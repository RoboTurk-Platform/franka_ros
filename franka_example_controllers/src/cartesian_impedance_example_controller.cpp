// Copyright (c) 2017 Franka Emika GmbH
// Use of this source code is governed by the Apache-2.0 license, see LICENSE
#include <franka_example_controllers/cartesian_impedance_example_controller.h>

#include <cmath>
#include <memory>

#include <controller_interface/controller_base.h>
#include <franka/robot_state.h>
#include <pluginlib/class_list_macros.h>
#include <ros/ros.h>

#include <franka_example_controllers/pseudo_inversion.h>

namespace franka_example_controllers {

bool CartesianImpedanceExampleController::init(hardware_interface::RobotHW* robot_hw,
                                               ros::NodeHandle& node_handle) {

  // ADDED: initialize variables here so that OpSpace is default
  use_impedance_ = false;
  use_joint_velocity_ = false;
  use_joint_torque_ = false;
  use_op_space_ = false;
  use_decoupling_ = false;

  // ADDED : one subscriber to rule them all
  sub_equilibrium_ = node_handle.subscribe(
      "/equilibrium_command", 20, &CartesianImpedanceExampleController::equilibriumCallback, this,
      ros::TransportHints().reliable().tcpNoDelay());

  std::string arm_id;
  if (!node_handle.getParam("arm_id", arm_id)) {
    ROS_ERROR_STREAM("CartesianImpedanceExampleController: Could not read parameter arm_id");
    return false;
  }
  std::vector<std::string> joint_names;
  if (!node_handle.getParam("joint_names", joint_names) || joint_names.size() != 7) {
    ROS_ERROR(
        "CartesianImpedanceExampleController: Invalid or no joint_names parameters provided, "
        "aborting controller init!");
    return false;
  }

  auto* model_interface = robot_hw->get<franka_hw::FrankaModelInterface>();
  if (model_interface == nullptr) {
    ROS_ERROR_STREAM(
        "CartesianImpedanceExampleController: Error getting model interface from hardware");
    return false;
  }
  try {
    model_handle_ = std::make_unique<franka_hw::FrankaModelHandle>(
        model_interface->getHandle(arm_id + "_model"));
  } catch (hardware_interface::HardwareInterfaceException& ex) {
    ROS_ERROR_STREAM(
        "CartesianImpedanceExampleController: Exception getting model handle from interface: "
        << ex.what());
    return false;
  }

  auto* state_interface = robot_hw->get<franka_hw::FrankaStateInterface>();
  if (state_interface == nullptr) {
    ROS_ERROR_STREAM(
        "CartesianImpedanceExampleController: Error getting state interface from hardware");
    return false;
  }
  try {
    state_handle_ = std::make_unique<franka_hw::FrankaStateHandle>(
        state_interface->getHandle(arm_id + "_robot"));
  } catch (hardware_interface::HardwareInterfaceException& ex) {
    ROS_ERROR_STREAM(
        "CartesianImpedanceExampleController: Exception getting state handle from interface: "
        << ex.what());
    return false;
  }

  // NOTE: this grabs handles for sending torque commands to the robot
  auto* effort_joint_interface = robot_hw->get<hardware_interface::EffortJointInterface>();
  if (effort_joint_interface == nullptr) {
    ROS_ERROR_STREAM(
        "CartesianImpedanceExampleController: Error getting effort joint interface from hardware");
    return false;
  }
  for (size_t i = 0; i < 7; ++i) {
    try {
      joint_handles_.push_back(effort_joint_interface->getHandle(joint_names[i]));
    } catch (const hardware_interface::HardwareInterfaceException& ex) {
      ROS_ERROR_STREAM(
          "CartesianImpedanceExampleController: Exception getting joint handles: " << ex.what());
      return false;
    }
  }

  // ADDED: initialize interface to command robot with joint velocities instead of joint torques
  // velocity_joint_interface_ = robot_hardware->get<hardware_interface::VelocityJointInterface>();
  // if (velocity_joint_interface_ == nullptr) {
  //   ROS_ERROR(
  //       "JointVelocityExampleController: Error getting velocity joint interface from hardware!");
  //   return false;
  // }
  // velocity_joint_handles_.resize(7);
  // for (size_t i = 0; i < 7; ++i) {
  //   try {
  //     velocity_joint_handles_[i] = velocity_joint_interface_->getHandle(joint_names[i]);
  //   } catch (const hardware_interface::HardwareInterfaceException& ex) {
  //     ROS_ERROR_STREAM(
  //         "JointVelocityExampleController: Exception getting joint handles: " << ex.what());
  //     return false;
  //   }
  // }

  dynamic_reconfigure_compliance_param_node_ =
      ros::NodeHandle(node_handle.getNamespace() + "dynamic_reconfigure_compliance_param_node");

  dynamic_server_compliance_param_ = std::make_unique<
      dynamic_reconfigure::Server<franka_example_controllers::compliance_paramConfig>>(

      dynamic_reconfigure_compliance_param_node_);
  dynamic_server_compliance_param_->setCallback(
      boost::bind(&CartesianImpedanceExampleController::complianceParamCallback, this, _1, _2));

  // ADDED: initialize all relevant control variables

  // for impedance and opspace control
  position_d_.setZero();
  orientation_d_.coeffs() << 0.0, 0.0, 0.0, 1.0;
  position_d_target_.setZero();
  orientation_d_target_.coeffs() << 0.0, 0.0, 0.0, 1.0;

  cartesian_stiffness_.setZero();
  cartesian_damping_.setZero();

  // for joint velocity control
  q_velocity_limits_ << 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0;
  q_velocity_kv_ << 8.0, 7.0, 6.0, 4.0, 2.0, 1.5, 2.5;
  q_velocity_d_.setZero();
  q_velocity_d_target_.setZero();

  // for joint torque control
  q_torque_limits_ << 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0;
  q_torque_d_.setZero();

  last_time_ = ros::Time(0);

  return true;
}

void CartesianImpedanceExampleController::starting(const ros::Time& /*time*/) {
  CartesianImpedanceExampleController::resetController();
}

void CartesianImpedanceExampleController::resetController() {
  // ADDED: initialize variables here so that OpSpace is default
  use_impedance_ = false;
  use_joint_velocity_ = false;
  use_joint_torque_ = false;
  use_op_space_ = false;
  use_decoupling_ = false;

  // compute initial velocity with jacobian and set x_attractor and q_d_nullspace
  // to initial configuration
  franka::RobotState initial_state = state_handle_->getRobotState();
  // get jacobian
  std::array<double, 42> jacobian_array =
      model_handle_->getZeroJacobian(franka::Frame::kEndEffector);
  // convert to eigen
  Eigen::Map<Eigen::Matrix<double, 7, 1>> q_initial(initial_state.q.data());
  Eigen::Affine3d initial_transform(Eigen::Matrix4d::Map(initial_state.O_T_EE.data()));

  // for impedance and opspace control

  // set equilibrium point to current state
  position_d_ = initial_transform.translation();
  orientation_d_ = Eigen::Quaterniond(initial_transform.linear());
  position_d_target_ = initial_transform.translation();
  orientation_d_target_ = Eigen::Quaterniond(initial_transform.linear());

  // set nullspace equilibrium configuration to initial q
  // q_d_nullspace_ = q_initial;
  q_d_nullspace_ << 0.0, -0.3135, 0.0, -2.515, 0.0, 2.226, 0.87;

  // for joint velocity control
  q_velocity_limits_ << 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0;
  q_velocity_kv_ << 8.0, 7.0, 6.0, 4.0, 2.0, 1.5, 2.5;
  q_velocity_d_.setZero();
  q_velocity_d_target_.setZero();

  // for joint torque control
  q_torque_limits_ << 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0;
  q_torque_d_.setZero();
}

void CartesianImpedanceExampleController::update(const ros::Time& /*time*/,
                                                 const ros::Duration& /*period*/) {
  // count for debugging
  count_ ++;
  // get state variables
  franka::RobotState robot_state = state_handle_->getRobotState();
  std::array<double, 7> coriolis_array = model_handle_->getCoriolis();
  std::array<double, 42> jacobian_array =
      model_handle_->getZeroJacobian(franka::Frame::kEndEffector);

  // convert to Eigen
  Eigen::Map<Eigen::Matrix<double, 7, 1>> coriolis(coriolis_array.data());
  Eigen::Map<Eigen::Matrix<double, 6, 7>> jacobian(jacobian_array.data());
  Eigen::Map<Eigen::Matrix<double, 7, 1>> q(robot_state.q.data());
  Eigen::Map<Eigen::Matrix<double, 7, 1>> dq(robot_state.dq.data());
  Eigen::Map<Eigen::Matrix<double, 7, 1>> tau_J_d(  // NOLINT (readability-identifier-naming)
      robot_state.tau_J_d.data());
  Eigen::Affine3d transform(Eigen::Matrix4d::Map(robot_state.O_T_EE.data()));
  Eigen::Vector3d position(transform.translation());
  Eigen::Quaterniond orientation(transform.linear());

  // OUR CONTROLLER IMPLEMENTATION

  // diagonal vector
  Eigen::VectorXd cartesian_stiffness_v = cartesian_stiffness_.diagonal();
  Eigen::VectorXd cartesian_damping_v = cartesian_damping_.diagonal();

  if (use_impedance_ || use_op_space_) {
    // compute error to desired pose

    // position error
    Eigen::Matrix<double, 6, 1> error;
    error.head(3) << position - position_d_;

    // orientation error
    if (orientation_d_.coeffs().dot(orientation.coeffs()) < 0.0) {
      // std::cout << "SHIT" << std::endl; 
      // std::cout << orientation_d_.coeffs() << std::endl; 
      // std::cout << orientation.coeffs() << std::endl; 
      orientation.coeffs() << -orientation.coeffs();
    }
    // "difference" quaternion
    Eigen::Quaterniond error_quaternion(orientation * orientation_d_.inverse());
    // convert to axis angle
    Eigen::AngleAxisd error_quaternion_angle_axis(error_quaternion);
    // compute "orientation error"
    // if(error_quaternion_angle_axis.angle() > 0.3)
    // {
    //  error_quaternion_angle_axis.angle() = 0.3; 
    // }else if (error_quaternion_angle_axis.angle() < -0.3)
    // {
    //   error_quaternion_angle_axis.angle() = -0.3;
    // }

    error.tail(3) << error_quaternion_angle_axis.axis() * error_quaternion_angle_axis.angle();
    if (count_ % 1000 == 0) {
      std::cout << "orn_error" << error.tail(3) << std::endl;
      std::cout << "pos_error" << error.head(3) << std::endl;
      std::cout << "orn_d_target" << orientation_d_target_.coeffs() << std::endl;
      std::cout << "orn_d" << orientation_d_.coeffs() << std::endl;
      std::cout << "orn" << orientation.coeffs() << std::endl;
    }
    // compute control
    // allocate variables
    Eigen::VectorXd tau_task(7), tau_nullspace(7), tau_d(7);

    if (use_op_space_) {
      // mass matrix M
      std::array<double, 49> mass_mat = model_handle_->getMass();
      Eigen::Map<Eigen::Matrix<double, 7, 7>> mass_matrix(mass_mat.data());

      mass_matrix(4, 4) += 0.10;
      mass_matrix(5, 5) += 0.10;
      mass_matrix(6, 6) += 0.10;

      if (count_ % 1000 == 0) {
        std::cout << mass_matrix << std::endl;
      }

      // M^-1
      Eigen::MatrixXd mass_matrix_inv = mass_matrix.inverse();

      // (J M^-1 J^T)^-1
      Eigen::MatrixXd lambda_matrix_inv = (jacobian * mass_matrix_inv) * jacobian.transpose();
      Eigen::MatrixXd lambda_matrix = lambda_matrix_inv.inverse();

      if (use_decoupling_) {
        // first get decoupled position and orientation matrices

        // Jx M^-1 Jx^T
        Eigen::MatrixXd jacobian_x = jacobian.block(0, 0, 3, 7);
        Eigen::MatrixXd lambda_x_matrix_inv = jacobian_x * mass_matrix_inv * jacobian_x.transpose();

        // Jr M^-1 Jr^T
        Eigen::MatrixXd jacobian_r = jacobian.block(3, 0, 3, 7);
        Eigen::MatrixXd lambda_r_matrix_inv = jacobian_r * mass_matrix_inv * jacobian_r.transpose();

        // threshold for SVD inversion to ensure stuff doesn't blow up
        double singularity_threshold = 0.00025;

        // take the inverse, but zero out elements in cases of a singularity (to take pseudoinverse)
        Eigen::JacobiSVD<Eigen::MatrixXd> svd_x(lambda_x_matrix_inv, Eigen::ComputeFullU | Eigen::ComputeFullV);
        Eigen::MatrixXd Ux = svd_x.matrixU();
        Eigen::MatrixXd Vx = svd_x.matrixV();
        Eigen::VectorXd Sx = svd_x.singularValues();

        for (size_t i = 0; i < Sx.size(); ++i) {
          if (Sx[i] < singularity_threshold) {
            Sx[i] = 0.;
          } else {
            Sx[i] = 1. / Sx[i];
          }
        }

        Eigen::MatrixXd lambda_x_matrix = Vx.transpose() * Sx.asDiagonal() * Ux.transpose();

        // take the inverse, but zero out elements in cases of a singularity (to take pseudoinverse)
        Eigen::JacobiSVD<Eigen::MatrixXd> svd_r(lambda_r_matrix_inv, Eigen::ComputeFullU | Eigen::ComputeFullV);
        Eigen::MatrixXd Ur = svd_r.matrixU();
        Eigen::MatrixXd Vr = svd_r.matrixV();
        Eigen::VectorXd Sr = svd_r.singularValues();
        for (size_t i = 0; i < Sr.size(); ++i) {
          if (Sr[i] < singularity_threshold) {
            Sr[i] = 0.;
          } else {
            Sr[i] = 1. / Sr[i];
          }
        }

        Eigen::MatrixXd lambda_r_matrix = Vr.transpose() * Sr.asDiagonal() * Ur.transpose();

        // compute decoupled force control from above matrices
        Eigen::Matrix<double, 6, 1> decoupled_control;
        Eigen::Matrix<double, 6, 1> ee_vel = jacobian * dq;

        // std::cout << "Ur" << Ur << std::endl;
        // std::cout << "Vr" << Vr << std::endl;
        // std::cout << "Sr" << Sr << std::endl;

        // std::cout << "lxm" << lambda_x_matrix << std::endl;
        // std::cout << "lrm" << lambda_r_matrix << std::endl;

        decoupled_control.head(3) << lambda_x_matrix * (-cartesian_stiffness_v.head(3).cwiseProduct(error.head(3)) - cartesian_damping_v.head(3).cwiseProduct(ee_vel.head(3)));
        decoupled_control.tail(3) << lambda_r_matrix * (-cartesian_stiffness_v.tail(3).cwiseProduct(error.tail(3)) - cartesian_damping_v.tail(3).cwiseProduct(ee_vel.tail(3)));
        // std::cout << "dc" << decoupled_control << std::endl;
        // compute appropriate task torque from decoupled force
        tau_task << jacobian.transpose() * decoupled_control;

      } else {
        // PD control law returns endpoint accelerations, so multiply by mass matrix to get forces
        // and J^T to get task torques
        tau_task << jacobian.transpose() *
                        lambda_matrix * (-cartesian_stiffness_v.cwiseProduct(error) - cartesian_damping_v.cwiseProduct(jacobian * dq));
      }

    } else {
      // Cartesian PD control with damping ratio = 1
      tau_task << jacobian.transpose() *
                      (-cartesian_stiffness_v.cwiseProduct(error) - cartesian_damping_v.cwiseProduct(jacobian * dq));
    }

    // pseudoinverse for nullspace handling
    // kinematic pseudoinverse
    Eigen::MatrixXd jacobian_transpose_pinv;
    pseudoInverse(jacobian.transpose(), jacobian_transpose_pinv);

    // nullspace PD control with damping ratio = 1
    tau_nullspace << (Eigen::MatrixXd::Identity(7, 7) -
                      jacobian.transpose() * jacobian_transpose_pinv) *
                         (nullspace_stiffness_ * (q_d_nullspace_ - q) -
                          (2.0 * sqrt(nullspace_stiffness_)) * dq);
    // Desired torque
    tau_d << tau_task + tau_nullspace + coriolis;
    if (count_ % 1000 == 0) {
      std::cout << "TORQQQQQ" << std::endl;
      std::cout << tau_d << std::endl;
    }
    // tau_d.setZero();

    // Saturate torque rate to avoid discontinuities
    tau_d << saturateTorqueRate(tau_d, tau_J_d);

    // std::cout << "TORQ111111" << std::endl;
    // std::cout << tau_d << std::endl;

    // Send torque command
    // std::array<double, 7> tau_offset;
    Eigen::Matrix<double, 7, 1> tau_offset;
    tau_offset << -0.3, -0.5, 0.1, 0.0, 0.0, 0.0, 0.0;
    Eigen::Matrix<double, 7, 1> tau_limit;
    tau_limit << 10.0, 10.0, 10.0, 10.0, 5.0, 5.0, 5.0;

    for (size_t i = 0; i < 7; ++i) {
      tau_d(i) += tau_offset(i);
      if (tau_d(i) > tau_limit(i)) {
        tau_d(i) = tau_limit(i);
      } else if (tau_d(i) < -tau_limit(i)) {
        tau_d(i) = -tau_limit(i);
      }
      joint_handles_[i].setCommand(tau_d(i));
    }
  } else if (use_joint_velocity_) {
    if (count_ % 1000 == 0) {
      std::cout << "joint_vel" << std::endl;
    }

    ros::Duration delay = ros::Time::now() - last_time_;

    if(last_time_ != ros::Time(0) && delay.toSec() >0.2)
    {
      q_velocity_d_target_.setZero();
      last_time_ = ros::Time(0);
      std::cout << "No signal received. Setting des vel to zero" << std::endl;

    }

    // allocate variables
    Eigen::VectorXd tau_d(7);

    // Clip velocity targets (these are smoothed and interpolated from raw targets coming in)
    Eigen::Matrix<double, 7, 1> clipped_targets;
    clipped_targets << clipTargets(q_velocity_d_);

    // Method 1: Compute appropriate joint torques based off of joint velocity target with P-controller.

    // P-Controller to compute torques 
    tau_d << -q_velocity_kv_.cwiseProduct(dq - clipped_targets);

    // for (size_t i = 0; i < 7; ++i) {
    //   tau_d[i] = -q_velocity_kv_[i] * (dq[i] - clipped_targets[i]);
    // }

    // Saturate torque rate to avoid discontinuities
    tau_d << saturateTorqueRate(tau_d, tau_J_d);

    // Send torque command
    for (size_t i = 0; i < 7; ++i) {
      joint_handles_[i].setCommand(tau_d(i));
    }

    // Method 2: Send joint velocity command directly to controller
    // for (size_t i = 0; i < 7; ++i) {
    //   velocity_joint_handles_[i].setCommand(clipped_targets(i));
    // }

  } else if (use_joint_torque_) {

    // allocate variables
    if (count_ % 1000 == 0) {
      std::cout << "joint_torque" << std::endl;
    }
    Eigen::VectorXd tau_d(7);

    // Clip commanded torque
    tau_d << clipTargets(q_torque_d_);

    // Saturate torque rate to avoid discontinuities
    tau_d << saturateTorqueRate(tau_d, tau_J_d);

    // Send torque command
    for (size_t i = 0; i < 7; ++i) {
      joint_handles_[i].setCommand(tau_d(i));
    }
  } 

  if (use_impedance_ || use_op_space_) {
    // update parameters changed online either through dynamic reconfigure or through the interactive
    // target by filtering
    cartesian_stiffness_ =
        filter_params_ * cartesian_stiffness_target_ + (1.0 - filter_params_) * cartesian_stiffness_;
    cartesian_damping_ =
        filter_params_ * cartesian_damping_target_ + (1.0 - filter_params_) * cartesian_damping_;
    nullspace_stiffness_ =
        filter_params_ * nullspace_stiffness_target_ + (1.0 - filter_params_) * nullspace_stiffness_;
    position_d_ = filter_params_ * position_d_target_ + (1.0 - filter_params_) * position_d_;
    Eigen::AngleAxisd aa_orientation_d(orientation_d_);
    Eigen::AngleAxisd aa_orientation_d_target(orientation_d_target_);
    aa_orientation_d.axis() = filter_params_ * aa_orientation_d_target.axis() +
                              (1.0 - filter_params_) * aa_orientation_d.axis();
    aa_orientation_d.angle() = filter_params_ * aa_orientation_d_target.angle() +
                               (1.0 - filter_params_) * aa_orientation_d.angle();
    // orientation_d_ = Eigen::Quaterniond(aa_orientation_d);
    // orientation_d_ = Eigen::Quaterniond(orientation_d_target_);

    // Spherical Linear Interpolation instead of their messed up shit
    orientation_d_ = orientation_d_.slerp(filter_params_, orientation_d_target_);
  } else if (use_joint_velocity_) {
    // ADDED: interpolation in joint velocity targets from the ones coming in over ROS
    q_velocity_d_ = filter_params_ * q_velocity_d_target_ + (1.0 - filter_params_) * q_velocity_d_;

  // THEIR CODE
  // // compute error to desired pose
  // // position error
  // Eigen::Matrix<double, 6, 1> error;
  // error.head(3) << position - position_d_;

  // // orientation error
  // if (orientation_d_.coeffs().dot(orientation.coeffs()) < 0.0) {
  //   orientation.coeffs() << -orientation.coeffs();
  // }
  // // "difference" quaternion
  // Eigen::Quaterniond error_quaternion(orientation.inverse() * orientation_d_);
  // error.tail(3) << error_quaternion.x(), error_quaternion.y(), error_quaternion.z();
  // // Transform to base frame
  // error.tail(3) << -transform.linear() * error.tail(3);

  // // compute control
  // // allocate variables
  // Eigen::VectorXd tau_task(7), tau_nullspace(7), tau_d(7);

  // // pseudoinverse for nullspace handling
  // // kinematic pseuoinverse
  // Eigen::MatrixXd jacobian_transpose_pinv;
  // pseudoInverse(jacobian.transpose(), jacobian_transpose_pinv);

  // // Cartesian PD control with damping ratio = 1
  // tau_task << jacobian.transpose() *
  //                 (-cartesian_stiffness_ * error - cartesian_damping_ * (jacobian * dq));
  // // nullspace PD control with damping ratio = 1
  // tau_nullspace << (Eigen::MatrixXd::Identity(7, 7) -
  //                   jacobian.transpose() * jacobian_transpose_pinv) *
  //                      (nullspace_stiffness_ * (q_d_nullspace_ - q) -
  //                       (2.0 * sqrt(nullspace_stiffness_)) * dq);
  // // Desired torque
  // tau_d << tau_task + tau_nullspace + coriolis;
  // // Saturate torque rate to avoid discontinuities
  // tau_d << saturateTorqueRate(tau_d, tau_J_d);
  // for (size_t i = 0; i < 7; ++i) {
  //   joint_handles_[i].setCommand(tau_d(i));
  // }

  // // update parameters changed online either through dynamic reconfigure or through the interactive
  // // target by filtering
  // cartesian_stiffness_ =
  //     filter_params_ * cartesian_stiffness_target_ + (1.0 - filter_params_) * cartesian_stiffness_;
  // cartesian_damping_ =
  //     filter_params_ * cartesian_damping_target_ + (1.0 - filter_params_) * cartesian_damping_;
  // nullspace_stiffness_ =
  //     filter_params_ * nullspace_stiffness_target_ + (1.0 - filter_params_) * nullspace_stiffness_;
  // std::lock_guard<std::mutex> position_d_target_mutex_lock(
  //     position_and_orientation_d_target_mutex_);
  // position_d_ = filter_params_ * position_d_target_ + (1.0 - filter_params_) * position_d_;
  // orientation_d_ = orientation_d_.slerp(filter_params_, orientation_d_target_);
}

Eigen::Matrix<double, 7, 1> CartesianImpedanceExampleController::saturateTorqueRate(
    const Eigen::Matrix<double, 7, 1>& tau_d_calculated,
    const Eigen::Matrix<double, 7, 1>& tau_J_d) {  // NOLINT (readability-identifier-naming)
  Eigen::Matrix<double, 7, 1> tau_d_saturated{};
  for (size_t i = 0; i < 7; i++) {
    double difference = tau_d_calculated[i] - tau_J_d[i];
    tau_d_saturated[i] =
        tau_J_d[i] + std::max(std::min(difference, delta_tau_max_), -delta_tau_max_);
  }
  return tau_d_saturated;
}

// ADDED: clipping functions for target joint velocities and torques
Eigen::Matrix<double, 7, 1> CartesianImpedanceExampleController::clipTargets(
    const Eigen::Matrix<double, 7, 1>& joint_targets) {  // NOLINT (readability-identifier-naming)
  Eigen::Matrix<double, 7, 1> clipped_targets{};
  for (size_t i = 0; i < 7; i++) {
    if (use_joint_velocity_) {
      clipped_targets[i] = std::max(std::min(joint_targets[i], q_velocity_limits_[i]), -q_velocity_limits_[i]);
    } else if (use_joint_torque_) {
      clipped_targets[i] = std::max(std::min(joint_targets[i], q_torque_limits_[i]), -q_torque_limits_[i]);
    } else {
      ROS_ERROR(
        "CartesianImpedanceExampleController: Invalid control options at top of cpp file!!!"
        "aborting clipping!");
    }
  }
  return clipped_targets;
}

void CartesianImpedanceExampleController::complianceParamCallback(
    franka_example_controllers::compliance_paramConfig& config,
    uint32_t /*level*/) {
  cartesian_stiffness_target_.setIdentity();

  // float t_stiffness = 100.0; //20.0;
  // float r_stiffness = 40.0;

  float t_stiffness = config.translational_stiffness; //20.0;
  float r_stiffness = config.rotational_stiffness;

  cartesian_stiffness_target_.topLeftCorner(3, 3)
      << t_stiffness * Eigen::Matrix3d::Identity();
  cartesian_stiffness_target_.bottomRightCorner(3, 3)
      << r_stiffness * Eigen::Matrix3d::Identity();
  cartesian_damping_target_.setIdentity();
  // Damping ratio = 1
  // cartesian_damping_target_.topLeftCorner(3, 3)
  //     << 2.0 * sqrt(config.translational_stiffness) * Eigen::Matrix3d::Identity();
  // cartesian_damping_target_.bottomRightCorner(3, 3)
  //     << 2.0 * sqrt(config.rotational_stiffness) * Eigen::Matrix3d::Identity();
  cartesian_damping_target_.topLeftCorner(3, 3)
      << 2.0 * sqrt(t_stiffness) * Eigen::Matrix3d::Identity();
  cartesian_damping_target_.bottomRightCorner(3, 3)
      << config.rotational_damping_factor * 2.0 * sqrt(r_stiffness) * Eigen::Matrix3d::Identity();
  nullspace_stiffness_target_ = config.nullspace_stiffness;

  filter_params_ = config.filter_coeff;
  q_velocity_kv_ << config.q_kp_0, config.q_kp_1, config.q_kp_2, config.q_kp_3, config.q_kp_4, config.q_kp_5, config.q_kp_6 ;
}

// void CartesianImpedanceExampleController::equilibriumPoseCallback(
//     const geometry_msgs::PoseStampedConstPtr& msg) {
//   std::lock_guard<std::mutex> position_d_target_mutex_lock(
//       position_and_orientation_d_target_mutex_);
//   position_d_target_ << msg->pose.position.x, msg->pose.position.y, msg->pose.position.z;
//   Eigen::Quaterniond last_orientation_d_target(orientation_d_target_);
//   orientation_d_target_.coeffs() << msg->pose.orientation.x, msg->pose.orientation.y,
//       msg->pose.orientation.z, msg->pose.orientation.w;
//   if (last_orientation_d_target.coeffs().dot(orientation_d_target_.coeffs()) < 0.0) {
//     orientation_d_target_.coeffs() << -orientation_d_target_.coeffs();
//   }
// }

// NOTE: callback for /equilibrium topic - for receiving control commands
void CartesianImpedanceExampleController::equilibriumCallback(
    const franka_example_controllers::MultiControllerCommandConstPtr& msg) {

  // controller_type: "impedance", "osc", "velocity", "torque"
  // pose_command
  // joint_command

  // clear control flags
  use_impedance_ = false;
  use_joint_velocity_ = false;
  use_joint_torque_ = false;
  use_op_space_ = false;
  use_decoupling_ = false;
  std::string ctrl_type(msg->controller_type.data.c_str());

  if (curr_ctrl_type_ != ctrl_type) {
    CartesianImpedanceExampleController::resetController();
  }
  curr_ctrl_type_ = ctrl_type;

  if (ctrl_type == std::string("velocity")) {

    // update controller variables
    use_joint_velocity_ = true;

    // set internal joint velocity command

    // convert to eigen from std::vector<double> first before assignment
    Eigen::Map<const Eigen::Matrix<double, 7, 1>> msg_velocity(msg->joint_command.velocity.data());
    q_velocity_d_target_ << msg_velocity;

  } else if (ctrl_type == std::string("torque")) {

    // update controller variables
    use_joint_torque_ = true;

    // set internal joint torque command

    // convert to eigen from std::vector<double> first before assignment
    Eigen::Map<const Eigen::Matrix<double, 7, 1>> msg_effort(msg->joint_command.effort.data());
    q_torque_d_ << msg_effort;

  } else if (ctrl_type == std::string("impedance") || ctrl_type == std::string("osc")) {

    if (ctrl_type == std::string("impedance")) {
      // update controller variables
      use_impedance_ = true;
    } else {
      use_op_space_ = true;
    }

    position_d_target_ << msg->pose_command.pose.position.x, msg->pose_command.pose.position.y, msg->pose_command.pose.position.z;
    Eigen::Quaterniond last_orientation_d_target(orientation_d_target_);
    orientation_d_target_.coeffs() << msg->pose_command.pose.orientation.x, msg->pose_command.pose.orientation.y,
        msg->pose_command.pose.orientation.z, msg->pose_command.pose.orientation.w;
    if (last_orientation_d_target.coeffs().dot(orientation_d_target_.coeffs()) < 0.0) {
      orientation_d_target_.coeffs() << -orientation_d_target_.coeffs();
    }
  }

  last_time_ = ros::Time::now();
}

}  // namespace franka_example_controllers

PLUGINLIB_EXPORT_CLASS(franka_example_controllers::CartesianImpedanceExampleController,
                       controller_interface::ControllerBase)
