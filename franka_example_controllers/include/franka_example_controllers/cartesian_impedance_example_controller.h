// Copyright (c) 2017 Franka Emika GmbH
// Use of this source code is governed by the Apache-2.0 license, see LICENSE
#pragma once

#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include <controller_interface/multi_interface_controller.h>
#include <dynamic_reconfigure/server.h>
#include <geometry_msgs/PoseStamped.h>
#include <sensor_msgs/JointState.h>
#include <franka_example_controllers/MultiControllerCommand.h>
#include <hardware_interface/joint_command_interface.h>
#include <hardware_interface/robot_hw.h>
#include <ros/node_handle.h>
#include <ros/time.h>
#include <Eigen/Dense>

#include <franka_example_controllers/compliance_paramConfig.h>
#include <franka_hw/franka_model_interface.h>
#include <franka_hw/franka_state_interface.h>

namespace franka_example_controllers {

class CartesianImpedanceExampleController : public controller_interface::MultiInterfaceController<
                                                franka_hw::FrankaModelInterface,
                                                hardware_interface::EffortJointInterface,
                                                franka_hw::FrankaStateInterface> {
 public:
  bool init(hardware_interface::RobotHW* robot_hw, ros::NodeHandle& node_handle) override;
  void starting(const ros::Time&) override;
  void update(const ros::Time&, const ros::Duration& period) override;
  void resetController();

 private:
  // Saturation
  Eigen::Matrix<double, 7, 1> saturateTorqueRate(
      const Eigen::Matrix<double, 7, 1>& tau_d_calculated,
      const Eigen::Matrix<double, 7, 1>& tau_J_d);  // NOLINT (readability-identifier-naming)

  // ADDED: clipping for joint commands
  Eigen::Matrix<double, 7, 1> clipTargets(
    const Eigen::Matrix<double, 7, 1>& joint_targets); // NOLINT (readability-identifier-naming)

  std::unique_ptr<franka_hw::FrankaStateHandle> state_handle_;
  std::unique_ptr<franka_hw::FrankaModelHandle> model_handle_;
  std::vector<hardware_interface::JointHandle> joint_handles_;

  // ADDED: internal varables for defining control type
  bool use_impedance_;
  bool use_joint_velocity_;
  bool use_joint_torque_;
  bool use_op_space_;
  bool use_decoupling_;

  // ADDED: handles for joint velocity control, in case we want to use their interface instead
  //        of computing our own torques
  // hardware_interface::VelocityJointInterface* velocity_joint_interface_;
  // std::vector<hardware_interface::JointHandle> velocity_joint_handles_;

  double filter_params_{1.0};
  double nullspace_stiffness_{20.0};
  double nullspace_stiffness_target_{20.0};
  const double delta_tau_max_{1.0};
  int count_{1};
  Eigen::Matrix<double, 6, 6> cartesian_stiffness_;
  Eigen::Matrix<double, 6, 6> cartesian_stiffness_target_;
  Eigen::Matrix<double, 6, 6> cartesian_damping_;
  Eigen::Matrix<double, 6, 6> cartesian_damping_target_;
  Eigen::Matrix<double, 7, 1> q_d_nullspace_;
  Eigen::Vector3d position_d_;
  Eigen::Quaterniond orientation_d_;
  std::mutex position_and_orientation_d_target_mutex_;
  Eigen::Vector3d position_d_target_;
  Eigen::Quaterniond orientation_d_target_;

  // ADDED: internal targets for joint velocity and torque controllers
  Eigen::Matrix<double, 7, 1> q_velocity_d_;
  Eigen::Matrix<double, 7, 1> q_torque_d_;
  Eigen::Matrix<double, 7, 1> q_velocity_d_target_;

  // ADDED: joint velocity and torque limits for control
  Eigen::Matrix<double, 7, 1> q_velocity_limits_;
  Eigen::Matrix<double, 7, 1> q_torque_limits_;

  // ADDED: joint velocity gains for joint velocity control
  Eigen::Matrix<double, 7, 1> q_velocity_kv_;

  // Dynamic reconfigure
  std::unique_ptr<dynamic_reconfigure::Server<franka_example_controllers::compliance_paramConfig>>
      dynamic_server_compliance_param_;
  ros::NodeHandle dynamic_reconfigure_compliance_param_node_;
  void complianceParamCallback(franka_example_controllers::compliance_paramConfig& config,
                               uint32_t level);

  // Equilibrium pose subscriber
  ros::Subscriber sub_equilibrium_;
  void equilibriumCallback(const franka_example_controllers::MultiControllerCommandConstPtr& msg);
  ros::Time last_time_;

  // ADDED: current controller type
  std::string curr_ctrl_type_{"null"};
};

}  // namespace franka_example_controllers
