import os

# Fail-safe: if not chosent otherwise before,
# communicate only on localhost to avoid network issues with ROS2
os.environ.setdefault("ROS_LOCALHOST_ONLY", "1")

print(
    "ROS 2 network mode:",
    "LOCALHOST" if os.environ["ROS_LOCALHOST_ONLY"] == "1" else "NETWORK",
)

import sys
import shlex
import subprocess
from pathlib import Path

dir_path = Path(__file__).resolve().parent
sys.path.append(str(dir_path / ".."))

ros_ws = dir_path / "ros2_ws"
setup_bash = ros_ws / "install" / "setup.bash"

if not setup_bash.exists():
    print("Building the msgs first...")
    subprocess.run(["colcon", "build"], cwd=ros_ws, check=True)

if os.environ.get("SIM2REAL_ROBOT_IDENTIFICATION_SOURCED") != "1":
    print("Sourcing ROS2 workspace and restarting script...")
    cmd = (
        f"source {shlex.quote(str(setup_bash))} && "
        "export SIM2REAL_ROBOT_IDENTIFICATION_SOURCED=1 && "
        f"exec {shlex.quote(sys.executable)} "
        + " ".join(shlex.quote(arg) for arg in [str(Path(__file__).resolve()), *sys.argv[1:]])
    )
    os.execv("/bin/bash", ["bash", "-c", cmd])


import rclpy 
from rclpy.node import Node 
from dls2_interface.msg import BlindState, ControlSignal

import time
import numpy as np
np.set_printoptions(precision=3, suppress=True)

import threading
import copy
import torch

import mujoco
import mujoco.viewer
import config
import collection_trajectories


# Set the priority of the process
pid = os.getpid()
print("PID: ", pid)
os.system("renice -n -21 -p " + str(pid))
os.system("echo -20 > /proc/" + str(pid) + "/autogroup")
#for real time, launch it with chrt -r 99 python3 run_controller.py

USE_MUJOCO_RENDER = config.USE_MUJOCO_RENDER
USE_MUJOCO_SIMULATION = config.USE_MUJOCO_SIMULATION


CONTROL_FREQ = config.frequency_collection # Hz 

def handle_parallel_gripper(array):
    if(config.robot == "piper_l"):
        return array[:-1]
    else:
        return array


class Data_Collection_Node(Node):
    def __init__(self):
        super().__init__('Data_Collection_Node')
        # Subscribers and Publishers
        self.subscription_blind_state = self.create_subscription(BlindState,"/blind_state_arm", self.get_blind_state_callback, 1)
        self.publisher_control_signal = self.create_publisher(ControlSignal,"/control_signal_arm", 1)
        self.sequence_id = 0 # To keep track of the last msg sent, useful for debugging and synchronization
        self.timer = self.create_timer(1.0/CONTROL_FREQ, self.compute_control)


        # Safety check to not do anything until a first base and blind state are received
        self.first_message_joints_arrived = False 

        # Timing stuff
        self.loop_time = 0.002
        self.last_start_time = None
        self.start_collection_time = None

        # Base State
        self.position = np.zeros(3)
        self.orientation = np.zeros(4)
        self.linear_velocity = np.zeros(3)
        self.angular_velocity = np.zeros(3)

        # Blind State
        self.joint_positions = np.zeros(12)
        self.joint_velocities = np.zeros(12)
        self.feet_contact = np.zeros(4)


        # Create the environment -----------------------------------------------------------
        self.mjModel = mujoco.MjModel.from_xml_path(str(dir_path) + "/robot_model/" + config.robot + "/scene_flat.xml")
        self.mjData = mujoco.MjData(self.mjModel)
        if USE_MUJOCO_SIMULATION:
            self.mjModel.opt.timestep = 1.0 / CONTROL_FREQ

        if(USE_MUJOCO_RENDER):
            self.viewer = mujoco.viewer.launch_passive(
                self.mjModel,
                self.mjData,
                show_left_ui=False,
                show_right_ui=False,
            )
            self.last_render_time = time.time()


        keyframe_home_id = mujoco.mj_name2id(self.mjModel, mujoco.mjtObj.mjOBJ_KEY, "home")
        keyframe_sys_id_1 = mujoco.mj_name2id(self.mjModel, mujoco.mjtObj.mjOBJ_KEY, "sys_id_1")
        keyframe_sys_id_2 = mujoco.mj_name2id(self.mjModel, mujoco.mjtObj.mjOBJ_KEY, "sys_id_2")
        self.home_position = self.mjModel.key_qpos[keyframe_sys_id_1]
        self.goal_position = self.mjModel.key_qpos[keyframe_sys_id_2]

        # Handling parallel gripper - if present, otherwise 
        # the func will return the same array
        self.home_position = handle_parallel_gripper(self.home_position)
        self.goal_position = handle_parallel_gripper(self.goal_position)
        self.idle_joint_position = copy.deepcopy(self.home_position)

        self.Kp = config.Kp
        self.Kd = config.Kd

        # Collection trajectory (see collection_trajectories.py): prepared and checked by the operator,
        # then executed once accepted
        self.pending_trajectory = None
        self.trajectory = None
        self.setpoint_preview = None  # posture shown in the viewer while waiting for confirmation
        self.setpoint_preview_viewer = None
        self.setpoint_preview_data = None

        self.saved_actual_joints_position = None
        self.saved_actual_joints_velocity = None
        self.saved_desired_joints_position = None
        self.saved_desired_joints_velocity = None
        self.saved_commanded_joints_torque = None
        # Interactive Command Line ----------------------------
        from console import Console
        self.console = Console(controller_node=self)
        thread_console = threading.Thread(target=self.console.interactive_command_line)
        thread_console.daemon = True
        thread_console.start()


    def get_blind_state_callback(self, msg):
        self.arm_joints_position = np.array(msg.joints_position)
        self.arm_joints_velocity = np.array(msg.joints_velocity)

        self.first_message_joints_arrived = True

    def prepare_trajectory(self, kind, **options):
        """Generate the trajectory of a collection mode and print its summary, without commanding the robot.

        kind is setpoint, chirp or excitation; options are the arguments of its generator (e.g. the speed scale).
        All of them stay inside the joint box of the sys_id_1 and sys_id_2 keyframes.
        """
        self._close_calibration_setpoint_preview()
        if kind == "setpoint":
            trajectory = collection_trajectories.manipulator_setpoint_trajectory(
                self.home_position, self.goal_position, CONTROL_FREQ
            )
        elif kind == "chirp":
            trajectory = collection_trajectories.manipulator_chirp_trajectory(
                self.home_position, self.goal_position, CONTROL_FREQ, **options
            )
        elif kind == "excitation":
            # starts from and returns to the current desired posture
            trajectory = collection_trajectories.manipulator_excitation_trajectory(
                self.home_position, self.goal_position, start=self.idle_joint_position,
                frequency=CONTROL_FREQ, **options,
            )
        elif kind == "falling":
            raise NotImplementedError("Falling collection is not implemented for manipulators.")
        else:
            raise ValueError(f"Unknown collection mode: {kind}")
        joint_names = [self.mjModel.joint(i).name for i in range(len(self.home_position))]
        print(trajectory.describe(joint_names))
        if kind == "setpoint":
            self.setpoint_preview = trajectory.target
            if USE_MUJOCO_RENDER:
                print("The proposed pose is shown in the MuJoCo viewer.")
            else:
                self._open_calibration_setpoint_preview()
                print("A temporary MuJoCo viewer is showing the proposed pose.")
        self.pending_trajectory = trajectory

    def start_trajectory(self):
        """Start timing only after the operator has approved the trajectory."""
        self._close_calibration_setpoint_preview()
        self.trajectory = self.pending_trajectory
        self.pending_trajectory = None
        self.setpoint_preview = None
        self.start_collection_time = time.time()

    def cancel_trajectory(self):
        self._close_calibration_setpoint_preview()
        self.pending_trajectory = None
        self.setpoint_preview = None

    def _finish_trajectory(self, desired_joint_pos):
        """Save the collected data and hold the final posture."""
        self.idle_joint_position = copy.deepcopy(desired_joint_pos)
        self._save_trajectory_data(self.trajectory.name, self.trajectory.metadata)
        print(f"{self.trajectory.label.capitalize()} collection completed.")
        self.trajectory = None
        self.console.isActivated = False

    def _set_preview_qpos(self, data):
        if config.robot == "piper_l":
            data.qpos[:-1] = self.setpoint_preview
            data.qpos[-1] = -self.setpoint_preview[-1]
        else:
            data.qpos[:] = self.setpoint_preview

    def _open_calibration_setpoint_preview(self):
        """Open a temporary viewer that cannot affect the controlled state."""
        self.setpoint_preview_data = mujoco.MjData(self.mjModel)
        self._set_preview_qpos(self.setpoint_preview_data)
        self.setpoint_preview_data.qvel[:] = 0.0
        mujoco.mj_forward(self.mjModel, self.setpoint_preview_data)
        self.setpoint_preview_viewer = mujoco.viewer.launch_passive(
            self.mjModel,
            self.setpoint_preview_data,
            show_left_ui=False,
            show_right_ui=False,
        )
        self.setpoint_preview_viewer.sync()

    def _close_calibration_setpoint_preview(self):
        if self.setpoint_preview_viewer is not None:
            self.setpoint_preview_viewer.close()
        self.setpoint_preview_viewer = None
        self.setpoint_preview_data = None

    def _render_calibration_setpoint_preview(self):
        """Render the proposed pose without changing the controller/simulation state."""
        saved_qpos = self.mjData.qpos.copy()
        saved_qvel = self.mjData.qvel.copy()

        self._set_preview_qpos(self.mjData)
        self.mjData.qvel[:] = 0.0
        mujoco.mj_forward(self.mjModel, self.mjData)
        self.viewer.sync()

        self.mjData.qpos[:] = saved_qpos
        self.mjData.qvel[:] = saved_qvel
        mujoco.mj_forward(self.mjModel, self.mjData)

    def _collect_trajectory_data(self, joints_pos, joints_vel, desired_joint_pos):
        """Collect trajectory data by concatenating and storing joint information"""
        
        concatenated_actual_joints_position = np.array(joints_pos, copy=True)
        concatenated_actual_joints_velocity = np.array(joints_vel, copy=True)
        concatenated_desired_joints_position = np.array(desired_joint_pos, copy=True)
        concatenated_desired_joints_velocity = np.zeros_like(desired_joint_pos)
        
        error_joints_pos = desired_joint_pos - joints_pos                
        concatenated_commanded_joints_torque = config.Kp * (error_joints_pos) - config.Kd * joints_vel

        if self.saved_actual_joints_position is None:
            self.saved_actual_joints_position = concatenated_actual_joints_position
            self.saved_actual_joints_velocity = concatenated_actual_joints_velocity
            self.saved_desired_joints_position = concatenated_desired_joints_position
            self.saved_desired_joints_velocity = concatenated_desired_joints_velocity
            self.saved_commanded_joints_torque = concatenated_commanded_joints_torque
        else:
            self.saved_actual_joints_position = np.vstack([self.saved_actual_joints_position, concatenated_actual_joints_position])
            self.saved_actual_joints_velocity = np.vstack([self.saved_actual_joints_velocity, concatenated_actual_joints_velocity])
            self.saved_desired_joints_position = np.vstack([self.saved_desired_joints_position, concatenated_desired_joints_position])
            self.saved_desired_joints_velocity = np.vstack([self.saved_desired_joints_velocity, concatenated_desired_joints_velocity])
            self.saved_commanded_joints_torque = np.vstack([self.saved_commanded_joints_torque, concatenated_commanded_joints_torque])

    def _save_trajectory_data(self, dataset_prefix, metadata=None):
        """Save collected trajectory data to file, with optional extra entries (e.g. the excitation seed)"""

        # HACK
        num_steps = self.saved_actual_joints_position.shape[0]
        time_data = torch.arange(num_steps, device="cpu") / CONTROL_FREQ
        dof_pos_buffer = torch.zeros(num_steps, 7, device="cpu")
        dof_vel_buffer = torch.zeros(num_steps, 7, device="cpu")
        dof_target_pos_buffer = torch.zeros(num_steps, 7, device="cpu")
        dof_target_vel_buffer = torch.zeros(num_steps, 7, device="cpu")
        dof_target_commanded_torque_buffer = torch.zeros(num_steps, 7, device="cpu")
        
        dof_pos_buffer[:, :] = torch.from_numpy(self.saved_actual_joints_position)
        dof_vel_buffer[:, :] = torch.from_numpy(self.saved_actual_joints_velocity)
        dof_target_pos_buffer[:, :] = torch.from_numpy(self.saved_desired_joints_position)
        #dof_target_vel_buffer[:, :] = torch.from_numpy(self.saved_desired_joints_velocity)
        dof_target_commanded_torque_buffer[:, :] = torch.from_numpy(self.saved_commanded_joints_torque)

        save_dir = "datasets/" + config.robot
        os.makedirs(save_dir, exist_ok=True)
        dataset_index = 1
        dataset_path = os.path.join(save_dir, f"{dataset_prefix}_{dataset_index}.pt")
        while os.path.exists(dataset_path):
            dataset_index += 1
            dataset_path = os.path.join(save_dir, f"{dataset_prefix}_{dataset_index}.pt")

        torch.save({
            "time": time_data.cpu(),
            "dof_pos": dof_pos_buffer.cpu(),
            "dof_vel": dof_vel_buffer.cpu(),
            "des_dof_pos": dof_target_pos_buffer.cpu(),
            "des_dof_vel": dof_target_vel_buffer.cpu(),
            "des_dof_torque": dof_target_commanded_torque_buffer.cpu(),
            "kp": self.Kp,
            "kd": self.Kd,
            **(metadata or {}),
        }, dataset_path)
        print(f"Dataset saved to {dataset_path}")

        self.saved_actual_joints_position = None
        self.saved_actual_joints_velocity = None
        self.saved_desired_joints_position = None
        self.saved_desired_joints_velocity = None
        self.saved_commanded_joints_torque = None

    def compute_control(self):
        # Update the loop time
        start_time = time.perf_counter()
        if(self.last_start_time is not None):
            self.loop_time = (start_time - self.last_start_time)
        self.last_start_time = start_time
        simulation_dt = self.loop_time

        # Safety check to not do anything until a first base and blind state are received
        if(not USE_MUJOCO_SIMULATION and self.first_message_joints_arrived==False):
            return


        # Update the mujoco model
        if(not USE_MUJOCO_SIMULATION):
            if(config.robot == "piper_l"):
                temp_arm_joint_pos = np.append(self.arm_joints_position, self.arm_joints_position[-1]*-1.)
                temp_arm_joint_vel = np.append(self.arm_joints_velocity, self.arm_joints_velocity[-1]*-1.)
            else:
                temp_arm_joint_pos = self.arm_joints_position
                temp_arm_joint_vel = self.arm_joints_velocity

            self.mjData.qpos = copy.deepcopy(temp_arm_joint_pos)
            self.mjData.qvel = copy.deepcopy(temp_arm_joint_vel)
            mujoco.mj_forward(self.mjModel, self.mjData)  


        joints_pos = self.mjData.qpos
        joints_vel = self.mjData.qvel

        # Handling parallel gripper - if present, otherwise 
        # the func will return the same array
        joints_pos = handle_parallel_gripper(joints_pos)
        joints_vel = handle_parallel_gripper(joints_vel)

        if(not self.console.isActivated):
            desired_joint_pos = self.idle_joint_position
            # Impedence Loop
            Kp = self.Kp
            Kd = self.Kd

        elif(self.trajectory is not None):
            # The whole trajectory is precomputed: sample it at the elapsed time
            elapsed_time = time.time() - self.start_collection_time
            desired_joint_pos = self.trajectory.sample(elapsed_time)
            Kp, Kd = self.trajectory.gains(elapsed_time, self.Kp, self.Kd)

            # Collect data
            self._collect_trajectory_data(joints_pos, joints_vel, desired_joint_pos)

            if self.trajectory.done(elapsed_time, joints_pos):
                self._finish_trajectory(desired_joint_pos)

        else:
            desired_joint_pos = self.idle_joint_position
            # Impedence Loop
            Kp = self.Kp*0.0
            Kd = self.Kd*0.0

        
        if USE_MUJOCO_SIMULATION:
            error_joints_pos = desired_joint_pos - joints_pos        
            self.mjData.ctrl = Kp * (error_joints_pos) - Kd * joints_vel
            mujoco.mj_step(self.mjModel, self.mjData)


        # Publish the desired joint positions to the control signal --------------------------------
        control_signal_msg = ControlSignal()
        #control_signal_msg.header.timestamp = int(self.get_clock().now().nanoseconds)
        #control_signal_msg.header.sequence_id = int(self.sequence_id % 1000)  # To avoid overflow, we reset the sequence id after it reaches a certain value
        #self.sequence_id += 1
        control_signal_msg.joints_position = np.array(desired_joint_pos).flatten().tolist()
        control_signal_msg.joints_velocity = np.zeros(len(desired_joint_pos)).tolist()
        control_signal_msg.joints_torques = np.zeros(len(desired_joint_pos)).tolist()
        control_signal_msg.kp = np.array(Kp).flatten().tolist()
        control_signal_msg.kd = np.array(Kd).flatten().tolist()
        self.publisher_control_signal.publish(control_signal_msg)
        
        
        
        # Render the simulation -----------------------------------------------------------------------------------
        if USE_MUJOCO_RENDER:
            RENDER_FREQ = 30
            # Render only at a certain frequency -----------------------------------------------------------------
            if time.time() - self.last_render_time > 1.0 / RENDER_FREQ:
                if self.setpoint_preview is not None:
                    self._render_calibration_setpoint_preview()
                else:
                    self.viewer.sync()
                self.last_render_time = time.time()





#---------------------------
if __name__ == '__main__':
    print('Hello from your lovely data_collection routine.')
    rclpy.init()
    data_collection_node = Data_Collection_Node()

    rclpy.spin(data_collection_node)
    data_collection_node.destroy_node()
    rclpy.shutdown()

    print("Data-Collection-Node is stopped")
    exit(0)
