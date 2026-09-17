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


# Set the priority of the process
pid = os.getpid()
print("PID: ", pid)
os.system("renice -n -21 -p " + str(pid))
os.system("echo -20 > /proc/" + str(pid) + "/autogroup")
#for real time, launch it with chrt -r 99 python3 run_controller.py

USE_MUJOCO_RENDER = config.USE_MUJOCO_RENDER
USE_MUJOCO_SIMULATION = config.USE_MUJOCO_SIMULATION


CONTROL_FREQ = config.frequency_collection # Hz 
SETPOINT_REACH_TOLERANCE = 0.1
SETPOINT_REACH_TIMEOUT = 2.0
SETPOINT_STATIC_DURATION = 0.5
INITIAL_CHIRP_TRAJECTORY_DURATION = 3.0


class Data_Collection_Node(Node):
    def __init__(self):
        super().__init__('Data_Collection_Node')
        # Subscribers and Publishers
        self.subscription_blind_state = self.create_subscription(BlindState,"/blind_state_legged", self.get_blind_state_callback, 1)
        self.publisher_control_signal = self.create_publisher(ControlSignal,"/control_signal_legged", 1)
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
        

        self.stand_up_and_down_actions = np.zeros(12)
        keyframe_down_id = mujoco.mj_name2id(self.mjModel, mujoco.mjtObj.mjOBJ_KEY, "down")
        keyframe_sys_id_1 = mujoco.mj_name2id(self.mjModel, mujoco.mjtObj.mjOBJ_KEY, "sys_id_1")
        keyframe_sys_id_2 = mujoco.mj_name2id(self.mjModel, mujoco.mjtObj.mjOBJ_KEY, "sys_id_2")
        
        self.stand_up_and_down_actions = self.mjModel.key_qpos[keyframe_sys_id_1][7:]
        self.idle_joint_positions = self.stand_up_and_down_actions.reshape(4, 3).copy()
        
        self.Kp = config.Kp
        self.Kd = config.Kd

        self.calibration_reference_joint_positions = None
        self.setpoint_reached_time = None
        self.setpoint_preview_active = False
        

        # Chirp Trajectory only variables
        self.chirp_traj_time = INITIAL_CHIRP_TRAJECTORY_DURATION
        self.calibration_reference_calf_trajectory = None
        self.calibration_reference_thigh_trajectory = None
        self.calibration_reference_hip_trajectory = None
        
        self.hip_setpoint2 = self.mjModel.key_qpos[keyframe_sys_id_2][7]
        self.thigh_setpoint2 = self.mjModel.key_qpos[keyframe_sys_id_2][8]
        self.calf_setpoin2 = self.mjModel.key_qpos[keyframe_sys_id_2][9]
        
        self.hip_setpoint1 = self.mjModel.key_qpos[keyframe_sys_id_1][7]
        self.thigh_setpoint1 = self.mjModel.key_qpos[keyframe_sys_id_1][8]
        self.calf_setpoint1 = self.mjModel.key_qpos[keyframe_sys_id_1][9]
        
        
        self.saved_actual_joints_position = None
        self.saved_actual_joints_velocity = None
        self.saved_desired_joints_position = None
        self.saved_desired_joints_velocity = None
        # Interactive Command Line ----------------------------
        from console import Console
        self.console = Console(controller_node=self)
        thread_console = threading.Thread(target=self.console.interactive_command_line)
        thread_console.daemon = True
        thread_console.start()


    def get_blind_state_callback(self, msg):
        
        self.joint_positions = np.array(msg.joints_position)
        self.joint_velocities = np.array(msg.joints_velocity)

        self.first_message_joints_arrived = True

        
    def prepare_calibration_setpoint(self):
        """Initialize calibration setpoint with random values"""
        hip_setpoint = np.random.uniform(-0.0, 0.5)
        thigh_setpoint = np.random.uniform(-1.0, 0.5)
        calf_setpoint = np.random.uniform(-0., 1.5)
        self.calibration_reference_joint_positions = np.zeros((4, 3))
        self.calibration_reference_joint_positions[0] = np.array([0.0+hip_setpoint, 1.21+thigh_setpoint, -2.794+calf_setpoint])
        self.calibration_reference_joint_positions[1] = np.array([0.0-hip_setpoint, 1.21+thigh_setpoint, -2.794+calf_setpoint])
        self.calibration_reference_joint_positions[2] = np.array([0.0+hip_setpoint, 1.21+thigh_setpoint, -2.794+calf_setpoint])
        self.calibration_reference_joint_positions[3] = np.array([0.0-hip_setpoint, 1.21+thigh_setpoint, -2.794+calf_setpoint])
        self.setpoint_preview_active = True
        print("\nProposed setpoint (rows: FL, FR, RL, RR; columns: hip, thigh, calf):")
        print(self.calibration_reference_joint_positions)
        if USE_MUJOCO_RENDER:
            print("The proposed pose is shown in the MuJoCo viewer.")

    def accept_calibration_setpoint(self):
        self.start_collection_time = time.time()
        self.setpoint_reached_time = None
        self.setpoint_preview_active = False

    def reject_calibration_setpoint(self):
        self.calibration_reference_joint_positions = None
        self.start_collection_time = None
        self.setpoint_reached_time = None
        self.setpoint_preview_active = False

    def _render_calibration_setpoint_preview(self):
        """Render the proposed pose without changing the controller/simulation state."""
        saved_qpos = self.mjData.qpos.copy()
        saved_qvel = self.mjData.qvel.copy()

        self.mjData.qpos[7:] = self.calibration_reference_joint_positions.flatten()
        self.mjData.qvel[:] = 0.0
        mujoco.mj_forward(self.mjModel, self.mjData)
        self.viewer.sync()

        self.mjData.qpos[:] = saved_qpos
        self.mjData.qvel[:] = saved_qvel
        mujoco.mj_forward(self.mjModel, self.mjData)

    def _initialize_calibration_setpoint(self):
        """Prepare and immediately accept a setpoint as a defensive fallback."""
        self.prepare_calibration_setpoint()
        self.accept_calibration_setpoint()

    def _initialize_calibration_trajectory(self):
        """Initialize calibration trajectory with random values"""
        print("Generating first a trajectory..")

        # Generate a linear trajectory between actual joint positions and two setpoint

        t = np.linspace(0, self.chirp_traj_time, num=100)

        self.calibration_reference_hip_trajectory = np.interp(
            t,
            [0, self.chirp_traj_time/2, self.chirp_traj_time],
            [self.hip_setpoint1, self.hip_setpoint2, self.hip_setpoint1]
        )

        self.calibration_reference_thigh_trajectory = np.interp(
            t,
            [0, self.chirp_traj_time/2, self.chirp_traj_time],
            [self.thigh_setpoint1, self.thigh_setpoint2, self.thigh_setpoint1]
        )

        self.calibration_reference_calf_trajectory = np.interp(
            t,
            [0, self.chirp_traj_time/2, self.chirp_traj_time],
            [self.calf_setpoint1, self.calf_setpoin2, self.calf_setpoint1]
        )

        self.start_collection_time = time.time()


    def _get_desired_positions_and_gains(self, ):
        """Get desired joint positions and control gains based on collection type"""
        if self.console.setpoint_collection:
            # Setpoint collection: maintain target position
            desired_joint_pos = np.zeros((4, 3))
            desired_joint_pos[0] = copy.deepcopy(self.calibration_reference_joint_positions[0])
            desired_joint_pos[1] = copy.deepcopy(self.calibration_reference_joint_positions[1])
            desired_joint_pos[2] = copy.deepcopy(self.calibration_reference_joint_positions[2])
            desired_joint_pos[3] = copy.deepcopy(self.calibration_reference_joint_positions[3])
            Kp = self.Kp
            Kd = self.Kd
            
        elif self.console.falling_collection:
            # Falling collection: target first, then free fall
            random_coin = np.random.randint(0, 4)
            if(random_coin == 0):
                random_time = 0.0 #No waiting time, immediate fall
            else:
                random_time = 1.8
            if time.time() - self.start_collection_time > (2.0-random_time):
                # Free falling!!
                Kp = self.Kp*0.0
                Kd = self.Kd*0.0
                desired_joint_pos = np.zeros((4, 3))
                desired_joint_pos[0] = np.zeros((1, int(self.mjModel.nu/4))) + 20.
                desired_joint_pos[1] = np.zeros((1, int(self.mjModel.nu/4))) + 20.
                desired_joint_pos[2] = np.zeros((1, int(self.mjModel.nu/4))) + 20.
                desired_joint_pos[3] = np.zeros((1, int(self.mjModel.nu/4))) + 20.
            else:
                # Reach the target
                Kp = self.Kp
                Kd = self.Kd
                desired_joint_pos = np.zeros((4, 3))
                desired_joint_pos[0] = copy.deepcopy(self.calibration_reference_joint_positions[0])
                desired_joint_pos[1] = copy.deepcopy(self.calibration_reference_joint_positions[1])
                desired_joint_pos[2] = copy.deepcopy(self.calibration_reference_joint_positions[2])
                desired_joint_pos[3] = copy.deepcopy(self.calibration_reference_joint_positions[3])
        
        elif self.console.trajectory_collection:
            # Trajectory collection: follow the reference trajectory
            Kp = self.Kp
            Kd = self.Kd

            time_traj = time.time() - self.start_collection_time
            trajectory_index = min(
                int((time_traj / self.chirp_traj_time) * len(self.calibration_reference_hip_trajectory)),
                len(self.calibration_reference_hip_trajectory) - 1,
            )
            desired_joint_pos = np.zeros((4, 3))
            desired_joint_pos[0, 0] = self.calibration_reference_hip_trajectory[trajectory_index]
            desired_joint_pos[0, 1] = self.calibration_reference_thigh_trajectory[trajectory_index]
            desired_joint_pos[0, 2] = self.calibration_reference_calf_trajectory[trajectory_index]
            desired_joint_pos[1] = copy.deepcopy(desired_joint_pos[0])
            if(config.robot == "pegasus"):
                desired_joint_pos[1, 0] = desired_joint_pos[1, 0]
                desired_joint_pos[2] = copy.deepcopy(desired_joint_pos[0])
                desired_joint_pos[3] = copy.deepcopy(desired_joint_pos[0])
                desired_joint_pos[3, 0] = desired_joint_pos[3, 0]  
            else:
                desired_joint_pos[1, 0] = -desired_joint_pos[1, 0]
                desired_joint_pos[2] = copy.deepcopy(desired_joint_pos[0])
                desired_joint_pos[3] = copy.deepcopy(desired_joint_pos[0])
                desired_joint_pos[3, 0] = -desired_joint_pos[3, 0]  
        return desired_joint_pos, Kp, Kd

    def _collect_trajectory_data(self, joints_pos, joints_vel, desired_joint_pos):
        """Collect trajectory data by concatenating and storing joint information"""
        concatenated_actual_joints_position = np.concatenate([joints_pos[0], joints_pos[1],
                                                            joints_pos[2], joints_pos[3]])
        concatenated_actual_joints_velocity = np.concatenate([joints_vel[0], joints_vel[1],
                                                            joints_vel[2], joints_vel[3]])
        concatenated_desired_joints_position = np.concatenate([desired_joint_pos[0], desired_joint_pos[1],
                                                            desired_joint_pos[2], desired_joint_pos[3]])
        concatenated_desired_joints_velocity = np.concatenate([joints_vel[0]*0.0, joints_vel[1]*0.0,
                                                            joints_vel[2]*0.0, joints_vel[3]*0.0])

        if self.saved_actual_joints_position is None:
            self.saved_actual_joints_position = concatenated_actual_joints_position
            self.saved_actual_joints_velocity = concatenated_actual_joints_velocity
            self.saved_desired_joints_position = concatenated_desired_joints_position
            self.saved_desired_joints_velocity = concatenated_desired_joints_velocity
        else:
            self.saved_actual_joints_position = np.vstack([self.saved_actual_joints_position, concatenated_actual_joints_position])
            self.saved_actual_joints_velocity = np.vstack([self.saved_actual_joints_velocity, concatenated_actual_joints_velocity])
            self.saved_desired_joints_position = np.vstack([self.saved_desired_joints_position, concatenated_desired_joints_position])
            self.saved_desired_joints_velocity = np.vstack([self.saved_desired_joints_velocity, concatenated_desired_joints_velocity])

    def _check_collection_complete(self, joints_pos, desired_joint_pos):
        """Check if data collection is complete based on collection type"""
        if self.console.setpoint_collection:
            # Complete when target is reached or timeout
            target_reached = (
                np.linalg.norm(desired_joint_pos[0] - joints_pos[0]) < SETPOINT_REACH_TOLERANCE
                and np.linalg.norm(desired_joint_pos[1] - joints_pos[1]) < SETPOINT_REACH_TOLERANCE
                and np.linalg.norm(desired_joint_pos[2] - joints_pos[2]) < SETPOINT_REACH_TOLERANCE
                and np.linalg.norm(desired_joint_pos[3] - joints_pos[3]) < SETPOINT_REACH_TOLERANCE
            )
            reach_timeout = time.time() - self.start_collection_time > SETPOINT_REACH_TIMEOUT
            if self.setpoint_reached_time is None and (target_reached or reach_timeout):
                self.setpoint_reached_time = time.time()
                reason = "target reached" if target_reached else "reach timeout"
                print(f"Setpoint {reason}; recording {SETPOINT_STATIC_DURATION:.1f} s of static data.")

            return (
                self.setpoint_reached_time is not None
                and time.time() - self.setpoint_reached_time >= SETPOINT_STATIC_DURATION
            )
            
        elif self.console.falling_collection:
            # Complete after falling phase timeout
            return time.time() - self.start_collection_time > 2.5

        elif self.console.trajectory_collection:
            # Complete after trajectory duration
            return time.time() - self.start_collection_time > self.chirp_traj_time

    def _save_trajectory_data(self, dataset_prefix):
        """Save collected trajectory data to file"""

        # HACK
        num_steps = self.saved_actual_joints_position.shape[0]
        time_data = torch.arange(num_steps, device="cpu") / CONTROL_FREQ
        dof_pos_buffer = torch.zeros(num_steps, 12, device="cpu")
        dof_vel_buffer = torch.zeros(num_steps, 12, device="cpu")
        dof_target_pos_buffer = torch.zeros(num_steps, 12, device="cpu")
        dof_target_vel_buffer = torch.zeros(num_steps, 12, device="cpu")
        dof_target_commanded_torque_buffer = torch.zeros(num_steps, 12, device="cpu")

        dof_pos_buffer[:, :] = torch.from_numpy(self.saved_actual_joints_position)
        dof_vel_buffer[:, :] = torch.from_numpy(self.saved_actual_joints_velocity)
        dof_target_pos_buffer[:] = torch.from_numpy(self.saved_desired_joints_position)
        #dof_target_vel_buffer[:] = torch.from_numpy(self.saved_desired_joints_velocity)
        #dof_target_commanded_torque_buffer[:] = torch.from_numpy(self.saved_commanded_torque)

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
            "kp": self.Kp,
            "kd": self.Kd,
        }, dataset_path)
        print(f"Dataset saved to {dataset_path}")

        self.saved_actual_joints_position = None
        self.saved_actual_joints_velocity = None
        self.saved_desired_joints_position = None
        self.saved_desired_joints_velocity = None

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
            self.mjData.qpos[0:3] = copy.deepcopy(self.position)
            self.mjData.qpos[3:7] = copy.deepcopy(self.orientation)
            self.mjData.qvel[0:3] = copy.deepcopy(self.linear_velocity)
            self.mjData.qvel[3:6] = copy.deepcopy(self.angular_velocity)
            self.mjData.qpos[7:] = copy.deepcopy(self.joint_positions)
            self.mjData.qvel[6:] = copy.deepcopy(self.joint_velocities)
            self.mjModel.opt.timestep = simulation_dt
            mujoco.mj_forward(self.mjModel, self.mjData)  
        else:
            self.mjData.qpos[0:3] = np.array([0, 0, 0.4])
            self.mjData.qpos[3:7] = np.array([1, 0, 0, 0])
            self.mjData.qvel[0:3] = np.array([0, 0, 0.])
            self.mjData.qvel[3:6] = np.array([0, 0, 0.0])


        qpos, qvel = self.mjData.qpos, self.mjData.qvel


        joints_pos = np.zeros((4, 3))
        joints_pos[0] = qpos[7:10]
        joints_pos[1] = qpos[10:13]
        joints_pos[2] = qpos[13:16]
        joints_pos[3] = qpos[16:19]

        joints_vel = np.zeros((4, 3))
        joints_vel[0] = qvel[6:9]
        joints_vel[1] = qvel[9:12]
        joints_vel[2] = qvel[12:15]
        joints_vel[3] = qvel[15:18]
    
        if(not self.console.isActivated):
            desired_joint_pos = copy.deepcopy(self.idle_joint_positions)

            # Impedence Loop
            Kp = self.Kp
            Kd = self.Kd
            

        elif(self.console.isActivated and (self.console.setpoint_collection or self.console.falling_collection)):
            
            # Initialize setpoint if needed
            if self.calibration_reference_joint_positions is None:
                self._initialize_calibration_setpoint()

            # Get desired joint positions and control gains based on collection type
            desired_joint_pos, Kp, Kd = self._get_desired_positions_and_gains()
            
            # Collect data
            self._collect_trajectory_data(joints_pos, joints_vel, desired_joint_pos)
            
            # Check if collection is complete
            collection_complete = self._check_collection_complete(joints_pos, desired_joint_pos)
            if collection_complete:
                dataset_prefix = "setpoint" if self.console.setpoint_collection else "falling"
                if self.console.setpoint_collection:
                    self.idle_joint_positions = copy.deepcopy(desired_joint_pos)
                self._save_trajectory_data(dataset_prefix)
                self.reject_calibration_setpoint()
                if self.console.setpoint_collection:
                    self.console.setpoint_collection = False
                    self.console.isActivated = False
                    print("Setpoint collection completed.")

        elif(self.console.isActivated and self.console.trajectory_collection):
            # Initialize setpoint if needed
            if self.calibration_reference_hip_trajectory is None:
                self._initialize_calibration_trajectory()

            # Get desired joint positions and control gains based on collection type
            desired_joint_pos, Kp, Kd = self._get_desired_positions_and_gains()

            # Collect data
            self._collect_trajectory_data(joints_pos, joints_vel, desired_joint_pos)

            # Check if collection is complete            
            collection_complete = self._check_collection_complete(joints_pos, desired_joint_pos)
            if collection_complete:
                self.calibration_reference_hip_trajectory = None
                self.calibration_reference_thigh_trajectory = None
                self.calibration_reference_calf_trajectory = None
                if(self.chirp_traj_time >= 0.6):
                    self.chirp_traj_time -= 0.2 # Reduce trajectory time for next trajectory
                else:
                    self.chirp_traj_time -= 0.1 # Reduce trajectory time for next trajectory slower
                if(self.chirp_traj_time < 0.2):
                    self._save_trajectory_data("trajectory")
                    self.chirp_traj_time = INITIAL_CHIRP_TRAJECTORY_DURATION
                    self.console.trajectory_collection = False
                    print("Trajectory collection completed.")
        else:
            desired_joint_pos = copy.deepcopy(self.idle_joint_positions)

            # Impedence Loop
            Kp = self.Kp*0.0
            Kd = self.Kd*0.0
            
        
        if USE_MUJOCO_SIMULATION:
            for j in range(10): #Hardcoded for now, if RL is 50Hz, this runs the simulation at 500Hz
                qpos, qvel = self.mjData.qpos, self.mjData.qvel
                joints_pos[0] = qpos[7:10]
                joints_pos[1] = qpos[10:13]
                joints_pos[2] = qpos[13:16]
                joints_pos[3] = qpos[16:19]

                joints_vel[0] = qvel[6:9]
                joints_vel[1] = qvel[9:12]
                joints_vel[2] = qvel[12:15]
                joints_vel[3] = qvel[15:18]

                error_joints_pos = np.zeros((4, 3))
                error_joints_pos[0] = desired_joint_pos[0] - joints_pos[0]
                error_joints_pos[1] = desired_joint_pos[1] - joints_pos[1]
                error_joints_pos[2] = desired_joint_pos[2] - joints_pos[2]
                error_joints_pos[3] = desired_joint_pos[3] - joints_pos[3]
                
                tau = np.zeros((4, 3))
                tau[0] = Kp[0:3] * (error_joints_pos[0]) - Kd[0:3] * joints_vel[0]
                tau[1] = Kp[3:6] * (error_joints_pos[1]) - Kd[3:6] * joints_vel[1]
                tau[2] = Kp[6:9] * (error_joints_pos[2]) - Kd[6:9] * joints_vel[2]
                tau[3] = Kp[9:12] * (error_joints_pos[3]) - Kd[9:12] * joints_vel[3]

                action = np.zeros(self.mjModel.nu)
                action = tau.flatten()
                self.mjData.ctrl = action
                mujoco.mj_step(self.mjModel, self.mjData)


        # Publish the desired joint positions to the control signal --------------------------------
        control_signal_msg = ControlSignal()
        control_signal_msg.timestamp = float(self.get_clock().now().nanoseconds)
        control_signal_msg.sequence_id = int(self.sequence_id % 1000)  # To avoid overflow, we reset the sequence id after it reaches a certain value
        self.sequence_id += 1
        control_signal_msg.joints_position = np.array([desired_joint_pos[0], desired_joint_pos[1], desired_joint_pos[2], desired_joint_pos[3]]).flatten().tolist()
        control_signal_msg.joints_velocity = np.zeros(12).tolist()
        control_signal_msg.joints_torques = np.zeros(12).tolist()
        control_signal_msg.kp = Kp.tolist()
        control_signal_msg.kd = Kd.tolist()
        self.publisher_control_signal.publish(control_signal_msg)
        
        
        
        # Render the simulation -----------------------------------------------------------------------------------
        if USE_MUJOCO_RENDER:
            RENDER_FREQ = 30
            # Render only at a certain frequency -----------------------------------------------------------------
            if time.time() - self.last_render_time > 1.0 / RENDER_FREQ:
                if self.setpoint_preview_active:
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
