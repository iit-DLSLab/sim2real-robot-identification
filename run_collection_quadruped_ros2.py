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
import gc
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
JOINT_NAMES = [f"{leg}_{joint}" for leg in ("FL", "FR", "RL", "RR") for joint in ("hip", "thigh", "calf")]


class Data_Collection_Node(Node):
    def __init__(self):
        super().__init__('Data_Collection_Node')
        # Subscribers and Publishers
        self.subscription_blind_state = self.create_subscription(BlindState,"/blind_state", self.get_blind_state_callback, 1)
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

        # Collection trajectory (see collection_trajectories.py): prepared and checked by the operator,
        # then executed once accepted
        self.pending_trajectory = None
        self.trajectory = None
        self.setpoint_preview = None  # posture shown in the viewer while waiting for confirmation

        # Joint box of the chirp and of the excitation (front-left leg)
        self.leg_setpoint_1 = self.mjModel.key_qpos[keyframe_sys_id_1][7:10].copy()
        self.leg_setpoint_2 = self.mjModel.key_qpos[keyframe_sys_id_2][7:10].copy()

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

        
    def prepare_trajectory(self, kind, **options):
        """Generate the trajectory of a collection mode and print its summary, without commanding the robot.

        kind is setpoint, falling, chirp or excitation; options are the arguments of its generator (e.g. the
        speed scale). The chirp and the excitation stay inside the joint box of the sys_id_1 and sys_id_2 keyframes.
        """
        # same hip convention for all trajectories: pegasus does not mirror the hips of the right legs
        mirror_right_hips = config.robot != "pegasus"
        if kind == "setpoint":
            trajectory = collection_trajectories.quadruped_setpoint_trajectory(CONTROL_FREQ)
        elif kind == "falling":
            trajectory = collection_trajectories.quadruped_falling_trajectory(CONTROL_FREQ)
        elif kind == "chirp":
            trajectory = collection_trajectories.quadruped_chirp_trajectory(
                self.leg_setpoint_1, self.leg_setpoint_2, CONTROL_FREQ,
                mirror_right_hips=mirror_right_hips, **options,
            )
        elif kind == "excitation":
            # starts from and returns to the current desired posture
            trajectory = collection_trajectories.quadruped_excitation_trajectory(
                self.leg_setpoint_1, self.leg_setpoint_2, start=self.idle_joint_positions.flatten(),
                frequency=CONTROL_FREQ, mirror_right_hips=mirror_right_hips, **options,
            )
        else:
            raise ValueError(f"Unknown collection mode: {kind}")
        print(trajectory.describe(JOINT_NAMES))
        if kind == "setpoint":
            self.setpoint_preview = trajectory.target.reshape(4, 3)
            if USE_MUJOCO_RENDER:
                print("The proposed pose is shown in the MuJoCo viewer.")
        self.pending_trajectory = trajectory

    def start_trajectory(self):
        """Start timing only after the operator has approved the trajectory."""
        self.trajectory = self.pending_trajectory
        self.pending_trajectory = None
        self.setpoint_preview = None
        self.start_collection_time = time.time()

    def cancel_trajectory(self):
        self.pending_trajectory = None
        self.setpoint_preview = None

    def _finish_trajectory(self, desired_joint_pos):
        """Save the collected data and choose what the node does next."""
        self._save_trajectory_data(self.trajectory.name, self.trajectory.metadata)
        if self.trajectory.name == "falling":
            # falling repeats, with a new random setpoint, until the node is stopped
            self.prepare_trajectory("falling")
            self.start_trajectory()
            return
        if self.trajectory.name == "setpoint":
            # hold the setpoint
            self.idle_joint_positions = copy.deepcopy(desired_joint_pos)
            self.console.isActivated = False
        # after the chirp and the excitation the node stays activated without a trajectory: zero gains
        print(f"{self.trajectory.label.capitalize()} collection completed.")
        self.trajectory = None

    def _render_calibration_setpoint_preview(self):
        """Render the proposed pose without changing the controller/simulation state."""
        saved_qpos = self.mjData.qpos.copy()
        saved_qvel = self.mjData.qvel.copy()

        self.mjData.qpos[7:] = self.setpoint_preview.flatten()
        self.mjData.qvel[:] = 0.0
        mujoco.mj_forward(self.mjModel, self.mjData)
        self.viewer.sync()

        self.mjData.qpos[:] = saved_qpos
        self.mjData.qvel[:] = saved_qvel
        mujoco.mj_forward(self.mjModel, self.mjData)

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

    def _save_trajectory_data(self, dataset_prefix, metadata=None):
        """Save collected trajectory data to file, with optional extra entries (e.g. the excitation seed)"""

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
            **(metadata or {}),
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

        elif(self.trajectory is not None):
            # The whole trajectory is precomputed: sample it at the elapsed time
            elapsed_time = time.time() - self.start_collection_time
            desired_joint_pos = self.trajectory.sample(elapsed_time).reshape(4, 3)
            Kp, Kd = self.trajectory.gains(elapsed_time, self.Kp, self.Kd)

            # Collect data
            self._collect_trajectory_data(joints_pos, joints_vel, desired_joint_pos)

            if self.trajectory.done(elapsed_time, joints_pos.flatten()):
                self._finish_trajectory(desired_joint_pos)

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
        #control_signal_msg.header.timestamp = int(self.get_clock().now().nanoseconds)
        #control_signal_msg.header.sequence_id = int(self.sequence_id % 1000)  # To avoid overflow, we reset the sequence id after it reaches a certain value
        #self.sequence_id += 1
        control_signal_msg.control_mode = 1 
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
                if self.setpoint_preview is not None:
                    self._render_calibration_setpoint_preview()
                else:
                    self.viewer.sync()
                self.last_render_time = time.time()




def create_executor():
    # EventsExecutor dispatches the callbacks in C++ (rclpy >= Kilted), with much less overhead than the
    # default SingleThreadedExecutor. It does not exist on Humble, where we fall back to SingleThreadedExecutor
    try:
        from rclpy.experimental import EventsExecutor
        print("Using EventsExecutor")
        return EventsExecutor()
    except ImportError:
        from rclpy.executors import SingleThreadedExecutor
        print("EventsExecutor not available in this ROS 2 distro, using SingleThreadedExecutor")
        return SingleThreadedExecutor()


#---------------------------
if __name__ == '__main__':
    print('Hello from your lovely data_collection routine.')
    rclpy.init()
    data_collection_node = Data_Collection_Node()
    # Move everything allocated at startup (modules, model, node) out of the garbage collector:
    # otherwise its full collections scan all of them and stall the control loop for tens of ms
    gc.collect()
    gc.freeze()

    rclpy.spin(data_collection_node, create_executor())
    data_collection_node.destroy_node()
    rclpy.shutdown()

    print("Data-Collection-Node is stopped")
    exit(0)
