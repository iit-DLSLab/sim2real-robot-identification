# Description: This script is used to run the policy on the real robot

# Authors:
# Giulio Turrisi

import rclpy 
from rclpy.node import Node 
from dls2_interface.msg import ArmState, ArmTrajectoryGenerator

import time
import numpy as np
np.set_printoptions(precision=3, suppress=True)

import threading
import copy
import torch

import sys
import os 
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir_path+"/mujoco/")
sys.path.append(dir_path+"/../")
sys.path.append(dir_path+"/../scripts/rsl_rl")

import mujoco
import mujoco.viewer
import config


# Set the priority of the process
pid = os.getpid()
print("PID: ", pid)
os.system("renice -n -21 -p " + str(pid))
os.system("echo -20 > /proc/" + str(pid) + "/autogroup")
#for real time, launch it with chrt -r 99 python3 run_controller.py

USE_MUJOCO_RENDER = True
USE_MUJOCO_SIMULATION = True


CONTROL_FREQ = config.frequency_collection # Hz 


class Data_Collection_Node(Node):
    def __init__(self):
        super().__init__('Data_Collection_Node')
        # Subscribers and Publishers
        self.subscription_arm_state = self.create_subscription(ArmState,"/arm_state", self.get_arm_blind_state_callback, 1)
        self.publisher_arm_trajectory_generator = self.create_publisher(ArmTrajectoryGenerator,"/arm_trajectory_generator", 1)
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
        self.mjModel = mujoco.MjModel.from_xml_path(dir_path + "/models/" + config.robot + "/" + config.robot + ".xml")
        self.mjData = mujoco.MjData(self.mjModel)

        if(USE_MUJOCO_RENDER):
            self.viewer = mujoco.viewer.launch_passive(
                self.mjModel,
                self.mjData,
                show_left_ui=False,
                show_right_ui=False,
            )
            self.last_render_time = time.time()


        keyframe_id = mujoco.mj_name2id(self.mjModel, mujoco.mjtObj.mjOBJ_KEY, "home")
        self.home_position = self.mjModel.key_qpos[keyframe_id]
        self.goal_position = copy.deepcopy(self.home_position)

        if(config.robot == "z1"):
            self.home_position += 0.2
            self.home_position[2] = -0.5
            
            self.goal_position += 0.9
            self.goal_position[2] = -1.0
        
        
        self.Kp = config.Kp
        self.Kd = config.Kd

        self.calibration_reference_joint_positions = None
        

        # Chirp Trajectory only variables
        self.chirp_traj_time = 3.0
        self.calibration_reference_trajectory = None
        
        self.saved_actual_joints_position = None
        self.saved_actual_joints_velocity = None
        self.saved_desired_joints_position = None
        self.saved_desired_joints_velocity = None
        self.saved_commanded_joints_torque = None
        self.num_traj_saved = 0



        # Interactive Command Line ----------------------------
        from console import Console
        self.console = Console(controller_node=self)
        thread_console = threading.Thread(target=self.console.interactive_command_line)
        thread_console.daemon = True
        thread_console.start()


    def get_arm_blind_state_callback(self, msg):        
        self.arm_joints_position = np.array(msg.joints_position)
        self.arm_joints_velocity = np.array(msg.joints_velocity)

        self.first_message_arm_joints_arrived = True

        

    def _initialize_calibration_trajectory(self):
        """Initialize calibration trajectory with random values"""
        print("Generating first a trajectory..")

        # Generate a linear trajectory between actual joint positions and two setpoint

        t = np.linspace(0, self.chirp_traj_time, num=100)
        
        # Interpolate for each joint separately
        self.calibration_reference_trajectory = np.zeros((100, len(self.home_position)))
        for joint_idx in range(len(self.home_position)):
            self.calibration_reference_trajectory[:, joint_idx] = np.interp(
                t,
                [0, self.chirp_traj_time/2, self.chirp_traj_time],
                [self.home_position[joint_idx], self.goal_position[joint_idx], self.home_position[joint_idx]]
            )
        

        self.start_collection_time = time.time()


    def _get_desired_positions_and_gains(self, ):
        """Get desired joint positions and control gains based on collection type"""
        
        if self.console.setpoint_collection:
            print("not implemented")
            
        elif self.console.falling_collection:
            print("not implemented")
        
        elif self.console.trajectory_collection:
            # Trajectory collection: follow the reference trajectory
            Kp = self.Kp
            Kd = self.Kd

            time_traj = self.start_collection_time - time.time()
            desired_joint_pos = self.calibration_reference_trajectory[int((time_traj/self.chirp_traj_time)*100)]

        return desired_joint_pos, Kp, Kd

    def _collect_trajectory_data(self, joints_pos, joints_vel, desired_joint_pos):
        """Collect trajectory data by concatenating and storing joint information"""
        
        concatenated_actual_joints_position = joints_pos
        concatenated_actual_joints_velocity = joints_vel
        concatenated_desired_joints_position = desired_joint_pos
        concatenated_desired_joints_velocity = desired_joint_pos*0.0
        
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

    def _check_collection_complete(self, joints_pos, desired_joint_pos):
        """Check if data collection is complete based on collection type"""
        
        if self.console.setpoint_collection:
            # Complete when target is reached or timeout
            print("not implemented")
            return False 
            
        elif self.console.falling_collection:
            # Complete after falling phase timeout
            print("not implemented")
            return False 


        elif self.console.trajectory_collection:
            # Complete after trajectory duration
            return time.time() - self.start_collection_time > self.chirp_traj_time

    def _save_trajectory_data(self):
        """Save collected trajectory data to file"""

        # Saving to file trajectory
        """desired_fps = CONTROL_FREQ
        data = {
            "joints_list": ["FL_hip_joint", "FL_thigh_joint", "FL_calf_joint", 
                                "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint", 
                                "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint", 
                                "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint"],
            "actual_joints_position": self.saved_actual_joints_position,
            "actual_joints_velocity": self.saved_actual_joints_velocity,
            "desired_joints_position": self.saved_desired_joints_position,
            "desired_joints_velocity": self.saved_desired_joints_velocity,
            "fps": desired_fps,
        }
        # Save the data to an .npy file
        output_file = "datasets/" + config.robot + f"/traj_{self.num_traj_saved}.npy"
        np.save(output_file, data)"""

        # HACK
        num_steps = self.saved_actual_joints_position.shape[0]
        duration = num_steps/CONTROL_FREQ
        time_data = torch.linspace(0, duration, steps=num_steps, device="cpu")
        dof_pos_buffer = torch.zeros(num_steps, 7, device="cpu")
        dof_vel_buffer = torch.zeros(num_steps, 7, device="cpu")
        dof_target_pos_buffer = torch.zeros(num_steps, 7, device="cpu")
        dof_target_vel_buffer = torch.zeros(num_steps, 7, device="cpu")
        dof_target_commanded_torque_buffer = torch.zeros(num_steps, 7, device="cpu")
        breakpoint()
        dof_pos_buffer[:, :] = torch.from_numpy(self.saved_actual_joints_position)
        dof_vel_buffer[:, :] = torch.from_numpy(self.saved_actual_joints_velocity)
        dof_target_pos_buffer[:, :] = torch.from_numpy(self.saved_desired_joints_position)
        #dof_target_vel_buffer[:, :] = torch.from_numpy(self.saved_desired_joints_velocity)
        dof_target_commanded_torque_buffer[:, :] = torch.from_numpy(self.saved_commanded_joints_torque)

        torch.save({
            "time": time_data.cpu(),
            "dof_pos": dof_pos_buffer.cpu(),
            "dof_vel": dof_vel_buffer.cpu(),
            "des_dof_pos": dof_target_pos_buffer.cpu(),
            "des_dof_vel": dof_target_vel_buffer.cpu(),
            "des_dof_torque": dof_target_commanded_torque_buffer.cpu(),
        }, "datasets/" + config.robot + f"/traj_{self.num_traj_saved}.pt")

        self.num_traj_saved += 1
        self.saved_actual_joints_position = None
        self.saved_actual_joints_velocity = None
        self.saved_desired_joints_position = None
        self.saved_desired_joints_velocity = None
        self.saved_commanded_joints_torque = None

        input("Press enter to continue.")


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
            self.mjData.qpos = copy.deepcopy(self.arm_joints_position)
            self.mjData.qvel = copy.deepcopy(self.arm_joints_velocity)
            mujoco.mj_forward(self.mjModel, self.mjData)  


        joints_pos = self.mjData.qpos
        joints_vel = self.mjData.qvel

        if(not self.console.isActivated):
            desired_joint_pos = self.home_position
            # Impedence Loop
            Kp = self.Kp
            Kd = self.Kd
            

        elif(self.console.isActivated and (self.console.setpoint_collection or self.console.falling_collection)):
            
            print("not implemented")

        elif(self.console.isActivated and self.console.trajectory_collection):
            # Initialize setpoint if needed
            if self.calibration_reference_trajectory is None:
                self._initialize_calibration_trajectory()

            # Get desired joint positions and control gains based on collection type
            desired_joint_pos, Kp, Kd = self._get_desired_positions_and_gains()

            # Collect data
            self._collect_trajectory_data(joints_pos, joints_vel, desired_joint_pos)

            # Check if collection is complete            
            collection_complete = self._check_collection_complete(joints_pos, desired_joint_pos)
            if collection_complete:
                self.calibration_reference_trajectory = None
                self.chirp_traj_time -= 0.2 # Reduce trajectory time for next trajectory
                if(self.chirp_traj_time < 0.4):
                    self._save_trajectory_data()
                    self.console.trajectory_collection = False
                    print("Trajectory collection completed.")
        else:
            desired_joint_pos = self.home_position
            # Impedence Loop
            Kp = self.Kp*0.0
            Kd = self.Kd*0.0

        
        if USE_MUJOCO_SIMULATION:
            for j in range(10): #Hardcoded for now, if RL is 50Hz, this runs the simulation at 500Hz

                error_joints_pos = desired_joint_pos - joints_pos                
                self.mjData.ctrl = Kp * (error_joints_pos) - Kd * joints_vel
                mujoco.mj_step(self.mjModel, self.mjData)


        # Publish the desired joint positions to the trajectory generator --------------------------------
        arm_trajectory_generator_msg = ArmTrajectoryGenerator()
        arm_trajectory_generator_msg.timestamp = float(self.get_clock().now().nanoseconds)
        arm_trajectory_generator_msg.desired_arm_joints_position = desired_joint_pos[0:-1].flatten().tolist()
        arm_trajectory_generator_msg.desired_arm_joints_velocity = np.zeros(6).tolist()
        arm_trajectory_generator_msg.desired_arm_gripper_position = desired_joint_pos[-1]
        arm_trajectory_generator_msg.desired_arm_gripper_velocity = 0.0
        arm_trajectory_generator_msg.arm_kp = (np.ones(6) * Kp).tolist()
        arm_trajectory_generator_msg.arm_kd = (np.ones(6) * Kd).tolist()
        self.publisher_arm_trajectory_generator.publish(arm_trajectory_generator_msg)
        
        
        
        # Render the simulation -----------------------------------------------------------------------------------
        if USE_MUJOCO_RENDER:
            RENDER_FREQ = 30
            # Render only at a certain frequency -----------------------------------------------------------------
            if time.time() - self.last_render_time > 1.0 / RENDER_FREQ:
                self.viewer.sync()





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