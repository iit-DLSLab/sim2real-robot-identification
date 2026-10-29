import readline
import readchar
import time

import numpy as np
import copy

import mujoco

from excitation_trajectory import DEFAULT_SEED, DEFAULT_SPEED_SCALE

class Console():
    def __init__(self, controller_node):
        self.controller_node = controller_node

        self.isDown = True
        self.setpoint_collection = False
        self.falling_collection = False
        self.trajectory_collection = False
        self.trajectory_type = "chirp"  # "chirp" or "excitation" (see excitation_trajectory.py)
        self.isActivated = False

        # Autocomplete setup
        self.commands = [
            "help", "goDown", "startGeneration", "setKp", "setKd"
        ]
        readline.set_completer(self.complete)
        readline.parse_and_bind("tab: complete")


    def complete(self, text, state):
        options = [cmd for cmd in self.commands if cmd.startswith(text)]
        if state < len(options):
            print(options[state])
            return options[state]
        else:
            return None


    def interactive_command_line(self, ):
        self.print_all_commands()
        while True:
            input_string = input(">>> ")
            try:
                    
                if(input_string == "startCollection"):

                    mode = input("Select the collection mode (setpoint/falling/trajectory): ")
                    if(mode == "setpoint"):
                        self.controller_node.prepare_calibration_setpoint()
                        confirmation = input("Execute this setpoint? [y/N]: ").strip().lower()
                        if confirmation not in ("y", "yes"):
                            self.controller_node.reject_calibration_setpoint()
                            self.setpoint_collection = False
                            self.falling_collection = False
                            self.trajectory_collection = False
                            print("Setpoint rejected. No motion will be executed.")
                            self.isActivated = False
                            continue

                        self.setpoint_collection = True
                        self.falling_collection = False
                        self.trajectory_collection = False
                        self.controller_node.accept_calibration_setpoint()
                        print("Setpoint collection mode activated")
                    elif(mode == "falling"):
                        self.setpoint_collection = False
                        self.falling_collection = True
                        self.trajectory_collection = False
                        print("Falling collection mode activated")
                    elif(mode == "trajectory"):
                        trajectory_type = input("Select the trajectory type (chirp/excitation) [chirp]: ").strip().lower() or "chirp"
                        if(trajectory_type == "excitation"):
                            # Per-joint chirps + posture sweep, generated once and checked before execution
                            seed_text = input(f"Excitation seed [{DEFAULT_SEED}] (use another seed for a validation trajectory): ").strip()
                            seed = int(seed_text) if seed_text else DEFAULT_SEED
                            speed_text = input(f"Speed scale in (0, 1] [{DEFAULT_SPEED_SCALE}] (1 = full speed): ").strip()
                            speed_scale = float(speed_text) if speed_text else DEFAULT_SPEED_SCALE
                            self.controller_node.prepare_excitation_trajectory(seed, speed_scale)
                            confirmation = input("Execute this excitation trajectory? [y/N]: ").strip().lower()
                            if confirmation not in ("y", "yes"):
                                self.controller_node.reject_excitation_trajectory()
                                print("Excitation trajectory rejected. No motion will be executed.")
                                self.isActivated = False
                                continue
                            self.controller_node.accept_excitation_trajectory()
                        elif(trajectory_type != "chirp"):
                            print("Invalid trajectory type selected")
                            self.isActivated = False
                            continue

                        self.trajectory_type = trajectory_type
                        self.setpoint_collection = False
                        self.falling_collection = False
                        self.trajectory_collection = True
                        print(f"Trajectory collection mode activated ({trajectory_type})")
                    else:
                        print("Invalid mode selected")
                        self.isActivated = False
                        continue
                    
                    self.isActivated = True

                elif(input_string == "help"):
                    self.print_all_commands()


                elif(input_string == "setKp"):
                    print("Kp stand_up_and_down: ", self.controller_node.Kp_stand_up_and_down)
                    temp = input("Enter Kp: ")
                    if(temp != ""):
                        self.controller_node.Kp_stand_up_and_down = float(temp)
                    


                elif(input_string == "setKd"):
                    print("Kd stand_up_and_down: ", self.controller_node.Kd_stand_up_and_down)
                    temp = input("Enter Kd: ")
                    if(temp != ""):
                        self.controller_node.Kd_stand_up_and_down = float(temp)


            
            except Exception as e:
                print("Error: ", e)
                print("Invalid Command")
                self.print_all_commands()


    def print_all_commands(self):
        print("\nAvailable Commands")
        print("help: Display all available messages")
        print("goDown: Move the robot down")
        print("startCollection: Start the collection process")
        print("setKp: Set the proportional gain")
        print("setKd: Set the derivative gain\n")
