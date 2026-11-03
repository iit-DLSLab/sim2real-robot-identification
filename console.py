import readline
import readchar
import time

import numpy as np
import copy

import mujoco

from collection_trajectories import DEFAULT_SEED, DEFAULT_SPEED_SCALE, DEFAULT_CHIRP_SPEED_SCALE

class Console():
    def __init__(self, controller_node):
        self.controller_node = controller_node

        self.isDown = True
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

                    mode = input("Select the collection mode (setpoint/falling/trajectory): ").strip().lower()
                    # Every mode is a trajectory of collection_trajectories.py, generated before execution
                    options = {}
                    if(mode == "trajectory"):
                        kind = input("Select the trajectory type (chirp/excitation) [chirp]: ").strip().lower() or "chirp"
                        if(kind == "excitation"):
                            # Per-joint chirps + posture sweep
                            seed_text = input(f"Excitation seed [{DEFAULT_SEED}] (use another seed for a validation trajectory): ").strip()
                            options["seed"] = int(seed_text) if seed_text else DEFAULT_SEED
                            speed_text = input(f"Speed scale in (0, 1] [{DEFAULT_SPEED_SCALE}] (1 = full speed): ").strip()
                            options["speed_scale"] = float(speed_text) if speed_text else DEFAULT_SPEED_SCALE
                        elif(kind == "chirp"):
                            # Every period of the chirp is divided by the speed scale (1 = original chirp)
                            speed_text = input(f"Chirp speed scale in (0, 1] [{DEFAULT_CHIRP_SPEED_SCALE}] (1 = original chirp): ").strip()
                            options["speed_scale"] = float(speed_text) if speed_text else DEFAULT_CHIRP_SPEED_SCALE
                        else:
                            print("Invalid trajectory type selected")
                            self.isActivated = False
                            continue
                    elif(mode in ("setpoint", "falling")):
                        kind = mode
                    else:
                        print("Invalid mode selected")
                        self.isActivated = False
                        continue

                    self.controller_node.prepare_trajectory(kind, **options)
                    # Falling repeats by itself until the node is stopped: no confirmation, as before
                    if(kind != "falling"):
                        confirmation = input(f"Execute this {kind}? [y/N]: ").strip().lower()
                        if confirmation not in ("y", "yes"):
                            self.controller_node.cancel_trajectory()
                            print(f"{kind.capitalize()} rejected. No motion will be executed.")
                            self.isActivated = False
                            continue

                    self.controller_node.start_trajectory()
                    print(f"{kind.capitalize()} collection activated")
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
