robot = 'go2'  # 'aliengo', 'go1', 'go2', 'b2', 'hyqreal2', 'z1' 

# ----------------------------------------------------------------------------------------------------------------
if(robot == "aliengo"):
    Kp_walking = 25.
    Kd_walking = 2.


elif(robot == "go2"):
    Kp_walking = 20.
    Kd_walking = 1.5

elif(robot == "b2"):
    Kp_walking = 20.
    Kd_walking = 1.5

elif(robot == "hyqreal2"):
    Kp_walking = 175.
    Kd_walking = 20.

elif(robot =="z1"):
    Kp = 150.
    Kd = 5.

else:
    raise ValueError(f"Robot {robot} not supported")

frequency_collection = 200#hz