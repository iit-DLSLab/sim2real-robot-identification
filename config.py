import numpy as np

robot = 'go2'  # 'aliengo', 'go1', 'go2', 'b2', 'hyqreal2', 'z1', 'piper_l', 'a2', "pegasus"


USE_MUJOCO_SIMULATION = False
USE_MUJOCO_RENDER = True
# ----------------------------------------------------------------------------------------------------------------
if(robot == "aliengo"):
    Kp = np.array([25., 25., 25.,
                   25., 25., 25.,
                   25., 25., 25.,
                   25., 25., 25.])
                   
    Kd = np.array([0.5, 0.5, 0.5,
                   0.5, 0.5, 0.5,
                   0.5, 0.5, 0.5,
                   0.5, 0.5, 0.5])

elif(robot == "go2"):
    # Order FL, FR, RL, RR
    Kp = np.array([20., 20., 20.,
                   20., 20., 20.,
                   20., 20., 20.,
                   20., 20., 20.])
                   
    Kd = np.array([1.5, 1.5, 1.5,
                   1.5, 1.5, 1.5,
                   1.5, 1.5, 1.5,
                   1.5, 1.5, 1.5])

elif(robot == "b2"):
    # Order FL, FR, RL, RR
    Kp = np.array([100., 100., 100.,
                   100., 100., 100.,
                   100., 100., 100.,
                   100., 100., 100.])

    Kd = np.array([5., 5., 5.,
                   5., 5., 5.,
                   5., 5., 5.,
                   5., 5., 5.])

elif(robot == "a2"):
    # Order FL, FR, RL, RR
    Kp = np.array([40., 40., 40.,
                   40., 40., 40.,
                   40., 40., 40.,
                   40., 40., 40.])

    Kd = np.array([2., 2., 2.,
                   2., 2., 2.,
                   2., 2., 2.,
                   2., 2., 2.])

elif(robot == "hyqreal2"):
    # Order FL, FR, RL, RR
    Kp = np.array([175., 175., 175.,
                   175., 175., 175.,
                   175., 175., 175.,
                   175., 175., 175.])

    Kd = np.array([20., 20., 20.,
                   20., 20., 20.,
                   20., 20., 20.,
                   20., 20., 20.])

elif(robot == "pegasus"):
    # Order FL, FR, RL, RR
    Kp = np.array([200., 200., 200.,
                   200., 200., 200.,
                   200., 200., 200.,
                   200., 200., 200.])

    Kd = np.array([10., 10., 10.,
                   10., 10., 10.,
                   10., 10., 10.,
                   10., 10., 10.])

elif(robot =="z1"):
    # Order joint1-joint2-joint3-joint4-joint5-joint6-gripper
    Kp = np.array([80., 120., 120.,
                   80., 80., 80.,
                   80.])

    Kd = np.array([2.5, 2.5, 2.5,
                   2.5, 2.5, 2.5,
                   2.5])

elif(robot =="piper_l"):
    # Order joint1-joint2-joint3-joint4-joint5-joint6-gripper
    Kp = np.array([125.0, 125.0, 125.0,
                   7.5, 7.5, 7.5,
                   10.0])

    Kd = np.array([10.0, 10.0, 10.0,
                   0.6, 0.6, 0.6,
                   0.8])

else:
    raise ValueError(f"Robot {robot} not supported")

frequency_collection = 200#hz