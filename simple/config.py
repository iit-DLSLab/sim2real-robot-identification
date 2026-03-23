robot = 'go2'  # 'aliengo', 'go1', 'go2', 'b2', 'hyqreal1', 'hyqreal2', 'mini_cheetah' 

# ----------------------------------------------------------------------------------------------------------------
if(robot == "aliengo"):
    Kp_walking = 25.
    Kd_walking = 2.
    search_Kp_bounds = [-10., 10.]
    search_Kd_bounds = [-1., 2.]

    friction_static = 0.2
    friction_dynamic = 0.6
    search_friction_static_bounds = [-0.2, 1.0]
    search_friction_dynamic_bounds = [-0.6, 1.0]


elif(robot == "go2"):
    Kp_walking = 20.
    Kd_walking = 1.5
    search_Kp_bounds = [-10., 10.]
    search_Kd_bounds = [-1., 2.]

    friction_static = 0.2
    friction_dynamic = 0.6   
    search_friction_static_bounds = [-0.2, 1.0]
    search_friction_dynamic_bounds = [-0.6, 1.0] 

    armature = 0.01
    search_armature_bounds = [-0.01, 0.04]

elif(robot == "b2"):
    Kp_walking = 20.
    Kd_walking = 1.5
    search_Kp_bounds = [-10., 10.]
    search_Kd_bounds = [-1., 2.]

    friction_static = 0.2
    friction_dynamic = 0.6
    search_friction_static_bounds = [-0.2, 1.0]
    search_friction_dynamic_bounds = [-0.6, 1.0]

    armature = 0.01
    search_armature_bounds = [-0.01, 0.04]

elif(robot == "hyqreal2"):
    Kp_walking = 175.
    Kd_walking = 20.
    search_Kp_bounds = [-50., 50.]
    search_Kd_bounds = [-5., 5.]

    friction_static = 0.2
    friction_dynamic = 0.6
    search_friction_static_bounds = [-0.2, 1.0]
    search_friction_dynamic_bounds = [-0.6, 1.0]
    
    armature = 0.01
    search_armature_bounds = [-0.01, 0.04]

else:
    raise ValueError(f"Robot {robot} not supported")


Kp_sampling_interval = 0.1
Kd_sampling_interval = 0.1
friction_static_sampling_interval = 0.05
friction_dynamic_sampling_interval = 0.05
armature_sampling_interval = 0.005

optimize_gain = False
optimize_friction = True
optimize_armature = True

isaaclab_physics_dt = 0.005  # seconds
frequency_collection = 200  # Hz

num_iterations = 1
num_best_candidates = 10

datasets_path = "./datasets/go2"