import mujoco
import numpy as np
from mujoco import MjData, MjModel
import mujoco.viewer as viewer

# Load the A1 model description
from robot_descriptions import a1_mj_description

# Load the A1 model from the provided path
model = MjModel.from_xml_path(a1_mj_description.MJCF_PATH)

# Create simulation data
data = MjData(model)

# Define target joint positions for a simple walking gait
# These positions will correspond to different leg movements for walking
# Adjust these values based on the structure of the A1 robot's joints
target_positions = np.array([0.0, 0.6, -1.2, 0.0, 0.6, -1.2, 0.0, 0.6, -1.2, 0.0, 0.6, -1.2])

# PD control gains (Kp and Kd)
Kp = 100  # Proportional gain
Kd = 2    # Derivative gain

# Function to step through the simulation
def simulate_steps(n_steps=1000):
    for _ in range(n_steps):
        # Implement a basic PD controller for each joint
        for i in range(model.nu):  # model.nu gives the number of actuators
            pos_error = target_positions[i] - data.qpos[i]   # Position error
            vel_error = -data.qvel[i]  # Velocity error (we want to reduce velocity to zero)
            control_signal = Kp * pos_error + Kd * vel_error  # PD control law
            data.ctrl[i] = control_signal  # Apply the control signal to the actuator

        # Step the simulation
        mujoco.mj_step(model, data)
        
        # Print joint positions (optional for debugging)
        print("Joint positions:", data.qpos)

# Optional: Render the simulation using MuJoCo's viewer
def render_simulation():
    viewer.launch(model, data)

# Simulate steps with the walking controller
simulate_steps(10000)

# Uncomment the following line to render the simulation (optional)
render_simulation()
