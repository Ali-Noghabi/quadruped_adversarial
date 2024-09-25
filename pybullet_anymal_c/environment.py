import pybullet as p
import pybullet_data
import time
import numpy as np
import torch
import matplotlib.pyplot as plt

class AnymalEnv:
    def __init__(self, gui=True, time_step=1./240):
        """ Initialize the simulation environment """
        # Connect to the PyBullet server
        if gui:
            self.client = p.connect(p.GUI)
        else:
            self.client = p.connect(p.DIRECT)
        
        # Load URDF paths
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        
        # Load the environment components
        self.plane_id = p.loadURDF("urdf/plane.urdf")
        self.robot_id = p.loadURDF("urdf/anymal.urdf", [0, 0, 0.5], useFixedBase=False)
        
        # Set gravity and time step
        p.setGravity(0, 0, -9.8)
        p.setTimeStep(time_step)
        
        self.n_joints = p.getNumJoints(self.robot_id)
        
        self.time_step = time_step
        self.done = False

    def reset(self):
        """ Reset the simulation environment """
        p.resetSimulation(self.client)
        p.setGravity(0, 0, -9.8)
        self.plane_id = p.loadURDF("urdf/plane.urdf")
        self.robot_id = p.loadURDF("urdf/anymal.urdf", [0, 0, 0.5], useFixedBase=False)
        
        self.done = False
        return self.get_observation()

    def get_observation(self):
        """ Collect observations from the robot """
        # Joint positions and velocities
        joint_positions = torch.zeros(self.n_joints)
        joint_velocities = torch.zeros(self.n_joints)
        for i in range(self.n_joints):
            joint_state = p.getJointState(self.robot_id, i)
            joint_positions[i] = joint_state[0]
            joint_velocities[i] = joint_state[1]

        # Base position, orientation, velocity
        base_position, base_orientation = p.getBasePositionAndOrientation(self.robot_id)
        base_velocity, base_angular_velocity = p.getBaseVelocity(self.robot_id)
        
        # Combine everything into a single observation tensor
        observation = torch.cat((
            joint_positions,
            joint_velocities,
            torch.tensor(base_position),
            torch.tensor(base_orientation),
            torch.tensor(base_velocity),
            torch.tensor(base_angular_velocity)
        ))

        return observation

    def step(self, action):
        """ 
        Apply the action to the robot and step the simulation. 
        Action should contain both the target joint positions and the forces.
        """
        assert len(action) == 2 * self.n_joints, "Action must include target positions and forces."

        # Split the action into target positions and forces
        target_positions = action[:self.n_joints]
        forces = action[self.n_joints:]

        # Apply action to each joint with the corresponding force
        for i in range(self.n_joints):
            p.setJointMotorControl2(
                bodyUniqueId=self.robot_id,
                jointIndex=i,
                controlMode=p.POSITION_CONTROL,
                targetPosition=target_positions[i],
                force=forces[i]  # Use the force specified in the action
            )
        
        # Step the simulation
        p.stepSimulation()
        
        # Get new observations
        observation = self.get_observation()

        # Calculate the reward
        reward = self.calculate_reward(observation, target_positions)

        # Check for done conditions
        done = self.check_done(observation)

        return observation, reward, done, {}

    def calculate_reward(self, observation, action):
        """ Calculate the reward based on the paper's formulation """
        # Extract relevant observation components
        joint_velocities = observation[:self.n_joints]  # First n_joints are velocities
        base_orientation = observation[-10:-6]  # Quaternion orientation (x, y, z, w)
        base_angular_velocity = observation[-6:-3]  # Angular velocity (omega_x, omega_y, omega_z)
        
        # Reward components
        g_z = base_orientation[2]  # Use z-component of gravity (orientation)
        shaking_penalty = torch.norm(base_angular_velocity[:2])  # Penalize for roll/pitch angular velocity
        torque_penalty = torch.sum(torch.relu(torch.abs(joint_velocities) - 1))  # Penalize excessive joint velocity
        
        # Final reward (negative reward for stability, positive for instability)
        reward = -1  # Penalize for being alive
        reward += g_z  # Encourage instability in the orientation
        reward += shaking_penalty  # Penalize stability in shaking
        reward += torque_penalty  # Penalize safe torque behavior
        
        return reward

    def check_done(self, observation):
        """ Check termination conditions (e.g., if the robot falls over) """
        base_position = observation[-13:-10]  # Base position x, y, z
        # Check if the base height is below a threshold (i.e., the robot has fallen)
        if base_position[2] < 0.2:
            return True
        return False

    def render(self):
        """ Optional: Render the environment for human viewing """
        time.sleep(self.time_step)
    
    def close(self):
        """ Close the simulation """
        p.disconnect(self.client)
