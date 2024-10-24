import pybullet as p
import pybullet_data
import time
import torch

class AnymalEnv:
    def __init__(self, gui=True, time_step=1./240, device='cpu'):
        """ Initialize the simulation environment """
        # Connect to the PyBullet server
        self.device = device  # Allows for potential future GPU support
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
        """ Collect observations from the robot and return them as PyTorch tensors """
        # Joint positions and velocities
        joint_positions = torch.zeros(self.n_joints, device=self.device)
        joint_velocities = torch.zeros(self.n_joints, device=self.device)
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
            torch.tensor(base_position, device=self.device),
            torch.tensor(base_orientation, device=self.device),
            torch.tensor(base_velocity, device=self.device),
            torch.tensor(base_angular_velocity, device=self.device)
        ))

        return observation

    def step(self, action):
        """ 
        Apply the action to the robot and step the simulation.
        Action should be a tensor containing both the target joint positions and the forces.
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
                targetPosition=target_positions[i].item(),  # Extract scalar from tensor
                force=forces[i].item()  # Extract scalar from tensor
            )
        
        # Step the simulation
        p.stepSimulation()
        
        # Get new observations
        observation = self.get_observation()

        torques = torch.zeros(self.n_joints, device=self.device)
        for i in range(self.n_joints):
            joint_state = p.getJointState(self.robot_id, i)
            torques[i] = joint_state[3]  # Joint torque is the fourth item in the tuple returned by getJointState

        # Calculate the reward
        reward , gz , shaking_reward , torque_penalty = self.calculate_reward(observation, torques)

        # Check for done conditions
        done = self.check_done(observation)

        # return observation, reward, done, {} , gz , shaking_reward , torque_penalty
        return observation, reward, done, {} 

    def calculate_reward(self, observation, torques):
        # Constants (you'll need to choose appropriate values)
        corient = 1.0  # Adjust based on testing
        cshake = 0.05   # Adjust based on testing
        ctorque = 0.05  # Adjust based on testing
        torque_limits = torch.tensor([20] * self.n_joints, device=self.device)  # Example limits, adjust as necessary

        # Extract base orientation quaternion
        quaternion = observation[-10:-6]
        _, _, z = p.getEulerFromQuaternion(quaternion.tolist())  # Assuming this gives us roll, pitch, yaw
        
        # Calculate gz from the pitch
        gz = -torch.cos(torch.tensor(z, device=self.device))
        
        # Extract angular velocities
        base_angular_velocity = observation[-3:]
        wx, wy = base_angular_velocity[0], base_angular_velocity[1]
        
        # Shaking Reward
        shaking_reward = cshake * (wx**2 + wy**2)

        # Torque Penalty
        # print(torques)
        relative_torques = torch.abs(torques) / torque_limits
        torque_penalty = ctorque * torch.sum(torch.relu(relative_torques - 1))

        # Combine all parts of the reward
        reward = corient * gz + shaking_reward + torque_penalty

            # Debugging print statement
        print(f"gz: {gz.item()}, "
          f"Shaking Reward: {shaking_reward.item()}, Torque Penalty: {torque_penalty}, "
          f"Total Reward: {reward.item()}")
        return reward ,gz, shaking_reward , torque_penalty

    def calculate_joint_torques(self, joint_velocities):
        """Calculate joint torques based on joint velocities. This is simplified as your setup may vary."""
        # In a real system, torques would be obtained via sensors or calculated based on system dynamics
        # Here we assume joint torques are proportional to velocities for simplicity.
        torques = joint_velocities * 0.1  # Simplified relation between velocity and torque
        return torques

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
