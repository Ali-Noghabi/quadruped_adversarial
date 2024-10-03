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

        # Calculate the reward
        reward = self.calculate_reward(observation, target_positions)

        # Check for done conditions
        done = self.check_done(observation)

        return observation, reward, done, {}

    def calculate_reward(self, observation, action):
        """ Calculate the reward based on the paper's formulation, using PyTorch tensors """

        # Extract relevant observation components
        joint_positions = observation[:self.n_joints]  # First n_joints are joint positions
        joint_velocities = observation[self.n_joints:self.n_joints*2]  # Joint velocities
        base_orientation = observation[-10:-6]  # Quaternion orientation (x, y, z, w)
        base_angular_velocity = observation[-6:-3]  # Angular velocity (omega_x, omega_y, omega_z)

        # Extract torque information (if available) or calculate from action if needed
        joint_torques = self.calculate_joint_torques(joint_velocities)

        # ---------------------- Paper-based Reward Components ----------------------

        # 1. Orientation Penalty (based on g_z component of gravity)
        g_z = base_orientation[2]  # z-component of orientation (gravity vector's effect)
        orientation_penalty = g_z  # Encourage bad orientation by increasing reward as g_z decreases

        # 2. Shaking Penalty (based on angular velocity in x and y axes)
        shaking_penalty = torch.norm(base_angular_velocity[:2])  # Penalize for angular velocity in roll/pitch axes (instability)

        # 3. Torque Penalty (penalize excessive joint torques)
        # The soft torque limits are applied here, so if torques exceed the limit, apply penalty
        torque_limits = torch.tensor([1.0] * self.n_joints, device=self.device)  # Define your torque limits
        torque_penalty = torch.sum(torch.relu(torch.abs(joint_torques) - torque_limits))  # Penalize if torque exceeds limits

        # 4. Lipschitz Regularization (for smooth adversarial actions)
        # Implementing Lipschitz regularization to prevent sharp/oscillating actions
        lipschitz_penalty = torch.sum(torch.abs(torch.diff(action, dim=0)))  # Penalize large differences between successive actions

        # 5. Base Penalty (penalize the robot for being alive)
        base_penalty = -1  # Constant penalty for the robot being alive to encourage destabilization

        # ---------------------- Combining All Reward Components ----------------------
        # Final reward calculation, combining all the components
        reward = base_penalty
        reward += orientation_penalty  # Add orientation penalty (encourages instability)
        reward += shaking_penalty  # Add shaking penalty (instability)
        reward += torque_penalty  # Add torque penalty (excessive torque)
        reward += lipschitz_penalty  # Add Lipschitz regularization (smooth adversarial actions)

        return reward


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
