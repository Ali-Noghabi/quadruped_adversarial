import torch
from adversarial_agent2 import AdversarialAgent
from dummy_legged_environment import DummyLeggedRobotEnv
# Initialize the environment with a control mode ('P' for position control, 'V' for velocity, 'T' for torque)
env = DummyLeggedRobotEnv(control_mode='T')  # Change 'T' to 'P' or 'V' for different modes
obs_dim = env.num_observations
action_dim = env.num_actions
attack_space_dim = 3  # For external forces (x, y, z)

# Initialize the adversarial agent
adversarial_agent = AdversarialAgent(obs_dim=obs_dim, action_dim=action_dim, attack_space_dim=attack_space_dim)

# Training Loop
def train_adversarial_agent(env, adversarial_agent, num_steps=1000, batch_size=64):
    obs = env.reset()  # Initial observations
    all_rewards = []
    log_probs = []
    old_actions = []
    dones_list = []

    for step in range(num_steps):
        # Generate robot actions (dummy policy for now, can be replaced with a learned policy)
        robot_actions = torch.zeros(env.num_envs, env.num_actions)

        # Adversarial agent generates perturbations and external forces
        perturbations, log_prob, external_forces = adversarial_agent.compute_action(obs)
        
        # Apply perturbations to robot's actions
        perturbed_actions = robot_actions + perturbations

        # Step the environment with perturbed actions
        obs, reward, dones, infos = env.step(perturbed_actions)
        
        # Compute adversarial reward based on whether the robot fell
        adversarial_reward = 1.0 if dones.any() else 0.0
        
        # Store rewards, log_probs, and actions for PPO update
        all_rewards.append(adversarial_reward)
        log_probs.append(log_prob)  # Append tensor log_prob
        old_actions.append(perturbations)
        dones_list.append(dones)

        # Every batch_size steps, update the adversarial agent
        if step % batch_size == 0:
            # Convert lists to tensors for the update step
            log_probs_tensor = torch.stack(log_probs)
            all_rewards_tensor = torch.tensor(all_rewards)
            old_actions_tensor = torch.stack(old_actions)
            dones_tensor = torch.stack(dones_list)

            adversarial_agent.update(
                old_obs=obs,
                old_actions=old_actions_tensor,
                log_probs=log_probs_tensor,
                rewards=all_rewards_tensor,
                dones=dones_tensor,
                new_obs=obs
            )
            
            # Reset buffers
            all_rewards = []
            log_probs = []
            old_actions = []
            dones_list = []
        
        # Reset environment if the robot falls
        if dones.any():
            obs = env.reset()

    print("Training completed.")

# Start the training
train_adversarial_agent(env, adversarial_agent)
