from adversarial_agent import PPOAgent
from environment import AnymalEnv
import torch

def process_trajectories(agent, trajectories, gamma=0.99):
    """ Process the collected trajectories to compute returns and advantages """
    states, actions, rewards, next_states, dones = zip(*trajectories)
    
    # Convert to torch tensors directly (instead of using np.array)
    states = torch.stack(states)
    actions = torch.stack(actions)
    rewards = torch.tensor(rewards, dtype=torch.float32)
    dones = torch.tensor(dones, dtype=torch.float32)

    # Compute returns and advantages
    returns = []
    G = 0
    for reward, done in zip(reversed(rewards), reversed(dones)):
        G = reward + gamma * G * (1 - done)
        returns.insert(0, G)
    
    returns = torch.FloatTensor(returns)
    
    # Calculate advantages (G_t - V_t)
    values = agent.value_net(states).detach().squeeze()  # Squeeze to ensure it's 1D
    advantages = returns - values  # Now this will be 1D

    # Print debugging information for tensor shapes
    # print(f"Values shape: {values.shape}")  # Should be [136]
    # print(f"Returns shape: {returns.shape}")  # Should be [136]
    # print(f"Advantages shape: {advantages.shape}")  # Should be [136]

    # Collect necessary data for PPO update
    return {
        'states': states,
        'actions': actions,
        'returns': returns,
        'advantages': advantages,
        'log_probs': agent.compute_log_probs(states, actions)
    }

def train(agent, env, num_episodes=1000, rollout_length=2048):
    """ Training loop for the agent """
    for episode in range(num_episodes):
        trajectories = []
        state = torch.FloatTensor(env.reset())  # Ensure state is a tensor

        for t in range(rollout_length):
            action = agent.select_action(state)
            action = torch.FloatTensor(action)  # Ensure action is a tensor
            print(f'action: {action}')
            next_state, reward, done, _ = env.step(action)  # Convert action to NumPy for the environment
            next_state = torch.FloatTensor(next_state)  # Ensure next_state is a tensor
            
            # Store everything as tensors
            trajectories.append((state, action, reward, next_state, done))
            
            state = next_state
            print(f'episode {episode} step {t} reward {reward} done {done}')
            if done:
                break
        
        # Process trajectories and update the agent
        processed_data = process_trajectories(agent, trajectories)
        agent.update(processed_data)
        
        # Print the episode summary
        if episode % 10 == 0:
            print(f"Episode {episode} complete")
            
if __name__ == "__main__":
    # Set up the environment
    env = AnymalEnv(gui=True)

    # Define observation and action dimensions based on your environment
    obs_dim = 2 * env.n_joints + 13  # Observation size from your environment's observation tensor
    action_dim = 2 * env.n_joints    # Number of actions (joint positions + forces)

    # Initialize the PPO agent
    agent = PPOAgent(obs_dim, action_dim)

    # Train the agent
    train(agent, env, num_episodes=1000, rollout_length=2048)

    # Save the trained model
    torch.save(agent.policy_net.state_dict(), "pretrained/policy_network.pth")
    torch.save(agent.value_net.state_dict(), "pretrained/value_network.pth")
