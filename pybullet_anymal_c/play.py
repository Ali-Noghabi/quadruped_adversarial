import torch
import torch.nn as nn
import numpy as np
from environment import AnymalEnv  # Import the environment class
import matplotlib.pyplot as plt

# Define the policy and value network architecture
class PolicyNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        action = torch.tanh(self.fc3(x))  # Output actions between -1 and 1
        return action

class ValueNetwork(nn.Module):
    def __init__(self, input_dim):
        super(ValueNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        value = self.fc3(x)  # Output a single value
        return value

def load_network(path, network_class, input_dim, output_dim=None, device='cpu'):
    """ Load a trained network (either policy or value) """
    if output_dim:
        network = network_class(input_dim, output_dim).to(device)
    else:
        network = network_class(input_dim).to(device)
    
    network.load_state_dict(torch.load(path, map_location=device))
    network.eval()  # Set the network to evaluation mode
    return network

def run_simulation(policy_net, value_net=None, chart=False):
    """ Main script to run the Anymal environment with the trained policy network """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = 'cpu'
    # Create the environment and dynamically find the number of joints
    env = AnymalEnv(gui=True, device=device)
    n_joints = env.n_joints  # Dynamically get the number of joints
    obs_dim = 2 * n_joints + 13  # Observation size (adjust dynamically)
    action_dim = 2 * n_joints    # Action size (joint positions + forces)
    
    # Reset the environment
    obs = env.reset().to(device)
    print(f"Initial observation: {obs}, Number of Joints: {n_joints}")

    steps = 0
    max_steps = 10000  # Set a limit for max steps
    done = False

    if chart:
        rewards = []  # List to store rewards
        values = []  # List to store state values if value_net is used
        plt.ion()  # Turn on interactive mode
        plt.show(block=False)  # Show the plot without blocking

    while steps < max_steps:
        # Use the policy network to generate actions based on the observation
        action = policy_net(obs).cpu().detach()  # Detach the tensor to avoid gradients
        
        # Take a step in the environment
        obs, reward, done, _ = env.step(action)
        obs = obs.to(device)  # Ensure the new observation is on the correct device
        
        print(f"Step: {steps} Reward: {reward}, Done: {done}")
        
        # Optionally, evaluate the state using the value network
        if value_net:
            state_value = value_net(obs).cpu().detach().item()
            print(f"Step: {steps}, Estimated Value of State: {state_value}")
        
        steps += 1
        env.render()  # Render the simulation (optional)
        
        if chart:
            # Update the plot
            rewards.append(reward.cpu().item())  # Append reward to list
            if value_net:
                values.append(state_value)  # Append value to list if value_net is available
            plt.clf()  # Clear the current figure
            plt.plot(rewards, label='Reward')
            if value_net:
                plt.plot(values, label='State Value')
            plt.title("Reward and Value over Time")  # Set title
            plt.xlabel("Step")  # Set x-axis label
            plt.ylabel("Value")  # Set y-axis label
            plt.legend()
            plt.pause(1./240)  # Pause for a short time to update the plot
        
        # Break early if robot becomes unalive
        if done:
            print("Robot has fallen or become unalive!")
            break
    
    # Close the environment
    env.close()
    
    if chart:
        plt.ioff()  # Turn off interactive mode
        plt.show()  # Show the final plot

if __name__ == "__main__":
    # Load the trained policy and value networks
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = 'cpu'
    # Dynamically set observation and action dimensions based on environment (after reset)
    dummy_env = AnymalEnv(gui=False)
    n_joints = dummy_env.n_joints  # Dynamically get the number of joints
    obs_dim = 2 * n_joints + 13  # Observation size based on joints
    action_dim = 2 * n_joints    # Action size (joint positions + forces)
    dummy_env.close()  # Close the dummy environment
    # Load the policy and value networks
    policy_net = load_network("pretrained/policy_network.pth", PolicyNetwork, obs_dim, action_dim, device=device)
    value_net = load_network("pretrained/value_network.pth", ValueNetwork, obs_dim, device=device)

    # Run the simulation using the trained policy network
    run_simulation(policy_net, value_net=value_net, chart=True)
