import torch
import numpy as np
import random
from environment import AnymalEnv  # Import the environment class
import matplotlib.pyplot as plt

def random_actions(n_joints, destabilize=False):
    """ 
    Generate random actions for both positions and forces.
    If destabilize=True, generate aggressive actions to make the robot collapse.
    """
    # Random joint positions
    if destabilize:
        target_positions = torch.tensor([random.uniform(-1.5, 1.5) for _ in range(n_joints)])
        forces = torch.tensor([random.uniform(50, 150) for _ in range(n_joints)])  # Higher forces for destabilization
    else:
        target_positions = torch.tensor([random.uniform(-0.2, 0.2) for _ in range(n_joints)])
        forces = torch.tensor([random.uniform(10, 50) for _ in range(n_joints)])  # Lower forces for stability

    # Concatenate positions and forces to form the complete action
    return torch.cat((target_positions, forces))

def update_plot(rewards, gz_values, shaking_rewards, torque_penalties, axs):
    """ Function to handle the plotting updates """
    # Clear the current figure
    for ax in axs:
        ax.clear()

    # Plot each metric in its own subplot
    axs[0].plot(rewards, label='Reward', color='blue')
    axs[0].set_title("Reward over Time")
    axs[0].set_ylabel("Reward")
    axs[0].legend()

    axs[1].plot(gz_values, label='gz', color='orange')
    axs[1].set_title("gz over Time")
    axs[1].set_ylabel("gz")
    axs[1].legend()

    axs[2].plot(shaking_rewards, label='Shaking Reward', color='green')
    axs[2].set_title("Shaking Reward over Time")
    axs[2].set_ylabel("Shaking Reward")
    axs[2].legend()

    axs[3].plot(torque_penalties, label='Torque Penalty', color='red')
    axs[3].set_title("Torque Penalty over Time")
    axs[3].set_ylabel("Torque Penalty")
    axs[3].legend()

    plt.xlabel("Step")
    plt.pause(1./240)  # Pause for a short time to update the plot

def run_simulation(chart=False):
    """ Main script to run the Anymal environment with random actions """
    # Create the environment
    env = AnymalEnv(gui=True, device='cuda')  # Set gui=True for visualization
    
    # Reset the environment
    obs = env.reset()
    print("Initial observation:", obs)
    steps = 0
    max_steps = 10000  # Set a limit for max steps
    done = False

    # Initialize lists to store values
    rewards = []
    gz_values = []
    shaking_rewards = []
    torque_penalties = []

    if chart:
        plt.ion()  # Turn on interactive mode
        fig, axs = plt.subplots(4, 1, figsize=(10, 10))  # Create 4 subplots in a single column
        plt.show(block=False)  # Show the plot without blocking

    while not done and steps < max_steps:
        if steps % 100 < 10:
            action = random_actions(env.n_joints, destabilize=True)
        else:
            action = random_actions(env.n_joints, destabilize=False)

        obs, reward, done, _, gz, shaking_reward, torque_penalty = env.step(action)

        print(f"Step: {steps} Reward: {reward}, Done: {done}, gz: {gz}, Shaking Reward: {shaking_reward}, Torque Penalty: {torque_penalty}")
        steps += 1
        env.render()

        if chart:
            # Update the lists
            rewards.append(reward.cpu().item())
            gz_values.append(gz.cpu().item())
            shaking_rewards.append(shaking_reward.cpu().item())
            torque_penalties.append(torque_penalty.cpu().item())
            
            # Call the update_plot function
            update_plot(rewards, gz_values, shaking_rewards, torque_penalties, axs)

        if done:
            print("Robot has fallen or become unalive!")
            # break

    # Close the environment
    env.close()

    if chart:
        plt.ioff()  # Turn off interactive mode
        plt.show()  # Show the final plot

if __name__ == "__main__":
    run_simulation(chart=True)
