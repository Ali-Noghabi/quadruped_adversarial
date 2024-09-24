import torch
import numpy as np
import random
from environment import AnymalEnv  # Import the environment class

def random_actions(n_joints, destabilize=False):
    """ 
    Generate random actions for both positions and forces.
    If destabilize=True, generate aggressive actions to make the robot collapse.
    """
    # Random joint positions
    if destabilize:
        target_positions = torch.tensor([random.uniform(-1.5, 1.5) for _ in range(n_joints)])
        forces = torch.tensor([random.uniform(50, 100) for _ in range(n_joints)])  # Higher forces for destabilization
    else:
        target_positions = torch.tensor([random.uniform(-0.2, 0.2) for _ in range(n_joints)])
        forces = torch.tensor([random.uniform(10, 50) for _ in range(n_joints)])  # Lower forces for stability

    # Concatenate positions and forces to form the complete action
    return torch.cat((target_positions, forces))

def run_simulation():
    """ Main script to run the Anymal environment with random actions """
    # Create the environment
    env = AnymalEnv(gui=True)  # Set gui=True for visualization
    
    # Reset the environment
    obs = env.reset()
    print("Initial observation:", obs)

    steps = 0
    max_steps = 10000  # Set a limit for max steps
    done = False
    
    while not done and steps < max_steps:
        if steps % 100 < 50:
            # First half of the steps: Random destabilizing actions
            action = random_actions(env.n_joints, destabilize=True)
        else:
            # Second half: Normal random actions
            action = random_actions(env.n_joints, destabilize=False)
        
        # Take a step in the environment
        obs, reward, done, _ = env.step(action)
        
        # Print the results for this step
        # print(f"Step: {steps}, Action: {action}")
        # print(f"Observation: {obs}")
        print(f"Step: {steps} Reward: {reward}, Done: {done}")
        
        steps += 1
        env.render()  # Render the simulation (optional)

        # Break early if robot becomes unalive
        if done:
            print("Robot has fallen or become unalive!")
            break
    
    # Close the environment
    env.close()

if __name__ == "__main__":
    run_simulation()