import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical

class AdversarialPolicy(nn.Module):
    def __init__(self, obs_dim, action_dim):
        super(AdversarialPolicy, self).__init__()
        self.fc1 = nn.Linear(obs_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, action_dim)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        actions = torch.tanh(self.fc3(x))  # Output actions as perturbations
        return actions

class PPOAdversary:
    def __init__(self, obs_dim, action_dim, lr=3e-4, gamma=0.99):
        self.policy = AdversarialPolicy(obs_dim, action_dim)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.gamma = gamma
        self.eps_clip = 0.2  # PPO clip parameter
        self.memory = []  # Store transitions for training

    def observe_and_act(self, obs):
        # Generate adversarial perturbations based on the observations
        obs = obs.clone().detach().float()
        action = self.policy(obs)
        return action

    def store_transition(self, transition):
        # Store state, action, reward, next_state for training
        self.memory.append(transition)

    def compute_advantages(self, rewards, dones):
        # Compute discounted rewards and advantages
        discounted_rewards = []
        running_reward = 0
        for reward, done in zip(reversed(rewards), reversed(dones)):
            if done:
                running_reward = 0  # Reset at episode end
            running_reward = reward + self.gamma * running_reward
            discounted_rewards.insert(0, running_reward)
        return torch.tensor(discounted_rewards, dtype=torch.float32)

    def train(self):
        if len(self.memory) == 0:
            return  # Skip training if no data collected

        # Prepare data for training (obs, actions, rewards, etc.)
        obs_batch = torch.stack([m['obs'].clone().detach().float() for m in self.memory])
        action_batch = torch.stack([m['action'] for m in self.memory])
        reward_batch = self.compute_advantages([m['reward'] for m in self.memory],
                                               [m['done'] for m in self.memory])

        # PPO: Clip the policy gradient and perform a policy update
        old_action_log_probs = [m['log_prob'] for m in self.memory]
        policy_loss = []
        for log_prob, reward in zip(old_action_log_probs, reward_batch):
            ratio = torch.exp(log_prob - log_prob.detach())  # PPO ratio
            clipped_ratio = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip)
            policy_loss.append(-torch.min(ratio * reward, clipped_ratio * reward))

        policy_loss = torch.stack(policy_loss).mean()
        self.optimizer.zero_grad()
        policy_loss.backward()
        self.optimizer.step()

        # Clear memory after training
        self.memory = []

# Example usage: Running the adversary in an environment
def run_adversary_simulation(env, ppo_adversary, max_steps=50000):
    obs = env.reset()
    done = False
    total_reward = 0
    step_count = 0

    while not done and step_count < max_steps:
        # Get adversary's action based on observation
        action = ppo_adversary.observe_and_act(obs)

        # Step the environment with the adversarial action
        next_obs, reward, done, _ = env.step(action.detach().numpy())

        # Log the transition
        ppo_adversary.store_transition({
            'obs': obs,
            'action': action,
            'reward': -reward,  # Adversary wants to destabilize (negative reward)
            'done': done,
            'log_prob': action.log()  # For PPO updates
        })

        obs = next_obs
        total_reward += reward
        step_count += 1

        # Every few steps, train the adversary policy
        if step_count % 20 == 0:
            ppo_adversary.train()

    # Final training at the end of the episode
    ppo_adversary.train()

    print(f"Total destabilization reward (negative of robot's stability): {total_reward}")

# Example: Replace this with the actual environment setup
class DummyEnv:
    def __init__(self):
        self.observation_space = torch.rand(10)
        self.action_space = torch.rand(4)
    
    def reset(self):
        return torch.rand(10)
    
    def step(self, action):
        action = torch.tensor(action) 
        obs = torch.rand(10)
        reward = -torch.norm(action)  # Simple reward based on action magnitude
        done = torch.rand(1).item() > 0.9  # Random done condition
        return obs, reward, done, {}

if __name__ == "__main__":
    env = DummyEnv()  # Replace this with your actual Isaac Gym environment
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    # Initialize PPO adversary
    ppo_adversary = PPOAdversary(obs_dim, action_dim)
    
    # Run simulation with live interaction and training
    run_adversary_simulation(env, ppo_adversary)
