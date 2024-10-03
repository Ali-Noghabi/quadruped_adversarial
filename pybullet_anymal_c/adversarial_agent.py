import torch
import torch.nn as nn
import torch.optim as optim

# Define the policy network
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

# Define the value network
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

# Define the PPO Agent class
class PPOAgent:
    def __init__(self, obs_dim, action_dim, lr=3e-4, gamma=0.99, clip_epsilon=0.2, entropy_coef=0.01):
        self.gamma = gamma
        self.clip_epsilon = clip_epsilon
        self.entropy_coef = entropy_coef
        
        # Networks
        self.policy_net = PolicyNetwork(obs_dim, action_dim)
        self.value_net = ValueNetwork(obs_dim)
        
        # Optimizers
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=lr)
    
    def select_action(self, state):
        """ Select an action based on the policy network output """
        state = torch.FloatTensor(state).unsqueeze(0)
        action = self.policy_net(state)
        return action.detach().cpu().numpy()[0]
    
    def compute_ppo_loss(self, old_log_probs, states, actions, advantages, returns):
        """ Compute the PPO loss with clipped objective """
        new_log_probs = self.compute_log_probs(states, actions)
        
        # Debugging print statements to track tensor sizes
        # print(f"States shape: {states.shape}")
        # print(f"Actions shape: {actions.shape}")
        # print(f"Advantages shape: {advantages.shape}")
        # print(f"Returns shape: {returns.shape}")
        # print(f"New log probs shape: {new_log_probs.shape}")
        # print(f"Old log probs shape: {old_log_probs.shape}")
        
        # Compute the ratio
        ratio = torch.exp(new_log_probs - old_log_probs)
        
        # print(f"Ratio shape: {ratio.shape}")
        
        # Broadcast the advantages to match the shape of ratio
        advantages = advantages.unsqueeze(1)  # Now advantages shape is [batch_size, 1]
        
        # print(f"Broadcasted Advantages shape: {advantages.shape}")

        # Surrogate loss
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * advantages

        # print(f"Surr1 shape: {surr1.shape}")
        # print(f"Surr2 shape: {surr2.shape}")
        
        policy_loss = -torch.min(surr1, surr2).mean()

        # Value loss
        values = self.value_net(states)
        value_loss = (returns - values).pow(2).mean()
        
        # Entropy loss (for exploration)
        entropy_loss = -self.entropy_coef * self.compute_entropy(states).mean()
        
        # Lipschitz regularization
        lipschitz_loss = self.lipschitz_regularization()
        
        # Total loss
        total_loss = policy_loss + value_loss + entropy_loss + lipschitz_loss
        return total_loss

    
    def update(self, trajectories):
        """ Update the policy and value network using trajectories from the environment """
        # Extract the necessary data from trajectories
        states = torch.FloatTensor(trajectories['states'])
        actions = torch.FloatTensor(trajectories['actions'])
        old_log_probs = torch.FloatTensor(trajectories['log_probs'])
        returns = torch.FloatTensor(trajectories['returns'])
        advantages = torch.FloatTensor(trajectories['advantages'])

        # Compute the loss
        loss = self.compute_ppo_loss(old_log_probs.detach(), states.detach(), actions.detach(), advantages.detach(), returns.detach())
        
        # Update policy network
        self.policy_optimizer.zero_grad()
        loss.backward()  # Compute the gradients
        self.policy_optimizer.step()  # Apply the gradients

        # Compute value loss (optional: separate value updates)
        value_loss = (returns - self.value_net(states).squeeze()).pow(2).mean()
        self.value_optimizer.zero_grad()
        value_loss.backward()  # Compute value gradients
        self.value_optimizer.step()  # Apply value gradients

    def compute_log_probs(self, states, actions):
        """ Compute the log probabilities of actions for PPO update """
        return torch.distributions.Normal(self.policy_net(states), 1).log_prob(actions)
    
    def compute_entropy(self, states):
        """ Compute entropy for exploration """
        return torch.distributions.Normal(self.policy_net(states), 1).entropy()
    
    def lipschitz_regularization(self):
        """ Compute Lipschitz regularization to ensure smooth actions """
        lipschitz_penalty = 0
        for param in self.policy_net.parameters():
            lipschitz_penalty += torch.norm(param, p=float('inf'))  # Apply infinity norm to penalize large changes
        return lipschitz_penalty

