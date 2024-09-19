import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import MultivariateNormal

class AdversarialAgent(nn.Module):
    def __init__(self, obs_dim, action_dim, attack_space_dim, control_mode='P', lr=3e-4, gamma=0.99, eps_clip=0.2):
        """
        control_mode: 'P' for Position control, 'V' for Velocity control, 'T' for Torque control
        """
        super(AdversarialAgent, self).__init__()
        # Set control mode (affects how the agent interprets its actions)
        self.control_mode = control_mode
        
        # Define the policy network (used to generate perturbations and external forces)
        self.policy_fc1 = nn.Linear(obs_dim, 64)
        self.policy_fc2 = nn.Linear(64, 64)
        self.action_fc = nn.Linear(64, action_dim)  # Perturb robot's actions
        self.force_fc = nn.Linear(64, attack_space_dim)  # Apply forces in the environment
        
        # Value network (used to estimate the value of a state)
        self.value_fc1 = nn.Linear(obs_dim, 64)
        self.value_fc2 = nn.Linear(64, 64)
        self.value_head = nn.Linear(64, 1)
        
        # Optimizer
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        
        # PPO-specific parameters
        self.gamma = gamma  # Discount factor
        self.eps_clip = eps_clip  # Clipping for PPO
        
        # Action covariance for stochastic action sampling
        self.cov_var = torch.full(size=(action_dim,), fill_value=0.5)  # Variance for action distribution
        self.cov_mat = torch.diag(self.cov_var)
    
    def forward(self, obs):
        """Forward pass to get perturbations and forces."""
        x = torch.relu(self.policy_fc1(obs))
        x = torch.relu(self.policy_fc2(x))
        
        # Perturbations (depending on control mode: positions, velocities, or torques)
        perturbations = torch.tanh(self.action_fc(x))  # Perturb actions (bounded between -1 and 1)
        external_forces = torch.tanh(self.force_fc(x))  # External force application (bounded between -1 and 1)
        
        return perturbations, external_forces
    
    def compute_value(self, obs):
        """Forward pass to compute the value of the state."""
        # Ensure the input `obs` has shape [batch_size, obs_dim]
        assert len(obs.shape) == 2, f"Expected 2D input for obs, got {obs.shape}"

        v = torch.relu(self.value_fc1(obs))
        v = torch.relu(self.value_fc2(v))
        value = self.value_head(v)
        
        # Output should have shape [batch_size], so we squeeze to remove the last dimension
        return value.squeeze(-1)  # Ensure it returns a 1D tensor [batch_size]


    def compute_action(self, obs):
        """
        Sample actions (perturbations and external forces) from the policy.
        The action interpretation will depend on the control mode ('P', 'V', or 'T').
        """
        perturbations, external_forces = self.forward(obs)
        
        # Create a normal distribution for the actions
        dist = MultivariateNormal(perturbations, self.cov_mat)
        action_sample = dist.sample()
        action_logprob = dist.log_prob(action_sample)
        
        # Depending on the control mode, the action will be interpreted differently:
        if self.control_mode == 'P':
            # In Position control, actions represent target positions
            action_sample = torch.clamp(action_sample, -1.0, 1.0)  # Clamp target positions
        elif self.control_mode == 'V':
            # In Velocity control, actions represent target velocities
            action_sample = torch.clamp(action_sample, -1.0, 1.0)  # Velocity bounds
        elif self.control_mode == 'T':
            # In Torque control, actions represent torques
            action_sample = torch.clamp(action_sample, -1.0, 1.0)  # Torque bounds
        
        return action_sample, action_logprob, external_forces

    def update(self, old_obs, old_actions, log_probs, rewards, dones, new_obs):
        # Compute discounted returns
        returns = self.compute_returns(rewards, dones)
        
        # Compute advantages
        values = self.compute_value(old_obs)  # No squeeze necessary here
        advantages = returns - values.detach()

        # Policy loss with PPO clipping
        new_actions, new_log_probs, _ = self.compute_action(old_obs)
        ratios = torch.exp(new_log_probs - log_probs)  # Ratio between new and old policy
        
        # Clipped surrogate objective
        surr1 = ratios * advantages
        surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()
        
        # Compute value loss
        new_values = self.compute_value(new_obs)  # Ensure it's shaped correctly with `.view(-1)`
        print(f"new_values shape: {new_values.shape}, returns shape: {returns.shape}")
        
        # Ensure both new_values and returns have the same shape
        value_loss = nn.MSELoss()(new_values, returns)
        
        # Total loss
        loss = policy_loss + 0.5 * value_loss
        
        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()



    def compute_returns(self, rewards, dones):
        """Compute discounted returns for each step."""
        returns = []
        G_t = 0
        for r, done in zip(reversed(rewards), reversed(dones)):
            if done:
                G_t = 0  # Reset return at the end of an episode
            G_t = r + self.gamma * G_t
            returns.insert(0, G_t)
        return torch.tensor(returns)
