import torch
import torch.nn as nn

class PPOActorCritic(nn.Module):
    """
    PPO Actor-Critic network for environments with continuous action spaces.
    
    This model outputs:
      - a stochastic policy (Normal distribution over actions)
      - a value estimate of the input state

    Args:
        obs_dim (int): Dimension of observation space
        act_dim (int): Dimension of action space

    Forward Input:
        x (torch.Tensor): Batch of observations, shape [batch_size, obs_dim]

    Forward Output:
        action_dist (torch.distributions.Normal): Distribution over actions
        value (torch.Tensor): Estimated state value, shape [batch_size, 1]
    """
    def __init__(self, obs_dim: int, act_dim: int):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
        )
        self.policy_mean = nn.Linear(128, act_dim)
        self.policy_logstd = nn.Parameter(torch.zeros(act_dim))
        self.value_head = nn.Linear(128, 1)
    
    def forward(self, x: torch.Tensor):
        """
        Forward pass through the network.

        Args:
            x (torch.Tensor): Input observation tensor [batch_size, obs_dim]

        Returns:
            action_dist (Normal): Gaussian distribution over actions
            value (torch.Tensor): State value estimate [batch_size, 1]
        """
        latent = self.shared(x)
        value = self.value_head(latent)
        action_mean = self.policy_mean(latent)
        action_std = self.policy_logstd.exp().expand_as(action_mean)
        action_dist = torch.distributions.Normal(action_mean, action_std)
        return action_dist, value

class PPOActorCriticLarge(nn.Module):
    """
    PPO Actor-Critic network for environments with continuous action spaces.
    This is a larger network compared to PPOActorCritic, designed to handle more 
    complecated taskes.
    
    This model outputs:
      - a stochastic policy (Normal distribution over actions)
      - a value estimate of the input state

    Args:
        obs_dim (int): Dimension of observation space
        act_dim (int): Dimension of action space

    Forward Input:
        x (torch.Tensor): Batch of observations, shape [batch_size, obs_dim]

    Forward Output:
        action_dist (torch.distributions.Normal): Distribution over actions
        value (torch.Tensor): Estimated state value, shape [batch_size, 1]
    """
    def __init__(self, obs_dim: int, act_dim: int):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
        )
        self.policy_mean = nn.Linear(256, act_dim)
        self.policy_logstd = nn.Parameter(torch.zeros(act_dim))
        self.value_head = nn.Linear(256, 1)
    
    def forward(self, x: torch.Tensor):
        """
        Forward pass through the network.

        Args:
            x (torch.Tensor): Input observation tensor [batch_size, obs_dim]

        Returns:
            action_dist (Normal): Gaussian distribution over actions
            value (torch.Tensor): State value estimate [batch_size, 1]
        """
        latent = self.shared(x)
        value = self.value_head(latent)
        action_mean = self.policy_mean(latent)
        action_std = self.policy_logstd.exp().expand_as(action_mean)
        action_dist = torch.distributions.Normal(action_mean, action_std)
        return action_dist, value