import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from typing import Tuple
from core.interfaces import IEpisodicMemory, IActor


class ContinuousActor(IActor):
    def __init__(self, d_model: int, action_dim: int):
        super().__init__()
        self.mu = nn.Linear(d_model, action_dim)
        self.log_std = nn.Linear(d_model, action_dim)
        
    def forward(self, state_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mu = self.mu(state_features)
        log_std = torch.clamp(self.log_std(state_features), -20, 2)
        return mu, log_std

    def sample(self, state_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mu, log_std = self.forward(state_features)
        std = torch.exp(log_std)
        normal_dist = Normal(mu, std)
        x_t = normal_dist.rsample() 
        action = torch.tanh(x_t)
        log_prob = normal_dist.log_prob(x_t) - torch.log(1 - action.pow(2) + 1e-6)
        return action, log_prob.sum(dim=-1, keepdim=True)


class ContinuousSACPolicy(nn.Module):
    def __init__(
        self, 
        backbone: nn.Module, 
        actor1: IActor, 
        actor2: IActor, 
        memory: IEpisodicMemory, 
        d_model: int
    ):
        super().__init__()
        self.backbone = backbone
        self.actor1 = actor1
        self.actor2 = actor2
        self.memory = memory  # Injected Memory Interface

        self.gate = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1),
            nn.Sigmoid()
        )

    def get_action(self, state):
        z = self.backbone.forward_pass(state)
        z_global = z.mean(dim=1) 
        identity_context = self.memory.get_identity_context(z_global)
        z_conscious = z_global + identity_context
        g = self.gate(z_conscious)
        mu1, log_std1 = self.actor1.forward(z_conscious)
        mu2, log_std2 = self.actor2.forward(z_conscious)
        blended_mu = g * mu1 + (1 - g) * mu2
        blended_log_std = g * log_std1 + (1 - g) * log_std2
        std = torch.exp(blended_log_std)
        normal_dist = Normal(blended_mu, std)
        x_t = normal_dist.rsample()
        final_action = torch.tanh(x_t)
        final_log_prob = normal_dist.log_prob(x_t) - torch.log(1 - final_action.pow(2) + 1e-6)
        return final_action, final_log_prob.sum(dim=-1, keepdim=True), g

class DoubleQCritic(nn.Module):
    def __init__(self, d_model: int, action_dim: int):
        super().__init__()
        self.critic1 = nn.Sequential(
            nn.Linear(d_model + action_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1)
        )
        self.critic2 = nn.Sequential(
            nn.Linear(d_model + action_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1)
        )

    def forward(self, z_global: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        sa_pair = torch.cat([z_global, action], dim=-1)
        return self.critic1(sa_pair), self.critic2(sa_pair)

    
class DiscreteValencePolicy(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Linear(512, action_dim)
        )
        self.value_manifold = nn.Linear(state_dim, 1)

    def get_empowerment(self, action_probs):
        marginal = action_probs.mean(dim=0)
        mi = torch.sum(action_probs * torch.log(action_probs / (marginal + 1e-9) + 1e-9), dim=-1)
        return mi.mean()

    def forward(self, state, valence):
        logits = self.actor(state)
        action_probs = F.softmax(logits + valence, dim=-1)
        
        empowerment = self.get_empowerment(action_probs)
        return action_probs, empowerment
