import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from core.interfaces import IEpisodicMemory

class LatentAligner(nn.Module):
    def __init__(
        self, 
        encoders: nn.ModuleDict,  # OCP Injection point for modular encoders
        d_model: int = 512, 
        backbone: Optional[nn.Module] = None,
        lr: float = 1e-4, 
        temperature: float = 0.07
    ):
        super().__init__()
        self.aligners = encoders
        
        # Information Bottleneck constraint
        self.bottleneck = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, d_model)
        )
        self.backbone = backbone
        self.temperature = temperature
        self.emotion_vocab = {'Angry': 0, 'Sad': 1, 'Happy': 2, 'Calm': 3}
        self.emotion_embeddings = nn.Embedding(len(self.emotion_vocab), d_model)
        self.optimizer = optim.AdamW(self.parameters(), lr=lr, weight_decay=1e-2)

    def forward(self, x: torch.Tensor, modality: str) -> torch.Tensor:
        if modality not in self.aligners:
            raise KeyError(f"Modality '{modality}' encoder registry not found.")
        z_raw = self.aligners[modality](x)
        return self.bottleneck(z_raw)
        
    def compute_infonce_loss(self, z_a: torch.Tensor, z_b: torch.Tensor) -> torch.Tensor:
        batch_size = z_a.shape[0]
        z_a = F.normalize(z_a, p=2, dim=1)
        z_b = F.normalize(z_b, p=2, dim=1)
        logits = (z_a @ z_b.T) / self.temperature
        labels = torch.arange(batch_size, device=z_a.device)
        return (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2

    def compute_emotional_alignment_loss(self, z_a: torch.Tensor, z_b: torch.Tensor, z_emotion: torch.Tensor) -> torch.Tensor:
        return (
            self.compute_infonce_loss(z_a, z_b) + 
            self.compute_infonce_loss(z_a, z_emotion) + 
            self.compute_infonce_loss(z_b, z_emotion)
        ) / 3

    def train_step(self, data_a: torch.Tensor, data_b: torch.Tensor, emotion_ids: Optional[torch.Tensor] = None) -> float:
        if self.backbone is None:
            raise AttributeError("Backbone network context must be injected for feature pass extraction.")
        self.optimizer.zero_grad()
        z_a_global = self.backbone.forward_pass(data_a).mean(dim=1)
        z_b_global = self.backbone.forward_pass(data_b).mean(dim=1)
        
        if emotion_ids is not None:
            loss = self.compute_emotional_alignment_loss(z_a_global, z_b_global, self.emotion_embeddings(emotion_ids))
        else:
            loss = self.compute_infonce_loss(z_a_global, z_b_global)

        loss.backward()
        self.optimizer.step()
        return loss.item()
