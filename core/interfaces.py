import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Tuple, Dict, Any, Optional
#To note I have written 'I' before every class is because it indicates Interface.


class IEpisodicMemory(nn.Module, ABC):
    @abstractmethod
    def store_episode(self, state_sequence: torch.Tensor, valence_sequence: torch.Tensor, empowerment_score: float) -> None:
        pass

    @abstractmethod
    def get_dream_batch(self, batch_size: int) -> Optional[torch.Tensor]:
        pass

    @abstractmethod
    def get_identity_context(self, current_latent: torch.Tensor) -> torch.Tensor:
        pass


    
class IActor(nn.Module, ABC):
    @abstractmethod
    def forward(self, state_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        pass

    @abstractmethod
    def sample(self, state_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        pass

class IWeightMergeStrategy(ABC):
    @abstractmethod
    def merge(self, target_model: nn.Module, ext_state_dict: Dict[str, torch.Tensor], ext_config: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        pass
