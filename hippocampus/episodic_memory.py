"""
hippocampus/episodic_memory.py — Significance-weighted episodic memory.

The hippocampus is the brain's memory formation centre. It encodes new
experiences, stores them with emotional significance weighting, and
replays them during rest (dream cycles) to consolidate learning.

This module implements the computational equivalent:
  - Significance-weighted episodic replay with recency decay
  - Narrative GRU that encodes the agent's recent trajectory into identity
  - Dream batch sampling for offline replay

Moved from: core/episodic_memory.py
"""

from __future__ import annotations

import math
from collections import deque
from typing import Optional

import torch
import torch.nn as nn

from interfaces.base import IEpisodicMemory


class EpisodicMemory(IEpisodicMemory):
    """
    Significance-weighted episodic memory.

    Each episode is a dict:
        { 'states': Tensor[T, D],
          'significance': float,
          'step': int }

    Significance:
        σ = |E[valence]| + empowerment_score

    Sampling weight at step `now`:
        w_i = σ_i · exp(-(now - step_i) / half_life)

    Identity context:
        seq      = stack(last K episode endpoints) ⊕ current_latent
        h_n      = GRU(seq)[-1]
        identity = h_n + self_token

    Args:
        latent_dim:        Latent dimensionality (default 512).
        capacity:          Max episodes retained.
        sequence_length:   Canonical episode length (pad/truncate to this).
        narrative_window:  How many recent episodes to feed the narrative GRU.
        half_life_steps:   Recency decay half-life in training steps.
        endpoint_only:     If True, store only the final state per episode
                           (D floats instead of T x D). Cuts memory by ~16x.
                           Significant episodes (boundary-flagged) keep
                           the full sequence regardless of this flag.
    """

    def __init__(
        self,
        latent_dim: int = 512,
        capacity: int = 10_000,
        sequence_length: int = 16,
        narrative_window: int = 8,
        half_life_steps: int = 5_000,
        endpoint_only: bool = False,
    ) -> None:
        super().__init__()
        self.capacity = capacity
        self.sequence_length = sequence_length
        self.narrative_window = narrative_window
        self.half_life_steps = float(half_life_steps)
        self.endpoint_only = endpoint_only
        self.latent_dim = latent_dim

        self._bank: deque = deque(maxlen=capacity)
        self._current_step: int = 0

        # Cached signature stack for fast recall (built on demand, invalidated
        # on every store_episode). Recall does cosine similarity against
        # signatures every tick, so caching avoids rebuilding the (N, D)
        # stack from scratch each time.
        self._sig_cache: Optional[torch.Tensor] = None     # (N, D) normalised
        self._sig_cache_mode: Optional[str] = None         # "endpoint" or "trajectory"
        self._sig_cache_size: int = 0                      # n episodes when cached

        self.narrative_encoder = nn.GRU(latent_dim, latent_dim, batch_first=True)
        self.self_token = nn.Parameter(torch.randn(1, 1, latent_dim))

    def set_current_step(self, step: int) -> None:
        self._current_step = int(step)

    @property
    def current_step(self) -> int:
        return self._current_step

    @property
    def size(self) -> int:
        return len(self._bank)

    def get_signature_cache(self, mode: str = "endpoint") -> Optional[torch.Tensor]:
        """
        Return a cached (N, D) stack of L2-normalised episode signatures
        for cosine similarity. Rebuilt only when the bank changes.

        Args:
            mode: "endpoint" or "trajectory" (averaging mode).

        Returns:
            (N, D) CPU tensor, or None if bank is empty.
        """
        if not self._bank:
            return None

        # Cache valid: same mode and same bank size.
        cache_valid = (
            self._sig_cache is not None
            and self._sig_cache_mode == mode
            and self._sig_cache_size == len(self._bank)
        )
        if cache_valid:
            return self._sig_cache

        # Rebuild
        import torch.nn.functional as F  # local import to avoid top-level cost
        sigs = []
        for ep in self._bank:
            states = ep["states"]
            if mode == "endpoint":
                sig = states[-1]
            else:  # trajectory
                sig = states.mean(dim=0)
            sig = sig.to('cpu') if sig.device.type != 'cpu' else sig
            sigs.append(F.normalize(sig.float(), p=2, dim=0))
        self._sig_cache = torch.stack(sigs)
        self._sig_cache_mode = mode
        self._sig_cache_size = len(self._bank)
        return self._sig_cache

    def store_episode(
        self,
        state_sequence: torch.Tensor,
        valence_sequence: torch.Tensor,
        empowerment_score: float,
        force_full_sequence: bool = False,
    ) -> None:
        significance = torch.abs(valence_sequence.mean()) + empowerment_score
        states = state_sequence.detach().to('cpu')
        if states.dim() == 1:
            states = states.unsqueeze(0)
        target_T = int(self.sequence_length)
        T = int(states.shape[0])
        if T < target_T:
            pad = states[-1:].expand(target_T - T, *states.shape[1:])
            states = torch.cat([states, pad], dim=0)
        elif T > target_T:
            states = states[-target_T:]

        # Endpoint-only storage: keep only the last state to save memory.
        # Significant episodes (force_full_sequence=True, e.g. boundary-flagged
        # or high-significance ones) keep the full sequence so dream batches
        # and consolidation still have temporal data to work with.
        if self.endpoint_only and not force_full_sequence and float(significance.item()) < 0.5:
            states = states[-1:]  # (1, D)

        self._bank.append(
            {
                "states": states,
                "significance": float(significance.item()),
                "step": self._current_step,
            }
        )
        # Invalidate signature cache — bank changed
        self._sig_cache = None

    # ------------------------------------------------------------------
    # Convenience: encode raw text into memory via the granite embedder
    # ------------------------------------------------------------------

    def store_text(
        self,
        text,
        valence: float = 0.0,
        empowerment_score: float = 0.0,
    ) -> None:
        """
        Encode raw text via the thalamus GraniteEmbedder and store it as
        an episode. This lets any region create memories from natural
        language without first projecting to the latent space themselves.

        Args:
            text:               A single string, or a list of strings forming
                                a temporal sequence (each string = one frame).
            valence:            Scalar emotional valence in [-1, 1].
            empowerment_score:  Optional empowerment contribution to significance.

        Notes:
            The granite-backed text encoder is loaded lazily on first call
            (a process-wide singleton), so the cost is amortised across all
            calls and across all regions that share this hippocampus.
        """
        from thalamus.granite_embedder import get_embedder

        embedder = get_embedder()
        items = [text] if isinstance(text, str) else list(text)
        z = embedder.encode(items)                # (N, D)
        if z.dim() == 1:
            z = z.unsqueeze(0)

        T = z.shape[0]
        valences = torch.full((T, 1), float(valence), device=z.device)
        self.store_episode(
            state_sequence=z,
            valence_sequence=valences,
            empowerment_score=empowerment_score,
        )

    def query_by_text(self, text: str) -> Optional[torch.Tensor]:
        """
        Encode `text` via the granite embedder and return the identity
        context grounded in that query — the hippocampus's recall response
        to a natural-language probe.

        Returns:
            (1, D) identity context tensor, or None if memory is empty.
        """
        from thalamus.granite_embedder import get_embedder

        if not self._bank:
            return None
        embedder = get_embedder()
        q = embedder.encode(text).unsqueeze(0)    # (1, D)
        return self.get_identity_context(q)

    def _effective_weights(self) -> torch.Tensor:
        if not self._bank:
            return torch.zeros(0)
        now = self._current_step
        weights = torch.tensor(
            [
                ep["significance"] * math.exp(-(now - ep["step"]) / self.half_life_steps)
                for ep in self._bank
            ],
            dtype=torch.float32,
        )
        return torch.clamp(weights, min=1e-6)

    def get_dream_batch(self, batch_size: int) -> Optional[torch.Tensor]:
        if len(self._bank) < batch_size:
            return None

        weights = self._effective_weights()
        target_T = int(self.sequence_length)

        # Build a mask of episodes that have full-length sequences (skip
        # endpoint-only stored episodes — they can't be dreamed about).
        full_mask = torch.tensor(
            [float(ep["states"].shape[0] == target_T) for ep in self._bank],
            dtype=torch.float32,
        )
        weights = weights * full_mask
        if weights.sum() <= 0:
            return None

        sampled: list = []
        attempts = 0
        max_attempts = max(batch_size * 4, batch_size + 16)
        while len(sampled) < batch_size and attempts < max_attempts:
            idx = torch.multinomial(weights, 1, replacement=True).item()
            ep = self._bank[idx]
            states = ep["states"]
            if states.dim() == 2 and states.shape[0] == target_T:
                sampled.append(states)
            attempts += 1
        if len(sampled) < batch_size:
            return None
        return torch.stack(sampled)

    def get_identity_context(self, current_latent: torch.Tensor) -> torch.Tensor:
        device = current_latent.device
        B, D = current_latent.shape
        self_token = self.self_token.to(device).squeeze(0).squeeze(0)

        if not self._bank:
            return self_token.unsqueeze(0).expand(B, -1)

        k = min(self.narrative_window, len(self._bank))
        recent = list(self._bank)[-k:]
        endpoints = torch.stack([ep["states"][-1] for ep in recent]).to(device)

        endpoints_seq = endpoints.unsqueeze(0).expand(B, -1, -1)
        current_token = current_latent.unsqueeze(1)
        seq = torch.cat([endpoints_seq, current_token], dim=1)

        # nn.GRU is not supported on DirectML (privateuseone) — run on CPU.
        # On CUDA/MPS/CPU the GRU runs natively on the device.
        if device.type == "privateuseone":
            with torch.no_grad():
                _, h_n = self.narrative_encoder.cpu()(seq.to('cpu'))
            identity = h_n[-1].to(device)
        else:
            with torch.no_grad():
                _, h_n = self.narrative_encoder(seq)
            identity = h_n[-1]
        return identity + self_token


__all__ = ["EpisodicMemory"]
