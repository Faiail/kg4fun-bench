import math
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class CosFaceLoss(nn.Module):
    """
    Large Margin Cosine Loss (CosFace) with optional cross-batch memory queue:
    - Supports 3-arg mode: forward(input_embs, target_embs, labels)
    - Supports 2-arg mode: forward(logits, labels) [when queue_size == 0]
    - Optional memory queue: stores FIFO target embeddings across batches to
      dramatically increase negative candidate diversity (essential when candidates >> batch size).
    - Optional learnable scale s (clamped between min_s and max_s).
    - Clean linear margin m subtracted in cosine space: logit_target = (cos - m) * s
    """

    def __init__(
        self,
        s: float = 25.0,
        m: float = 0.10,
        queue_size: int = 0,
        feat_dim: int = 512,
        learnable_s: bool = False,
        min_s: float = 1.0,
        max_s: float = 50.0,
        warmup_steps: int = 0,
    ):
        super().__init__()
        self.s = float(s)
        self.m = float(m)
        self.queue_size = int(queue_size) if queue_size is not None else 0
        self.feat_dim = int(feat_dim)
        self.learnable_s = learnable_s
        self.min_s = float(min_s)
        self.max_s = float(max_s)
        self.warmup_steps = int(warmup_steps) if warmup_steps is not None else 0
        self.current_step = 0
        self.current_m = 0.0 if self.warmup_steps > 0 else self.m

        if self.learnable_s:
            self.log_s = nn.Parameter(torch.tensor(math.log(s), dtype=torch.float32))
        else:
            self.log_s = None

        if self.queue_size > 0:
            # Memory queue initialized with unit-normalized random vectors
            self.register_buffer(
                "queue",
                F.normalize(torch.randn(self.queue_size, self.feat_dim), dim=-1),
            )
            self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    def step_margin(self):
        """Advances margin schedule if warmup_steps is configured."""
        if self.warmup_steps > 0:
            self.current_step += 1
            if self.current_step >= self.warmup_steps:
                self.current_m = self.m
            else:
                self.current_m = self.m * (self.current_step / self.warmup_steps)

    @property
    def current_s(self) -> float:
        if self.learnable_s:
            with torch.no_grad():
                return float(self.log_s.exp().clamp(self.min_s, self.max_s).item())
        return self.s

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys: torch.Tensor):
        """FIFO update of the memory queue."""
        keys = keys.detach()
        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr.item())

        if ptr + batch_size <= self.queue_size:
            self.queue[ptr : ptr + batch_size] = keys
            ptr = (ptr + batch_size) % self.queue_size
        else:
            first_chunk = self.queue_size - ptr
            self.queue[ptr:] = keys[:first_chunk]
            self.queue[: batch_size - first_chunk] = keys[first_chunk:]
            ptr = batch_size - first_chunk

        self.queue_ptr[0] = ptr

    def forward(
        self,
        input_or_logits: torch.Tensor,
        target_or_labels: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Supports:
          - 3-arg mode: forward(input_embs, target_embs, labels)
          - 2-arg mode: forward(logits, labels)
        """
        if labels is None:
            # 2-arg mode: input is already the cosine similarity logits
            logits = input_or_logits
            target_labels = target_or_labels
        else:
            # 3-arg mode: embeddings are passed directly
            input_embs = input_or_logits
            target_embs = target_or_labels
            target_labels = labels

            # Compute in-batch cosine similarities
            batch_logits = input_embs @ target_embs.T

            if self.queue_size > 0:
                # Include queue negatives: clone and detach to avoid in-place graph conflicts
                queue_embs = self.queue.clone().detach().to(
                    device=input_embs.device, dtype=input_embs.dtype
                )
                queue_logits = input_embs @ queue_embs.T
                logits = torch.cat([batch_logits, queue_logits], dim=1)

                if self.training:
                    self._dequeue_and_enqueue(target_embs)
            else:
                logits = batch_logits

        # Determine scale factor
        if self.learnable_s:
            scale = torch.clamp(
                self.log_s.to(logits.device).exp(), min=self.min_s, max=self.max_s
            )
        else:
            scale = self.s

        # Apply CosFace margin: subtract (m * scale) from positive target logits
        diff = -self.current_m * scale
        scaled_logits = logits * scale
        target_labels_idx = target_labels.unsqueeze(1)
        diff_tensor = diff if isinstance(diff, torch.Tensor) else torch.tensor(diff, device=logits.device, dtype=logits.dtype)
        scaled_logits = scaled_logits.scatter_add(
            1,
            target_labels_idx,
            diff_tensor.expand_as(target_labels_idx),
        )

        return F.cross_entropy(scaled_logits, target_labels)

