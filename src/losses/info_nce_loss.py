import torch
import torch.nn as nn

class InfoNCELoss(nn.Module):
    def __init__(self, temperature: float = 0.05, learnable: bool = False):
        """
        Args:
            temperature (float): The initial temperature value (e.g., 0.05 is a standard default).
            learnable (bool): If True, the model will dynamically learn the optimal temperature.
        """
        super().__init__()
        self.learnable = learnable
        
        if self.learnable:
            # We store log(1 / temperature) for numerical stability (Standard practice from CLIP)
            initial_scale = torch.log(torch.tensor(1.0 / temperature))
            self.logit_scale = nn.Parameter(torch.ones([]) * initial_scale)
        else:
            self.temperature = temperature
            
        self.criterion = nn.CrossEntropyLoss()

    def forward(
        self,
        input_or_logits: torch.Tensor,
        target_or_labels: torch.Tensor,
        labels: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Supports:
          - 3-arg mode: forward(input_embs, target_embs, labels)
          - 2-arg mode: forward(logits, labels)
        """
        if labels is None:
            logits = input_or_logits
            target_labels = target_or_labels
        else:
            logits = input_or_logits @ target_or_labels.T
            target_labels = labels

        # 1. Apply the temperature scale
        if self.learnable:
            # Clamp to prevent the scale from blowing up and causing NaN losses
            scale = torch.clamp(self.logit_scale.exp(), max=100.0)
            scaled_logits = logits * scale
        else:
            scaled_logits = logits / self.temperature

        # 2. Compute the standard cross-entropy loss
        return self.criterion(scaled_logits, target_labels)