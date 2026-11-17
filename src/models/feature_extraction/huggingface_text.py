from transformers import AutoModel
import torch


class HuggingfaceTextModel(torch.nn.Module):
    def __init__(self, config: dict = None, **kwargs):
        super().__init__()
        cfg = config if config is not None else kwargs
        self.model = AutoModel.from_pretrained(**cfg)

    def _pool(self, output, attention_mask: torch.Tensor = None) -> torch.Tensor:
        if attention_mask is None:
            return output.last_hidden_state[:, 0]
        mask = attention_mask.unsqueeze(-1)
        return (output.last_hidden_state * mask).sum(
            dim=1
        ) / mask.sum(dim=1).clamp(min=1e-9)

    def forward(self, input_dict) -> torch.Tensor:
        output = self.model(**input_dict)
        pooled = self._pool(output, input_dict.get("attention_mask", None))
        return torch.nn.functional.normalize(pooled, p=2, dim=-1)
