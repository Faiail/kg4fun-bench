import torch
from transformers import AutoModel
from src.models.utils import ReturnKeys


class SharedTextEncoder(torch.nn.Module):
    def __init__(
        self,
        bbone_config: dict,
        bbone_hidden_size: int,
        final_hidden_size: int,
    ) -> None:
        super().__init__()
        self.encoder = AutoModel.from_pretrained(**bbone_config)
        self.projector = torch.nn.Linear(bbone_hidden_size, final_hidden_size)
        self.layernorm = torch.nn.LayerNorm(final_hidden_size)
        self.schema_hidden_size = final_hidden_size

    def _pool(self, output, input_dict: dict) -> torch.Tensor:
        if "attention_mask" in input_dict:
            mask = input_dict["attention_mask"].unsqueeze(-1).to(output.last_hidden_state.dtype)
            return (output.last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
        return output.last_hidden_state[:, 0]

    def encode(self, input_dict) -> torch.Tensor:
        output = self.encoder(**input_dict)
        pooled = self._pool(output, input_dict)
        return torch.nn.functional.normalize(self.layernorm(self.projector(pooled)), p=2, dim=-1)

    def forward(self, input: dict, target: dict) -> dict[str, torch.Tensor]:
        input_emb = self.encode(input)
        target_emb = self.encode(target)
        return {ReturnKeys.INPUT: input_emb, ReturnKeys.TARGET: target_emb}
