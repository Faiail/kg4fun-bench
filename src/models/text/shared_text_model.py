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

    def encode(self, input_dict) -> torch.Tensor:
        output = self.encoder(**input_dict)
        cls_token = (
            output.pooler_output
            if "pooler_output" in output
            else output.last_hidden_state[:, 0]
        )
        return self.layernorm(self.projector(cls_token))

    def forward(self, input: dict, target: dict) -> dict[str, torch.Tensor]:
        input_emb = self.encode(input)
        target_emb = self.encode(target)
        return {ReturnKeys.INPUT: input_emb, ReturnKeys.TARGET: target_emb}
