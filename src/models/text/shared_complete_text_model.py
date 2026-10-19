import torch
from transformers import AutoModel
from src.models.utils import ReturnKeys


class SharedCompleteTextEncoder(torch.nn.Module):
    def __init__(
        self,
        bbone_config: dict,
        bbone_hidden_size: int,
        final_hidden_size: int,
    ) -> None:
        super().__init__()
        self.encoder = AutoModel.from_pretrained(**bbone_config)
        self.node_projector = torch.nn.Linear(bbone_hidden_size, final_hidden_size)
        self.node_layernorm = torch.nn.LayerNorm(final_hidden_size)
        self.edge_projector = torch.nn.Linear(bbone_hidden_size, final_hidden_size)
        self.edge_layernorm = torch.nn.LayerNorm(final_hidden_size)
        self.schema_hidden_size = final_hidden_size

    def encode(self, input_dict: dict, task: str = "node") -> torch.Tensor:
        output = self.encoder(**input_dict)
        cls_token = output.last_hidden_state[:, 0]
        if task == "edge":
            return torch.nn.functional.normalize(
                self.edge_layernorm(self.edge_projector(cls_token)), p=2, dim=-1
            )
        return torch.nn.functional.normalize(
            self.node_layernorm(self.node_projector(cls_token)), p=2, dim=-1
        )

    def forward(
        self,
        input_head: dict,
        input_rel: dict,
        input_tail: dict,
        target_head: dict,
        target_rel: dict,
        target_tail: dict,
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        input_head_emb = self.encode(input_head, task="node")
        input_rel_emb = self.encode(input_rel, task="edge")
        input_tail_emb = self.encode(input_tail, task="node")
        target_head_emb = self.encode(target_head, task="node")
        target_rel_emb = self.encode(target_rel, task="edge")
        target_tail_emb = self.encode(target_tail, task="node")
        return (
            {
                ReturnKeys.HEAD: input_head_emb,
                ReturnKeys.REL: input_rel_emb,
                ReturnKeys.TAIL: input_tail_emb,
            },
            {
                ReturnKeys.HEAD: target_head_emb,
                ReturnKeys.REL: target_rel_emb,
                ReturnKeys.TAIL: target_tail_emb,
            },
        )
