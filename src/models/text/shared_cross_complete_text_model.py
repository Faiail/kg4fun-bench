import torch
from transformers import AutoModel
from src.models.utils import ReturnKeys
import src.models.text.cross_units as cross_unit_pkg


class SharedCompleteCrossEncodedTextModel(torch.nn.Module):
    def __init__(
        self,
        bbone_config: dict,
        unit_type_dict: dict[str, str],
        bbone_hidden_size: int,
        final_hidden_size: int = 512,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.encoder = AutoModel.from_pretrained(**bbone_config)
        self.schema_hidden_size = bbone_hidden_size

        self.cross_encoders = torch.nn.ModuleDict(
            {
                k: cross_unit_pkg.__dict__[v](
                    hidden_dim=bbone_hidden_size,
                    unit_hidden_dim=final_hidden_size,
                    dropout=dropout,
                )
                for k, v in unit_type_dict.items()
            }
        )
        assert set(list(self.cross_encoders.keys())) == set(
            [ReturnKeys.NODE, ReturnKeys.EDGE]
        )

    def encode(self, input_dict: dict) -> dict[str, torch.Tensor]:
        output = self.encoder(**input_dict)
        return {
            ReturnKeys.TOKEN: output.last_hidden_state,
            ReturnKeys.CLS: output.last_hidden_state[:, 0],
            ReturnKeys.MASK: input_dict.attention_mask,
        }

    def forward(
        self,
        input_head: dict,
        input_rel: dict,
        input_tail: dict,
        target_head: dict,
        target_rel: dict,
        target_tail: dict,
    ) -> tuple[torch.Tensor]:
        input_head_emb = self.encode(input_head)
        input_rel_emb = self.encode(input_rel)
        input_tail_emb = self.encode(input_tail)
        target_head_emb = self.encode(target_head)
        target_rel_emb = self.encode(target_rel)
        target_tail_emb = self.encode(target_tail)

        return {
            ReturnKeys.HEAD: self.cross_encoders[ReturnKeys.NODE](
                input_cls=input_head_emb[ReturnKeys.CLS],
                input_tokens=input_head_emb[ReturnKeys.TOKEN],
                input_mask=input_head_emb[ReturnKeys.MASK],
                target_cls=target_head_emb[ReturnKeys.CLS],
                target_tokens=target_head_emb[ReturnKeys.TOKEN],
                target_mask=target_head_emb[ReturnKeys.MASK],
            ),
            ReturnKeys.REL: self.cross_encoders[ReturnKeys.EDGE](
                input_cls=input_rel_emb[ReturnKeys.CLS],
                input_tokens=input_rel_emb[ReturnKeys.TOKEN],
                input_mask=input_rel_emb[ReturnKeys.MASK],
                target_cls=target_rel_emb[ReturnKeys.CLS],
                target_tokens=target_rel_emb[ReturnKeys.TOKEN],
                target_mask=target_rel_emb[ReturnKeys.MASK],
            ),
            ReturnKeys.TAIL: self.cross_encoders[ReturnKeys.NODE](
                input_cls=input_tail_emb[ReturnKeys.CLS],
                input_tokens=input_tail_emb[ReturnKeys.TOKEN],
                input_mask=input_tail_emb[ReturnKeys.MASK],
                target_cls=target_tail_emb[ReturnKeys.CLS],
                target_tokens=target_tail_emb[ReturnKeys.TOKEN],
                target_mask=target_tail_emb[ReturnKeys.MASK],
            ),
        }

    def match(self, input_dict, target_dict, unit: str) -> torch.Tensor:
        return self.cross_encoders[unit](
            input_cls=input_dict[ReturnKeys.CLS],
            input_tokens=input_dict[ReturnKeys.TOKEN],
            input_mask=input_dict[ReturnKeys.MASK],
            target_cls=target_dict[ReturnKeys.CLS],
            target_tokens=target_dict[ReturnKeys.TOKEN],
            target_mask=target_dict[ReturnKeys.MASK],
        )
