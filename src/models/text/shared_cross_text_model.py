import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel
from src.models.utils import ReturnKeys
import src.models.text.cross_units as cross_unit_pkg


class SharedCrossEncodedTextModel(nn.Module):
    def __init__(
        self,
        bbone_config: dict,
        bbone_hidden_size: int,
        final_hidden_size: int = 512,
        dropout: float = 0.1,
        unit_type: str = "TextCrossEncodingUnit",
    ) -> None:
        super().__init__()
        self.encoder = AutoModel.from_pretrained(**bbone_config)
        self.schema_hidden_size = bbone_hidden_size

        self.cross_encoder_unit = cross_unit_pkg.__dict__[unit_type](
            hidden_dim=bbone_hidden_size,
            unit_hidden_dim=final_hidden_size,
            dropout=dropout,
        )

    def encode(self, input_dict: dict) -> dict[str, torch.Tensor]:
        """Independent encoding used at inference time to cache token representations."""
        output = self.encoder(**input_dict)
        return {
            ReturnKeys.TOKEN: output.last_hidden_state,
            ReturnKeys.CLS: output.last_hidden_state[:, 0],
            ReturnKeys.MASK: input_dict.attention_mask,
        }

    def forward(self, input: dict, target: dict) -> torch.Tensor:
        """Pairwise cross-encoding returning direct matching scores (B_in, B_tgt)."""
        out_input = self.encode(input)
        out_target = self.encode(target)

        input_tokens = out_input[ReturnKeys.TOKEN]
        target_tokens = out_target[ReturnKeys.TOKEN]

        input_cls = out_input[ReturnKeys.CLS]
        target_cls = out_target[ReturnKeys.CLS]

        input_mask = out_input[ReturnKeys.MASK]
        target_mask = out_target[ReturnKeys.MASK]

        scores = self.cross_encoder_unit(
            input_cls=input_cls,
            input_tokens=input_tokens,
            target_cls=target_cls,
            target_tokens=target_tokens,
            input_mask=input_mask,
            target_mask=target_mask,
        )

        return scores

    def match(
        self,
        input_dict: dict[str, torch.Tensor],
        target_dict: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        return self.cross_encoder_unit(
            input_cls=input_dict[ReturnKeys.CLS],
            input_tokens=input_dict[ReturnKeys.TOKEN],
            input_mask=input_dict[ReturnKeys.MASK],
            target_cls=target_dict[ReturnKeys.CLS],
            target_tokens=target_dict[ReturnKeys.TOKEN],
            target_mask=target_dict[ReturnKeys.MASK],
        )
