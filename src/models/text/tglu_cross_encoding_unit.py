import torch
import torch.nn as nn
import torch.nn.functional as F


class TGLUCrossEncodingUnit(nn.Module):
    """
    Text-based Global-Local interaction Unit (T-GLU) from Luo et al. (ACM MM 2023):
    'MIMIC: Multi-Grained Multimodal Interaction Network for Entity Linking'
    
    Architecture:
    1. Global-to-Global (G2G): CLS cosine similarity between input entity and target schema class.
    2. Global-to-Local (G2L): Target tokens query Input tokens via scaled dot-product cross-attention,
       aggregated into a context vector that is scored by the projected target CLS.
    3. Matching Score: (G2L + G2G) / 2
    """
    def __init__(
        self,
        hidden_dim: int,
        unit_hidden_dim: int,
        dropout: float = 0.0,
        **kwargs,
    ):
        super().__init__()
        self.unit_hidden_dim = unit_hidden_dim

        # Projections as defined in Luo et al.
        self.fc_query = nn.Linear(hidden_dim, unit_hidden_dim)
        self.fc_key = nn.Linear(hidden_dim, unit_hidden_dim)
        self.fc_value = nn.Linear(hidden_dim, unit_hidden_dim)
        self.fc_cls = nn.Linear(hidden_dim, unit_hidden_dim)

        self.layer_norm = nn.LayerNorm(unit_hidden_dim)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

    def forward(
        self,
        input_cls: torch.Tensor,
        input_tokens: torch.Tensor,
        target_cls: torch.Tensor,
        target_tokens: torch.Tensor,
        input_mask: torch.Tensor,
        target_mask: torch.Tensor,
    ) -> torch.Tensor:
        B_in, L_in, _ = input_tokens.shape
        B_tgt, L_tgt, _ = target_tokens.shape

        # 1. Global-to-Global (G2G) Cosine Similarity: (B_in, B_tgt)
        c_in_norm = F.normalize(input_cls, p=2, dim=-1)
        c_tgt_norm = F.normalize(target_cls, p=2, dim=-1)
        g2g_score = torch.matmul(c_in_norm, c_tgt_norm.transpose(-1, -2))

        # 2. Linear Projections for Attention
        # Target tokens form Query: (B_tgt, 1, L_tgt, d)
        query = self.dropout(self.fc_query(target_tokens)).unsqueeze(1)
        # Input tokens form Key and Value: (1, B_in, L_in, d)
        key = self.dropout(self.fc_key(input_tokens)).unsqueeze(0)
        values = self.dropout(self.fc_value(input_tokens)).unsqueeze(0)

        # 3. Scaled Dot-Product Attention: (B_tgt, B_in, L_tgt, L_in)
        scale = float(self.unit_hidden_dim) ** 0.5
        attention_scores = torch.matmul(query, key.transpose(-1, -2)) / scale

        # Mask padding in input tokens (dim -1 is L_in)
        pad_in = ~input_mask.bool().unsqueeze(0).unsqueeze(2)  # (1, B_in, 1, L_in)
        attention_scores = attention_scores.masked_fill(pad_in, -1e4)
        attention_probs = F.softmax(attention_scores, dim=-1)

        # 4. Context Aggregation from Input tokens: (B_tgt, B_in, L_tgt, d)
        context = torch.matmul(attention_probs, values)

        # Masked average over valid target tokens (dim -2 is L_tgt)
        valid_tgt = target_mask.unsqueeze(1).unsqueeze(-1)      # (B_tgt, 1, L_tgt, 1)
        tgt_lens = target_mask.sum(dim=-1, keepdim=True).unsqueeze(1).clamp(min=1)
        context = (context * valid_tgt).sum(dim=-2) / tgt_lens # (B_tgt, B_in, d)
        context = self.layer_norm(context)

        # 5. Global-to-Local (G2L) Scoring
        target_cls_fc = self.fc_cls(target_cls).unsqueeze(1)    # (B_tgt, 1, d)
        target_cls_fc_norm = F.normalize(target_cls_fc, p=2, dim=-1)
        context_norm = F.normalize(context, p=2, dim=-1)

        # Cosine similarity between target CLS and aggregated context
        g2l_score = torch.sum(target_cls_fc_norm * context_norm, dim=-1).transpose(0, 1) # (B_in, B_tgt)

        # 6. Combined Matching Score: (G2L + G2G) / 2
        matching_score = (g2l_score + g2g_score) / 2 # (B_in, B_tgt)
        return matching_score

