import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================
# FiLM Conditioning
# ============================
class FiLM(nn.Module):
    def __init__(self, d_model, embed_dim):
        super().__init__()
        self.gamma = nn.Linear(embed_dim, d_model)
        self.beta = nn.Linear(embed_dim, d_model)

    def forward(self, x, cond):
        # x: (B, d_model)
        # cond: (B, embed_dim)
        gamma = self.gamma(cond)
        beta = self.beta(cond)
        return x * gamma + beta


# ============================
# Transformer Encoder for x_struct
# ============================
class TSFLEncoder(nn.Module):
    def __init__(self, feature_dim, d_model, n_heads, num_layers):
        super().__init__()
        self.linear_in = nn.Linear(feature_dim, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x):
        # x: (B, 30, F)
        h = self.linear_in(x)
        h = self.encoder(h)
        return h  # (B, 30, d_model)


# ============================
# Cross-Attention over neighbors
# ============================
class NeighborAttention(nn.Module):
    def __init__(self, embed_dim, d_model, horizon, num_heads=4):
        super().__init__()
        self.query_proj = nn.Linear(embed_dim, d_model)
        self.key_proj = nn.Linear(embed_dim, d_model)
        self.value_proj = nn.Linear(horizon, d_model)

        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            batch_first=True
        )

    def forward(self, q, keys, values):
        # q:     (B, embed_dim)
        # keys:  (B, K, embed_dim)
        # values:(B, K, horizon)

        Q = self.query_proj(q).unsqueeze(1)    # (B, 1, d_model)
        K = self.key_proj(keys)                # (B, K, d_model)
        V = self.value_proj(values)            # (B, K, d_model)

        out, attn_weights = self.attn(Q, K, V)
        # out: (B, 1, d_model)
        # attn_weights: (B, 1, K)

        return out.squeeze(1), attn_weights, V


class MemoryHop(nn.Module):
    def __init__(self, d_model, num_heads=4):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=num_heads, batch_first=True)

    def forward(self, state, memory):
        # state: (B, 1, d_model)
        # memory: (B, K, d_model)
        out, _ = self.attn(state, memory, memory)
        return out


# ============================
# Full TSFM Model
# ============================
class TSFM(nn.Module):
    def __init__(
        self,
        feature_dim=11,
        embed_dim=3072,
        d_model=256,
        n_heads=4,
        num_layers=2,
        K=5,
        horizon=5
    ):
        super().__init__()

        # encodes 30-day structured window
        self.encoder = TSFLEncoder(feature_dim, d_model, n_heads, num_layers)

        # FiLM for conditioning structured features using event embedding
        self.film = FiLM(d_model, embed_dim)

        # cross-attention using neighbors
        self.neighbor_attn = NeighborAttention(embed_dim, d_model, horizon)
        self.memory_hop = MemoryHop(d_model)

        # fusion layer
        self.fusion = nn.Linear(d_model * 2, d_model)

        # prediction head
        self.pred_head = nn.Linear(d_model, horizon)

    def forward(self, x_struct, x_query, x_keys, x_values):
        """
        Inputs:
            x_struct: (B, 30, F)
            x_query:  (B, embed_dim)
            x_keys:   (B, K, embed_dim)
            x_values: (B, K, 5)

        Outputs:
            y_pred: (B, 5)
            attn_weights: (B, 1, K)
        """

        # 1. structured encoder
        h_struct = self.encoder(x_struct)      # (B, 30, d_model)
        h_struct = h_struct.mean(dim=1)        # (B, d_model)

        # 2. FiLM conditioning
        h_struct = self.film(h_struct, x_query)

        # 3. neighbor cross-attention
        h_attn, attn_weights, memory = self.neighbor_attn(x_query, x_keys, x_values)

        hop_input = h_attn.unsqueeze(1)  # (B, 1, d_model)
        hop_out = self.memory_hop(hop_input, memory).squeeze(1)
        h_attn = 0.5 * (h_attn + hop_out)

        # 4. fusion
        h = torch.cat([h_struct, h_attn], dim=-1)
        h = F.relu(self.fusion(h))

        # 5. predict future 5-day trajectory
        y_pred = self.pred_head(h)

        return y_pred, attn_weights
