import torch
import torch.nn as nn
import torch.nn.functional as F

class GumbelSparseAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, seq_len, tau=0.1):
        super(GumbelSparseAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.seq_len = seq_len
        self.tau = tau  # Gumbel-Softmax temperature parameter

        # Predict sparse mask based on sequence length
        self.gumbel_predictor = nn.Linear(seq_len, seq_len)
        self.scaling = (embed_dim // num_heads) ** -0.5

    def forward(self, query, key, value):
        batch_size, seq_len, embed_dim = query.size()

        # Project to multiple heads
        query = query.view(batch_size, seq_len, self.num_heads, -1).permute(0, 2, 1, 3)
        key = key.view(batch_size, seq_len, self.num_heads, -1).permute(0, 2, 1, 3)
        value = value.view(batch_size, seq_len, self.num_heads, -1).permute(0, 2, 1, 3)

        # Attention scores
        attn_weights = torch.einsum("bhld,bhmd->bhlm", query, key) * self.scaling

        # Gumbel-Softmax to generate sparse mask
        query_mean = query.mean(dim=-1)  # Shape: (B, H, L)
        logits = self.gumbel_predictor(query_mean)  # Shape: (B, H, L)
        mask = F.gumbel_softmax(logits, tau=self.tau, hard=True)  # Discrete-like sparse mask

        # Expand and apply mask
        mask = mask.unsqueeze(-1).expand(-1, -1, -1, seq_len)
        attn_weights = attn_weights * mask
        attn_weights = F.softmax(attn_weights, dim=-1)

        # Weighted value vectors
        attn_output = torch.einsum("bhlm,bhmd->bhld", attn_weights, value)
        attn_output = attn_output.permute(0, 2, 1, 3).contiguous().view(batch_size, seq_len, embed_dim)

        return attn_output

class GSformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, feedforward_dim, seq_len, tau=0.1, dropout=0.1):
        super(GSformerBlock, self).__init__()
        self.attention = GumbelSparseAttention(embed_dim, num_heads, seq_len, tau)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.dropout1 = nn.Dropout(dropout)

        self.feedforward = nn.Sequential(
            nn.Linear(embed_dim, feedforward_dim),
            nn.GELU(),
            nn.Linear(feedforward_dim, embed_dim)
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        # Gumbel-Softmax Sparse Attention
        attn_output = self.attention(x, x, x)
        x = x + self.dropout1(attn_output)
        x = self.norm1(x)

        # Feedforward network
        feedforward_output = self.feedforward(x)
        x = x + self.dropout2(feedforward_output)
        x = self.norm2(x)

        return x

class GSformer(nn.Module):
    def __init__(self, input_dim, embed_dim, num_heads, num_layers, feedforward_dim, seq_len, output_dim, tau=0.1, dropout=0.1):
        super(GSformer, self).__init__()
        self.embedding = nn.Linear(input_dim, embed_dim)
        self.layers = nn.ModuleList([
            GSformerBlock(embed_dim, num_heads, feedforward_dim, seq_len, tau, dropout) for _ in range(num_layers)
        ])
        self.output_layer = nn.Linear(embed_dim, output_dim)  # Updated to use output_dim

    def forward(self, x):
        x = self.embedding(x)
        for layer in self.layers:
            x = layer(x)
        x = self.output_layer(x)
        return x

# Example usage
if __name__ == "__main__":
    # Input tensor: (batch_size, sequence_length, input_dim)
    batch_size = 64
    seq_len = 96
    input_dim = 8
    output_dim = 16  # Specified output dimension

    model = GSformer(input_dim=input_dim, embed_dim=64, num_heads=8, num_layers=4, feedforward_dim=128, seq_len=seq_len, output_dim=output_dim, tau=0.1, dropout=0.1)
    x = torch.rand(batch_size, seq_len, input_dim)
    output = model(x)

    print("Input shape:", x.shape)
    print("Output shape:", output.shape)
    # 关注的是不同时刻时间的注意力关系

