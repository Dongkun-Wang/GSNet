import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class GumbelSparseAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, seq_len, tau=0.1):
        super(GumbelSparseAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.seq_len = seq_len
        self.tau = tau  # Gumbel-Softmax温度参数

        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "embed_dim必须能被num_heads整除"

        # 定义线性投影层
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        # 预测器，用于生成针对键位置的logits
        self.gumbel_predictor = nn.Linear(self.head_dim, seq_len)
        self.scaling = self.head_dim ** -0.5

    def forward(self, query, key, value):
        batch_size, seq_len, _ = query.size()

        # 对query、key、value进行投影并重塑为多头形式
        query = self.q_proj(query).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, L, D)
        key = self.k_proj(key).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)      # (B, H, L, D)
        value = self.v_proj(value).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, L, D)

        # 计算注意力分数
        attn_scores = torch.matmul(query, key.transpose(-2, -1)) * self.scaling  # (B, H, L, L)

        # 生成Gumbel-Softmax掩码
        logits = self.gumbel_predictor(query)  # (B, H, L, seq_len)
        mask = F.gumbel_softmax(logits, tau=self.tau, hard=True)  # (B, H, L, seq_len)

        # 应用掩码，将被掩码的位置的注意力分数设为负无穷
        attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))

        # 计算注意力权重
        attn_probs = F.softmax(attn_scores, dim=-1)

        # 计算注意力输出
        attn_output = torch.matmul(attn_probs, value)  # (B, H, L, D)

        # 将多头的结果拼接起来
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)
        attn_output = self.out_proj(attn_output)

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
            nn.Dropout(dropout),
            nn.Linear(feedforward_dim, embed_dim)
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        # Gumbel-Softmax稀疏注意力
        attn_output = self.attention(x, x, x)
        x = x + self.dropout1(attn_output)
        x = self.norm1(x)

        # 前馈网络
        feedforward_output = self.feedforward(x)
        x = x + self.dropout2(feedforward_output)
        x = self.norm2(x)

        return x

class GSformer(nn.Module):
    def __init__(self, input_dim, embed_dim, num_heads, num_layers, feedforward_dim, seq_len, output_dim, tau=0.1, dropout=0.1):
        super(GSformer, self).__init__()
        self.embed_dim = embed_dim
        self.seq_len = seq_len

        # 输入嵌入层
        self.embedding = nn.Linear(input_dim, embed_dim)
        # 位置编码，可学习的
        self.pos_embedding = nn.Parameter(torch.zeros(1, seq_len, embed_dim))

        # Transformer层
        self.layers = nn.ModuleList([
            GSformerBlock(embed_dim, num_heads, feedforward_dim, seq_len, tau, dropout) for _ in range(num_layers)
        ])
        # 输出层
        self.output_layer = nn.Linear(embed_dim, output_dim)

    def forward(self, x):
        # 输入嵌入和位置编码
        x = self.embedding(x) + self.pos_embedding

        # 通过Transformer层
        for layer in self.layers:
            x = layer(x)

        # 输出层
        x = self.output_layer(x)
        return x

# 示例用法
if __name__ == "__main__":
    # 输入张量形状：(batch_size, seq_len, input_dim)
    batch_size = 64
    seq_len = 96
    input_dim = 8
    output_dim = 16

    model = GSformer(
        input_dim=input_dim,
        embed_dim=64,
        num_heads=8,
        num_layers=4,
        feedforward_dim=128,
        seq_len=seq_len,
        output_dim=output_dim,
        tau=0.1,
        dropout=0.1
    )

    x = torch.rand(batch_size, seq_len, input_dim)
    output = model(x)

    print("输入形状:", x.shape)
    print("输出形状:", output.shape)