import torch
import torch.nn as nn
import torch.nn.functional as F


class GSAT(nn.Module):
    def __init__(self, in_features, out_features, num_heads, tau=1.0, epsilon=1e-9):
        super(GSAT, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_heads = num_heads
        self.tau = tau  # Gumbel-Softmax temperature
        self.epsilon = epsilon  # To avoid log(0)

        # Define learnable parameters for attention
        self.W = nn.Linear(in_features, num_heads * out_features, bias=False)
        self.a_src = nn.Parameter(torch.zeros(size=(num_heads, out_features)))
        self.a_dst = nn.Parameter(torch.zeros(size=(num_heads, out_features)))

        # Initialize parameters
        nn.init.xavier_uniform_(self.W.weight)
        nn.init.xavier_uniform_(self.a_src)
        nn.init.xavier_uniform_(self.a_dst)

        # Output layer to combine multi-head outputs
        self.output_layer = nn.Linear(num_heads * out_features, out_features, bias=False)

    def gumbel_softmax(self, logits, tau):
        """Gumbel-Softmax implementation for sparse attention."""
        gumbel_noise = -torch.log(-torch.log(torch.rand_like(logits) + self.epsilon) + self.epsilon)
        y = logits + gumbel_noise
        return F.softmax(y / tau, dim=-1)

    def forward(self, x, adj):
        """
        Forward pass of GAT with Gumbel-Softmax attention.

        Args:
        x: Input features (batch_size, num_nodes, in_features)
        adj: Adjacency matrix (num_nodes, num_nodes), shared across the batch.

        Returns:
        Updated node embeddings (batch_size, num_nodes, out_features)
        """
        batch_size,num_nodes,_  = x.size()

        # Linear transformation and reshape for multi-head attention
        h = self.W(x).view(batch_size, num_nodes, self.num_heads,
                           self.out_features)  # (batch_size, num_nodes, num_heads, out_features)
        h_src = torch.einsum("bnhd,hd->bnh", h, self.a_src)  # (batch_size, num_nodes, num_heads)
        h_dst = torch.einsum("bnhd,hd->bnh", h, self.a_dst)  # (batch_size, num_nodes, num_heads)

        # Compute attention scores
        scores = h_src.unsqueeze(2) + h_dst.unsqueeze(1)  # (batch_size, num_nodes, num_nodes, num_heads) TODO：计算节点之间的注意力 为空间注意力
        scores = scores.mean(dim=-1)  # Aggregate across heads -> (batch_size, num_nodes, num_nodes)

        # Apply adjacency mask
        scores = scores * adj.unsqueeze(0)  # Mask with adjacency, shape (batch_size, num_nodes, num_nodes)

        # Apply Gumbel-Softmax to compute sparse attention
        gumbel_attention = self.gumbel_softmax(scores, self.tau)  # (batch_size, num_nodes, num_nodes)

        # Normalize sparse attention
        sparse_attention = F.softmax(gumbel_attention, dim=-1)  # (batch_size, num_nodes, num_nodes)

        # Message passing: Weighted sum of neighbor features
        h_prime = torch.einsum("bmn,bmhd->bnhd", sparse_attention,
                               h)  # Aggregate -> (batch_size, num_nodes, num_heads, out_features)

        # Combine multi-head outputs
        h_prime = h_prime.reshape(batch_size, num_nodes,
                                  -1)  # Flatten heads -> (batch_size, num_nodes, num_heads * out_features)
        h_prime = self.output_layer(h_prime)  # Combine heads -> (batch_size, num_nodes, out_features)

        return h_prime

# Example usage
if __name__ == "__main__":
    # 初始化模型
    gat_model = GSAT(
        in_features=128, out_features=64, num_heads=8, tau=0.5
    )

    # 输入数据
    batch_size = 32
    num_nodes = 100
    in_features = 128

    x = torch.randn(batch_size, num_nodes, in_features)  # 节点特征
    adj = (torch.rand(num_nodes, num_nodes) > 0.8).float()  # 全局邻接矩阵

    # 前向传播
    output = gat_model(x, adj)
    print(output.shape)  # (batch_size, num_nodes, out_features)
    #计算的是不同节点之间的注意力关系