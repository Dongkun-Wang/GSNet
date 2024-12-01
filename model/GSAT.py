import torch
import torch.nn as nn
import torch.nn.functional as F


class GSAT(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, num_heads, tau=1.0, epsilon=1e-9):
        super(GSAT, self).__init__()
        self.in_features = in_features
        self.hidden_features = hidden_features
        self.out_features = out_features
        self.num_heads = num_heads
        self.tau = tau  # Gumbel-Softmax 温度参数
        self.epsilon = epsilon  # 防止 log(0)

        # 定义第一层注意力的可学习参数
        self.W1 = nn.Linear(in_features, num_heads * hidden_features, bias=False)
        self.a_src1 = nn.Parameter(torch.zeros(size=(num_heads, hidden_features)))
        self.a_dst1 = nn.Parameter(torch.zeros(size=(num_heads, hidden_features)))

        # 定义第二层注意力的可学习参数
        self.W2 = nn.Linear(hidden_features, num_heads * out_features, bias=False)
        self.a_src2 = nn.Parameter(torch.zeros(size=(num_heads, out_features)))
        self.a_dst2 = nn.Parameter(torch.zeros(size=(num_heads, out_features)))

        # 参数初始化
        nn.init.xavier_uniform_(self.W1.weight)
        nn.init.xavier_uniform_(self.a_src1)
        nn.init.xavier_uniform_(self.a_dst1)
        nn.init.xavier_uniform_(self.W2.weight)
        nn.init.xavier_uniform_(self.a_src2)
        nn.init.xavier_uniform_(self.a_dst2)

        # 用于合并多头输出的输出层
        self.output_layer1 = nn.Linear(num_heads * hidden_features, hidden_features, bias=False)
        self.output_layer2 = nn.Linear(num_heads * out_features, out_features, bias=False)

    def gumbel_softmax(self, logits, tau):
        """Gumbel-Softmax 的稀疏注意力实现。"""
        gumbel_noise = -torch.log(-torch.log(torch.rand_like(logits) + self.epsilon) + self.epsilon)
        y = logits + gumbel_noise
        return F.softmax(y / tau, dim=-1)

    def forward(self, x, adj):
        """
        带有 Gumbel-Softmax 注意力的 GAT 前向传播。

        参数：
        x: 输入特征，形状为 (batch_size, num_nodes, in_features)
        adj: 邻接矩阵，形状为 (num_nodes, num_nodes)，在批次间共享。

        返回：
        更新后的节点嵌入，形状为 (batch_size, num_nodes, out_features)
        """
        batch_size, num_nodes, _ = x.size()

        #### 第一层图卷积 ####

        # 线性变换并重塑以适应多头注意力
        h1 = self.W1(x).view(batch_size, num_nodes, self.num_heads, self.hidden_features)
        h1_src = torch.einsum("bnhd,hd->bnh", h1, self.a_src1)
        h1_dst = torch.einsum("bnhd,hd->bnh", h1, self.a_dst1)

        # 计算注意力得分
        scores1 = h1_src.unsqueeze(2) + h1_dst.unsqueeze(1)
        scores1 = scores1.mean(dim=-1)

        # 应用邻接矩阵掩码
        scores1 = scores1 * adj.unsqueeze(0)

        # 使用 Gumbel-Softmax 计算稀疏注意力
        gumbel_attention1 = self.gumbel_softmax(scores1, self.tau)

        # 归一化稀疏注意力
        sparse_attention1 = F.softmax(gumbel_attention1, dim=-1)

        # 信息传递：邻居特征的加权和
        h_prime1 = torch.einsum("bmn,bmhd->bnhd", sparse_attention1, h1)

        # 合并多头输出
        h_prime1 = h_prime1.reshape(batch_size, num_nodes, -1)
        h_prime1 = self.output_layer1(h_prime1)

        #### 第二层图卷积 ####

        # 线性变换并重塑以适应多头注意力
        h2 = self.W2(h_prime1).view(batch_size, num_nodes, self.num_heads, self.out_features)
        h2_src = torch.einsum("bnhd,hd->bnh", h2, self.a_src2)
        h2_dst = torch.einsum("bnhd,hd->bnh", h2, self.a_dst2)

        # 计算注意力得分
        scores2 = h2_src.unsqueeze(2) + h2_dst.unsqueeze(1)
        scores2 = scores2.mean(dim=-1)

        # 应用邻接矩阵掩码
        scores2 = scores2 * adj.unsqueeze(0)

        # 使用 Gumbel-Softmax 计算稀疏注意力
        gumbel_attention2 = self.gumbel_softmax(scores2, self.tau)

        # 归一化稀疏注意力
        sparse_attention2 = F.softmax(gumbel_attention2, dim=-1)

        # 信息传递：邻居特征的加权和
        h_prime2 = torch.einsum("bmn,bmhd->bnhd", sparse_attention2, h2)

        # 合并多头输出
        h_prime2 = h_prime2.reshape(batch_size, num_nodes, -1)
        h_prime2 = self.output_layer2(h_prime2)

        return h_prime2

# Example usage
if __name__ == "__main__":
    # 初始化模型
    gat_model = GSAT(
        in_features=128,hidden_features=32, out_features=64, num_heads=8, tau=0.5
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