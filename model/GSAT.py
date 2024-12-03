import torch
import torch.nn as nn
import torch.nn.functional as F

class GSAT(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, num_heads, tau=1.0):
        super(GSAT, self).__init__()
        self.in_features = in_features
        self.hidden_features = hidden_features
        self.out_features = out_features
        self.num_heads = num_heads
        self.tau = tau  # Gumbel-Softmax 温度参数

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
        eps = torch.finfo(logits.dtype).eps  # 获取数据类型的机器精度
        uniform_noise = torch.rand_like(logits)
        gumbel_noise = -torch.log(-torch.log(uniform_noise + eps) + eps)
        y = logits + gumbel_noise
        return F.softmax(y / tau, dim=-2)  # 注意这里的维度

    def attention_layer(self, h, a_src, a_dst, adj):
        # 调整 a_src 和 a_dst 的维度
        a_src_expanded = a_src.unsqueeze(0).unsqueeze(1)  # 形状：(1, 1, num_heads, features)
        a_dst_expanded = a_dst.unsqueeze(0).unsqueeze(1)

        # 计算源和目标节点的注意力投影
        h_src = (h * a_src_expanded).sum(dim=-1)  # 形状：(batch_size, num_nodes, num_heads)
        h_dst = (h * a_dst_expanded).sum(dim=-1)

        # 计算注意力得分
        scores = h_src.unsqueeze(2) + h_dst.unsqueeze(1)  # 形状：(batch_size, num_nodes, num_nodes, num_heads)

        # 应用邻接矩阵掩码
        adj_expanded = adj.unsqueeze(0).unsqueeze(-1)  # 形状：(1, num_nodes, num_nodes, 1)
        scores = scores * adj_expanded

        # 使用 Gumbel-Softmax 计算稀疏注意力
        sparse_attention = self.gumbel_softmax(scores, self.tau)  # 在第二维（邻居）上 softmax

        # 信息传递：邻居特征的加权和
        h_prime = torch.einsum("bmnk,bmhd->bnhd", sparse_attention, h)  # 形状：(batch_size, num_nodes, num_heads, features)

        return h_prime

    def forward(self, x, adj):
        batch_size, num_nodes, _ = x.size()

        #### 第一层图卷积 ####

        # 线性变换并重塑以适应多头注意力
        h1 = self.W1(x).view(batch_size, num_nodes, self.num_heads, self.hidden_features)

        # 使用注意力层
        h_prime1 = self.attention_layer(h1, self.a_src1, self.a_dst1, adj)

        # 合并多头输出
        h_prime1 = h_prime1.reshape(batch_size, num_nodes, -1)
        h_prime1 = F.leaky_relu(self.output_layer1(h_prime1))

        #### 第二层图卷积 ####

        h2 = self.W2(h_prime1).view(batch_size, num_nodes, self.num_heads, self.out_features)
        h_prime2 = self.attention_layer(h2, self.a_src2, self.a_dst2, adj)
        h_prime2 = h_prime2.reshape(batch_size, num_nodes, -1)
        h_prime2 = F.leaky_relu(self.output_layer2(h_prime2))

        return h_prime2

# 示例用法
if __name__ == "__main__":
    # 初始化模型
    gat_model = GSAT(
        in_features=128, hidden_features=32, out_features=64, num_heads=8, tau=0.5
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