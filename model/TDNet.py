import os
import sys

from torch.nn.functional import dropout

from model.GSAT import GSAT
from model.GSformer import GSformer
import torch
import torch.nn as nn
import torch.nn.functional as F

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)


class GCN_Layer(nn.Module):
    def __init__(self, num_of_features, num_of_filter):
        """One layer of GCN
        
        Arguments:
            num_of_features {int} -- the dimension of node feature
            num_of_filter {int} -- the number of graph filters
        """
        super(GCN_Layer, self).__init__()
        self.gcn_layer = nn.Sequential(
            nn.Linear(in_features=num_of_features,
                      out_features=num_of_filter),
            nn.ReLU()
        )

    def forward(self, input, adj):
        """计算一层GCN
        
        Arguments:
            input {Tensor} -- signal matrix,shape (batch_size,N,T*D)
            adj {np.array} -- adjacent matrix，shape (N,N)
        Returns:
            {Tensor} -- output,shape (batch_size,N,num_of_filter)
        """
        adj = adj.unsqueeze(0).expand(input.size(0), -1, -1)
        input = torch.bmm(adj, input)
        output = self.gcn_layer(input)
        return output


class STGeoModule(nn.Module):
    def __init__(self, grid_in_channel, num_of_gru_layers, seq_len,
                 gru_hidden_size, num_of_target_time_feature):
        """[summary]
        
        Arguments:
            grid_in_channel {int} -- the number of grid data feature (batch_size,T,D,W,H),grid_in_channel=D 48
            num_of_gru_layers {int} -- the number of GRU layers
            seq_len {int} -- the time length of input
            gru_hidden_size {int} -- the hidden size of GRU
            num_of_target_time_feature {int} -- the number of target time feature，为24(hour)+7(week)+1(holiday)=32
        """
        super(STGeoModule, self).__init__()
        self.grid_conv = nn.Sequential(
            nn.Conv2d(in_channels=grid_in_channel, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=grid_in_channel, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        self.grid_gru = nn.GRU(grid_in_channel, gru_hidden_size, num_of_gru_layers, batch_first=True)
        #calculate attention score
        self.grid_att_fc1 = nn.Linear(in_features=gru_hidden_size, out_features=1)
        self.grid_att_fc2 = nn.Linear(in_features=num_of_target_time_feature, out_features=seq_len)
        self.grid_att_bias = nn.Parameter(torch.zeros(1))
        self.grid_att_softmax = nn.Softmax(dim=-1)

    def forward(self, grid_input, target_time_feature):
        """
        Arguments:
            grid_input {Tensor} -- grid input，shape：(batch_size,seq_len,D,W,H)
            target_time_feature {Tensor} -- the feature of target time，shape：(batch_size,num_target_time_feature)
        Returns:
            {Tensor} -- shape：(batch_size,hidden_size,W,H)
        """
        batch_size, T, D, W, H = grid_input.shape  #64*7*48*20*20

        grid_input = grid_input.view(-1, D, W, H)
        conv_output = self.grid_conv(grid_input)

        conv_output = conv_output.view(batch_size, -1, D, W, H) \
            .permute(0, 3, 4, 1, 2) \
            .contiguous() \
            .view(-1, T, D)  #25600*7*48
        gru_output, _ = self.grid_gru(conv_output)  # 25600*7*256

        grid_target_time = torch.unsqueeze(target_time_feature, 1).repeat(1, W * H, 1).view(batch_size * W * H, -1)
        grid_att_fc1_output = torch.squeeze(self.grid_att_fc1(gru_output))
        grid_att_fc2_output = self.grid_att_fc2(grid_target_time)
        grid_att_score = self.grid_att_softmax(F.relu(grid_att_fc1_output + grid_att_fc2_output + self.grid_att_bias))
        grid_att_score = grid_att_score.view(batch_size * W * H, -1, 1)
        grid_output = torch.sum(gru_output * grid_att_score, dim=1)

        grid_output = grid_output.view(batch_size, W, H, -1).permute(0, 3, 1, 2).contiguous()  #64*256*20*20

        return grid_output


class STSemModule(nn.Module):
    def __init__(self, num_of_graph_feature, nums_of_graph_filters,
                 seq_len, num_of_gru_layers, gru_hidden_size,
                 num_of_target_time_feature, north_south_map, west_east_map):
        """
        Arguments:
            num_of_graph_feature {int} -- the number of graph node feature，(batch_size,seq_len,D,N),num_of_graph_feature=D
            nums_of_graph_filters {list} -- the number of GCN output feature
            seq_len {int} -- the time length of input
            num_of_gru_layers {int} -- the number of GRU layers
            gru_hidden_size {int} -- the hidden size of GRU
            num_of_target_time_feature {int} -- the number of target time feature，为24(hour)+7(week)+1(holiday)=32
            north_south_map {int} -- the weight of grid data
            west_east_map {int} -- the height of grid data

        """
        super(STSemModule, self).__init__()
        self.north_south_map = north_south_map
        self.west_east_map = west_east_map
        self.road_gcn = nn.ModuleList()
        self.risk_gcn = nn.ModuleList()
        # create road and risk gcn，num_of_filter=3 (in/out flow + risk)
        for idx, num_of_filter in enumerate(nums_of_graph_filters):
            if idx == 0:
                self.road_gcn.append(GCN_Layer(num_of_graph_feature, num_of_filter))
                self.risk_gcn.append(GCN_Layer(num_of_graph_feature, num_of_filter))
            else:
                self.road_gcn.append(GCN_Layer(nums_of_graph_filters[idx - 1], num_of_filter))
                self.risk_gcn.append(GCN_Layer(nums_of_graph_filters[idx - 1], num_of_filter))

        self.graph_gru = nn.GRU(num_of_filter, gru_hidden_size, num_of_gru_layers, batch_first=True)
        #calculate attention score
        self.graph_att_fc1 = nn.Linear(in_features=gru_hidden_size, out_features=1)
        self.graph_att_fc2 = nn.Linear(in_features=num_of_target_time_feature, out_features=seq_len)
        self.graph_att_bias = nn.Parameter(torch.zeros(1))
        self.graph_att_softmax = nn.Softmax(dim=-1)

    def forward(self, graph_feature, road_adj, risk_adj,
                target_time_feature, grid_node_map):
        """
        Arguments:
            graph_feature {Tensor} -- Graph signal matrix，(batch_size,T,D1,N)
            road_adj {np.array} -- road adjacent matrix，shape：(N,N)
            risk_adj {np.array} -- risk adjacent matrix，shape：(N,N)
            target_time_feature {Tensor} -- the feature of target time，shape：(batch_size,num_target_time_feature)
            grid_node_map {np.array} -- map graph data to grid data,shape (W*H,N)
        Returns:
            {Tensor} -- shape：(batch_size,pre_len,north_south_map,west_east_map)
        """
        batch_size, T, D1, N = graph_feature.shape  #64*7*3*243
        road_graph_output = graph_feature.view(-1, D1, N).permute(0, 2, 1).contiguous()

        for gcn_layer in self.road_gcn:
            road_graph_output = gcn_layer(road_graph_output, road_adj)

        risk_graph_output = graph_feature.view(-1, D1, N).permute(0, 2, 1).contiguous()
        for gcn_layer in self.risk_gcn:
            risk_graph_output = gcn_layer(risk_graph_output, risk_adj)

        graph_output = road_graph_output + risk_graph_output

        graph_output = graph_output.view(batch_size, T, N, -1) \
            .permute(0, 2, 1, 3) \
            .contiguous() \
            .view(batch_size * N, T, -1)  #15552*7*64
        graph_output, _ = self.graph_gru(graph_output)  #15552*7*256

        graph_target_time = torch.unsqueeze(target_time_feature, 1).repeat(1, N, 1).view(batch_size * N, -1)
        graph_att_fc1_output = torch.squeeze(self.graph_att_fc1(graph_output))
        graph_att_fc2_output = self.graph_att_fc2(graph_target_time)
        graph_att_score = self.graph_att_softmax(
            F.relu(graph_att_fc1_output + graph_att_fc2_output + self.graph_att_bias))
        graph_att_score = graph_att_score.view(batch_size * N, -1, 1)
        graph_output = torch.sum(graph_output * graph_att_score, dim=1)
        graph_output = graph_output.view(batch_size, N, -1).contiguous()

        grid_node_map_tmp = torch.from_numpy(grid_node_map) \
            .to(graph_feature.device) \
            .repeat(batch_size, 1, 1)
        graph_output = torch.bmm(grid_node_map_tmp, graph_output) \
            .permute(0, 2, 1) \
            .view(batch_size, -1, self.north_south_map, self.west_east_map)  #TODO：243映射回400
        return graph_output


class TDNet(nn.Module):
    def __init__(self, seq_len, pre_len, num_of_nodes, num_of_input_feature,
                 st_embedding_dim, north_south_map, west_east_map):
        """[summary]
        
        Arguments:
            grid_in_channel {int} -- the number of grid data feature (batch_size,T,D,W,H),grid_in_channel=D
            num_of_gru_layers {int} -- the number of GRU layers
            seq_len {int} -- the time length of input
            pre_len {int} -- the time length of prediction
            gru_hidden_size {int} -- the hidden size of GRU
            nums_of_graph_filters {list} -- the number of GCN output feature
            north_south_map {int} -- the weight of grid data
            west_east_map {int} -- the height of grid data
            spatial_temporal_embedding_dim {int} -- the spatial temporal embedding dim
        """
        super(TDNet, self).__init__()
        self.north_south_map = north_south_map
        self.west_east_map = west_east_map
        final_embedding_dim = 32
        # TODO：匹配维度
        self.spatial_model = GSAT(in_features=seq_len * num_of_input_feature,
                                  out_features=st_embedding_dim, num_heads=8, hidden_features=128)

        self.temporal_model = GSformer(input_dim=num_of_nodes, output_dim=self.north_south_map * self.west_east_map,
                                       embed_dim=64, num_heads=4, num_layers=3, feedforward_dim=128,
                                       seq_len=st_embedding_dim, tau=0.1, dropout=0.1)

        self.middle_layer = nn.Linear(st_embedding_dim * north_south_map * west_east_map,
                                      final_embedding_dim * north_south_map * west_east_map)
        self.dropout = nn.Dropout(0.2)
        self.activation =nn.GELU()
        self.output_layer = nn.Linear(final_embedding_dim * north_south_map * west_east_map,
                                      pre_len * north_south_map * west_east_map)

    def forward(self, grid_input, target_time_feature, graph_feature,
                road_adj, risk_adj, grid_node_map):
        """
        Arguments:
            grid_input {Tensor} -- grid input，shape：(batch_size,T,D,W,H)
            graph_feature {Tensor} -- Graph signal matrix，(batch_size,T,D1,N)
            target_time_feature {Tensor} -- the feature of target time，shape：(batch_size,num_target_time_feature)
            road_adj {np.array} -- road adjacent matrix，shape：(N,N)
            risk_adj {np.array} -- risk adjacent matrix，shape：(N,N)
            grid_node_map {np.array} -- map graph data to grid data,shape (W*H,N)

        Returns:
            {Tensor} -- shape：(batch_size,pre_len,north_south_map,west_east_map)
        """
        batch_size, _, _, _, _ = grid_input.shape

        spatial_grid_input = grid_input.view(batch_size, -1, self.north_south_map * self.west_east_map)

        spatial_graph_input = torch.matmul(spatial_grid_input, torch.from_numpy(grid_node_map) \
                                           .to(graph_feature.device))

        spatial_graph_input = spatial_graph_input.permute(0, 2, 1).contiguous()
        spatial_output = self.spatial_model(spatial_graph_input, road_adj)  #64，243，128
        temporal_input = spatial_output.permute(0, 2, 1).contiguous()
        temporal_output = self.temporal_model(temporal_input)
        #grid_output = self.st_geo_module(grid_input,target_time_feature)
        #graph_output = self.st_sem_module(graph_feature,road_adj,risk_adj,
        #target_time_feature,grid_node_map)
        temporal_output = temporal_output.view(batch_size, -1)
        middle_output = self.middle_layer(temporal_output)
        middle_output = self.activation(middle_output)
        middle_output = self.dropout(middle_output)
        final_output = self.output_layer(middle_output).view(batch_size, -1, self.north_south_map, self.west_east_map)
        return final_output
        #TODO：输出还是看齐20x20的而不是243，243是中间过渡维度，双路的output输出都是20x20
