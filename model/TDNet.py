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
            road_adj {np.array} -- road adjacent matrix，shape：(N,N)
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
        temporal_output = temporal_output.view(batch_size, -1)

        middle_output = self.middle_layer(temporal_output)
        middle_output = self.activation(middle_output)
        middle_output = self.dropout(middle_output)
        final_output = self.output_layer(middle_output).view(batch_size, -1, self.north_south_map, self.west_east_map)

        return final_output
