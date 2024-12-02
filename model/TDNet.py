import os
import sys
from model.GSAT import GSAT
from model.GSformer import GSformer
import torch
import torch.nn as nn

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)


class TDNet(nn.Module):
    def __init__(self, seq_len, pre_len, num_of_nodes, num_of_input_feature,
                 st_embedding_dim, north_south_map, west_east_map, grid_node_map):
        """
        Args:
            seq_len (int): The time length of input.
            pre_len (int): The time length of prediction.
            num_of_nodes (int): Number of nodes in the graph.
            num_of_input_feature (int): Number of input features.
            st_embedding_dim (int): Spatial-temporal embedding dimension.
            north_south_map (int): Height of the grid data.
            west_east_map (int): Width of the grid data.
            grid_node_map (np.array): Map graph data to grid data, shape (W*H, N).
        """
        super(TDNet, self).__init__()
        self.north_south_map = north_south_map
        self.west_east_map = west_east_map
        self.num_of_nodes = num_of_nodes
        final_embedding_dim = 32

        # Convert grid_node_map to a torch tensor once during initialization
        self.register_buffer('grid_node_map', torch.from_numpy(grid_node_map))

        # Spatial model
        self.spatial_model = GSAT(
            in_features=seq_len * num_of_input_feature,
            out_features=st_embedding_dim,
            num_heads=8,
            hidden_features=128
        )

        # Temporal model
        self.temporal_model = GSformer(
            input_dim=num_of_nodes,
            output_dim=north_south_map * west_east_map,
            embed_dim=64,
            num_heads=4,
            num_layers=3,
            feedforward_dim=128,
            seq_len=st_embedding_dim,
            tau=0.1,
            dropout=0.1
        )

        # Middle and output layers
        self.middle_layer = nn.Linear(
            st_embedding_dim * north_south_map * west_east_map,
            final_embedding_dim * north_south_map * west_east_map
        )
        self.dropout = nn.Dropout(0.2)
        self.activation = nn.GELU()
        self.output_layer = nn.Linear(
            final_embedding_dim * north_south_map * west_east_map,
            pre_len * north_south_map * west_east_map
        )

    def forward(self, grid_input, target_time_feature, graph_feature,
                road_adj, risk_adj):
        """
        Args:
            grid_input (Tensor): Grid input, shape (batch_size, T, D, W, H).
            target_time_feature (Tensor): Time features of the target.
            graph_feature (Tensor): Graph signal matrix, shape (batch_size, T, D1, N).
            road_adj (Tensor): Road adjacency matrix, shape (N, N).
            risk_adj (Tensor): Risk adjacency matrix, shape (N, N).

        Returns:
            Tensor: Predicted output, shape (batch_size, pre_len, north_south_map, west_east_map).
        """
        batch_size = grid_input.size(0)

        # Reshape grid input
        spatial_grid_input = grid_input.view(batch_size, -1, self.north_south_map * self.west_east_map)

        # Move grid_node_map to the appropriate device
        grid_node_map = self.grid_node_map.to(grid_input.device)

        # Multiply with grid_node_map
        spatial_graph_input = torch.matmul(spatial_grid_input, grid_node_map)

        # Permute dimensions for the spatial model
        spatial_graph_input = spatial_graph_input.permute(0, 2, 1).contiguous()

        # Pass through spatial model
        spatial_output = self.spatial_model(spatial_graph_input, road_adj)  # Shape: (batch_size, N, st_embedding_dim)

        # Prepare input for temporal model
        temporal_input = spatial_output.permute(0, 2, 1).contiguous()  # Shape: (batch_size, st_embedding_dim, N)

        # Pass through temporal model
        temporal_output = self.temporal_model(temporal_input)  # Shape: (batch_size, st_embedding_dim, output_dim)

        # Flatten temporal output
        temporal_output = temporal_output.view(batch_size, -1)

        # Middle layer processing
        middle_output = self.middle_layer(temporal_output)
        middle_output = self.activation(middle_output)
        middle_output = self.dropout(middle_output)

        # Output layer
        final_output = self.output_layer(middle_output)
        final_output = final_output.view(batch_size, -1, self.north_south_map, self.west_east_map)

        return final_output
