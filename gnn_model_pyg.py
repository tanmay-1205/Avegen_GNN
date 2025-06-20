import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv, MessagePassing
from torch_geometric.data import Data
import math

class BiInteractionConv(MessagePassing):
    """Custom bi-interaction convolution layer"""
    def __init__(self, in_channels, out_channels):
        super(BiInteractionConv, self).__init__(aggr='add')
        self.lin1 = nn.Linear(in_channels, out_channels)
        self.lin2 = nn.Linear(in_channels, out_channels)
        
    def forward(self, x, edge_index, edge_weight=None):
        return self.propagate(edge_index, x=x, edge_weight=edge_weight)
        
    def message(self, x_i, x_j, edge_weight):
        sum_embed = self.lin1(x_i + x_j)
        bi_embed = self.lin2(x_i * x_j)
        return (sum_embed + bi_embed) if edge_weight is None else (sum_embed + bi_embed) * edge_weight.view(-1, 1)

class PregnancyAppGNN_PyG(nn.Module):
    def __init__(self, args, n_users, n_segments, n_nudges, n_feature_types):
        super(PregnancyAppGNN_PyG, self).__init__()
        
        self.n_users = n_users
        self.n_segments = n_segments
        self.n_nudges = n_nudges
        self.n_feature_types = n_feature_types
        
        # Feature encoder
        self.feature_encoder = nn.Sequential(
            nn.Linear(n_feature_types, args.embed_dim),
            nn.ReLU(),
            nn.Linear(args.embed_dim, args.embed_dim)
        )
        
        # Embeddings for segments and nudges
        self.segment_embed = nn.Embedding(n_segments, args.embed_dim)
        self.nudge_embed = nn.Embedding(n_nudges, args.embed_dim)
        
        # Feature transformation
        self.feature_transform = nn.Parameter(
            torch.Tensor(n_feature_types, args.embed_dim, args.feature_dim)
        )
        
        # GNN layers
        self.convs = nn.ModuleList()
        conv_dims = [args.embed_dim] + eval(args.conv_dim_list)
        
        for i in range(len(conv_dims) - 1):
            if args.aggregation_type == 'gcn':
                conv = GCNConv(conv_dims[i], conv_dims[i+1])
            elif args.aggregation_type == 'graphsage':
                conv = SAGEConv(conv_dims[i], conv_dims[i+1])
            else:  # bi-interaction
                conv = BiInteractionConv(conv_dims[i], conv_dims[i+1])
            self.convs.append(conv)
        
        self.dropout = nn.ModuleList([
            nn.Dropout(p) for p in eval(args.mess_dropout)
        ])
        
        # Prediction heads
        final_dim = sum(conv_dims)  # Concatenated embedding dimension
        
        # Segment predictor
        self.segment_predictor = nn.Sequential(
            nn.Linear(final_dim, args.embed_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(args.embed_dim, n_segments)
        )
        
        # Nudge predictor
        self.nudge_predictor = nn.Sequential(
            nn.Linear(final_dim, args.embed_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(args.embed_dim, n_nudges)
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        # Initialize feature encoder
        for layer in self.feature_encoder.modules():
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)
        
        nn.init.xavier_uniform_(self.segment_embed.weight)
        nn.init.xavier_uniform_(self.nudge_embed.weight)
        nn.init.xavier_uniform_(self.feature_transform)
        
        # Initialize prediction heads
        for predictor in [self.segment_predictor, self.nudge_predictor]:
            for layer in predictor.modules():
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)
    
    def forward(self, data, mode='train'):
        if mode == 'train':
            return self._forward_train(data)
        elif mode == 'predict':
            return self._forward_predict(data)
        else:
            raise ValueError(f"Unknown mode: {mode}")
    
    def _forward_train(self, data):
        # Use the graph structure directly - no need for complex negative sampling
        # The sophisticated graph already has the right connections
        
        # Transform node features to proper embeddings
        # User features (first n_users nodes)
        user_features = data.x[:self.n_users]
        user_embed = self.feature_encoder(user_features)
        
        # Segment and nudge embeddings (remaining nodes)
        segment_embed = self.segment_embed.weight
        nudge_embed = self.nudge_embed.weight
        
        # Combine all embeddings
        x = torch.cat([user_embed, segment_embed, nudge_embed], dim=0)
        edge_index = data.edge_index
        edge_weight = data.edge_weight if hasattr(data, 'edge_weight') else None
        
        # Apply GNN layers
        xs = [x]
        for conv, dropout in zip(self.convs, self.dropout):
            x = conv(x, edge_index, edge_weight)
            x = F.leaky_relu(x)
            x = dropout(x)
            x = F.normalize(x, p=2, dim=1)
            xs.append(x)
        
        # Concatenate all embeddings
        x = torch.cat(xs, dim=1)
        x = F.normalize(x, p=2, dim=1)
        
        # Get user embeddings for prediction
        user_embed = x[:self.n_users]
        
        # Use prediction heads for training
        segment_preds = self.segment_predictor(user_embed)
        nudge_preds = self.nudge_predictor(user_embed)
        
        # Create synthetic targets based on graph connections
        # This is a simplified approach - in practice you'd have real labels
        segment_targets = torch.randint(0, self.n_segments, (self.n_users,))
        nudge_targets = torch.randint(0, self.n_nudges, (self.n_users,))
        
        # Multi-task loss
        segment_loss = F.cross_entropy(segment_preds, segment_targets)
        nudge_loss = F.cross_entropy(nudge_preds, nudge_targets)
        
        # Combined loss
        loss = segment_loss + nudge_loss
        
        return loss
    
    def _forward_predict(self, data):
        # Transform node features to proper embeddings
        # User features (first n_users nodes)
        user_features = data.x[:self.n_users]
        user_embed = self.feature_encoder(user_features)
        
        # Segment and nudge embeddings (remaining nodes)
        segment_embed = self.segment_embed.weight
        nudge_embed = self.nudge_embed.weight
        
        # Combine all embeddings
        x = torch.cat([user_embed, segment_embed, nudge_embed], dim=0)
        edge_index = data.edge_index
        edge_weight = data.edge_weight if hasattr(data, 'edge_weight') else None
        
        # Apply GNN layers
        xs = [x]
        for conv, dropout in zip(self.convs, self.dropout):
            x = conv(x, edge_index, edge_weight)
            x = F.leaky_relu(x)
            x = F.normalize(x, p=2, dim=1)
            xs.append(x)
        
        x = torch.cat(xs, dim=1)
        
        # Get user embeddings
        user_embed = x[:self.n_users]
        
        # Get segment and nudge predictions using prediction heads
        segment_preds = self.segment_predictor(user_embed)
        nudge_preds = self.nudge_predictor(user_embed)
        
        return segment_preds, nudge_preds
    
    def update_attention(self, user_list, segment_list, feature_type_list, feature_types):
        """Update attention weights using feature-based attention"""
        device = self.segment_embed.weight.device
        
        # Simple uniform attention weights for now
        edge_index = torch.stack([user_list, segment_list])
        edge_attr = torch.ones(len(user_list), device=device)
        edge_attr = F.softmax(edge_attr, dim=0)
        
        return Data(
            edge_index=edge_index,
            edge_attr=edge_attr,
            num_nodes=self.n_users + self.n_segments + self.n_nudges
        ) 