import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv, GATConv
from torch_geometric.nn import global_mean_pool

class PregnancyGNN(torch.nn.Module):
    def __init__(self, 
                 input_dim, 
                 hidden_dim=64, 
                 num_layers=3,
                 dropout=0.2,
                 conv_type='sage'):
        """
        Initialize the GNN model for pregnancy engagement prediction
        
        Args:
            input_dim: Dimension of input node features
            hidden_dim: Dimension of hidden layers
            num_layers: Number of GNN layers
            dropout: Dropout rate
            conv_type: Type of GNN layer ('gcn', 'sage', or 'gat')
        """
        super(PregnancyGNN, self).__init__()
        
        self.num_layers = num_layers
        self.dropout = dropout
        self.hidden_dim = hidden_dim
        
        # Initialize list to hold GNN layers
        self.convs = nn.ModuleList()
        
        # Input layer
        if conv_type == 'gcn':
            self.convs.append(GCNConv(input_dim, hidden_dim))
        elif conv_type == 'sage':
            self.convs.append(SAGEConv(input_dim, hidden_dim))
        elif conv_type == 'gat':
            self.convs.append(GATConv(input_dim, hidden_dim))
        
        # Hidden layers
        for _ in range(num_layers - 1):
            if conv_type == 'gcn':
                self.convs.append(GCNConv(hidden_dim, hidden_dim))
            elif conv_type == 'sage':
                self.convs.append(SAGEConv(hidden_dim, hidden_dim))
            elif conv_type == 'gat':
                self.convs.append(GATConv(hidden_dim, hidden_dim))
        
        # Output layers for different tasks
        self.segment_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 6)  # 6 segments
        )
        
        self.nudge_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 5)  # 5 nudge types
        )
        
    def forward(self, x, edge_index, batch=None):
        """
        Forward pass of the model
        
        Args:
            x: Node features
            edge_index: Graph connectivity
            batch: Batch vector for multiple graphs (optional)
        """
        # Initial features
        h = x
        
        # Message passing layers
        for i in range(self.num_layers):
            h = self.convs[i](h, edge_index)
            if i != self.num_layers - 1:  # No activation on last layer
                h = F.relu(h)
                h = F.dropout(h, p=self.dropout, training=self.training)
        
        # Get predictions for different node types
        segment_preds = self.segment_predictor(h)
        nudge_preds = self.nudge_predictor(h)
        
        return segment_preds, nudge_preds
    
    def get_embeddings(self, x, edge_index):
        """
        Get node embeddings from the model
        """
        h = x
        for i in range(self.num_layers):
            h = self.convs[i](h, edge_index)
            if i != self.num_layers - 1:
                h = F.relu(h)
        return h

# Example usage
if __name__ == "__main__":
    # Load the graph data
    from graph_construction import GraphConstructor
    from data_preprocessing import PregnancyDataPreprocessor
    import pandas as pd
    
    # Load and preprocess data
    print("Loading and preprocessing data...")
    df = pd.read_csv('synthetic_pregnancy_app_data.csv')
    preprocessor = PregnancyDataPreprocessor()
    df_cleaned, df_engagement, df_health = preprocessor.fit_transform(df)
    
    # Create graph
    print("Constructing graph...")
    graph_constructor = GraphConstructor()
    graph_data = graph_constructor.create_graph(df_cleaned, df_engagement, df_health)
    
    # Initialize model
    print("Initializing model...")
    input_dim = graph_data.x.size(1)  # Feature dimension
    model = PregnancyGNN(input_dim=input_dim)
    
    # Test forward pass
    print("Testing forward pass...")
    segment_preds, nudge_preds = model(graph_data.x, graph_data.edge_index)
    
    # Print model statistics
    print("\nModel Statistics:")
    print(f"Input dimension: {input_dim}")
    print(f"Segment predictions shape: {segment_preds.shape}")
    print(f"Nudge predictions shape: {nudge_preds.shape}")
    
    # Get embeddings
    print("\nGenerating embeddings...")
    embeddings = model.get_embeddings(graph_data.x, graph_data.edge_index)
    print(f"Embedding shape: {embeddings.shape}")