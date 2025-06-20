import argparse
import torch
import torch.optim as optim
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import pandas as pd
import numpy as np
from gnn_model_pyg import PregnancyAppGNN_PyG
from graph_construction import GraphConstructor
import logging
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser()
    
    # Model arguments
    parser.add_argument('--embed_dim', type=int, default=128)
    parser.add_argument('--feature_dim', type=int, default=64)
    parser.add_argument('--aggregation_type', type=str, default='bi-interaction',
                      choices=['gcn', 'graphsage', 'bi-interaction'])
    parser.add_argument('--conv_dim_list', type=str, default='[128, 64, 32]')
    parser.add_argument('--mess_dropout', type=str, default='[0.2, 0.2, 0.2]')
    
    # Training arguments
    parser.add_argument('--lr', type=float, default=0.0005)
    parser.add_argument('--n_epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--n_segments', type=int, default=5)
    parser.add_argument('--n_neg_samples', type=int, default=5)
    
    # Data arguments
    parser.add_argument('--data_path', type=str, default='preprocessed_data/preprocessed_data.csv')
    
    return parser.parse_args()

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('training_pyg.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def create_pyg_data_sophisticated(df, feature_columns):
    """Create sophisticated PyG data object using GraphConstructor"""
    logger = logging.getLogger(__name__)
    logger.info("Creating sophisticated graph structure...")
    
    # Split features for graph construction (matching graph_construction.py requirements)
    engagement_features = [
        'session_start_count', 'watch_now_count', 'community_opened_count',
        'health_tracking_submission_count', 'content_completion_count',
        'total_engagement_score', 'content_interaction_ratio', 
        'health_tracking_ratio', 'activity_consistency', 'engagement_trend'
    ]
    
    health_features = ['bmi']
    
    # Select only existing columns
    available_engagement = [col for col in engagement_features if col in df.columns]
    available_health = [col for col in health_features if col in df.columns]
    
    logger.info(f"Available engagement features: {len(available_engagement)}")
    logger.info(f"Available health features: {len(available_health)}")
    
    # Create feature dataframes
    df_engagement = df[available_engagement] if available_engagement else pd.DataFrame()
    df_health = df[available_health] if available_health else pd.DataFrame()
    
    # If no health features available, create dummy BMI column
    if df_health.empty:
        df_health = pd.DataFrame({'bmi': np.random.normal(0, 1, len(df))})
        logger.info("Created dummy BMI column for graph construction")
    
    # Create graph using GraphConstructor
    graph_constructor = GraphConstructor()
    data = graph_constructor.create_graph(df, df_engagement, df_health)
    
    logger.info(f"Sophisticated graph created:")
    logger.info(f"  - Node features: {data.x.size()}")
    logger.info(f"  - Edge index: {data.edge_index.size()}")
    logger.info(f"  - Edge types: {len(torch.unique(data.edge_type))}")
    logger.info(f"  - Edge weights: {data.edge_weight.size()}")
    
    return data

def generate_negative_samples_heterogeneous(edge_index, node_type, n_neg):
    """Generate negative samples for heterogeneous graph"""
    # Get user-segment edges only (edge_type 0 and 1 are user-segment connections)
    user_indices = []
    segment_indices = []
    
    for i in range(edge_index.size(1)):
        src, dst = edge_index[0, i].item(), edge_index[1, i].item()
        # Check if this is a user-segment edge (user nodes have type 0, segment nodes have type 1)
        if node_type[src].item() == 0 and node_type[dst].item() == 1:
            user_indices.append(src)
            # Convert absolute segment index to relative segment index
            segment_idx = dst - (node_type == 0).sum().item()  # Subtract number of users
            segment_indices.append(segment_idx)
    
    if not user_indices:
        # Fallback: create dummy negative samples
        return torch.zeros((1, n_neg), dtype=torch.long)
    
    # Generate negative samples for each positive user-segment pair
    n_segments = (node_type == 1).sum().item()
    neg_segments = []
    
    for pos_seg in segment_indices:
        available_segments = list(set(range(n_segments)) - {pos_seg})
        if len(available_segments) < n_neg:
            neg = np.random.choice(available_segments, size=n_neg, replace=True)
        else:
            neg = np.random.choice(available_segments, size=n_neg, replace=False)
        neg_segments.append(neg)
    
    return torch.LongTensor(neg_segments)

def train_epoch(model, optimizer, train_loader, device, args):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    n_batches = 0
    
    for batch in train_loader:
        batch = batch.to(device)
        
        # Forward pass - no need for negative sampling with new approach
        loss = model(batch, mode='train')
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        total_loss += loss.item()
        n_batches += 1
    
    return total_loss / n_batches

def evaluate(model, data_loader, device):
    """Evaluate the model"""
    model.eval()
    total_loss = 0
    n_batches = 0
    
    with torch.no_grad():
        for batch in data_loader:
            batch = batch.to(device)
            
            # Forward pass for evaluation
            loss = model(batch, mode='train')
            
            total_loss += loss.item()
            n_batches += 1
    
    return total_loss / n_batches

def main():
    args = parse_args()
    logger = setup_logging()
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Load data
    logger.info("Loading data...")
    df = pd.read_csv(args.data_path)
    
    # Extract feature columns
    exclude_columns = ['user_id_hashed', 'user_id', 'city', 'selected_occupation', 'selected_reasons', 
                      'conditions', 'last_session_date', 'last_health_check_date', 'device_type', 
                      'app_version', 'city_tier', 'age_group', 'bmi_category', 'trimester', 'risk_level',
                      'pregnancy_experience']
    feature_columns = [col for col in df.columns if col not in exclude_columns]
    
    # Create sophisticated PyG data using GraphConstructor
    data = create_pyg_data_sophisticated(df, feature_columns)
    logger.info(f"Data loaded successfully with sophisticated graph structure")
    
    # Split into train/val
    train_size = int(0.8 * len(data.edge_index[0]))
    perm = torch.randperm(len(data.edge_index[0]))
    
    train_data = Data(
        x=data.x,
        edge_index=data.edge_index[:, perm[:train_size]],
        edge_weight=data.edge_weight[perm[:train_size]],
        edge_type=data.edge_type[perm[:train_size]],
        node_type=data.node_type,
        num_nodes=data.num_nodes
    )
    
    val_data = Data(
        x=data.x,
        edge_index=data.edge_index[:, perm[train_size:]],
        edge_weight=data.edge_weight[perm[train_size:]],
        edge_type=data.edge_type[perm[train_size:]],
        node_type=data.node_type,
        num_nodes=data.num_nodes
    )
    
    logger.info(f"Train data - edges: {train_data.edge_index.size(1)}, types: {len(torch.unique(train_data.edge_type))}")
    logger.info(f"Val data - edges: {val_data.edge_index.size(1)}, types: {len(torch.unique(val_data.edge_type))}")
    
    # Create data loaders
    train_loader = DataLoader([train_data], batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader([val_data], batch_size=args.batch_size)
    
    # Initialize model
    logger.info("Initializing model...")
    
    # Calculate number of users from node_type (node_type == 0 are users)
    n_users = (data.node_type == 0).sum().item()
    n_segments = (data.node_type == 1).sum().item()  # Should be 6 from GraphConstructor
    n_nudges = (data.node_type == 2).sum().item()    # Should be 5 from GraphConstructor
    n_feature_types = data.x.size(1)  # Feature dimension from the graph
    
    logger.info(f"Model parameters: users={n_users}, segments={n_segments}, nudges={n_nudges}, features={n_feature_types}")
    
    model = PregnancyAppGNN_PyG(
        args=args,
        n_users=n_users,
        n_segments=n_segments,
        n_nudges=n_nudges,
        n_feature_types=n_feature_types
    ).to(device)
    
    # Initialize optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # Training loop
    logger.info("Starting training...")
    best_val_loss = float('inf')
    patience = 10
    patience_counter = 0
    
    for epoch in range(args.n_epochs):
        # Train
        train_loss = train_epoch(model, optimizer, train_loader, device, args)
        
        # Validate
        val_loss = evaluate(model, val_loader, device)
        
        # Log progress
        logger.info(f"Epoch {epoch+1}/{args.n_epochs}")
        logger.info(f"Train Loss: {train_loss:.4f}")
        logger.info(f"Val Loss: {val_loss:.4f}")
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), 'best_model_pyg.pt')
            logger.info("Saved new best model")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info("Early stopping triggered")
                break
    
    logger.info("Training completed!")

if __name__ == '__main__':
    main() 