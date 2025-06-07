import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data
import networkx as nx

class GraphConstructor:
    def __init__(self):
        # Define node types
        self.node_types = {
            'user': 0,
            'segment': 1,
            'nudge': 2
        }
        
        # Define edge types
        self.edge_types = {
            'belongs_to_segment': 0,
            'receives_nudge': 1
        }
        
        # Define segments (as discussed earlier)
        self.segments = [
            'SEGMENT_0_INACTIVE_WOMAN',
            'SEGMENT_1_ACTIVE_WOMAN',
            'SEGMENT_2_HEALTHY_WOMAN',
            'SEGMENT_3_WEAK_WOMAN',
            'SEGMENT_4_LOW_CONTENT_CONSUMER',
            'SEGMENT_5_HIGH_CONTENT_CONSUMER'
        ]
        
        # Define nudge types
        self.nudge_types = [
            'NUDGE_0_REMINDER_AND_PROMPTS',
            'NUDGE_1_PROGRESS_FEEDBACK',
            'NUDGE_2_PERSONALIZED_TIPS',
            'NUDGE_3_GOAL_SETTINGS',
            'NUDGE_4_SOCIAL_CUES'
        ]

    def assign_user_segments(self, df_engagement, df_health):
        """
        Assign users to segments based on their engagement and health features
        """
        segment_assignments = []
        
        for idx, row in df_engagement.iterrows():
            user_segments = []
            
            # Activity-based segmentation
            if row['session_start_count'] > 0.5:  # Using scaled values
                user_segments.append('SEGMENT_1_ACTIVE_WOMAN')
            else:
                user_segments.append('SEGMENT_0_INACTIVE_WOMAN')
                
            # Content consumption segmentation
            if row['watch_now_count'] > 0.5:
                user_segments.append('SEGMENT_5_HIGH_CONTENT_CONSUMER')
            else:
                user_segments.append('SEGMENT_4_LOW_CONTENT_CONSUMER')
            
            # Health-based segmentation
            health_row = df_health.iloc[idx]
            if health_row['bmi'] >= 25:
                user_segments.append('SEGMENT_3_WEAK_WOMAN')
            else:
                user_segments.append('SEGMENT_2_HEALTHY_WOMAN')
                
            segment_assignments.append(user_segments)
            
        return segment_assignments

    def create_graph(self, df_cleaned, df_engagement, df_health):
        """
        Create a heterogeneous graph from the preprocessed data
        """
        print("Creating graph structure...")
        
        num_users = len(df_cleaned)
        num_segments = len(self.segments)
        num_nudges = len(self.nudge_types)
        
        # Calculate feature dimensions
        user_feature_dim = df_engagement.shape[1] + 1  # engagement features + bmi
        
        # 1. Create node features
        print("Creating node features...")
        
        # User features: combine engagement and health features
        user_features = torch.cat([
            torch.tensor(df_engagement.values, dtype=torch.float),
            torch.tensor(df_health[['bmi']].values, dtype=torch.float)
        ], dim=1)
        
        # Pad segment and nudge features to match user feature dimensions
        segment_features = torch.zeros((num_segments, user_feature_dim))
        segment_features[:, :num_segments] = torch.eye(num_segments)  # One-hot encoding in first positions
        
        nudge_features = torch.zeros((num_nudges, user_feature_dim))
        nudge_features[:, :num_nudges] = torch.eye(num_nudges)  # One-hot encoding in first positions
        
        # Combine all node features
        x = torch.cat([user_features, segment_features, nudge_features], dim=0)
        
        # 2. Create edges
        print("Creating edges...")
        
        edge_index_list = []
        edge_type_list = []
        
        # User-Segment edges
        segment_assignments = self.assign_user_segments(df_engagement, df_health)
        for user_idx, user_segments in enumerate(segment_assignments):
            for segment in user_segments:
                segment_idx = self.segments.index(segment) + num_users  # offset by num_users
                edge_index_list.append([user_idx, segment_idx])
                edge_type_list.append(self.edge_types['belongs_to_segment'])
        
        # Segment-Nudge edges (for now, connect each segment to all nudges)
        segment_base_idx = num_users
        nudge_base_idx = num_users + num_segments
        for segment_idx in range(num_segments):
            for nudge_idx in range(num_nudges):
                edge_index_list.append([
                    segment_base_idx + segment_idx,
                    nudge_base_idx + nudge_idx
                ])
                edge_type_list.append(self.edge_types['receives_nudge'])
        
        edge_index = torch.tensor(edge_index_list, dtype=torch.long).t()
        edge_type = torch.tensor(edge_type_list, dtype=torch.long)
        
        # 3. Create PyG Data object
        data = Data(
            x=x,
            edge_index=edge_index,
            edge_type=edge_type,
            num_nodes=len(x)
        )
        
        print("Graph construction complete!")
        print(f"Number of nodes: {data.num_nodes}")
        print(f"Number of edges: {data.num_edges}")
        
        return data

# Example usage
if __name__ == "__main__":
    # Load preprocessed data
    df = pd.read_csv('synthetic_pregnancy_app_data.csv')
    
    # First preprocess the data
    from data_preprocessing import PregnancyDataPreprocessor
    preprocessor = PregnancyDataPreprocessor()
    df_cleaned, df_engagement, df_health = preprocessor.fit_transform(df)
    
    # Create graph
    graph_constructor = GraphConstructor()
    graph_data = graph_constructor.create_graph(df_cleaned, df_engagement, df_health)
    
    # Print graph statistics
    print("\nGraph Statistics:")
    print(f"Node feature dimensions: {graph_data.x.size()}")
    print(f"Edge index dimensions: {graph_data.edge_index.size()}")
    print(f"Number of edge types: {len(torch.unique(graph_data.edge_type))}")