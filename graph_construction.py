import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data
import networkx as nx
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

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
            'segment_to_user': 1,  # Bidirectional edge
            'receives_nudge': 2,
            'nudge_to_segment': 3  # Bidirectional edge
        }
        
        # Define segments with their criteria
        self.segments = {
            'SEGMENT_0_INACTIVE_WOMAN': {
                'description': 'Users with low engagement and activity',
                'criteria': {
                    'session_start_count': (-float('inf'), -0.5),
                    'total_engagement_score': (-float('inf'), -0.3)
                }
            },
            'SEGMENT_1_ACTIVE_WOMAN': {
                'description': 'Users with high engagement and regular activity',
                'criteria': {
                    'session_start_count': (0.5, float('inf')),
                    'total_engagement_score': (0.3, float('inf'))
                }
            },
            'SEGMENT_2_HEALTHY_WOMAN': {
                'description': 'Users with good health metrics',
                'criteria': {
                    'bmi': (-1, 1),  # Normal BMI range after standardization
                    'health_tracking_ratio': (0.3, float('inf'))
                }
            },
            'SEGMENT_3_WEAK_WOMAN': {
                'description': 'Users needing health attention',
                'criteria': {
                    'bmi': (1, float('inf')),  # High BMI
                    'health_tracking_ratio': (-float('inf'), 0.3)
                }
            },
            'SEGMENT_4_LOW_CONTENT_CONSUMER': {
                'description': 'Users with low content consumption',
                'criteria': {
                    'content_interaction_ratio': (-float('inf'), -0.3),
                    'watch_now_count': (-float('inf'), -0.3)
                }
            },
            'SEGMENT_5_HIGH_CONTENT_CONSUMER': {
                'description': 'Users with high content consumption',
                'criteria': {
                    'content_interaction_ratio': (0.3, float('inf')),
                    'watch_now_count': (0.3, float('inf'))
                }
            }
        }
        
        # Define nudge types with their target segments
        self.nudge_types = {
            'NUDGE_0_REMINDER_AND_PROMPTS': {
                'description': 'Regular reminders and activity prompts',
                'target_segments': ['SEGMENT_0_INACTIVE_WOMAN'],
                'weight': 1.0
            },
            'NUDGE_1_PROGRESS_FEEDBACK': {
                'description': 'Positive reinforcement of progress',
                'target_segments': ['SEGMENT_1_ACTIVE_WOMAN', 'SEGMENT_2_HEALTHY_WOMAN'],
                'weight': 0.8
            },
            'NUDGE_2_PERSONALIZED_TIPS': {
                'description': 'Health and wellness tips',
                'target_segments': ['SEGMENT_3_WEAK_WOMAN'],
                'weight': 0.9
            },
            'NUDGE_3_GOAL_SETTINGS': {
                'description': 'Personalized goal recommendations',
                'target_segments': ['SEGMENT_1_ACTIVE_WOMAN', 'SEGMENT_4_LOW_CONTENT_CONSUMER'],
                'weight': 0.7
            },
            'NUDGE_4_SOCIAL_CUES': {
                'description': 'Community engagement prompts',
                'target_segments': ['SEGMENT_5_HIGH_CONTENT_CONSUMER'],
                'weight': 0.6
            }
        }
        
        self.scaler = MinMaxScaler()

    def assign_user_segments(self, df_engagement, df_health):
        """
        Assign users to segments based on sophisticated criteria
        Returns both segment assignments and confidence scores
        """
        segment_assignments = []
        confidence_scores = []
        
        # Combine relevant features
        features_df = pd.concat([
            df_engagement[['session_start_count', 'watch_now_count', 'total_engagement_score', 
                         'content_interaction_ratio', 'health_tracking_ratio']],
            df_health[['bmi']]
        ], axis=1)
        
        for idx, row in features_df.iterrows():
            user_segments = []
            segment_scores = []
            
            for segment_name, segment_info in self.segments.items():
                score = 0
                criteria_met = 0
                
                for feature, (min_val, max_val) in segment_info['criteria'].items():
                    if feature in row and min_val <= row[feature] <= max_val:
                        criteria_met += 1
                        score += abs(row[feature])  # Use feature value as confidence
                
                if criteria_met == len(segment_info['criteria']):
                    user_segments.append(segment_name)
                    segment_scores.append(score / len(segment_info['criteria']))
            
            # Ensure at least one segment assignment
            if not user_segments:
                user_segments.append('SEGMENT_0_INACTIVE_WOMAN')
                segment_scores.append(0.5)
            
            segment_assignments.append(user_segments)
            confidence_scores.append(segment_scores)
        
        return segment_assignments, confidence_scores

    def select_user_features(self, df_engagement, df_health):
        """
        Select and combine relevant features for user nodes
        """
        key_features = [
            'session_start_count',
            'watch_now_count',
            'community_opened_count',
            'health_tracking_submission_count',
            'content_completion_count',
            'total_engagement_score',
            'content_interaction_ratio',
            'health_tracking_ratio',
            'activity_consistency',
            'engagement_trend'
        ]
        
        health_features = ['bmi']
        
        # Combine features
        user_features = pd.concat([
            df_engagement[key_features],
            df_health[health_features]
        ], axis=1)
        
        return torch.tensor(user_features.values, dtype=torch.float)

    def calculate_edge_weight(self, source_type, target_type, source_features, target_features, base_weight=1.0):
        """
        Calculate edge weights based on feature similarity and node types
        """
        if source_type == 'user' and target_type == 'segment':
            # Calculate feature similarity for user-segment edges
            similarity = torch.cosine_similarity(source_features.unsqueeze(0), target_features.unsqueeze(0))
            return float(similarity) * base_weight
        elif source_type == 'segment' and target_type == 'nudge':
            # Use predefined weights for segment-nudge edges
            return base_weight
        else:
            return base_weight

    def create_graph(self, df_cleaned, df_engagement, df_health):
        """
        Create a heterogeneous graph with improved structure and edge weights
        """
        print("Creating graph structure...")
        
        num_users = len(df_cleaned)
        num_segments = len(self.segments)
        num_nudges = len(self.nudge_types)
        
        # 1. Create node features
        print("Creating node features...")
        
        # User features with improved selection
        user_features = self.select_user_features(df_engagement, df_health)
        feature_dim = user_features.size(1)
        
        # Segment features
        segment_features = torch.zeros((num_segments, feature_dim))
        for i, segment in enumerate(self.segments.keys()):
            segment_features[i, :num_segments] = torch.eye(num_segments)[i]
        
        # Nudge features
        nudge_features = torch.zeros((num_nudges, feature_dim))
        for i, nudge in enumerate(self.nudge_types.keys()):
            nudge_features[i, :num_nudges] = torch.eye(num_nudges)[i]
        
        # Combine all node features
        x = torch.cat([user_features, segment_features, nudge_features], dim=0)
        
        # Create node type indicators
        node_type = torch.zeros(len(x), dtype=torch.long)
        node_type[num_users:num_users + num_segments] = 1  # Segment nodes
        node_type[num_users + num_segments:] = 2  # Nudge nodes
        
        # 2. Create edges with sophisticated weights
        print("Creating edges...")
        
        edge_index_list = []
        edge_type_list = []
        edge_weight_list = []
        
        # User-Segment edges (bidirectional)
        segment_assignments, confidence_scores = self.assign_user_segments(df_engagement, df_health)
        for user_idx, (user_segments, scores) in enumerate(zip(segment_assignments, confidence_scores)):
            current_user_features = user_features[user_idx]
            for segment, score in zip(user_segments, scores):
                segment_idx = list(self.segments.keys()).index(segment) + num_users
                segment_features_slice = segment_features[segment_idx - num_users]
                
                # Calculate sophisticated edge weight
                edge_weight = self.calculate_edge_weight(
                    'user', 'segment',
                    current_user_features,
                    segment_features_slice,
                    base_weight=score
                )
                
                # User to segment edge
                edge_index_list.append([user_idx, segment_idx])
                edge_type_list.append(self.edge_types['belongs_to_segment'])
                edge_weight_list.append(edge_weight)
                
                # Segment to user edge
                edge_index_list.append([segment_idx, user_idx])
                edge_type_list.append(self.edge_types['segment_to_user'])
                edge_weight_list.append(edge_weight)
        
        # Segment-Nudge edges (bidirectional with weights)
        segment_base_idx = num_users
        nudge_base_idx = num_users + num_segments
        
        for segment_name, segment_idx in zip(self.segments.keys(), range(num_segments)):
            for nudge_name, nudge_info in self.nudge_types.items():
                nudge_idx = list(self.nudge_types.keys()).index(nudge_name)
                
                if segment_name in nudge_info['target_segments']:
                    # Segment to nudge edge
                    edge_index_list.append([
                        segment_base_idx + segment_idx,
                        nudge_base_idx + nudge_idx
                    ])
                    edge_type_list.append(self.edge_types['receives_nudge'])
                    edge_weight_list.append(float(nudge_info['weight']))
                    
                    # Nudge to segment edge
                    edge_index_list.append([
                        nudge_base_idx + nudge_idx,
                        segment_base_idx + segment_idx
                    ])
                    edge_type_list.append(self.edge_types['nudge_to_segment'])
                    edge_weight_list.append(float(nudge_info['weight']))
        
        edge_index = torch.tensor(edge_index_list, dtype=torch.long).t()
        edge_type = torch.tensor(edge_type_list, dtype=torch.long)
        edge_weight = torch.tensor(edge_weight_list, dtype=torch.float)
        
        # 3. Create PyG Data object
        data = Data(
            x=x,
            edge_index=edge_index,
            edge_type=edge_type,
            edge_weight=edge_weight,
            node_type=node_type,
            num_nodes=len(x)
        )
        
        print("Graph construction complete!")
        print(f"Number of nodes: {data.num_nodes}")
        print(f"Number of edges: {data.num_edges}")
        
        return data

    def visualize_graph(self, data, output_path='graph_visualization.png'):
        """
        Create a comprehensive visualization of the graph structure
        """
        # Convert to networkx for visualization
        G = nx.Graph()
        
        # Add nodes with different colors for different types
        colors = []
        node_labels = {}
        
        num_users = (data.node_type == 0).sum().item()
        num_segments = (data.node_type == 1).sum().item()
        num_nudges = (data.node_type == 2).sum().item()
        
        # Add user nodes (blue)
        for i in range(num_users):
            G.add_node(i)
            colors.append('lightblue')
            node_labels[i] = f'U{i}'
        
        # Add segment nodes (green)
        for i in range(num_segments):
            node_idx = num_users + i
            G.add_node(node_idx)
            colors.append('lightgreen')
            node_labels[node_idx] = f'S{i}'
        
        # Add nudge nodes (red)
        for i in range(num_nudges):
            node_idx = num_users + num_segments + i
            G.add_node(node_idx)
            colors.append('salmon')
            node_labels[node_idx] = f'N{i}'
        
        # Add edges with weights
        edge_index = data.edge_index.t().numpy()
        edge_weights = data.edge_weight.numpy() if hasattr(data, 'edge_weight') else None
        
        for i in range(edge_index.shape[0]):
            source, target = edge_index[i]
            weight = edge_weights[i] if edge_weights is not None else 1.0
            G.add_edge(source, target, weight=weight)
        
        # Create the visualization
        plt.figure(figsize=(15, 10))
        pos = nx.spring_layout(G, k=1, iterations=50)
        
        # Draw nodes
        nx.draw_networkx_nodes(G, pos, node_color=colors, node_size=500)
        nx.draw_networkx_labels(G, pos, node_labels)
        
        # Draw edges with varying width based on weight
        edge_weights = [G[u][v]['weight'] for u, v in G.edges()]
        nx.draw_networkx_edges(G, pos, width=[w * 2 for w in edge_weights], alpha=0.5)
        
        plt.title('Pregnancy Healthcare Graph Structure')
        plt.axis('off')
        
        # Add legend
        legend_elements = [
            plt.Line2D([0], [0], marker='o', color='w', label='Users',
                      markerfacecolor='lightblue', markersize=10),
            plt.Line2D([0], [0], marker='o', color='w', label='Segments',
                      markerfacecolor='lightgreen', markersize=10),
            plt.Line2D([0], [0], marker='o', color='w', label='Nudges',
                      markerfacecolor='salmon', markersize=10)
        ]
        plt.legend(handles=legend_elements, loc='upper right')
        
        # Save the visualization
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
        plt.close()
        print(f"Graph visualization saved to {output_path}")

# Example usage
if __name__ == "__main__":
    try:
        # Load preprocessed data
        print("Loading preprocessed data...")
        df_cleaned = pd.read_csv('./preprocessed_data/preprocessed_data.csv')
        
        # Split features for graph construction
        engagement_features = [
            'session_start_count', 'watch_now_count', 'community_opened_count',
            'health_tracking_submission_count', 'content_completion_count',
            'total_engagement_score', 'content_interaction_ratio', 
            'health_tracking_ratio', 'activity_consistency', 'engagement_trend'
        ]
        
        health_features = ['bmi']
        
        # Select only existing columns
        available_engagement = [col for col in engagement_features if col in df_cleaned.columns]
        available_health = [col for col in health_features if col in df_cleaned.columns]
        
        df_engagement = df_cleaned[available_engagement]
        df_health = df_cleaned[available_health]
        
        # Create graph
        print("\nConstructing graph...")
        graph_constructor = GraphConstructor()
        graph_data = graph_constructor.create_graph(df_cleaned, df_engagement, df_health)
        
        # Print graph statistics
        print("\nGraph Statistics:")
        print(f"Node feature dimensions: {graph_data.x.size()}")
        print(f"Edge index dimensions: {graph_data.edge_index.size()}")
        print(f"Number of edge types: {len(torch.unique(graph_data.edge_type))}")
        print(f"Number of node types: {len(torch.unique(graph_data.node_type))}")
        
        # Visualize graph
        print("\nCreating graph visualization...")
        graph_constructor.visualize_graph(graph_data)
        
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        raise