import torch
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple

from data_preprocessing import PregnancyDataPreprocessor
from graph_construction import GraphConstructor
from gnn_model import PregnancyGNN

class EngagementPredictor:
    def __init__(self, model_path: str = 'best_model.pt'):
        """
        Initialize the predictor with a trained model
        
        Args:
            model_path: Path to the saved model weights
        """
        # Load preprocessor and graph constructor
        self.preprocessor = PregnancyDataPreprocessor()
        self.graph_constructor = GraphConstructor()
        
        # Initialize model
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.input_dim = 6  # Same as training
        self.model = PregnancyGNN(input_dim=self.input_dim)
        
        # Load trained weights
        self.model.load_state_dict(torch.load(model_path))
        self.model.to(self.device)
        self.model.eval()
        
        # Define segment and nudge mappings
        self.segments = [
            'SEGMENT_0_INACTIVE_WOMAN',
            'SEGMENT_1_ACTIVE_WOMAN',
            'SEGMENT_2_HEALTHY_WOMAN',
            'SEGMENT_3_WEAK_WOMAN',
            'SEGMENT_4_LOW_CONTENT_CONSUMER',
            'SEGMENT_5_HIGH_CONTENT_CONSUMER'
        ]
        
        self.nudges = [
            'NUDGE_0_REMINDER_AND_PROMPTS',
            'NUDGE_1_PROGRESS_FEEDBACK',
            'NUDGE_2_PERSONALIZED_TIPS',
            'NUDGE_3_GOAL_SETTINGS',
            'NUDGE_4_SOCIAL_CUES'
        ]
        
    def preprocess_user_data(self, user_data: Dict) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Preprocess single user data
        
        Args:
            user_data: Dictionary containing user information
            
        Returns:
            Processed features and graph structure
        """
        # Convert single user data to DataFrame
        df = pd.DataFrame([user_data])
        
        # Preprocess
        df_cleaned, df_engagement, df_health = self.preprocessor.fit_transform(df)
        
        # Create graph structure
        graph_data = self.graph_constructor.create_graph(df_cleaned, df_engagement, df_health)
        
        return graph_data.x, graph_data.edge_index
    
    def get_top_nudges(self, 
                      nudge_scores: torch.Tensor, 
                      k: int = 3) -> List[Dict]:
        """
        Get top-k nudges with their scores
        """
        # Get top-k indices
        top_k_values, top_k_indices = torch.topk(nudge_scores, k)
        
        # Create recommendations list
        recommendations = []
        for score, idx in zip(top_k_values, top_k_indices):
            recommendations.append({
                'nudge_type': self.nudges[idx],
                'confidence': float(score),
                'description': self.get_nudge_description(self.nudges[idx])
            })
            
        return recommendations
    
    def get_segments(self, 
                    segment_scores: torch.Tensor, 
                    threshold: float = 0.3) -> List[Dict]:
        """
        Get segments above threshold
        """
        # Apply softmax to get probabilities
        segment_probs = torch.softmax(segment_scores, dim=0)
        
        # Get segments above threshold
        selected_segments = []
        for idx, prob in enumerate(segment_probs):
            if prob > threshold:
                selected_segments.append({
                    'segment': self.segments[idx],
                    'probability': float(prob)
                })
                
        return selected_segments
    
    def get_nudge_description(self, nudge_type: str) -> str:
        """
        Get detailed description for each nudge type
        """
        descriptions = {
            'NUDGE_0_REMINDER_AND_PROMPTS': 
                "Gentle reminder to log your daily health parameters and stay consistent with your tracking.",
            'NUDGE_1_PROGRESS_FEEDBACK': 
                "Personalized feedback on your health journey and progress so far.",
            'NUDGE_2_PERSONALIZED_TIPS': 
                "Custom health tips based on your current trimester and health status.",
            'NUDGE_3_GOAL_SETTINGS': 
                "Suggestions for setting achievable health and wellness goals.",
            'NUDGE_4_SOCIAL_CUES': 
                "Connect with other mothers in similar stages of their pregnancy journey."
        }
        return descriptions.get(nudge_type, "General engagement recommendation")
    
    def predict(self, user_data: Dict) -> Dict:
        """
        Make predictions for a single user
        
        Args:
            user_data: Dictionary containing user information
            
        Returns:
            Dictionary containing segments and recommended nudges
        """
        # Preprocess user data
        x, edge_index = self.preprocess_user_data(user_data)
        
        # Move to device
        x = x.to(self.device)
        edge_index = edge_index.to(self.device)
        
        # Get predictions
        with torch.no_grad():
            segment_preds, nudge_preds = self.model(x, edge_index)
            
            # Get predictions for the user node (first node)
            user_segment_preds = segment_preds[0]
            user_nudge_preds = nudge_preds[0]
        
        # Get segments and nudges
        segments = self.get_segments(user_segment_preds)
        recommended_nudges = self.get_top_nudges(user_nudge_preds)
        
        return {
            'user_segments': segments,
            'recommended_nudges': recommended_nudges
        }

# Example usage
if __name__ == "__main__":
    # Initialize predictor
    predictor = EngagementPredictor()
    
    # Example user data
    example_user = {
        'user_id_hashed': 'USER_TEST',
        'city': 'Mumbai',
        'session_start_count': 5,
        'watch_now_count': 3,
        'community_opened_count': 2,
        'courses_screen_view_count': 4,
        'health_tracking_submission_count': 3,
        'selected_height': 165.0,
        'selected_weight': 68.0,
        'selected_occupation': 'Working Professional',
        'selected_reasons': 'First Pregnancy',
        'latest_week': 16,
        'conditions': 'None'
    }
    
    # Make predictions
    print("\nMaking predictions for example user...")
    predictions = predictor.predict(example_user)
    
    # Print results
    print("\nPredicted Segments:")
    for segment in predictions['user_segments']:
        print(f"- {segment['segment']}: {segment['probability']:.3f}")
    
    print("\nRecommended Nudges:")
    for i, nudge in enumerate(predictions['recommended_nudges'], 1):
        print(f"\n{i}. {nudge['nudge_type']}")
        print(f"   Confidence: {nudge['confidence']:.3f}")
        print(f"   Description: {nudge['description']}") 