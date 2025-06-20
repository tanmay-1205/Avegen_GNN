import torch
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
from torch_geometric.data import Data

from gnn_model_pyg import PregnancyAppGNN_PyG
from train_pyg import parse_args, create_pyg_data_sophisticated

class EngagementPredictor:
    def __init__(self, model_path: str = 'best_model_pyg.pt'):
        """
        Initialize the predictor with a trained model
        
        Args:
            model_path: Path to the saved model weights
        """
        # Get model arguments
        self.args = parse_args()
        
        # Initialize model
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Load preprocessed data to get feature dimensions
        df = pd.read_csv(self.args.data_path)
        feature_columns = [col for col in df.columns if col not in [
            'user_id_hashed', 'user_id', 'city', 'selected_occupation', 'selected_reasons', 
            'conditions', 'last_session_date', 'last_health_check_date', 'device_type', 
            'app_version', 'city_tier', 'age_group', 'bmi_category', 'trimester', 'risk_level',
            'pregnancy_experience'
        ]]
        
        # Create a sample graph to get correct dimensions
        sample_data = create_pyg_data_sophisticated(df.head(10), feature_columns)
        
        # Calculate correct model parameters
        n_users = (sample_data.node_type == 0).sum().item() * len(df) // 10  # Scale up
        n_segments = (sample_data.node_type == 1).sum().item()
        n_nudges = (sample_data.node_type == 2).sum().item()
        n_feature_types = sample_data.x.size(1)
        
        # Initialize model with correct parameters
        self.model = PregnancyAppGNN_PyG(
            args=self.args,
            n_users=len(df),  # Use actual number of users
            n_segments=n_segments,
            n_nudges=n_nudges,
            n_feature_types=n_feature_types
        )
        
        # Load trained weights
        self.model.load_state_dict(torch.load(model_path))
        self.model.to(self.device)
        self.model.eval()
        
        # Store feature columns
        self.feature_columns = feature_columns
        
        # Define segment descriptions (matching GraphConstructor)
        self.segments = [
            'SEGMENT_0_INACTIVE_WOMAN',
            'SEGMENT_1_ACTIVE_WOMAN', 
            'SEGMENT_2_HEALTHY_WOMAN',
            'SEGMENT_3_WEAK_WOMAN',
            'SEGMENT_4_LOW_CONTENT_CONSUMER',
            'SEGMENT_5_HIGH_CONTENT_CONSUMER'
        ]
        
        # Define nudge categories
        self.nudges = [
            'NUDGE_0_REMINDER_AND_PROMPTS',
            'NUDGE_1_PROGRESS_FEEDBACK',
            'NUDGE_2_PERSONALIZED_TIPS',
            'NUDGE_3_GOAL_SETTINGS',
            'NUDGE_4_SOCIAL_CUES'
        ]
    
    def preprocess_user_data(self, user_data: Dict) -> Data:
        """
        Preprocess single user data
        
        Args:
            user_data: Dictionary containing user information
            
        Returns:
            PyG Data object
        """
        # Convert single user data to DataFrame
        df = pd.DataFrame([user_data])
        
        # Create PyG data using sophisticated graph construction
        data = create_pyg_data_sophisticated(df, self.feature_columns)
        
        return data
    
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
                    'probability': float(prob),
                    'description': self.get_segment_description(self.segments[idx])
                })
                
        return selected_segments
    
    def get_top_nudges(self, 
                      nudge_scores: torch.Tensor, 
                      k: int = 3) -> List[Dict]:
        """
        Get top-k nudges with their scores
        """
        # Apply softmax to get probabilities
        nudge_probs = torch.softmax(nudge_scores, dim=0)
        
        # Get top-k indices
        top_k_values, top_k_indices = torch.topk(nudge_probs, k)
        
        # Create recommendations list
        recommendations = []
        for score, idx in zip(top_k_values, top_k_indices):
            recommendations.append({
                'nudge_type': self.nudges[idx],
                'confidence': float(score),
                'description': self.get_nudge_description(self.nudges[idx])
            })
            
        return recommendations
    
    def get_segment_description(self, segment: str) -> str:
        """
        Get detailed description for each segment
        """
        descriptions = {
            'SEGMENT_0_INACTIVE_WOMAN': 
                "Users with low engagement and activity. May need motivation and engagement strategies.",
            'SEGMENT_1_ACTIVE_WOMAN': 
                "Users with high engagement and regular activity. Can benefit from advanced features and community engagement.",
            'SEGMENT_2_HEALTHY_WOMAN': 
                "Users with good health metrics. Primarily interested in maintaining their health status.",
            'SEGMENT_3_WEAK_WOMAN': 
                "Users needing health attention. Require health-focused content and medical guidance.",
            'SEGMENT_4_LOW_CONTENT_CONSUMER': 
                "Users with low content consumption. May benefit from content recommendations and engagement strategies.",
            'SEGMENT_5_HIGH_CONTENT_CONSUMER': 
                "Users with high content consumption. Actively consume educational content, articles, and videos."
        }
        return descriptions.get(segment, "General user segment")
    
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
        data = self.preprocess_user_data(user_data)
        
        # Move to device
        data = data.to(self.device)
        
        # Get predictions
        with torch.no_grad():
            segment_preds, nudge_preds = self.model(data, mode='predict')
            
            # Get predictions for the first user (since we're predicting for one user)
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
    # Example user data
    example_user = {
        'session_start_count': 15,
        'watch_now_count': 8,
        'community_opened_count': 3,
        'health_tracking_submission_count': 12,
        'content_completion_count': 6,
        'selected_height': 165,
        'selected_weight': 60,
        'latest_week': 20,
        'average_session_duration': 25.5,
        'notification_response_count': 10,
        'previous_pregnancies': 0,
        'bmi': 22.0,
        'total_engagement_score': 44,
        'content_interaction_ratio': 0.4,
        'community_engagement_ratio': 0.2,
        'health_tracking_ratio': 0.8,
        'activity_consistency': 0.7,
        'engagement_trend': 0.6,
        'health_engagement_score': 22,
        'days_since_last_session': 1,
        'days_since_last_health_check': 2
    }
    
    # Initialize predictor
    predictor = EngagementPredictor()
    
    # Make predictions
    predictions = predictor.predict(example_user)
    
    # Display results
    print("User Segmentation Results:")
    print("=" * 50)
    
    print("\nIdentified Segments:")
    for i, segment in enumerate(predictions['user_segments'], 1):
        print(f"\n{i}. {segment['segment']}")
        print(f"   Probability: {segment['probability']:.3f}")
        print(f"   Description: {segment['description']}")
    
    print("\nRecommended Nudges:")
    for i, nudge in enumerate(predictions['recommended_nudges'], 1):
        print(f"\n{i}. {nudge['nudge_type']}")
        print(f"   Confidence: {nudge['confidence']:.3f}")
        print(f"   Description: {nudge['description']}") 