import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder

class PregnancyDataPreprocessor:
    def __init__(self):
        # Initialize scalers and encoders
        self.numerical_scaler = StandardScaler()
        self.categorical_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        
        # Define feature groups
        self.engagement_features = [
            'session_start_count',
            'watch_now_count',
            'community_opened_count',
            'courses_screen_view_count',
            'health_tracking_submission_count'
        ]
        
        self.health_features = [
            'selected_height',
            'selected_weight',
            'latest_week',
            'conditions'
        ]
        
        self.categorical_features = [
            'city',
            'selected_occupation',
            'selected_reasons',
            'conditions'
        ]

    def clean_data(self, df):
        """
        Basic cleaning of the dataframe
        """
        df_clean = df.copy()
        
        # 1. Handle missing values
        # For numerical columns, fill with median
        numerical_cols = df_clean.select_dtypes(include=['float64', 'int64']).columns
        for col in numerical_cols:
            df_clean[col] = df_clean[col].fillna(df_clean[col].median())
            
        # For categorical columns, fill with mode
        categorical_cols = df_clean.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            df_clean[col] = df_clean[col].fillna(df_clean[col].mode()[0])
            
        # 2. Handle outliers in health metrics using IQR method
        for col in ['selected_height', 'selected_weight']:
            Q1 = df_clean[col].quantile(0.25)
            Q3 = df_clean[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            df_clean[col] = df_clean[col].clip(lower_bound, upper_bound)
            
        return df_clean

    def process_engagement_features(self, df):
        """
        Process engagement-related features
        """
        df_engagement = df[self.engagement_features].copy()
        
        # Scale engagement features
        df_engagement_scaled = pd.DataFrame(
            self.numerical_scaler.fit_transform(df_engagement),
            columns=df_engagement.columns
        )
        
        return df_engagement_scaled

    def process_health_features(self, df):
        """
        Process health-related features
        """
        df_health = df[self.health_features].copy()
        
        # Calculate BMI
        df_health['bmi'] = df_health['selected_weight'] / ((df_health['selected_height']/100) ** 2)
        
        # Create trimester feature
        df_health['trimester'] = pd.cut(
            df_health['latest_week'],
            bins=[0, 13, 26, 42],
            labels=['First', 'Second', 'Third']
        )
        
        return df_health

    def fit_transform(self, df):
        """
        Main method to preprocess the data
        """
        print("Starting data preprocessing...")
        
        # 1. Clean the data
        print("Cleaning data...")
        df_cleaned = self.clean_data(df)
        
        # 2. Process engagement features
        print("Processing engagement features...")
        df_engagement = self.process_engagement_features(df_cleaned)
        
        # 3. Process health features
        print("Processing health features...")
        df_health = self.process_health_features(df_cleaned)
        
        print("Preprocessing complete!")
        
        return df_cleaned, df_engagement, df_health

# Example usage
if __name__ == "__main__":
    # Load the synthetic data
    df = pd.read_csv('synthetic_pregnancy_app_data.csv')
    
    # Initialize and run preprocessor
    preprocessor = PregnancyDataPreprocessor()
    df_cleaned, df_engagement, df_health = preprocessor.fit_transform(df)
    
    # Print sample results
    print("\nSample of processed engagement features:")
    print(df_engagement.head())
    print("\nSample of processed health features:")
    print(df_health.head())