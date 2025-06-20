import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
import os
from datetime import datetime, timedelta
import joblib

class DataPreprocessor:
    def __init__(self, data_path='synthetic_pregnancy_app_data.csv'):
        """
        Initialize the preprocessor with the path to the data file
        """
        self.data_path = data_path
        self.scalers = {}
        self.label_encoders = {}
        
    def load_data(self):
        """
        Load the data from CSV file
        """
        self.df = pd.read_csv(self.data_path)
        print(f"Loaded data with shape: {self.df.shape}")
        return self.df
    
    def handle_missing_values(self):
        """
        Handle missing values in the dataset
        """
        # For numerical columns, fill with median
        numerical_cols = self.df.select_dtypes(include=['float64', 'int64']).columns
        for col in numerical_cols:
            if self.df[col].isnull().sum() > 0:
                self.df[col] = self.df[col].fillna(self.df[col].median())
        
        # For categorical columns, fill with mode
        categorical_cols = self.df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if self.df[col].isnull().sum() > 0:
                self.df[col] = self.df[col].fillna(self.df[col].mode()[0])
        
        print("Missing values handled")
        return self.df
    
    def handle_outliers(self, columns=['selected_height', 'selected_weight']):
        """
        Handle outliers using IQR method
        """
        for col in columns:
            Q1 = self.df[col].quantile(0.25)
            Q3 = self.df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            # Replace outliers with bounds
            self.df[col] = np.where(self.df[col] < lower_bound, lower_bound, self.df[col])
            self.df[col] = np.where(self.df[col] > upper_bound, upper_bound, self.df[col])
        
        print("Outliers handled")
        return self.df
    
    def normalize_numerical_features(self):
        """
        Normalize numerical features using StandardScaler
        """
        numerical_cols = [
            # Engagement metrics
            'session_start_count', 'watch_now_count', 'community_opened_count',
            'hamburger_menu_nearby_hospitals_click_count', 'courses_screen_view_count',
            'self_assessment_card_button_click_count', 'course_start_tap_count',
            'health_tracking_submission_count', 'view_tracking_questions_click_home_scre_count',
            'cr_card_tap_count', 'week_content_share_app_click_count',
            'herstore_opened_count', 'herhealth_opened_count',
            
            # Health metrics
            'selected_height', 'selected_weight', 'latest_week',
            'average_session_duration', 'notification_response_count',
            'content_completion_count', 'previous_pregnancies'
        ]
        
        for col in numerical_cols:
            if col in self.df.columns:
                scaler = StandardScaler()
                self.df[col] = scaler.fit_transform(self.df[[col]])
                self.scalers[col] = scaler
        
        print("Numerical features normalized")
        return self.df
    
    def encode_categorical_features(self):
        """
        Encode categorical features using LabelEncoder
        """
        categorical_cols = [
            'city', 'city_tier', 'selected_occupation', 'selected_reasons', 
            'conditions', 'device_type', 'app_version', 'bmi_category',
            'age_group', 'trimester', 'risk_level'
        ]
        
        for col in categorical_cols:
            if col in self.df.columns:
                le = LabelEncoder()
                self.df[col] = le.fit_transform(self.df[col])
                self.label_encoders[col] = le
        
        # Handle boolean columns
        if 'health_records_uploaded' in self.df.columns:
            self.df['health_records_uploaded'] = self.df['health_records_uploaded'].astype(int)
        
        print("Categorical features encoded")
        return self.df
    
    def prepare_date_features(self):
        """
        Process date features
        """
        # Convert dob to age
        self.df['dob'] = pd.to_datetime(self.df['dob'])
        self.df['age'] = (pd.Timestamp.now() - self.df['dob']).dt.total_seconds() / (365.25 * 24 * 60 * 60)
        self.df.drop('dob', axis=1, inplace=True)
        
        # Normalize age
        scaler = StandardScaler()
        self.df['age'] = scaler.fit_transform(self.df[['age']])
        self.scalers['age'] = scaler
        
        print("Date features processed")
        return self.df
    
    def save_preprocessed_data(self, output_dir='preprocessed_data'):
        """
        Save preprocessed data and preprocessing objects using joblib
        """
        # Create output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Save preprocessed data
        self.df.to_csv(os.path.join(output_dir, 'preprocessed_data.csv'), index=False)
        
        # Save preprocessing objects using joblib
        joblib.dump(self.scalers, os.path.join(output_dir, 'scalers.joblib'))
        joblib.dump(self.label_encoders, os.path.join(output_dir, 'label_encoders.joblib'))
        
        print(f"Preprocessed data and objects saved to {output_dir}")
    
    def create_user_ids(self):
        """
        Create sequential user IDs starting from 0
        """
        self.df['user_id'] = range(len(self.df))
        print("User IDs created")
        return self.df
    
    def calculate_bmi(self):
        """
        Calculate BMI using height and weight
        Formula: BMI = weight (kg) / (height (m))²
        """
        # Convert height from cm to meters
        height_in_meters = self.df['selected_height'] / 100
        
        # Calculate BMI
        self.df['bmi'] = self.df['selected_weight'] / (height_in_meters ** 2)
        
        # Add BMI category
        self.df['bmi_category'] = pd.cut(
            self.df['bmi'],
            bins=[0, 18.5, 24.9, 29.9, float('inf')],
            labels=['Underweight', 'Normal', 'Overweight', 'Obese']
        )
        
        print("BMI calculated and categorized")
        return self.df
    
    def create_age_groups(self):
        """
        Create age groups from age
        """
        self.df['age_group'] = pd.cut(
            self.df['age'],
            bins=[-float('inf'), 24, 30, 35, 40, float('inf')],
            labels=['18-24', '25-30', '31-35', '36-40', '40+']
        )
        print("Age groups created")
        return self.df

    def calculate_engagement_metrics(self):
        """
        Calculate additional engagement metrics with safety checks
        """
        # Total engagement score
        engagement_columns = [
            'session_start_count', 'watch_now_count', 'community_opened_count',
            'courses_screen_view_count', 'health_tracking_submission_count'
        ]
        self.df['total_engagement_score'] = self.df[engagement_columns].sum(axis=1)
        
        # Add small epsilon to prevent division by zero
        epsilon = 1e-10
        safe_session_count = self.df['session_start_count'] + epsilon
        
        # Content interaction ratio
        content_columns = ['watch_now_count', 'courses_screen_view_count']
        self.df['content_interaction_ratio'] = (
            self.df[content_columns].sum(axis=1) / safe_session_count
        )
        
        # Community engagement ratio
        community_columns = ['community_opened_count', 'cr_card_tap_count']
        self.df['community_engagement_ratio'] = (
            self.df[community_columns].sum(axis=1) / safe_session_count
        )
        
        # Health tracking engagement
        health_columns = ['health_tracking_submission_count', 'view_tracking_questions_click_home_scre_count']
        self.df['health_tracking_ratio'] = (
            self.df[health_columns].sum(axis=1) / safe_session_count
        )
        
        print("Engagement metrics calculated")
        return self.df

    def calculate_trimester(self):
        """
        Calculate trimester based on pregnancy week
        """
        self.df['trimester'] = pd.cut(
            self.df['latest_week'],
            bins=[0, 13, 26, 42],
            labels=['First', 'Second', 'Third']
        )
        print("Trimester calculated")
        return self.df

    def calculate_risk_level(self):
        """
        Calculate pregnancy risk level based on various factors
        """
        # Initialize risk score
        risk_score = pd.Series(0, index=self.df.index)
        
        # Age-based risk
        risk_score += np.where(self.df['age'] > 35, 1, 0)
        risk_score += np.where(self.df['age'] > 40, 1, 0)
        
        # BMI-based risk
        risk_score += np.where(self.df['bmi'] < 18.5, 1, 0)  # Underweight
        risk_score += np.where(self.df['bmi'] > 30, 1, 0)    # Obese
        
        # Medical conditions risk
        risk_score += np.where(self.df['conditions'] != 'None', 1, 0)
        
        # Categorize risk level
        self.df['risk_level'] = pd.cut(
            risk_score,
            bins=[-float('inf'), 0, 1, float('inf')],
            labels=['Low', 'Medium', 'High']
        )
        
        print("Risk level calculated")
        return self.df

    def calculate_activity_patterns(self):
        """
        Calculate user activity patterns
        """
        # Activity consistency (ratio of days with activity)
        total_possible_days = 30  # Assuming 30-day period
        active_days = self.df['session_start_count'].clip(upper=total_possible_days)
        self.df['activity_consistency'] = active_days / total_possible_days
        
        # Engagement trend (ratio of recent to total engagement)
        recent_engagement = self.df[['watch_now_count', 'community_opened_count']].sum(axis=1)
        total_engagement = self.df['total_engagement_score']
        self.df['engagement_trend'] = (recent_engagement / total_engagement).fillna(0)
        
        print("Activity patterns calculated")
        return self.df

    def process_temporal_features(self):
        """
        Process temporal features and calculate time-based metrics
        """
        current_date = datetime.now()
        
        # Process last session date
        if 'last_session_date' in self.df.columns:
            self.df['last_session_date'] = pd.to_datetime(self.df['last_session_date'])
            self.df['days_since_last_session'] = (
                current_date - self.df['last_session_date']
            ).dt.total_seconds() / (24 * 60 * 60)  # Convert to days
            
        # Process last health check date
        if 'last_health_check_date' in self.df.columns:
            self.df['last_health_check_date'] = pd.to_datetime(self.df['last_health_check_date'])
            self.df['days_since_last_health_check'] = (
                current_date - self.df['last_health_check_date']
            ).dt.total_seconds() / (24 * 60 * 60)  # Convert to days
        
        # Normalize the new temporal features
        temporal_cols = ['days_since_last_session', 'days_since_last_health_check']
        for col in temporal_cols:
            if col in self.df.columns:
                scaler = StandardScaler()
                self.df[col] = scaler.fit_transform(self.df[[col]])
                self.scalers[col] = scaler
        
        print("Temporal features processed")
        return self.df

    def calculate_health_metrics(self):
        """
        Calculate additional health-related metrics
        """
        if 'previous_pregnancies' in self.df.columns:
            # Create pregnancy experience level
            self.df['pregnancy_experience'] = pd.cut(
                self.df['previous_pregnancies'],
                bins=[-1, 0, 1, float('inf')],
                labels=['First Time', 'Second Time', 'Experienced']
            )
        
        # Calculate health engagement score
        health_cols = [
            'health_tracking_submission_count',
            'view_tracking_questions_click_home_scre_count'
        ]
        
        if all(col in self.df.columns for col in health_cols):
            self.df['health_engagement_score'] = self.df[health_cols].sum(axis=1)
            
            # Normalize health engagement score
            scaler = StandardScaler()
            self.df['health_engagement_score'] = scaler.fit_transform(self.df[['health_engagement_score']])
            self.scalers['health_engagement_score'] = scaler
        
        print("Health metrics calculated")
        return self.df

    def preprocess(self):
        """
        Run all preprocessing steps
        """
        self.load_data()
        self.create_user_ids()
        self.handle_missing_values()
        self.handle_outliers()
        
        # Process dates first to create age
        self.prepare_date_features()
        
        # Then create groups that depend on processed features
        self.create_age_groups()
        self.calculate_bmi()
        self.calculate_engagement_metrics()
        self.calculate_trimester()
        self.calculate_risk_level()
        self.calculate_activity_patterns()
        self.process_temporal_features()
        self.calculate_health_metrics()
        
        # Standard preprocessing
        self.normalize_numerical_features()
        self.encode_categorical_features()
        self.save_preprocessed_data()
        return self.df

if __name__ == "__main__":
    # Create preprocessor and run preprocessing
    preprocessor = DataPreprocessor()
    preprocessed_data = preprocessor.preprocess()
    
    # Print summary of preprocessed data
    print("\nPreprocessed Data Summary:")
    print("-" * 50)
    print(f"Shape: {preprocessed_data.shape}")
    print("\nFeature Statistics:")
    print(preprocessed_data.describe())
    
    # Print categorical feature distributions
    print("\nCategorical Feature Distributions:")
    print("-" * 50)
    categorical_features = [
        'age_group', 'trimester', 'risk_level', 'city_tier',
        'pregnancy_experience', 'bmi_category'
    ]
    for feature in categorical_features:
        if feature in preprocessed_data.columns:
            print(f"\n{feature} distribution:")
            print(preprocessed_data[feature].value_counts(normalize=True))