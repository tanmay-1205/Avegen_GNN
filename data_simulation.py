import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

def generate_synthetic_data(n_users=1000):
    """
    Generate synthetic user data for pregnancy support app
    
    Parameters:
    n_users: Number of users to generate
    
    Returns:
    DataFrame with synthetic user data
    """
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Generate user IDs
    user_ids = [f"USER_{str(i).zfill(6)}" for i in range(n_users)]
    
    # Define possible values for categorical fields
    cities = ['Mumbai', 'Delhi', 'Bangalore', 'Chennai', 'Kolkata', 'Pune', 'Hyderabad']
    occupations = ['Working Professional', 'Homemaker', 'Self Employed', 'Other']
    reasons = ['First Pregnancy', 'Second Pregnancy', 'Health Monitoring', 'General Wellness']
    conditions = ['None', 'Gestational Diabetes', 'Thyroid', 'Anemia', 'Hypertension']
    
    # Generate data
    data = {
        'user_id_hashed': user_ids,
        'city': np.random.choice(cities, n_users),
        
        # Session and engagement counts (following realistic patterns)
        'session_start_count': np.random.negative_binomial(5, 0.5, n_users),
        'watch_now_count': np.random.negative_binomial(3, 0.4, n_users),
        'community_opened_count': np.random.negative_binomial(2, 0.3, n_users),
        'hamburger_menu_nearby_hospitals_click_count': np.random.negative_binomial(1, 0.2, n_users),
        'courses_screen_view_count': np.random.negative_binomial(4, 0.4, n_users),
        'self_assessment_card_button_click_count': np.random.negative_binomial(2, 0.3, n_users),
        'course_start_tap_count': np.random.negative_binomial(3, 0.4, n_users),
        'health_tracking_submission_count': np.random.negative_binomial(4, 0.5, n_users),
        'view_tracking_questions_click_home_scre_count': np.random.negative_binomial(2, 0.3, n_users),
        'cr_card_tap_count': np.random.negative_binomial(2, 0.3, n_users),
        'week_content_share_app_click_count': np.random.negative_binomial(1, 0.2, n_users),
        'herstore_opened_count': np.random.negative_binomial(2, 0.3, n_users),
        'herhealth_opened_count': np.random.negative_binomial(2, 0.3, n_users),
        
        # User attributes
        'selected_height': np.random.normal(160, 10, n_users).round(1),  # in cm
        'selected_weight': np.random.normal(65, 12, n_users).round(1),   # in kg
        'selected_occupation': np.random.choice(occupations, n_users),
        'selected_reasons': np.random.choice(reasons, n_users),
        
        # Pregnancy specific
        'latest_week': np.random.randint(1, 40, n_users),
        'conditions': np.random.choice(conditions, n_users, p=[0.7, 0.1, 0.1, 0.05, 0.05])
    }
    
    # Generate dates of birth (25-40 years range)
    today = datetime.now()
    dates = [today - timedelta(days=random.randint(25*365, 40*365)) for _ in range(n_users)]
    data['dob'] = dates
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Add some data quality issues to make it realistic
    # 1. Add some missing values
    for col in ['selected_height', 'selected_weight', 'selected_occupation']:
        mask = np.random.random(n_users) < 0.05  # 5% missing values
        df.loc[mask, col] = np.nan
    
    # 2. Add some outliers
    height_outliers = np.random.random(n_users) < 0.02  # 2% outliers
    weight_outliers = np.random.random(n_users) < 0.02  # 2% outliers
    df.loc[height_outliers, 'selected_height'] = np.random.normal(200, 10, sum(height_outliers))
    df.loc[weight_outliers, 'selected_weight'] = np.random.normal(120, 15, sum(weight_outliers))
    
    return df

# Generate the data
df = generate_synthetic_data(1000)

# Basic data quality check
print("\nDataset Overview:")
print("-" * 50)
print(f"Number of records: {len(df)}")
print("\nMissing values:")
print(df.isnull().sum())
print("\nData sample:")
print(df.head())
print("\nData statistics:")
print(df.describe())

# Save to CSV
df.to_csv('synthetic_pregnancy_app_data.csv', index=False)