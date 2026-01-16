import pandas as pd
import numpy as np
from datetime import datetime

def engineer_features(df):
    """
    Engineer features for fraud detection from transaction data.
    
    Parameters:
    df (pd.DataFrame): Raw transaction data
    
    Returns:
    pd.DataFrame: Data with engineered features
    """
    
    # Make a copy
    df = df.copy()
    
    # Convert timestamp to datetime if not already
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Temporal features
    df['hour_of_day'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    df['is_night'] = ((df['hour_of_day'] >= 22) | (df['hour_of_day'] <= 5)).astype(int)
    
    # Amount-based features
    df['amount_log'] = np.log1p(df['amount'])
    df['is_high_amount'] = (df['amount'] > df['amount'].quantile(0.95)).astype(int)
    
    # Customer behavior features (simplified - in real scenario would use historical data)
    customer_stats = df.groupby('customer_id')['amount'].agg(['mean', 'std', 'count']).reset_index()
    customer_stats.columns = ['customer_id', 'customer_avg_amount', 'customer_std_amount', 'customer_transaction_count']
    customer_stats['customer_std_amount'] = customer_stats['customer_std_amount'].fillna(0)
    
    df = df.merge(customer_stats, on='customer_id', how='left')
    
    # Deviation from customer average
    df['amount_deviation'] = (df['amount'] - df['customer_avg_amount']) / (df['customer_std_amount'] + 1)
    
    # Category risk scores (simplified)
    category_risk = {
        'Grocery': 0.1,
        'Restaurant': 0.2,
        'Online Shopping': 0.8,
        'Utilities': 0.1,
        'Entertainment': 0.7,
        'Transportation': 0.3,
        'Healthcare': 0.2,
        'Education': 0.1,
        'Telecom': 0.4,
        'Banking': 0.9
    }
    df['category_risk'] = df['merchant_category'].map(category_risk)
    
    # Location risk (simplified)
    location_risk = {
        'Mumbai': 0.8,
        'Delhi': 0.9,
        'Bangalore': 0.7,
        'Chennai': 0.5,
        'Kolkata': 0.6,
        'Pune': 0.4,
        'Hyderabad': 0.5,
        'Ahmedabad': 0.4,
        'Jaipur': 0.3,
        'Surat': 0.2
    }
    df['location_risk'] = df['location'].map(location_risk)
    
    # Combined risk score
    df['combined_risk'] = (
        0.3 * df['category_risk'] +
        0.3 * df['location_risk'] +
        0.2 * df['is_night'] +
        0.2 * df['is_high_amount']
    )
    
    # Frequency features (simplified - using customer stats instead of rolling)
    # Note: Rolling window features removed for simplicity in synthetic data demo
    df['transactions_last_hour'] = df['customer_transaction_count'] / 24  # Simplified proxy
    df['amount_last_hour'] = df['customer_avg_amount'] * df['transactions_last_hour']  # Simplified proxy
    
    # Fill NaN values
    df = df.fillna(0)
    
    return df

if __name__ == "__main__":
    # Load data
    df = pd.read_csv('data/upi_transactions.csv')
    
    # Engineer features
    df_featured = engineer_features(df)
    
    # Save featured data
    df_featured.to_csv('data/upi_transactions_featured.csv', index=False)
    
    print(f"Engineered features for {len(df_featured)} transactions")
    print("Featured data saved to data/upi_transactions_featured.csv")