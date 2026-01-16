import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

def generate_synthetic_upi_data(n_transactions=10000, fraud_rate=0.02):
    """
    Generate synthetic UPI transaction data with realistic features.
    
    Parameters:
    n_transactions (int): Number of transactions to generate
    fraud_rate (float): Proportion of fraudulent transactions
    
    Returns:
    pd.DataFrame: Synthetic transaction data
    """
    
    # Set random seed for reproducibility
    np.random.seed(42)
    random.seed(42)
    
    # Calculate number of fraud transactions
    n_fraud = int(n_transactions * fraud_rate)
    n_legitimate = n_transactions - n_fraud
    
    # Merchant categories
    merchant_categories = [
        'Grocery', 'Restaurant', 'Online Shopping', 'Utilities', 'Entertainment',
        'Transportation', 'Healthcare', 'Education', 'Telecom', 'Banking'
    ]
    
    # Locations (Indian cities)
    locations = [
        'Mumbai', 'Delhi', 'Bangalore', 'Chennai', 'Kolkata', 'Pune', 
        'Hyderabad', 'Ahmedabad', 'Jaipur', 'Surat'
    ]
    
    # Generate legitimate transactions
    legitimate_data = []
    for i in range(n_legitimate):
        customer_id = f'CUST_{random.randint(1000, 9999)}'
        
        # Generate timestamp (last 30 days)
        base_date = datetime.now() - timedelta(days=30)
        timestamp = base_date + timedelta(
            days=random.randint(0, 29),
            hours=random.randint(0, 23),
            minutes=random.randint(0, 59)
        )
        
        # Transaction amount (legitimate: lower amounts, normal distribution)
        amount = np.random.lognormal(mean=6, sigma=1)  # Around ₹400-500 average
        
        # Other features
        merchant_category = random.choice(merchant_categories)
        location = random.choice(locations)
        
        legitimate_data.append({
            'transaction_id': f'TXN_{i+1:06d}',
            'customer_id': customer_id,
            'timestamp': timestamp,
            'amount': round(amount, 2),
            'merchant_category': merchant_category,
            'location': location,
            'is_fraud': 0
        })
    
    # Generate fraudulent transactions
    fraud_data = []
    for i in range(n_fraud):
        customer_id = f'CUST_{random.randint(1000, 9999)}'
        
        # Fraudulent transactions often at odd hours
        base_date = datetime.now() - timedelta(days=30)
        hour = random.choice([2, 3, 4, 5, 22, 23, 0, 1])  # Late night/early morning
        timestamp = base_date + timedelta(
            days=random.randint(0, 29),
            hours=hour,
            minutes=random.randint(0, 59)
        )
        
        # Higher amounts for fraud
        amount = np.random.lognormal(mean=8, sigma=1.5)  # Higher amounts
        
        # Fraud often in certain categories
        fraud_categories = ['Online Shopping', 'Entertainment', 'Banking']
        merchant_category = random.choice(fraud_categories)
        
        # Fraud locations might be different
        fraud_locations = ['Mumbai', 'Delhi', 'Bangalore']  # Major cities
        location = random.choice(fraud_locations)
        
        fraud_data.append({
            'transaction_id': f'TXN_{n_legitimate + i + 1:06d}',
            'customer_id': customer_id,
            'timestamp': timestamp,
            'amount': round(amount, 2),
            'merchant_category': merchant_category,
            'location': location,
            'is_fraud': 1
        })
    
    # Combine and shuffle
    all_data = legitimate_data + fraud_data
    df = pd.DataFrame(all_data)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    return df

if __name__ == "__main__":
    # Generate data
    df = generate_synthetic_upi_data()
    
    # Save to CSV
    df.to_csv('data/upi_transactions.csv', index=False)
    
    print(f"Generated {len(df)} transactions")
    print(f"Fraud rate: {df['is_fraud'].mean():.2%}")
    print("Data saved to data/upi_transactions.csv")