import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from xgboost import XGBClassifier
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
TORCH_AVAILABLE = False
TORCH_GEOMETRIC_AVAILABLE = False

if TORCH_AVAILABLE:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset

if TORCH_GEOMETRIC_AVAILABLE:
    from torch_geometric.data import Data
    from torch_geometric.nn import GCNConv
from sklearn.preprocessing import LabelEncoder

def train_models(df):
    """
    Train multiple ML models for fraud detection.
    
    Parameters:
    df (pd.DataFrame): Featured transaction data
    
    Returns:
    dict: Trained models
    """
    
    # Feature columns (exclude IDs, timestamps, target)
    exclude_cols = ['transaction_id', 'customer_id', 'timestamp', 'is_fraud', 
                   'merchant_category', 'location']
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    X = df[feature_cols]
    y = df['is_fraud']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Train Random Forest
    rf_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        class_weight='balanced'
    )
    rf_model.fit(X_train, y_train)
    
    # Train XGBoost
    xgb_model = XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        scale_pos_weight=len(y_train[y_train==0])/len(y_train[y_train==1])
    )
    xgb_model.fit(X_train, y_train)
    
    # Ensemble (simple average of probabilities)
    class EnsembleModel:
        def __init__(self, models):
            self.models = models
        
        def predict_proba(self, X):
            probs = [model.predict_proba(X)[:, 1] for model in self.models]
            return np.mean(probs, axis=0)
        
        def predict(self, X, threshold=0.5):
            probs = self.predict_proba(X)
            return (probs >= threshold).astype(int)
    
    ensemble_model = EnsembleModel([rf_model, xgb_model])
    
    if TORCH_GEOMETRIC_AVAILABLE:
        # Build graph for GNN
        # Nodes: customers and merchants
        customer_encoder = LabelEncoder()
        merchant_encoder = LabelEncoder()
        
        df['customer_encoded'] = customer_encoder.fit_transform(df['customer_id'])
        df['merchant_encoded'] = merchant_encoder.fit_transform(df['merchant_category'])
        
        # Edges: transactions between customers and merchants
        edge_index = torch.tensor([df['customer_encoded'].values, df['merchant_encoded'].values], dtype=torch.long)
        
        # Node features: simple one-hot or aggregated features
        num_customers = len(customer_encoder.classes_)
        num_merchants = len(merchant_encoder.classes_)
        
        # For simplicity, use random features or aggregated
        customer_features = torch.randn(num_customers, 10)  # Placeholder
        merchant_features = torch.randn(num_merchants, 10)  # Placeholder
        x = torch.cat([customer_features, merchant_features], dim=0)
        
        # Labels for nodes (fraud if any transaction is fraud)
        customer_labels = df.groupby('customer_encoded')['is_fraud'].max().values
        merchant_labels = df.groupby('merchant_encoded')['is_fraud'].max().values
        y = torch.tensor(np.concatenate([customer_labels, merchant_labels]), dtype=torch.long)
        
        graph_data = Data(x=x, edge_index=edge_index, y=y)
        
        # GNN Model
        class GNNModel(nn.Module):
            def __init__(self, in_channels, hidden_channels, out_channels):
                super(GNNModel, self).__init__()
                self.conv1 = GCNConv(in_channels, hidden_channels)
                self.conv2 = GCNConv(hidden_channels, out_channels)
            
            def forward(self, data):
                x, edge_index = data.x, data.edge_index
                x = self.conv1(x, edge_index)
                x = torch.relu(x)
                x = self.conv2(x, edge_index)
                return torch.sigmoid(x)
        
        gnn_model = GNNModel(in_channels=10, hidden_channels=16, out_channels=1)
        optimizer = optim.Adam(gnn_model.parameters(), lr=0.01)
        criterion = nn.BCELoss()
        
        # Train GNN (simple training)
        gnn_model.train()
        for epoch in range(10):
            optimizer.zero_grad()
            out = gnn_model(graph_data)
            loss = criterion(out.squeeze(), graph_data.y.float())
            loss.backward()
            optimizer.step()
    else:
        gnn_model = None
        customer_encoder = None
        merchant_encoder = None
    
    if TORCH_AVAILABLE:
        # LSTM for sequences
        # Group transactions by customer, sort by time
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values(['customer_id', 'timestamp'])
        
        # Create sequences: last 5 transactions per customer
        sequences = []
        labels = []
        for customer in df['customer_id'].unique():
            cust_df = df[df['customer_id'] == customer]
            features = cust_df[feature_cols].values
            if len(features) >= 5:
                for i in range(5, len(features)):
                    seq = features[i-5:i]
                    label = cust_df.iloc[i]['is_fraud']
                    sequences.append(seq)
                    labels.append(label)
        
        if sequences:
            X_seq = torch.tensor(np.array(sequences), dtype=torch.float32)
            y_seq = torch.tensor(np.array(labels), dtype=torch.float32)
            
            class LSTMModel(nn.Module):
                def __init__(self, input_size, hidden_size, output_size):
                    super(LSTMModel, self).__init__()
                    self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
                    self.fc = nn.Linear(hidden_size, output_size)
                
                def forward(self, x):
                    _, (hn, _) = self.lstm(x)
                    out = self.fc(hn.squeeze(0))
                    return torch.sigmoid(out)
            
            lstm_model = LSTMModel(input_size=len(feature_cols), hidden_size=50, output_size=1)
            optimizer_lstm = optim.Adam(lstm_model.parameters(), lr=0.01)
            criterion_lstm = nn.BCELoss()
            
            # Train LSTM
            dataset = TensorDataset(X_seq, y_seq)
            dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
            lstm_model.train()
            for epoch in range(10):
                for batch_x, batch_y in dataloader:
                    optimizer_lstm.zero_grad()
                    out = lstm_model(batch_x)
                    loss = criterion_lstm(out.squeeze(), batch_y)
                    loss.backward()
                    optimizer_lstm.step()
        else:
            lstm_model = None
    else:
        lstm_model = None
    
    models = {
        'random_forest': rf_model,
        'xgboost': xgb_model,
        'ensemble': ensemble_model,
        'gnn': gnn_model if TORCH_GEOMETRIC_AVAILABLE else None,
        'lstm': lstm_model if TORCH_AVAILABLE else None
    }
    
    # Evaluate models
    for name, model in models.items():
        if name in ['random_forest', 'xgboost', 'ensemble']:
            if name == 'ensemble':
                y_pred = model.predict(X_test)
                y_pred_proba = model.predict_proba(X_test)
            else:
                y_pred = model.predict(X_test)
                y_pred_proba = model.predict_proba(X_test)[:, 1]
            
            print(f"\n{name.upper()} Model Results:")
            print(classification_report(y_test, y_pred))
            print(f"ROC-AUC: {roc_auc_score(y_test, y_pred_proba):.4f}")
            
            # Confusion matrix
            cm = confusion_matrix(y_test, y_pred)
            plt.figure(figsize=(6, 4))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.title(f'{name.upper()} Confusion Matrix')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            plt.savefig(f'reports/{name}_confusion_matrix.png')
            plt.close()
        elif name == 'gnn' and TORCH_GEOMETRIC_AVAILABLE:
            gnn_model.eval()
            with torch.no_grad():
                pred = gnn_model(graph_data).squeeze()
                pred_binary = (pred > 0.5).int()
                print(f"\nGNN Model Results:")
                print(classification_report(graph_data.y.numpy(), pred_binary.numpy()))
                print(f"ROC-AUC: {roc_auc_score(graph_data.y.numpy(), pred.numpy()):.4f}")
        elif name == 'lstm' and TORCH_AVAILABLE and sequences:
            lstm_model.eval()
            with torch.no_grad():
                pred = lstm_model(X_seq).squeeze()
                pred_binary = (pred > 0.5).int()
                print(f"\nLSTM Model Results:")
                print(classification_report(y_seq.numpy(), pred_binary.numpy()))
                print(f"ROC-AUC: {roc_auc_score(y_seq.numpy(), pred.numpy()):.4f}")
    
    return models, X_test, y_test, customer_encoder, merchant_encoder

def save_models(models):
    """Save trained models to disk."""
    for name, model in models.items():
        if name not in ['ensemble', 'gnn', 'lstm']:  # Can't pickle ensemble, GNN, LSTM easily
            joblib.dump(model, f'models/{name}_model.joblib')
    
    # Save encoders if available
    if 'customer_encoder' in globals() and customer_encoder:
        joblib.dump(customer_encoder, 'models/customer_encoder.joblib')
    if 'merchant_encoder' in globals() and merchant_encoder:
        joblib.dump(merchant_encoder, 'models/merchant_encoder.joblib')
    
    print("Models saved to models/ directory")

def real_time_risk_engine(transaction, models, feature_cols, customer_encoder, merchant_encoder):
    """
    Real-time decision engine that assigns risk score to a transaction and triggers actions.
    
    Parameters:
    transaction (dict): Transaction data
    models (dict): Trained models
    feature_cols (list): Feature columns
    customer_encoder, merchant_encoder: Encoders for GNN
    
    Returns:
    float: Risk score (probability of fraud)
    str: Decision (allow/block)
    dict: Triggered actions
    """
    # Prepare features
    df_trans = pd.DataFrame([transaction])
    df_trans['timestamp'] = pd.to_datetime(df_trans['timestamp'])
    # Assume feature engineering is done elsewhere or add here
    
    X = df_trans[feature_cols]
    
    # Ensemble prediction (use RF if ensemble not available)
    if 'ensemble' in models and models['ensemble']:
        ensemble = models['ensemble']
        risk_score = ensemble.predict_proba(X)[0]
    else:
        # Fallback to Random Forest
        rf = models.get('random_forest')
        if rf:
            risk_score = rf.predict_proba(X)[:, 1][0]
        else:
            risk_score = 0.5  # Default
    
    # If GNN available, incorporate
    if 'gnn' in models and models['gnn'] is not None:
        cust_enc = customer_encoder.transform([transaction['customer_id']])[0]
        merch_enc = merchant_encoder.transform([transaction['merchant_category']])[0]
        # Node prediction
        gnn_pred = models['gnn'](Data(x=torch.randn(1,10), edge_index=torch.tensor([[cust_enc], [merch_enc]], dtype=torch.long))).item()
        risk_score = (risk_score + gnn_pred) / 2  # Average
    
    decision = 'block' if risk_score > 0.5 else 'allow'
    
    # Trigger actions
    actions = {}
    if decision == 'block':
        actions['alert'] = f"High-risk transaction detected for customer {transaction['customer_id']}. Risk score: {risk_score:.4f}"
        actions['block_transaction'] = True
        actions['notify_customer'] = True
    else:
        actions['allow_transaction'] = True
    
    return risk_score, decision, actions

if __name__ == "__main__":
    # Load featured data
    df = pd.read_csv('data/upi_transactions_featured.csv')
    
    # Train models
    models, X_test, y_test, customer_encoder, merchant_encoder = train_models(df)
    
    # Save models
    save_models(models)
    
    print("Model training completed!")