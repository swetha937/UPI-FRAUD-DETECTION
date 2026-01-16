import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, precision_recall_curve

def evaluate_models(models, X_test, y_test):
    """
    Comprehensive evaluation of trained models.
    
    Parameters:
    models (dict): Trained models
    X_test (pd.DataFrame): Test features
    y_test (pd.Series): Test labels
    """
    
    plt.figure(figsize=(12, 8))
    
    # ROC Curves
    plt.subplot(2, 2, 1)
    for name, model in models.items():
        if name == 'ensemble':
            y_pred_proba = model.predict_proba(X_test)
        else:
            y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        plt.plot(fpr, tpr, label=f'{name.upper()} (AUC = {roc_auc_score(y_test, y_pred_proba):.3f})')
    
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves')
    plt.legend()
    plt.grid(True)
    
    # Precision-Recall Curves
    plt.subplot(2, 2, 2)
    for name, model in models.items():
        if name == 'ensemble':
            y_pred_proba = model.predict_proba(X_test)
        else:
            y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
        plt.plot(recall, precision, label=name.upper())
    
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curves')
    plt.legend()
    plt.grid(True)
    
    # Feature Importance (Random Forest)
    plt.subplot(2, 2, 3)
    rf_model = models['random_forest']
    feature_importance = pd.DataFrame({
        'feature': X_test.columns,
        'importance': rf_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    sns.barplot(data=feature_importance.head(10), x='importance', y='feature')
    plt.title('Top 10 Feature Importances (Random Forest)')
    plt.xlabel('Importance')
    
    # Fraud Distribution by Hour
    plt.subplot(2, 2, 4)
    # Load original data for this plot
    df_orig = pd.read_csv('data/upi_transactions.csv')
    df_orig['timestamp'] = pd.to_datetime(df_orig['timestamp'])
    df_orig['hour'] = df_orig['timestamp'].dt.hour
    
    fraud_by_hour = df_orig.groupby('hour')['is_fraud'].mean()
    fraud_by_hour.plot(kind='bar')
    plt.title('Fraud Rate by Hour of Day')
    plt.xlabel('Hour')
    plt.ylabel('Fraud Rate')
    plt.xticks(rotation=0)
    
    plt.tight_layout()
    plt.savefig('reports/model_evaluation.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Evaluation plots saved to reports/model_evaluation.png")

def calculate_business_impact(models, X_test, y_test):
    """
    Calculate business impact metrics.
    
    Parameters:
    models (dict): Trained models
    X_test (pd.DataFrame): Test features
    y_test (pd.Series): Test labels
    """
    
    # Assume average transaction amount and cost assumptions
    avg_transaction_amount = 500  # INR
    fraud_prevention_cost = 50    # INR per flagged transaction
    fraud_loss = 500             # INR average loss per fraud
    
    print("\nBusiness Impact Analysis:")
    print("=" * 50)
    
    for name, model in models.items():
        if name == 'ensemble':
            y_pred = model.predict(X_test)
        else:
            y_pred = model.predict(X_test)
        
        # Calculate metrics
        tp = ((y_pred == 1) & (y_test == 1)).sum()  # True positives
        fp = ((y_pred == 1) & (y_test == 0)).sum()  # False positives
        fn = ((y_pred == 0) & (y_test == 1)).sum()  # False negatives
        
        # Financial calculations
        fraud_prevented = tp * fraud_loss
        prevention_cost = (tp + fp) * fraud_prevention_cost
        net_benefit = fraud_prevented - prevention_cost
        
        print(f"\n{name.upper()} Model:")
        print(f"  Fraud Transactions Prevented: {tp}")
        print(f"  False Positives: {fp}")
        print(f"  Fraud Loss Prevented: ₹{fraud_prevented:,.0f}")
        print(f"  Prevention Cost: ₹{prevention_cost:,.0f}")
        print(f"  Net Benefit: ₹{net_benefit:,.0f}")
        print(f"  ROI: {net_benefit/prevention_cost*100:.1f}%" if prevention_cost > 0 else "  ROI: N/A")

if __name__ == "__main__":
    # This would be called after training
    # For now, just print that evaluation is ready
    print("Evaluation module ready. Run after model training.")