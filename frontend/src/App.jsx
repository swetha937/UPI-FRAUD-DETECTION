import React, { useState } from 'react';
import './App.css';

function App() {
  const [formData, setFormData] = useState({
    customer_id: '',
    amount: '',
    merchant_category: '',
    location: '',
    timestamp: ''
  });
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleChange = (e) => {
    setFormData({
      ...formData,
      [e.target.name]: e.target.value
    });
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    
    // Simple fraud detection logic
    const transaction = {
      customer_id: formData.customer_id,
      amount: parseFloat(formData.amount),
      merchant_category: formData.merchant_category,
      location: formData.location,
      timestamp: new Date(formData.timestamp)
    };
    
    // Compute risk score based on simple rules
    let risk_score = 0;
    
    // Amount-based risk
    if (transaction.amount > 1000) risk_score += 0.3;
    else if (transaction.amount > 500) risk_score += 0.1;
    
    // Category risk
    const categoryRisk = {
      'Banking': 0.4,
      'Online Shopping': 0.3,
      'Entertainment': 0.2,
      'Transportation': 0.1,
      'Healthcare': 0.05,
      'Grocery': 0.05,
      'Utilities': 0.05,
      'Education': 0.05,
      'Telecom': 0.1,
      'Restaurant': 0.1
    };
    risk_score += categoryRisk[transaction.merchant_category] || 0;
    
    // Location risk
    const locationRisk = {
      'Delhi': 0.3,
      'Mumbai': 0.25,
      'Bangalore': 0.2,
      'Chennai': 0.15,
      'Kolkata': 0.15,
      'Pune': 0.1,
      'Hyderabad': 0.1,
      'Ahmedabad': 0.05,
      'Jaipur': 0.05,
      'Surat': 0.05
    };
    risk_score += locationRisk[transaction.location] || 0;
    
    // Time-based risk
    const hour = transaction.timestamp.getHours();
    if (hour >= 22 || hour <= 5) risk_score += 0.2; // Night time
    
    // Clamp risk_score between 0 and 1
    risk_score = Math.min(Math.max(risk_score, 0), 1);
    
    const decision = risk_score > 0.5 ? 'block' : 'allow';
    
    const actions = {};
    if (decision === 'block') {
      actions.alert = `High-risk transaction detected for customer ${transaction.customer_id}. Risk score: ${risk_score.toFixed(4)}`;
      actions.block_transaction = true;
      actions.notify_customer = true;
    } else {
      actions.allow_transaction = true;
    }
    
    setResult({
      risk_score: risk_score,
      decision: decision,
      actions: actions
    });
    
    setLoading(false);
  };

  return (
    <div className="App">
      <header className="App-header">
        <h1>UPI Fraud Detection System</h1>
      </header>
      <main className="container">
        <form onSubmit={handleSubmit} className="fraud-form">
          <div className="form-group">
            <label>Customer ID:</label>
            <input
              type="text"
              name="customer_id"
              value={formData.customer_id}
              onChange={handleChange}
              required
            />
          </div>
          
          <div className="form-group">
            <label>Amount:</label>
            <input
              type="number"
              step="0.01"
              name="amount"
              value={formData.amount}
              onChange={handleChange}
              required
            />
          </div>
          
          <div className="form-group">
            <label>Merchant Category:</label>
            <input
              type="text"
              name="merchant_category"
              value={formData.merchant_category}
              onChange={handleChange}
              required
            />
          </div>
          
          <div className="form-group">
            <label>Location:</label>
            <input
              type="text"
              name="location"
              value={formData.location}
              onChange={handleChange}
              required
            />
          </div>
          
          <div className="form-group">
            <label>Timestamp:</label>
            <input
              type="datetime-local"
              name="timestamp"
              value={formData.timestamp}
              onChange={handleChange}
              required
            />
          </div>
          
          <button type="submit" disabled={loading}>
            {loading ? 'Checking...' : 'Check Transaction'}
          </button>
        </form>
        
        {result && (
          <div className={`result ${result.decision === 'allow' ? 'safe' : 'fraud'}`}>
            {result.error ? (
              <p>{result.error}</p>
            ) : (
              <>
                <h2>Result</h2>
                <p><strong>Risk Score:</strong> {result.risk_score?.toFixed(4)}</p>
                <p><strong>Decision:</strong> {result.decision}</p>
                <p><strong>Actions:</strong> {JSON.stringify(result.actions)}</p>
              </>
            )}
          </div>
        )}
      </main>
    </div>
  );
}

export default App;