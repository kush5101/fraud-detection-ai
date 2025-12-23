import streamlit as st
import pandas as pd
import numpy as np
import random
import time
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
import io

# ==========================================
# 1. SYNTHETIC DATA GENERATOR
# ==========================================
def generate_synthetic_data(num_samples=1000, fraud_ratio=0.1):
    """
    Generates a synthetic dataset for credit card fraud detection.
    """
    ids = [f"TXN-{random.randint(100000, 999999)}" for _ in range(num_samples)]
    
    # Base Features
    amounts = []
    times = []
    
    # Advanced Features
    ip_risks = [] # 0.0 to 1.0
    distances = [] # km from home
    use_chip = [] # 1: Chip, 0: Swipe/Online
    merchant_risks = [] # 0.0 to 1.0

    is_fraud_labels = []

    for _ in range(num_samples):
        # Determine if this sample will be fraud based on ratio
        is_fraud = 1 if random.random() < fraud_ratio else 0
        is_fraud_labels.append(is_fraud)

        if is_fraud:
            # Fraud patterns
            amounts.append(round(random.uniform(200, 10000), 2))
            times.append(random.choice([0, 1, 2, 3, 4, 23])) # Night
            
            # High risk indicators
            ip_risks.append(random.uniform(0.6, 1.0)) # High IP risk
            distances.append(random.uniform(100, 5000)) # Far from home
            merchant_risks.append(random.uniform(0.5, 1.0)) # Risky merchant
            use_chip.append(0) # Likely online or swipe
        
        else:
            # Normal patterns
            amounts.append(round(random.uniform(5, 300), 2))
            times.append(random.randint(6, 22))
            
            # Low risk indicators
            ip_risks.append(random.uniform(0.0, 0.2))
            distances.append(random.uniform(0, 50)) # Close to home
            merchant_risks.append(random.uniform(0.0, 0.3))
            use_chip.append(random.choice([0, 1]))

    df = pd.DataFrame({
        'TransactionID': ids,
        'Amount': amounts,
        'Hour': times,
        'IP_Risk': ip_risks,
        'Distance_km': distances,
        'Merchant_Risk': merchant_risks,
        'Use_Chip': use_chip,
        'IsFraud': is_fraud_labels
    })
    
    return df

# ==========================================
# 2. HYBRID MODEL CLASS
# ==========================================
class HybridFraudDetector:
    def __init__(self):
        # Initialize individual models
        self.lr = LogisticRegression(random_state=42)
        self.dt = DecisionTreeClassifier(random_state=42)
        self.rf = RandomForestClassifier(n_estimators=100, random_state=42)
        self.svm = SVC(probability=True, random_state=42)
        self.nn = MLPClassifier(hidden_layer_sizes=(50, 25), max_iter=500, random_state=42)
        
        # Create the Ensemble (Voting Classifier)
        self.model = VotingClassifier(
            estimators=[
                ('lr', self.lr),
                ('dt', self.dt),
                ('rf', self.rf),
                ('svm', self.svm),
                ('nn', self.nn)
            ],
            voting='soft'
        )
        
        self.scaler = StandardScaler()

    def train(self, X_train, y_train):
        """Trains the ensemble model."""
        X_scaled = self.scaler.fit_transform(X_train)
        self.model.fit(X_scaled, y_train)

    def predict(self, X):
        """Returns predictions and probabilities."""
        X_scaled = self.scaler.transform(X)
        preds = self.model.predict(X_scaled)
        probs = self.model.predict_proba(X_scaled)[:, 1] # Probability of fraud
        return preds, probs

    def get_individual_confidences(self, X):
        """Returns the probability outcome from each individual model for visualization."""
        X_scaled = self.scaler.transform(X)
        
        confidences = {
            'Logistic Regression': self.model.named_estimators_['lr'].predict_proba(X_scaled)[:, 1],
            'Decision Tree': self.model.named_estimators_['dt'].predict_proba(X_scaled)[:, 1],
            'Random Forest': self.model.named_estimators_['rf'].predict_proba(X_scaled)[:, 1],
            'SVM': self.model.named_estimators_['svm'].predict_proba(X_scaled)[:, 1],
            'Neural Network': self.model.named_estimators_['nn'].predict_proba(X_scaled)[:, 1]
        }
        return confidences

# ==========================================
# 3. STREAMLIT DASHBOARD
# ==========================================

# Page Config
st.set_page_config(
    page_title="Hybrid AI Fraud Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Session State Initialization
if 'history' not in st.session_state:
    st.session_state.history = pd.DataFrame(columns=['ID', 'Amount', 'Time', 'IP Risk', 'Dist(km)', 'Risk Score', 'Status'])
if 'stats' not in st.session_state:
    st.session_state.stats = {'total': 0, 'safe': 0, 'fraud': 0}

# Custom CSS for Cyberpunk/Tech look (Glassmorphism)
st.markdown("""
<style>
    .reportview-container {
        background: radial-gradient(circle at 10% 20%, rgb(15, 23, 42) 0%, rgb(0, 0, 0) 90%);
    }
    .metric-card {
        background: rgba(255, 255, 255, 0.03);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        padding: 20px;
        border-radius: 15px;
        margin-bottom: 20px;
        transition: transform 0.3s ease;
    }
    .metric-card:hover {
        transform: translateY(-5px);
        border: 1px solid rgba(16, 185, 129, 0.3);
    }
    .metric-value {
        font-size: 2.2rem;
        font-weight: 700;
        background: -webkit-linear-gradient(45deg, #10b981, #3b82f6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .metric-label {
        color: #94a3b8;
        font-family: 'Helvetica Neue', sans-serif;
        text-transform: uppercase;
        letter-spacing: 1px;
        font-size: 0.8rem;
    }
    div[data-testid="stMetric"] {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 10px;
        padding: 10px;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
</style>
""", unsafe_allow_html=True)

# --- IN-APP MODEL TRAINING ---
@st.cache_resource
def get_trained_model():
    with st.spinner("Initializing AI Core & Training Hybrid Models..."):
        # 1. Generate Training Data
        df_train = generate_synthetic_data(num_samples=3000, fraud_ratio=0.15)
        X_train = df_train[['Amount', 'Hour', 'IP_Risk', 'Distance_km', 'Merchant_Risk', 'Use_Chip']]
        y_train = df_train['IsFraud']
        
        # 2. Train Model
        detect = HybridFraudDetector()
        detect.train(X_train, y_train)
        return detect

# Initialize
try:
    detector = get_trained_model()
    st.success("System Online: AI Models Active (LR, RF, SVM, NN, DT)")
except Exception as e:
    st.error(f"Critical System Failure: {e}")
    st.stop()

# --- SIDEBAR ---
st.sidebar.title("🛡️ SOC Control Panel")
st.sidebar.markdown("---")
st.sidebar.subheader("Simulation Parameters")
speed = st.sidebar.slider("Tick Speed (ms)", 100, 3000, 1000)
fraud_prob = st.sidebar.slider("Fraud Injection Rate", 0.0, 1.0, 0.15)
st.sidebar.markdown("---")
st.sidebar.info("Model: Hybrid Ensemble (Voting)")

# --- MAIN LAYOUT ---
st.title("🛡️ Hybrid AI Fraud Detection System")

# Top Metrics
col1, col2, col3, col4 = st.columns(4)
# Create placeholders for dynamic updates
p1 = col1.empty()
p2 = col2.empty()
p3 = col3.empty()
p4 = col4.empty()

def render_metric(placeholder, label, value, color="#10b981"):
    placeholder.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value" style="color: {color};">{value}</div>
    </div>
    """, unsafe_allow_html=True)

# Initial Render
render_metric(p1, "Processed TXNs", st.session_state.stats['total'])
render_metric(p2, "Safe Verified", st.session_state.stats['safe'])
render_metric(p3, "Fraud Detected", st.session_state.stats['fraud'], "#ef4444")
render_metric(p4, "Global Threat Level", "LOW", "#f59e0b")

# Main Feed & Charts
col_feed, col_chart = st.columns([2, 1])

with col_feed:
    st.subheader("Live Transaction Stream")
    feed_placeholder = st.empty()

with col_chart:
    st.subheader("AI Confidence Levels")
    chart_placeholder = st.empty()


# Simulation Control
if st.button("▶ START / ⏹ STOP MONITORING"):
    st.session_state.running = not st.session_state.get('running', False)

# --- SIMULATION LOOP ---
if st.session_state.get('running', False):
    
    # Generate 1 Live Transaction
    new_data = generate_synthetic_data(num_samples=1, fraud_ratio=fraud_prob)
    X_new = new_data[['Amount', 'Hour', 'IP_Risk', 'Distance_km', 'Merchant_Risk', 'Use_Chip']]
    
    # Prediction
    pred, prob = detector.predict(X_new)
    is_fraud = pred[0] == 1
    risk_score = prob[0]
    
    # Update Stats
    st.session_state.stats['total'] += 1
    if is_fraud:
        st.session_state.stats['fraud'] += 1
        status = "DETECTED"
    else:
        st.session_state.stats['safe'] += 1
        status = "SAFE"
        
    # Calculate Threat Level
    fraud_rate = st.session_state.stats['fraud'] / st.session_state.stats['total'] if st.session_state.stats['total'] > 0 else 0
    if fraud_rate < 0.05:
        threat_level, threat_color = "LOW", "#10b981"
    elif fraud_rate < 0.15:
        threat_level, threat_color = "MODERATE", "#f59e0b"
    elif fraud_rate < 0.30:
        threat_level, threat_color = "HIGH", "#ef4444"
    else:
        threat_level, threat_color = "CRITICAL", "#7f1d1d"

    # UI Updates (Dynamic Cards)
    render_metric(p1, "Processed TXNs", st.session_state.stats['total'])
    render_metric(p2, "Safe Verified", st.session_state.stats['safe'])
    render_metric(p3, "Fraud Detected", st.session_state.stats['fraud'], "#ef4444")
    render_metric(p4, "Global Threat Level", threat_level, threat_color)
    
    # Feed Update
    new_row = {
        'ID': new_data['TransactionID'][0],
        'Amount': f"${new_data['Amount'][0]}",
        'Time': f"{new_data['Hour'][0]}:00",
        'IP Risk': f"{new_data['IP_Risk'][0]:.2f}",
        'Dist(km)': f"{new_data['Distance_km'][0]:.1f}", 
        'Risk Score': f"{risk_score:.1%}",
        'Status': status
    }
    st.session_state.history = pd.concat([pd.DataFrame([new_row]), st.session_state.history], ignore_index=True).head(100)
    
    def color_status(val):
        color = '#ef4444' if val == 'DETECTED' else '#10b981'
        return f'color: {color}; font-weight: bold;'
    
    feed_placeholder.dataframe(
        st.session_state.history.style.applymap(color_status, subset=['Status']),
        use_container_width=True
    )
    
    # --- VISUALIZATION (Plotly) ---
    individual_confs = detector.get_individual_confidences(X_new)
    chart_data = {k: v[0] for k,v in individual_confs.items()}
    
    # 1. 3D Model Confidence
    fig_bar = go.Figure(go.Bar(
        x=list(chart_data.values()),
        y=list(chart_data.keys()),
        orientation='h',
        marker=dict(
            color=list(chart_data.values()),
            colorscale='Bluered',
            showscale=False
        )
    ))
    fig_bar.update_layout(
        title="🤖 Model Consensus",
        xaxis_title="Confidence Probability",
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        height=300,
        margin=dict(l=0, r=0, t=30, b=0)
    )
    chart_placeholder.plotly_chart(fig_bar, use_container_width=True)
    
    # 2. 3D Risk Dimensions
    fig_3d = go.Figure(data=[go.Scatter3d(
        x=[new_data['Amount'][0]],
        y=[new_data['Distance_km'][0]],
        z=[new_data['IP_Risk'][0]],
        mode='markers',
        marker=dict(
            size=12,
            color=[risk_score],
            colorscale='Bluered',
            opacity=0.8
        ),
        text=[f"Risk: {risk_score:.2%}"],
        hoverinfo='text'
    )])
    
    # Add baseline point for reference
    fig_3d.add_trace(go.Scatter3d(
        x=[20], y=[5], z=[0.1], # Typical user behavior
        mode='markers',
        marker=dict(size=8, color='cyan', symbol='diamond'),
        name='Baseline'
    ))

    fig_3d.update_layout(
        title="3D Risk Vector Analysis",
        scene=dict(
            xaxis_title='Amount ($)',
            yaxis_title='Distance (km)',
            zaxis_title='IP Risk',
            bgcolor='rgba(0,0,0,0)'
        ),
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        height=300,
        margin=dict(l=0, r=0, t=30, b=0)
    )
    
    chart_placeholder.plotly_chart(fig_3d, use_container_width=True)
    
    # Helper for deviation analysis
    def decode_location(dist): return "San Francisco (Home)" if dist < 10 else f"Unknown ({dist:.0f}km away)"
    def decode_ip(risk): return "Comcast Cable (Trusted)" if risk < 0.2 else "Tor Exit Node (High Risk)"
    
    curr_loc = decode_location(new_data['Distance_km'][0])
    curr_ip = decode_ip(new_data['IP_Risk'][0])
    
    # Compare
    st.markdown("#### 🔬 feature Analysis & Anomaly Detection")
    
    with st.container(border=True):
        col_d1, col_d2, col_d3 = st.columns(3)
        
        # 1. Location Analysis
        with col_d1:
            delta_color = "normal" if new_data['Distance_km'][0] < 50 else "inverse"
            val_str = "✅ Verified Location" if new_data['Distance_km'][0] < 50 else f"⚠️ {curr_loc}"
            st.metric("Location Status", val_str, delta=f"{new_data['Distance_km'][0]:.0f}km from Base", delta_color=delta_color)
                
        # 2. IP Network Analysis
        with col_d2:
            is_safe_ip = new_data['IP_Risk'][0] < 0.5
            delta_color = "normal" if is_safe_ip else "inverse"
            val_str = "✅ Trusted Network" if is_safe_ip else "❌ Suspicious Proxy"
            st.metric("Network Risk", val_str, delta=curr_ip, delta_color=delta_color)
                
        # 3. Transaction Type / Device
        with col_d3:
            is_chip = new_data['Use_Chip'][0] == 1
            delta_color = "normal" if is_chip else "off"
            val_str = "💳 Chip Card" if is_chip else "⚠️ Card-Not-Present"
            st.metric("Transaction Mode", val_str, delta="Baseline: Chip", delta_color=delta_color)

    time.sleep(speed / 1000)
    st.rerun()

elif not st.session_state.get('running', False):
    st.info("System Standby. Press START to begin real-time analysis.")

# --- EXPORT SECTION ---
st.sidebar.markdown("### 💾 Export Data")
if not st.session_state.history.empty:
    csv = st.session_state.history.to_csv(index=False).encode('utf-8')
    st.sidebar.download_button(
        label="Download TXN Log (CSV)",
        data=csv,
        file_name='fraud_detection_log.csv',
        mime='text/csv',
    )
