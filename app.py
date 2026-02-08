"""
Fraud Detection Web Application
A Streamlit-based dashboard for detecting fraudulent transactions using ML
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import google.generativeai as genai
import firebase_admin
from firebase_admin import credentials, firestore
from datetime import datetime
import json
import os

# ========================================
# PAGE CONFIGURATION
# ========================================

st.set_page_config(
    page_title="Fraud Detection System",
    page_icon="🔒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ========================================
# CUSTOM STYLING
# ========================================

st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .stAlert {
        background-color: #fff3cd;
        border-left: 4px solid #ffc107;
    }
    h1 {
        color: #1f4788;
        font-weight: 700;
    }
    h2 {
        color: #2c5aa0;
        font-weight: 600;
    }
    .metric-card {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    </style>
""", unsafe_allow_html=True)

# ========================================
# FIREBASE INITIALIZATION
# ========================================

def initialize_firebase():
    """Initialize Firebase connection (if not already initialized)"""
    if not firebase_admin._apps:
        try:
            # Option 1: Using service account JSON file
            if os.path.exists('firebase-credentials.json'):
                cred = credentials.Certificate('firebase-credentials.json')
                firebase_admin.initialize_app(cred)
                return firestore.client()
            
            # Option 2: Using environment variable with JSON string
            elif os.getenv('FIREBASE_CREDENTIALS'):
                cred_dict = json.loads(os.getenv('FIREBASE_CREDENTIALS'))
                cred = credentials.Certificate(cred_dict)
                firebase_admin.initialize_app(cred)
                return firestore.client()
            
            else:
                st.warning("⚠️ Firebase credentials not found. Fraud cases won't be saved to database.")
                return None
                
        except Exception as e:
            st.error(f"Firebase initialization error: {str(e)}")
            return None
    else:
        return firestore.client()

# ========================================
# GEMINI AI INITIALIZATION
# ========================================

def initialize_gemini(api_key):
    """Initialize Gemini API"""
    try:
        genai.configure(api_key=api_key)
        return genai.GenerativeModel('gemini-pro')
    except Exception as e:
        st.error(f"Gemini API initialization error: {str(e)}")
        return None

# ========================================
# DATA PROCESSING
# ========================================

class DataProcessor:
    """Handle CSV upload and data cleaning"""
    
    @staticmethod
    def load_and_clean_data(uploaded_file):
        """Load CSV and perform basic cleaning"""
        try:
            df = pd.read_csv(uploaded_file)
            
            # Validate required columns
            required_cols = ['Amount', 'Time']
            missing_cols = [col for col in required_cols if col not in df.columns]
            
            if missing_cols:
                st.error(f"Missing required columns: {', '.join(missing_cols)}")
                return None
            
            # Remove duplicates
            initial_rows = len(df)
            df = df.drop_duplicates()
            
            # Handle missing values
            df = df.dropna(subset=['Amount', 'Time'])
            
            # Convert to numeric
            df['Amount'] = pd.to_numeric(df['Amount'], errors='coerce')
            df['Time'] = pd.to_numeric(df['Time'], errors='coerce')
            
            # Remove rows with NaN after conversion
            df = df.dropna(subset=['Amount', 'Time'])
            
            cleaned_rows = len(df)
            
            if cleaned_rows < initial_rows:
                st.info(f"🧹 Cleaned data: Removed {initial_rows - cleaned_rows} rows (duplicates/missing values)")
            
            return df
            
        except Exception as e:
            st.error(f"Error loading data: {str(e)}")
            return None
    
    @staticmethod
    def add_transaction_id(df):
        """Add unique transaction ID if not present"""
        if 'TransactionID' not in df.columns:
            df.insert(0, 'TransactionID', range(1, len(df) + 1))
        return df

# ========================================
# FRAUD DETECTION MODEL
# ========================================

class FraudDetector:
    """Isolation Forest-based anomaly detection"""
    
    def __init__(self, contamination=0.1, random_state=42):
        self.contamination = contamination
        self.model = IsolationForest(
            contamination=contamination,
            random_state=random_state,
            n_estimators=100
        )
        self.scaler = StandardScaler()
        
    def fit_predict(self, df):
        """Fit model and predict anomalies"""
        # Prepare features
        features = df[['Amount', 'Time']].values
        
        # Scale features
        features_scaled = self.scaler.fit_transform(features)
        
        # Predict (-1 for anomalies, 1 for normal)
        predictions = self.model.fit_predict(features_scaled)
        
        # Get anomaly scores (lower = more anomalous)
        scores = self.model.score_samples(features_scaled)
        
        # Add predictions to dataframe
        df['Prediction'] = predictions
        df['AnomalyScore'] = scores
        df['IsFraud'] = df['Prediction'].apply(lambda x: 'Fraud' if x == -1 else 'Normal')
        
        return df

# ========================================
# AI EXPLANATION GENERATOR
# ========================================

class AIExplainer:
    """Generate natural language explanations using Gemini"""
    
    def __init__(self, model):
        self.model = model
    
    def explain_fraud(self, transaction_row, df_stats):
        """Generate one-sentence explanation for flagged transaction"""
        if self.model is None:
            return "AI explanation unavailable (API key not configured)"
        
        try:
            # Prepare context
            prompt = f"""You are a fraud detection analyst. Explain in ONE SENTENCE why this transaction looks suspicious:

Transaction Details:
- Amount: ${transaction_row['Amount']:.2f}
- Time: {transaction_row['Time']} seconds
- Anomaly Score: {transaction_row['AnomalyScore']:.4f}

Dataset Statistics:
- Average Amount: ${df_stats['avg_amount']:.2f}
- Std Dev Amount: ${df_stats['std_amount']:.2f}
- Average Time: {df_stats['avg_time']:.2f}
- Max Amount: ${df_stats['max_amount']:.2f}

Provide a clear, professional explanation in one sentence focusing on what makes this transaction unusual."""

            response = self.model.generate_content(prompt)
            return response.text.strip()
            
        except Exception as e:
            return f"Unable to generate explanation: {str(e)}"

# ========================================
# FIREBASE DATABASE HANDLER
# ========================================

class FirebaseHandler:
    """Handle Firestore operations for fraud cases"""
    
    def __init__(self, db):
        self.db = db
    
    def save_fraud_case(self, transaction_row, explanation):
        """Save flagged fraud case to Firestore"""
        if self.db is None:
            return False
            
        try:
            doc_ref = self.db.collection('fraud_cases').document()
            
            fraud_data = {
                'transaction_id': str(transaction_row.get('TransactionID', 'N/A')),
                'amount': float(transaction_row['Amount']),
                'time': float(transaction_row['Time']),
                'anomaly_score': float(transaction_row['AnomalyScore']),
                'explanation': explanation,
                'detected_at': datetime.now(),
                'status': 'pending_review'
            }
            
            # Add any additional columns from the transaction
            for col in transaction_row.index:
                if col not in ['TransactionID', 'Amount', 'Time', 'AnomalyScore', 
                              'Prediction', 'IsFraud']:
                    fraud_data[col.lower()] = str(transaction_row[col])
            
            doc_ref.set(fraud_data)
            return True
            
        except Exception as e:
            st.error(f"Error saving to Firebase: {str(e)}")
            return False
    
    def save_batch_fraud_cases(self, fraud_df, explanations):
        """Save multiple fraud cases in batch"""
        if self.db is None:
            return 0
            
        saved_count = 0
        for idx, row in fraud_df.iterrows():
            explanation = explanations.get(idx, "No explanation available")
            if self.save_fraud_case(row, explanation):
                saved_count += 1
        
        return saved_count

# ========================================
# VISUALIZATION
# ========================================

class Visualizer:
    """Create interactive visualizations"""
    
    @staticmethod
    def create_scatter_plot(df):
        """Create scatter plot with anomalies highlighted"""
        fig = px.scatter(
            df,
            x='Time',
            y='Amount',
            color='IsFraud',
            color_discrete_map={'Normal': '#4CAF50', 'Fraud': '#F44336'},
            hover_data=['TransactionID', 'AnomalyScore'],
            title='Transaction Analysis: Amount vs Time',
            labels={'Time': 'Time (seconds)', 'Amount': 'Amount ($)'}
        )
        
        fig.update_layout(
            plot_bgcolor='white',
            paper_bgcolor='white',
            font=dict(family="Arial, sans-serif", size=12),
            height=500,
            xaxis=dict(gridcolor='#E0E0E0'),
            yaxis=dict(gridcolor='#E0E0E0')
        )
        
        return fig
    
    @staticmethod
    def create_distribution_plot(df):
        """Create distribution plot for amounts"""
        fig = go.Figure()
        
        # Normal transactions
        normal_data = df[df['IsFraud'] == 'Normal']['Amount']
        fraud_data = df[df['IsFraud'] == 'Fraud']['Amount']
        
        fig.add_trace(go.Histogram(
            x=normal_data,
            name='Normal',
            marker_color='#4CAF50',
            opacity=0.7
        ))
        
        fig.add_trace(go.Histogram(
            x=fraud_data,
            name='Fraud',
            marker_color='#F44336',
            opacity=0.7
        ))
        
        fig.update_layout(
            title='Transaction Amount Distribution',
            xaxis_title='Amount ($)',
            yaxis_title='Frequency',
            barmode='overlay',
            plot_bgcolor='white',
            paper_bgcolor='white',
            height=400
        )
        
        return fig

# ========================================
# MAIN APPLICATION
# ========================================

def main():
    # Header
    st.title("🔒 Fraud Detection System")
    st.markdown("**AI-Powered Transaction Anomaly Detection**")
    st.markdown("---")
    
    # ========================================
    # SIDEBAR CONFIGURATION
    # ========================================
    
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # API Keys
        st.subheader("API Keys")
        gemini_api_key = st.text_input(
            "Gemini API Key",
            type="password",
            help="Enter your Google Gemini API key for AI explanations"
        )
        
        # Model Parameters
        st.subheader("Model Settings")
        contamination = st.slider(
            "Contamination Rate",
            min_value=0.01,
            max_value=0.30,
            value=0.10,
            step=0.01,
            help="Expected proportion of anomalies (10% default)"
        )
        
        # File Upload
        st.subheader("Data Upload")
        uploaded_file = st.file_uploader(
            "Upload Transaction CSV",
            type=['csv'],
            help="CSV must contain 'Amount' and 'Time' columns"
        )
        
        # Firebase Toggle
        use_firebase = st.checkbox(
            "Save to Firebase",
            value=True,
            help="Store fraud cases in Firestore database"
        )
        
        st.markdown("---")
        st.info("📊 Upload your transaction data to begin analysis")
    
    # ========================================
    # MAIN CONTENT AREA
    # ========================================
    
    if uploaded_file is not None:
        # Load and clean data
        with st.spinner("📥 Loading and cleaning data..."):
            processor = DataProcessor()
            df = processor.load_and_clean_data(uploaded_file)
        
        if df is not None:
            df = processor.add_transaction_id(df)
            
            # Display data preview
            st.subheader("📋 Data Preview")
            st.dataframe(df.head(10), use_container_width=True)
            
            # Run fraud detection
            with st.spinner("🔍 Detecting anomalies..."):
                detector = FraudDetector(contamination=contamination)
                df_with_predictions = detector.fit_predict(df)
            
            # Calculate statistics
            total_transactions = len(df_with_predictions)
            fraud_count = len(df_with_predictions[df_with_predictions['IsFraud'] == 'Fraud'])
            fraud_percentage = (fraud_count / total_transactions) * 100
            
            # Metrics
            st.markdown("---")
            st.subheader("📊 Detection Summary")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Transactions", f"{total_transactions:,}")
            with col2:
                st.metric("Fraud Detected", f"{fraud_count:,}", delta=f"{fraud_percentage:.1f}%")
            with col3:
                st.metric("Normal Transactions", f"{total_transactions - fraud_count:,}")
            with col4:
                avg_fraud_amount = df_with_predictions[df_with_predictions['IsFraud'] == 'Fraud']['Amount'].mean()
                st.metric("Avg Fraud Amount", f"${avg_fraud_amount:,.2f}")
            
            # Visualizations
            st.markdown("---")
            st.subheader("📈 Visualizations")
            
            col1, col2 = st.columns(2)
            
            with col1:
                scatter_fig = Visualizer.create_scatter_plot(df_with_predictions)
                st.plotly_chart(scatter_fig, use_container_width=True)
            
            with col2:
                dist_fig = Visualizer.create_distribution_plot(df_with_predictions)
                st.plotly_chart(dist_fig, use_container_width=True)
            
            # Fraud Cases Analysis
            st.markdown("---")
            st.subheader("🚨 Flagged Fraud Cases")
            
            fraud_cases = df_with_predictions[df_with_predictions['IsFraud'] == 'Fraud'].copy()
            fraud_cases = fraud_cases.sort_values('AnomalyScore', ascending=True)
            
            if len(fraud_cases) > 0:
                # Initialize Gemini if API key provided
                gemini_model = None
                if gemini_api_key:
                    gemini_model = initialize_gemini(gemini_api_key)
                    explainer = AIExplainer(gemini_model)
                    
                    # Generate explanations
                    df_stats = {
                        'avg_amount': df_with_predictions['Amount'].mean(),
                        'std_amount': df_with_predictions['Amount'].std(),
                        'avg_time': df_with_predictions['Time'].mean(),
                        'max_amount': df_with_predictions['Amount'].max()
                    }
                    
                    explanations = {}
                    with st.spinner("🤖 Generating AI explanations..."):
                        for idx, row in fraud_cases.head(10).iterrows():
                            explanations[idx] = explainer.explain_fraud(row, df_stats)
                            fraud_cases.loc[idx, 'AI_Explanation'] = explanations[idx]
                else:
                    st.warning("⚠️ Enter Gemini API key for AI explanations")
                
                # Display fraud cases
                display_cols = ['TransactionID', 'Amount', 'Time', 'AnomalyScore']
                if 'AI_Explanation' in fraud_cases.columns:
                    display_cols.append('AI_Explanation')
                
                st.dataframe(
                    fraud_cases[display_cols].head(20),
                    use_container_width=True,
                    height=400
                )
                
                # Firebase Integration
                if use_firebase and gemini_api_key:
                    st.markdown("---")
                    st.subheader("💾 Database Storage")
                    
                    if st.button("Save Fraud Cases to Firebase", type="primary"):
                        with st.spinner("Saving to Firebase..."):
                            db = initialize_firebase()
                            if db:
                                handler = FirebaseHandler(db)
                                saved_count = handler.save_batch_fraud_cases(
                                    fraud_cases.head(10),
                                    explanations
                                )
                                st.success(f"✅ Successfully saved {saved_count} fraud cases to Firebase!")
                            else:
                                st.error("❌ Firebase not configured. Please add credentials.")
                
                # Download Results
                st.markdown("---")
                csv = fraud_cases.to_csv(index=False)
                st.download_button(
                    label="📥 Download Fraud Report (CSV)",
                    data=csv,
                    file_name=f"fraud_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
                
            else:
                st.success("✅ No fraudulent transactions detected!")
                
    else:
        # Welcome Screen
        st.info("""
        ### Welcome to the Fraud Detection System
        
        **Getting Started:**
        1. **Upload your transaction CSV** file in the sidebar (must contain 'Amount' and 'Time' columns)
        2. **Configure the model** parameters (contamination rate)
        3. **Enter your Gemini API key** for AI-powered explanations
        4. **Optionally configure Firebase** for storing fraud cases
        
        The system will automatically detect anomalies and provide detailed insights!
        """)
        
        # Sample data format
        st.subheader("📄 Expected CSV Format")
        sample_data = pd.DataFrame({
            'TransactionID': [1, 2, 3, 4, 5],
            'Amount': [45.23, 1250.00, 32.50, 8900.00, 67.89],
            'Time': [120, 350, 480, 620, 750],
            'Merchant': ['Amazon', 'BestBuy', 'Walmart', 'Luxury Store', 'Target']
        })
        st.dataframe(sample_data, use_container_width=True)

if __name__ == "__main__":
    main()