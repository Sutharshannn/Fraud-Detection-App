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
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

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
# HARDCODED API KEY (For Demo/Hackathon)
# ========================================
# Load API key from .env file (more secure than hardcoding)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

# Validate API key is loaded
if not GEMINI_API_KEY:
    print("⚠️ WARNING: GEMINI_API_KEY not found in .env file!")
    print("Create a .env file with: GEMINI_API_KEY=your_api_key_here")

# ========================================
# CUSTOM STYLING
# ========================================

st.markdown("""
    <style>
    .main {
        background-color: #0e1117;
        padding-top: 0 !important;
    }
    .stAlert {
        background-color: #fff3cd;
        border-left: 4px solid #ffc107;
    }
    
    /* Sticky Header Container */
    .sticky-header {
        position: fixed;
        top: 0;
        left: 0;
        right: 0;
        z-index: 999;
        background: linear-gradient(180deg, #0e1117 0%, #0e1117 85%, rgba(14, 17, 23, 0) 100%);
        padding: 1.5rem 2rem 2rem 2rem;
        text-align: center;
    }
    
    .header-spacer {
        height: 180px;
    }
    
    h1 {
        color: #ffffff;
        font-weight: 700;
        text-align: center;
        font-size: 3rem !important;
        margin-bottom: 0.3rem !important;
        margin-top: 0 !important;
    }
    h2 {
        color: #a0a0a0;
        font-weight: 400;
        text-align: center;
        font-size: 1.1rem !important;
        margin-bottom: 0 !important;
        margin-top: 0 !important;
    }
    h3 {
        color: #ffffff;
        text-align: center;
        margin-top: 1rem;
        margin-bottom: 0.5rem;
        font-size: 1.3rem !important;
    }
    .metric-card {
        background-color: #1e1e1e;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.3);
    }
    .compact-section {
        margin-bottom: 1rem;
    }
    .stButton>button {
        width: 100%;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        font-size: 1.1rem;
        border-radius: 10px;
        font-weight: 600;
        transition: transform 0.2s;
    }
    .stButton>button:hover {
        transform: scale(1.05);
        box-shadow: 0 5px 20px rgba(102, 126, 234, 0.4);
    }
    div[data-testid="stFileUploader"] {
        background-color: rgba(255, 255, 255, 0.05);
        border: 2px dashed rgba(255, 255, 255, 0.2);
        border-radius: 15px;
        padding: 1.5rem;
    }
    .stSlider {
        padding: 0.5rem 0;
    }
    .stTextInput > div > div {
        background-color: rgba(255, 255, 255, 0.05);
    }
    /* Hide sidebar by default */
    [data-testid="stSidebar"] {
        display: none;
    }
    /* Compact labels */
    .stTextInput label, .stSlider label {
        font-size: 0.9rem !important;
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
        return genai.GenerativeModel('gemini-2.5-flash')  # Updated to latest model
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
    # Sticky Header
    st.markdown("""
        <div class="sticky-header">
            <h1>🔒 Fraud Detection System</h1>
            <h2>AI-Powered Transaction Anomaly Detection</h2>
        </div>
    """, unsafe_allow_html=True)
    
    # Spacer to prevent content from hiding under sticky header
    st.markdown('<div class="header-spacer"></div>', unsafe_allow_html=True)
    
    # ========================================
    # CENTERED UPLOAD SECTION
    # ========================================
    
    # Create centered container
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        # Model Settings - More compact
        st.markdown("### ⚙️ Detection Sensitivity")
        contamination = st.slider(
            "Fraud Detection Rate",
            min_value=0.01,
            max_value=0.30,
            value=0.10,
            step=0.01,
            help="Percentage of transactions to flag as potential fraud"
        )
        st.caption(f"Currently set to detect ~{int(contamination*100)}% as potential fraud")
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # File Upload - Centered
        st.markdown("### 📂 Upload Transaction Data")
        uploaded_file = st.file_uploader(
            "Choose CSV file",
            type=['csv'],
            help="CSV must contain 'Amount' and 'Time' columns"
        )
        
        # Firebase Toggle - More compact
        use_firebase = st.checkbox(
            "💾 Save to Firebase (optional)",
            value=False,
            help="Requires Firebase credentials"
        )
        
        st.markdown("<br>", unsafe_allow_html=True)
    
    # ========================================
    # PROCESSING & RESULTS SECTION
    # ========================================
    
    if uploaded_file is not None:
        
        # Centered "Analyze" button
        with col2:
            analyze_button = st.button("🔍 Analyze Transactions", type="primary", use_container_width=True)
        
        if analyze_button or 'analyzed' in st.session_state:
            st.session_state['analyzed'] = True
            
            # Loading Animation - Centered
            st.markdown("<br><br>", unsafe_allow_html=True)
            
            with st.spinner(""):
                # Custom loading message
                loading_placeholder = st.empty()
                loading_placeholder.markdown("""
                    <div style='text-align: center; padding: 3rem;'>
                        <h2 style='color: #667eea; margin-bottom: 1rem;'>🔍 Analyzing Your Transactions...</h2>
                        <p style='color: #a0a0a0; font-size: 1.1rem;'>
                            Detecting anomalies using machine learning<br>
                            This may take a few moments
                        </p>
                    </div>
                """, unsafe_allow_html=True)
                
                # Load and clean data
                processor = DataProcessor()
                df = processor.load_and_clean_data(uploaded_file)
                
                if df is not None:
                    df = processor.add_transaction_id(df)
                    
                    # Run fraud detection
                    detector = FraudDetector(contamination=contamination)
                    df_with_predictions = detector.fit_predict(df)
                    
                    loading_placeholder.empty()
                    
                    # Success message
                    st.success("✅ Analysis Complete!")
                    
                    st.markdown("<br><br>", unsafe_allow_html=True)
                    st.markdown("<h2 style='text-align: center; color: white;'>📊 Analysis Results</h2>", unsafe_allow_html=True)
                    st.markdown("<p style='text-align: center; color: #a0a0a0; margin-bottom: 3rem;'>Scroll down to view detailed insights</p>", unsafe_allow_html=True)
                    
                    # Calculate statistics
                    total_transactions = len(df_with_predictions)
                    fraud_count = len(df_with_predictions[df_with_predictions['IsFraud'] == 'Fraud'])
                    fraud_percentage = (fraud_count / total_transactions) * 100
                    
                    # Metrics - Full Width
                    st.markdown("### 📈 Key Metrics")
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Total Transactions", f"{total_transactions:,}")
                    with col2:
                        st.metric("🚨 Fraud Detected", f"{fraud_count:,}", delta=f"{fraud_percentage:.1f}%", delta_color="inverse")
                    with col3:
                        st.metric("✅ Normal Transactions", f"{total_transactions - fraud_count:,}")
                    with col4:
                        avg_fraud_amount = df_with_predictions[df_with_predictions['IsFraud'] == 'Fraud']['Amount'].mean()
                        st.metric("Avg Fraud Amount", f"${avg_fraud_amount:,.2f}")
                    
                    st.markdown("<br>", unsafe_allow_html=True)
                    
                    # Data Preview
                    with st.expander("📋 View Raw Data Preview", expanded=False):
                        st.dataframe(df.head(20), use_container_width=True)
                    
                    st.markdown("<br>", unsafe_allow_html=True)
                    
                    # Visualizations
                    st.markdown("### 📊 Visual Analysis")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        scatter_fig = Visualizer.create_scatter_plot(df_with_predictions)
                        st.plotly_chart(scatter_fig, use_container_width=True)
                    
                    with col2:
                        dist_fig = Visualizer.create_distribution_plot(df_with_predictions)
                        st.plotly_chart(dist_fig, use_container_width=True)
                    
                    st.markdown("<br><br>", unsafe_allow_html=True)
                    
                    # Fraud Cases Analysis
                    st.markdown("### 🚨 Detected Fraud Cases")
                    
                    fraud_cases = df_with_predictions[df_with_predictions['IsFraud'] == 'Fraud'].copy()
                    fraud_cases = fraud_cases.sort_values('AnomalyScore', ascending=True)
                    
                    if len(fraud_cases) > 0:
                        # Use API key from environment variable
                        gemini_model = None
                        if GEMINI_API_KEY:
                            with st.spinner("🤖 Generating AI explanations for flagged transactions..."):
                                gemini_model = initialize_gemini(GEMINI_API_KEY)
                                explainer = AIExplainer(gemini_model)
                                
                                # Generate explanations
                                df_stats = {
                                    'avg_amount': df_with_predictions['Amount'].mean(),
                                    'std_amount': df_with_predictions['Amount'].std(),
                                    'avg_time': df_with_predictions['Time'].mean(),
                                    'max_amount': df_with_predictions['Amount'].max()
                                }
                                
                                explanations = {}
                                for idx, row in fraud_cases.head(10).iterrows():
                                    explanations[idx] = explainer.explain_fraud(row, df_stats)
                                    fraud_cases.loc[idx, 'AI_Explanation'] = explanations[idx]
                        else:
                            st.warning("⚠️ AI explanations unavailable - Configure GEMINI_API_KEY in .env file")
                            explanations = {}
                        
                        # Display fraud cases
                        display_cols = ['TransactionID', 'Amount', 'Time', 'AnomalyScore']
                        if 'AI_Explanation' in fraud_cases.columns:
                            display_cols.append('AI_Explanation')
                        
                        st.dataframe(
                            fraud_cases[display_cols].head(20),
                            use_container_width=True,
                            height=400
                        )
                        
                        st.markdown("<br>", unsafe_allow_html=True)
                        
                        # Action buttons in columns
                        btn_col1, btn_col2, btn_col3 = st.columns(3)
                        
                        with btn_col1:
                            # Download Results
                            csv = fraud_cases.to_csv(index=False)
                            st.download_button(
                                label="📥 Download Fraud Report",
                                data=csv,
                                file_name=f"fraud_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                mime="text/csv",
                                use_container_width=True
                            )
                        
                        with btn_col2:
                            # Firebase Integration
                            if use_firebase and GEMINI_API_KEY:
                                if st.button("💾 Save to Firebase", type="secondary", use_container_width=True):
                                    with st.spinner("Saving to database..."):
                                        db = initialize_firebase()
                                        if db:
                                            handler = FirebaseHandler(db)
                                            saved_count = handler.save_batch_fraud_cases(
                                                fraud_cases.head(10),
                                                explanations
                                            )
                                            st.success(f"✅ Saved {saved_count} cases to Firebase!")
                                        else:
                                            st.error("❌ Firebase credentials not found")
                        
                        with btn_col3:
                            if st.button("🔄 Analyze New File", use_container_width=True):
                                del st.session_state['analyzed']
                                st.rerun()
                        
                    else:
                        st.success("🎉 Great news! No fraudulent transactions detected in your dataset.")
                        
                        if st.button("🔄 Analyze Another File", use_container_width=True):
                            del st.session_state['analyzed']
                            st.rerun()
                
                else:
                    loading_placeholder.empty()
                    st.error("❌ Error processing the uploaded file. Please check the format.")
                    
    else:
        # Welcome Screen - Centered
        st.markdown("<br><br>", unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.info("""
            ### 📚 How It Works
            
            **Step 1:** Upload your transaction CSV file above
            
            **Step 2:** Adjust the detection sensitivity if needed
            
            **Step 3:** Click "Analyze Transactions"
            
            **Step 4:** Review results, AI explanations, and download report
            
            ---
            
            **Required CSV Format:** Must contain `Amount` and `Time` columns
            """)
            
            # Sample data format
            st.markdown("### 📄 Example CSV Format")
            sample_data = pd.DataFrame({
                'TransactionID': [1, 2, 3, 4, 5],
                'Amount': [45.23, 1250.00, 32.50, 8900.00, 67.89],
                'Time': [120, 350, 480, 620, 750],
                'Merchant': ['Amazon', 'BestBuy', 'Walmart', 'Luxury Store', 'Target']
            })
            st.dataframe(sample_data, use_container_width=True)

if __name__ == "__main__":
    main()