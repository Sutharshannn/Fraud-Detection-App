# 🔒 Fraud Detection Web Application

A professional, AI-powered fraud detection system built with Streamlit, scikit-learn, and Google Gemini AI for hackathons and real-world applications.

## 🌟 Features

- **Anomaly Detection**: Uses Isolation Forest ML algorithm to detect fraudulent transactions
- **AI Explanations**: Integrates Google Gemini API for natural language fraud explanations
- **Cloud Storage**: Firebase Firestore integration for storing flagged cases
- **Interactive Visualizations**: Plotly charts for transaction analysis
- **Professional UI**: Clean, financial-grade dashboard interface
- **Real-time Processing**: Upload CSV and get instant results

## 🚀 Quick Start

### 1. Installation

```bash
# Clone or download the project
cd fraud-detection-app

# Install dependencies
pip install -r requirements.txt
```

### 2. Configuration

#### Option A: Gemini API Key (Required for AI Explanations)
1. Get your free API key from [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Enter it in the sidebar when you run the app

#### Option B: Firebase Setup (Optional - for database storage)

**Method 1: Service Account File**
1. Go to [Firebase Console](https://console.firebase.google.com/)
2. Create a new project or use existing
3. Go to Project Settings → Service Accounts
4. Click "Generate New Private Key"
5. Save the JSON file as `firebase-credentials.json` in the app directory

**Method 2: Environment Variable**
```bash
export FIREBASE_CREDENTIALS='{"type":"service_account","project_id":"your-project",...}'
```

### 3. Run the Application

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

## 📊 Using the App

### Step 1: Prepare Your Data
Your CSV file must contain at least these columns:
- `Amount`: Transaction amount (numeric)
- `Time`: Transaction timestamp in seconds (numeric)

Optional columns (will be preserved):
- `TransactionID`: Unique identifier
- `Merchant`: Merchant name
- Any other transaction metadata

### Step 2: Configure Detection
- **Contamination Rate**: Set the expected fraud percentage (default: 10%)
- Lower values = stricter detection
- Higher values = more transactions flagged

### Step 3: Upload & Analyze
1. Upload your CSV file
2. The system will automatically:
   - Clean and validate data
   - Detect anomalies using Isolation Forest
   - Generate visualizations
   - Provide AI explanations (if API key configured)

### Step 4: Review Results
- View summary metrics
- Explore interactive charts
- Review flagged transactions with AI explanations
- Download fraud report
- Save to Firebase (optional)

## 🏗️ Architecture

### Data Flow
```
CSV Upload → Data Cleaning → Feature Scaling → Isolation Forest → 
Anomaly Detection → AI Explanation → Visualization → Firebase Storage
```

### Key Components

#### 1. **DataProcessor**
```python
# Handles CSV upload and cleaning
- Removes duplicates
- Handles missing values
- Validates required columns
- Converts data types
```

#### 2. **FraudDetector**
```python
# ML-based anomaly detection
- Uses IsolationForest algorithm
- StandardScaler for feature normalization
- Generates anomaly scores
- Configurable contamination rate
```

#### 3. **AIExplainer**
```python
# Gemini API integration
- Generates natural language explanations
- Context-aware analysis
- One-sentence summaries
- Professional tone
```

#### 4. **FirebaseHandler**
```python
# Cloud database integration
- Batch saving of fraud cases
- Timestamp tracking
- Metadata preservation
- Error handling
```

#### 5. **Visualizer**
```python
# Interactive charts
- Scatter plot (Time vs Amount)
- Distribution histograms
- Color-coded fraud indicators
- Hover tooltips
```

## 📁 Project Structure

```
fraud-detection-app/
│
├── app.py                      # Main Streamlit application
├── requirements.txt            # Python dependencies
├── firebase-credentials.json   # Firebase service account (optional)
├── sample_transactions.csv     # Sample data for testing
└── README.md                   # This file
```

## 🔧 Customization

### Adjust Model Sensitivity
In the sidebar, modify the **Contamination Rate**:
- **0.01-0.05**: Very strict (1-5% flagged)
- **0.10**: Balanced (10% flagged) - **DEFAULT**
- **0.15-0.30**: Aggressive (15-30% flagged)

### Modify Features
Edit the feature selection in `FraudDetector.fit_predict()`:
```python
# Current features
features = df[['Amount', 'Time']].values

# Add more features
features = df[['Amount', 'Time', 'Merchant_Encoded', 'Location']].values
```

### Customize AI Prompts
Edit the prompt in `AIExplainer.explain_fraud()` to change explanation style:
```python
prompt = f"""Your custom prompt here..."""
```

## 📈 Sample Data Format

```csv
TransactionID,Amount,Time,Merchant,Location
1,45.23,120,Amazon,Online
2,1250.00,350,BestBuy,NYC
3,32.50,480,Walmart,LA
4,8900.00,620,Luxury Store,Miami
5,67.89,750,Target,Chicago
```

## 🐛 Troubleshooting

### "Missing required columns" Error
- Ensure your CSV has `Amount` and `Time` columns
- Check for typos in column names (case-sensitive)

### Gemini API Errors
- Verify API key is correct
- Check [quota limits](https://ai.google.dev/pricing)
- Ensure you're using a valid Gemini model

### Firebase Connection Issues
- Verify `firebase-credentials.json` is in the correct location
- Check Firebase project permissions
- Ensure Firestore is enabled in Firebase console

### Visualization Not Showing
- Clear browser cache
- Update Plotly: `pip install --upgrade plotly`
- Check browser console for JavaScript errors

## 💡 Tips for Hackathons

1. **Demo Preparation**: Use the sample data to create compelling visualizations
2. **Story Telling**: Focus on the AI explanation feature - it's unique
3. **Scalability**: Mention batch processing capabilities
4. **Security**: Highlight the professional financial UI design
5. **Extensions**: Suggest future features (real-time monitoring, email alerts)

## 🔐 Security Notes

- **Never commit** `firebase-credentials.json` to version control
- Add to `.gitignore`:
```
firebase-credentials.json
*.env
.streamlit/secrets.toml
```
- Use environment variables for production deployments

## 📚 Additional Resources

- [Streamlit Documentation](https://docs.streamlit.io/)
- [scikit-learn Isolation Forest](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.IsolationForest.html)
- [Google Gemini API](https://ai.google.dev/)
- [Firebase Admin SDK](https://firebase.google.com/docs/admin/setup)
- [Plotly Python](https://plotly.com/python/)

## 🎯 Future Enhancements

- [ ] Real-time transaction monitoring
- [ ] Email/SMS alerts for high-risk transactions
- [ ] Multi-model ensemble (Random Forest + Isolation Forest)
- [ ] User authentication and role-based access
- [ ] Historical trend analysis
- [ ] Export to PDF reports
- [ ] Integration with payment gateways

## 📄 License

This project is open source and available for hackathon use.

## 🤝 Contributing

Feel free to fork, modify, and enhance this project for your hackathon needs!

---

**Built for hackathons with ❤️**

Good luck with your demo! 🚀
