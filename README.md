# 🔒 AI-Powered Fraud Detection System

**Tagline:** Bridging Machine Learning and Generative AI for Intelligent, Explainable Fraud Analysis.

---

## 📌 Project Overview
This project is a full-stack anomaly detection platform designed to identify and explain suspicious financial transactions in real-time. By combining **Unsupervised Machine Learning** with **Generative AI**, the system not only flags potential fraud but provides natural language explanations for why a transaction was deemed suspicious, making the data actionable for human analysts.

### 🚀 Key Features
* **Anomaly Detection**: Uses the **Isolation Forest** algorithm to detect outliers in transaction datasets based on Amount and Time features.
* **Explainable AI (XAI)**: Integrates **Gemini 2.5 Flash** to generate professional, one-sentence explanations for every flagged case.
* **Batch Processing Optimization**: Implements custom batching logic to process multiple transactions in a single API call, staying within Free Tier rate limits.
* **Interactive Dashboard**: A **Streamlit**-based UI featuring real-time sensitivity sliders and Plotly data visualizations.
* **Cloud Persistence**: Seamless integration with **Firebase Firestore** to save and track fraudulent cases for audit.

---

## 🛠️ Built With
* **Streamlit** (Frontend UI)
* **Gemini 2.5 Flash** (Generative AI)
* **Scikit-learn** (Isolation Forest ML Model)
* **Firebase Firestore** (NoSQL Database)
* **Plotly** (Data Visualization)
* **Pandas & NumPy** (Data Manipulation)
* **Tenacity** (Exponential Backoff & API Resilience)

---

## 🔬 How It Works

### 1. Isolation Forest Logic
The system identifies fraud by measuring how easily a transaction can be "isolated" from the rest of the group. Points that require fewer splits to isolate are assigned lower anomaly scores.



### 2. Batch Processing (API Efficiency)
To maximize the Gemini Free Tier (15 RPM), the app bundles up to 10 detected cases into a single JSON-structured prompt. This reduces API overhead and ensures a smooth user experience.



---

## ⚙️ Installation & Setup

1. **Clone the Repository**
   ```bash
   git clone [https://github.com/Sutharshannn/Fraud-Detection-App.git](https://github.com/Sutharshannn/Fraud-Detection-App.git)
   cd Fraud-Detection-App
