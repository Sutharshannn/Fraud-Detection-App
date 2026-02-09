# 🛡️ AI-Powered Fraud Detection System

**Fraud-Guard** is a robust financial security application that bridges the gap between unsupervised machine learning and human-readable intelligence. By leveraging **Gemini 2.5 Flash**, it transforms complex anomaly scores into actionable forensic insights.

---

## 🚀 Key Features

* **Isolation Forest Engine**: Employs unsupervised machine learning to detect high-risk outliers in transaction datasets based on Amount and Time features.
* **Explainable AI (XAI)**: Utilizes Gemini 2.5 Flash to generate professional, one-sentence natural language explanations for every flagged transaction.
* **Batch-Optimized Inference**: Implements custom batch processing to bundle multiple detections into a single API request, maximizing performance within free-tier rate limits.
* **Sensitivity Calibration**: Features an interactive contamination slider that allows analysts to adjust the model's fraud detection threshold in real-time.
* **Cloud Persistence**: Integrated with Firebase Firestore for secure, cloud-based storage and status tracking of all detected fraud cases.

---

## 🛠️ Built With

* **Python**
* **Streamlit**
* **Google Gemini 2.5 Flash API**
* **Scikit-learn**
* **Firebase Firestore**
* **Plotly**

---

## 🔬 How It Works

**1. Data Ingestion** Users upload a standard transaction CSV containing Amount and Time columns for analysis.

**2. Anomaly Detection** The Isolation Forest algorithm isolates outliers by randomly splitting features; transactions requiring fewer splits receive higher priority scores.



**3. Batch Explanation** Flagged cases are bundled into a JSON-structured batch and sent to the AI for rapid, contextual analysis against dataset statistics.



**4. Persistence** Verified fraud cases are saved to the Firebase database with their AI-generated explanations for follow-up investigation.

---

## ⚙️ Installation and Setup

**1. Clone the Repository** Use your terminal to clone the project files and enter the project directory.

**2. Install Dependencies** Install the required libraries including scikit-learn, streamlit, and the Google GenAI SDK.

**3. Configure Environment Variables** Create a .env file in the root folder and add your GEMINI_API_KEY.

**4. Run the Application** Execute the Streamlit run command to launch the interactive fraud dashboard.

---

## 🎓 Academic Context
Developed as a specialized project during the University of Windsor hackathon series in early 2026.

**Author:** Sutharshan Suthakaran, Computer Science Student at the University of Windsor.

---
*Disclaimer: This tool is intended for ethical financial research and educational purposes only.*
