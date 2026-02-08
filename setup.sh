#!/bin/bash

# Fraud Detection App - Quick Setup Script
# This script helps you set up the application quickly

echo "🔒 Fraud Detection App - Setup Script"
echo "======================================"
echo ""

# Check Python version
echo "📋 Checking Python version..."
python_version=$(python3 --version 2>&1)
if [[ $? -eq 0 ]]; then
    echo "✅ $python_version detected"
else
    echo "❌ Python 3 is required but not found"
    echo "Please install Python 3.8 or higher from https://www.python.org/"
    exit 1
fi

echo ""

# Create virtual environment
echo "🔧 Creating virtual environment..."
if [ -d "venv" ]; then
    echo "⚠️  Virtual environment already exists. Skipping..."
else
    python3 -m venv venv
    echo "✅ Virtual environment created"
fi

echo ""

# Activate virtual environment
echo "🔌 Activating virtual environment..."
source venv/bin/activate || . venv/Scripts/activate

echo ""

# Install dependencies
echo "📦 Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

if [[ $? -eq 0 ]]; then
    echo "✅ All dependencies installed successfully"
else
    echo "❌ Error installing dependencies"
    exit 1
fi

echo ""
echo "======================================"
echo "✅ Setup Complete!"
echo ""
echo "📝 Next Steps:"
echo ""
echo "1. Get your Gemini API key:"
echo "   Visit: https://makersuite.google.com/app/apikey"
echo ""
echo "2. (Optional) Set up Firebase:"
echo "   - Create project at: https://console.firebase.google.com/"
echo "   - Download credentials and save as 'firebase-credentials.json'"
echo ""
echo "3. Run the application:"
echo "   streamlit run app.py"
echo ""
echo "4. Upload the sample data:"
echo "   Use 'sample_transactions.csv' to test the app"
echo ""
echo "======================================"
echo "🚀 Ready to detect fraud!"
echo ""
