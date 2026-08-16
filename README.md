# HeartCare AI Health System

AI-powered web application for heart disease risk prediction using Machine Learning.

## Overview
HeartCare AI Health System is a Flask-based web application that predicts a user's heart disease risk using clinical input data. The system uses a trained Random Forest Classifier and displays results through an interactive healthcare dashboard.

## Features
- Heart disease risk prediction with probability score
- Prediction history tracking (SQLite database)
- Dashboard with risk distribution overview
- Statistics and reports pages
- Profile and settings management

## Machine Learning Pipeline
1. Dataset loading and duplicate removal
2. Exploratory Data Analysis (statistics, missing values, correlation heatmap)
3. Train/test split
4. Feature scaling using StandardScaler
5. Model training and comparison:
   - Logistic Regression
   - Decision Tree
   - Random Forest ⭐ (Final Model)
6. Model and scaler saved using Joblib

## Technology Stack
- **Backend:** Python, Flask
- **Frontend:** HTML, CSS
- **Machine Learning:** Scikit-learn
- **Data Handling:** Pandas, NumPy
- **Visualization:** Matplotlib, Seaborn
- **Database:** SQLite

## Project Structure
HeartCare-AI-Health-System/
├── app.py
├── heart_prediction.py
├── models/
│ ├── heart_model.pkl
│ └── scaler.pkl
├── data/
│ └── data.csv
├── templates/
├── static/
├── heart_history.db
└── requirements.txt

## Installation & Setup
```bash
git clone https://github.com/nousheenfazal113-ctrl/HeartCare-AI-Health-System.git
cd HeartCare-AI-Health-System
pip install -r requirements.txt
python app.py
```
Then open `http://127.0.0.1:5000` in your browser.

## Medical Disclaimer
This project is an educational prototype and is **not** intended to replace professional medical diagnosis or advice.