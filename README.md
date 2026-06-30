# 🏠 Smart House Price Prediction System

A Machine Learning web application that predicts house prices using the California Housing Dataset. The project is built with **Python**, **Scikit-Learn**, **FastAPI**, **Pickle**, **HTML**, **CSS**, **Bootstrap 5**, and **Chart.js**.

---

## 📌 Project Overview

This project demonstrates the complete Machine Learning deployment process:

- Data Loading
- Data Preprocessing
- Model Training
- Model Evaluation
- Model Serialization using Pickle
- FastAPI Backend
- HTML + CSS Frontend
- Dynamic Prediction Results
- Interactive Charts

Users can enter property details and Machine Learning features to receive an estimated house price.

---

## 🚀 Technologies Used

- Python 3.11+
- FastAPI
- Scikit-Learn
- Random Forest Regressor
- Pickle
- NumPy
- Pandas
- HTML5
- CSS3
- Bootstrap 5
- Chart.js
- Jinja2 Templates

---

## 📂 Project Structure

```
Mini_Project/

│
├── app.py
├── train_Model.py
├── house_price_model.pkl
│
├── templates/
│   └── index.html
│
├── static/
│   └── css/
│       └── style.css
│
└── README.md
```

---

## ⚙ Features

- Modern Responsive UI
- Customer Information Form
- Property Details Form
- Machine Learning Prediction
- Random Forest Regression Model
- Pickle Model Deployment
- Interactive Chart.js Visualization
- Beautiful Bootstrap Interface
- FastAPI Backend
- Jinja2 Template Rendering

---

## 🤖 Machine Learning Model

Algorithm Used:

- Random Forest Regressor

Dataset:

- California Housing Dataset
- Total Records: 20,640
- Features: 8

Features used for prediction:

- Median Income
- House Age
- Average Rooms
- Average Bedrooms
- Population
- Average Occupancy
- Latitude
- Longitude

---

## 📊 Model Performance

Example Results

| Metric | Value |
|---------|-------|
| Training Accuracy | 97% |
| Testing Accuracy | 80% |
| MAE | 0.327 |
| RMSE | 0.504 |
| R² Score | 0.806 |

---

## 💻 Installation

Clone the repository

```bash
git clone https://github.com/yourusername/house-price-prediction.git
```

Move into the project folder

```bash
cd house-price-prediction
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run the application

```bash
uvicorn app:app --reload
```

Open your browser

```
http://127.0.0.1:8000
```

---

## 📸 Application Workflow

1. Enter customer details.
2. Enter property details.
3. Provide Machine Learning input features.
4. Click **Predict House Price**.
5. View:
   - Estimated House Price
   - Customer Summary
   - Property Details
   - Interactive Chart

---

## 📈 Future Improvements

- Real Estate Dataset
- Database Integration
- User Authentication
- PDF Report Generation
- Model Comparison
- Feature Engineering
- Deep Learning Model
- Cloud Deployment

---

## 🎓 Academic Purpose

This project was developed as a **Bachelor of Computer Applications (BCA) AI & Data Analytics Mini Project** to demonstrate Machine Learning model deployment using FastAPI.

---

## 👩‍💻 Author

**Mantasha Anish Shaikh**

BCA (Artificial Intelligence & Data Analytics)

---

## 📜 License

This project is developed for educational purposes.
