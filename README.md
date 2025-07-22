```markdown
# 💰 Pay Predict

AI-powered salary prediction tailored to the Indian tech market. Built with **Streamlit**, **scikit-learn** and an interactive dark-orange UI.

---

## 🚀 Live Demo
Run the app locally (instructions below) and open it in your browser; Streamlit starts on `localhost:8501` by default.

---

## ✨ Features
- **89 % R² accuracy** via Gradient Boosting Regressor.  
- **Real-time results** on every input change.  
- **Dark + orange theme** with mobile-friendly layout.  
- **Market benchmarks** & interactive Plotly charts (job, city, education).  

---

## 🏗️ Project Layout
```
pay-predict/
├── app_streamlit.py         # Streamlit front-end
├── train_model.py           # Model training pipeline
├── requirements.txt         # Python dependencies
├── model.joblib             # Trained model (generated)
├── label_encoders.joblib    # Encoders (generated)
└── indian_salary_data_500.csv
```

---

## ⚙️ Tech Stack
| Layer            | Tools / Libraries |
|------------------|-------------------|
| User Interface   | Streamlit         |
| Machine Learning | scikit-learn (Gradient Boosting) |
| Data Handling    | pandas · numpy    |
| Visualisation    | Plotly            |
| Persistence      | joblib            |

---

## 🛠️ Quick Start
```
# 1. Clone repository
git clone 
cd pay-predict

# 2. Install dependencies
pip install -r requirements.txt

# 3. Add dataset
#   Place indian_salary_data_500.csv in the project root

# 4. Train the model (generates model.joblib & label_encoders.joblib)
python train_model.py

# 5. Run the application
streamlit run app_streamlit.py
```

---

## 📈 Dataset
Indian software-sector salary survey with  
• 500+ records • 8 input features (age, gender, education, experience, job title, location, city, nationality)  
• Target: annual salary in ₹ lakhs.

---

## 🤖 Model
GradientBoostingRegressor  
• 200 estimators • max_depth 6 • learning_rate 0.1  
Performance on held-out test set: **R² ≈ 0.89**.

---

## 📝 License
MIT © 2025 Pay Predict
```