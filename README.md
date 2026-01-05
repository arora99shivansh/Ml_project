🎓 Student Performance Prediction – End-to-End ML Project

An end-to-end Machine Learning project that predicts a student’s Math score based on demographic details and academic performance.
The project includes data ingestion from MySQL, data preprocessing, model training, and deployment using Streamlit.

🚀 Features:
📥 Data ingestion from MySQL database
🧹 Data preprocessing using Scikit-Learn Pipelines
🤖 Multiple ML models with hyperparameter tuning
🏆 Automatic best model selection
💾 Saved artifacts:
        model.pkl
        preprocessor.pkl
📊 Model evaluation using R² Score
🌐 Streamlit web app for live predictions
🧱 Modular, production-ready project structure
🪵 Centralized logging & custom exception handling

Machine Learning Workflow:
Data Ingestion
1)Reads data from MySQL
2)Splits into train & test datasets

Data Transformation
1)Numerical: Median Imputation + Standard Scaling
2)Categorical: Mode Imputation + One-Hot Encoding

Saved as preprocessor.pkl

Model Training
  Models used:
    Linear Regression
    Random Forest
    Gradient Boosting
    Decision Tree
    XGBoost
    CatBoost
    AdaBoost

Best model selected based on R² Score
Saved as model.pkl

Deployment
Streamlit UI for real-time predictions

🌐 Run Streamlit App
streamlit run streamlit_app.py


Model Performance
Best Model: Linear Regression
R² Score: ~0.88

<img width="1903" height="907" alt="image" src="https://github.com/user-attachments/assets/90697304-e382-4527-bd8c-a75b7f2a48b6" />

