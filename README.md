# Heart Disease Risk Prediction & Patient Clustering

This project is a **Graduation Project** for predicting heart disease risk and clustering patients based on clinical features, using 70,000 patient records from the Cardiovascular Disease Dataset.

---

## User Stories & Requirements

### 1. Data Preprocessing
**As a data engineer at CardioCare Analytics**  
I want to build a robust data preprocessing pipeline for 70,000 patient health records  
So that we ensure high-quality, consistent data for all ML models

**Requirements:**
- Handle outliers in systolic/diastolic blood pressure (ap_hi > 250, ap_lo > 200)  
- Address class imbalance using SMOTE or stratified sampling  
- Encode categorical variables (cholesterol: 1=normal, 2=above normal, 3=well above normal)  
- Scale numerical features (age, height, weight) using RobustScaler  
- Create temporal splits to prevent data leakage  

### 2. Exploratory Data Analysis (EDA)
**As a medical researcher**  
I want to analyze patient records to identify cardiovascular risk patterns  

**Requirements:**
- Correlation analysis between all 12 clinical features and cardiovascular disease  
- Age distribution analysis with 5-year bins and disease prevalence  
- Blood pressure category visualization (normal, stage 1, stage 2 hypertension)  
- Interactive Plotly dashboards showing risk factor combinations  
- Statistical significance testing using chi-square and t-tests  

### 3. Supervised Machine Learning
**As a machine learning engineer**  
I want to build ensemble models that predict cardiovascular disease with 85%+ accuracy  

**Requirements:**
- Implement XGBoost with hyperparameter tuning (n_estimators=500, max_depth=8)  
- Build Random Forest with feature importance analysis  
- Train Logistic Regression with regularization for interpretability  
- Evaluate using stratified 5-fold cross-validation with AUC-ROC scores  
- Create calibration curves to assess prediction reliability  

### 4. Unsupervised Machine Learning
**As a healthcare strategist**  
I want to segment patients into clinically meaningful clusters  

**Requirements:**
- Apply KMeans clustering (k=6)  
- Use DBSCAN to detect outlier patients  
- Implement hierarchical clustering for nested patient groupings  
- Reduce dimensions with t-SNE for intuitive cluster visualization  
- Profile each cluster with comprehensive clinical statistics  

### 5. Deep Learning
**As an AI specialist**  
I want to develop deep neural networks that outperform traditional ML  

**Requirements:**
- Build DNN classifier with 5 hidden layers (256,128,64,32,16 neurons)  
- Implement wide & deep architecture combining raw features and embeddings  
- Use Autoencoders (12→8→4→8→12) for unsupervised feature learning  
- Apply batch normalization and dropout (0.3)  
- Compare performance against ensemble methods using statistical tests  

### 6. Streamlit Deployment
**As a full-stack developer**  
I want to create a production-ready web application serving patient insights  

**Requirements:**
- Patient data input form with 11 clinical parameters  
- Real-time cardiovascular risk prediction with confidence intervals  
- Interactive cluster assignment showing similar patient profiles  
- SHAP force plots explaining feature contributions  
- Batch processing capability for multiple patient assessments  

---

## Files
- `Graduation project.ipynb` : Main notebook with code and analysis  
- `cardio_train.csv` : Dataset used for training and analysis  

## Tools & Libraries
- Python 3, Jupyter Notebook  
- Pandas, NumPy, Scikit-learn, XGBoost  
- Plotly, Streamlit, SHAP  
