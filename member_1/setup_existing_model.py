"""
استخدام موديل موجود وإنشاء الباقي
Complete Integration Script
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import joblib
import os

print("="*60)
print("Setting Up Models from Existing Model")
print("="*60)

# ==================== الخطوة 1: تحميل الموديل الموجود ====================
print("\n[Step 1/4] Loading existing model...")

# حدد مسار الموديل اللي عندك
EXISTING_MODEL_PATH =  "C:\\Users\\nadam\\Downloads\\-Heart-Disease-Risk-Prediction-Patient-Clustering-main (1)\\-Heart-Disease-Risk-Prediction-Patient-Clustering-main\\ml_model.ipynb" # 👈 غير المسار هنا!

try:
    existing_model = joblib.load(EXISTING_MODEL_PATH)
    print(f"Model loaded successfully from: {EXISTING_MODEL_PATH}")
    print(f"Model type: {type(existing_model).__name__}")
except FileNotFoundError:
    print(f"Model not found at: {EXISTING_MODEL_PATH}")
    print("Please update EXISTING_MODEL_PATH with your model location")
    print("   Example: 'models/my_model.pkl' or 'C:/path/to/model.pkl'")
    exit(1)
except Exception as e:
    print(f"Error loading model: {str(e)}")
    exit(1)

# ==================== الخطوة 2: فحص الموديل ====================
print("\n[Step 2/4] Inspecting model...")

# فحص نوع الموديل
model_type = type(existing_model).__name__
print(f"   Model type: {model_type}")

# محاولة الحصول على معلومات عن الـ features
try:
    if hasattr(existing_model, 'n_features_in_'):
        n_features = existing_model.n_features_in_
        print(f"   Expected features: {n_features}")
    
    if hasattr(existing_model, 'feature_names_in_'):
        feature_names = existing_model.feature_names_in_
        print(f"   Feature names: {list(feature_names)}")
except:
    print("Could not extract feature information")

# ==================== الخطوة 3: إنشاء Scaler ====================
print("\n[Step 3/4] Creating scaler...")

# إنشاء بيانات تجريبية لتدريب الـ scaler
np.random.seed(42)
n_samples = 300

# البيانات التجريبية بنفس الـ features المتوقعة
sample_data = pd.DataFrame({
    'age': np.random.randint(30, 80, n_samples),
    'sex': np.random.randint(0, 2, n_samples),
    'cp': np.random.randint(0, 4, n_samples),
    'trestbps': np.random.randint(90, 200, n_samples),
    'chol': np.random.randint(150, 400, n_samples),
    'fbs': np.random.randint(0, 2, n_samples),
    'restecg': np.random.randint(0, 3, n_samples),
    'thalach': np.random.randint(80, 200, n_samples),
    'exang': np.random.randint(0, 2, n_samples),
    'oldpeak': np.random.uniform(0, 6, n_samples),
    'slope': np.random.randint(0, 3, n_samples),
    'ca': np.random.randint(0, 4, n_samples),
    'thal': np.random.randint(0, 4, n_samples),
})

# تدريب scaler
scaler = StandardScaler()
scaled_data = scaler.fit_transform(sample_data)

# حفظ الـ scaler
os.makedirs('models', exist_ok=True)
scaler_path = 'models/scaler.pkl'
joblib.dump(scaler, scaler_path)
print(f"Scaler created and saved: {scaler_path}")

# ==================== الخطوة 4: إنشاء Clustering Model ====================
print("\n[Step 4/4] Creating clustering model...")

# إنشاء نموذج clustering
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
kmeans.fit(scaled_data)

# حفظ نموذج الـ clustering
cluster_path = 'models/cluster_model.pkl'
joblib.dump(kmeans, cluster_path)
print(f"Cluster model created and saved: {cluster_path}")

# ==================== الخطوة 5: نسخ الموديل الأصلي ====================
print("\n[Step 5/5] Copying original model to models folder...")

model_path = 'models/heart_disease_model.pkl'
if EXISTING_MODEL_PATH != model_path:
    joblib.dump(existing_model, model_path)
    print(f"Model copied to: {model_path}")
else:
    print(f"Model already in correct location: {model_path}")

# ==================== اختبار النظام الكامل ====================
print("\n" + "="*60)
print("Testing Complete System")
print("="*60)

# بيانات اختبار
test_patient = pd.DataFrame([{
    'age': 63,
    'sex': 1,
    'cp': 3,
    'trestbps': 145,
    'chol': 233,
    'fbs': 1,
    'restecg': 0,
    'thalach': 150,
    'exang': 0,
    'oldpeak': 2.3,
    'slope': 0,
    'ca': 0,
    'thal': 1
}])

try:
    # تطبيق scaling
    test_scaled = scaler.transform(test_patient)
    
    # التنبؤ
    prediction = existing_model.predict(test_scaled)[0]
    
    # محاولة الحصول على probability
    if hasattr(existing_model, 'predict_proba'):
        probability = existing_model.predict_proba(test_scaled)[0]
        print(f" Test Prediction:")
        print(f"   - Risk: {'HIGH' if prediction == 1 else 'LOW'}")
        print(f"   - Probability: {probability[1]:.1%}")
    else:
        print(f" Test Prediction:")
        print(f"   - Risk: {'HIGH' if prediction == 1 else 'LOW'}")
        print(f"    Model doesn't support probability prediction")
    
    # Cluster prediction
    cluster = kmeans.predict(test_scaled)[0]
    print(f"   - Cluster: {cluster}")
    
except Exception as e:
    print(f" Test failed: {str(e)}")
    print(" This might mean your model expects different features.")
    print("   Please check your model's training data format.")

# ==================== ملخص النتائج ====================
print("\n" + "="*60)
print(" SETUP COMPLETE!")
print("="*60)
print("Created/Updated files:")
print(f"  1. {model_path}")
print(f"  2. {scaler_path}")
print(f"  3. {cluster_path}")

print(" Next Steps:")
print("  1. If test passed: Run 'streamlit run app.py'")
print("  2. If test failed: Check feature names/order in your model")
print("  3. Adjust pipeline if needed based on your model's requirements")

print(" Important Notes:")
print("  - Make sure your model expects the same 13 features")
print("  - If features differ, update prediction_pipeline.py")
print("  - The scaler is fitted on sample data - adjust if needed")
print("="*60)