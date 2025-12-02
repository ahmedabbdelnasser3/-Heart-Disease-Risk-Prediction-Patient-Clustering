
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
import joblib
import os

print("="*60)
print("Creating Heart Disease Models")
print("="*60)

# Create models folder
os.makedirs('models', exist_ok=True)
print("[Step 1/5] Created models folder")

# Generate realistic sample data (300 patients)
np.random.seed(42)
n_samples = 300

print(f"[Step 2/5] Generating {n_samples} patient records")

# Create features with realistic distributions
age = np.random.randint(30, 80, n_samples)
sex = np.random.randint(0, 2, n_samples)
cp = np.random.randint(0, 4, n_samples)
trestbps = np.random.randint(90, 200, n_samples)
chol = np.random.randint(150, 400, n_samples)
fbs = np.random.randint(0, 2, n_samples)
restecg = np.random.randint(0, 3, n_samples)
thalach = np.random.randint(80, 200, n_samples)
exang = np.random.randint(0, 2, n_samples)
oldpeak = np.random.uniform(0, 6, n_samples)
slope = np.random.randint(0, 3, n_samples)
ca = np.random.randint(0, 4, n_samples)
thal = np.random.randint(0, 4, n_samples)

# Create realistic target based on risk factors
risk_score = (
    (age > 60).astype(int) * 25 +
    (chol > 250).astype(int) * 20 +
    (trestbps > 140).astype(int) * 15 +
    (thalach < 120).astype(int) * 20 +
    (oldpeak > 2).astype(int) * 15 +
    cp * 5 +
    np.random.randint(-10, 10, n_samples)
)

target = (risk_score > 50).astype(int)

# Create DataFrame
df = pd.DataFrame({
    'age': age,
    'sex': sex,
    'cp': cp,
    'trestbps': trestbps,
    'chol': chol,
    'fbs': fbs,
    'restecg': restecg,
    'thalach': thalach,
    'exang': exang,
    'oldpeak': oldpeak,
    'slope': slope,
    'ca': ca,
    'thal': thal,
    'target': target
})

print(f"   Total patients: {len(df)}")
print(f"   With heart disease: {target.sum()} ({target.sum()/len(df)*100}%)")
print(f"   Healthy: {(1-target).sum()} ({(1-target).sum()/len(df)*100}%)")

# Separate features and target
X = df.drop('target', axis=1)
y = df['target']

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"   Training samples: {len(X_train)}")
print(f"   Testing samples: {len(X_test)}")

# Create and save Scaler
print("\n[Step 3/5] Creating and saving Scaler")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

scaler_path = 'models/scaler.pkl'
joblib.dump(scaler, scaler_path)
print(f"   SAVED: {scaler_path}")

# Train and save Classification Model
print("\n[Step 4/5] Training Classification Model")
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    random_state=42,
    n_jobs=-1
)
model.fit(X_train_scaled, y_train)

train_accuracy = model.score(X_train_scaled, y_train)
test_accuracy = model.score(X_test_scaled, y_test)

print(f"   Training accuracy: {train_accuracy:.1%}")
print(f"   Testing accuracy: {test_accuracy:.1%}")

model_path = 'models/heart_disease_model.pkl'
joblib.dump(model, model_path)
print(f"   SAVED: {model_path}")

# Create and save Clustering Model
print("[Step 5/5] Creating Clustering Model")
X_all_scaled = scaler.transform(X)

kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X_all_scaled)

# Show cluster statistics
print("  Cluster distribution:")
for i in range(3):
    cluster_size = (clusters == i).sum()
    cluster_disease_rate = df[clusters == i]['target'].mean()
    risk_level = "LOW" if cluster_disease_rate < 0.3 else "MODERATE" if cluster_disease_rate < 0.7 else "HIGH"
    print(f"   - Cluster {i}: {cluster_size} patients, {cluster_disease_rate:.1%} disease rate ({risk_level} RISK)")

cluster_path = 'models/cluster_model.pkl'
joblib.dump(kmeans, cluster_path)
print(f"   SAVED: {cluster_path}")

# Quick test
print("\n" + "="*60)
print("Testing Models with Sample Patient")
print("="*60)

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

test_scaled = scaler.transform(test_patient)
prediction = model.predict(test_scaled)[0]
probability = model.predict_proba(test_scaled)[0]
cluster = kmeans.predict(test_scaled)[0]

print(f"Test Results:")
print(f"  Risk Level: {'HIGH RISK' if prediction == 1 else 'LOW RISK'}")
print(f"  Probability: {probability[1]:.1%}")
print(f"  Confidence: {max(probability):.1%}")
print(f"  Assigned Cluster: {cluster}")

print("\n" + "="*60)
print("SUCCESS! All Models Created Successfully")
print("="*60)
print("Created files:")
print("  1. models/heart_disease_model.pkl  (Classification)")
print("  2. models/scaler.pkl               (Feature Scaling)")
print("  3. models/cluster_model.pkl        (Risk Clustering)")
print("Next step:")
print("  Run: python test_pipeline.py")
print("="*60)