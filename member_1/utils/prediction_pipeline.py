"""
Prediction Pipeline for Cardiovascular Disease (11 Features)
"""
import sklearn
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier

# Patch for sklearn version incompatibility
missing_attrs = ["monotonic_cst", "monotonic_constraints", "n_outputs_"]

for cls in [DecisionTreeClassifier, ExtraTreeClassifier]:
    for attr in missing_attrs:
        if not hasattr(cls, attr):
            setattr(cls, attr, None)


import numpy as np
import pandas as pd
from typing import Dict, Tuple
import joblib
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SimpleClusterModel:
    def __init__(self, model, scaler):
        self.model = model
        self.scaler = scaler
        self.n_clusters = 3
    
    def predict(self, X):
        if not hasattr(X, 'shape'):
            X = np.array(X)
        if len(X.shape) == 1:
            X = X.reshape(1, -1)
        proba = self.model.predict_proba(X)[:, 1]
        clusters = np.zeros(len(proba), dtype=int)
        clusters[proba < 0.3] = 0
        clusters[(proba >= 0.3) & (proba < 0.7)] = 1
        clusters[proba >= 0.7] = 2
        return clusters
    
    def fit(self, X, y=None):
        return self
    
    def fit_predict(self, X):
        return self.predict(X)

class PredictionPipeline:
    def __init__(self, model_path: str, scaler_path: str, cluster_model_path: str):
        try:
            self.model = joblib.load(model_path)
            self.scaler = joblib.load(scaler_path)
            try:
                self.cluster_model = joblib.load(cluster_model_path)
            except:
                logger.warning("Creating new cluster model...")
                self.cluster_model = SimpleClusterModel(self.model, self.scaler)
                print("CLUSTER MODEL TYPE:", type(self.cluster_model))
            logger.info("Models loaded successfully")
        except Exception as e:
            logger.error(f"Error loading models: {str(e)}")
            raise

    def predict_risk(self, input_data: Dict) -> Dict:
        try:
            df = self._prepare_input(input_data)
            scaled_features = self.scaler.transform(df)
            prediction = self.model.predict(scaled_features)[0]
            probability = self.model.predict_proba(scaled_features)[0]
            return {
                'prediction': int(prediction),
                'risk_level': 'High Risk' if prediction == 1 else 'Low Risk',
                'probability': float(probability[1]),
                'confidence': float(max(probability))
            }
        except Exception as e:
            logger.error(f"Error in prediction: {str(e)}")
            raise ValueError(f"Prediction failed: {str(e)}")

    def calculate_confidence_interval(self, probability: float, confidence_level: float = 0.95) -> Tuple[float, float]:
        try:
            from scipy import stats
            n = 100
            se = np.sqrt((probability * (1 - probability)) / n)
            z_score = stats.norm.ppf((1 + confidence_level) / 2)
            margin = z_score * se
            lower_bound = max(0, probability - margin)
            upper_bound = min(1, probability + margin)
            return (round(lower_bound, 3), round(upper_bound, 3))
        except Exception as e:
            logger.error(f"Error calculating CI: {str(e)}")
            return (probability, probability)

    def assign_cluster(self, input_data: Dict) -> Dict:
        try:
            df = self._prepare_input(input_data)
            scaled_features = self.scaler.transform(df)
            cluster = self.cluster_model.predict(scaled_features)[0]
            cluster_descriptions = {
                0: "Low Risk - Healthy Profile",
                1: "Moderate Risk - Monitor Closely",
                2: "High Risk - Immediate Attention Required"
            }
            return {
                'cluster_id': int(cluster),
                'cluster_description': cluster_descriptions.get(cluster, "Unknown"),
                'risk_category': self._get_risk_category(cluster)
            }
        except Exception as e:
            logger.error(f"Error in cluster assignment: {str(e)}")
            raise ValueError(f"Cluster assignment failed: {str(e)}")
    
    def _get_risk_category(self, cluster: int) -> str:
        if cluster == 0:
            return "LOW"
        elif cluster == 1:
            return "MODERATE"
        else:
            return "HIGH"

    def _prepare_input(self, input_data: Dict) -> pd.DataFrame:
        expected_features = ['age', 'gender', 'height', 'weight', 'ap_hi', 'ap_lo', 'cholesterol', 'gluc', 'smoke', 'alco', 'active']
        missing_features = [f for f in expected_features if f not in input_data]
        if missing_features:
            raise ValueError(f"Missing features: {missing_features}")
        df = pd.DataFrame([input_data])[expected_features]
        self._validate_input(df)
        return df

    def _validate_input(self, df: pd.DataFrame):
        validations = {
            'age': (0, 100), 'gender': (1, 2), 'height': (100, 250), 'weight': (30, 200),
            'ap_hi': (80, 250), 'ap_lo': (40, 150), 'cholesterol': (1, 3), 'gluc': (1, 3),
            'smoke': (0, 1), 'alco': (0, 1), 'active': (0, 1)
        }
        for feature, (min_val, max_val) in validations.items():
            value = df[feature].iloc[0]
            if not (min_val <= value <= max_val):
                raise ValueError(f"{feature} value {value} out of range [{min_val}, {max_val}]")

    def get_complete_prediction(self, input_data: Dict) -> Dict:
        try:
            prediction_result = self.predict_risk(input_data)
            ci_lower, ci_upper = self.calculate_confidence_interval(prediction_result['probability'])
            cluster_result = self.assign_cluster(input_data)
            return {
                'success': True,
                'prediction': prediction_result,
                'confidence_interval': {'lower': ci_lower, 'upper': ci_upper, 'level': '95%'},
                'cluster': cluster_result,
                'timestamp': pd.Timestamp.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Complete prediction failed: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'timestamp': pd.Timestamp.now().isoformat()
            }
        