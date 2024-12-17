import os
import logging
import joblib
import numpy as np
from xgboost import Booster, DMatrix
from sklearn.ensemble import RandomForestClassifier

logger = logging.getLogger(__name__)

class ModelManager:
    """Manages loading and prediction for different model types"""
    
    MODEL_CONFIGS = {
        'xgboost_cosine': {
            'model_path': 'xgboost_cosine_model.json',
            'scaler_path': 'xgboost_cosine_scaler.joblib',
            'features': ['cosine_similarity'],
            'type': 'xgboost',
            'display_name': 'XGBoost with Cosine Similarity'
        },
        'xgboost_top3': {
            'model_path': 'xgboost_top3_model.json',
            'scaler_path': 'xgboost_top3_scaler.joblib',
            'features': ['cosine_similarity', 'correlation', 'area_ratio'],
            'type': 'xgboost',
            'display_name': 'XGBoost with Top 3 Features'
        },
        'rf_cosine': {
            'model_path': 'random_forest_cosine_model.joblib',
            'scaler_path': 'random_forest_cosine_scaler.joblib',
            'features': ['cosine_similarity'],
            'type': 'random_forest',
            'display_name': 'Random Forest with Cosine Similarity'
        },
        'rf_top3': {
            'model_path': 'random_forest_top3_model.joblib',
            'scaler_path': 'random_forest_top3_scaler.joblib',
            'features': ['cosine_similarity', 'correlation', 'area_ratio'],
            'type': 'random_forest',
            'display_name': 'Random Forest with Top 3 Features'
        }
    }

    def __init__(self, model_dir):
        self.model_dir = model_dir
        self.models = {}
        self.scalers = {}
        logger.info(f"Initializing ModelManager with directory: {model_dir}")
        self._load_all_models()

    def _load_all_models(self):
        """Load all models and their corresponding scalers"""
        logger.info(f"Loading models from directory: {self.model_dir}")
        
        if not os.path.exists(self.model_dir):
            logger.error(f"Model directory does not exist: {self.model_dir}")
            raise FileNotFoundError(f"Model directory not found: {self.model_dir}")
        
        logger.info(f"Directory contents: {os.listdir(self.model_dir)}")
        
        for model_name, config in self.MODEL_CONFIGS.items():
            try:
                model_path = os.path.join(self.model_dir, config['model_path'])
                scaler_path = os.path.join(self.model_dir, config['scaler_path'])
                
                logger.info(f"Attempting to load {model_name}...")
                logger.info(f"Model path: {model_path}")
                logger.info(f"Scaler path: {scaler_path}")
                
                if not os.path.exists(model_path):
                    logger.warning(f"Model file not found: {model_path}")
                    continue
                    
                if not os.path.exists(scaler_path):
                    logger.warning(f"Scaler file not found: {scaler_path}")
                    continue
                
                # Load scaler
                self.scalers[model_name] = joblib.load(scaler_path)
                logger.info(f"Loaded scaler for {model_name}")
                
                # Load model based on type
                if config['type'] == 'xgboost':
                    model = Booster()
                    model.load_model(model_path)
                    logger.info(f"Loaded XGBoost model for {model_name}")
                else:  # random_forest
                    model = joblib.load(model_path)
                    logger.info(f"Loaded Random Forest model for {model_name}")
                    
                self.models[model_name] = model
                logger.info(f"Successfully loaded {model_name}")
                
            except Exception as e:
                logger.error(f"Error loading {model_name}: {str(e)}", exc_info=True)
                continue

        logger.info(f"Loaded models: {list(self.models.keys())}")

    def predict(self, model_name, features):
        """Make prediction using specified model"""
        try:
            logger.info(f"Starting prediction for model: {model_name}")
            logger.info(f"Input features: {features}")
            
            if model_name not in self.models:
                logger.error(f"Model {model_name} not found. Available models: {list(self.models.keys())}")
                raise ValueError(f"Model {model_name} not found")
            
            model = self.models[model_name]
            scaler = self.scalers[model_name]
            
            # Convert features dict to array in correct order
            feature_names = self.MODEL_CONFIGS[model_name]['features']
            feature_values = [features[fname] for fname in feature_names]
            features_array = np.array(feature_values)
            
            logger.info(f"Feature names: {feature_names}")
            logger.info(f"Feature values: {feature_values}")
            
            # Reshape features for scaler
            features_array = features_array.reshape(1, -1)
            logger.info(f"Features shape before scaling: {features_array.shape}")
            
            # Scale features
            try:
                features_scaled = scaler.transform(features_array)
                logger.info(f"Scaled features shape: {features_scaled.shape}")
            except Exception as e:
                logger.error(f"Error during feature scaling: {str(e)}", exc_info=True)
                raise
            
            # Make prediction
            try:
                if self.MODEL_CONFIGS[model_name]['type'] == 'xgboost':
                    logger.info("Using XGBoost model for prediction")
                    dmatrix = DMatrix(features_scaled)
                    prob = float(model.predict(dmatrix)[0])
                    pred = 1 if prob >= 0.5 else 0
                else:  # random_forest
                    logger.info("Using Random Forest model for prediction")
                    pred = int(model.predict(features_scaled)[0])
                    prob = float(model.predict_proba(features_scaled)[0][1])
                
                logger.info(f"Prediction successful: pred={pred}, prob={prob}")
                return pred, prob
                
            except Exception as e:
                logger.error(f"Error during model prediction: {str(e)}", exc_info=True)
                raise
                
        except Exception as e:
            logger.error(f"Error in predict method: {str(e)}", exc_info=True)
            raise

    def get_available_models(self):
        """Get dictionary of available models and their display names"""
        available = {
            name: config['display_name']
            for name, config in self.MODEL_CONFIGS.items()
            if name in self.models
        }
        logger.info(f"Available models: {available}")
        return available
