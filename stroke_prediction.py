import numpy as np
import pandas as pd
import os
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
import catboost as cb
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import joblib
import warnings
from datetime import datetime
from PIL import Image, ImageDraw, ImageFont
from huggingface_hub import hf_hub_download
import requests
warnings.filterwarnings('ignore')

class StrokePredictionInterface:
    def __init__(self, model_paths):
        """
        Initialize the prediction interface with paths to downloaded models
        
        Args:
            model_paths: Dictionary mapping filenames to their cached local paths
        """
        print("Loading Brain Stroke Prediction Models...")
        print("="*50)
        
        self.model_paths = model_paths
        
        # Load U-Net model
        unet_path = self.model_paths.get("best_unet_model.h5")
        if not unet_path or not os.path.exists(unet_path):
            raise ValueError(f"U-Net model not found at {unet_path}")
            
        print(f"Loading U-Net model from: {unet_path}")
        self.unet_model = keras.models.load_model(unet_path)
        
        # Get input specifications from U-Net
        unet_input_shape = self.unet_model.input_shape
        self.img_size = (unet_input_shape[1], unet_input_shape[2])
        self.input_channels = unet_input_shape[3] if len(unet_input_shape) > 3 else 1
        
        print(f"U-Net input shape: {unet_input_shape}")
        print(f"Image size: {self.img_size}")
        print(f"Input channels: {self.input_channels}")
        
        # Load ensemble models
        print("\nLoading ensemble models...")
        self._load_ensemble()
        
        # Check if all models are loaded
        self.models_loaded = all([
            self.lightgbm_model is not None,
            self.catboost_model is not None,
            self.adaboost_model is not None,
            self.decision_tree_meta is not None
        ])
        
        if self.models_loaded:
            print("\n✓ All models loaded successfully!")
        else:
            print("\n⚠ Some models failed to load. Predictions may not work properly.")
    
    def _load_ensemble(self):
        """Load ensemble models from provided paths"""
        models_to_load = {
            'lightgbm_model': 'my_stroke_ensemble_lightgbm.pkl',
            'catboost_model': 'my_stroke_ensemble_catboost.pkl',
            'adaboost_model': 'my_stroke_ensemble_adaboost.pkl',
            'decision_tree_meta': 'my_stroke_ensemble_decision_tree_meta.pkl',
            'scaler': 'my_stroke_ensemble_scaler.pkl'
        }
        
        for attr_name, filename in models_to_load.items():
            path = self.model_paths.get(filename)
            try:
                if not path or not os.path.exists(path):
                    raise FileNotFoundError(f"File {filename} not found in model paths.")
                
                model = joblib.load(path)
                setattr(self, attr_name, model)
                display_name = attr_name.replace('_model', '').replace('_', ' ').title()
                print(f"✓ {display_name} loaded")
            except Exception as e:
                print(f"✗ Error loading {attr_name}: {e}")
                setattr(self, attr_name, None)
        
        # Fallback scaler if not loaded
        if not hasattr(self, 'scaler') or self.scaler is None:
            self.scaler = StandardScaler()
    
    def preprocess_image(self, image_path):
        """
        Preprocess a single image for prediction
        
        Args:
            image_path: Path to the CT scan image
            
        Returns:
            Preprocessed image array
        """
        try:
            color_mode = 'grayscale' if self.input_channels == 1 else 'rgb'
            img = load_img(image_path, target_size=self.img_size, color_mode=color_mode)
            img_array = img_to_array(img) / 255.0
            
            # Ensure correct channels
            if self.input_channels == 1 and len(img_array.shape) == 3 and img_array.shape[-1] == 3:
                img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
                img_array = np.expand_dims(img_array, axis=-1)
            elif self.input_channels == 3 and len(img_array.shape) == 3 and img_array.shape[-1] == 1:
                img_array = np.repeat(img_array, 3, axis=-1)
            
            return img_array
            
        except Exception as e:
            print(f"Error preprocessing image {image_path}: {str(e)}")
            return None
    
    def extract_unet_features(self, img_array):
        """
        Extract features from preprocessed image using U-Net
        
        Args:
            img_array: Preprocessed image array
            
        Returns:
            Extracted features
        """
        try:
            # Add batch dimension
            img_batch = np.expand_dims(img_array, axis=0)
            
            # Find suitable layer for feature extraction
            layer_info = []
            for i, layer in enumerate(self.unet_model.layers):
                try:
                    output_shape = layer.output_shape
                    if isinstance(output_shape, tuple) and len(output_shape) == 4:
                        h, w = output_shape[1], output_shape[2]
                        if h is not None and w is not None and h <= 56 and w <= 56:
                            layer_info.append((i, layer.name, output_shape))
                except:
                    continue
            
            # Use a suitable layer or fallback
            if layer_info:
                feature_layer_name = layer_info[0][1]
            else:
                mid_idx = len(self.unet_model.layers) // 2
                feature_layer_name = self.unet_model.layers[mid_idx].name
            
            # Create feature extractor with global average pooling
            base_output = self.unet_model.get_layer(feature_layer_name).output
            
            if len(base_output.shape) == 4:
                pooled_output = tf.keras.layers.GlobalAveragePooling2D()(base_output)
            else:
                pooled_output = base_output
            
            feature_extractor = keras.Model(
                inputs=self.unet_model.input,
                outputs=pooled_output
            )
            
            # Extract features
            features = feature_extractor.predict(img_batch, verbose=0)
            
            return features.flatten()
            
        except Exception as e:
            print(f"Error extracting U-Net features: {str(e)}")
            return self.create_simple_features(img_array)
    
    def create_simple_features(self, img_array):
        """
        Create simple statistical features from image
        
        Args:
            img_array: Image array
            
        Returns:
            Statistical features
        """
        # Convert to grayscale for analysis
        if len(img_array.shape) == 3:
            if img_array.shape[-1] == 3:
                gray_img = cv2.cvtColor(img_array.astype(np.float32), cv2.COLOR_RGB2GRAY)
            else:
                gray_img = img_array[:, :, 0].astype(np.float32)
        else:
            gray_img = img_array.astype(np.float32)
        
        # Calculate statistics
        features = [
            np.mean(gray_img),
            np.std(gray_img),
            np.min(gray_img),
            np.max(gray_img),
            np.median(gray_img),
            np.percentile(gray_img, 25),
            np.percentile(gray_img, 75)
        ]
        
        # Add histogram features
        hist = cv2.calcHist([gray_img], [0], None, [16], [gray_img.min(), gray_img.max()])
        hist_features = hist.flatten() / (hist.sum() + 1e-7)
        
        return np.array(features + hist_features.tolist())
    
    def predict_single_image(self, image_path):
        """
        Predict stroke probability for a single image
        
        Args:
            image_path: Path to CT scan image
            
        Returns:
            Dictionary containing prediction results
        """
        if not self.models_loaded:
            return {"error": "Models not properly loaded"}
        
        # Preprocess image
        img_array = self.preprocess_image(image_path)
        if img_array is None:
            return {"error": "Failed to preprocess image"}
        
        try:
            # Extract features
            unet_features = self.extract_unet_features(img_array)
            tabular_features = self.create_simple_features(img_array)
            
            # Get expected feature counts
            lgb_features = getattr(self.lightgbm_model, 'n_features_in_', None) or getattr(self.lightgbm_model, 'n_features_', None)
            
            # Create features to match model expectations
            if lgb_features is not None:
                target_features = lgb_features
                combined_raw = np.hstack([unet_features, tabular_features])
                
                if len(combined_raw) == target_features:
                    model_features = combined_raw.reshape(1, -1)
                elif len(combined_raw) > target_features:
                    model_features = combined_raw[:target_features].reshape(1, -1)
                else:
                    padded = np.zeros(target_features)
                    padded[:len(combined_raw)] = combined_raw
                    model_features = padded.reshape(1, -1)
                
                # Scale features
                try:
                    model_features = self.scaler.transform(model_features)
                except:
                    pass  # Use unscaled if scaling fails
            else:
                model_features = np.hstack([unet_features, tabular_features]).reshape(1, -1)
            
            # Get predictions from individual models
            lgb_pred = self.lightgbm_model.predict_proba(model_features)[0, 1]
            cat_pred = self.catboost_model.predict_proba(model_features)[0, 1]
            ada_pred = self.adaboost_model.predict_proba(model_features)[0, 1]
            
            # Meta-features for decision tree
            meta_features = np.array([[lgb_pred, cat_pred, ada_pred]])
            
            # Final prediction
            final_prediction = self.decision_tree_meta.predict(meta_features)[0]
            final_probabilities = self.decision_tree_meta.predict_proba(meta_features)[0]
            
            # Prepare results
            result = {
                "image_path": image_path,
                "prediction": int(final_prediction),
                "prediction_label": "Stroke Detected" if final_prediction == 1 else "Normal",
                "stroke_probability": float((lgb_pred + cat_pred + ada_pred) / 3),
                "individual_predictions": {
                    "lightgbm": float(lgb_pred),
                    "catboost": float(cat_pred), 
                    "adaboost": float(ada_pred)
                },
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }
            
            return result
            
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            print(f"Full error details:\n{error_details}")
            return {"error": f"Prediction failed: {str(e)}", "details": error_details}
    
    def predict_multiple_images(self, image_paths):
        """
        Predict on multiple images
        
        Args:
            image_paths: List of image paths
            
        Returns:
            List of prediction results
        """
        results = []
        
        print(f"Processing {len(image_paths)} images...")
        print("-" * 50)
        
        for i, image_path in enumerate(image_paths, 1):
            print(f"Processing image {i}/{len(image_paths)}: {os.path.basename(image_path)}")
            
            result = self.predict_single_image(image_path)
            results.append(result)
            
            if "error" not in result:
                prediction = result["prediction_label"]
                print(f"Result: {prediction}")
            else:
                print(f"Error: {result['error']}")
            
            print()
        
        return results


# Example usage
def main():
    """
    Main function demonstrating how to use the prediction interface with Hugging Face
    """
    from download_model import download_models
    
    # 1. Download models first
    repo_id = "saishhh/brain-stroke-model"  # Replace with your HF repo if different
    try:
        model_paths = download_models(repo_id=repo_id)
        
        # 2. Initialize predictor with the downloaded paths
        predictor = StrokePredictionInterface(model_paths=model_paths)
        
        # Example: Predict on a single image
        print("\n" + "="*50)
        print("SINGLE IMAGE PREDICTION")
        print("="*50)
        
        single_image_path = "test_image.jpg"
        if os.path.exists(single_image_path):
            result = predictor.predict_single_image(single_image_path)
            
            if "error" not in result:
                print(f"Image: {os.path.basename(result['image_path'])}")
                print(f"Prediction: {result['prediction_label']}")
                print(f"Stroke Probability: {result['stroke_probability']:.3f}")
            else:
                print(f"Error: {result['error']}")
        else:
            print(f"Test image '{single_image_path}' not found for demonstration.")
            
    except Exception as e:
        print(f"Failed to run demonstration: {e}")

if __name__ == "__main__":
    main()