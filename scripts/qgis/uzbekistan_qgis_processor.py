"""
QGIS Processing script for Uzbekistan land cover classification
This script handles local processing, classification, and validation using QGIS.
"""

import os
import sys
import numpy as np
import pandas as pd
import geopandas as gpd
from pathlib import Path
import rasterio
from rasterio.windows import Window
from rasterio.warp import calculate_default_transform, reproject, Resampling
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# QGIS imports
try:
    from qgis.core import QgsApplication, QgsRasterLayer
    from qgis.analysis import QgsNativeAlgorithms
    import processing
    QGIS_AVAILABLE = True
except ImportError:
    print("⚠️  QGIS not available. Some functions will be limited.")
    QGIS_AVAILABLE = False
    # Create dummy classes for when QGIS is not available
    class QgsApplication:
        @staticmethod
        def setPrefixPath(*args): pass
        def __init__(self, *args): pass
        def initQgis(self): pass
        @staticmethod
        def processingRegistry(): 
            class DummyRegistry:
                def addProvider(self, *args): pass
            return DummyRegistry()
    
    class QgsRasterLayer:
        def __init__(self, *args): pass

# Add config path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'config'))
from land_cover_config import LAND_COVER_CLASSES, MERGED_CLASSES, QGIS_CONFIG, ML_CONFIG

class UzbekistanQGISProcessor:
    def __init__(self, data_dir):
        """Initialize the QGIS processor."""
        self.data_dir = Path(data_dir)
        self.results_dir = self.data_dir / 'results'
        self.training_dir = self.data_dir / 'training'
        
        # Create directories if they don't exist
        self.results_dir.mkdir(exist_ok=True)
        self.training_dir.mkdir(exist_ok=True)
        
        # Initialize QGIS if available
        if QGIS_AVAILABLE:
            self.init_qgis()
        
        print(f"📂 Working directory: {self.data_dir}")
    
    def init_qgis(self):
        """Initialize QGIS application."""
        # Initialize QGIS application
        QgsApplication.setPrefixPath("/path/to/qgis", True)  # Update with your QGIS path
        qgs = QgsApplication([], False)
        qgs.initQgis()
        
        # Initialize processing
        from processing.core.Processing import Processing
        Processing.initialize()
        QgsApplication.processingRegistry().addProvider(QgsNativeAlgorithms())
    
    def load_training_data(self, training_csv_path):
        """Load and prepare training data from CSV exported from GEE."""
        print(f"📊 Loading training data from {training_csv_path}")
        
        # Load training data
        training_df = pd.read_csv(training_csv_path)
        
        # Remove any rows with null values
        training_df = training_df.dropna()
        
        # Prepare features and labels
        feature_columns = [col for col in training_df.columns if col in ML_CONFIG['features']]
        X = training_df[feature_columns]
        y = training_df['layer_id']  # Use layer_id as the target
        
        print(f"✅ Loaded {len(training_df)} training samples")
        print(f"📈 Features: {feature_columns}")
        print(f"🎯 Classes: {sorted(y.unique())}")
        
        return X, y, feature_columns
    
    def train_classifier(self, X, y, algorithm='random_forest'):
        """Train a machine learning classifier."""
        print(f"🤖 Training {algorithm} classifier")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=ML_CONFIG['validation_split'], random_state=42, stratify=y
        )
        
        # Select algorithm
        if algorithm == 'random_forest':
            clf = RandomForestClassifier(**ML_CONFIG['algorithms']['random_forest'], random_state=42)
        elif algorithm == 'gradient_boosting':
            clf = GradientBoostingClassifier(**ML_CONFIG['algorithms']['gradient_boosting'], random_state=42)
        elif algorithm == 'svm':
            clf = SVC(**ML_CONFIG['algorithms']['svm'], random_state=42)
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")
        
        # Train classifier
        clf.fit(X_train, y_train)
        
        # Validate
        y_pred = clf.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"✅ Training complete. Accuracy: {accuracy:.3f}")
        
        # Cross-validation
        cv_scores = cross_val_score(clf, X, y, cv=ML_CONFIG['cross_validation_folds'])
        print(f"📊 Cross-validation accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
        
        # Classification report
        print("\n📋 Classification Report:")
        print(classification_report(y_test, y_pred))
        
        return clf, accuracy, y_test, y_pred
    
    def save_model(self, classifier, algorithm, feature_columns, accuracy):
        """Save the trained model."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        model_filename = f"uzbekistan_classifier_{algorithm}_{timestamp}_acc{accuracy:.3f}.joblib"
        model_path = self.results_dir / model_filename
        
        # Save model with metadata
        model_data = {
            'classifier': classifier,
            'algorithm': algorithm,
            'feature_columns': feature_columns,
            'accuracy': accuracy,
            'land_cover_classes': LAND_COVER_CLASSES,
            'training_timestamp': timestamp
        }
        
        joblib.dump(model_data, model_path)
        print(f"💾 Model saved: {model_path}")
        
        return model_path
    
    def load_model(self, model_path):
        """Load a trained model."""
        model_data = joblib.load(model_path)
        print(f"📤 Model loaded: {model_path}")
        return model_data
    
    def create_confusion_matrix_plot(self, y_true, y_pred, algorithm):
        """Create and save confusion matrix plot."""
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=sorted(set(y_true)), 
                   yticklabels=sorted(set(y_true)))
        plt.title(f'Confusion Matrix - {algorithm}')
        plt.xlabel('Predicted Class')
        plt.ylabel('True Class')
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        plot_path = self.results_dir / f"confusion_matrix_{algorithm}_{timestamp}.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Confusion matrix saved: {plot_path}")
        return plot_path
    
    def classify_raster_tiles(self, raster_path, model_data, output_path):
        """Classify a large raster in tiles to manage memory."""
        print(f"🗺️  Classifying raster: {raster_path}")
        
        classifier = model_data['classifier']
        feature_columns = model_data['feature_columns']
        
        with rasterio.open(raster_path) as src:
            profile = src.profile.copy()
            profile.update({
                'dtype': 'uint8',
                'count': 1,
                'compress': 'lzw'
            })
            
            # Calculate tile size
            tile_size = QGIS_CONFIG['tile_size']
            height, width = src.height, src.width
            
            with rasterio.open(output_path, 'w', **profile) as dst:
                # Process in tiles
                for row in range(0, height, tile_size):
                    for col in range(0, width, tile_size):
                        # Calculate window
                        window_height = min(tile_size, height - row)
                        window_width = min(tile_size, width - col)
                        window = Window(col, row, window_width, window_height)
                        
                        # Read tile
                        tile_data = src.read(window=window)
                        
                        # Reshape for classification
                        original_shape = tile_data.shape
                        tile_reshaped = tile_data.reshape(tile_data.shape[0], -1).T
                        
                        # Create DataFrame with feature names
                        tile_df = pd.DataFrame(tile_reshaped, columns=[f'B{i+1}' for i in range(tile_data.shape[0])])
                        
                        # Select only the features used in training
                        available_features = [col for col in feature_columns if col in tile_df.columns]
                        if len(available_features) != len(feature_columns):
                            print(f"⚠️  Warning: Missing features. Available: {available_features}")
                        
                        tile_features = tile_df[available_features]
                        
                        # Handle NaN values
                        tile_features = tile_features.fillna(0)
                        
                        # Classify
                        try:
                            classified = classifier.predict(tile_features)
                            classified = classified.reshape(window_height, window_width)
                        except Exception as e:
                            print(f"❌ Error classifying tile: {e}")
                            classified = np.zeros((window_height, window_width), dtype=np.uint8)
                        
                        # Write classified tile
                        dst.write(classified, 1, window=window)
                        
                        print(f"✅ Processed tile: row {row}/{height}, col {col}/{width}")
        
        print(f"🎯 Classification complete: {output_path}")
        return output_path
    
    def post_process_classification(self, classified_raster_path, output_path):
        """Post-process classification results using QGIS algorithms."""
        if not QGIS_AVAILABLE:
            print("⚠️  QGIS not available for post-processing")
            return classified_raster_path
        
        print("🔧 Post-processing classification results")
        
        # Load classified raster
        classified_layer = QgsRasterLayer(classified_raster_path, "classified")
        
        # Apply majority filter to reduce noise
        majority_params = {
            'INPUT': classified_layer,
            'RADIUS': 1,
            'OUTPUT': output_path
        }
        
        try:
            processing.run("native:majorityfilterbycategory", majority_params)
            print(f"✅ Post-processing complete: {output_path}")
            return output_path
        except Exception as e:
            print(f"❌ Post-processing failed: {e}")
            return classified_raster_path
    
    def create_classification_statistics(self, classified_raster_path):
        """Generate classification statistics and area calculations."""
        print("📊 Generating classification statistics")
        
        with rasterio.open(classified_raster_path) as src:
            classified_data = src.read(1)
            pixel_area = src.res[0] * src.res[1]  # Area per pixel in square degrees
            
            # Calculate areas for each class
            unique_classes, counts = np.unique(classified_data[classified_data > 0], return_counts=True)
            
            stats = []
            for class_id, count in zip(unique_classes, counts):
                if class_id in LAND_COVER_CLASSES:
                    class_info = LAND_COVER_CLASSES[class_id]
                    area_sq_degrees = count * pixel_area
                    # Convert to approximate square kilometers (rough conversion)
                    area_sq_km = area_sq_degrees * 111.32 * 111.32  # Very rough approximation
                    
                    stats.append({
                        'class_id': int(class_id),
                        'class_name': class_info['name'],
                        'pixel_count': int(count),
                        'area_sq_km': round(area_sq_km, 2),
                        'percentage': round((count / np.sum(counts)) * 100, 2)
                    })
            
            # Create statistics DataFrame
            stats_df = pd.DataFrame(stats)
            stats_df = stats_df.sort_values('area_sq_km', ascending=False)
            
            # Save statistics
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            stats_path = self.results_dir / f"classification_statistics_{timestamp}.csv"
            stats_df.to_csv(stats_path, index=False)
            
            print("📈 Classification Statistics:")
            print(stats_df.to_string(index=False))
            print(f"💾 Statistics saved: {stats_path}")
            
            return stats_df, stats_path

def main():
    """Main execution function."""
    print("🚀 Starting Uzbekistan Land Cover Classification - QGIS Processing")
    
    # Setup paths
    base_dir = Path(__file__).parent.parent.parent
    data_dir = base_dir / 'data'
    
    # Initialize processor
    processor = UzbekistanQGISProcessor(data_dir)
    
    # Check for training data
    training_csv = data_dir / 'training' / 'uzbekistan_training_data.csv'
    if not training_csv.exists():
        print(f"❌ Training data not found: {training_csv}")
        print("🔄 Please run the GEE script first to generate training data")
        return
    
    # Load training data
    X, y, feature_columns = processor.load_training_data(training_csv)
    
    # Train classifiers
    algorithms = ['random_forest', 'gradient_boosting']
    best_model = None
    best_accuracy = 0
    
    for algorithm in algorithms:
        print(f"\n{'='*50}")
        print(f"Training {algorithm}")
        print('='*50)
        
        clf, accuracy, y_test, y_pred = processor.train_classifier(X, y, algorithm)
        
        # Save model
        model_path = processor.save_model(clf, algorithm, feature_columns, accuracy)
        
        # Create confusion matrix
        processor.create_confusion_matrix_plot(y_test, y_pred, algorithm)
        
        # Track best model
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = {
                'classifier': clf,
                'algorithm': algorithm,
                'feature_columns': feature_columns,
                'accuracy': accuracy,
                'path': model_path
            }
    
    print(f"\n🏆 Best model: {best_model['algorithm']} (accuracy: {best_accuracy:.3f})")
    
    # Look for satellite imagery to classify
    composite_files = list(data_dir.glob('*composite*.tif'))
    if composite_files:
        composite_path = composite_files[0]
        print(f"🛰️  Found composite image: {composite_path}")
        
        # Classify the composite
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        classified_path = processor.results_dir / f"uzbekistan_classified_{timestamp}.tif"
        
        processor.classify_raster_tiles(composite_path, best_model, classified_path)
        
        # Post-process
        postprocessed_path = processor.results_dir / f"uzbekistan_classified_postprocessed_{timestamp}.tif"
        final_path = processor.post_process_classification(classified_path, postprocessed_path)
        
        # Generate statistics
        processor.create_classification_statistics(final_path)
        
    else:
        print("⚠️  No composite imagery found. Please run GEE script first.")
    
    print("\n🎉 Classification workflow complete!")
    print(f"📁 Results saved in: {processor.results_dir}")

if __name__ == "__main__":
    main()
