"""
Main workflow script for Uzbekistan Land Cover Classification
This script orchestrates the entire classification workflow.
"""

import os
import sys
import subprocess
from pathlib import Path
from datetime import datetime
import argparse

def setup_environment():
    """Set up the Python environment and install dependencies."""
    print("🔧 Setting up environment...")
    
    # Install requirements
    subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], check=True)
    print("✅ Dependencies installed")

def run_gee_processing(training_shapefile=None):
    """Run Google Earth Engine processing."""
    print("\n" + "="*60)
    print("📡 STEP 1: Google Earth Engine Processing")
    print("="*60)
    
    # Add scripts to path
    gee_script_path = Path(__file__).parent / "scripts" / "gee"
    sys.path.insert(0, str(gee_script_path))
    
    try:
        from uzbekistan_gee_processor import UzbekistanEEProcessor
        
        processor = UzbekistanEEProcessor()
        
        # Process satellite data
        satellite_data = processor.load_satellite_data()
        composite = processor.create_classification_composite(satellite_data)
        
        # Export composite
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        export_task = processor.export_composite_image(composite, f"uzbekistan_composite_{timestamp}")
        
        # If training shapefile provided, process training data
        if training_shapefile and os.path.exists(training_shapefile):
            print(f"📊 Processing training data from {training_shapefile}")
            training_fc = processor.prepare_training_data(training_shapefile)
            training_data = processor.extract_features_for_training(composite, training_fc)
            processor.export_training_data(training_data, f"training_data_{timestamp}")
        
        print("✅ Google Earth Engine processing complete")
        print("📥 Check Google Drive for exported data")
        
    except Exception as e:
        print(f"❌ GEE processing failed: {e}")
        print("🔑 Make sure you're authenticated with Google Earth Engine:")
        print("   earthengine authenticate")
        return False
    
    return True

def run_qgis_processing():
    """Run QGIS local processing."""
    print("\n" + "="*60)
    print("🗺️  STEP 2: QGIS Local Processing")
    print("="*60)
    
    # Add scripts to path
    qgis_script_path = Path(__file__).parent / "scripts" / "qgis"
    sys.path.insert(0, str(qgis_script_path))
    
    try:
        from uzbekistan_qgis_processor import UzbekistanQGISProcessor
        
        data_dir = Path(__file__).parent / "data"
        processor = UzbekistanQGISProcessor(data_dir)
        
        # Check for training data
        training_csv = data_dir / "training" / "uzbekistan_training_data.csv"
        if not training_csv.exists():
            print(f"⚠️  Training data not found: {training_csv}")
            print("📥 Please download training data from Google Drive and place it in data/training/")
            return False
        
        # Load and train
        X, y, feature_columns = processor.load_training_data(training_csv)
        
        # Train multiple algorithms and select best
        algorithms = ['random_forest', 'gradient_boosting']
        best_model = None
        best_accuracy = 0
        
        for algorithm in algorithms:
            clf, accuracy, y_test, y_pred = processor.train_classifier(X, y, algorithm)
            model_path = processor.save_model(clf, algorithm, feature_columns, accuracy)
            processor.create_confusion_matrix_plot(y_test, y_pred, algorithm)
            
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_model = {
                    'classifier': clf,
                    'algorithm': algorithm,
                    'feature_columns': feature_columns,
                    'accuracy': accuracy
                }
        
        print(f"🏆 Best model: {best_model['algorithm']} (accuracy: {best_accuracy:.3f})")
        
        # Classify imagery if available
        composite_files = list(data_dir.glob("*composite*.tif"))
        if composite_files:
            composite_path = composite_files[0]
            print(f"🛰️  Classifying: {composite_path}")
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            classified_path = processor.results_dir / f"uzbekistan_classified_{timestamp}.tif"
            
            processor.classify_raster_tiles(composite_path, best_model, classified_path)
            
            # Generate statistics
            processor.create_classification_statistics(classified_path)
            
        print("✅ QGIS processing complete")
        
    except Exception as e:
        print(f"❌ QGIS processing failed: {e}")
        return False
    
    return True

def create_project_summary():
    """Create a summary of the project and results."""
    print("\n" + "="*60)
    print("📋 PROJECT SUMMARY")
    print("="*60)
    
    project_info = f"""
🌍 Uzbekistan Land Cover Classification Project
===============================================

📅 Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

🎯 Objective:
   • Classify land cover across entire country of Uzbekistan
   • Use 200k training features across 12 land cover classes
   • Combine Google Earth Engine satellite data with local QGIS processing

📊 Land Cover Classes:
"""
    
    # Add class information
    from config.land_cover_config import LAND_COVER_CLASSES
    for class_id, class_info in LAND_COVER_CLASSES.items():
        project_info += f"   {class_id:2d}. {class_info['name']:20} - {class_info['description']}\n"
    
    project_info += f"""
🛠️  Workflow:
   1. Google Earth Engine: Satellite data acquisition and preprocessing
   2. Feature extraction: Spectral indices, topographic features
   3. Local processing: Machine learning classification using QGIS
   4. Validation: Cross-validation and accuracy assessment
   5. Post-processing: Majority filtering and area calculations

📁 Project Structure:
   • data/training/     - Training data and shapefiles
   • data/results/      - Classification outputs and statistics
   • scripts/gee/       - Google Earth Engine scripts
   • scripts/qgis/      - QGIS processing scripts
   • config/           - Configuration files
   • notebooks/        - Jupyter notebooks for analysis

🚀 Next Steps:
   1. Download exported data from Google Drive
   2. Place training shapefile in data/training/
   3. Run: python main_workflow.py --full
   4. Review results in data/results/
"""
    
    print(project_info)
    
    # Save summary
    summary_path = Path(__file__).parent / "PROJECT_SUMMARY.md"
    with open(summary_path, 'w') as f:
        f.write(project_info)
    
    print(f"💾 Summary saved: {summary_path}")

def main():
    """Main workflow execution."""
    parser = argparse.ArgumentParser(description="Uzbekistan Land Cover Classification Workflow")
    parser.add_argument("--setup", action="store_true", help="Setup environment only")
    parser.add_argument("--gee", action="store_true", help="Run GEE processing only")
    parser.add_argument("--qgis", action="store_true", help="Run QGIS processing only")
    parser.add_argument("--full", action="store_true", help="Run full workflow")
    parser.add_argument("--training-shapefile", type=str, help="Path to training shapefile")
    parser.add_argument("--summary", action="store_true", help="Create project summary only")
    
    args = parser.parse_args()
    
    print("🚀 Uzbekistan Land Cover Classification Workflow")
    print("=" * 50)
    
    if args.summary or (not any([args.setup, args.gee, args.qgis, args.full])):
        create_project_summary()
        return
    
    if args.setup or args.full:
        try:
            setup_environment()
        except subprocess.CalledProcessError as e:
            print(f"❌ Environment setup failed: {e}")
            return
    
    if args.gee or args.full:
        success = run_gee_processing(args.training_shapefile)
        if not success and args.full:
            print("⚠️  GEE processing failed, skipping QGIS processing")
            return
    
    if args.qgis or args.full:
        run_qgis_processing()
    
    create_project_summary()
    print("\n🎉 Workflow complete!")

if __name__ == "__main__":
    main()
