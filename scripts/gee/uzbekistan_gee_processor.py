"""
Google Earth Engine script for Uzbekistan land cover classification
This script handles satellite data acquisition, preprocessing, and feature extraction.
"""

import ee
import geemap
import pandas as pd
import geopandas as gpd
from datetime import datetime
import sys
import os

# Add config path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'config'))
from land_cover_config import GEE_CONFIG, LAND_COVER_CLASSES, ML_CONFIG

class UzbekistanEEProcessor:
    def __init__(self):
        """Initialize the Earth Engine processor."""
        self.initialize_ee()
        self.uzbekistan_aoi = self.create_uzbekistan_aoi()
        
    def initialize_ee(self):
        """Initialize Google Earth Engine."""
        try:
            ee.Initialize()
            print("✅ Google Earth Engine initialized successfully")
        except Exception as e:
            print(f"❌ Error initializing Earth Engine: {e}")
            print("Please authenticate with: earthengine authenticate")
            sys.exit(1)
    
    def create_uzbekistan_aoi(self):
        """Create Uzbekistan area of interest polygon."""
        coords = GEE_CONFIG['uzbekistan_bounds']['coordinates']
        return ee.Geometry.Polygon(coords)
    
    def load_satellite_data(self, start_date=None, end_date=None):
        """Load and preprocess satellite imagery for Uzbekistan."""
        start_date = start_date or GEE_CONFIG['date_range']['start']
        end_date = end_date or GEE_CONFIG['date_range']['end']
        
        print(f"📡 Loading satellite data from {start_date} to {end_date}")
        
        # Load Sentinel-2 data
        sentinel2 = (ee.ImageCollection(GEE_CONFIG['satellite_collections']['sentinel2'])
                    .filterBounds(self.uzbekistan_aoi)
                    .filterDate(start_date, end_date)
                    .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', GEE_CONFIG['cloud_filter']))
                    .map(self.mask_s2_clouds)
                    .median()
                    .clip(self.uzbekistan_aoi))
        
        # Load Landsat 8/9 data as backup
        landsat8 = (ee.ImageCollection(GEE_CONFIG['satellite_collections']['landsat8'])
                   .filterBounds(self.uzbekistan_aoi)
                   .filterDate(start_date, end_date)
                   .filter(ee.Filter.lt('CLOUD_COVER', GEE_CONFIG['cloud_filter']))
                   .map(self.mask_landsat_clouds)
                   .median()
                   .clip(self.uzbekistan_aoi))
        
        return {
            'sentinel2': sentinel2,
            'landsat8': landsat8
        }
    
    def mask_s2_clouds(self, image):
        """Mask clouds in Sentinel-2 imagery."""
        qa = image.select('QA60')
        cloud_bit_mask = 1 << 10
        cirrus_bit_mask = 1 << 11
        mask = qa.bitwiseAnd(cloud_bit_mask).eq(0).And(qa.bitwiseAnd(cirrus_bit_mask).eq(0))
        return image.updateMask(mask).divide(10000)
    
    def mask_landsat_clouds(self, image):
        """Mask clouds in Landsat imagery."""
        qa = image.select('QA_PIXEL')
        cloud = qa.bitwiseAnd(1 << 3)
        cloud_shadow = qa.bitwiseAnd(1 << 4)
        mask = cloud.eq(0).And(cloud_shadow.eq(0))
        return image.updateMask(mask).multiply(0.0000275).add(-0.2)
    
    def calculate_spectral_indices(self, image):
        """Calculate various spectral indices."""
        # NDVI - Normalized Difference Vegetation Index
        ndvi = image.normalizedDifference(['B8', 'B4']).rename('NDVI')
        
        # NDWI - Normalized Difference Water Index
        ndwi = image.normalizedDifference(['B3', 'B8']).rename('NDWI')
        
        # NDBI - Normalized Difference Built-up Index
        ndbi = image.normalizedDifference(['B11', 'B8']).rename('NDBI')
        
        # EVI - Enhanced Vegetation Index
        evi = image.expression(
            '2.5 * ((NIR - RED) / (NIR + 6 * RED - 7.5 * BLUE + 1))',
            {
                'NIR': image.select('B8'),
                'RED': image.select('B4'),
                'BLUE': image.select('B2')
            }
        ).rename('EVI')
        
        # SAVI - Soil Adjusted Vegetation Index
        savi = image.expression(
            '((NIR - RED) / (NIR + RED + 0.5)) * (1 + 0.5)',
            {
                'NIR': image.select('B8'),
                'RED': image.select('B4')
            }
        ).rename('SAVI')
        
        return image.addBands([ndvi, ndwi, ndbi, evi, savi])
    
    def add_topographic_features(self, image):
        """Add elevation, slope, and aspect from SRTM."""
        srtm = ee.Image('USGS/SRTMGL1_003').clip(self.uzbekistan_aoi)
        elevation = srtm.select('elevation')
        slope = ee.Terrain.slope(elevation)
        aspect = ee.Terrain.aspect(elevation)
        
        return image.addBands([elevation, slope, aspect])
    
    def prepare_training_data(self, training_shapefile_path):
        """Prepare training data from shapefile."""
        print(f"📊 Preparing training data from {training_shapefile_path}")
        
        # Load training shapefile
        training_gdf = gpd.read_file(training_shapefile_path)
        
        # Convert to Earth Engine FeatureCollection
        training_features = []
        for idx, row in training_gdf.iterrows():
            # Extract geometry and properties
            geometry = ee.Geometry(row.geometry.__geo_interface__)
            properties = {'class': row.get('class', 'unknown'), 'layer_id': row.get('layer_id', 0)}
            feature = ee.Feature(geometry, properties)
            training_features.append(feature)
        
        training_fc = ee.FeatureCollection(training_features)
        print(f"✅ Loaded {len(training_features)} training features")
        
        return training_fc
    
    def extract_features_for_training(self, image, training_fc):
        """Extract features from satellite imagery for training points."""
        print("🔍 Extracting features for training data")
        
        # Add spectral indices and topographic features
        image_with_features = self.add_topographic_features(
            self.calculate_spectral_indices(image)
        )
        
        # Select feature bands
        feature_bands = ML_CONFIG['features']
        feature_image = image_with_features.select(feature_bands)
        
        # Sample training data
        training_data = feature_image.sampleRegions(
            collection=training_fc,
            properties=['class', 'layer_id'],
            scale=10,  # 10m resolution
            tileScale=16
        )
        
        return training_data
    
    def export_training_data(self, training_data, filename):
        """Export training data to Google Drive."""
        print(f"💾 Exporting training data to {filename}")
        
        task = ee.batch.Export.table.toDrive(
            collection=training_data,
            description=f'uzbekistan_training_data_{filename}',
            fileFormat='CSV'
        )
        task.start()
        
        print(f"✅ Export task started. Check Google Drive for {filename}")
        return task
    
    def create_classification_composite(self, satellite_data):
        """Create a composite image for classification."""
        # Use Sentinel-2 as primary, fallback to Landsat
        composite = satellite_data['sentinel2']
        
        # Add features
        composite_with_features = self.add_topographic_features(
            self.calculate_spectral_indices(composite)
        )
        
        return composite_with_features.select(ML_CONFIG['features'])
    
    def export_composite_image(self, composite, filename, scale=10):
        """Export composite image for local processing."""
        print(f"📤 Exporting composite image: {filename}")
        
        task = ee.batch.Export.image.toDrive(
            image=composite,
            description=f'uzbekistan_composite_{filename}',
            folder='uzbekistan_classification',
            fileNamePrefix=filename,
            region=self.uzbekistan_aoi,
            scale=scale,
            maxPixels=1e13,
            crs='EPSG:4326'
        )
        task.start()
        
        print(f"✅ Export task started for {filename}")
        return task

def main():
    """Main execution function."""
    print("🚀 Starting Uzbekistan Land Cover Classification - GEE Processing")
    
    # Initialize processor
    processor = UzbekistanEEProcessor()
    
    # Load satellite data
    satellite_data = processor.load_satellite_data()
    
    # Create classification composite
    composite = processor.create_classification_composite(satellite_data)
    
    # Export composite for local processing
    export_task = processor.export_composite_image(
        composite, 
        f"uzbekistan_composite_{datetime.now().strftime('%Y%m%d')}"
    )
    
    print("🎯 Google Earth Engine processing initiated!")
    print("📁 Check your Google Drive for exported data")
    print("⏭️  Next step: Run local QGIS processing script")

if __name__ == "__main__":
    main()
