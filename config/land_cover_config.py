"""
Land Cover Classification Configuration for Uzbekistan
Defines all land cover classes and their properties for the classification system.
"""

# Land cover class definitions
LAND_COVER_CLASSES = {
    1: {
        'name': 'residential',
        'description': 'Residential areas and housing',
        'color': '#FF6B6B',  # Red
        'source': 'APPHOUSE_production_ready'
    },
    2: {
        'name': 'agriculture',
        'description': 'Agricultural fields and farmland',
        'color': '#4ECDC4',  # Teal
        'source': 'AgricultureFields_production_ready'
    },
    3: {
        'name': 'residential',
        'description': 'Buildings and residential structures',
        'color': '#FF6B6B',  # Red (same as class 1)
        'source': 'Buildings_production_ready'
    },
    4: {
        'name': 'forest',
        'description': 'Forest agency lands and wooded areas',
        'color': '#45B7D1',  # Blue
        'source': 'ForestAgencyLands_production_ready'
    },
    5: {
        'name': 'residential_private',
        'description': 'High resolution private residential areas',
        'color': '#F7DC6F',  # Yellow
        'source': 'HighResPrivate_production_ready'
    },
    6: {
        'name': 'roads_highways',
        'description': 'Major highways and road networks',
        'color': '#2C3E50',  # Dark blue-gray
        'source': 'Highways_production_ready'
    },
    7: {
        'name': 'land_stock',
        'description': 'Land stock and reserve areas',
        'color': '#E67E22',  # Orange
        'source': 'LandStock_production_ready'
    },
    8: {
        'name': 'non_residential',
        'description': 'Non-residential buildings and infrastructure',
        'color': '#9B59B6',  # Purple
        'source': 'NotResidential_production_ready'
    },
    9: {
        'name': 'protected',
        'description': 'Protected areas and conservation zones',
        'color': '#27AE60',  # Green
        'source': 'ProtectedAreas_production_ready'
    },
    10: {
        'name': 'railways',
        'description': 'Railway lines and rail infrastructure',
        'color': '#34495E',  # Dark gray
        'source': 'Railways_production_ready'
    },
    11: {
        'name': 'shared_lands',
        'description': 'Shared and communal land areas',
        'color': '#F39C12',  # Amber
        'source': 'SharedLands_production_ready'
    },
    12: {
        'name': 'water',
        'description': 'Water bodies, rivers, and lakes',
        'color': '#3498DB',  # Light blue
        'source': 'Water_production_ready'
    }
}

# Merged class mapping for simplified classification
MERGED_CLASSES = {
    'residential': [1, 3, 5],  # Combine all residential types
    'agriculture': [2],
    'forest': [4],
    'infrastructure': [6, 10],  # Roads and railways
    'land_stock': [7],
    'non_residential': [8],
    'protected': [9],
    'shared_lands': [11],
    'water': [12]
}

# Google Earth Engine configuration
GEE_CONFIG = {
    'project_id': 'your-gee-project-id',  # Replace with your GEE project ID
    'uzbekistan_bounds': {
        'coordinates': [
            [55.997, 37.172],  # Southwest corner
            [73.055, 37.172],  # Southeast corner
            [73.055, 45.573],  # Northeast corner
            [55.997, 45.573],  # Northwest corner
            [55.997, 37.172]   # Close polygon
        ]
    },
    'satellite_collections': {
        'sentinel2': 'COPERNICUS/S2_SR_HARMONIZED',
        'landsat8': 'LANDSAT/LC08/C02/T1_L2',
        'landsat9': 'LANDSAT/LC09/C02/T1_L2'
    },
    'date_range': {
        'start': '2023-01-01',
        'end': '2024-12-31'
    },
    'cloud_filter': 20  # Maximum cloud coverage percentage
}

# QGIS Processing configuration
QGIS_CONFIG = {
    'crs': 'EPSG:4326',  # WGS84
    'utm_zone': 'EPSG:32642',  # UTM Zone 42N for Uzbekistan
    'tile_size': 10000,  # 10km tiles for processing
    'buffer_size': 500,  # 500m buffer for tiles
    'output_format': 'GeoTIFF'
}

# Machine Learning configuration
ML_CONFIG = {
    'algorithms': {
        'random_forest': {
            'n_estimators': 100,
            'max_depth': 10,
            'min_samples_split': 2
        },
        'gradient_boosting': {
            'n_estimators': 100,
            'learning_rate': 0.1,
            'max_depth': 8
        },
        'svm': {
            'kernel': 'rbf',
            'C': 1.0,
            'gamma': 'scale'
        }
    },
    'features': [
        'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B11', 'B12',  # Sentinel-2 bands
        'NDVI', 'NDWI', 'NDBI', 'EVI', 'SAVI',  # Spectral indices
        'elevation', 'slope', 'aspect'  # Topographic features
    ],
    'validation_split': 0.2,
    'cross_validation_folds': 5
}
