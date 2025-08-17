"""
Tile-based Download Configuration for Uzbekistan Land Cover Classification

This configuration file defines parameters for the individual tile download approach.
"""

from datetime import datetime, timedelta

# ============ GEOGRAPHIC CONFIGURATION ============
def calculate_optimal_grid_size(bounds, target_tile_size_km=150):
    """
    Calculate optimal grid size based on country dimensions and target tile size.
    
    Args:
        bounds: [min_lon, min_lat, max_lon, max_lat]
        target_tile_size_km: Target tile size in kilometers
    
    Returns:
        Dictionary with grid configuration
    """
    min_lon, min_lat, max_lon, max_lat = bounds
    
    # Approximate conversion from degrees to km at Uzbekistan's latitude
    # 1 degree longitude ≈ 111.32 * cos(latitude) km
    # 1 degree latitude ≈ 111.32 km
    avg_lat = (min_lat + max_lat) / 2
    km_per_degree_lon = 111.32 * abs(np.cos(np.radians(avg_lat)))
    km_per_degree_lat = 111.32
    
    # Calculate country dimensions
    width_degrees = max_lon - min_lon
    height_degrees = max_lat - min_lat
    width_km = width_degrees * km_per_degree_lon
    height_km = height_degrees * km_per_degree_lat
    
    # Calculate optimal grid size
    grid_x = max(1, round(width_km / target_tile_size_km))
    grid_y = max(1, round(height_km / target_tile_size_km))
    
    # Adjust for reasonable tile counts (typically 9-25 tiles work well)
    total_tiles = grid_x * grid_y
    if total_tiles > 25:
        # Scale down proportionally
        scale_factor = (25 / total_tiles) ** 0.5
        grid_x = max(1, round(grid_x * scale_factor))
        grid_y = max(1, round(grid_y * scale_factor))
    elif total_tiles < 9:
        # Scale up proportionally
        scale_factor = (9 / total_tiles) ** 0.5
        grid_x = max(1, round(grid_x * scale_factor))
        grid_y = max(1, round(grid_y * scale_factor))
    
    return {
        'size_x': grid_x,
        'size_y': grid_y,
        'total_tiles': grid_x * grid_y,
        'overlap_buffer': 0.005,  # Small overlap between tiles (degrees)
        'calculated_tile_size_km': {
            'width': width_km / grid_x,
            'height': height_km / grid_y
        },
        'country_dimensions_km': {
            'width': width_km,
            'height': height_km
        }
    }

# Import numpy for calculations
try:
    import numpy as np
except ImportError:
    # Fallback calculation without numpy
    def calculate_optimal_grid_size(bounds, target_tile_size_km=150):
        min_lon, min_lat, max_lon, max_lat = bounds
        avg_lat = (min_lat + max_lat) / 2
        
        # Simplified calculation without numpy
        import math
        km_per_degree_lon = 111.32 * abs(math.cos(math.radians(avg_lat)))
        km_per_degree_lat = 111.32
        
        width_km = (max_lon - min_lon) * km_per_degree_lon
        height_km = (max_lat - min_lat) * km_per_degree_lat
        
        grid_x = max(1, round(width_km / target_tile_size_km))
        grid_y = max(1, round(height_km / target_tile_size_km))
        
        # Keep reasonable bounds
        grid_x = min(max(grid_x, 3), 6)
        grid_y = min(max(grid_y, 3), 6)
        
        return {
            'size_x': grid_x,
            'size_y': grid_y,
            'total_tiles': grid_x * grid_y,
            'overlap_buffer': 0.005,
            'calculated_tile_size_km': {
                'width': width_km / grid_x,
                'height': height_km / grid_y
            },
            'country_dimensions_km': {
                'width': width_km,
                'height': height_km
            }
        }

UZBEKISTAN_CONFIG = {
    # Main bounding box for Uzbekistan (verified coordinates)
    'bounds': [55.9, 37.2, 73.2, 45.6],  # [min_lon, min_lat, max_lon, max_lat]
    
    # Calculate optimal grid automatically
    'grid': None,  # Will be populated below
    
    # UTM zone for Uzbekistan (for area calculations if needed)
    'utm_zone': 'EPSG:32642'  # UTM Zone 42N
}

# Calculate and set the optimal grid
UZBEKISTAN_CONFIG['grid'] = calculate_optimal_grid_size(
    UZBEKISTAN_CONFIG['bounds'], 
    target_tile_size_km=120  # Landsat scenes are ~185km, so smaller tiles work better
)

# ============ TEMPORAL CONFIGURATION ============
def get_date_ranges():
    """Generate date ranges for different seasons/periods"""
    today = datetime.now()
    
    return {
        'recent_3_months': (
            (today - timedelta(days=90)).strftime('%Y-%m-%d'),
            today.strftime('%Y-%m-%d')
        ),
        'recent_6_months': (
            (today - timedelta(days=180)).strftime('%Y-%m-%d'),
            today.strftime('%Y-%m-%d')
        ),
        'summer_2024': ('2024-06-01', '2024-08-31'),
        'winter_2023_2024': ('2023-12-01', '2024-02-29'),
        'spring_2024': ('2024-03-01', '2024-05-31'),
        'autumn_2023': ('2023-09-01', '2023-11-30'),
        'summer_2023': ('2023-06-01', '2023-08-31'),
    }

DATE_RANGES = get_date_ranges()

# ============ SATELLITE DATA CONFIGURATION ============
SATELLITE_CONFIG = {
    'landsat9': {
        'collection_id': 'LANDSAT/LC09/C02/T1_L2',
        'cloud_filter_max': 85,  # Maximum cloud coverage for initial filtering
        'preferred_cloud_max': 50,  # Preferred maximum cloud coverage per tile
        'scale': 30,  # Native resolution in meters
        'bands': {
            'optical': ['SR_B1', 'SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B6', 'SR_B7'],
            'thermal': ['ST_B10'],
            'qa': ['QA_PIXEL']
        },
        'description': 'Landsat 9 Collection 2 Level 2 (primary choice)'
    },
    'landsat8': {
        'collection_id': 'LANDSAT/LC08/C02/T1_L2',
        'cloud_filter_max': 85,
        'preferred_cloud_max': 50,
        'scale': 30,
        'bands': {
            'optical': ['SR_B1', 'SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B6', 'SR_B7'],
            'thermal': ['ST_B10'],
            'qa': ['QA_PIXEL']
        },
        'description': 'Landsat 8 Collection 2 Level 2 (backup option)'
    },
    'landsat_combined': {
        'collection_ids': ['LANDSAT/LC09/C02/T1_L2', 'LANDSAT/LC08/C02/T1_L2'],
        'cloud_filter_max': 85,
        'preferred_cloud_max': 50,
        'scale': 30,
        'bands': {
            'optical': ['SR_B1', 'SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B6', 'SR_B7'],
            'thermal': ['ST_B10'],
            'qa': ['QA_PIXEL']
        },
        'description': 'Combined Landsat 8 & 9 for maximum coverage'
    }
}

# ============ QUALITY ASSESSMENT CONFIGURATION ============
QUALITY_CONFIG = {
    'min_valid_pixel_ratio': 0.6,  # Minimum ratio of valid pixels per tile
    'preferred_valid_pixel_ratio': 0.8,  # Preferred ratio for high-quality tiles
    'max_candidates_to_evaluate': 15,  # Maximum number of images to evaluate per tile
    'scoring_weights': {
        'cloud_cover': 0.4,  # Weight for cloud coverage in scoring
        'valid_pixels': 0.6  # Weight for valid pixel ratio in scoring
    }
}

# ============ FEATURE EXTRACTION CONFIGURATION ============
FEATURE_CONFIG = {
    'spectral_indices': {
        'vegetation': ['NDVI', 'EVI', 'SAVI', 'GNDVI'],
        'water': ['NDWI', 'MNDWI', 'AWEI_NS', 'AWEI_SH'],
        'built_up': ['NDBI', 'NDBAI', 'UI'],
        'bare_soil': ['BSI', 'NDSI']
    },
    'topographic_features': {
        'elevation': True,
        'slope': True,
        'aspect': True,
        'hillshade': False  # Can be computationally expensive
    },
    'texture_features': {
        'enabled': True,
        'kernel_size': 3,  # 3x3 kernel for texture analysis
        'bands_for_texture': ['NIR']  # Which bands to compute texture for
    },
    'thermal_features': {
        'enabled': True,
        'convert_to_celsius': True
    }
}

# ============ EXPORT CONFIGURATION ============
EXPORT_CONFIG = {
    'google_drive': {
        'enabled': True,
        'folder_name': 'uzbekistan_tiles',
        'file_format': 'GeoTIFF',
        'max_pixels': 1e9
    },
    'local_export': {
        'enabled': False,  # Set to True for local downloads via geemap
        'require_geemap': True,
        'folder_name': 'downloaded_tiles'
    },
    'naming_convention': {
        'pattern': 'uzbekistan_{period}_{tile_id}_enhanced',
        'include_date': False,
        'include_sensor': False
    }
}

# ============ PROCESSING CONFIGURATION ============
PROCESSING_CONFIG = {
    'batch_size': 1,  # Number of tiles to process simultaneously
    'retry_attempts': 3,  # Number of retry attempts for failed tiles
    'delay_between_exports': 2,  # Seconds to wait between export tasks
    'memory_optimization': True,  # Use memory-efficient processing
    'progress_tracking': {
        'save_metadata': True,
        'metadata_format': 'json',
        'include_statistics': True
    }
}

# ============ DEBUGGING AND LOGGING ============
DEBUG_CONFIG = {
    'verbose_logging': True,
    'save_processing_logs': True,
    'log_image_statistics': True,
    'log_quality_metrics': True,
    'dry_run_mode': False  # Set to True to test without actually exporting
}

# ============ DERIVED CONFIGURATIONS ============
def get_tile_bounds(tile_x: int, tile_y: int) -> list:
    """
    Calculate the bounds for a specific tile given its grid position.
    
    Args:
        tile_x: X position in grid (0 to grid_size_x-1)
        tile_y: Y position in grid (0 to grid_size_y-1)
    
    Returns:
        [min_lon, min_lat, max_lon, max_lat]
    """
    bounds = UZBEKISTAN_CONFIG['bounds']
    grid = UZBEKISTAN_CONFIG['grid']
    
    min_lon, min_lat, max_lon, max_lat = bounds
    
    lon_step = (max_lon - min_lon) / grid['size_x']
    lat_step = (max_lat - min_lat) / grid['size_y']
    
    tile_min_lon = min_lon + tile_x * lon_step
    tile_max_lon = min_lon + (tile_x + 1) * lon_step
    tile_min_lat = min_lat + tile_y * lat_step
    tile_max_lat = min_lat + (tile_y + 1) * lat_step
    
    return [tile_min_lon, tile_min_lat, tile_max_lon, tile_max_lat]

def get_all_tile_info() -> list:
    """
    Generate information for all tiles in the grid.
    
    Returns:
        List of dictionaries with tile information
    """
    grid = UZBEKISTAN_CONFIG['grid']
    tiles = []
    
    for i in range(grid['size_x']):
        for j in range(grid['size_y']):
            tile_info = {
                'id': f"tile_{i:02d}_{j:02d}",
                'grid_x': i,
                'grid_y': j,
                'bounds': get_tile_bounds(i, j),
                'area_km2': None  # Can be calculated later if needed
            }
            tiles.append(tile_info)
    
    return tiles

# Make commonly used configurations easily accessible
DEFAULT_SATELLITE = 'landsat_combined'  # Use both Landsat 8 & 9 for best coverage
LANDSAT_ONLY_MODE = True  # Flag to indicate Landsat-only processing
DEFAULT_PERIODS = ['recent_3_months', 'summer_2024', 'winter_2023_2024']

# Print configuration summary
def print_configuration_summary():
    """Print a summary of the current configuration."""
    grid = UZBEKISTAN_CONFIG['grid']
    print(f"🗺️  Uzbekistan Coverage Configuration:")
    print(f"   📐 Grid size: {grid['size_x']} × {grid['size_y']} = {grid['total_tiles']} tiles")
    if 'calculated_tile_size_km' in grid:
        tile_size = grid['calculated_tile_size_km']
        print(f"   📏 Avg tile size: {tile_size['width']:.0f} × {tile_size['height']:.0f} km")
        country_size = grid['country_dimensions_km']
        print(f"   🌍 Country size: {country_size['width']:.0f} × {country_size['height']:.0f} km")
    print(f"   🛰️  Satellite: {DEFAULT_SATELLITE.upper()} (Landsat only: {LANDSAT_ONLY_MODE})")
    print(f"   📅 Default periods: {len(DEFAULT_PERIODS)} time ranges")

# Calculate total tiles dynamically
TOTAL_TILES = UZBEKISTAN_CONFIG['grid']['total_tiles']
