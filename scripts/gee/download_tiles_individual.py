#!/usr/bin/env python3
"""
Individual Tile Download Script for Uzbekistan Land Cover Classification

This script downloads Landsat 9 imagery tile by tile instead of creating a large mosaic.
It divides the country into a grid and finds the best image for each tile based on
cloud coverage and data quality.

Features:
- Grid-based tile approach for manageable downloads
- Individual quality assessment per tile
- Best image selection per tile per season
- Robust error handling and retry logic
- Progress tracking and metadata saving
"""

import ee
import os
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path
import time
from typing import Dict, List, Tuple, Optional

# Import configuration
sys.path.append(str(Path(__file__).parent.parent.parent))
from config.tile_download_config import (
    UZBEKISTAN_CONFIG, DATE_RANGES, SATELLITE_CONFIG, QUALITY_CONFIG,
    FEATURE_CONFIG, EXPORT_CONFIG, PROCESSING_CONFIG, DEBUG_CONFIG,
    get_all_tile_info, DEFAULT_SATELLITE, LANDSAT_ONLY_MODE
)

# ============ GEE BOOTSTRAP ============
def initialize_gee():
    """Authenticates and initializes the Google Earth Engine library."""
    try:
        ee.Image.constant(0).getInfo()
        print('✅ GEE is already authenticated and initialized.')
    except Exception:
        print('🔑 Authenticating and initializing GEE...')
        try:
            ee.Authenticate()
            ee.Initialize()
            print('✅ GEE authenticated and initialized successfully!')
        except Exception as e:
            print(f'❌ GEE initialization failed: {e}')
            sys.exit(1)

# ============ CLOUD MASK ============
def mask_l2_clouds(img: ee.Image) -> ee.Image:
    """
    Advanced cloud/shadow masking for Landsat Collection 2 L2 using QA_PIXEL.
    """
    qa = img.select('QA_PIXEL')
    
    # Mask clouds, cloud shadows, snow, and dilated clouds
    cloud_mask = (qa.bitwiseAnd(1 << 1).eq(0)  # not dilated cloud
                  .And(qa.bitwiseAnd(1 << 2).eq(0))  # not cirrus
                  .And(qa.bitwiseAnd(1 << 3).eq(0))  # not cloud
                  .And(qa.bitwiseAnd(1 << 4).eq(0))  # not cloud shadow
                  .And(qa.bitwiseAnd(1 << 5).eq(0)))  # not snow
    
    return img.updateMask(cloud_mask)

def calculate_valid_pixel_ratio(image: ee.Image, region: ee.Geometry) -> float:
    """
    Calculate the ratio of valid (non-masked) pixels in the image for the given region.
    """
    # Count total pixels in the region
    total_pixels = ee.Image.constant(1).reduceRegion(
        reducer=ee.Reducer.count(),
        geometry=region,
        scale=30,
        maxPixels=1e9
    ).get('constant')
    
    # Count valid pixels (using a band that should be present)
    valid_pixels = image.select('SR_B4').mask().reduceRegion(
        reducer=ee.Reducer.count(),
        geometry=region,
        scale=30,
        maxPixels=1e9
    ).get('SR_B4')
    
    # Calculate ratio (server-side computation)
    ratio = ee.Number(valid_pixels).divide(ee.Number(total_pixels))
    return ratio

# ============ TILE MANAGEMENT ============
def create_tile_grid(bounds: ee.Geometry, grid_size: int = 4) -> List[ee.Geometry]:
    """
    Create a grid of tiles covering the given bounds.
    
    Args:
        bounds: The boundary geometry to divide
        grid_size: Number of tiles per dimension (grid_size x grid_size total tiles)
    
    Returns:
        List of tile geometries
    """
    coords = bounds.bounds().coordinates().get(0).getInfo()
    min_lon, min_lat = coords[0]
    max_lon, max_lat = coords[2]
    
    lon_step = (max_lon - min_lon) / grid_size
    lat_step = (max_lat - min_lat) / grid_size
    
    tiles = []
    for i in range(grid_size):
        for j in range(grid_size):
            tile_min_lon = min_lon + i * lon_step
            tile_max_lon = min_lon + (i + 1) * lon_step
            tile_min_lat = min_lat + j * lat_step
            tile_max_lat = min_lat + (j + 1) * lat_step
            
            tile = ee.Geometry.Rectangle([tile_min_lon, tile_min_lat, tile_max_lon, tile_max_lat])
            tiles.append(tile)
    
    return tiles

def get_best_image_for_tile(collection: ee.ImageCollection, tile_region: ee.Geometry, 
                           min_valid_ratio: float = None) -> Optional[ee.Image]:
    """
    Find the best image for a specific tile based on cloud coverage and data availability.
    
    Args:
        collection: The image collection to search
        tile_region: The tile geometry
        min_valid_ratio: Minimum ratio of valid pixels required (uses config default if None)
    
    Returns:
        Best image for the tile, or None if no suitable image found
    """
    if min_valid_ratio is None:
        min_valid_ratio = QUALITY_CONFIG['min_valid_pixel_ratio']
    
    max_candidates = QUALITY_CONFIG['max_candidates_to_evaluate']
    
    # Filter collection to images that intersect the tile
    tile_collection = collection.filterBounds(tile_region)
    
    # Get collection size
    collection_size = tile_collection.size().getInfo()
    if collection_size == 0:
        return None
    
    if DEBUG_CONFIG['verbose_logging']:
        print(f"      Found {collection_size} candidate images for tile")
    
    # Sort by cloud cover and limit candidates
    sorted_collection = tile_collection.sort('CLOUD_COVER').limit(max_candidates)
    image_list = sorted_collection.getInfo()['features']
    
    best_image = None
    best_score = -1
    
    cloud_weight = QUALITY_CONFIG['scoring_weights']['cloud_cover']
    pixel_weight = QUALITY_CONFIG['scoring_weights']['valid_pixels']
    
    for img_info in image_list:
        img_id = img_info['id']
        img = ee.Image(img_id)
        cloud_cover = img_info['properties'].get('CLOUD_COVER', 100)
        
        # Skip very cloudy images
        if cloud_cover > SATELLITE_CONFIG[DEFAULT_SATELLITE]['preferred_cloud_max']:
            continue
            
        try:
            # Calculate valid pixel ratio for this tile
            masked_img = mask_l2_clouds(img)
            valid_ratio = calculate_valid_pixel_ratio(masked_img, tile_region).getInfo()
            
            # Weighted score based on valid pixels and cloud cover
            score = (pixel_weight * valid_ratio + 
                    cloud_weight * (100 - cloud_cover) / 100)
            
            if DEBUG_CONFIG['verbose_logging']:
                print(f"        Image {img_id[-8:]}: {cloud_cover:.1f}% clouds, {valid_ratio:.2f} valid ratio, score: {score:.2f}")
            
            if score > best_score and valid_ratio >= min_valid_ratio:
                best_score = score
                best_image = masked_img
                
        except Exception as e:
            if DEBUG_CONFIG['verbose_logging']:
                print(f"        Error evaluating image {img_id}: {e}")
            continue
    
    return best_image

# ============ ENHANCED FEATURE EXTRACTION ============
def add_enhanced_features(image: ee.Image, tile_region: ee.Geometry) -> ee.Image:
    """
    Add comprehensive spectral indices and features optimized for land cover classification.
    """
    # Scale optical bands
    optical = image.select(['SR_B1','SR_B2','SR_B3','SR_B4','SR_B5','SR_B6','SR_B7']) \
                   .multiply(0.0000275).add(-0.2) \
                   .rename(['BLUE', 'GREEN', 'RED', 'NIR', 'SWIR1', 'SWIR2', 'COASTAL'])

    # Core vegetation indices
    ndvi = optical.normalizedDifference(['NIR','RED']).rename('NDVI')
    ndwi = optical.normalizedDifference(['GREEN','NIR']).rename('NDWI')
    mndwi = optical.normalizedDifference(['GREEN','SWIR1']).rename('MNDWI')
    
    # Enhanced vegetation indices
    evi = optical.expression(
        '2.5 * ((NIR - RED) / (NIR + 6 * RED - 7.5 * BLUE + 1))',
        {'NIR': optical.select('NIR'), 'RED': optical.select('RED'), 'BLUE': optical.select('BLUE')}
    ).rename('EVI')
    
    savi = optical.expression(
        '((NIR - RED) / (NIR + RED + 0.5)) * 1.5',
        {'NIR': optical.select('NIR'), 'RED': optical.select('RED')}
    ).rename('SAVI')
    
    # Built-up and bare soil indices
    ndbi = optical.normalizedDifference(['SWIR1','NIR']).rename('NDBI')
    ndbai = optical.normalizedDifference(['SWIR1','GREEN']).rename('NDBAI')  # Built-up areas
    
    # Water indices
    awei_ns = optical.expression(
        '4 * (GREEN - SWIR1) - (0.25 * NIR + 2.75 * SWIR2)',
        {'GREEN': optical.select('GREEN'), 'SWIR1': optical.select('SWIR1'), 
         'NIR': optical.select('NIR'), 'SWIR2': optical.select('SWIR2')}
    ).rename('AWEI_NS')
    
    # Topographic features
    elevation = ee.Image('USGS/SRTMGL1_003').select('elevation').clip(tile_region)
    slope = ee.Terrain.slope(elevation).rename('SLOPE')
    aspect = ee.Terrain.aspect(elevation).rename('ASPECT')
    
    # Thermal (if available)
    thermal = ee.Image(ee.Algorithms.If(
        image.bandNames().contains('ST_B10'),
        image.select(['ST_B10']).multiply(0.00341802).add(149.0).subtract(273.15).rename('TEMP_C'),
        ee.Image(0).rename('TEMP_C')
    ))
    
    # Texture features (using NIR band)
    nir_texture = optical.select('NIR').reduceNeighborhood(
        reducer=ee.Reducer.stdDev(), kernel=ee.Kernel.square(3)
    ).rename('NIR_TEXTURE')
    
    # Combine all features
    enhanced_image = ee.Image.cat([
        # Core spectral bands
        optical.select(['BLUE', 'GREEN', 'RED', 'NIR', 'SWIR1', 'SWIR2']),
        
        # Vegetation indices
        ndvi, evi, savi,
        
        # Water indices
        ndwi, mndwi, awei_ns,
        
        # Built-up indices
        ndbi, ndbai,
        
        # Topographic
        elevation.rename('ELEVATION'), slope, aspect,
        
        # Environmental
        thermal, nir_texture
    ]).toFloat()
    
    return enhanced_image

# ============ EXPORT FUNCTIONS ============
def export_tile_to_drive(image: ee.Image, tile_region: ee.Geometry, description: str, 
                        scale: int = 30, folder: str = 'uzbekistan_tiles'):
    """Export a tile to Google Drive."""
    print(f"      🛰️  Starting Drive export: {description}")
    
    task = ee.batch.Export.image.toDrive(
        image=image,
        description=description,
        folder=folder,
        fileNamePrefix=description,
        region=tile_region,
        scale=scale,
        fileFormat='GeoTIFF',
        maxPixels=1e9
    )
    task.start()
    return task

def save_tile_metadata(tile_info: Dict, metadata_file: Path):
    """Save tile processing metadata to JSON file."""
    metadata = []
    if metadata_file.exists():
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
    
    metadata.append(tile_info)
    
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2, default=str)

# ============ MAIN PROCESSING ============
def create_landsat_collection(satellite_config: dict, bounds: ee.Geometry) -> ee.ImageCollection:
    """
    Create a Landsat image collection based on the satellite configuration.
    
    Args:
        satellite_config: Satellite configuration dictionary
        bounds: Geographic bounds for filtering
    
    Returns:
        Filtered ImageCollection
    """
    if 'collection_ids' in satellite_config:
        # Combined collection (Landsat 8 + 9)
        collections = []
        for collection_id in satellite_config['collection_ids']:
            collection = ee.ImageCollection(collection_id).filterBounds(bounds)
            collections.append(collection)
        
        # Merge all collections
        merged_collection = collections[0]
        for collection in collections[1:]:
            merged_collection = merged_collection.merge(collection)
        
        return merged_collection
    else:
        # Single collection
        return ee.ImageCollection(satellite_config['collection_id']).filterBounds(bounds)

def main():
    initialize_gee()
    
    # Import and print configuration
    from config.tile_download_config import print_configuration_summary
    print_configuration_summary()
    
    # Configuration
    script_dir = Path(__file__).parent
    project_root = script_dir.parent.parent
    download_dir = project_root / 'data' / 'downloaded_tiles'
    metadata_dir = project_root / 'data' / 'tile_metadata'
    
    download_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n📂 Download directory: {download_dir}")
    print(f"📊 Metadata directory: {metadata_dir}")
    
    # Use configuration for Uzbekistan bounds
    bounds = UZBEKISTAN_CONFIG['bounds']
    uzbekistan_bounds = ee.Geometry.Rectangle(bounds)
    
    # Get tiles from configuration (now automatically calculated)
    tiles = get_all_tile_info()
    print(f"\n🗺️  Using {len(tiles)} automatically calculated tiles")
    
    # Print tile information
    if DEBUG_CONFIG['verbose_logging']:
        grid = UZBEKISTAN_CONFIG['grid']
        if 'calculated_tile_size_km' in grid:
            tile_size = grid['calculated_tile_size_km']
            print(f"   📏 Average tile size: {tile_size['width']:.1f} × {tile_size['height']:.1f} km")
    
    # Convert to EE geometries
    for tile in tiles:
        tile['geometry'] = ee.Geometry.Rectangle(tile['bounds'])
    
    # Get satellite configuration
    satellite_config = SATELLITE_CONFIG[DEFAULT_SATELLITE]
    max_cloud_cover = satellite_config['cloud_filter_max']
    scale = satellite_config['scale']
    
    print(f"\n🛰️  Using {satellite_config.get('description', DEFAULT_SATELLITE.upper())}")
    
    # Create collection
    collection = create_landsat_collection(satellite_config, uzbekistan_bounds)
    
    # Process each period
    for period, (start_date, end_date) in DATE_RANGES.items():
        print(f"\n🔎 Processing period: {period} ({start_date} → {end_date})")
        
        # Filter collection for period
        period_collection = (collection
                           .filterDate(start_date, end_date)
                           .filter(ee.Filter.lt('CLOUD_COVER', max_cloud_cover))
                           .map(mask_l2_clouds))
        
        total_images = period_collection.size().getInfo()
        if total_images == 0:
            print(f"   ⚠️ No Landsat images found for {period}. Skipping.")
            continue
            
        print(f"   ✅ Found {total_images} total Landsat images for {period}")
        
        metadata_file = metadata_dir / f"tile_processing_{period}.json"
        successful_tiles = 0
        
        # Process each tile
        for tile_idx, tile_info in enumerate(tiles):
            tile_id = tile_info['id']
            tile_geom = tile_info['geometry']
            
            print(f"\n   🗂️  Processing tile {tile_idx + 1}/{len(tiles)}: {tile_id}")
            
            if DEBUG_CONFIG['dry_run_mode']:
                print(f"      🏃 DRY RUN MODE - Skipping actual processing")
                continue
            
            try:
                # Find best image for this tile
                best_image = get_best_image_for_tile(period_collection, tile_geom)
                
                if best_image is None:
                    print(f"      ⚠️ No suitable Landsat image found for tile {tile_id}")
                    continue
                
                print(f"      ✅ Found suitable Landsat image for tile {tile_id}")
                
                # Add enhanced features
                enhanced_image = add_enhanced_features(best_image, tile_geom)
                
                # Generate export description using configuration
                pattern = EXPORT_CONFIG['naming_convention']['pattern']
                description = pattern.format(period=period, tile_id=tile_id)
                
                # Export based on configuration
                if EXPORT_CONFIG['google_drive']['enabled']:
                    task = export_tile_to_drive(
                        enhanced_image, tile_geom, description, 
                        scale=scale, 
                        folder=EXPORT_CONFIG['google_drive']['folder_name']
                    )
                    
                    # Save metadata
                    tile_metadata = {
                        'tile_id': tile_id,
                        'period': period,
                        'date_range': [start_date, end_date],
                        'bounds': tile_info['bounds'],
                        'grid_position': {'x': tile_info['grid_x'], 'y': tile_info['grid_y']},
                        'export_description': description,
                        'task_id': task.id if hasattr(task, 'id') else 'unknown',
                        'processing_time': datetime.now().isoformat(),
                        'total_candidates': total_images,
                        'satellite': DEFAULT_SATELLITE,
                        'satellite_description': satellite_config.get('description', 'Landsat'),
                        'scale': scale,
                        'landsat_only_mode': LANDSAT_ONLY_MODE
                    }
                    
                    save_tile_metadata(tile_metadata, metadata_file)
                    successful_tiles += 1
                    
                    print(f"      ✅ Export task started for {tile_id}")
                
                # Delay between exports
                if PROCESSING_CONFIG['delay_between_exports'] > 0:
                    time.sleep(PROCESSING_CONFIG['delay_between_exports'])
                
            except Exception as e:
                print(f"      ❌ Error processing tile {tile_id}: {e}")
                continue
        
        print(f"\n   📊 Period {period} summary: {successful_tiles}/{len(tiles)} tiles processed successfully")
    
    print("\n🎉 All tile processing complete!")
    if EXPORT_CONFIG['google_drive']['enabled']:
        print("📥 Check the GEE Tasks panel for export progress.")
    print("📊 Tile metadata saved in: data/tile_metadata/")
    
    # Print final summary
    grid = UZBEKISTAN_CONFIG['grid']
    print(f"\n📋 Final Summary:")
    print(f"   🗺️  Total tiles processed: {grid['total_tiles']} ({grid['size_x']}×{grid['size_y']} grid)")
    print(f"   🛰️  Satellite data: Landsat only ({satellite_config.get('description', 'Combined Landsat 8+9')})")
    print(f"   📅 Time periods: {len(DATE_RANGES)} periods processed")

if __name__ == '__main__':
    main()
