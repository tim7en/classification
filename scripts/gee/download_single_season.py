#!/usr/bin/env python3
"""
Single Season Tile Download Script for Uzbekistan Land Cover Classification

This script downloads tiles for ONE SEASON ONLY - useful for testing the full workflow
with a manageable amount of data before processing all seasons.

This version processes Summer 2024 (June-August 2024) across all 24 tiles.
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
    get_all_tile_info, DEFAULT_SATELLITE, LANDSAT_ONLY_MODE, print_configuration_summary
)

# Import download functions from main script
from download_tiles_individual import (
    initialize_gee, create_landsat_collection, get_best_image_for_tile,
    add_enhanced_features, export_tile_to_drive, save_tile_metadata,
    mask_l2_clouds
)

def main():
    """Run single season download."""
    initialize_gee()
    
    print("🌞 SINGLE SEASON DOWNLOAD - SUMMER 2024")
    print("="*60)
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
    
    # Get tiles from configuration
    tiles = get_all_tile_info()
    print(f"\n🗺️  Processing {len(tiles)} tiles for single season")
    
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
    
    # SINGLE SEASON PROCESSING - Summer 2024
    target_season = 'summer_2024'
    start_date, end_date = DATE_RANGES[target_season]
    
    print(f"\n🌞 Processing SINGLE SEASON: {target_season} ({start_date} → {end_date})")
    print(f"   📅 This covers June-August 2024 - peak growing season")
    
    # Filter collection for the target season
    period_collection = (collection
                       .filterDate(start_date, end_date)
                       .filter(ee.Filter.lt('CLOUD_COVER', max_cloud_cover))
                       .map(mask_l2_clouds))
    
    total_images = period_collection.size().getInfo()
    if total_images == 0:
        print(f"   ❌ No Landsat images found for {target_season}. Exiting.")
        return
        
    print(f"   ✅ Found {total_images} total Landsat images for {target_season}")
    
    metadata_file = metadata_dir / f"single_season_{target_season}.json"
    successful_tiles = 0
    failed_tiles = []
    
    # Process each tile for the single season
    for tile_idx, tile_info in enumerate(tiles):
        tile_id = tile_info['id']
        tile_geom = tile_info['geometry']
        
        print(f"\n🗂️  Processing tile {tile_idx + 1}/{len(tiles)}: {tile_id}")
        
        if DEBUG_CONFIG['dry_run_mode']:
            print(f"      🏃 DRY RUN MODE - Skipping actual processing")
            continue
        
        try:
            # Find best image for this tile
            best_image = get_best_image_for_tile(period_collection, tile_geom)
            
            if best_image is None:
                print(f"      ⚠️ No suitable Landsat image found for tile {tile_id}")
                failed_tiles.append(tile_id)
                continue
            
            print(f"      ✅ Found suitable Landsat image for tile {tile_id}")
            
            # Add enhanced features
            enhanced_image = add_enhanced_features(best_image, tile_geom)
            
            # Generate export description
            description = f"uzbekistan_{target_season}_{tile_id}_enhanced"
            
            # Export to Google Drive
            task = export_tile_to_drive(
                enhanced_image, tile_geom, description, 
                scale=scale, 
                folder='uzbekistan_summer2024_tiles'  # Special folder for single season
            )
            
            # Save metadata
            tile_metadata = {
                'tile_id': tile_id,
                'period': target_season,
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
                'landsat_only_mode': LANDSAT_ONLY_MODE,
                'single_season_run': True
            }
            
            save_tile_metadata(tile_metadata, metadata_file)
            successful_tiles += 1
            
            print(f"      ✅ Export task started for {tile_id}")
            
            # Progress indicator
            progress = (tile_idx + 1) / len(tiles) * 100
            print(f"      📊 Progress: {progress:.1f}% ({tile_idx + 1}/{len(tiles)} tiles)")
            
            # Delay between exports
            if PROCESSING_CONFIG['delay_between_exports'] > 0:
                time.sleep(PROCESSING_CONFIG['delay_between_exports'])
            
        except Exception as e:
            print(f"      ❌ Error processing tile {tile_id}: {e}")
            failed_tiles.append(tile_id)
            continue
    
    # Final summary
    print(f"\n🎉 SINGLE SEASON DOWNLOAD COMPLETE!")
    print(f"="*60)
    print(f"📊 Season: {target_season} (June-August 2024)")
    print(f"✅ Successful tiles: {successful_tiles}/{len(tiles)}")
    
    if failed_tiles:
        print(f"❌ Failed tiles: {len(failed_tiles)}")
        print(f"   Failed tile IDs: {', '.join(failed_tiles)}")
    
    grid = UZBEKISTAN_CONFIG['grid']
    print(f"\n📋 Processing Summary:")
    print(f"   🗺️  Grid: {grid['size_x']}×{grid['size_y']} = {grid['total_tiles']} tiles")
    print(f"   🛰️  Satellite: {satellite_config.get('description', 'Combined Landsat 8+9')}")
    print(f"   📅 Season: Summer 2024 only")
    print(f"   📁 Google Drive folder: 'uzbekistan_summer2024_tiles'")
    print(f"   📊 Metadata file: {metadata_file}")
    
    if EXPORT_CONFIG['google_drive']['enabled']:
        print(f"\n📥 Next steps:")
        print(f"   1. Check the GEE Tasks panel for export progress")
        print(f"   2. Monitor Google Drive folder: 'uzbekistan_summer2024_tiles'")
        print(f"   3. Once complete, you can run classification on summer 2024 data")
        print(f"   4. If satisfied, run full multi-season download later")
    
    if successful_tiles == len(tiles):
        print(f"\n🎉 Perfect! All {len(tiles)} tiles processed successfully!")
    elif successful_tiles > 0:
        print(f"\n✅ Good! {successful_tiles} tiles processed. Failed tiles can be re-run individually.")
    else:
        print(f"\n⚠️  No tiles were processed successfully. Check configuration and try again.")

if __name__ == '__main__':
    main()
