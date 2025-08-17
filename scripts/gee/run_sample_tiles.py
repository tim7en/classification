#!/usr/bin/env python3
"""
Sample Tile Download Script

This script downloads just a few tiles for testing purposes.
Use this to validate the workflow before running the full download.
"""

import ee
import os
import json
import sys
from datetime import datetime
from pathlib import Path

# Import configuration
sys.path.append(str(Path(__file__).parent.parent.parent))
from config.tile_download_config import (
    UZBEKISTAN_CONFIG, DATE_RANGES, SATELLITE_CONFIG,
    get_all_tile_info, DEFAULT_SATELLITE, print_configuration_summary
)

# Import download functions
from download_tiles_individual import (
    initialize_gee, create_landsat_collection, get_best_image_for_tile,
    add_enhanced_features, export_tile_to_drive, save_tile_metadata,
    mask_l2_clouds
)

def main():
    """Run sample tile downloads."""
    initialize_gee()
    
    print("🧪 SAMPLE TILE DOWNLOAD")
    print("="*50)
    print_configuration_summary()
    
    # Configuration
    script_dir = Path(__file__).parent
    project_root = script_dir.parent.parent
    metadata_dir = project_root / 'data' / 'tile_metadata'
    metadata_dir.mkdir(parents=True, exist_ok=True)
    
    # Get tiles and select just a few for testing
    all_tiles = get_all_tile_info()
    
    # Select 3 tiles from different areas
    sample_indices = [0, len(all_tiles)//2, len(all_tiles)-1]  # First, middle, last
    sample_tiles = [all_tiles[i] for i in sample_indices]
    
    print(f"\n🗂️  Selected {len(sample_tiles)} sample tiles from {len(all_tiles)} total:")
    for tile in sample_tiles:
        print(f"   - {tile['id']}: {tile['bounds']}")
    
    # Convert to EE geometries
    for tile in sample_tiles:
        tile['geometry'] = ee.Geometry.Rectangle(tile['bounds'])
    
    # Use Uzbekistan bounds for collection filtering
    bounds = UZBEKISTAN_CONFIG['bounds']
    uzbekistan_bounds = ee.Geometry.Rectangle(bounds)
    
    # Get satellite configuration
    satellite_config = SATELLITE_CONFIG[DEFAULT_SATELLITE]
    scale = satellite_config['scale']
    
    # Create collection
    collection = create_landsat_collection(satellite_config, uzbekistan_bounds)
    
    # Use just one recent period for testing
    test_period = 'recent_3_months'
    start_date, end_date = DATE_RANGES[test_period]
    
    print(f"\n📅 Processing period: {test_period} ({start_date} → {end_date})")
    
    # Filter collection
    period_collection = (collection
                       .filterDate(start_date, end_date)
                       .filter(ee.Filter.lt('CLOUD_COVER', satellite_config['cloud_filter_max']))
                       .map(mask_l2_clouds))
    
    total_images = period_collection.size().getInfo()
    print(f"   🛰️  Found {total_images} total Landsat images")
    
    if total_images == 0:
        print("   ❌ No images found. Exiting.")
        return
    
    metadata_file = metadata_dir / f"sample_tiles_{test_period}.json"
    successful_tiles = 0
    
    # Process each sample tile
    for tile_idx, tile_info in enumerate(sample_tiles):
        tile_id = tile_info['id']
        tile_geom = tile_info['geometry']
        
        print(f"\n🗂️  Processing sample tile {tile_idx + 1}/{len(sample_tiles)}: {tile_id}")
        
        try:
            # Find best image for this tile
            best_image = get_best_image_for_tile(period_collection, tile_geom)
            
            if best_image is None:
                print(f"      ⚠️ No suitable image found for tile {tile_id}")
                continue
            
            print(f"      ✅ Found suitable image for tile {tile_id}")
            
            # Add enhanced features
            enhanced_image = add_enhanced_features(best_image, tile_geom)
            
            # Export description
            description = f"uzbekistan_sample_{test_period}_{tile_id}_enhanced"
            
            # Export to Google Drive
            task = export_tile_to_drive(
                enhanced_image, tile_geom, description, 
                scale=scale, 
                folder='uzbekistan_sample_tiles'
            )
            
            # Save metadata
            tile_metadata = {
                'tile_id': tile_id,
                'period': test_period,
                'date_range': [start_date, end_date],
                'bounds': tile_info['bounds'],
                'grid_position': {'x': tile_info['grid_x'], 'y': tile_info['grid_y']},
                'export_description': description,
                'task_id': task.id if hasattr(task, 'id') else 'unknown',
                'processing_time': datetime.now().isoformat(),
                'total_candidates': total_images,
                'satellite': DEFAULT_SATELLITE,
                'scale': scale,
                'sample_run': True
            }
            
            save_tile_metadata(tile_metadata, metadata_file)
            successful_tiles += 1
            
            print(f"      ✅ Export task started: {description}")
            
        except Exception as e:
            print(f"      ❌ Error processing tile {tile_id}: {e}")
            continue
    
    print(f"\n📊 Sample run summary:")
    print(f"   ✅ {successful_tiles}/{len(sample_tiles)} sample tiles processed successfully")
    print(f"   📥 Check GEE Tasks panel for export progress")
    print(f"   📊 Metadata saved to: {metadata_file}")
    
    if successful_tiles > 0:
        print(f"\n🎉 Sample download started successfully!")
        print(f"   📁 Files will appear in Google Drive folder: 'uzbekistan_sample_tiles'")
        print(f"   🔍 Once verified, run the full script: python scripts/gee/download_tiles_individual.py")
    else:
        print(f"\n⚠️  No tiles were processed successfully. Check the configuration.")

if __name__ == '__main__':
    main()
