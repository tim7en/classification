#!/usr/bin/env python3
"""
Utility Script for Selective Tile Downloads

This script allows you to download specific tiles or periods instead of running
the full batch. Useful for testing or selective processing.
"""

import ee
import sys
import json
from pathlib import Path
import argparse
from datetime import datetime

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

# Import our modules
try:
    from config.tile_download_config import (
        UZBEKISTAN_CONFIG, DATE_RANGES, SATELLITE_CONFIG, 
        get_all_tile_info, DEFAULT_SATELLITE
    )
    from scripts.gee.download_tiles_individual import (
        initialize_gee, mask_l2_clouds, get_best_image_for_tile,
        add_enhanced_features, export_tile_to_drive, save_tile_metadata
    )
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure you're running from the project root directory")
    sys.exit(1)

def list_available_options():
    """Show available tiles and periods"""
    print("📋 Available Options:")
    print("\n🗂️  Available Tiles:")
    tiles = get_all_tile_info()
    for i, tile in enumerate(tiles):
        print(f"   {i+1:2d}. {tile['id']} - {tile['bounds']}")
    
    print("\n📅 Available Periods:")
    for i, (period, (start, end)) in enumerate(DATE_RANGES.items()):
        print(f"   {i+1:2d}. {period}: {start} to {end}")

def download_specific_tiles(tile_ids, periods):
    """Download specific tiles for specific periods"""
    print(f"🎯 Downloading tiles: {tile_ids}")
    print(f"📅 For periods: {periods}")
    
    initialize_gee()
    
    # Setup directories
    script_dir = Path(__file__).parent
    project_root = script_dir.parent.parent
    metadata_dir = project_root / 'data' / 'tile_metadata'
    metadata_dir.mkdir(parents=True, exist_ok=True)
    
    # Get all tiles and filter requested ones
    all_tiles = get_all_tile_info()
    tiles_to_process = [tile for tile in all_tiles if tile['id'] in tile_ids]
    
    if not tiles_to_process:
        print("❌ No valid tiles found matching the specified IDs")
        return
    
    # Convert to EE geometries
    for tile in tiles_to_process:
        tile['geometry'] = ee.Geometry.Rectangle(tile['bounds'])
    
    # Setup satellite collection
    uzbekistan_bounds = ee.Geometry.Rectangle(UZBEKISTAN_CONFIG['bounds'])
    satellite_config = SATELLITE_CONFIG[DEFAULT_SATELLITE]
    collection = ee.ImageCollection(satellite_config['collection_id']).filterBounds(uzbekistan_bounds)
    
    total_tasks = 0
    successful_tasks = 0
    
    # Process each period
    for period in periods:
        if period not in DATE_RANGES:
            print(f"⚠️  Unknown period: {period}")
            continue
            
        start_date, end_date = DATE_RANGES[period]
        print(f"\n🔎 Processing period: {period} ({start_date} → {end_date})")
        
        # Filter collection for period
        period_collection = (collection
                           .filterDate(start_date, end_date)
                           .filter(ee.Filter.lt('CLOUD_COVER', satellite_config['cloud_filter_max']))
                           .map(mask_l2_clouds))
        
        total_images = period_collection.size().getInfo()
        if total_images == 0:
            print(f"   ⚠️ No images found for {period}")
            continue
            
        print(f"   ✅ Found {total_images} images")
        
        metadata_file = metadata_dir / f"selective_download_{period}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        # Process each requested tile
        for tile_info in tiles_to_process:
            tile_id = tile_info['id']
            tile_geom = tile_info['geometry']
            
            print(f"\n   🗂️  Processing tile: {tile_id}")
            total_tasks += 1
            
            try:
                # Find best image
                best_image = get_best_image_for_tile(period_collection, tile_geom)
                
                if best_image is None:
                    print(f"      ⚠️ No suitable image found")
                    continue
                
                # Add features
                enhanced_image = add_enhanced_features(best_image, tile_geom)
                
                # Export
                description = f"uzbekistan_{period}_{tile_id}_selective"
                task = export_tile_to_drive(enhanced_image, tile_geom, description, 
                                          scale=satellite_config['scale'])
                
                # Save metadata
                tile_metadata = {
                    'tile_id': tile_id,
                    'period': period,
                    'date_range': [start_date, end_date],
                    'bounds': tile_info['bounds'],
                    'export_description': description,
                    'task_id': task.id if hasattr(task, 'id') else 'unknown',
                    'processing_time': datetime.now().isoformat(),
                    'mode': 'selective_download'
                }
                
                save_tile_metadata(tile_metadata, metadata_file)
                successful_tasks += 1
                
                print(f"      ✅ Export started: {description}")
                
            except Exception as e:
                print(f"      ❌ Error: {e}")
    
    print(f"\n📊 Summary: {successful_tasks}/{total_tasks} tasks started successfully")
    if successful_tasks > 0:
        print("📥 Check GEE Tasks panel for progress")

def main():
    parser = argparse.ArgumentParser(description='Selective tile download utility')
    parser.add_argument('--list', action='store_true', help='List available tiles and periods')
    parser.add_argument('--tiles', nargs='+', help='Tile IDs to download (e.g., tile_00_00 tile_01_01)')
    parser.add_argument('--periods', nargs='+', help='Periods to download (e.g., recent_3_months summer_2024)')
    parser.add_argument('--test-tile', type=str, help='Test download for a single tile')
    parser.add_argument('--test-period', type=str, default='recent_3_months', help='Period for test download')
    
    args = parser.parse_args()
    
    if args.list:
        list_available_options()
        return
    
    if args.test_tile:
        # Test download for single tile
        print(f"🧪 Testing download for tile: {args.test_tile}")
        download_specific_tiles([args.test_tile], [args.test_period])
        return
    
    if args.tiles and args.periods:
        download_specific_tiles(args.tiles, args.periods)
        return
    
    # No arguments provided, show help
    print("🔧 Selective Tile Download Utility")
    print("\nUsage examples:")
    print("  python scripts/gee/selective_download.py --list")
    print("  python scripts/gee/selective_download.py --test-tile tile_01_01")
    print("  python scripts/gee/selective_download.py --tiles tile_00_00 tile_01_01 --periods recent_3_months")
    print("\nUse --help for full options")

if __name__ == '__main__':
    main()
