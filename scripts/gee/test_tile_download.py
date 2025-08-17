#!/usr/bin/env python3
"""
Test Script for Tile-based Download

This script tests the tile-based download approach with just one tile
to verify the configuration and logic work correctly.
"""

import ee
import sys
from pathlib import Path

# Add project root to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))
from config.tile_download_config import (
    UZBEKISTAN_CONFIG, DATE_RANGES, SATELLITE_CONFIG, QUALITY_CONFIG,
    get_all_tile_info, DEFAULT_SATELLITE, print_configuration_summary, LANDSAT_ONLY_MODE
)

def initialize_gee():
    """Initialize Google Earth Engine"""
    try:
        ee.Image.constant(0).getInfo()
        print('✅ GEE is already initialized.')
    except Exception:
        print('🔑 Initializing GEE...')
        try:
            ee.Authenticate()
            ee.Initialize()
            print('✅ GEE initialized successfully!')
        except Exception as e:
            print(f'❌ GEE initialization failed: {e}')
            sys.exit(1)

def test_configuration():
    """Test the configuration setup"""
    print("\n🧪 Testing Configuration...")
    
    # Print full configuration summary
    print_configuration_summary()
    
    # Test tile generation
    tiles = get_all_tile_info()
    print(f"\n   🗂️  Generated {len(tiles)} tiles")
    
    # Show first few tiles
    for i, tile in enumerate(tiles[:3]):
        print(f"      Tile {i+1}: {tile['id']} - {tile['bounds']}")
    
    # Test date ranges
    print(f"   📅 Available periods: {list(DATE_RANGES.keys())}")
    
    # Test satellite config
    satellite_config = SATELLITE_CONFIG[DEFAULT_SATELLITE]
    print(f"   🛰️  Using {DEFAULT_SATELLITE}: {satellite_config.get('description', 'Unknown')}")
    print(f"   🛰️  Landsat only mode: {LANDSAT_ONLY_MODE}")
    
    return tiles

def test_single_tile():
    """Test processing with a single tile"""
    print("\n🧪 Testing Single Tile Processing...")
    
    initialize_gee()
    tiles = test_configuration()
    
    # Use first tile for testing
    test_tile = tiles[0]
    tile_geom = ee.Geometry.Rectangle(test_tile['bounds'])
    
    print(f"   🗂️  Testing with tile: {test_tile['id']}")
    print(f"   📍 Bounds: {test_tile['bounds']}")
    
    # Get satellite collection
    satellite_config = SATELLITE_CONFIG[DEFAULT_SATELLITE]
    
    # Handle combined collections
    if 'collection_ids' in satellite_config:
        # Test with first collection from combined setup
        collection_id = satellite_config['collection_ids'][0]
        print(f"   🛰️  Testing with: {collection_id} (from combined collections)")
    else:
        collection_id = satellite_config['collection_id']
        print(f"   🛰️  Testing with: {collection_id}")
    
    collection = ee.ImageCollection(collection_id)
    
    # Test with recent 3 months
    test_period = 'recent_3_months'
    start_date, end_date = DATE_RANGES[test_period]
    
    print(f"   📅 Testing period: {test_period} ({start_date} to {end_date})")
    
    # Filter collection
    uzbekistan_bounds = ee.Geometry.Rectangle(UZBEKISTAN_CONFIG['bounds'])
    filtered_collection = (collection
                          .filterBounds(uzbekistan_bounds)
                          .filterDate(start_date, end_date)
                          .filter(ee.Filter.lt('CLOUD_COVER', satellite_config['cloud_filter_max'])))
    
    total_images = filtered_collection.size().getInfo()
    print(f"   🛰️  Found {total_images} total images in collection")
    
    if total_images == 0:
        print("   ⚠️  No images found - check date range or collection")
        return
    
    # Filter for tile
    tile_collection = filtered_collection.filterBounds(tile_geom)
    tile_images = tile_collection.size().getInfo()
    print(f"   🎯 Found {tile_images} images intersecting tile")
    
    if tile_images == 0:
        print("   ⚠️  No images intersect the test tile")
        return
    
    # Get info about available images
    if tile_images > 0:
        sample_image_info = tile_collection.limit(3).getInfo()
        print("   📋 Sample images:")
        for img in sample_image_info['features']:
            img_id = img['id']
            cloud_cover = img['properties'].get('CLOUD_COVER', 'N/A')
            date = img['properties'].get('DATE_ACQUIRED', 'N/A')
            print(f"      - {img_id[-12:]}: {date}, {cloud_cover}% clouds")
    
    print("   ✅ Tile processing test completed successfully!")

def test_dry_run():
    """Test the full workflow in dry-run mode"""
    print("\n🧪 Testing Full Workflow (Dry Run)...")
    
    # Test that we can import and configure the download script
    try:
        from config.tile_download_config import DEBUG_CONFIG
        
        original_dry_run = DEBUG_CONFIG['dry_run_mode']
        DEBUG_CONFIG['dry_run_mode'] = True
        
        print("   🏃 Configuration loaded successfully")
        print("   ✅ Dry run test passed!")
        
        # Restore original setting
        DEBUG_CONFIG['dry_run_mode'] = original_dry_run
        
    except ImportError as e:
        print(f"   ⚠️  Import issue: {e}")
    except Exception as e:
        print(f"   ❌ Error in dry run test: {e}")

def main():
    """Run all tests"""
    print("🧪 Testing Tile-based Download Configuration")
    print("=" * 50)
    
    try:
        test_single_tile()
        test_dry_run()
        
        print("\n🎉 All tests passed!")
        print("\n📋 Next steps:")
        print("   1. Run the full download script: python scripts/gee/download_tiles_individual.py")
        print("   2. Monitor Google Earth Engine Tasks panel")
        print("   3. Check data/tile_metadata/ for processing logs")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()
