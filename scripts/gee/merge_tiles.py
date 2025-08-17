"""
Merge downloaded tiles into a single mosaic for classification.
This script combines individual tiles downloaded from Google Earth Engine
into seamless mosaics covering the entire area of interest.
"""

import os
import json
import glob
import sys
from pathlib import Path
import numpy as np
from osgeo import gdal, osr
import argparse
from datetime import datetime

def setup_gdal():
    """Configure GDAL settings for optimal performance."""
    gdal.SetConfigOption('GDAL_CACHEMAX', '2048')  # 2GB cache
    gdal.SetConfigOption('GDAL_NUM_THREADS', 'ALL_CPUS')
    gdal.UseExceptions()

def load_tile_metadata(metadata_file):
    """Load tile metadata from JSON file."""
    print(f"📋 Loading tile metadata: {metadata_file}")
    
    with open(metadata_file, 'r') as f:
        metadata = json.load(f)
    
    print(f"   📊 Period: {metadata['period']}")
    print(f"   🧩 Total tiles: {metadata['total_tiles']}")
    print(f"   📅 Export date: {metadata['export_timestamp']}")
    
    return metadata

def find_tile_files(tiles_dir, period, tile_descriptions):
    """Find downloaded tile files matching the metadata."""
    print(f"🔍 Searching for tile files in: {tiles_dir}")
    
    tile_files = {}
    missing_tiles = []
    
    for tile_meta in tile_descriptions:
        tile_desc = tile_meta['description']
        # Look for files matching the pattern
        pattern = f"uzbekistan_{period}_{tile_desc}*.tif"
        matches = glob.glob(os.path.join(tiles_dir, pattern))
        
        if matches:
            # Take the first match (should be only one)
            tile_files[tile_desc] = matches[0]
        else:
            missing_tiles.append(tile_desc)
    
    print(f"   ✅ Found {len(tile_files)} tile files")
    if missing_tiles:
        print(f"   ⚠️ Missing {len(missing_tiles)} tiles:")
        for missing in missing_tiles[:5]:  # Show first 5
            print(f"      - {missing}")
        if len(missing_tiles) > 5:
            print(f"      ... and {len(missing_tiles) - 5} more")
    
    return tile_files, missing_tiles

def get_tile_extent(tile_file):
    """Get the geographic extent of a tile."""
    ds = gdal.Open(tile_file)
    if ds is None:
        raise RuntimeError(f"Could not open tile: {tile_file}")
    
    gt = ds.GetGeoTransform()
    cols = ds.RasterXSize
    rows = ds.RasterYSize
    
    # Calculate extent
    min_x = gt[0]
    max_y = gt[3]
    max_x = min_x + (cols * gt[1])
    min_y = max_y + (rows * gt[5])
    
    extent = [min_x, min_y, max_x, max_y]  # [west, south, east, north]
    
    projection = ds.GetProjection()
    bands = ds.RasterCount
    
    ds = None
    
    return extent, projection, bands

def calculate_mosaic_extent(tile_files):
    """Calculate the overall extent that will contain all tiles."""
    print("📐 Calculating mosaic extent...")
    
    all_extents = []
    projection = None
    band_count = None
    
    for tile_desc, tile_file in tile_files.items():
        try:
            extent, proj, bands = get_tile_extent(tile_file)
            all_extents.append(extent)
            
            if projection is None:
                projection = proj
                band_count = bands
            elif projection != proj:
                print(f"⚠️ Projection mismatch in tile {tile_desc}")
                
        except Exception as e:
            print(f"❌ Error reading tile {tile_desc}: {e}")
            continue
    
    if not all_extents:
        raise RuntimeError("No valid tiles found")
    
    # Calculate overall bounds
    all_extents = np.array(all_extents)
    mosaic_extent = [
        np.min(all_extents[:, 0]),  # min west
        np.min(all_extents[:, 1]),  # min south
        np.max(all_extents[:, 2]),  # max east
        np.max(all_extents[:, 3])   # max north
    ]
    
    print(f"   📏 Mosaic extent: {mosaic_extent}")
    print(f"   🎯 Projection: {projection[:50]}...")
    print(f"   📊 Bands: {band_count}")
    
    return mosaic_extent, projection, band_count

def create_mosaic_vrt(tile_files, output_vrt, mosaic_extent=None):
    """Create a VRT (Virtual Dataset) that references all tiles."""
    print(f"🔗 Creating VRT mosaic: {output_vrt}")
    
    # Get list of tile file paths
    tile_paths = list(tile_files.values())
    
    if not tile_paths:
        raise RuntimeError("No tile files to mosaic")
    
    # Create VRT options
    vrt_options = gdal.BuildVRTOptions(
        resolution='highest',  # Use highest resolution
        resampleAlg='bilinear',  # Resampling algorithm
        addAlpha=False,  # Don't add alpha channel
        separate=False   # Don't separate bands
    )
    
    # Add extent if provided
    if mosaic_extent:
        vrt_options = gdal.BuildVRTOptions(
            resolution='highest',
            resampleAlg='bilinear',
            addAlpha=False,
            separate=False,
            outputBounds=mosaic_extent  # [west, south, east, north]
        )
    
    # Build VRT
    vrt_ds = gdal.BuildVRT(output_vrt, tile_paths, options=vrt_options)
    
    if vrt_ds is None:
        raise RuntimeError("Failed to create VRT")
    
    print(f"   ✅ VRT created: {vrt_ds.RasterXSize} x {vrt_ds.RasterYSize} pixels")
    print(f"   📊 Bands: {vrt_ds.RasterCount}")
    
    vrt_ds = None
    return output_vrt

def translate_vrt_to_geotiff(vrt_file, output_tiff, compress='LZW'):
    """Convert VRT to a single GeoTIFF file."""
    print(f"💾 Converting VRT to GeoTIFF: {output_tiff}")
    
    # Translation options for optimized GeoTIFF
    translate_options = gdal.TranslateOptions(
        format='GTiff',
        creationOptions=[
            f'COMPRESS={compress}',
            'PREDICTOR=2',  # Horizontal differencing for better compression
            'TILED=YES',    # Create tiled TIFF
            'BLOCKXSIZE=512',
            'BLOCKYSIZE=512',
            'BIGTIFF=IF_SAFER'  # Use BigTIFF if needed
        ]
    )
    
    # Perform translation
    ds = gdal.Translate(output_tiff, vrt_file, options=translate_options)
    
    if ds is None:
        raise RuntimeError("Failed to create GeoTIFF")
    
    print(f"   ✅ GeoTIFF created: {ds.RasterXSize} x {ds.RasterYSize} pixels")
    print(f"   📦 File size: {os.path.getsize(output_tiff) / (1024**3):.2f} GB")
    
    ds = None
    return output_tiff

def merge_tiles_for_period(tiles_dir, metadata_file, output_dir, create_vrt_only=False):
    """Merge all tiles for a specific period."""
    print(f"\n{'='*60}")
    print(f"🔄 Merging tiles for period")
    print(f"{'='*60}")
    
    # Load metadata
    metadata = load_tile_metadata(metadata_file)
    period = metadata['period']
    
    # Find tile files
    tile_files, missing_tiles = find_tile_files(tiles_dir, period, metadata['tiles'])
    
    if not tile_files:
        print("❌ No tile files found. Cannot create mosaic.")
        return None
    
    if len(missing_tiles) > len(tile_files):
        print("⚠️ More tiles missing than available. Mosaic may have gaps.")
    
    # Calculate mosaic extent
    mosaic_extent, projection, band_count = calculate_mosaic_extent(tile_files)
    
    # Create output files
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    vrt_file = os.path.join(output_dir, f'uzbekistan_{period}_mosaic_{timestamp}.vrt')
    tiff_file = os.path.join(output_dir, f'uzbekistan_{period}_mosaic_{timestamp}.tif')
    
    # Create VRT
    create_mosaic_vrt(tile_files, vrt_file, mosaic_extent)
    
    if create_vrt_only:
        print(f"✅ VRT mosaic created: {vrt_file}")
        return vrt_file
    
    # Convert to GeoTIFF
    translate_vrt_to_geotiff(vrt_file, tiff_file)
    
    print(f"✅ Mosaic complete: {tiff_file}")
    return tiff_file

def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description="Merge downloaded tiles into mosaics")
    parser.add_argument("--tiles-dir", type=str, help="Directory containing downloaded tiles")
    parser.add_argument("--metadata-dir", type=str, help="Directory containing tile metadata")
    parser.add_argument("--output-dir", type=str, help="Directory for output mosaics")
    parser.add_argument("--period", type=str, help="Specific period to process (optional)")
    parser.add_argument("--vrt-only", action="store_true", help="Create VRT only (no GeoTIFF conversion)")
    
    args = parser.parse_args()
    
    # Set up paths
    script_dir = Path(__file__).parent
    project_root = script_dir.parent.parent
    
    tiles_dir = args.tiles_dir or project_root / "data" / "downloaded_tiles"
    metadata_dir = args.metadata_dir or project_root / "data" / "tile_metadata"
    output_dir = args.output_dir or project_root / "data" / "merged_tiles"
    
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("🗂️ Uzbekistan Tile Merger")
    print("=" * 50)
    print(f"📂 Tiles directory: {tiles_dir}")
    print(f"📋 Metadata directory: {metadata_dir}")
    print(f"💾 Output directory: {output_dir}")
    
    # Set up GDAL
    setup_gdal()
    
    # Find metadata files
    if args.period:
        metadata_files = [metadata_dir / f"tile_metadata_{args.period}.json"]
    else:
        metadata_files = list(Path(metadata_dir).glob("tile_metadata_*.json"))
    
    if not metadata_files:
        print("❌ No tile metadata files found.")
        print(f"   Expected location: {metadata_dir}")
        print(f"   Expected pattern: tile_metadata_*.json")
        return
    
    print(f"\n🔍 Found {len(metadata_files)} period(s) to process:")
    for mf in metadata_files:
        print(f"   - {mf.name}")
    
    # Process each period
    created_mosaics = []
    
    for metadata_file in metadata_files:
        try:
            output_file = merge_tiles_for_period(
                tiles_dir, 
                metadata_file, 
                output_dir, 
                create_vrt_only=args.vrt_only
            )
            if output_file:
                created_mosaics.append(output_file)
        except Exception as e:
            print(f"❌ Error processing {metadata_file}: {e}")
            continue
    
    print(f"\n🎉 Tile merging complete!")
    print(f"📁 Created {len(created_mosaics)} mosaic(s):")
    for mosaic in created_mosaics:
        print(f"   - {mosaic}")
    
    if created_mosaics:
        print(f"\n💡 Next steps:")
        print(f"   1. Review mosaics in QGIS or other GIS software")
        print(f"   2. Run classification workflow on merged mosaics")
        print(f"   3. Use VRT files for faster preview/analysis")

if __name__ == "__main__":
    main()
