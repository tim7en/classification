"""
Uzbekistan Land Cover Classification - Tile Management
This script provides a simplified interface for downloading and merging tiles.
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path
from datetime import datetime

def run_tile_download():
    """Run the tile download script."""
    print("🚀 Starting tile download from Google Earth Engine...")
    
    script_dir = Path(__file__).parent
    download_script = script_dir / "download_tiles.py"
    
    if not download_script.exists():
        print(f"❌ Download script not found: {download_script}")
        return False
    
    try:
        # Run the download script
        result = subprocess.run([sys.executable, str(download_script)], 
                              capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Tile download script completed successfully")
            print("📥 Check your Google Drive for exported tiles")
            return True
        else:
            print("❌ Tile download script failed:")
            print(result.stderr)
            return False
            
    except Exception as e:
        print(f"❌ Error running download script: {e}")
        return False

def check_downloaded_tiles(tiles_dir):
    """Check what tiles have been downloaded."""
    tiles_dir = Path(tiles_dir)
    
    if not tiles_dir.exists():
        print(f"📂 Tiles directory doesn't exist: {tiles_dir}")
        return {}
    
    print(f"🔍 Checking for downloaded tiles in: {tiles_dir}")
    
    # Find all .tif files
    tif_files = list(tiles_dir.glob("uzbekistan_*.tif"))
    
    if not tif_files:
        print("   ⚠️ No tile files found")
        return {}
    
    # Group by period
    periods = {}
    for tif_file in tif_files:
        # Extract period from filename
        # Expected format: uzbekistan_{period}_tile_{row}_{col}.tif
        name_parts = tif_file.stem.split('_')
        if len(name_parts) >= 3:
            period = name_parts[1]  # Should be the period
            if period not in periods:
                periods[period] = []
            periods[period].append(tif_file)
    
    print(f"   📊 Found tiles for {len(periods)} period(s):")
    for period, files in periods.items():
        print(f"      - {period}: {len(files)} tiles")
    
    return periods

def run_tile_merge(tiles_dir, metadata_dir, output_dir, period=None, vrt_only=False):
    """Run the tile merge script."""
    print("🔗 Starting tile merge process...")
    
    script_dir = Path(__file__).parent
    merge_script = script_dir / "merge_tiles.py"
    
    if not merge_script.exists():
        print(f"❌ Merge script not found: {merge_script}")
        return False
    
    # Build command arguments
    cmd = [sys.executable, str(merge_script)]
    cmd.extend(["--tiles-dir", str(tiles_dir)])
    cmd.extend(["--metadata-dir", str(metadata_dir)])
    cmd.extend(["--output-dir", str(output_dir)])
    
    if period:
        cmd.extend(["--period", period])
    
    if vrt_only:
        cmd.append("--vrt-only")
    
    try:
        # Run the merge script
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Tile merge completed successfully")
            print(result.stdout)
            return True
        else:
            print("❌ Tile merge failed:")
            print(result.stderr)
            return False
            
    except Exception as e:
        print(f"❌ Error running merge script: {e}")
        return False

def setup_directories():
    """Set up the required directory structure."""
    script_dir = Path(__file__).parent
    project_root = script_dir.parent.parent
    
    directories = {
        'downloaded_tiles': project_root / "data" / "downloaded_tiles",
        'tile_metadata': project_root / "data" / "tile_metadata", 
        'merged_tiles': project_root / "data" / "merged_tiles"
    }
    
    print("📁 Setting up directories...")
    for name, path in directories.items():
        path.mkdir(parents=True, exist_ok=True)
        print(f"   ✅ {name}: {path}")
    
    return directories

def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description="Uzbekistan Tile Management")
    parser.add_argument("--download", action="store_true", help="Download tiles from GEE")
    parser.add_argument("--merge", action="store_true", help="Merge downloaded tiles")
    parser.add_argument("--check", action="store_true", help="Check downloaded tiles")
    parser.add_argument("--full", action="store_true", help="Download and merge (full workflow)")
    parser.add_argument("--period", type=str, help="Specific period to process")
    parser.add_argument("--vrt-only", action="store_true", help="Create VRT only (no GeoTIFF)")
    
    args = parser.parse_args()
    
    print("🌍 Uzbekistan Land Cover Classification - Tile Management")
    print("=" * 60)
    
    # Set up directories
    dirs = setup_directories()
    
    # Default action if no specific action is specified
    if not any([args.download, args.merge, args.check, args.full]):
        args.check = True
    
    success = True
    
    # Download tiles
    if args.download or args.full:
        success = run_tile_download()
        if not success and args.full:
            print("⚠️ Download failed, skipping merge")
            return
    
    # Check downloaded tiles
    if args.check or args.full:
        periods = check_downloaded_tiles(dirs['downloaded_tiles'])
        
        if not periods:
            print("📥 No tiles found. You may need to:")
            print("   1. Run --download to start the GEE export process")
            print("   2. Download tiles manually from Google Drive")
            print("   3. Place downloaded tiles in:", dirs['downloaded_tiles'])
            if not args.full:
                return
    
    # Merge tiles
    if args.merge or args.full:
        success = run_tile_merge(
            dirs['downloaded_tiles'],
            dirs['tile_metadata'],
            dirs['merged_tiles'],
            period=args.period,
            vrt_only=args.vrt_only
        )
    
    if success:
        print("\n🎉 Tile management workflow completed!")
        print("\n💡 Next steps:")
        print("   1. Review merged mosaics in QGIS")
        print("   2. Run classification on merged tiles")
        print("   3. Analyze results")
    else:
        print("\n❌ Workflow completed with errors")

if __name__ == "__main__":
    main()
