# Tile-Based Download and Processing

This directory contains scripts for downloading satellite imagery as individual tiles instead of attempting to download entire country mosaics. This approach is more reliable for large areas like Uzbekistan.

## Overview

The tile-based approach breaks down the large Uzbekistan territory into smaller, manageable tiles (~55km × 55km each) that can be downloaded individually from Google Earth Engine and then merged locally.

## Scripts

### 1. `download_tiles.py`
**Purpose**: Downloads individual tiles from Google Earth Engine
- Creates a grid of tiles covering Uzbekistan
- Filters tiles with sufficient data coverage
- Exports each tile individually to Google Drive
- Generates metadata for tile reconstruction

**Key Features**:
- Configurable tile size (default: 0.5° ≈ 55km)
- Quality assessment and cloud masking
- Spectral indices and contextual bands
- Export limit protection (max 50 tiles per period)

### 2. `merge_tiles.py`
**Purpose**: Merges downloaded tiles into seamless mosaics
- Reads tile metadata from JSON files
- Creates Virtual Raster Datasets (VRT) for fast access
- Optionally converts to GeoTIFF for permanent storage
- Handles missing tiles gracefully

**Key Features**:
- GDAL-based efficient merging
- Compressed GeoTIFF output
- VRT-only option for large datasets
- Automatic extent calculation

### 3. `tile_manager.py`
**Purpose**: Simplified interface for the entire tile workflow
- One-stop script for download and merge operations
- Status checking and validation
- Directory management
- Error handling and user guidance

## Workflow

### Step 1: Download Tiles from Google Earth Engine

```bash
# Option A: Use the tile manager (recommended)
python scripts/gee/tile_manager.py --download

# Option B: Run download script directly
python scripts/gee/download_tiles.py
```

This will:
- Create a grid of ~50-100 tiles covering Uzbekistan
- Export tiles to Google Drive folder: `uzbekistan_tiles`
- Generate metadata files in `data/tile_metadata/`

### Step 2: Download Tiles from Google Drive

1. Go to your Google Drive
2. Navigate to the `uzbekistan_tiles` folder
3. Download all `.tif` files to your local `data/downloaded_tiles/` directory

### Step 3: Merge Tiles into Mosaics

```bash
# Option A: Use the tile manager (recommended)
python scripts/gee/tile_manager.py --merge

# Option B: Run merge script directly
python scripts/gee/merge_tiles.py
```

This will:
- Read tile metadata
- Find corresponding downloaded files
- Create merged mosaics in `data/merged_tiles/`

### Step 4: Check Status

```bash
python scripts/gee/tile_manager.py --check
```

## Directory Structure

```
data/
├── downloaded_tiles/          # Downloaded .tif files from Google Drive
├── tile_metadata/            # JSON files with tile grid information
└── merged_tiles/             # Final merged mosaics
    ├── *.vrt                # Virtual datasets (fast access)
    └── *.tif                # GeoTIFF mosaics (permanent storage)
```

## Configuration

### Tile Size
Default: 0.5° (approximately 55km × 55km)

You can modify the `TILE_SIZE_DEGREES` parameter in `download_tiles.py`:
- Smaller tiles = more reliable downloads, more files to manage
- Larger tiles = fewer files, higher chance of export failure

### Export Limits
Default: Maximum 50 tiles per time period

Modify `MAX_TILES_PER_PERIOD` in `download_tiles.py` to change this limit.

### Time Periods
Currently configured for:
- `recent_3_months`: Last 3 months of available data
- `summer_2023`: June-August 2023
- `winter_2023_2024`: December 2023 - February 2024

## Troubleshooting

### No tiles found after download
1. Check Google Earth Engine Tasks panel for export status
2. Verify authentication: `earthengine authenticate`
3. Check Google Drive for exported files

### Missing tiles during merge
- This is normal - not all tiles may have sufficient data
- The merge process handles missing tiles gracefully
- Resulting mosaics may have gaps in areas with no data

### Large file sizes
- Use VRT files for analysis when possible (`--vrt-only` option)
- VRT files reference original tiles without copying data
- Convert to GeoTIFF only when needed for specific applications

### Memory issues during merge
- Reduce the number of tiles being processed simultaneously
- Use VRT-only mode for large datasets
- Increase system virtual memory if needed

## Advanced Usage

### Process specific time period only
```bash
python scripts/gee/tile_manager.py --merge --period recent_3_months
```

### Create VRT only (no GeoTIFF conversion)
```bash
python scripts/gee/tile_manager.py --merge --vrt-only
```

### Full workflow (download + merge)
```bash
python scripts/gee/tile_manager.py --full
```

## Integration with Classification Workflow

The merged mosaics can be used directly with the existing classification scripts:

1. Place merged `.tif` files in the expected data directory
2. Run the main classification workflow
3. The classification scripts will automatically detect and process the mosaics

## Performance Notes

- **Download**: Individual tiles download more reliably than large mosaics
- **Storage**: VRT files provide fast access without duplicating data
- **Processing**: Local merging is faster than GEE-based mosaicking for large areas
- **Memory**: GDAL handles large datasets efficiently through tiling and caching
