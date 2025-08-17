# landsat9_uzbekistan_tiles.py
import ee
import os
import sys
from datetime import datetime, timedelta

# landsat9_uzbekistan_tiles.py
import ee
import os
import sys
from datetime import datetime, timedelta

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



# ============ CLOUD MASK (L2 QA_PIXEL) ============
def mask_l2_clouds(img: ee.Image) -> ee.Image:
    """
    Basic cloud/snow masking for Landsat Collection 2 L2 using QA_PIXEL.
    Bits used (QA_PIXEL):
      1: Dilated Cloud, 2: Cirrus, 3: Cloud, 4: Cloud Shadow, 5: Snow
    """
    qa = img.select('QA_PIXEL')
    mask = (qa.bitwiseAnd(1 << 1).eq(0)  # not dilated cloud
            .And(qa.bitwiseAnd(1 << 2).eq(0))  # not cirrus
            .And(qa.bitwiseAnd(1 << 3).eq(0))  # not cloud
            .And(qa.bitwiseAnd(1 << 4).eq(0))  # not cloud shadow
            .And(qa.bitwiseAnd(1 << 5).eq(0)))  # not snow
    return img.updateMask(mask)

# ============ IMAGE PICKER & MOSAICKING ============
def create_country_mosaic(collection: ee.ImageCollection, region: ee.Geometry) -> ee.Image:
    """
    Creates a seamless mosaic covering the entire country by:
    1. Creating a quality mosaic using cloud-free pixels
    2. Using median composite as fallback for any remaining gaps
    3. Ensuring complete coverage of the region
    """
    # Method 1: Quality mosaic - prioritizes cloud-free pixels
    quality_mosaic = collection.qualityMosaic('SR_B4')  # Uses red band for quality assessment
    
    # Method 2: Median composite as fallback for any gaps
    median_composite = collection.median()
    
    # Combine: use quality mosaic where available, fill gaps with median
    # First, create a mask of valid pixels in quality mosaic
    quality_mask = quality_mosaic.select('SR_B4').mask()
    
    # Fill any remaining gaps with median composite
    final_mosaic = quality_mosaic.where(quality_mask.Not(), median_composite)
    
    return final_mosaic.clip(region)

def get_least_cloudy_image(collection: ee.ImageCollection, region: ee.Geometry) -> ee.Image:
    """
    Legacy function - kept for compatibility but now calls create_country_mosaic
    """
    return create_country_mosaic(collection, region)

# ============ CONTEXTUAL BANDS ============
def add_contextual_bands(image: ee.Image, region: ee.Geometry) -> ee.Image:
    """
    Creates a clean, optimized dataset with only essential bands for classification.
    Removes redundant bands and creates a streamlined feature set.
    """
    # Scale optical SR bands and rename for clarity
    optical = image.select(['SR_B1','SR_B2','SR_B3','SR_B4','SR_B5','SR_B6','SR_B7']) \
                   .multiply(0.0000275).add(-0.2) \
                   .rename(['BLUE', 'GREEN', 'RED', 'NIR', 'SWIR1', 'SWIR2', 'COASTAL'])

    # Spectral indices (key for land cover classification)
    ndvi = optical.normalizedDifference(['NIR','RED']).rename('NDVI')
    ndwi = optical.normalizedDifference(['GREEN','NIR']).rename('NDWI')
    mndwi = optical.normalizedDifference(['GREEN','SWIR1']).rename('MNDWI')
    ndbi = optical.normalizedDifference(['SWIR1','NIR']).rename('NDBI')

    evi = optical.expression(
        '2.5 * ((NIR - RED) / (NIR + 6 * RED - 7.5 * BLUE + 1))',
        {'NIR': optical.select('NIR'),
         'RED': optical.select('RED'),
         'BLUE': optical.select('BLUE')}
    ).rename('EVI')

    savi = optical.expression(
        '((NIR - RED) / (NIR + RED + 0.5)) * 1.5',
        {'NIR': optical.select('NIR'),
         'RED': optical.select('RED')}
    ).rename('SAVI')

    # Terrain features
    elevation = ee.Image('USGS/SRTMGL1_003').select('elevation').clip(region)
    slope = ee.Terrain.slope(elevation).rename('SLOPE')

    # Thermal band (if available)
    band_names = image.bandNames()
    has_thermal = band_names.contains('ST_B10')
    thermal = ee.Image(ee.Algorithms.If(
        has_thermal,
        image.select(['ST_B10']).multiply(0.00341802).add(149.0).subtract(273.15).rename('TEMP_C'),
        ee.Image(0).rename('TEMP_C')
    ))

    # Texture feature (NIR texture for structure detection)
    nir_texture = optical.select('NIR').reduceNeighborhood(
        reducer=ee.Reducer.stdDev(), kernel=ee.Kernel.square(3)
    ).rename('NIR_TEXTURE')

    # Create final optimized image with clear band organization
    final_image = ee.Image.cat([
        # Core spectral bands (7 bands)
        optical.select(['BLUE', 'GREEN', 'RED', 'NIR', 'SWIR1', 'SWIR2']),  # Skip coastal for most use cases
        
        # Vegetation indices (3 bands) - most important for land cover
        ndvi, evi, savi,
        
        # Water/built-up indices (3 bands)
        ndwi, mndwi, ndbi,
        
        # Terrain (2 bands)
        elevation.rename('ELEVATION'), slope,
        
        # Environmental (2 bands)
        thermal, nir_texture
    ]).toFloat()
    
    return final_image

# ============ EXPORT HELPERS ============
def export_to_drive(image: ee.Image, region: ee.Geometry, description: str, scale: int = 30, folder: str = 'earthengine_exports'):
    print(f"🛰️  Starting Drive export: {description}")
    task = ee.batch.Export.image.toDrive(
        image=image,
        description=description,
        folder=folder,
        fileNamePrefix=description,
        region=region,
        scale=scale,
        fileFormat='GeoTIFF',
        maxPixels=1e10
    )
    task.start()
    print(f"✅ Drive task '{description}' started. Monitor GEE Tasks.")

def export_local(image: ee.Image, region: ee.Geometry, out_path: str, scale: int = 30):
    """
    Optional: direct local export using geemap (smaller areas recommended).
    pip install geemap
    """
    try:
        import geemap
    except ImportError:
        raise RuntimeError("geemap is not installed. `pip install geemap` to enable local export.")
    print(f"💾 Exporting locally: {out_path}")
    geemap.ee_export_image(
        image=image,
        filename=out_path,
        scale=scale,
        region=region,
        file_per_band=False
    )
    print("✅ Local export complete.")

# ============ MAIN ============
def main():
    initialize_gee()

    # --- Config ---
    USE_LOCAL_EXPORT = False   # True => save to disk via geemap; False => export to Google Drive
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, '..', '..'))
    download_dir = os.path.join(project_root, 'data', 'downloaded_tiles')
    os.makedirs(download_dir, exist_ok=True)
    print(f"📂 Local path (if local export is enabled): {download_dir}")

    # Uzbekistan AOI (rough bbox)
    uzbekistan_bounds = ee.Geometry.Rectangle([55.9, 37.2, 73.2, 45.6])

    # Periods
    today = datetime.now()
    date_ranges = {
        'recent_3_months': (
            (today - timedelta(days=90)).strftime('%Y-%m-%d'),
            today.strftime('%Y-%m-%d')
        ),
        'summer_2023': ('2023-06-01', '2023-08-31'),
        'winter_2023_2024': ('2023-12-01', '2024-02-29'),
    }

    # Landsat 9 Collection 2, Tier 1, L2 SR + ST
    landsat9 = (ee.ImageCollection('LANDSAT/LC09/C02/T1_L2')
                .filterBounds(uzbekistan_bounds))

    for period, (start_date, end_date) in date_ranges.items():
        print(f"\n🔎 Period: {period} ({start_date} → {end_date})")

        period_coll = (landsat9
                       .filterDate(start_date, end_date)
                       .filterBounds(uzbekistan_bounds)
                       .map(mask_l2_clouds)
                       .filter(ee.Filter.lt('CLOUD_COVER', 60)))  # More permissive for mosaicking

        count = period_coll.size().getInfo()
        if count == 0:
            print("   - ⚠️ No Landsat 9 images found after filtering. Skipping.")
            continue
        print(f"   - ✅ {count} scenes found for mosaicking.")
        
        # Get unique path/row combinations to understand coverage
        paths_rows = period_coll.aggregate_array('WRS_PATH').cat(
            period_coll.aggregate_array('WRS_ROW')
        ).distinct().size().getInfo()
        print(f"   - 📍 Coverage: ~{paths_rows//2} unique Landsat path/row combinations")

        print("   - 🧩 Creating country-wide mosaic...")
        mosaic = create_country_mosaic(period_coll, uzbekistan_bounds)

        # Verify we have valid data
        band_count = mosaic.bandNames().size().getInfo()
        if band_count == 0:
            print("   - ⚠️ Could not create valid mosaic. Skipping.")
            continue
        print(f"   - ✅ Mosaic created with {band_count} bands")

        print("   - ➕ Adding contextual bands…")
        out_img = add_contextual_bands(mosaic, uzbekistan_bounds)
        
        # Display band information for transparency
        final_band_names = out_img.bandNames().getInfo()
        final_band_count = len(final_band_names)
        print(f"   - ✅ Final optimized image: {final_band_count} bands")
        print(f"   - 📊 Bands: {', '.join(final_band_names)}")
        
        desc = f'uzbekistan_mosaic_{period}_L9_comprehensive'

        if USE_LOCAL_EXPORT:
            out_tif = os.path.join(download_dir, f'{desc}.tif')
            export_local(out_img, uzbekistan_bounds, out_tif, scale=30)
        else:
            export_to_drive(out_img, uzbekistan_bounds, desc, scale=30, folder='earthengine_exports')

    print("\n🎉 All exports triggered.")
    if not USE_LOCAL_EXPORT:
        print("📥 Check the GEE Tasks panel; files will land in Google Drive → /earthengine_exports.")
    else:
        print(f"📂 Files saved under: {download_dir}")

if __name__ == '__main__':
    main()
