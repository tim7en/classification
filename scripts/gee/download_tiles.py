import ee
import geemap
import os
import sys
from datetime import datetime, timedelta

def initialize_gee():
    """Authenticates and initializes the Google Earth Engine library."""
    try:
        # Check if GEE is already initialized
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

def get_least_cloudy_image(collection, region):
    """
    Filters an ImageCollection to get the single least cloudy image.
    
    Args:
        collection (ee.ImageCollection): The collection to filter.
        region (ee.Geometry): The region of interest to clip the image.
        
    Returns:
        ee.Image: The least cloudy image, clipped to the region.
    """
    # Sort by cloud cover and get the least cloudy image
    return collection.sort('CLOUD_COVER').first().clip(region)

def add_contextual_bands(image, region):
    """
    Adds comprehensive contextual bands to a Landsat image for classification.
    Includes spectral indices, terrain features, and environmental context layers.

    Args:
        image (ee.Image): The input Landsat image.
        region (ee.Geometry): The region of interest for clipping contextual layers.

    Returns:
        ee.Image: The image with added contextual bands.
    """
    # Scale optical bands (B1-B7)
    optical_bands = image.select(['SR_B1', 'SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B6', 'SR_B7']) \
                         .multiply(0.0000275).add(-0.2)

    # === SPECTRAL INDICES ===
    # 1. Normalized Difference Vegetation Index (NDVI) - Vegetation health
    ndvi = optical_bands.normalizedDifference(['SR_B5', 'SR_B4']).rename('NDVI')
    
    # 2. Normalized Difference Water Index (NDWI) - Water bodies detection
    ndwi = optical_bands.normalizedDifference(['SR_B3', 'SR_B5']).rename('NDWI')
    
    # 3. Modified Normalized Difference Water Index (MNDWI) - Better water detection
    mndwi = optical_bands.normalizedDifference(['SR_B3', 'SR_B6']).rename('MNDWI')
    
    # 4. Normalized Difference Built-up Index (NDBI) - Urban/built areas
    ndbi = optical_bands.normalizedDifference(['SR_B6', 'SR_B5']).rename('NDBI')
    
    # 5. Enhanced Vegetation Index (EVI) - Better for dense vegetation
    evi = optical_bands.expression(
        '2.5 * ((NIR - RED) / (NIR + 6 * RED - 7.5 * BLUE + 1))',
        {
            'NIR': optical_bands.select('SR_B5'),
            'RED': optical_bands.select('SR_B4'), 
            'BLUE': optical_bands.select('SR_B2')
        }
    ).rename('EVI')
    
    # 6. Soil Adjusted Vegetation Index (SAVI) - Reduces soil background effects
    savi = optical_bands.expression(
        '((NIR - RED) / (NIR + RED + 0.5)) * (1.5)',
        {
            'NIR': optical_bands.select('SR_B5'),
            'RED': optical_bands.select('SR_B4')
        }
    ).rename('SAVI')

    # === TERRAIN FEATURES ===
    # 7. Elevation data from SRTM
    elevation = ee.Image('USGS/SRTMGL1_003').select('elevation').clip(region)
    
    # 8. Slope calculated from elevation
    slope = ee.Terrain.slope(elevation).rename('SLOPE')
    
    # 9. Aspect (direction of slope)
    aspect = ee.Terrain.aspect(elevation).rename('ASPECT')
    
    # 10. Hillshade (terrain illumination)
    hillshade = ee.Terrain.hillshade(elevation).rename('HILLSHADE')

    # === CONTEXTUAL LAYERS ===
    # 11. Terrain Context: Mountainous areas (elevation > 1500m)
    terrain_mountain = elevation.gte(1500).rename('TERRAIN_MOUNTAIN')
    
    # 12. Water mask (using MNDWI threshold)
    water_mask = mndwi.gt(0.3).rename('WATER_MASK')
    
    # 13. Vegetation density categories
    veg_sparse = ndvi.gt(0.1).And(ndvi.lte(0.3)).rename('VEG_SPARSE')
    veg_moderate = ndvi.gt(0.3).And(ndvi.lte(0.6)).rename('VEG_MODERATE') 
    veg_dense = ndvi.gt(0.6).rename('VEG_DENSE')
    
    # 14. Brightness Temperature (thermal band for water/moisture detection)
    thermal = image.select('ST_B10').multiply(0.00341802).add(149.0).subtract(273.15).rename('TEMP_C')

    # === TEXTURE FEATURES ===
    # 15. Standard deviation of NIR (texture indicator)
    nir_texture = optical_bands.select('SR_B5').reduceNeighborhood(
        reducer=ee.Reducer.stdDev(),
        kernel=ee.Kernel.square(3)
    ).rename('NIR_TEXTURE')

    # Combine all bands into the comprehensive classification image
    # Cast all bands to Float32 to ensure consistent data types
    final_image = optical_bands.toFloat() \
        .addBands(ndvi.toFloat()) \
        .addBands(ndwi.toFloat()) \
        .addBands(mndwi.toFloat()) \
        .addBands(ndbi.toFloat()) \
        .addBands(evi.toFloat()) \
        .addBands(savi.toFloat()) \
        .addBands(elevation.toFloat()) \
        .addBands(slope.toFloat()) \
        .addBands(aspect.toFloat()) \
        .addBands(hillshade.toFloat()) \
        .addBands(terrain_mountain.toFloat()) \
        .addBands(water_mask.toFloat()) \
        .addBands(veg_sparse.toFloat()) \
        .addBands(veg_moderate.toFloat()) \
        .addBands(veg_dense.toFloat()) \
        .addBands(thermal.toFloat()) \
        .addBands(nir_texture.toFloat())
    
    return final_image

def download_gee_tile(image, region, description, folder='earthengine_exports', scale=30):
    """
    Exports a GEE Image to Google Drive.
    
    Args:
        image (ee.Image): The image to export.
        region (ee.Geometry): The bounding box for the export.
        description (str): The name for the export task and output file.
        folder (str): The Google Drive folder to save the export to.
        scale (int): The pixel scale in meters.
    """
    print(f"🛰️  Starting export task for: {description}")
    try:
        # Create an export task to save the image to Google Drive
        task = ee.batch.Export.image.toDrive(
            image=image,
            description=description,
            folder=folder,
            fileNamePrefix=description,
            region=region,
            scale=scale,
            fileFormat='GeoTIFF',
            maxPixels=1e10  # Increase pixel limit for large exports
        )
        task.start()
        print(f"✅ Export task '{description}' started. Check the 'Tasks' tab in the GEE Asset Manager.")
        # Add a new line for better log readability
        print("-" * 50)
    except Exception as e:
        print(f"❌ Error starting export task for {description}: {e}")
        print("-" * 50)

def main():
    """
    Main function to download seasonal and recent satellite tiles for Uzbekistan.
    """
    initialize_gee()

    # --- Configuration ---
    # Define project paths relative to this script's location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, '..', '..'))
    download_dir = os.path.join(project_root, 'data', 'downloaded_tiles')
    os.makedirs(download_dir, exist_ok=True)
    
    print(f"📂 Files will be saved to: {download_dir}")

    # Define the Area of Interest (AOI) for Uzbekistan
    uzbekistan_bounds = ee.Geometry.Rectangle([55.9, 37.2, 73.2, 45.6])

    # Define date ranges for the tiles you want to download
    today = datetime.now()
    date_ranges = {
        'recent_3_months': (
            (today - timedelta(days=90)).strftime('%Y-%m-%d'),
            today.strftime('%Y-%m-%d')
        ),
        'summer_2023': ('2023-06-01', '2023-08-31'),
        'winter_2023_2024': ('2023-12-01', '2024-02-29'),
    }

    # --- Processing ---
    # Base Landsat 8 Collection 2, Surface Reflectance
    landsat_collection = ee.ImageCollection('LANDSAT/LC08/C02/T1_L2') \
        .filterBounds(uzbekistan_bounds)

    for period, (start_date, end_date) in date_ranges.items():
        print(f"\n🔎 Processing period: {period} ({start_date} to {end_date})")
        
        # Filter the collection for the specific date range and low cloud cover
        period_collection = landsat_collection \
            .filterDate(start_date, end_date) \
            .filter(ee.Filter.lt('CLOUD_COVER', 20))

        image_count = period_collection.size().getInfo()
        if image_count == 0:
            print(f"   - ⚠️ No images found for this period. Skipping.")
            continue
        
        print(f"   - ✅ Found {image_count} images for this period.")

        # Get the best (least cloudy) image from the filtered collection
        best_image = get_least_cloudy_image(period_collection, uzbekistan_bounds)
        
        # Robustness check: ensure a valid image was found
        if best_image.bandNames().size().getInfo() == 0:
            print(f"   - ⚠️ Could not retrieve a valid image for this period. Skipping.")
            continue

        # Add all necessary contextual bands for classification
        print("   - ➕ Adding comprehensive features:")
        print("        • Spectral indices: NDVI, NDWI, MNDWI, NDBI, EVI, SAVI")
        print("        • Terrain features: Elevation, Slope, Aspect, Hillshade")
        print("        • Context layers: Water mask, Vegetation density, Temperature")
        print("        • Texture features: NIR texture")
        output_image = add_contextual_bands(best_image, uzbekistan_bounds)

        # Define the output filename and start the export task
        output_description = f'uzbekistan_tile_{period}_comprehensive'
        download_gee_tile(output_image, uzbekistan_bounds, output_description)

    print("\n🎉 All export tasks have been started.")
    print("📥 Please monitor the 'Tasks' tab in the Google Earth Engine console.")
    print("📂 Once complete, find your files in the 'earthengine_exports' folder in your Google Drive.")

if __name__ == '__main__':
    main()
