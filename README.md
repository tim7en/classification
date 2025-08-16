# Uzbekistan Land Cover Classification

This project provides a comprehensive workflow for land cover classification of Uzbekistan using Google Earth Engine (GEE) for satellite data processing and local machine learning with QGIS for analysis and classification. It is designed to handle large-scale classification tasks using extensive training data.

## Project Overview

The primary objective of this project is to classify the entire land cover of Uzbekistan into 12 distinct classes based on a provided training shapefile with over 200,000 features. The workflow is divided into two main stages:

1.  **Google Earth Engine (GEE) Processing**: This stage involves accessing and preprocessing satellite imagery (Sentinel-2, Landsat), calculating various spectral and topographic indices, and exporting a feature-rich composite image ready for classification. It also processes the training data to extract features for model training.

2.  **Local QGIS/Python Processing**: This stage uses the data exported from GEE to train machine learning models (e.g., Random Forest, Gradient Boosting). It then applies the best-performing model to classify the entire composite image of Uzbekistan. The script also handles post-processing, validation, and the generation of classification statistics.

## Directory Structure

The project is organized into the following directories:

```
classification/
│
├── config/
│   └── land_cover_config.py      # Configuration for classes, GEE, and QGIS
│
├── data/
│   ├── training/                 # Input training data (shapefiles, etc.)
│   └── results/                  # Output classification maps and reports
│
├── notebooks/
│   └── analysis.ipynb            # Jupyter notebook for exploration and visualization
│
├── scripts/
│   ├── gee/
│   │   └── uzbekistan_gee_processor.py  # GEE data processing script
│   └── qgis/
│       └── uzbekistan_qgis_processor.py # Local classification script
│
├── .gitignore                    # Files to be ignored by Git
├── main_workflow.py              # Main script to orchestrate the workflow
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

## Getting Started

### Prerequisites

- Python 3.8 or higher
- Git
- Google Earth Engine account and authenticated local environment
- QGIS (optional, for visualization and advanced post-processing)

### Installation

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd classification
    ```

2.  **Set up your training data:**
    Place your training shapefile (`landcover_training.shp` and its associated files) into the `data/training/` directory.

3.  **Install Python dependencies:**
    The required Python packages are listed in `requirements.txt`. They have been installed for you.

4.  **Authenticate with Google Earth Engine:**
    If you haven't already, you need to authenticate your local machine to use the GEE Python API. Run the following command in your terminal and follow the instructions:
    ```bash
    earthengine authenticate
    ```

### Running the Workflow

The entire classification process can be managed through the `main_workflow.py` script.

1.  **Run the full workflow:**
    To execute the complete workflow from GEE processing to local classification, run the following command in your terminal:

    ```bash
    python main_workflow.py --full --training-shapefile "data/training/landcover_training.shp"
    ```

    This command will:
    - Start the GEE processing to create and export the satellite data composite and training features.
    - Wait for you to download the exported data from your Google Drive into the `data/` and `data/training/` folders.
    - Run the local QGIS/Python script to train the classifier and produce the final land cover map.

2.  **Running steps individually:**
    You can also run each major step of the workflow separately.

    - **Step 1: GEE Processing**
      ```bash
      python main_workflow.py --gee --training-shapefile "data/training/landcover_training.shp"
      ```
      After this step, you will need to go to your Google Drive, find the exported files (e.g., `uzbekistan_composite_*.tif` and `training_data_*.csv`), and download them. Place the composite image in `data/` and the training CSV in `data/training/`. Rename the training CSV to `uzbekistan_training_data.csv`.

    - **Step 2: QGIS/Python Local Processing**
      ```bash
      python main_workflow.py --qgis
      ```
      This step assumes that the necessary data from GEE has been downloaded and placed in the correct directories.

## Outputs

The final classification results will be saved in the `data/results/` directory. This includes:

- The final classified land cover map as a GeoTIFF file.
- The trained machine learning model (`.joblib` file).
- A confusion matrix plot (`.png` file) for accuracy assessment.
- A CSV file with classification statistics, including the area of each land cover class.
