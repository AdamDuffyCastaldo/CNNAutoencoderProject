# Data Directory

This directory contains all data for the SAR autoencoder project.

## Structure

```
data/
├── raw/                    # Raw Sentinel-1 products
│   └── S1A_IW_GRDH_*.SAFE/ # Downloaded .SAFE directories
│
├── processed/              # Preprocessed full images
│   └── *.npy              # Preprocessed images as numpy arrays
│
└── patches/               # Extracted training patches
    ├── patches.npy        # Training patches (N, 256, 256)
    └── preprocess_params.npy  # Preprocessing parameters
```

## Data Sources

1. **Copernicus Data Space**: https://dataspace.copernicus.eu/
2. **Alaska Satellite Facility**: https://search.asf.alaska.edu/
3. **Google Earth Engine**: https://earthengine.google.com/

## Preprocessing Pipeline

```python
from src.data import preprocess_sar_image, extract_patches

# Load raw data
image = load_geotiff('data/raw/...')

# Preprocess
normalized, params = preprocess_sar_image(image)

# Extract patches
patches, positions = extract_patches(normalized)

# Save
np.save('data/patches/patches.npy', patches)
np.save('data/patches/preprocess_params.npy', params)
```

## Notes

- Raw Sentinel-1 GRD products are ~1 GB each
- Processed patches typically 100-500 MB
- Keep preprocessing parameters with patches for inverse transform
