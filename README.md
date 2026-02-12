# Sentinel-2-Vegetation-Urban-Change-Analysis-2016-2026-
This repository contains a Python project analyzing vegetation and urban dynamics in Islamabad, Pakistan, using Sentinel-2 imagery from 2016, 2020, and 2026. The analysis includes NDVI (Normalized Difference Vegetation Index), NDBI (Normalized Difference Built-up Index), vegetation-to-urban conversions, and visualizations.

---

## Features

- **NDVI Analysis**: Measures vegetation health and identifies areas of vegetation gain, loss, or stability.  
- **NDBI Analysis**: Tracks urban expansion and built-up area dynamics.  
- **Vegetation → Urban Conversion**: Identifies areas where vegetation has been converted to urban land.  
- **Comparison with Google Earth**: Side-by-side visualization of satellite data and reference imagery.  
- **Summary Statistics**: Percentage change in vegetation and urban areas over time, as well as conversion areas in km².  
- **Visualizations**: Includes color-coded maps, pie charts, and bar charts for easy interpretation.  

---

## Key Outputs

### Vegetation Change
`Outputs/Vegetation_Change_2016-2026.png`  
Red: vegetation loss | Blue: vegetation gain | Gray: stable.

### Urban Change
`output/Urban_Change_2016-2026.png`  
Red: urban gain | Blue: urban loss | Gray: stable.

### Vegetation → Urban Conversion
`output/Vegetation_to_Urban_2016-2026.png`  
Red: converted | Black: no change.

### Comparison with Google Earth
`output/Comparison_with_GoogleEarth.png`  
Top-left: Google Earth reference | Top-right: Vegetation change  
Bottom-left: Urban change | Bottom-right: Vegetation → Urban

### Delta Statistics
- **NDVI Changes Pie Chart**: `outputs/Delta_NDVI_pie.png`  
- **NDBI Changes Pie Chart**: `outputs/Delta_NDBI_pie.png`  
- **Vegetation → Urban Bar Chart**: `outputs/Veg_to_Urban_Bar.png`  

---

## Usage

1. Clone this repository:

```bash
git clone https://github.com/yourusername/sentinel2-veg-urban.git
cd sentinel2-veg-urban
