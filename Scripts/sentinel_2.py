# -*- coding: utf-8 -*-
"""
Sentinel-2 Vegetation & Urban Change Analysis (2016–2026)
@author: Uzair

"""

#----------Libraries---------
import rasterio
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from rasterio.windows import from_bounds
from rasterio.warp import transform_bounds



#==========Importing Data=============

#----Sentinel-1 Bands

#2016

red_path_16 = "Sentinel2/2016/B04_R10m.jp2"
nir_path_16 = "Sentinel2/2016/B08_R10m.jp2"
green_path_16 = "Sentinel2/2016/B03_R10m.jp2"
blue_path_16 = "Sentinel2/2016/B02_R10m.jp2"
swir_path_16 = "Sentinel2/2016/B11_R20m.jp2"

#2020

red_path_20 = "Sentinel2/2020/B04_R10m.jp2"
nir_path_20 = "Sentinel2/2020/B08_R10m.jp2"
green_path_20 = "Sentinel2/2020/B03_R10m.jp2"
blue_path_20 = "Sentinel2/2020/B02_R10m.jp2"
swir_path_20 = "Sentinel2/2020/B11_R20m.jp2"

#2026

red_path_26 = "Sentinel2/2026/B04_R10m.jp2"
nir_path_26 = "Sentinel2/2026/B08_R10m.jp2"
green_path_26 = "Sentinel2/2026/B03_R10m.jp2"
blue_path_26 = "Sentinel2/2026/B02_R10m.jp2"
swir_path_26 = "Sentinel2/2026/B11_R20m.jp2"


#===========Area of interest (Islamabad)==========

#---------Clipping-------------

west, south, east, north = 72.85, 33.55, 73.15, 33.85

def read_clipped_band(path):
    with rasterio.open(path) as src:
        bounds = transform_bounds(
            "EPSG:4326",
            src.crs,
            west, south, east, north,
            densify_pts=21
        )
        window = from_bounds(*bounds, transform=src.transform)
        band = src.read(1, window=window).astype(np.float32)
    return band

#----Loading bands---------

#2016
red_16 = read_clipped_band(red_path_16)
nir_16 = read_clipped_band(nir_path_16)
green_16 = read_clipped_band(green_path_16)
blue_16 = read_clipped_band(blue_path_16)
swir_16 = read_clipped_band(swir_path_16)

#2020

red_20 = read_clipped_band(red_path_20)
nir_20 = read_clipped_band(nir_path_20)
green_20 = read_clipped_band(green_path_20)
blue_20 = read_clipped_band(blue_path_20)
swir_20 = read_clipped_band(swir_path_20)

#2026

red_26 = read_clipped_band(red_path_26)
nir_26 = read_clipped_band(nir_path_26)
green_26 = read_clipped_band(green_path_26)
blue_26 = read_clipped_band(blue_path_26)
swir_26 = read_clipped_band(swir_path_26)

#=============================================================
#====================NDVI Calculation=========================
#=============================================================


#--------Scaling----

red_16 /= 10000
red_20 /= 10000
red_26 /= 10000
nir_16 /= 10000
nir_20 /= 10000
nir_26 /= 10000
     
def compute_ndvi(nir, red):
    ndvi = np.where(
        (nir + red) == 0,
        np.nan,
        (nir - red) / (nir + red)
    )
    return ndvi
   
ndvi_16 = compute_ndvi(nir_16, red_16)
ndvi_20 = compute_ndvi(nir_20, red_20)
ndvi_26 = compute_ndvi(nir_26, red_26)

#-----Change in NDVI------

delta_16_20 = ndvi_20 - ndvi_16
delta_20_26 = ndvi_26 - ndvi_20
delta_16_26 = ndvi_26 - ndvi_16

#print("2016 NDVI:", np.nanmin(ndvi_16), np.nanmax(ndvi_16), np.nanmean(ndvi_16))
#print("2020 NDVI:", np.nanmin(ndvi_20), np.nanmax(ndvi_20), np.nanmean(ndvi_20))
#print("2026 NDVI:", np.nanmin(ndvi_26), np.nanmax(ndvi_26), np.nanmean(ndvi_26))



#---------Visualising NDVI---------
'''
# Clip NDVI for display
ndvi_16_plot = np.clip(ndvi_16, -0.1, 0.7)
ndvi_20_plot = np.clip(ndvi_20, -0.1, 0.7)
ndvi_26_plot = np.clip(ndvi_26, -0.1, 0.7)


plt.figure(figsize=(6,6))
plt.imshow(ndvi_16, cmap="RdYlGn", vmin=-0.1, vmax=0.8)
plt.colorbar(label="NDVI")
plt.title("NDVI – Islamabad (2016)")
plt.axis("off")
plt.show()

'''


#---------------NDVI for each year---------------
fig, axes = plt.subplots(1, 3, figsize=(18,6))

years = ["2016", "2020", "2026"]
ndvis = [ndvi_16, ndvi_20, ndvi_26]

for ax, ndvi, year in zip(axes, ndvis, years):
    im = ax.imshow(ndvi, cmap="RdYlGn", vmin=-0.1, vmax=0.7)
    ax.set_title(f"NDVI {year}")
    ax.axis("off")

fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.7, label='GREEN = Healthy Vegetation | Red = Urban', pad=0.02)
plt.savefig("Outputs/NDVI_2016-2020-2026.png", dpi=600, bbox_inches='tight')
plt.show()

'''#--------Change in NDVI over the years-----------
fig, axes = plt.subplots(1,3, figsize=(18,6))
diff_maps = [delta_16_20, delta_20_26, delta_16_26]
titles = ['NDVI Change 2016-2020', 'NDVI Change 2020-2026', 'NDVI Change 2016-2026']

for ax, diff, title in zip(axes, diff_maps, titles):
    im = ax.imshow(diff, cmap='bwr', vmin=-0.4, vmax=0.4)
    ax.set_title(title)
    ax.axis('off')

fig.colorbar(im, ax=axes, fraction=0.03, pad=0.04)
plt.show()'''


#==========Vegetation gain and loss============

#--------Threshold-----------

loss_thr = -0.1
gain_thr = 0.1

#-------Classifying the change------

#2016-2026
loss_16_26 = delta_16_26 < loss_thr
gain_16_26 = delta_16_26 > gain_thr
stable_16_26 = (delta_16_26 >= loss_thr) & (delta_16_26 <= gain_thr)

#--------Calculating the statistics-----------

total_pixels = np.sum(~np.isnan(delta_16_26))

loss_pixels = np.sum(loss_16_26)
gain_pixels = np.sum(gain_16_26)
stable_pixels = np.sum(stable_16_26)

print("Vegetation NDVI decline %:", (loss_pixels)/(total_pixels)*100)
print("Vegetation NDVI gain %:", (gain_pixels / total_pixels) * 100)
print("Stable NDVI %:", (stable_pixels / total_pixels) * 100)

#------Visualising the gain/loss-------

veg_change_map = np.zeros(delta_16_26.shape)
veg_change_map[loss_16_26] = -1
veg_change_map[gain_16_26] = 1

plt.figure(figsize=(7,7))
plt.imshow(veg_change_map, cmap='bwr', vmin=-1, vmax=1)
plt.title("Vegetation Change (2016–2026)")
plt.colorbar(label='-1 = Loss | 0 = Stable | 1 = Gain')
plt.axis('off')
plt.savefig("Outputs/Vegetation_Change_2016-2026.png", dpi=600, bbox_inches='tight')
plt.show()

#=============================================================
#====================NDVI END=================================
#=============================================================


#=============================================================
#====================NDBI Calculation=========================
#=============================================================

#--------Scaling B11 SWIR--------------

swir_16 /= 10000
swir_20 /= 10000
swir_26 /= 10000

#----------Resampling NIR to 20m--------------

def downsample_band(band): return band[::2, ::2] 
nir_16_r = downsample_band(nir_16) 
nir_20_r = downsample_band(nir_20) 
nir_26_r = downsample_band(nir_26) 

# Resize downsampled NIR to match SWIR 
def match_shape(swir, nir_downsampled):
    swir_h, swir_w = swir.shape 
    nir_h, nir_w = nir_downsampled.shape 
    return nir_downsampled[:swir_h, :swir_w] # crop if needed 

nir_16_r = match_shape(swir_16, nir_16_r) 
nir_20_r = match_shape(swir_20, nir_20_r) 
nir_26_r = match_shape(swir_26, nir_26_r)

#---------Calcualting NDBI---------------

#2016
ndbi_16 = np.where(
    (swir_16 + nir_16_r) == 0,
    np.nan,
    (swir_16 - nir_16_r) / (swir_16 + nir_16_r)
)
#2020
ndbi_20 = np.where(
    (swir_20 + nir_20_r) == 0,
    np.nan,
    (swir_20 - nir_20_r) / (swir_20 + nir_20_r)
)

#2026
ndbi_26 = np.where(
    (swir_26 + nir_26_r) == 0,
    np.nan,
    (swir_26 - nir_26_r) / (swir_26 + nir_26_r)
)

#-----Change in NDBI------

urban_delta_16_20 = ndbi_20 - ndbi_16
urban_delta_20_26 = ndbi_26 - ndbi_20
urban_delta_16_26 = ndbi_26 - ndbi_16



#==============Visualisation================

#---------------NDBI for each year---------------
from matplotlib.colors import TwoSlopeNorm

vmin_clip, vmax_clip = -0.4, 0.2  # narrow the range for better contrast
norm = TwoSlopeNorm(vmin=vmin_clip, vcenter=0, vmax=vmax_clip)

fig, axes = plt.subplots(1, 3, figsize=(18,6))
years = ["2016", "2020", "2026"]
ndbis = [ndbi_16, ndbi_20, ndbi_26]

for ax, ndbi, year in zip(axes, ndbis, years):
    im = ax.imshow(ndbi, cmap="bwr", norm=norm)
    ax.set_title(f"NDBI {year}")
    ax.axis("off")

fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.7, label='BLUE = Vegetation | Red = Urban', pad=0.02)
plt.savefig("Outputs/NDBI_2016-2020-2026.png", dpi=600, bbox_inches='tight')
plt.show()

'''#--------Change in NDBI over the years-----------
fig, axes = plt.subplots(1,3, figsize=(18,6))
diff_maps = [urban_delta_16_20, urban_delta_20_26, urban_delta_16_26]
titles = ['NDBI Change 2016-2020', 'NDBI Change 2020-2026', 'NDBI Change 2016-2026']

for ax, diff, title in zip(axes, diff_maps, titles):
    diff = np.where(np.abs(diff) < 0.05, np.nan, diff)
    im = ax.imshow(diff, cmap='bwr', vmin=-0.4, vmax=0.4)
    ax.set_title(title)
    ax.axis('off')

fig.colorbar(im, ax=axes, fraction=0.03, pad=0.04)

plt.show()'''

#=============Urbanisation Gain & Loss================

#--------Threshold-----------

urban_loss_thr = -0.07
urban_gain_thr = 0.07

#-------Classifying the change------

#2016-2026
urban_loss_16_26 = urban_delta_16_26 < urban_loss_thr
urban_gain_16_26 = urban_delta_16_26 > urban_gain_thr
urban_stable_16_26 = (urban_delta_16_26 >= urban_loss_thr) & (urban_delta_16_26 <= urban_gain_thr)

#--------Calculating the statistics-----------

valid = ~np.isnan(urban_delta_16_26)

urban_total_pixels = np.sum(valid)

urban_loss_pixels = np.sum(urban_loss_16_26 & valid)
urban_gain_pixels = np.sum(urban_gain_16_26 & valid)
urban_stable_pixels = np.sum(urban_stable_16_26 & valid)


print("Urban NDBI decline %:", (urban_loss_pixels)/(urban_total_pixels)*100)
print("Urban NDBI gain %:", (urban_gain_pixels / urban_total_pixels) * 100)
print("Stable NDBI %:", (urban_stable_pixels / urban_total_pixels) * 100)

#------Visualising the gain/loss-------

urban_change_map = np.zeros_like(urban_delta_16_26)
urban_change_map[urban_loss_16_26] = -1
urban_change_map[urban_gain_16_26] = 1

plt.figure(figsize=(7,7))
plt.imshow(urban_change_map, cmap='bwr', vmin=-1, vmax=1)
plt.title("Urban Change (2016–2026)")
plt.colorbar(label='-1 = Loss | 0 = Stable | 1 = Gain')
plt.axis('off')
plt.savefig("Outputs/Urban_Change_2016-2026.png", dpi=600, bbox_inches='tight')
plt.show()

#=================TEST=======================
'''
valid = ~np.isnan(urban_delta_16_26)
print("Min:", urban_delta_16_26[valid].min())
print("Max:", urban_delta_16_26[valid].max())
print("Mean:", urban_delta_16_26[valid].mean())
'''
#=============================================================
#====================NDBI END=================================
#=============================================================

#=============================================================
#====================NDVi AND NDBI OVERLAY====================
#=============================================================

#-------------Resampling to match B11--------
from skimage.transform import resize

delta_16_26_d = resize(
    delta_16_26,
    urban_delta_16_26.shape,
    order=1,              # bilinear
    preserve_range=True,
    anti_aliasing=True
)
delta_16_20_d = resize(
    delta_16_20,
    urban_delta_16_20.shape,
    order=1,              # bilinear
    preserve_range=True,
    anti_aliasing=True
)
delta_20_26_d = resize(
    delta_20_26,
    urban_delta_20_26.shape,
    order=1,              # bilinear
    preserve_range=True,
    anti_aliasing=True
)

#-------------Thresholds----------------------
ndvi_loss_thr = -0.1
ndbi_gain_thr = 0.03

veg_to_urban_1 = (delta_16_26_d < ndvi_loss_thr) & (urban_delta_16_26 > ndbi_gain_thr)
veg_to_urban_2 = (delta_16_20_d < ndvi_loss_thr) & (urban_delta_16_20 > ndbi_gain_thr)
veg_to_urban_3 = (delta_20_26_d < ndvi_loss_thr) & (urban_delta_20_26 > ndbi_gain_thr)


#------------Visualisation-----------------

cmap = ListedColormap(["black", "red"])  # 0 = no change, 1 = conversion
plt.figure(figsize=(7,7))
plt.imshow(veg_to_urban_1.astype(int), cmap=cmap)
plt.title("Vegetation → Urban Conversion (2016–2026)")
plt.colorbar(ticks=[0,1], label="0 = No | 1 = Conversion")
plt.axis("off")
plt.savefig("Outputs/Vegetation_to_Urban_2016-2026.png", dpi=600, bbox_inches='tight')
plt.show()

#-----------Area Completion----------------
pixel_size = 20  # meters
pixel_area_m2 = pixel_size ** 2
conversion_pixels = np.sum(veg_to_urban_1)
conversion_area_km2_1 = (conversion_pixels * pixel_area_m2) / 1e6
conversion_pixels = np.sum(veg_to_urban_2)
conversion_area_km2_2 = (conversion_pixels * pixel_area_m2) / 1e6
conversion_pixels = np.sum(veg_to_urban_3)
conversion_area_km2_3 = (conversion_pixels * pixel_area_m2) / 1e6
print("Vegetation → Urban (2016–2020 km²):", conversion_area_km2_2)
print("Vegetation → Urban (2020–2026 km²):", conversion_area_km2_3)
print("Vegetation → Urban (2016–2026 km²):", conversion_area_km2_1)


from PIL import Image

# Load screenshot properly
google_img = Image.open("Outputs/Refrence_map.png")

fig, axes = plt.subplots(2, 2, figsize=(11,13))



#===========SUBPLOT============================
# --- 1 Vegetation Change ---
axes[0,1].imshow(veg_change_map, cmap='bwr', vmin=-1, vmax=1)
axes[0,1].set_title("Vegetation Change (2016–2026)")
axes[0,1].axis("off")

# --- 2 Urban Change ---
axes[1,0].imshow(urban_change_map, cmap='bwr', vmin=-1, vmax=1)
axes[1,0].set_title("Urban Change (2016–2026)")
axes[1,0].axis("off")

# --- 3 Google Earth --
axes[0,0].imshow(google_img)
axes[0,0].set_title("Google Earth (Reference)")
axes[0,0].axis("off")

# --- 4 Vegetation → Urban ---
axes[1,1].imshow(veg_to_urban_1, cmap=cmap, vmin=0, vmax=1)
axes[1,1].set_title("Vegetation → Urban Conversion")
axes[1,1].axis("off")

plt.tight_layout()
plt.savefig("Outputs/Comparison_with_GoogleEarth.png", dpi=600, bbox_inches='tight')
plt.show()


#===============PIE CHARTS====================================

#-----------------DELTA NDVI--------------------
veg_vals = [12.48, 13.41, 74.11]
veg_labels = ["Loss", "Gain", "Stable"]

plt.figure(figsize=(6,6))
plt.pie(
    veg_vals,
    labels=veg_labels,
    autopct="%.1f%%",
    startangle=90,
    colors=["#d73027", "#1a9850", "#cccccc"]
)
plt.title("Delta %pixels w.r.t NDVI threshold\n (Index change 2016-2026)")
plt.savefig("Outputs/Delta_NDVI_pie.png", dpi=600, bbox_inches='tight')
plt.show()

#-----------------DELTA NDBI--------------------
urban_vals = [19.82, 15.64, 64.54]
urban_labels = ["Loss", "Gain", "Stable"]

plt.figure(figsize=(6,6))
plt.pie(
    urban_vals,
    labels=urban_labels,
    autopct="%.1f%%",
    startangle=90,
    colors=["#4575b4", "#fee090", "#cccccc"]
)
plt.title("Delta %pixels w.r.t NDBI threshold\n (Index change 2016-2026)")
plt.savefig("Outputs/Delta_NDBI_pie.png", dpi=600, bbox_inches='tight')
plt.show()

#----------------DELTA AREA------------------------
labels = ["2016–2020", "2020–2026", "2016–2026"]
values = [
    conversion_area_km2_2,
    conversion_area_km2_3,
    conversion_area_km2_1
]
plt.figure(figsize=(6,4))
plt.bar(labels, values)
plt.ylabel("Area (km²)")
plt.title("Vegetation Converted to Urban")
plt.grid(axis="y", alpha=0.3)
plt.savefig("Outputs/Veg_to_Urban_Bar.png", dpi=600, bbox_inches='tight')
plt.show()


#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#+============================================================+
#+===================END OF SENTINEL 2========================+
#+============================================================+
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++































