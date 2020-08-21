import rasterio.features
from geojson import Point, Feature, FeatureCollection, dump
import numpy as np
import pandas as pd
import os
import sys
from PIL import Image
import copy
from matplotlib.colors import Normalize
from matplotlib import cm

# Get centroid of each tile because plotly doesn't support hover text for a polygon shape, 
# drawing tiny scatter points at the center of each tile so that when we hover over a tile, 
# the text will display
def get_centers(geojson):
    lon, lat = [], []

    for k in range(len(geojson['features'])):
        geometry = geojson['features'][k]['geometry']

        if geometry['type'] == 'Polygon':
            coords=np.array(geometry['coordinates'][0])

        # the centroids
        lon.append(sum(coords[:,0]) / len(coords[:,0]))
        lat.append(sum(coords[:,1]) / len(coords[:,1]))

    return lon, lat

def scalarmappable(cmap, cmin, cmax):
        colormap = cm.get_cmap(cmap)
        norm = Normalize(vmin=cmin, vmax=cmax)
        return cm.ScalarMappable(norm=norm, cmap=colormap)

def get_scatter_colors(sm, df):
    grey = 'rgba(128,128,128,1)'
    return ['rgba' + str(sm.to_rgba(m, bytes = True, alpha = 1)) if not np.isnan(m) else grey for m in df]

def get_colorscale(sm, df, cmin, cmax):
    xrange = np.linspace(0, 1, len(df))
    values = np.linspace(cmin, cmax, len(df))

    return [[i, 'rgba' + str(sm.to_rgba(v, bytes = True))] for i,v in zip(xrange, values) ]
  
'''def get_hover_text(df) :
    text_value = df
    with_data = '<b>{}</b> <br> {} mangrove'
    no_data = '<b>{}</b> <br> no data'

    return [with_data.format(p,v) if v != 'nan%' else no_data.format(p) for p,v in zip(df.index, text_value)]
'''
# create dictionary that plotly uses to draw the shape border
def make_sources(geojson):
    sources = []
    geojson_copy = copy.deepcopy(geojson['features']) # do not overwrite the original file

    for feature in geojson_copy:
        sources.append(dict(type = 'FeatureCollection',
                            features = [feature]))
    return sources

# return the bounds of the reassembled tif
def get_latlonminmax(geojson):

  lats = []
  lons = []
  for i in range(len(geojson['features'])):
    for coord_list in geojson['features'][i]['geometry']['coordinates']:
      for coord_pair in coord_list:
        lons.append(coord_pair[0])
        lats.append(coord_pair[1])

  latmin= min(lats)
  latmax= max(lats)
  lonmin= min(lons)
  lonmax= max(lons)
  return latmin, latmax, lonmin, lonmax


def rgb_to_hsv(rgb):
    # Translated from source of colorsys.rgb_to_hsv
    # r,g,b should be a numpy arrays with values between 0 and 255
    # rgb_to_hsv returns an array of floats between 0.0 and 1.0.
    rgb = rgb.astype('float')
    hsv = np.zeros_like(rgb)
    # in case an RGBA array was passed, just copy the A channel
    hsv[..., 3:] = rgb[..., 3:]
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    maxc = np.max(rgb[..., :3], axis=-1)
    minc = np.min(rgb[..., :3], axis=-1)
    hsv[..., 2] = maxc
    mask = maxc != minc
    hsv[mask, 1] = (maxc - minc)[mask] / maxc[mask]
    rc = np.zeros_like(r)
    gc = np.zeros_like(g)
    bc = np.zeros_like(b)
    rc[mask] = (maxc - r)[mask] / (maxc - minc)[mask]
    gc[mask] = (maxc - g)[mask] / (maxc - minc)[mask]
    bc[mask] = (maxc - b)[mask] / (maxc - minc)[mask]
    hsv[..., 0] = np.select(
        [r == maxc, g == maxc], [bc - gc, 2.0 + rc - bc], default=4.0 + gc - rc)
    hsv[..., 0] = (hsv[..., 0] / 6.0) % 1.0
    return hsv

def hsv_to_rgb(hsv):
    # Translated from source of colorsys.hsv_to_rgb
    # h,s should be a numpy arrays with values between 0.0 and 1.0
    # v should be a numpy array with values between 0.0 and 255.0
    # hsv_to_rgb returns an array of uints between 0 and 255.
    rgb = np.empty_like(hsv)
    rgb[..., 3:] = hsv[..., 3:]
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    i = (h * 6.0).astype('uint8')
    f = (h * 6.0) - i
    p = v * (1.0 - s)
    q = v * (1.0 - s * f)
    t = v * (1.0 - s * (1.0 - f))
    i = i % 6
    conditions = [s == 0.0, i == 1, i == 2, i == 3, i == 4, i == 5]
    rgb[..., 0] = np.select(conditions, [v, q, p, p, t, v], default=v)
    rgb[..., 1] = np.select(conditions, [v, v, v, q, p, p], default=t)
    rgb[..., 2] = np.select(conditions, [v, p, t, v, v, q], default=p)
    return rgb.astype('uint8')


def shift_hue(arr,hout):
    hsv=rgb_to_hsv(arr)
    hsv[...,0]=hout
    rgb=hsv_to_rgb(hsv)
    return rgb

# input: filename
# output: returns a downsampled image object
def downsample_im(FILENAME, ds_factor):
    image = Image.open(FILENAME)
    (width, height) = (int(image.width / ds_factor), int(image.height / ds_factor))
    image = image.resize((width,height), Image.ANTIALIAS)


def get_im(FILENAME, hue):
    Image.MAX_IMAGE_PIXELS = None

    # downsample by 
    # image = downsample_im(FILENAME, ds_factor)
    image = Image.open(FILENAME)
    arr = np.array(image)

    image_hue = Image.fromarray(shift_hue(arr, hue), 'RGBA')
    return image_hue


# create geojson file to store the polygon
def create_geojson(FILENAME, final_filename):
    # list of GeoJSON feature objects (later this becomes a FeatureCollection)
    features = []

    if os.path.isfile(FILENAME):
        print('File that is being opened: ', FILENAME)
        dataset = rasterio.open(os.path.abspath(FILENAME))

        # Read the dataset's valid data mask as a ndarray.
        mask = dataset.dataset_mask()

        # Extract feature shapes and values from the array.
        for geom, val in rasterio.features.shapes(mask, transform=dataset.transform):
        # val is the value of the raster feature corresponding to the shape
        # val = 0: no shape and val = 255 means shape (drone footage, aka tiles we want)
            if (val == 255.0):  

                # Transform shapes from the dataset's own coordinate reference system to CRS84 (EPSG:4326) tbh idk what this means
                geom = rasterio.warp.transform_geom(dataset.crs, 'EPSG:4326', geom, precision=30)

                # store GeoJSON shapes to features list.
                # might have to put the probabilty value in properties ... tbd
                features.append(Feature(geometry=geom, properties={'name': FILENAME}))

        # all features become a feature collection
        feature_collection = FeatureCollection(features)

        # Feature collection goes into a geojson file
        with open(final_filename, 'w') as f:
            dump(feature_collection, f)
        
        return feature_collection
    else:
        return None