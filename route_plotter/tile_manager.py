from __future__ import print_function, division
import numpy as np
import os
from collections import namedtuple
from random import choice

try:
  from PIL import Image
except ImportError:
  import Image

try:
  from urllib.request import urlretrieve
except ImportError:
  from urllib import urlretrieve

__all__ = ['stitch_tiles', 'coords_to_bbox']

# template URL for map tiles
TILESET_URL_TEMPLATE = 'http://{abc}.tile.openstreetmap.org/{zoom}/{x}/{y}.png'
# width/height of tile images
TILE_SIZE_PX = 256
# more readable form of bounding box tuple
_BBox = namedtuple('BBox', ('lon_min', 'lon_max', 'lat_min', 'lat_max'))


def coords_to_bbox(coords):
  coords = np.asarray(coords)
  lat_min, lon_min = coords.min(axis=0)
  lat_max, lon_max = coords.max(axis=0)
  return _BBox(lon_min, lon_max, lat_min, lat_max)


def stitch_tiles(bbox, zoom=None, cachedir='.', flatten=True,
                 margin_scale=0.05, ideal_width=1024, max_tiles=100):
  bbox = _BBox(*bbox)

  if zoom is None:
    zoom = _ideal_zoom(bbox, ideal_width)
  elif not (0 <= zoom < 18):
    raise ValueError('Invalid zoom level: %s' % zoom)

  # convert the requested bbox to tile coordinates
  x0, y0 = _latlon_to_tile(bbox.lat_min, bbox.lon_min, zoom)
  x1, y1 = _latlon_to_tile(bbox.lat_max, bbox.lon_max, zoom)
  tile_xs = list(range(x0, x1+1) if x0 <= x1 else range(x1, x0+1))
  tile_ys = list(range(y0, y1+1) if y0 <= y1 else range(y1, y0+1))

  # make sure we aren't requesting too many
  num_tiles = len(tile_xs) * len(tile_ys)
  if num_tiles > max_tiles:
    raise ValueError('Too many tiles requested (%d), '
                     'try lowering the zoom level' % num_tiles)

  # get the tile images and concatenate them
  cache_tpl = os.path.join(cachedir, '{zoom}_{x}_{y}.png')
  images = []
  for y in tile_ys:
    row = [_get_tile(zoom, x, y, cache_tpl, flatten) for x in tile_xs]
    images.append(np.hstack(row))
  tile_image = np.vstack(images)

  # compute the bounding box of the map image
  tiles_bbox = coords_to_bbox([
      _tile_to_latlon(tile_xs[0], tile_ys[0], zoom),
      _tile_to_latlon(tile_xs[-1]+1, tile_ys[-1]+1, zoom)
  ])

  # pad the bounding box, up to the tile bbox limits
  margin_lon = (bbox.lon_max - bbox.lon_min) * margin_scale
  margin_lat = (bbox.lat_max - bbox.lat_min) * margin_scale
  padded_bbox = _BBox(
      lon_min=max(tiles_bbox.lon_min, bbox.lon_min - margin_lon),
      lon_max=min(tiles_bbox.lon_max, bbox.lon_max + margin_lon),
      lat_min=max(tiles_bbox.lat_min, bbox.lat_min - margin_lat),
      lat_max=min(tiles_bbox.lat_max, bbox.lat_max + margin_lat))

  # crop the image to avoid big empty bits
  px_lat, px_lon = tile_image.shape[:2]
  scale_lon = px_lon / (tiles_bbox.lon_max - tiles_bbox.lon_min)
  scale_lat = px_lat / (tiles_bbox.lat_max - tiles_bbox.lat_min)
  lon_idx = (np.array(padded_bbox[:2]) - tiles_bbox.lon_min) * scale_lon
  lat_idx = (np.array(padded_bbox[2:]) - tiles_bbox.lat_min) * scale_lat
  lon_idx[0], lat_idx[0] = np.floor([lon_idx[0], lat_idx[0]])
  lon_idx[1], lat_idx[1] = np.ceil([lon_idx[1], lat_idx[1]])
  j0, j1 = np.clip(lon_idx.astype(int), 0, px_lon)
  i1, i0 = np.clip(px_lat - lat_idx.astype(int), 0, px_lat)
  tile_image = tile_image[i0:i1, j0:j1]

  # sanity check
  if 0 in tile_image.shape:
    raise ValueError("Invalid image shape: %s" % tile_image.shape)

  return tile_image, padded_bbox


def _get_tile(zoom, x, y, cache_tpl, flatten):
  cache_path = cache_tpl.format(zoom=zoom, x=x, y=y)
  try:
    return _imread(cache_path, flatten=flatten)
  except IOError:
    _download_tile(x, y, zoom, cache_path)
  return _imread(cache_path, flatten=flatten)


def _ideal_zoom(bbox, ideal_width):
  ideal_tiles = ideal_width / TILE_SIZE_PX
  lat_diff = bbox.lat_max - bbox.lat_min
  return int(np.log2(ideal_tiles * 360 / lat_diff))


def _latlon_to_tile(lat, lon, zoom):
  n = 2 ** zoom
  x = n * (lon + 180) / 360.
  y = n * (1 - (np.arcsinh(np.tan(np.deg2rad(lat))) / np.pi)) / 2.
  return int(x), int(y)


def _tile_to_latlon(x, y, zoom):
  n = 2. ** zoom
  lon_deg = (x / n) * 360 - 180
  lat_rad = np.arctan(np.sinh(np.pi * (1 - 2 * y / n)))
  return np.rad2deg(lat_rad), lon_deg


def _download_tile(x, y, zoom, cache_path):
  url = TILESET_URL_TEMPLATE.format(zoom=zoom, x=x, y=y, abc=choice('abc'))
  print('Downloading', url, 'to', cache_path)
  urlretrieve(url, cache_path)


def _imread(path, flatten=False):
  img = Image.open(path)
  if img.mode == 'P':
    img = img.convert('RGB')
  if flatten:
    img = img.convert('F')
  return np.array(img)
