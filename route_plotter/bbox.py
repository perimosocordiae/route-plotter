from __future__ import print_function, division, absolute_import
import numpy as np
from collections import namedtuple

class BBox(namedtuple('_BBox',
    ('lon_min', 'lon_max', 'lat_min', 'lat_max'))):

  @classmethod
  def from_coords(cls, coords):
    coords = np.asarray(coords)
    lat_min, lon_min = coords.min(axis=0)
    lat_max, lon_max = coords.max(axis=0)
    return cls(lon_min, lon_max, lat_min, lat_max)

  @classmethod
  def from_routes(cls, routes):
    min_coord = np.min([r.min_coord() for r in routes], axis=0)
    max_coord = np.max([r.max_coord() for r in routes], axis=0)
    return cls.from_coords([min_coord, max_coord])

  def pad(self, margin_scale):
    margin_lon = (self.lon_max - self.lon_min) * margin_scale
    margin_lat = (self.lat_max - self.lat_min) * margin_scale
    return BBox(lon_min=self.lon_min - margin_lon,
                lon_max=self.lon_max + margin_lon,
                lat_min=self.lat_min - margin_lat,
                lat_max=self.lat_max + margin_lat)

  def crop(self, outer_bbox):
    return BBox(lon_min=max(outer_bbox.lon_min, self.lon_min),
                lon_max=min(outer_bbox.lon_max, self.lon_max),
                lat_min=max(outer_bbox.lat_min, self.lat_min),
                lat_max=min(outer_bbox.lat_max, self.lat_max))

