from __future__ import print_function, division, absolute_import
import numpy as np

from .bbox import BBox
from .parsers import parse_route

__all__ = ['Route', 'filter_routes', 'greatcircle_distance']


class Route(object):
  def __init__(self, coords, seconds=None, start_time=None, elevations=None):
    if seconds is not None:
      assert coords.shape[:1] == seconds.shape
    if elevations is not None:
      assert coords.shape[:1] == elevations.shape
    self.coords = coords
    self.seconds = seconds
    self.start_time = start_time
    self.elevations = elevations

  @classmethod
  def parse(cls, file_or_path, read_elevation=False):
    if hasattr(file_or_path, 'read'):
      fh = file_or_path
    else:
      fh = open(file_or_path)
    result = parse_route(fh, return_time=True, return_elevation=read_elevation)
    coords, times = result[:2]
    elevs = result[2] if read_elevation else None
    if coords.ndim != 2 or coords.shape[0] < 2 or coords.shape[1] != 2:
      raise ValueError('Invalid coords shape %s in %s' % (coords.shape, file_or_path))
    seconds = (times - times[0]).astype('timedelta64[s]').astype(int)
    return cls(coords, seconds=seconds, start_time=times[0], elevations=elevs)

  def start_coord(self):
    return self.coords[0]

  def stop_coord(self):
    return self.coords[-1]

  def min_coord(self):
    return self.coords.min(axis=0)

  def max_coord(self):
    return self.coords.max(axis=0)


def filter_routes(routes, loop_gap_threshold=200, start_cluster_radius=200):
  """Selects routes that:
    - start and end in roughly the same place (forming a loop)
    - start close to the mean starting location
  """
  starts = np.array([r.start_coord() for r in routes])
  ends = np.array([r.stop_coord() for r in routes])

  # convert lat/lon displacements into approximate distances (meters)
  gap_dists = greatcircle_distance(starts, ends)
  mask = gap_dists < loop_gap_threshold

  # iterative thresholding to narrow down the center location
  thresh = np.inf
  while thresh > start_cluster_radius:
    center = starts[mask].mean(axis=0)
    start_dist = greatcircle_distance(starts, center)
    thresh = max(np.percentile(start_dist[mask], 95),
                 start_cluster_radius)
    mask &= start_dist < thresh

  keep_indices, = np.where(mask)
  return [routes[i] for i in keep_indices]


def greatcircle_distance(latlon0, latlon1):
  EARTH_RADIUS = 6371000  # approximate spherical radius
  lat0, lon0 = np.deg2rad(latlon0).T
  lat1, lon1 = np.deg2rad(latlon1).T
  return EARTH_RADIUS * np.arccos(
      np.sin(lat0) * np.sin(lat1) +
      np.cos(lat0) * np.cos(lat1) * np.cos(np.abs(lon0 - lon1)))

