#!/usr/bin/env python
"""Create a matplotlib animation from route data."""
from __future__ import print_function, division
import numpy as np
import os
import sys
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation

from route_plotter import parse_route, stitch_tiles, coords_to_bbox


def main():
  args = parse_args()
  routes = _parse_routes(args.route)
  if not routes:
    print('No valid route data provided.', file=sys.stderr)
    return
  print('Parsed', len(routes), 'valid routes.')

  routes = _filter_routes(routes, loop_gap_threshold=args.max_loop_gap,
                          start_cluster_radius=args.max_start_dist)
  assert bool(routes), '_filter_routes removed all routes'
  print('Filtered to', len(routes), 'routes.')

  if not os.path.exists(args.tile_cache):
    os.makedirs(args.tile_cache)
  bg_img, bg_extent = stitch_tiles(_bounding_box(routes), zoom=args.zoom,
                                   cachedir=args.tile_cache,
                                   flatten=(not args.color_tiles),
                                   margin_scale=args.max_margin)
  print('Assembled %d x %d background map image.' % bg_img.shape[:2])

  fig, ax = _setup_figure(bg_img, bg_extent, scale=args.scale)
  lines = ax.plot(np.zeros((0, len(routes))), c='r', alpha=0.75, zorder=1)
  heads = ax.scatter([], [], c='b', zorder=2)
  no_head = np.full(2, np.nan)
  end_pts = np.zeros((len(routes), 2))
  plot_data = [(np.flipud(r.T), l) for r, l in zip(routes, lines)]

  def update_lines(num):
    for i, (route, line) in enumerate(plot_data):
      if num < route.shape[0]:
        line.set_data(route[:, :num])
        end_pts[i] = route[:, num]
      elif num == route.shape[1]:
        line.set_data(route[:, :num])
        end_pts[i] = no_head
      else:
        end_pts[i] = no_head
    heads.set_offsets(end_pts)
    return lines

  num_frames = max(r.shape[0] for r in routes)
  dpi = plt.rcParams.get('figure.dpi', 100.0)
  anim = FuncAnimation(fig, update_lines, frames=num_frames, blit=True,
                       interval=100, repeat=True, repeat_delay=150)
  if args.save:
    anim.save(args.save, dpi=dpi, savefig_kwargs=dict(pad_inches=0))
  else:
    plt.show()


def parse_args():
  cache_dir = os.path.join(os.path.dirname(__file__), 'tiles')
  ap = ArgumentParser(description=__doc__,
                      formatter_class=ArgumentDefaultsHelpFormatter)
  ap.add_argument('--color-tiles', action='store_true',
                  help='Use full-color map tiles.')
  ap.add_argument('--scale', type=float, default=1,
                  help='Figure size scale fraction.')
  ap.add_argument('--zoom', type=int, default=None, help='Map zoom level.')
  ap.add_argument('--tile-cache', type=str, default=cache_dir,
                  help='Directory for storing cached map tiles.')
  ap.add_argument('--save', type=str, help='Save animation to disk.')
  ap.add_argument('--max-margin', type=float, default=0.05,
                  help='Keep up to this fraction of width/height as margin.')
  ap.add_argument('--max-loop-gap', type=float, default=200,
                  help='Maximum gap between start/end points, in meters.')
  ap.add_argument('--max-start-dist', type=float, default=200,
                  help=('Maximum distance from route start to the '
                        'mean starting location, in meters.'))
  ap.add_argument('route', type=open, nargs='+', help='Route file(s) to use.')
  return ap.parse_args()


def _setup_figure(bg_img, bg_extent, scale=1.0):
  plt.rc('figure', autolayout=False)  # turn off tight_layout
  dpi = plt.rcParams.get('figure.dpi', 100.0)
  fig = plt.figure(dpi=dpi, frameon=False)

  # scale the figure to fit the bg image
  bg_height, bg_width = bg_img.shape[:2]
  fig.set_size_inches(bg_width / dpi * scale, bg_height / dpi * scale)

  ax = fig.add_axes([0, 0, 1, 1])
  ax.set_axis_off()
  ax.xaxis.set_major_locator(plt.NullLocator())
  ax.yaxis.set_major_locator(plt.NullLocator())
  ax.imshow(bg_img, zorder=0, extent=bg_extent, cmap='Greys_r')
  ax.autoscale(False)
  ax.margins(0, 0)
  return fig, ax


def _parse_routes(file_paths):
  all_coords = []
  for fh in file_paths:
    coords = parse_route(fh)
    if coords.ndim == 2 and coords.shape[0] >= 2 and coords.shape[1] == 2:
      all_coords.append(coords)
  return all_coords


def _filter_routes(routes, loop_gap_threshold=200, start_cluster_radius=200):
  """Selects routes that:
    - start and end in roughly the same place (forming a loop)
    - start close to the mean starting location
  """
  starts = np.array([r[0] for r in routes])
  ends = np.array([r[-1] for r in routes])

  # convert lat/lon displacements into approximate distances (meters)
  gap_dists = _greatcircle_distance(starts, ends)
  mask = gap_dists < loop_gap_threshold

  # iterative thresholding to narrow down the center location
  thresh = np.inf
  while thresh > start_cluster_radius:
    start_dist = _greatcircle_distance(starts, starts[mask].mean(axis=0))
    thresh = np.percentile(start_dist[mask], 95)
    mask &= start_dist < thresh
  return [routes[i] for i in np.where(mask)[0]]


def _bounding_box(routes):
  min_coord = np.array([r.min(axis=0) for r in routes]).min(axis=0)
  max_coord = np.array([r.max(axis=0) for r in routes]).max(axis=0)
  return coords_to_bbox([min_coord, max_coord])


def _greatcircle_distance(latlon0, latlon1):
  EARTH_RADIUS = 6371000  # approximate spherical radius
  lat0, lon0 = np.deg2rad(latlon0).T
  lat1, lon1 = np.deg2rad(latlon1).T
  return EARTH_RADIUS * np.arccos(
      np.sin(lat0) * np.sin(lat1) +
      np.cos(lat0) * np.cos(lat1) * np.cos(np.abs(lon0 - lon1)))


if __name__ == '__main__':
  main()
