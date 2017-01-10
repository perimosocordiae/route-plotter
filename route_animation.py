#!/usr/bin/env python
"""Create a matplotlib animation from route data."""
from __future__ import print_function, division
import glob
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
  max_duration = max(s[-1] for _,s in routes)
  frame_times = np.linspace(0, max_duration, args.num_frames)
  anim = _animate(fig, ax, routes, frame_times)
  print('Prepared animation loop.')

  if args.save:
    dpi = plt.rcParams.get('figure.dpi', 100.0)
    anim.save(args.save, dpi=dpi, savefig_kwargs=dict(pad_inches=0))
  else:
    plt.show()


def _animate(fig, ax, routes, frame_data):
  lines = ax.plot(np.zeros((0, len(routes))), c='r', alpha=0.75, zorder=1)
  heads = ax.scatter([], [], c='b', zorder=2)
  timer = ax.text(1, 0, '0:00:00', transform=ax.transAxes,
                  bbox=dict(facecolor='white'),
                  verticalalignment='bottom', horizontalalignment='right')
  no_head = np.full(2, np.nan)
  end_pts = np.zeros((len(routes), 2))
  plot_data = []
  for i, line in enumerate(lines):
    r, s = routes[i]
    plot_data.append((np.fliplr(r), s, line))
  artists = lines + [heads, timer]

  def update_frame(num_seconds):
    for i, (xy, time, line) in enumerate(plot_data):
      idx = np.searchsorted(time, num_seconds)
      if idx < xy.shape[0]:
        line.set_data(xy[:idx].T)
        end_pts[i] = xy[idx]
      elif idx == xy.shape[0]:
        line.set_data(xy[:idx].T)
        end_pts[i] = no_head
      else:
        end_pts[i] = no_head
    heads.set_offsets(end_pts)
    # update the clock
    mins, secs = divmod(num_seconds, 60)
    hours, mins = divmod(mins, 60)
    timer.set_text('%d:%02d:%02d' % (hours, mins, secs))
    return artists

  return FuncAnimation(fig, update_frame, frames=frame_data, blit=True,
                       interval=100, repeat=True, repeat_delay=150)


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
  ap.add_argument('--num-frames', type=int, default=500,
                  help='Number of frames to animate.')
  ap.add_argument('route', type=str, nargs='+', help='Route file(s) to use.')
  args = ap.parse_args()
  if len(args.route) == 1:
    args.route = glob.glob(args.route[0])
  args.route = map(open, args.route)
  return args


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
    coords, times = parse_route(fh, return_time=True)
    if coords.ndim == 2 and coords.shape[0] >= 2 and coords.shape[1] == 2:
      seconds = (times - times[0]).astype('timedelta64[s]').astype(int)
      assert coords.shape[0] == seconds.shape[0]
      all_coords.append((coords, seconds))
  return all_coords


def _filter_routes(routes, loop_gap_threshold=200, start_cluster_radius=200):
  """Selects routes that:
    - start and end in roughly the same place (forming a loop)
    - start close to the mean starting location
  """
  starts = np.array([r[0] for r,_ in routes])
  ends = np.array([r[-1] for r,_ in routes])

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
  min_coord = np.array([r.min(axis=0) for r,_ in routes]).min(axis=0)
  max_coord = np.array([r.max(axis=0) for r,_ in routes]).max(axis=0)
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
