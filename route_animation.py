#!/usr/bin/env python3
"""Create a matplotlib animation from route data."""
import glob
import numpy as np
import os
import sys
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.collections import LineCollection

# workaround for older matplotlib versions
try:
  from matplotlib.colors import to_rgb
except ImportError:
  import matplotlib.colors
  to_rgb = matplotlib.colors.colorConverter.to_rgb

# appdirs is an optional dependency
try:
  from appdirs import user_cache_dir
except ImportError:
  def user_cache_dir(appname=None):
    return os.path.join(os.getcwd(), 'tiles')

from route_plotter import BBox, Route, filter_routes, stitch_tiles


def main():
  args = parse_args()
  routes = [Route.parse(f) for f in args.route]
  if not routes:
    print('No valid route data provided.', file=sys.stderr)
    return
  print('Parsed', len(routes), 'valid routes.')

  routes = filter_routes(routes, loop_gap_threshold=args.max_loop_gap,
                         start_cluster_radius=args.max_start_dist)
  assert bool(routes), '_filter_routes removed all routes'
  print('Filtered to', len(routes), 'routes.')

  frame_data = _prepare_frame_data(routes, num_frames=args.num_frames,
                                   fade_secs=args.head_fade)
  print('Split into', len(frame_data[0]), 'segments.')

  if not os.path.exists(args.tile_cache):
    os.makedirs(args.tile_cache)
  bg_img, bg_extent = stitch_tiles(BBox.from_routes(routes), zoom=args.zoom,
                                   cachedir=args.tile_cache,
                                   flatten=(not args.color_tiles),
                                   margin_scale=args.max_margin)
  print('Assembled %d x %d background map image.' % bg_img.shape[:2])

  fig, ax = _setup_figure(bg_img, bg_extent, scale=args.scale)
  anim = _animate(fig, ax, frame_data, line_width=args.line_width, fps=args.fps,
                  tail_color=args.tail_color, head_color=args.head_color,
                  tail_alpha=args.tail_alpha, head_alpha=args.head_alpha)
  print('Prepared animation loop.')

  if args.save:
    print('Saving animation to', args.save)
    dpi = plt.rcParams.get('figure.dpi', 100.0)
    anim.save(args.save, dpi=dpi, bitrate=args.bitrate,
              savefig_kwargs=dict(pad_inches=0))
  else:
    plt.show()


def _prepare_frame_data(routes, num_frames=500, fade_secs=90):
  max_duration = max(r.seconds[-1] for r in routes)
  frame_times, dt = np.linspace(0, max_duration, num_frames, retstep=True)

  # convert fade_secs to frame units
  fade_frames = fade_secs / dt

  # generate timestamps
  mins, secs = divmod(frame_times, 60)
  hours, mins = divmod(mins, 60)
  timestamps = ['%d:%02d:%02d' % tt for tt in zip(hours, mins, secs)]

  # break up routes into per-frame segments
  frame_idxs = np.column_stack([np.searchsorted(r.seconds, frame_times)
                                for r in routes])
  segments = []
  segment_frames = []
  start_idxs = np.zeros(len(routes), dtype=int)
  for frame, stop_idxs in enumerate(frame_idxs):
    updated, = np.where(start_idxs < stop_idxs)
    for i in updated:
      r = routes[i].coords
      segments.append(r[start_idxs[i]:stop_idxs[i], ::-1])
      segment_frames.append(frame)
    start_idxs[updated] = stop_idxs[updated] - 1
  segment_frames = np.array(segment_frames)

  return segments, segment_frames, timestamps, fade_frames


def _animate(fig, ax, frame_data, line_width=2, fps=10, tail_color='r',
             tail_alpha=0.75, head_color='b', head_alpha=1):
  segments, segment_frames, timestamps, fade_frames = frame_data
  head_color = np.array(to_rgb(head_color))[None]
  tail_color = np.array(to_rgb(tail_color))[None]

  # start with all segments transparent
  segment_colors = np.zeros((len(segments), 4), dtype=float)
  lines = LineCollection(segments, linewidths=line_width, colors=segment_colors)
  ax.add_collection(lines)

  timer = ax.text(1, 0, '0:00:00', transform=ax.transAxes, zorder=3,
                  verticalalignment='bottom', horizontalalignment='right',
                  bbox=dict(facecolor='white'))

  def update_frame(frame_idx):
    frame_diff = frame_idx - segment_frames
    mask = frame_diff < 0
    # compute head alpha
    alpha1 = 1 - np.minimum(frame_diff/fade_frames, 1)
    alpha1[mask] = 0
    alpha1 *= head_alpha
    # compute tail alpha
    alpha2 = (1 - mask) * tail_alpha
    # composite alpha and colors
    color, alpha = _blend_alpha(head_color, alpha1, tail_color, alpha2)
    segment_colors[:, :3] = color
    segment_colors[:, 3] = alpha

    lines.set_color(segment_colors)
    timer.set_text(timestamps[frame_idx])
    return lines, timer

  interval = 1000. / fps
  return FuncAnimation(fig, update_frame, frames=len(timestamps), blit=True,
                       interval=interval, repeat=True)


def _blend_alpha(c1, a1, c2, a2):
  # for details, see: https://en.wikipedia.org/wiki/Alpha_compositing
  a21 = a2 * (1 - a1)
  a = a1 + a21
  c = c1 * a1[:,None] + c2 * a21[:,None]
  denom = np.where(a == 0, 1, a)[:, None]
  c /= denom
  return c, a


def parse_args():
  cache_dir = user_cache_dir(appname='route-plotter')
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
  ap.add_argument('--fps', type=float, default=10, help='Frames per second.')
  ap.add_argument('--bitrate', type=int, help='Bitrate when saving animation.')
  ap.add_argument('--line-width', type=float, default=2.5, help='Line width.')
  ap.add_argument('--tail-color', type=str, default='blue',
                  help='Color of trailing line.')
  ap.add_argument('--tail-alpha', type=float, default=0.2,
                  help='Opacity of trailing line.')
  ap.add_argument('--head-color', type=str, default='red', help='Head color.')
  ap.add_argument('--head-alpha', type=float, default=1, help='Head opacity.')
  ap.add_argument('--head-fade', type=float, default=90,
                  help=('Number of seconds (in route time) over which the head'
                        ' of the line fades out.'))
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
  ax.imshow(bg_img, zorder=0, extent=bg_extent, cmap='Greys_r', aspect='auto')
  ax.autoscale(False)
  ax.margins(0, 0)
  return fig, ax


if __name__ == '__main__':
  main()
