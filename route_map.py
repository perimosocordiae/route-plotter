#!/usr/bin/env python
"""Creates an HTML+JS-based map of given GPS routes."""
from __future__ import print_function
import fileinput
import folium
import numpy as np
import sys
from argparse import ArgumentParser
from matplotlib.colors import rgb2hex
from matplotlib.cm import ScalarMappable

from route_plotter import parse_route

TILES = [
    "OpenStreetMap",
    "Stamen Terrain",
    "Stamen Toner",
]
METADATA_FNS = {
    'start-time': lambda times, elevations: times[0],
    'duration': lambda times, elevations: times[-1] - times[0],
    'height': lambda times, elevations: elevations.max(),
    'none': lambda times, elevations: 0,
}
GRAY_CSS = r'.leaflet-tile{filter:grayscale(1);-webkit-filter:grayscale(1);}'


def main():
  args = parse_args()

  coords, meta = aggregate_route_data(args.route, METADATA_FNS[args.color_by])
  loc, mask = calc_center(coords, args.exclude_stdv)
  coords, meta = coords[mask], meta[mask]

  run_map = folium.Map(location=loc, tiles=args.tileset, zoom_start=14)

  if args.color_by == 'none':
    lines = folium.MultiPolyLine(coords, color='red', opacity=0.2)
    run_map.add_children(lines)
  else:
    # TODO: make the colors/labels work for simple numeric metas
    hex_colors = color_mapping(meta.astype(int), args.line_colormap)
    color_labels = meta.astype(str)
    for traj, c, label in zip(coords, hex_colors, color_labels):
      line = folium.PolyLine(traj, color=c, popup=label)
      run_map.add_child(line)

  # save the map to the output html file
  run_map.save(args.output, close_file=True)

  # reduce file size and optionally add a css tweak to make the map grayscale
  dont_add_css = bool(args.color_tiles)
  for line in fileinput.input(args.output, inplace=True):
    line = line.strip()
    if not line:
      continue
    if dont_add_css or line != '<style>':
      print(line)
    else:
      print(line, GRAY_CSS)
      dont_add_css = True
  print("Results written to:", args.output)


def parse_args():
  ap = ArgumentParser(description=__doc__, version='0.1')
  ap.add_argument('--tileset', choices=TILES, default=TILES[0],
                  help='Map tileset. [%(default)s]')
  ap.add_argument('--line-colormap', default='YlOrRd',
                  help='Color mapping for routes. [%(default)s]')
  ap.add_argument('--color-tiles', action='store_true',
                  help='If passed, use full-color map tiles.')
  ap.add_argument('--exclude-stdv', default=3, type=float,
                  help='Exclude paths with a center > N standard deviations '
                       'from the overall center.')
  ap.add_argument('--color-by', choices=METADATA_FNS, default='start-time',
                  help='Data to color by. [%(default)s]')
  ap.add_argument('-o', '--output', default='map.html',
                  help='Name of the output html file.')
  ap.add_argument('route', type=open, nargs='+')
  return ap.parse_args()


def aggregate_route_data(file_paths, metadata_fn):
  all_coords = []
  metadata = []
  for fh in file_paths:
    coords, times, elevations = parse_route(fh, return_time=True,
                                            return_elevation=True)
    if coords.shape[0] < 2 or coords.shape[1] != 2:
      print('No coordinates in route file:', fh.name, file=sys.stderr)
    else:
      all_coords.append(coords)
      metadata.append(metadata_fn(times, elevations))
  return np.array(all_coords, dtype=object), np.array(metadata)


def calc_center(all_coords, exclude_stdv):
  centers = np.array([path.mean(axis=0) for path in all_coords])
  center_mean = np.mean(centers, axis=0)
  center_stdv = np.std(centers, axis=0)
  stdv_from_center = (np.abs(centers - center_mean) / center_stdv).max(axis=1)
  mask = stdv_from_center <= exclude_stdv
  center = np.mean(centers[mask], axis=0)
  return tuple(center), mask


def color_mapping(arr, cmap):
  sm = ScalarMappable(cmap=cmap)
  sm.set_array(arr)
  sm.autoscale()
  return map(rgb2hex, sm.to_rgba(arr))


if __name__ == '__main__':
  main()
