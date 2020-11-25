#!/usr/bin/env python3
"""Creates an HTML+JS-based map of given GPS routes."""
import fileinput
import folium
import numpy as np
import sys
from argparse import ArgumentParser
from matplotlib.colors import rgb2hex
from matplotlib.cm import ScalarMappable

from route_plotter import Route, filter_routes

TILES = [
    "OpenStreetMap",
    "Stamen Terrain",
    "Stamen Toner",
]
METADATA_FNS = {
    'start-time': lambda route: route.start_time,
    'duration': lambda route: route.seconds[-1],
    'height': lambda route: np.nanmax(route.elevations),
    'none': lambda route: 0,
}
GRAY_CSS = r'.leaflet-tile{filter:grayscale(1);-webkit-filter:grayscale(1);}'


def main():
  args = parse_args()
  routes = [Route.parse(f, read_elevation=True) for f in args.route]
  if not routes:
    print('No valid route data provided.', file=sys.stderr)
    return

  routes = filter_routes(routes, start_cluster_radius=args.max_start_dist)
  if not routes:
    print('No routes selected.', file=sys.stderr)
    return

  loc = np.mean([r.start_coord() for r in routes], axis=0).tolist()
  run_map = folium.Map(location=loc, tiles=args.tileset, zoom_start=14)

  coords = [r.coords for r in routes]
  if args.color_by == 'none':
    lines = folium.PolyLine(coords, color='red', opacity=0.6)
    run_map.add_child(lines)
  else:
    meta_fn = METADATA_FNS[args.color_by]
    meta = np.array([meta_fn(route) for route in routes])
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
    if dont_add_css or line != '</style>':
      print(line)
    else:
      print(GRAY_CSS, line)
      dont_add_css = True
  print("Results written to:", args.output)


def parse_args():
  ap = ArgumentParser(description=__doc__)
  ap.add_argument('--tileset', choices=TILES, default=TILES[0],
                  help='Map tileset. [%(default)s]')
  ap.add_argument('--line-colormap', default='YlOrRd',
                  help='Color mapping for routes. [%(default)s]')
  ap.add_argument('--color-tiles', action='store_true',
                  help='If passed, use full-color map tiles.')
  ap.add_argument('--max-start-dist', type=float, default=200,
                  help='Maximum distance from route start to the '
                       'mean starting location, in meters.')
  ap.add_argument('--color-by', choices=METADATA_FNS, default='start-time',
                  help='Data to color by. [%(default)s]')
  ap.add_argument('-o', '--output', default='map.html',
                  help='Name of the output html file.')
  ap.add_argument('route', type=open, nargs='+')
  return ap.parse_args()


def color_mapping(arr, cmap):
  sm = ScalarMappable(cmap=cmap)
  sm.set_array(arr)
  sm.autoscale()
  return map(rgb2hex, sm.to_rgba(arr))


if __name__ == '__main__':
  main()
