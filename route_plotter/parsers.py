import numpy as np
import os
try:
  from xml.etree import cElementTree as ElementTree
except ImportError:
  from xml.etree import ElementTree

__all__ = ['parse_route']


def parse_route(fh, kind=None, return_time=False, return_elevation=False):
  """Parse a route file.

  fh : file-like object

  kind : str, one of {'gpx', 'tcx'}
    Inferred from file extension if not given explicitly.
  """
  if kind is None:
    kind = os.path.splitext(fh.name)[1][1:]
  try:
    parse_fn = PARSERS[kind.lower()]
  except KeyError:
    raise ValueError('Unknown route file type: %s' % kind.lower())

  coords, times, elevations = [], [], []
  for latlon, time, elev in parse_fn(fh):
    coords.append(latlon)
    if return_time:
      times.append(time)
    if return_elevation:
      elevations.append(elev)

  ret = [np.array(coords, dtype=float)]
  if return_time:
    ret.append(np.array(times, dtype=np.datetime64))
  if return_elevation:
    ret.append(np.array(elevations, dtype=float))
  if len(ret) == 1:
    return ret[0]
  return tuple(ret)


def gpx_parser(fh):
  it = ElementTree.iterparse(fh, events=('start','end'))
  # look for the start gpx tag to fail fast
  for event, elem in it:
    if event == 'start' and elem.tag.endswith('}gpx'):
      break
  else:
    raise ValueError('Not a gpx file: %s' % fh.name)

  # do the main parse
  for event, elem in it:
    if event == 'end' and elem.tag.endswith('}trkpt'):
      latlon = (float(elem.attrib['lat']),
                float(elem.attrib['lon']))
      elev = np.nan
      time = None
      for child in elem:
        tag_name = child.tag.rsplit('}', 1)[1]
        if tag_name == 'ele':
          elev = float(child.text)
        elif tag_name == 'time':
          time = child.text
      yield latlon, time, elev
      elem.clear()


def tcx_parser(fh):
  it = ElementTree.iterparse(fh, events=('start','end'))
  # look for the start TrainingCenterDatabase tag to fail fast
  for event, elem in it:
    if event == 'start' and elem.tag.endswith('}TrainingCenterDatabase'):
      break
  else:
    raise ValueError('Not a tcx file: %s' % fh.name)

  # do the main parse
  for event, elem in it:
    if event == 'end' and elem.tag.endswith('}Trackpoint'):
      latlon = None
      elev = np.nan
      time = None
      for child in elem:
        tag_name = child.tag.rsplit('}', 1)[1]
        if tag_name == 'Time':
          time = child.text
        elif tag_name == 'AltitudeMeters':
          elev = float(child.text)
        elif tag_name == 'Position':
          vals = dict((c.tag.rsplit('}', 1)[1], float(c.text)) for c in child)
          latlon = (vals['LatitudeDegrees'], vals['LongitudeDegrees'])
      if latlon is not None:
        yield latlon, time, elev
      elem.clear()


PARSERS = dict(gpx=gpx_parser, tcx=tcx_parser)
