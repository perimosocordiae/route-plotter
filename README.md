# route-plotter

Visualizations for running/walking/cycling route data.

## Usage

```sh
git clone https://github.com/perimosocordiae/route-plotter.git
cd route-plotter
python setup.py install

# create an animation
route_animation.py ~/path/to/route_data/*
route_animation.py ~/path/to/route_data/* --save anim.mp4

# generate an interactive map
route_map.py ~/path/to/route_data/* --output map.html
open map.html
```
