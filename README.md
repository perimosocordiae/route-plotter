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

## Animation options

`route_animation.py` is very configurable.
If you don't like how something looks, try changing its options:

### Route Filtering

 * `--max-loop-gap N` excludes routes that don't end within N meters of where they started. If you don't care about plotting loops, you can set this to infinity: `--max-loop-gap inf`.

 * `--max-start-dist N` excludes routes that start more than N meters from the average starting location. I wouldn't recommend setting this to infinity, in case you accidentally include routes that are several hundred miles apart.

### Video Options

 * `--num-frames N` controls how many frames the video will contain.
 * `--fps N` controls the framerate of the video, in frames/second.
