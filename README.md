# route-plotter

Visualizations for running/walking/cycling route data.

## Usage (macOS/Linux/programmers)

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

## Usage (Windows/non-programmers)

 1. [Install Anaconda](https://www.continuum.io/downloads#windows)
 2. Download and extract [the route-plotter code](https://github.com/perimosocordiae/route-plotter/archive/master.zip).
 3. Open a Command Prompt (`cmd.exe`) and navigate to the `route-plotter` folder.
 4. Type:
  * `python route_animation.py c:\Users\your_username\RouteData\*`
  * `python route_map.py c:\Users\your_username\RouteData\* --output map.html`

To get the path to your running data, you can open the folder in File Explorer,
click on the address bar, then copy the path that shows up.

## Animation options

`route_animation.py` is very configurable.
If you don't like how something looks, try changing its options:

### Route Filtering

 * `--max-loop-gap N` excludes routes that don't end within N meters of where they started. If you don't care about plotting loops, you can set this to infinity: `--max-loop-gap inf`.

 * `--max-start-dist N` excludes routes that start more than N meters from the average starting location. I wouldn't recommend setting this to infinity, in case you accidentally include routes that are several hundred miles apart.

### Video Options

 * `--num-frames N` controls how many frames the video will contain.
 * `--fps N` controls the framerate of the video, in frames/second.
