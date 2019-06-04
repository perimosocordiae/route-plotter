#!/usr/bin/env python
from setuptools import setup

setup(
    name='route_plotter',
    version='0.1.0',
    description='Visualizations for running/walking/cycling route data',
    author='CJ Carey',
    author_email='perimosocordiae@gmail.com',
    url='https://github.com/perimosocordiae/route-plotter',
    license='MIT',
    classifiers=[
        'Development Status :: 4 - Beta',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python',
    ],
    packages=['route_plotter'],
    scripts=[
        'route_animation.py',
        'route_map.py',
    ],
    install_requires=[
        'numpy',
        'matplotlib >= 1.5',
        'folium',
        'requests',
        'appdirs >= 1.4',
        'Pillow >= 2.0'
    ],
)
