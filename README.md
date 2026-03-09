# River Bend Analysis

A Python application for detecting and analyzing river meander bends from centerline data.
The project computes geometric bend metrics, visualizes their distributions, and explores bend shape patterns through clustering.

## Overview

River meanders are fundamental geomorphic features that provide insight into river dynamics and landscape evolution.
This project processes river centerline data to:

* detect inflection points
* segment the river into individual bends
* compute geometric bend metrics
* visualize feature distributions
* cluster bends based on their geometric characteristics

The application also includes a **desktop graphical interface built with Qt (PySide6)** for interactive data exploration.

## Features

* River centerline preprocessing
* Inflection point detection
* Bend segmentation
* Computation of geometric metrics:

  * bend amplitude
  * normalized amplitude
  * sinuosity
  * asymmetry ratio
  * openness
  * arc length and chord length
* Histogram visualization of bend features
* Feature threshold visualization
* Clustering of bends using machine learning
* Interactive GUI for exploring results

## Project Structure

```
river-bends-analysis
│
├── app/           # Pipeline logic and data loading
├── clustering/    # Clustering algorithms and evaluation
├── geometry/      # Bend detection and geometric metrics
├── gui/           # PySide6 graphical interface + entry point for GUI usage
├── outputs/       # Generated plots and CSV results
├── data/          # Input datasets (not included by default)
│
├── main.py        # Application entry point without gui - for scientific purposes
├── cluster_bends.py
└── README.md
```

## Installation

Clone the repository:

```
git clone https://github.com/YOUR_USERNAME/river-bends-analysis.git
cd river-bends-analysis
```

## Running the Application

Start the application with:

```
python main.py

```
or
```
python gui/main.py

```

The GUI will open and allow you to load bend metric CSV files and visualize feature distributions.

## Dependencies

Main libraries used:

* Python
* PySide6 (Qt GUI)
* NumPy
* Pandas
* Matplotlib
* scikit-learn
* GeoPandas
* Shapely

## Data

Large datasets are not included in this repository.
Users can place their input data inside the `data/` folder.
In the repository there is a .shp file for Somesul Mic river.

## Example Output

The application generates visualizations such as:

* bend centerline previews
* feature histograms
* clustering plots (K-means)

Example output is available in the `outputs/` directory.

Mathematics and geometrical algorithms belong to Limaye et. all (2025) research paper.
