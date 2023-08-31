# QuadrilateralFitter
<img alt="QuadrilateralFitter Logo" title="QuadrilateralFitter" src="https://raw.githubusercontent.com/Eric-Canas/quadrilateral-fitter/main/resources/logo.png" width="20%" align="left"> QuadrilateralFitter is an efficient and easy-to-use library for fitting irregular quadrilaterals from polygons or point clouds.

**QuadrilateralFitter** helps you find that four corners polygon that **best approximates** your noisy data or detection, so you can apply further processing steps like: _perspective correction_ or _pattern matching_, without worrying about noise or non-expected vertex.

Optimal **Fitted Quadrilateral** is the smallest area quadrilateral that contains all the points inside a given polygon.

## Installation

To install QuadrilateralFitter, you can use pip:

```bash
pip install quadrilateral-fitter
```

## Usage

There is only one line you need to use QuadrilateralFitter:

```python
from quadrilateral_fitter import QuadrilateralFitter

# Fit an input polygon of N sides
fitted_quadrilateral = QuadrilateralFitter(polygon=your_noisy_polygon).fit()
```