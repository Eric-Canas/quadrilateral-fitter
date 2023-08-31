# QuadrilateralFitter
<img alt="QuadrilateralFitter Logo" title="QuadrilateralFitter" src="https://raw.githubusercontent.com/Eric-Canas/quadrilateral-fitter/main/resources/logo.png" width="20%" align="left"> **QuadrilateralFitter** is an efficient and easy-to-use library for fitting irregular quadrilaterals from polygons or point clouds.

**QuadrilateralFitter** helps you find that four corners polygon that **best approximates** your noisy data or detection, so you can apply further processing steps like: _perspective correction_ or _pattern matching_, without worrying about noise or non-expected vertex.

Optimal **Fitted Quadrilateral** is the smallest area quadrilateral that contains all the points inside a given polygon.

## Installation

You can install **QuadrilateralFitter** with pip:

```bash
pip install quadrilateral-fitter
```

## Usage

There is only one line you need to use **QuadrilateralFitter**:

```python
from quadrilateral_fitter import QuadrilateralFitter

# Fit an input polygon of N sides
fitted_quadrilateral = QuadrilateralFitter(polygon=your_noisy_polygon).fit()
```

#### Example

Let's simulate a real case scenario where we detect a noisy polygon from a form that we know should be a perfect rectangle (only deformed by perspective).

```python
import numpy as np
import cv2

image = cv2.cvtColor(cv2.imread('./resources/input_sample.jpg'), cv2.COLOR_BGR2RGB)   

# Save the Ground Truth corners
true_corners = np.array([[50., 100.], [370., 0.], [421., 550.], [0., 614.], [50., 100.]], dtype=np.float32)

# Generate a simulated noisy detection
sides = [np.linspace([x1, y1], [x2, y2], 20) + np.random.normal(scale=10, size=(20, 2))
         for (x1, y1), (x2, y2) in zip(true_corners[:-1], true_corners[1:])]
noisy_corners = np.concatenate(sides, axis=0)

# To simplify, we will clip the corners to be within the image
noisy_corners[:, 0] = np.clip(noisy_corners[:, 0], a_min=0., a_max=image.shape[1])
noisy_corners[:, 1] = np.clip(noisy_corners[:, 1], a_min=0., a_max=image.shape[0])
```

<img alt="Input Sample" title="Input Sample" src="https://raw.githubusercontent.com/Eric-Canas/quadrilateral-fitter/main/resources/input_noisy_detection.png" width="50%" align="center">


And now, let's run **QuadrilateralFitter** to find the quadrilateral that best approximates our noisy detection (without leaving points outside).

```python
from quadrilateral_fitter import QuadrilateralFitter

# Define the fitter (we want to keep it for reading internal variables later)
fitter = QuadrilateralFitter(polygon=noisy_corners)

# Get the fitted quadrilateral that contains all the points inside the input polygon
fitted_quadrilateral = np.array(fitter.fit(), dtype=np.float32)
# If you wanna to get a tighter mask, less likely to contain points outside the real quadrilateral, 
# but that cannot ensure to always contain all the points within the input polygon, you can use:
tight_quadrilateral = np.array(fitter.tight_quadrilateral, dtype=np.float32)

# To show the plot of the fitting process
fitter.plot()
```

<div align="center">
  <img alt="Fitting Process" title="Fitting Process" src="https://raw.githubusercontent.com/Eric-Canas/quadrilateral-fitter/main/resources/fitting_process.png" height="720px">
  <img alt="Fitted Quadrilateral" title="Fitted Quadrilateral" src="https://raw.githubusercontent.com/Eric-Canas/quadrilateral-fitter/main/resources/fitted_quadrilateral.png" height="720px">
</div>



