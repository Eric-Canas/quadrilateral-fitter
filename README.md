# QuadrilateralFitter
<img alt="QuadrilateralFitter Logo" title="QuadrilateralFitter" src="https://raw.githubusercontent.com/Eric-Canas/quadrilateral-fitter/main/resources/logo.png" width="20%" align="left"> QuadrilateralFitter is an efficient and easy-to-use library for fitting irregular quadrilaterals from polygons or point clouds.

There are a lot of applications, where you have noisy data, like _segmentation masks_ or _sensor inputs_, that you know should be a quadrilateral in a perfect scenario. **QuadrilateralFitter** helps you find that four corners polygon that best approximates your data, so you can apply further processing steps like: _perspective correction_ or _pattern matching_, without worrying about noise or non-expected vertex.

Optimal **Fitted Quadrilateral** is the smallest area quadrilateral that contains all the points inside the input polygon.