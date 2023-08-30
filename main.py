from trapezoid_fitter import TrapezoidFitter
import numpy as np

if __name__ == '__main__':

    # 1. Deformed trapezoid
    num_points = 20
    left_side = np.linspace([0.2, 0.2], [0.2, 0.8], num_points) + np.random.normal(scale=0.01, size=(num_points, 2))
    right_side = np.linspace([0.8, 0.3], [0.8, 0.7], num_points) + np.random.normal(scale=0.01, size=(num_points, 2))
    top_side = np.linspace([0.2, 0.8], [0.8, 0.7], num_points) + np.random.normal(scale=0.01, size=(num_points, 2))
    bottom_side = np.linspace([0.2, 0.2], [0.8, 0.3], num_points) + np.random.normal(scale=0.01, size=(num_points, 2))
    deformed_trapezoid = np.vstack([left_side, right_side, top_side, bottom_side])

    # 2. Perfect square
    square = np.array([[0, 0], [0, 1], [1, 1], [1, 0], [0, 0]])

    # 3. Deformed circle
    theta = np.linspace(0, 2*np.pi, 100)
    circle_x = 0.5 + 0.4 * np.cos(theta) + np.random.normal(scale=0.03, size=theta.shape)
    circle_y = 0.5 + 0.4 * np.sin(theta) + np.random.normal(scale=0.03, size=theta.shape)
    deformed_circle = np.vstack((circle_x, circle_y)).T

    # Running the tests
    test_data = [deformed_trapezoid, square, deformed_circle]
    results = []

    for data in test_data:
        fitter = TrapezoidFitter(data)
        fitter.fit()
        fitter._plot()
        results.append(fitter.best_trapezoid)