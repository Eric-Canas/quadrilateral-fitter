from quadrilateral_fitter import QuadrilateralFitter
import numpy as np
import cv2
from matplotlib import pyplot as plt

def yugioh_test():
    image = cv2.cvtColor(cv2.imread('./resources/input_sample.jpg'), cv2.COLOR_BGR2RGB)
    true_corners = np.array([[50., 100.], [370., 0.], [421., 550.], [0., 614.], [50., 100.]], dtype=np.float32)

    # Generate the noisy corners
    sides = [np.linspace([x1, y1], [x2, y2], 50) + np.random.normal(scale=5, size=(50, 2))
             for (x1, y1), (x2, y2) in zip(true_corners[:-1], true_corners[1:])]
    noisy_corners = np.concatenate(sides, axis=0)

    # To simplify, we will clip the corners to be within the image
    noisy_corners[:, 0] = np.clip(noisy_corners[:, 0], a_min=0., a_max=image.shape[1])
    noisy_corners[:, 1] = np.clip(noisy_corners[:, 1], a_min=0., a_max=image.shape[0])

    fig, ax = plt.subplots()
    ax.imshow(image), ax.axis('off')

    # Draw the true corners
    ax.plot(true_corners[:, 0], true_corners[:, 1], linestyle='--', marker='o', alpha=0.9, label='True Polygon')
    # Draw the noisy corners
    ax.plot(noisy_corners[:, 0], noisy_corners[:, 1], linestyle='-', alpha=0.6, label='Noisy Detection')

    ax.legend(fontsize='small')
    plt.show()
    #plt.savefig('./resources/yugioh_noisy_detection2.png', dpi=600, bbox_inches='tight')
    # Fit an input polygon of N sides
    fitter = QuadrilateralFitter(polygon=noisy_corners)
    fitted_quadrilateral = fitter.fit()
    fitter.plot()

    # Draw the fitted quadrilateral on the image
    fig, ax = plt.subplots()
    ax.imshow(image), ax.axis('off')
    # Draw the input
    ax.plot(noisy_corners[:, 0], noisy_corners[:, 1], linestyle='-', alpha=0.7, label='Input Detection')
    # Draw the fitted quadrilateral
    fitted_quadrilateral = np.array(fitted_quadrilateral+(fitted_quadrilateral[0],), dtype=np.float32)

    ax.plot(fitted_quadrilateral[:, 0], fitted_quadrilateral[:, 1], linestyle='--', marker='o', alpha=0.7,
            label='Fitted Quadrilateral')
    ax.legend(fontsize='small')
    #plt.savefig('./resources/yugioh_fitted_quadrilateral4.png', dpi=600, bbox_inches='tight')
    plt.show()



if __name__ == '__main__':


    yugioh_test()
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

    for data in test_data:
        fitter = QuadrilateralFitter(data)
        fitter.fit()
        fitter.plot()