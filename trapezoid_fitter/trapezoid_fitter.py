from __future__ import annotations

# Importing required libraries
from scipy.spatial import ConvexHull
from shapely.geometry import Polygon
from itertools import combinations
import numpy as np

from trapezoid_fitter import _Line

class TrapezoidFitter:
    def __init__(self, input_polygon: np.ndarray):
        """
        Constructor for initializing the TrapezoidFitter object.

        :param input_polygon: np.ndarray. A NumPy array of shape (N, 2) representing the input polygon,
                              where N is the number of vertices.
        """
        assert input_polygon.shape[1] == 2, "Input polygon should have a shape of (N, 2)"

        # Compute the Convex Hull of the input polygon and store its vertices
        convex_hull = ConvexHull(input_polygon)
        self.convex_hull_vertices = input_polygon[convex_hull.vertices]

        # Initialize variables to hold the initial and best-fitting trapezoids
        self._initial_trapezoid = None
        self.best_trapezoid = None

    def fit(self) -> np.ndarray:
        """
        Method to fit the best approximating trapezoid to the Convex Hull of the input polygon.

        :return: np.ndarray. Returns a NumPy array with shape (4, 2) representing the vertices
                 of the best-fitting trapezoid.
        """
        self._initial_trapezoid = self.__find_initial_trapezoid()
        self.best_trapezoid = self.__refine_trapezoid(self._initial_trapezoid)
        return self.best_trapezoid


    def __find_initial_trapezoid(self) -> np.ndarray:
        """
        Internal method to find the initial approximating trapezoid based on the vertices of the Convex Hull.
        1. Calculate the line equation for each side of the trapezoid
        2. Move each line in their orthogonal direction (outwards) until it contains all the points
        3. Find the intersection points between the lines to form the refined trapezoid
        :return: np.ndarray. Returns a NumPy array with shape (4, 2) representing the vertices
                 of the initial best-fitting trapezoid.
        """
        best_iou = 0  # Variable to store the best IoU
        best_trapezoid_vertices = None  # Variable to store the vertices of the best trapezoid

        # Create a Shapely Polygon object for the Convex Hull
        convex_hull_polygon = Polygon(self.convex_hull_vertices)

        # Iterate through all 4-vertex combinations to form potential trapezoids
        for vertices_combination in combinations(self.convex_hull_vertices, 4):
            current_trapezoid_polygon = Polygon(vertices_combination)

            if not current_trapezoid_polygon.is_valid:
                continue

            # Calculate the Intersection over Union (IoU) between the Convex Hull and the current trapezoid
            intersection = convex_hull_polygon.intersection(current_trapezoid_polygon).area
            union = convex_hull_polygon.area + current_trapezoid_polygon.area - intersection
            iou = intersection / union if union != 0 else 0

            if iou > best_iou:
                best_iou = iou
                best_trapezoid_vertices = vertices_combination

        return best_trapezoid_vertices


    def __refine_trapezoid(self, trapezoid_vertices: np.ndarray) -> np.ndarray:
        """
        Internal method to refine the approximating trapezoid to make it contain all the points
        in the Convex Hull of the input polygon.

        :param trapezoid_vertices: np.ndarray. A NumPy array with shape (4, 2) representing the vertices
                                   of the initial approximating trapezoid.
        :return: np.ndarray. A NumPy array with shape (4, 2) representing the vertices of the refined trapezoid.
        """
        line_equations = self.__polygon_vertices_to_line_equations(vertices=trapezoid_vertices)
        centroid_x, centroid_y = np.mean(trapezoid_vertices, axis=0)

        for line in line_equations:
            self.__move_line_to_contain_all_points(line, centroid_x, centroid_y)

        refined_trapezoid_vertices = self.__calculate_refined_vertices(line_equations)
        return refined_trapezoid_vertices

    def __move_line_to_contain_all_points(self, line: _Line, centroid_x: float, centroid_y: float):
        """
        Internal method to move a line until it contains all points in the Convex Hull.

        :param line: _Line. The line to be moved.
        :param centroid_x: float. The x-coordinate of the trapezoid's centroid.
        :param centroid_y: float. The y-coordinate of the trapezoid's centroid.
        """
        centroid_position = line.point_line_position(centroid_x, centroid_y)
        max_distance = 0
        best_point = None

        for point in self.convex_hull_vertices:
            point_x, point_y = point
            point_position = line.point_line_position(point_x, point_y)

            if np.sign(point_position) != np.sign(centroid_position):
                distance = line.distance_from_point(point_x, point_y)
                if distance > max_distance:
                    max_distance = distance
                    best_point = point

        if best_point is not None:
            line.move_line_to_intersect_point(best_point[0], best_point[1])

    def __calculate_refined_vertices(self, line_equations: list[_Line]) -> np.ndarray:
        """
        Internal method to calculate the vertices of the refined trapezoid based on the moved lines.

        :param line_equations: list[_Line]. A list of _Line objects representing the moved lines.
        :return: np.ndarray. A NumPy array with shape (4, 2) containing the vertices of the refined trapezoid.
        """
        refined_trapezoid_vertices = np.zeros((4, 2))
        for i, line1 in enumerate(line_equations):
            line2 = line_equations[(i + 1) % 4]
            x, y = line1.get_intersection(line2)
            refined_trapezoid_vertices[i] = [x, y]

        return refined_trapezoid_vertices

    @staticmethod
    def __polygon_vertices_to_line_equations(vertices: np.ndarray | tuple) -> list[_Line]:
        """
        Internal static method to convert polygon vertices to line equations.

        :param vertices: np.ndarray | tuple. The vertices of the polygon in either a NumPy array of shape (N, 2)
                         or a tuple containing the vertices.
        :return: list[_Line]. A list of _Line objects representing the sides of the polygon.
        """

        if isinstance(vertices, (tuple, list)):
            vertices = np.array(vertices)

        N, coords = vertices.shape
        assert coords == 2, f"Vertices should have a shape of (N, 2). Got {(N, coords)} instead."

        # Initialize an empty list to store the line equations
        lines = []
        # Loop through each pair of vertices to calculate the line equations
        for i in range(N):
            x1, y1 = vertices[i]
            x2, y2 = vertices[(i + 1) % N]  # The modulo operation ensures that the last point connects to the first one
            lines.append(_Line(x1=x1, y1=y1, x2=x2, y2=y2))

        return lines

    def _plot(self):
        """
        Plot the convex hull and the best-fitting trapezoid for debugging purposes.
        This function imports matplotlib.pyplot locally, so the library is not required for the entire class.
        """
        import matplotlib.pyplot as plt

        # Plot the convex hull as a filled polygon
        plt.fill(self.convex_hull_vertices[:, 0], self.convex_hull_vertices[:, 1], alpha=0.4, label='Convex Hull')

        # Plot the initial trapezoid if it exists as a semi-transparent dashed line
        if self._initial_trapezoid is not None:
            initial_trapezoid_polygon = Polygon(self._initial_trapezoid)
            plt.plot(*initial_trapezoid_polygon.exterior.xy, linestyle='--', alpha=0.5, label='Initial Approximation')

        # Plot the best trapezoid if it exists
        if self.best_trapezoid is not None:
            best_trapezoid_polygon = Polygon(self.best_trapezoid)
            plt.plot(*best_trapezoid_polygon.exterior.xy, label='Best Approximating Trapezoid')

            # Mark the corners of the best trapezoid with 'X'
            plt.scatter(self.best_trapezoid[:, 0], self.best_trapezoid[:, 1], marker='x', color='red')

        plt.axis('equal')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title('Convex Hull and Best Approximating Trapezoid')
        plt.legend()
        plt.grid(True)
        plt.show()

