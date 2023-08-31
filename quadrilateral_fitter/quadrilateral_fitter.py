from __future__ import annotations

from shapely.geometry import Polygon, mapping
from itertools import combinations


from quadrilateral_fitter import _Line  # Assuming you'll also rename the module

class QuadrilateralFitter:
    def __init__(self, polygon: 'np.ndarray' | tuple | list):
        """
        Constructor for initializing the QuadrilateralFitter object.

        :param polygon: np.ndarray. A NumPy array of shape (N, 2) representing the input polygon,
                              where N is the number of vertices.
        """
        assert polygon.shape[1] == 2, "Input polygon should have a shape of (N, 2)"
        self._polygon = polygon
        self.convex_hull_polygon = Polygon(polygon).convex_hull

        self._initial_guess = None
        self.fitted_quadrilateral = None

    @property
    def tight_quadrilateral(self):
        return self._initial_guess

    def fit(self) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float], tuple[float, float]]:
        """
        Fits an irregular quadrilateral around the input polygon. The quadrilateral is optimized to minimize
        the Intersection over Union (IoU) with the input polygon.

        This method performs the following steps:
        1. Computes the convex hull of the input polygon.
        2. Finds an initial quadrilateral that closely approximates the convex hull.
        3. Refines this initial quadrilateral to ensure it fully circumscribes the convex hull.

        Note: The input polygon should be of shape (N, 2), where N is the number of vertices.

        :return: A tuple containing four tuples, each of which has two float elements representing the (x, y)
                coordinates of the quadrilateral's vertices. The vertices are order clockwise.

        :raises AssertionError: If the input polygon does not have a shape of (N, 2).
        """
        self._initial_guess = self.__find_initial_quadrilateral()
        self.fitted_quadrilateral = self.__expand_quadrilateral(self._initial_guess)
        return self.fitted_quadrilateral


    def __find_initial_quadrilateral(self, simplify_polygons_larger_than: int | None = 10) -> Polygon:
        """
        Internal method to find the initial approximating quadrilateral based on the vertices of the Convex Hull.
        To find the initial quadrilateral, we iterate through all 4-vertex combinations of the Convex Hull vertices
        and find the one with the highest Intersection over Union (IoU) with the Convex Hull. It will ensure that
        it is the best possible quadrilateral approximation to the input polygon.
        :param simplify_polygons_larger_than: int|None. If a number is specified, the method will make a
                        preliminar Douglas-Peucker simplification of the Convex Hull if it has more than
                        simplify_polygons_larger_than vertices. This will speed up the process, but may
                        lead to a sub-optimal quadrilateral approximation.

        :return: Polygon. A Shapely Polygon object representing the initial quadrilateral approximation.
        """
        best_iou, best_quadrilateral = 0., None  # Variable to store the vertices of the best quadrilateral
        convex_hull_area = self.convex_hull_polygon.area

        # Simplify the Convex Hull if it has more than simplify_polygons_larger_than vertices
        simplified_polygon = self.__simplify_polygon(polygon=self.convex_hull_polygon,
                                                     max_sides=simplify_polygons_larger_than)

        # Iterate through all 4-vertex combinations to form potential quadrilaterals
        for vertices_combination in combinations(mapping(simplified_polygon)['coordinates'][0], 4):
            current_quadrilateral = Polygon(vertices_combination)
            assert current_quadrilateral.is_valid, f"Quadrilaterals generated from an ordered Convex Hull should be " \
                                                       f"always valid."

            # Calculate the Intersection over Union (IoU) between the Convex Hull and the current quadrilateral
            iou = self.__iou(polygon1=self.convex_hull_polygon, polygon2=current_quadrilateral,
                             precomputed_polygon_1_area=convex_hull_area)

            if iou > best_iou:
                best_iou, best_quadrilateral = iou, current_quadrilateral
                if iou >= 1.:
                    assert iou == 1., f"IoU should never be > 1.0. Got{iou}"
                    break  # We found the best possible quadrilateral, so we can stop iterating

        assert best_quadrilateral is not None, "No quadrilateral was found. This should never happen."

        return best_quadrilateral


    def __expand_quadrilateral(self, quadrilateral: Polygon) -> Polygon:
        """
        Internal method that expands the initial quadrilateral approximation to make sure it contains all the vertices
        of the input polygon Convex Hull.
        Method:
            1. Calculate the line equation for each side of the quadrilateral
            2. Move each line in their orthogonal direction (outwards) until it contains (or intersects)
               all the points of the Convex Hull in its inward direction
            3. Find the intersection points between the lines to calculate the vertices of the
               new expanded quadrilateral

        :param quadrilateral: Polygon. A Shapely Polygon object representing the initial quadrilateral approximation.

        :return: Polygon. A Shapely Polygon object representing the expanded quadrilateral.
        """
        # 1. Calculate the line equation for each side of the quadrilateral
        line_equations = self.__polygon_vertices_to_line_equations(polygon=quadrilateral)
        # 2. Move each line in their orthogonal direction (outwards) until it contains (or intersects)
        #    all the points of the Convex Hull in its inward direction
        for line in line_equations:
            self.__move_line_to_contain_all_points(line=line, polygon=quadrilateral)
        # 3. Find the intersection points between the lines to calculate the vertices of the
        #    new expanded quadrilateral
        new_quadrilateral_vertices = self.__find_polygon_vertices_from_lines(line_equations=line_equations)
        return new_quadrilateral_vertices

    def __find_polygon_vertices_from_lines(self, line_equations: tuple[_Line]) -> tuple[tuple[float, float], ...]:
        """
        Internal method to calculate the vertices of a polygon from a tuple of line equations.

        :param line_equations: tuple[_Line]. A tuple of _Line objects representing the sides of the polygon.
        :return: tuple[tuple[float, float], ...]. A tuple of tuples representing the vertices of the polygon.
        """
        # Find the intersection between each line and its next one
        return tuple(line1.get_intersection(other_line=line_equations[(i + 1) % len(line_equations)])
                     for i, line1 in enumerate(line_equations))

    @staticmethod
    def __polygon_vertices_to_line_equations(polygon: Polygon) -> tuple[_Line]:
        """
        Internal static method to convert Polygons to a tuple of line equations.

        :param polygon: Polygon. A Shapely Polygon object.
        :return: tuple[_Line]. A tuple of _Line objects representing the sides of the polygon.
        """
        assert isinstance(polygon, Polygon), f"Expected a Shapely Polygon, got {type(polygon)} instead."
        coords = polygon.exterior.coords
        # Loop through each pair of vertices to calculate the line equations (Last coord is same as first (Shapely))
        return tuple(_Line(x1=x1, y1=y1, x2=x2, y2=y2)
                     for (x1, y1), (x2, y2) in zip(coords[:-1], coords[1:]))

    def __move_line_to_contain_all_points(self, line: _Line, polygon: Polygon) -> bool:
        """
        Internal method to move a line until it contains all points in the Convex Hull.

        :param line: _Line. The line to be moved.
        :param polygon: Polygon. The polygon to be contained by the line after moving it.

        :return: bool. True if the line was moved, False otherwise.
        """
        centroid = polygon.centroid
        centroid_sign = self.__sign(x=line.point_line_position(x=centroid.x, y=centroid.y))
        assert centroid_sign != 0, "The centroid of the polygon should never be on the line."

        max_distance, best_point = 0., None

        for (x, y) in self.convex_hull_polygon.exterior.coords[:-1]:
            point_position = line.point_line_position(x=x, y=y)
            if self.__sign(x=point_position) != centroid_sign:
                distance = line.distance_from_point(x=x, y=y)
                if distance > max_distance:
                    max_distance, best_point = distance, (x, y)

        if best_point is not None:
            x, y = best_point
            line.move_line_to_intersect_point(x=x, y=y)
            return True
        return False


    # -------------------------------- HELPER METHODS -------------------------------- #

    def __simplify_polygon(self, polygon: Polygon, max_sides: int|None,
                           max_epsilon: float = 0.3, initial_epsilon: float = 0.01,
                           epsilon_increment: float = 0.005) -> Polygon:
        """
        Internal method to simplify a polygon using the Douglas-Peucker algorithm.
        :param polygon: Polygon. The polygon to simplify.
        :param max_sides: int|None. The maximum number of sides the polygon can have after simplification.
                            If None, no simplification will be performed.
        :param max_epsilon: float. The maximum tolerance value for the Douglas-Peucker algorithm.
        :param initial_epsilon: float. The initial tolerance value for the Douglas-Peucker algorithm.
        :param epsilon_increment: float. The incremental step for the tolerance value.

        :return: Polygon. The simplified polygon.
        """
        if max_sides is None:
            return polygon  # No simplification needed

        assert 0. < max_epsilon <= 1., "max_epsilon should be a float between 0 and 1"
        assert 0. < initial_epsilon <= 1., "initial_epsilon should be a float between 0 and 1"
        assert 0. < epsilon_increment <= 1., "epsilon_increment should be a float between 0 and 1"

        epsilon, increment = 0.01, 0.005  # Initial tolerance value and incremental step for tolerance
        simplified_polygon = polygon

        while len(simplified_polygon.exterior.coords) - 1 > max_sides:  # -1 because the polygon is closed
            simplified_polygon = polygon.simplify(epsilon, preserve_topology=True)
            epsilon += increment
            if epsilon > 0.3:
                break

        return simplified_polygon

    def __iou(self, polygon1: Polygon, polygon2: Polygon, precomputed_polygon_1_area: float | None = None) -> float:
        """
        Calculate the Intersection over Union (IoU) between two polygons.

        :param polygon1: Polygon. The first polygon.
        :param polygon2: Polygon. The second polygon.
        :param precomputed_polygon_1_area: float|None. The area of the first polygon. If None, it will be computed.
        :return: float. The IoU value.
        """
        if precomputed_polygon_1_area is None:
            precomputed_polygon_1_area = polygon1.area
        # Calculate the intersection and union areas
        intersection = polygon1.intersection(polygon2).area
        union = precomputed_polygon_1_area + polygon2.area - intersection
        # Return the IoU value
        return (intersection / union) if union != 0. else 0.

    @staticmethod
    def __sign(x: int | float) -> int:
        """
        Return the sign of a number.
        :param x: float. The number to check.
        :return: int. 1 if x > 0, -1 if x < 0, 0 if x == 0.
        """
        return 1 if x > 0 else (-1 if x < 0 else 0)

    def plot(self):
        """
        Plot the convex hull and the best-fitting quadrilateral for debugging purposes.
        This function imports matplotlib.pyplot locally, so the library is not required for the entire class.
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError("This function requires matplotlib to be installed. Please install it first.")

        # Plot the original polygon as a set of alpha 0.4 points
        x, y = zip(*list(self._polygon))
        plt.scatter(x, y, alpha=0.2, label='Input Polygon')

        # Plot the convex hull as a filled polygon
        x, y = self.convex_hull_polygon.exterior.xy
        plt.fill(x, y, alpha=0.4, label='Convex Hull')

        # Plot the initial quadrilateral if it exists as a semi-transparent dashed line
        if self._initial_guess is not None:
            # Calculate the IoU between the Convex Hull and the best-fitting quadrilateral
            iou = self.__iou(polygon1=self.convex_hull_polygon, polygon2=Polygon(self._initial_guess))
            x, y = self._initial_guess.exterior.xy
            plt.plot(x, y, linestyle='--', alpha=0.5, label=f'Initial Guess (IoU={iou:.3f})')

        # Plot the best quadrilateral if it exists
        if self.fitted_quadrilateral is not None:
            # Calculate the IoU between the Convex Hull and the best-fitting quadrilateral
            iou = self.__iou(polygon1=self.convex_hull_polygon, polygon2=Polygon(self.fitted_quadrilateral))
            x, y = zip(*self.fitted_quadrilateral)
            plt.plot(x + (x[0],), y + (y[0],), label=f'Fitted Quadrilateral (IoU={iou:.3f})')

            # Mark the corners of the best quadrilateral with 'X'
            plt.scatter(x, y, marker='x', color='red')

        plt.axis('equal')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title('Quadrilateral Fitting')
        plt.legend()
        plt.grid(True)
        plt.show()


