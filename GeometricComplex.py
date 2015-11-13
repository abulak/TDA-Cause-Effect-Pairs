import numpy as np
import logging
import os
import sys
import FilteredComplex as FC

path = os.path.abspath(os.path.join(os.getcwd(), os.pardir, os.pardir))
sys.path.append(os.path.join(path, "Dionysus-python3/build/bindings/python"))


class GeometricComplex:
    """Class abstracting geometric complex on a given set of points in R^d.
    The points need to be standardised!
    Attributes:
    full_complex
    the_alpha
    limited_complex
    x_filtration
    x_inv_filtration
    y_filtration
    y_inv_filtration
    """
    dionysus = __import__('dionysus')  # own copy of the not thread-safe library

    def __init__(self, cleaned_data, x_range=range(0, 1), y_range=range(1, 2)):

        self.points = cleaned_data
        logging.info("Creating GeometricComplex on %d points in dimension %d",
                     self.points.shape[0], self.points.shape[1])
        self.dimension = self.points.shape[1]
        self.standardise_data()

        self.maxima = np.array([np.max(self.points[:, i])
                                for i in range(self.dimension)])
        self.minima = np.array([np.min(self.points[:, i])
                                for i in range(self.dimension)])

        self.x_range = x_range
        logging.info("Variable X range: %s", " ".join([str(i) for i in
                                                       self.x_range]))
        self.y_range = y_range
        logging.info("Variable Y range: %s", " ".join([str(i) for i in
                                                       self.y_range]))
        self.filtered_complexes = {
            "X": [],
            "Y": [],
            "X_inverted": [],
            "Y_inverted": []}
        self.annotated_simplices = []

    def do_all_filtrations(self):
        for i in self.x_range:
            logging.info("X-variable: Projecting on %d-th axis", i)
            self.filtered_complexes['X'].append(self.filtered_complex(
                axis=i, inverse=False))
            self.filtered_complexes['X_inverted'].append(self.filtered_complex(
                axis=i, inverse=True))
        for i in self.y_range:
            logging.info("Y-variable: Projecting on %d-th axis", i)
            self.filtered_complexes['Y'].append(self.filtered_complex(
                axis=i, inverse=False))
            self.filtered_complexes['Y_inverted'].append(self.filtered_complex(
                axis=i, inverse=True))

    def filtered_complex(self, axis, inverse):
        """
        Returns complex filtered along projection on axis in one of
        the directions.
        :param axis: integer: the index of axis to project onto
        :param inverse: bool: if true, the filtration is veversed
        :return: dionysus.Filtration: SORTED filtration of self.limited_complex
        """
        self.annotated_simplices.sort(key=lambda s: s.data[int(inverse)][axis])
        cmplx = self.dionysus.Filtration(self.annotated_simplices)
        return FC.SweepFilteredComplex(cmplx, axis, inverse)

    def annotate_simplex(self, simplex):
        """
        Attaches a list of all sweeps as simplex.data[0]
        Attaches a list of all inv_sweeps as simplex.data[1]
        :param simplex: dionysus.Simplex
        :return: dionysus.Simplex
        """
        sweeps = self.sweep_function(simplex, inverse=False)
        inv_sweeps = self.sweep_function(simplex, inverse=True)
        simplex.data = [sweeps, inv_sweeps]
        return simplex

    def sweep_function(self, simplex, inverse):
        """
        Given a simplex returns max value of the orthogonal projection
        on the axis.
        If inverse is set to true, it returns min value
        :param simplex: dionysus.Simplex object
        :param inverse: bool
        :return: float
        """

        # this turns out to be much (20?!) faster than list(simplex.vertices)
        vertices = simplex.vertices
        if simplex.dimension() == 0:
            simplex_real_coordinates = self.points[next(vertices)]
            if not inverse:
                return simplex_real_coordinates - self.minima
            if inverse:
                return self.maxima - simplex_real_coordinates
        elif simplex.dimension() == 1:
            simplex_real_coordinates = [self.points[next(vertices)],
                                        self.points[next(vertices)]]
        else:
            logging.warning("There shouldn't be any simplices of dim >1?!")
            vertices_list = [v for v in vertices]
            simplex_real_coordinates = self.points[vertices_list]

        if not inverse:
            return np.maximum(simplex_real_coordinates[0],
                              simplex_real_coordinates[1]) - self.minima
        if inverse:
            return self.maxima - np.minimum(simplex_real_coordinates[0],
                                            simplex_real_coordinates[1])

    def standardise_data(self):
        """Standardise self.points IN-PLACE i.e.
        mean = 0 and standard deviation = 1 in all dimensions"""
        for i in range(self.dimension):
            p = self.points[:, i]
            mean = np.mean(p)
            std = np.std(p)
            p -= mean
            if std:
                p /= std

    @staticmethod
    def get_real_edges_from_smpl(edges, points):
        """
        Computes the real edges coordinates; returns list ready to
        supply to LineCollection
        i.e. list of tuples ((begin_x,begin_y), (end_x,end_y))
        :param edges: list of edges (simplicial ones)
        :param points: list of points (real coordinates)
        :return: list of tuples of tuples
        """

        lines = []
        for edge in edges:
            begin = points[edge[0]]
            end = points[edge[1]]
            lines.append((list(begin), list(end)))
        return lines

    def get_real_edges(self, cmplx):

        edges = []
        for simplex in cmplx:
            if simplex.dimension() == 1:
                x = simplex.vertices
                edge = [next(x), next(x)]
                edges.append(edge)
        real_edges = self.get_real_edges_from_smpl(edges, self.points)
        return real_edges


class RipsGeometricComplex(GeometricComplex):

    def __init__(self, cleaned_data, x_range=range(0, 1), y_range=range(1, 2),
                 full_initialisation=True):
        GeometricComplex.__init__(self, cleaned_data,
                                  x_range=x_range, y_range=y_range)
        radius = np.sqrt(self.dimension)
        self.full_complex = self.create_full_complex(radius)
        self.cutoff = self.compute_the_last_death(radius)
        self.limited_simplices = [s for s in self.full_complex
                                  if s.data <= self.cutoff]
        logging.info("The threshold %f limits the Rips complex size to "
                     "%d", self.cutoff, len(self.limited_simplices))
        self.annotated_simplices = [self.annotate_simplex(s)
                                    for s in self.limited_simplices]
        logging.info("Annotating simplices done.")

        if full_initialisation:
            self.do_all_filtrations()

    def create_full_complex(self, radius):
        """
        Creates the SORTED full neighbouring graph of the Vietoris-Rips complex.
        Note that VR complex may quickly become huge for dense datasets.
        We restrict to 1-skeleton (ie. points & edges) and build edges of
        length at most radius.
        :param radius: float
        :return: return dionysus.Filtration object
        """

        logging.info("Using Rips-complex with radius %f. This may be slow "
                     "for dense sets!", radius)
        full_complex = self.rips_graph(radius)
        logging.info("Created full Rips complex of size %d",
                     full_complex.__len__())
        return full_complex

    def rips_graph(self, radius):
        """
        Bruteforce creation of neighbouring graph
        :param radius: float
        :return: dionysus.Filtration: weighted filtration of neighbouring graph
        """
        from scipy.spatial.distance import cdist
        distances = cdist(self.points, self.points)
        simplices = []
        for i in range(self.points.shape[0]):
            simplices.append(self.dionysus.Simplex([i], 0))
            for j in range(i+1, self.points.shape[0]):
                d = float(distances[i][j])
                if j != i and d < radius:
                    simplices.append(self.dionysus.Simplex([i, j], d))
        simplices.sort(key=lambda x: x.data)
        full_complex = self.dionysus.Filtration(simplices)
        return full_complex

    def compute_the_last_death(self, radius):
        """
        finds the minimal filtration s.t. self.full_complex is connected.
        It builds FilteredComplex object on dion_complex, computes its
        0-th homology and finds the last death (i.e. the minimal radius
        parameter that will make the VR-radius complex connected).

        If the full_complex is not connected it doubles the radius, creates
        self.full_complex again and repeats.

        :param radius: float:   radius that was used to build full_complex in
                                the first place
        :return: float:         the minimal radius that we need for building
                                limited_complex
        """

        connected = False
        while not connected:
            z = FC.FilteredComplex(self.full_complex)
            homology_0 = z.compute_homology()
            if len(homology_0["undying"]) > 1:
                logging.info("The complex seems to be disconected, doubling "
                             "the threshold")
                radius *= 2
                self.full_complex = self.create_full_complex(radius)
            else:
                connected = True
        deaths = [x[1] for x in homology_0["dying"]]
        return max(deaths)


class AlphaGeometricComplex(GeometricComplex):

    def __init__(self, cleaned_data, x_range=range(0, 1), y_range=range(1, 2),
                 full_initialisation=True):
        GeometricComplex.__init__(self, cleaned_data,
                                  x_range=x_range, y_range=y_range)

        self.full_complex = self.create_full_complex()

        self.cutoff = self.compute_the_last_death()
        self.limited_simplices = [s for s in self.full_complex
                                  if s.data[0] <= self.cutoff and
                                  s.dimension() < 2]
        logging.info("The threshold %f limits the Alpha complex size to "
                     "%d", self.cutoff, len(self.limited_simplices))
        self.annotated_simplices = [self.annotate_simplex(s)
                                    for s in self.limited_simplices]
        logging.info("Annotating simplices done.")

        if full_initialisation:
            self.do_all_filtrations()

    def create_full_complex(self):
        """
        Creates the SORTED alpha complex (i.e. dionysus object) on
        self.points
        """
        full_complex = self.dionysus.Filtration()
        self.dionysus.fill_alpha_complex(self.points.tolist(),
                                         full_complex)
        one_skeleton = [smpl for smpl in full_complex if smpl.dimension() <= 1]
        full_complex = self.dionysus.Filtration(one_skeleton)
        logging.info("Created full Alpha complex of size %d",
                     len(full_complex))
        full_complex.sort(self.dionysus.data_dim_cmp)
        return full_complex

    def compute_the_last_death(self):
        """finds the minimal filtration s.t. the full_complex is connected
        It assumes that the complex will is CONNECTED (as alpha complexes are).
        Faster than Rips version
        """
        full_persistence = self.dionysus.StaticPersistence(self.full_complex)
        full_persistence.pair_simplices()
        smap = full_persistence.make_simplex_map(self.full_complex)

        deaths = [smap[i.pair()].data[0] for i in full_persistence
                  if smap[i].dimension() == 0]
        return max(deaths)
