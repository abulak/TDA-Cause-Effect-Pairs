import numpy as np
import numpy.ma as ma
import os
import sys
import logging

path = os.path.abspath(os.path.join(os.getcwd(), os.pardir, os.pardir,
                                    os.pardir))
sys.path.append(os.path.join(path, "Dionysus-python3/build/bindings/python"))


class FilteredComplex:

    """Class abstracting geometric/algebraic filtration of a complex;
    (fcomplex should be dionysus.Filtration() object).
    Attributes:
    h_0                          pairs of points on the diagram
    persistence_diagram_0        the diagram (dionysus object)
    h_1                          (uncomment, if needed)
    persistence_diagram_1        (uncomment, if needed)
    bottleneck_dist              distance of 0-diagram and the empty diagram

    Methods:
    compute_persistence_pairs
    create_persistence_diagrams
    distance
    """
    dionysus = __import__('dionysus')
    empty_diagram = dionysus.PersistenceDiagram(0)
    empty_diagram.append((0, 0))

    def __init__(self, fcomplex):

        logging.debug("Initialising Static Presistence")
        self.persistence = self.dionysus.StaticPersistence(fcomplex)

        logging.debug("Pairing Simplices")
        self.persistence.pair_simplices()
        self.smap = self.persistence.make_simplex_map(fcomplex)

    def distance(self, p=0):
        """Returns p-th Wasserstein distance between the filtration's 0-diagram
        and the empty diagram.
        If p=0 then the bottleneck is returned.

        TODO: higher dimensions"""

        if not hasattr(self,"persistance_diagram_0"):
            self.persistence_diagram_0 = self.create_persistence_diagrams()

        if p > 0:
            return self.dionysus.wasserstein_distance(
                self.persistence_diagram_0, self.empty_diagram, p)
        else:
            return self.dionysus.bottleneck_distance(
                self.persistence_diagram_0, self.empty_diagram)

    def compute_homology(self, dimension=0):
        """ Computes the homology persistence pairs for the filtered complex"""

        logging.debug("Computing 0-th Persistence Homology Pairs")

        homology = {"dying": [], "undying": []}
        undying = 0
        for i in self.persistence:
            if i.sign() > 0:
                birth_simplex = self.smap[i]
                birth = birth_simplex.data
                if birth_simplex.dimension() == dimension:
                    if i.unpaired():
                        death = float('inf')
                        homology["undying"].append([birth, death])
                        logging.debug("Undying simplex: %s at %f",
                                      birth_simplex, birth_simplex.data)
                        undying += 1
                        if undying > 1:
                            logging.debug("The complex seems to be "
                                            "disconnected?!")
                    else:
                        killing_simplex = self.smap[i.pair()]
                        death = killing_simplex.data
                        if death > birth:
                            homology["dying"].append([birth, death])
                        elif death < birth:
                            logging.warning("You can not die before You were "
                                            "born!")
                            logging.warning(birth_simplex, birth,
                                            killing_simplex, death)
                elif birth_simplex.dimension() > 1:
                    logging.warning("There should be no simplices of dim >1?! "
                                    "but there is: %s", birth_simplex)
        return homology

    def create_persistence_diagrams(self):
        logging.debug("Creating Diagrams")

        if self.homology["dying"]:
            persistence_diagram = self.dionysus.PersistenceDiagram(0)
            all_pairs = [tuple(x) for x in self.homology["dying"]]
            for x in all_pairs:
                persistence_diagram.append(x)
        else:
            persistence_diagram = self.empty_diagram

        return persistence_diagram


class SweepFilteredComplex(FilteredComplex):

    def __init__(self, fcomplex, axis, inverse):
        self.axis = axis
        self.inverse = int(inverse)

        FilteredComplex.__init__(self, fcomplex)

        self.homology_0 = self.compute_homology(dimension=0)

    def compute_homology(self, dimension=0):
        logging.debug("Computing 0-th Persistence Homology Pairs")

        homology = {"dying": [], "undying": []}
        undying = 0
        for i in self.persistence:
            if i.sign() > 0:
                birth_simplex = self.smap[i]
                birth = birth_simplex.data[self.inverse][self.axis]
                if birth_simplex.dimension() == dimension:
                    if i.unpaired():
                        death = float('inf')
                        homology["undying"].append([birth, death])
                        logging.debug("Undying simplex: %s at %f",
                                      birth_simplex, birth)
                        undying += 1
                        if undying > 1:
                            logging.warning("The complex seems to be "
                                            "disconnected?!")
                    else:
                        killing_simplex = self.smap[i.pair()]
                        death = killing_simplex.data[self.inverse][self.axis]
                        if death > birth:
                            homology["dying"].append([birth, death])
                        elif death < birth:
                            logging.warning("You can not die before You were "
                                            "born!")
                            logging.warning(birth_simplex, birth,
                                            killing_simplex, death)
                elif birth_simplex.dimension() > 1:
                    logging.warning("There should be no simplices of dim >1?! "
                                    "but there is: %s", birth_simplex)
        return homology


class AlphaFilteredComplex(FilteredComplex):

    def __init__(self, fcomplex):
        FilteredComplex.__init__(self, fcomplex)

        self.homology_0 = self.compute_homology(dimension=0)

    def compute_homology(self, dimension=0):
        logging.debug("Computing 0-th Persistence Homology Pairs")

        homology = {"dying": [], "undying": []}
        undying = 0
        for i in self.persistence:
            if i.sign() > 0:
                birth_simplex = self.smap[i]
                birth = birth_simplex.data[0]
                if birth_simplex.dimension() == dimension:
                    if i.unpaired():
                        death = float('inf')
                        homology["undying"].append([birth, death])
                        logging.debug("Undying simplex: %s at %f",
                                      birth_simplex, birth)
                        undying += 1
                        if undying > 1:
                            logging.warning("The complex seems to be "
                                            "disconnected?!")
                    else:
                        killing_simplex = self.smap[i.pair()]
                        death = killing_simplex.data[0]
                        if death > birth:
                            homology["dying"].append([birth, death])
                        elif death < birth:
                            logging.warning("You can not die before You were "
                                            "born!")
                            logging.warning(birth_simplex, birth,
                                            killing_simplex, death)
                elif birth_simplex.dimension() > 1:
                    logging.warning("There should be no simplices of dim >1?! "
                                    "but there is: %s", birth_simplex)
        return homology


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
        self.filtrations = {}
        #         "X": [],
        #         "X_inverted": [],
        #         "Y": [],
        #         "Y_inverted": []}

    def do_all_filtrations(self):
        for i in self.x_range:
            logging.info("X-variable: Projecting on %d-th axis", i)
            self.filtrations['X'] = self.filtered_complex(axis=i,
                                                          inverse=False)
            self.filtrations['X_inverted'] = self.filtered_complex(axis=i,
                                                                inverse=True)
        for i in self.y_range:
            logging.info("Y-variable: Projecting on %d-th axis", i)
            self.filtrations['Y'] = self.filtered_complex(axis=i,
                                                          inverse=False)
            self.filtrations['Y_inverted'] = self.filtered_complex(axis=i,
                                                                inverse=True)

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
        return SweepFilteredComplex(cmplx, axis, inverse)

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
        """ Given a simplex returns max value of the orthogonal projection
        on the axis.
        If inverse is set to true, it returns min value """

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
            p /= std

    def get_real_edges_from_smpl(self, edges, points):
        """Computes the real edges of a filtration; returns list ready to
        supply to LineCollection
        i.e. list of tuples ((begin_x,begin_y), (end_x,end_y))"""
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
        length  2*sqrt(3).
        This relies on the assumption that we deal with STANDARDISED DATA
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
        :return: dionysus weighted filtration of neighbouring graph
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
        """finds the minimal filtration s.t. the full_complex is connected"""
        connected = False
        while not connected:
            z = FilteredComplex(self.full_complex)
            homology = z.compute_homology()
            if len(homology["undying"]) > 1:
                logging.info("The complex seems to be disconected, doubling "
                             "the threshold")
                self.full_complex = self.create_full_complex(2 * radius)
            else:
                connected = True
        deaths = [x[1] for x in homology["dying"]]
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
        self.points"""
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
        """
        full_persistence = self.dionysus.StaticPersistence(self.full_complex)
        full_persistence.pair_simplices()
        smap = full_persistence.make_simplex_map(self.full_complex)

        deaths = [smap[i.pair()].data[0] for i in full_persistence
                  if smap[i].dimension() == 0]
        return max(deaths)


class CauseEffectPair:
    """
    Encapsulates the whole logical concept behind Cause-Effect Pair.
    I.e. contains points, the whole list of outliers, metadata of a pair, etc.
    """

    def __init__(self, model):
        self.current_dir = os.getcwd()
        self.name = self.current_dir[-8:]

        logging.basicConfig(filename=self.name+".log", level=logging.INFO,
                            format='%(asctime)s - %(levelname)s - %(message)s')

        logging.info("Starting CauseEffectPair")
        self.model = model
        pairs_dir = os.path.abspath(os.path.join(self.current_dir, os.pardir,
                                                 os.pardir, 'pairs'))
        all_pairs_metadata = np.loadtxt(os.path.join(pairs_dir, 'pairmeta.txt'))
        self.metadata = all_pairs_metadata[int(self.name[-4:]) - 1]
        # metadata is a list of the form
        # 0: pair-number,
        # 1: cause-first-coord, (all starting from one)
        # 2: cause-last-coord,
        # 3: effect-first-coord,
        # 4: effect-last-coord,
        # 5: weight
        self.x_range = range(int(self.metadata[1])-1, int(self.metadata[2]))
        self.y_range = range(int(self.metadata[3])-1, int(self.metadata[4]))

        self.prepare_points()

        self.compute_topological_summary()

    def prepare_points(self):

        std_points_file = os.path.join(self.current_dir, "std_points")
        self.std_points = np.loadtxt(std_points_file)
        self.dimension = int(self.std_points.shape[1])

        outliers_file = os.path.join(self.current_dir, "outliers_" + self.model)
        self.outliers = np.loadtxt(outliers_file).astype(np.int)

    def remove_outliers(self, i):
        """
        :param i: number outliers to remove from self.std_points
        :return: numpy array of std_points without outliers[:i]
        """
        points_masked = ma.MaskedArray(self.std_points)
        for outlier in self.outliers[:i]:
            points_masked[outlier] = ma.masked
        cleaned_points = points_masked.compressed().reshape(
                self.std_points.shape[0] - i, self.dimension)

        return cleaned_points

    def compute_topological_summary(self):
        """
        For each in self.outliers generate cleaned_points. Then construct
        GeometricComplex(cleaned_points) and compute its persistant 0-th
        homology.

        We save the 0-persistence pairs in the _list_
        self.persistence_pairs.

        persistence_pairs[outlier] contains dictionary with self-explaining
        keys:
        x_filtration_H0
        x_inv_filtration_H0
        y_filtration_H0
        y_inv_filtration_H0

        values are arrays of persistance pairs
        """

        self.persistence_pairs = []
        self.extrema = []
        for i, outlier in enumerate(self.outliers, start=1):
            logging.info("Outlier: %d of %d", i, self.outliers.shape[0])

            cleaned_points = self.remove_outliers(i)
            if self.dimension <= 3:
                self.geometric_complex = AlphaGeometricComplex(
                    cleaned_points, self.x_range, self.y_range,
                    full_initialisation=True)
            elif self.dimension >= 4:
                self.geometric_complex = RipsGeometricComplex(
                    cleaned_points, self.x_range, self.y_range,
                    full_initialisation=True)

            self.extrema.append({
                "maxima": list(self.geometric_complex.maxima),
                "minima": list(self.geometric_complex.minima)
                })

            self.persistence_pairs.append(
                {"x_filtration_H0":
                    self.geometric_complex.filtrations[
                        'X'].homology_0['dying'],
                 "x_inv_filtration_H0":
                    self.geometric_complex.filtrations[
                        'X_inverted'].homology_0['dying'],
                 "y_filtration_H0":
                    self.geometric_complex.filtrations[
                        'Y'].homology_0['dying'],
                 "y_inv_filtration_H0":
                    self.geometric_complex.filtrations[
                        'Y_inverted'].homology_0['dying']})

    def save_topological_summary(self):
        """
        Saves $model-OutlierModels persistence_pairs to "diagram_$model"
        located in the pairXXXX directory.

        persistence_pairs is an outlier-indexed dictionary of 4-tuples:
        0: x_filtration pairs
        1: x_inv_filtration paris
        2: y_filtration pairs
        3: y_inv_filtration pairs
        """

        self.save_diagrams()
        self.save_extrema()

    def save_diagrams(self, filename='diagrams_'):
        import json
        file = os.path.join(self.current_dir, filename + self.model)
        with open(file, 'w') as f:
            json.dump(self.persistence_pairs, f)
            # for line in self.knn.persistence_pairs:
            #     f.write(line)

    def save_extrema(self, filename='extrema_'):
        import json
        file = os.path.join(self.current_dir, filename + self.model)
        with open(file, 'w') as f:
            json.dump(self.extrema, f)


def workflow(model):
    p = CauseEffectPair(model)
    p.save_topological_summary()

if __name__ == "__main__":
    if len(sys.argv) == 2:
        workflow(sys.argv[1])
    else:
        print("Usage: TDA.py $MODEL")
