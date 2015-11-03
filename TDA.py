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

    def __init__(self, fcomplex):
        self.empty_diagram = self.dionysus.PersistenceDiagram(0)
        self.empty_diagram.append((0, 0))
        logging.debug("Initialising Static Presistence")
        self.persistence = self.dionysus.StaticPersistence(fcomplex)

        logging.debug("Pairing Simplices")
        self.persistence.pair_simplices()
        self.smap = self.persistence.make_simplex_map(fcomplex)

        logging.debug("Computing Persistence Pairs")
        self.compute_persistence_pairs()

        logging.debug("Creating Diagrams")
        self.create_persistence_diagrams()

    def distance(self, p=0):
        """Returns p-th Wasserstein distance between the filtration's 0-diagram
        and the empty diagram.
        If p=0 then the bottleneck is returned.

        TODO: higher dimensions"""

        if p > 0:
            return self.dionysus.wasserstein_distance(
                self.persistence_diagram_0, self.empty_diagram, p)
        else:
            return self.dionysus.bottleneck_distance(
                self.persistence_diagram_0, self.empty_diagram)

    def compute_persistence_pairs(self):
        """ Computes the persistence pairs for the filered complex"""
        h0 = []
        inf_life_0 = []
        # h1 = []
        # Inf_life_1 = []
        undying = 0
        for i in self.persistence:
            if i.sign() > 0:
                birth_simplex = self.smap[i]
                birth = birth_simplex.data
                if i.unpaired():
                    death = float('inf')

                    if birth_simplex.dimension() == 0:
                        inf_life_0.append([birth, death])
                        logging.debug("Undying simplex: %s at %f",
                                      birth_simplex, birth_simplex.data)
                        undying += 1
                        if undying > 1:
                            logging.warning("The complex seems to be "
                                            "disconnected?!")
                    elif birth_simplex.dimension() == 1:
                        pass
                    #     Inf_life_1.append([birth, death])
                    else:
                        logging.warning("There should be no simplices of "
                                        "dim >1?! but there is: %s",
                                        birth_simplex)
                else:
                    killing_simplex = self.smap[i.pair()]
                    death = killing_simplex.data
                    if death > birth:
                        if birth_simplex.dimension() == 0:
                            h0.append([birth, death])
                        elif birth_simplex.dimension() == 1:
                            pass
                        #     h1.append([birth, death])
                        else:
                            logging.warning("There should be no simplices of "
                                            "dim >1?! but there is: %s",
                                            birth_simplex)
                    elif death < birth:
                        logging.warning("You can not die before You were born!")
                        logging.warning(birth_simplex, birth,
                                        killing_simplex, death)
                    else:
                        pass

        self.h_0 = h0
        # self.h_1 = h1
        self.inf_life_0 = inf_life_0
        # self.inf_life_1 = inf_life_1
        # return h0, h1, inf_life_0, inf_life_1

    def create_persistence_diagrams(self):
        if self.h_0:
            self.persistence_diagram_0 = self.dionysus.PersistenceDiagram(0)
            all_pairs = [tuple(x) for x in self.h_0]
            for x in all_pairs:
                self.persistence_diagram_0.append(x)
        else:
            self.persistence_diagram_0 = self.empty_diagram
#         if self.h_1:
#             self.persistence_diagram_1 = dionysus.PersistenceDiagram(1)
#             all_pairs = [tuple(x) for x in self.h_1]
#             for x in all_pairs:
#                 self.persistence_diagram_1.append(x)
#         else:
#             self.persistence_diagram_1 = self.empty_diagram


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

    def __init__(self, cleaned_data, x_range=range(0, 1), y_range=range(1, 2),
                 full_initialisation=True):

        self.points = cleaned_data
        logging.info("Creating GeometricComplex on %d points",
                     self.points.shape[0])
        self.dimension = self.points[0].shape[0]
        self.standardise_data()

        if self.dimension <= 3:
            self.complex_model = "alpha"
        else:
            self.complex_model = "rips"

        self.x_range = x_range
        logging.info("Variable X range: %s", " ".join([str(i) for i in
                                                    self.x_range]))
        self.y_range = y_range
        logging.info("Variable Y range: %s", " ".join([str(i) for i in
                                                    self.y_range]))
        self.maxima = [np.max(self.points[:, i])
                       for i in range(self.dimension)]
        self.minima = [np.min(self.points[:, i])
                       for i in range(self.dimension)]
        self.full_complex = self.create_full_complex(
            radius=np.sqrt(self.dimension))
        self.the_cutoff = self.compute_the_last_death()

        self.limited_complex = self.create_limited_complex(
            threshold=self.the_cutoff)

        if full_initialisation:
            self.x_filtrations = []
            self.x_inv_filtrations = []
            for i in self.x_range:
                logging.info("X-variable: Projecting on %d-th axis", i)
                self.x_filtrations.append(FilteredComplex(
                    self.filtered_complex(i)))
                self.x_inv_filtrations.append(FilteredComplex(
                    self.filtered_complex(i, inverse=True)))

            self.y_filtrations = []
            self.y_inv_filtrations = []
            for i in self.y_range:
                logging.info("Y-variable: Projecting on %d-th axis", i)
                self.y_filtrations.append(FilteredComplex(
                    self.filtered_complex(i)))
                self.y_inv_filtrations.append(FilteredComplex(
                    self.filtered_complex(i, inverse=True)))

    def create_full_complex(self, radius):
        """
        Creates the SORTED full complex (i.e. dionysus object) on the
        self.points.
        Depending on the dimension n of the points it may be alpha-complex
        (for n=2,3) or Rips-complex (for n>3).

        Note that Rips complex may quickly become huge for dense datasets.
        We generate Rips complex for higher-dimensional data;
        We restrict to 1-skeleton (ie. points & edges) and build edges of
        length  <=1.
        This relies on the assumption that we deal with STANDARDISED DATA
        """
        full_complex = self.dionysus.Filtration()

        if self.complex_model == "alpha":
            self.dionysus.fill_alpha_complex(self.points.tolist(),
                                             full_complex)
            one_skeleton = [smpl for smpl in full_complex
                            if smpl.dimension() <= 1]
            full_complex = self.dionysus.Filtration(one_skeleton)
            full_complex.sort(self.dionysus.data_dim_cmp)

        elif self.complex_model == "rips":
            logging.info("Using Rips-complex with radius %f. This may be slow "
                         "for dense sets!", radius)

            full_complex = self.exact_rips_graph(radius)

        logging.info("Created %s full complex of size %d", self.complex_model,
                     full_complex.__len__())

        return full_complex

    def exact_rips_graph(self, radius):
        """
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

    def compute_the_last_death(self):
        """finds the minimal filtration s.t. the full_complex is connected"""
        full_persistence = self.dionysus.StaticPersistence(self.full_complex)
        full_persistence.pair_simplices()
        smap = full_persistence.make_simplex_map(self.full_complex)
        if self.complex_model == 'alpha':
            deaths = [smap[i.pair()].data[0] for i in full_persistence
                      if smap[i].dimension() == 0]
        else:  # self.complex_model == "rips":
            deaths = [smap[i.pair()].data for i in full_persistence
                      if smap[i].dimension() == 0]
        return max(deaths)

    def create_limited_complex(self, threshold):
        """ Creates complex by limiting simplices of self.full_complex
        to those which have data[0] equal or smaller than cutoff"""
        if self.complex_model == 'alpha':
            limited_simplices = [s for s in self.full_complex
                                 if s.data[0] <= threshold and
                                 s.dimension() < 2]
        else:  # self.complex_model == "rips":
            limited_simplices = [s for s in self.full_complex
                                 if s.data <= threshold]
        limited_complex = self.dionysus.Filtration(limited_simplices)
        logging.info("The threshold %f limits the complex size to "
                     "%d", threshold, limited_complex.__len__())
        return limited_complex

    def filtered_complex(self, axis, inverse=False):
        """This method returns self.limited_complex filtered along
        projection on $axis
        in direction:
        ascending when inverse=False;
        descending when inverse=True"""
        weighted_simplices = []
        for simplex in self.limited_complex:
            d = self.sweep_function(simplex, axis, inverse)
            simplex.data = d
            weighted_simplices.append(simplex)
        weighted_simplices.sort(key=lambda s: s.data)
        filtered_complex = self.dionysus.Filtration(weighted_simplices)
        return filtered_complex

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

    def real_coords(self, vertices):
        """returns the physical coordinates of a list of vertices"""
        return self.points[vertices]

    def sweep_function(self, simplex, axis, inverse):
        """ Given a simplex returns max value of the orthogonal projection
        on the axis.
        If inverse is set to true, it returns min value """

        # this turns out to be much (20?!) faster than list(simplex.vertices)
        x = simplex.vertices
        if simplex.dimension() == 0:
            vert = [next(x)]
        elif simplex.dimension() == 1:
            vert = [next(x), next(x)]
        else:
            logging.warning("There shouldn't be any simplices of dim >1?!")
            vert = [v for v in x]

        simplex_real_coordinates = self.real_coords(vertices=vert)
        simplex_projection = simplex_real_coordinates[:, axis]
        if not inverse:
            return max(simplex_projection) - self.minima[axis]
        if inverse:
            return self.maxima[axis] - min(simplex_projection)

    def standardise_data(self):
        """Standardise self.points IN-PLACE i.e.
        mean = 0 and standard deviation = 1 in all dimensions"""
        for i in range(self.dimension):
            p = self.points[:, i]
            mean = np.mean(p)
            std = np.std(p)
            p -= mean
            p /= std


class CauseEffectPair:
    """
    Encapsulates the whole logical concept behind Cause-Effect Pair.
    I.e. contains points, the whole list of outliers, metadata of a pair, etc.
    """

    def __init__(self, model):
        self.current_dir = os.getcwd()
        self.name = self.current_dir[-8:]

        # logging.basicConfig(filename=self.name+".log", level=logging.INFO,
        #                     format='%(asctime)s - %(levelname)s - %(message)s')

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
        self.dimension = int(self.std_points[0].shape[0])

        outliers_file = os.path.join(self.current_dir, "outliers_" + self.model)
        self.outliers = np.loadtxt(outliers_file).astype(np.int)

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

        points_masked = ma.MaskedArray(self.std_points)
        self.persistence_pairs = []
        self.extrema = []
        for i, outlier in enumerate(self.outliers, start=1):
            logging.info("Outlier: %d of %d", i, self.outliers.shape[0])
            points_masked[outlier] = ma.masked
            cleaned_points = points_masked.compressed().reshape(
                self.std_points.shape[0] - i, self.dimension)
            self.geometric_complex = GeometricComplex(cleaned_points,
                                                      self.x_range,
                                                      self.y_range)

            self.extrema.append({
                "maxima": self.geometric_complex.maxima,
                "minima": self.geometric_complex.minima
                })

            self.persistence_pairs.append(
                {"x_filtration_H0":
                    [f.h_0 for f in self.geometric_complex.x_filtrations],
                 "x_inv_filtration_H0":
                    [f.h_0 for f in self.geometric_complex.x_inv_filtrations],
                 "y_filtration_H0":
                    [f.h_0 for f in self.geometric_complex.y_filtrations],
                 "y_inv_filtration_H0":
                    [f.h_0 for f in self.geometric_complex.y_inv_filtrations]})

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
