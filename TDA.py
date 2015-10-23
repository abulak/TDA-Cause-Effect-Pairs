import numpy as np
import numpy.ma as ma
import os
import sys
import re

import matplotlib
matplotlib.use('Agg')
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt

sys.path.append("../Dionysus-python3/build/bindings/python")

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

        self.persistence = self.dionysus.StaticPersistence(fcomplex)
        self.persistence.pair_simplices()
        self.smap = self.persistence.make_simplex_map(fcomplex)
        self.compute_persistence_pairs()
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
        """ Computes the persistence pairs for the filered complex
        (-1 as death stands for infty)"""
        h0 = []
        # h1 = []
#         Inf_life_0 = []
#         Inf_life_1 = []
        for i in self.persistence:
            if i.sign() > 0:
                birth_simplex = self.smap[i]
                birth = birth_simplex.data
                if i.unpaired():
                    pass
#                     death = float('inf')
                    # print(birth_simplex.data)
#                     if birth_simplex.dimension() == 0:
#                         Inf_life_0.append([birth, death])
#                     if birth_simplex.dimension() == 1:
#                         Inf_life_1.append([birth, death])
                else:
                    killing_simplex = self.smap[i.pair()]
                    death = killing_simplex.data
                    if birth_simplex.dimension() == 0:
                        h0.append([birth, death])
                    # if birth_simplex.dimension() == 1:
                    #     h1.append([birth, death])

        self.h_0 = np.asarray([x for x in h0 if (x[0] - x[1]) ** 2 > 0])
        # list([x for x in H0]))
        # self.h_1 = np.asarray(h1)
        # list([x for x in H1]))

#         h0 = np.asarray(list([x for x in h0 if (x[0] - x[1]) ** 2 > 0]))
#         h1 = np.asarray(list([x for x in h1 if (x[0] - x[1]) ** 2 > 0]))
#         i_life_0 = np.asarray(inf_life_0)
#         i_life_1 = np.asarray(inf_life_1)
#         return h0, h1, i_life_0, i_life_1

    def create_persistence_diagrams(self):
        if self.h_0.any():
            self.persistence_diagram_0 = self.dionysus.PersistenceDiagram(0)
            all_pairs = [tuple(x) for x in self.h_0]
            for x in all_pairs:
                self.persistence_diagram_0.append(x)
        else:
            self.persistence_diagram_0 = self.empty_diagram
#         if self.h_1.any():
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
    dionysus = __import__('dionysus')

    def __init__(self, cleaned_data):

        self.points = cleaned_data
        self.dimension = self.points[0].shape[0]
        self.standardise_data()

        self.maximums = [np.max(self.points[:, i])
                         for i in range(self.dimension)]
        self.minimums = [np.min(self.points[:, i])
                         for i in range(self.dimension)]

        self.__create_full_complex__()
        self.the_alpha = self.compute_the_last_death()
        self.__create_limited_complex__(threshold=self.the_alpha)
        self.x_filtration = FilteredComplex(self.filtered_complex(0))
        self.y_filtration = FilteredComplex(self.filtered_complex(1))
        self.x_inv_filtration = FilteredComplex(
            self.filtered_complex(0, inverse=True))
        self.y_inv_filtration = FilteredComplex(
            self.filtered_complex(1, inverse=True))

    def __create_full_complex__(self):
        """Creates the full complex (i.e. dionysus object) on the self.points.
        Depending on the dimension n of the points it may be alpha-complex
        (for n=2,3) or Rips-complex (for n>3).

        Note that Rips complex may quickly become huge for dense datasets."""
        self.full_complex = self.dionysus.Filtration()

        if self.dimension == 2:
            self.dionysus.fill_alpha2D_complex(self.points.tolist(),
                                               self.full_complex)
        if self.dimension == 3:
            self.dionysus.fill_alpha3D_complex(self.points.tolist(),
                                               self.full_complex)
        if self.dimension > 3:
            print("Using Rips-complex. This may (or may not) be slow!")
#             distances = dionysus.PairwiseDistances(self.points.tolist())
            distances = self.dionysus.PairwiseDistances(self.points.tolist())
            rips = self.dionysus.Rips(distances)
            # We generate Rips complex for higher-dimensional data;
            # We restrict to 1-skeleton (ie. points & edges)
            # We restrict all edges to length of 2 as we deal with
            # STANDARDISED DATA
            rips.generate(1, 2, self.full_complex.append)
            for s in self.full_complex:
                s.data = rips.eval(s)

        self.full_complex.sort(self.dionysus.data_dim_cmp)

    def compute_the_last_death(self):
        """finds the minimal filtration s.t. the full_complex is connected"""
        full_persistence = self.dionysus.StaticPersistence(self.full_complex)
        full_persistence.pair_simplices()
        smap = full_persistence.make_simplex_map(self.full_complex)
        deaths = [smap[i.pair()].data[0] for i in full_persistence
                  if smap[i].dimension() == 0]
        return max(deaths)

    def __create_limited_complex__(self, threshold):
        """ Creates complex by limiting simplices of self.full_complex
        to those which have data[0] equal or smaller than cutoff"""
        limited_simplices = [s for s in self.full_complex
                             if s.data[0] <= threshold and s.dimension() < 2]
        self.limited_complex = self.dionysus.Filtration(limited_simplices)

    def filtered_complex(self, axis, inverse=False):
        """This method is actually a function. Returs filtered
        self.limited_complex along axis and in direction:
        ascending when inverse=False;
        descending when inverse=True"""
        weighted_simplices = []
        for simplex in self.limited_complex:
            simplex.data = self.sweep_function(simplex, axis, inverse)
            weighted_simplices.append(simplex)
        weighted_simplices.sort(key=lambda s: s.data)
        filtered_complex = self.dionysus.Filtration(weighted_simplices)
        # filtered_complex.sort(dionysus.data_dim_cmp)
        return filtered_complex

    def real_coords(self, vertex):
        """returns the physical coordinates of a vertex """
        return self.points[vertex]

    def sweep_function(self, simplex, axis, inverse):
        """ Given a simplex returns max value of the orthogonal projection
        on the axis.
        If inverse is set to true, it returns min value """

        simplex_real_coordinates = [self.real_coords(i)
                                    for i in simplex.vertices]
        simplex_projection = [point[axis]
                              for point in simplex_real_coordinates]

        if not inverse:
            return max(simplex_projection) - self.minimums[axis]
        if inverse:
            return self.maximums[axis] - min(simplex_projection)

    def standardise_data(self):
        """Standardise self.points IN-PLACE i.e.
        mean = 0 and standard deviation = 1 in all dimensions"""
        for i in range(self.dimension):
            p = self.points[:, i]
            mean = np.mean(p)
            std = np.std(p)
            p -= mean
            p /= std


class OutliersModel:

    def __init__(self, orig_points_list, outliers_list):
        self.orig_points = orig_points_list
        self.dimension = self.orig_points[0].shape[0]
        self.outliers = outliers_list

        self.x_causes_y_scores = np.zeros(self.outliers.shape[0]+1)
        self.y_causes_x_scores = np.zeros(self.outliers.shape[0]+1)

        self.geometric_complex = GeometricComplex(self.orig_points)

        self.x_causes_y_scores[0] = max(
            self.geometric_complex.y_filtration.distance(),
            self.geometric_complex.y_inv_filtration.distance()
            )
        self.y_causes_x_scores[0] = max(
            self.geometric_complex.x_filtration.distance(),
            self.geometric_complex.x_inv_filtration.distance()
            )

        self.compute_topological_summary()

    def compute_topological_summary(self):
        """
        For each in self.outliers generate cleaned_points. Then construct
        GeometricComplex(cleaned_points) and compute its persistant 0-th
        homology.

        We save the 0-persistence pairs in the _list_
        self.persistence_pairs.

        persistence_pairs[outlier] contains 4 items:
        0: 0-persistence pairs for x_filtration
        1: 0-persistence pairs for x_inv_filtration
        2: 0-persistence pairs for y_filtration
        3: 0-persistence pairs for y_inv_filtration

        Based on that we generate scores for hypotheses.

        In particular the self.x_causes_y_scores and self.y_causes_x_scores
        are populated by this function.
        """

        points_masked = ma.MaskedArray(self.orig_points)
        self.persistence_pairs = []

        for i, outlier in enumerate(self.outliers, start=1):
            print(str(self.outliers.shape[0]-i), end="; ",
                  flush=True)
            points_masked[outlier] = ma.masked
            cleaned_points = points_masked.compressed().reshape(
                self.orig_points.shape[0] - i, self.dimension)
            self.geometric_complex = GeometricComplex(cleaned_points)

            self.persistence_pairs.append([
                self.geometric_complex.x_filtration.persistence_diagram_0,
                self.geometric_complex.x_inv_filtration.persistence_diagram_0,
                self.geometric_complex.y_filtration.persistence_diagram_0,
                self.geometric_complex.y_inv_filtration.persistence_diagram_0])

            self.x_causes_y_scores[i] = max(
                self.geometric_complex.y_filtration.distance(),
                self.geometric_complex.y_inv_filtration.distance())
            self.y_causes_x_scores[i] = max(
                self.geometric_complex.x_filtration.distance(),
                self.geometric_complex.x_inv_filtration.distance())


class Pair:
    """
    Encapsulates the whole logical concept behind Cause-Effect Pair.
    I.e. contains, the whole list of outliers, etc.
    """
    def __init__(self, prefix, pair_name):
        self.type = type
        self.prefix = prefix

        if pair_name[-4:] == '.txt':
            pair_name = pair_name[:-4]
        self.name = pair_name
        print(self.name, end=" ", flush=True)

        self.prefix_dir = os.path.join(os.getcwd(), self.prefix)
        self.directory = os.path.join(self.prefix_dir, self.name)

        self.prepare_points()

        # self.all = OutliersModel(self.orig_points, self.outliers_all)
        # print("all-model scores stability done")
        self.knn = OutliersModel(self.orig_points, self.outliers_knn)
        print("knn-model scores stability done")
        print(self.name, "done!")

    def prepare_points(self):
        self.orig_points = np.loadtxt(os.path.join(self.directory,
                                                   "orig_points"))
        self.dimension = int(self.orig_points[0].shape[0])

        self.outliers_all = np.loadtxt(os.path.join(self.directory,
            "outliers_all")).astype(np.int)
        self.outliers_knn = np.loadtxt(os.path.join(self.directory,
            "outliers_knn")).astype(np.int)

    def save_topological_summary(self):
        """
        Saves knn and all OutlierModels
        persistence_pairs to "persistence_pairs_knn" and "persistence_pairs_all"
        causality_scores to "scores_knn" and "scores_all"
        located in the pairXXXX directory.

        persistence_pairs is an outlier-indexed list of 4-tuples:
        0: x_filtration pairs
        1: x_inv_filtration paris
        2: y_filtration pairs
        3: y_inv_filtration pairs

        causality_score is a numpy array of two rows:
        [0:,] x_causes_y_scores
        [1:,] y_causes_x_scores
        is numpy.savetxt-ed.
        """
        self.save_scores()
        self.save_diagrams()

    def save_diagrams(self, filename='diagrams'):
        file = os.path.join(self.directory, filename+'_knn')
        with open(file, 'w') as f:
            for line in self.knn.persistence_pairs:
                f.write(line)

        file = os.path.join(self.directory, filename+'_all')
        with open(file, 'w') as f:
            for line in self.knn.persistence_pairs:
                f.write(line)


    def save_scores(self, filename='scores'):
        file = os.path.join(self.directory, filename+'_knn')
        z = np.zeros((2, self.knn.x_causes_y_scores.shape[0]))
        z[0, :] = self.knn.x_causes_y_scores
        z[1, :] = self.knn.y_causes_x_scores
        np.savetxt(file, z)

        file = os.path.join(self.directory, filename+'_all')
        z = np.zeros((2, self.all.x_causes_y_scores.shape[0]))
        z[0, :] = self.all.x_causes_y_scores
        z[1, :] = self.all.y_causes_x_scores
        np.savetxt(file, z)

    def plot_scores(self):
        pdf_file = os.path.join(self.directory, 'scores.pdf')
        with PdfPages(pdf_file) as pdf:
            plt.figure(figsize=(12, 12))
            plt.title(self.name+" knn scores")
            plt.plot(self.knn.x_causes_y_scores, label='x->y')
            plt.plot(self.knn.y_causes_x_scores, label='y->x')
            plt.legend()
            pdf.savefig()
            plt.close()

            plt.figure(figsize=(12, 12))
            plt.title(self.name+" all scores")
            plt.plot(self.all.x_causes_y_scores, label='x->y')
            plt.plot(self.all.y_causes_x_scores, label='y->x')
            plt.legend()
            pdf.savefig()
            plt.close()


def workflow(pair, prefix):

    target_directory_list = os.listdir(os.path.join(os.getcwd(), prefix, pair))
    # if "scores_knn" in target_directory_list and "scores_all" in \
    #         target_directory_list:
    #     print("Scores for", pair, "seem to be already computed")
    #     return 0
    #
    #
    # else:
    p = Pair(prefix, pair)
    p.save_scores()
    p.save_diagrams()
    p.plot_scores()
    # print('Saving results of', pair, 'done')
    return 1

if __name__ == "__main__":

    print(sys.argv)
    if len(sys.argv) == 2:

        prefix = sys.argv[1]

        pattern = re.compile('pair[0-9]+$')
        prefix_path_list= os.listdir(os.path.join(os.getcwd(), prefix))
        pairs = sorted([x for x in prefix_path_list if pattern.match(
                x)])
        # print(pairs)

        # for pair in pairs:
        #     workflow(pair, prefix)

        import multiprocessing as mproc

        from functools import partial
        partial_work = partial(workflow, prefix=prefix)

        with mproc.Pool(mproc.cpu_count()) as pool:
            pool.map(partial_work, pairs)
    else:
        print("Usage:\n"
              "TDA_classes_new.py prefix_dir\n"
              "where prefix_dir is directory with precomputed "
              "pairs_dirs containing orig_points and outliers")
