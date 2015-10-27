import os
import sys

import numpy as np
import numpy.ma as ma

from sklearn.neighbors import NearestNeighbors
import scipy.spatial as spsp


class DataPair:

    """A Class encapsulating everything what we do to the pairs-data
    string      self.name           file name
    np.array    self.raw_data       data loaded from file
                self.orig_points    (potentially) sampled points
                self.points         points we use for all computations
                                    watch out using them!
                self.cleaned_points orig_points - outliers
    list        self.outliers       list of indices of detected outlers in
                                    orig_points
    string      self.target_dir     (path) where to save all the points/outliers
    """

    def __init__(self, model):
        self.current_dir = os.getcwd()
        self.name = self.current_dir[-8:]
        self.points = np.loadtxt(os.path.join(self.current_dir, 'std_points'))
        self.model = model
        self.dimension = self.points[0].shape[0]

        self.n_of_outliers = 15 * int(self.points.shape[0] / 100)

    def find_outliers_knn(self, k_nearest):

        print("Finding knn", self.n_of_outliers, "outliers in", self.name)

        neigh = NearestNeighbors()
        neigh.fit(self.points)
        distances, indices = neigh.kneighbors(self.points,
                                              k_nearest + self.n_of_outliers)
        self.outliers = []

        for each in range(self.n_of_outliers):
            # print(self.name[-2:] + "::" + str(self.n_of_outliers - each),
            #       end="; ", flush=True)
            distances_partial = distances[:, 1:k_nearest+1]
            distances_vector = distances_partial.sum(1)
            outlier = distances_vector.argmax()
            self.outliers.append(outlier)

            distances[outlier] = np.zeros(k_nearest + self.n_of_outliers)

            for i, row in enumerate(indices):
                if outlier in row:
                    distances[i][np.where(row == outlier)[0][0]] = 1000
                    distances[i].sort()
        return self.outliers

    def find_outliers_all(self):

        print("Finding all", self.n_of_outliers, "outliers in", self.name)

        distances_matrix = spsp.distance_matrix(self.points, self.points)
        self.outliers = []

        distances_vector = ma.masked_array(np.sum(distances_matrix, axis=1))
        for i in range(self.n_of_outliers):
            outlier = distances_vector.argmax()
            # print(self.name[-2:] + ":" + str(self.n_of_outliers - i),
            #       end='; ', flush=True)
            self.outliers.append(outlier)
            distances_vector -= distances_matrix[:, outlier]
            distances_vector[outlier] = ma.masked
        return self.outliers

    def find_outliers(self):
        """Procedure finding outliers based on nearest neighbours.
        if neighbours == 0 then all other points are taken into the account
        Outliers (their indexes in self.points) are stored in self.outliers"""

        if model == 'all':  # outlier based on max distance to all others
            self.outliers = self.find_outliers_all()
        if model == 'knn':
            nearest_neighbours = 2 * int(self.points.shape[0] / 100) + 2
            self.outliers = self.find_outliers_knn(nearest_neighbours)

        print(self.name + ' Done with outliers!')

    def save_outliers(self):
        np.savetxt(os.path.join(self.current_dir, "outliers_" + self.model),
                   np.asarray(self.outliers, dtype=int), fmt='%d')


def workflow(model):

    p = DataPair(model)
    p.find_outliers()
    p.save_outliers()


if __name__ == "__main__":

    if len(sys.argv) == 2:
        model = sys.argv[1]
        workflow(model)
    else:
        print("Usage: identify-outliers.py $MODEL")
#
# def main(prefix, size=2000, jobs=1):
#
#     random.seed(0)
#
#     pairs_dir = os.path.join(os.getcwd(), 'pairs')
#
#     blacklist = ['pair0023.txt',
#                  'pair0033.txt',
#                  'pair0037.txt',
#                  'pair0047.txt',
#                  'pair0070.txt']  # pairs that do not fit the model at all
#
#     pattern = re.compile('pair00..\.txt')
#     files = sorted([pattern.match(x).group() for x in os.listdir(pairs_dir)
#                     if pattern.match(x)])
#
#     # for now we use only 2d data:
#     for f in files:
#         with open(os.path.join(pairs_dir, f), 'r') as file:
#             first = file.readline().strip()
#             if len(first.split()) != 2:
#                 print(f+" has to many dimensions")
#                 blacklist.append(f)
#
#     done = []
#     import shutil
#
#     for f in files:
#         dest_dir = os.path.join(os.getcwd(), prefix, f[:-4])
#         # if the destination directory exists and contains "done" file we're
#         # done
#         if os.path.isdir(dest_dir) and "done" in os.listdir(dest_dir):
#             print("Directory " + dest_dir +
#                   " exists and appears to contain finished computations!")
#             done.append(f)
#         else:
#             # if it doesn't contain the "done" file, lets nuke the content
#             if os.path.isdir(dest_dir):
#                 shutil.rmtree(dest_dir)
#
#     files = [f for f in files if f not in blacklist if f not in done]
#     if not files:
#         print("""No pairs to process. To delete old data try running
#               with --clean switch!""")
#     for f in files:
#         print(f)
#     print("Number of files to process: ", len(files))
#
# #     serial version:
# #     for f in files:
# #         workflow(f)
#
# #     parallel version:
#
#     from functools import partial
#
#     work = partial(workflow, prefix=prefix, size=size)
#
#     with Pool(jobs) as p:
#         p.map(work, files)
#
# #     call(["tar", "-acf", prefix + "_pairs.tar.xz", target_dir])
#
# if __name__ == "__main__":
#     if len(sys.argv) < 2:
#         print("No name for the prefix given. Aborting!")
#         print("Usage: identify_outliers [-j N] [-f] [-k]")
#         print(
#             "\t prefix \t dictionary name where cleaned data will be written")
#         print("\t -j \t\t outlier removal will be performed in parallel")
#         print("\t\t\t autodetect number of workers (=no of cpus)")
#         print("\t -j N \t\t N parallel workers")
#         print("\t -f, --force \t overwrites files")
#         print(
#             "\t -k, --clean \t removes the whole prefix directory for clean start")
#     else:
#         if '-j' in sys.argv[2:]:
#             k = sys.argv.index('-j')
#             if k + 1 != len(sys.argv) and sys.argv[k + 1].isdecimal():
#                 jobs = int(sys.argv[k + 1])
#             else:
#                 jobs = cpu_count()
#                 print("Decided automatically on ", jobs, " jobs.")
#         else:
#             jobs = 1
#
#         prefix = sys.argv[1]
#         prefix_dir = os.path.join(os.getcwd(), prefix)
#         if not os.path.isdir(prefix_dir):
#             print("Creating directory", prefix)
#             os.mkdir(prefix_dir)
#             main(prefix=prefix, jobs=jobs)
#
#         elif '-f' in sys.argv[2:] or '--force' in sys.argv[2:]:
#             print("Directory already exists. Overwriting files inside...")
#             if '-k' in sys.argv[2:] or '--clean' in sys.argv[2:]:
#                 print("Deleting content of", prefix_dir)
#                 import shutil
#                 shutil.rmtree(prefix_dir)
#                 os.mkdir(prefix_dir)
#             del prefix_dir
#             main(prefix=prefix, jobs=jobs)
#
#         else:
#             print("""Directory already exists. If You want to overwrite
#                         files inside try -f or --force""")
