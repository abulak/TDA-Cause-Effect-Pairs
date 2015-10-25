import os
import random
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

    def __init__(self, filename, prefix, size=2000):
        print(filename)
        print(prefix)
        self.filename = filename
        self.name = self.filename[:-4]
        self.prepare_dirs(prefix)

        self.raw_data = np.loadtxt(
            os.path.join(self.initial_dir, 'pairs', self.filename))

        if size < 1:
            self.orig_points = self.raw_data
        else:
            if self.raw_data.shape[0] > size:
                indexes = np.random.randint(0, self.raw_data.shape[0], size)
                self.orig_points = self.raw_data[indexes]
            else:
                self.orig_points = self.raw_data

        self.points = self.orig_points
        self.dimension = self.points[0].shape[0]

        self.n_of_outliers = 15 * int(self.points.shape[0] / 100)

    def prepare_dirs(self, prefix):
        """file IO operations:
            create target directory where all files will be saved."""

        self.initial_dir = os.getcwd()
        # print(self.initial_dir, prefix, self.name)
        self.target_dir = os.path.join(self.initial_dir, prefix, self.name)
        # os.mkdir(self.target_dir)

    def linearise(self):
        """Fit self.points to [0,1] interval in both coordinates"""
        for i in range(self.dimension):
            p = self.points[:, i]
            maximum = np.max(p)
            minimum = np.min(p)
            p -= minimum
            p /= (maximum - minimum)

    def standardise(self):
        """Standardise self.points, i.e.
        mean = 0 and standard deviation = 1 in both dimensions"""
        for i in range(self.dimension):
            p = self.points[:, i]
            mean = np.mean(p)
            std = np.std(p)
            p -= mean
            p /= std

    def find_outliers_knn(self, k_nearest):

        print("Finding knn", self.n_of_outliers, "outliers in", self.name)

        neigh = NearestNeighbors()
        neigh.fit(self.points)
        distances, indices = neigh.kneighbors(self.points,
                                              k_nearest + self.n_of_outliers)
        self.outliers = []

        for each in range(self.n_of_outliers):
            print(self.name[-2:] + "::" + str(self.n_of_outliers - each),
                  end="; ", flush=True)
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
            print(self.name[-2:] + ":" + str(self.n_of_outliers - i),
                  end='; ', flush=True)
            self.outliers.append(outlier)
            distances_vector -= distances_matrix[:, outlier]
            distances_vector[outlier] = ma.masked
        return self.outliers

    def find_outliers(self, neighbours=0):
        """Procedure finding outliers based on nearest neighbours.
        if neighbours == 0 then all other points are taken into the account
        Outliers (their indexes in self.points) are stored in self.outliers"""

        self.outliers = []

        n = int(neighbours)
        if n <= 0:  # outlier based on max distance to all others
            self.outliers = self.find_outliers_all()
            suffix = 'all'
        else:
            self.outliers = self.find_outliers_knn(neighbours)
            suffix = 'knn'
        self.save_outliers(suffix=suffix)
        print(self.name + ' Done with outliers!')

    def save_outliers(self, suffix=''):
        print(self.outliers)
        np.savetxt(os.path.join(self.target_dir, "outliers_" + str(suffix)),
                   np.asarray(self.outliers, dtype=int), fmt='%d')

    def save_points_to_file(self, target_dir='', name="orig_points"):
        """Saves numpy.array accesible under name attribute to
        target_dir/name_self.filename"""

        if target_dir == '':
            target_dir = self.target_dir

        np.savetxt(os.path.join(target_dir, name),
                   self.__getattribute__(name))

    def save_cleaned_points(self, target_dir='', suffix=''):
        """Saves the cleaned point into target_dir/cleaned_self.filename"""
        if target_dir == '':
            target_dir = self.target_dir

        data = ma.masked_array(self.points)

        for outlier in self.outliers:
            data[outlier] = ma.masked
        if suffix:
            suffix = "_" + str(suffix)
        compressed_data = data.compressed().reshape(
            self.points.shape[0] - len(self.outliers),
            self.dimension)
        np.savetxt(os.path.join(target_dir,
                                "cleaned_" + self.filename + suffix),
                   compressed_data)

    def plot_points_pdf(self, suffix=''):
        print("Generating pdf with plots of ", self.name)
        import matplotlib
        matplotlib.use('Agg')
        from matplotlib.backends.backend_pdf import PdfPages
        import matplotlib.pyplot as plt

        pdf_file = os.path.join(self.target_dir,
                                self.filename[:-4] +
                                '_' + str(suffix) + '.pdf')
        with PdfPages(pdf_file) as pdf:
            new_points = ma.masked_array(self.points)
            for i in range(-1, len(self.outliers)):
                # print(self.name[-2:0]+"-"+str(i), end=' ', flush=True)
                plt.figure(figsize=(12, 12))
                if i == -1:
                    plt.title(self.name)
                    plt.scatter(self.points[:, 0], self.points[:, 1],
                                color='black', alpha=0.7, s=15)
                else:
                    outlier = self.outliers[i]
                    removed_points = self.points[self.outliers[:i + 1]]
                    new_points[outlier] = ma.masked
                    to_plot = new_points.compressed().reshape(
                        self.points.shape[0] - i - 1, 2)
                    plt.figure(figsize=(12, 12))
                    plt.title(self.name + ", outliers: " + str(i))
                    plt.scatter(to_plot[:, 0], to_plot[:, 1],
                                color='black', alpha=0.7, s=15)
                    plt.scatter(removed_points[:, 0], removed_points[:, 1],
                                color='red', alpha=0.7, s=15)
                pdf.savefig()
                plt.close()
        print(self.name, "Done pdf!")


def workflow(prefix, filename, size):
    prefix_dir = os.path.join(os.getcwd(), prefix)
    p = DataPair(filename, prefix, size)
    if p.dimension == 2:
        p.standardise()
        p.save_points_to_file()

        p.find_outliers(neighbours=0)
        # if p.dimension == 2:
        #     p.plot_points_pdf(suffix='all')

        p.find_outliers(neighbours=2 * int(p.orig_points.shape[0] / 100))
        # if p.dimension == 2:
        #     p.plot_points_pdf(suffix='knn')

        # else:
        #     print(p.name, p.dimension, "is too many to plot!")
        done_file = os.path.join(prefix_dir, filename[:-4], "done")
        print("Creating file", done_file)
        os.mknod(done_file)
        return 0
    else:
        return 1

if __name__ == "__main__":

    random.seed(0)

    blacklist = ['pair0023.txt',
                 'pair0033.txt',
                 'pair0037.txt',
                 'pair0047.txt',
                 'pair0070.txt']

    if len(sys.argv) < 3:
        print("Usage: identify-outliers.py $PREFIX $FILENAME")
    else:
        prefix = sys.argv[1]
        file = sys.argv[2]
        if file not in blacklist:
            workflow(prefix, file, size=1000)
        else:
            print(file, "is blacklisted! (it doesn't fit the model?)")
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
