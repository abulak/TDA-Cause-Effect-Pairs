import numpy as np
import sys
import os
import logging


def quantise(points):
    """
    if one of the axes in points is heavily digitised, digitise all the other
    to the same number of bins.
    :param points: np.array
    :return: np.array: digitised points to the same number of bins
    """

    number_of_bins = [len(set(points[:, i]))
                      for i in range(points.shape[1])]
    logging.info("Numbers of different values along axes (bins): %s",
                 str(number_of_bins))
    m = min(number_of_bins)

    logging.info("Pair %s has been quantised.", filename)
    for i in range(points.shape[1]):
        points[:, i] = fit_to_bins(points[:, i], m)

    return points


def fit_to_bins(column, m):
    """
    fits values in column to m equally spaced bins between column.min() and
    column.max()
    :param column: np.array of shape (n,1)
    :param m: int: number of bins
    :return: the column of values rounded to the center of the nearest bin
    """
    bins, step = np.linspace(column.min(), column.max()+0.001, m+1,
                             retstep=True)
    inds = np.digitize(column, bins)
    digitised = np.array([bins[i]-step/2 for i in inds])
    return digitised


def workflow(filename, size=1000, quant=False):

    logging.basicConfig(filename=filename[:-4]+".log",
                        filemode='w',
                        level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')

    logging.info("Sampling up to %d points from %s", size, filename)
    target_dir = os.path.join(os.getcwd())

    raw_data = np.loadtxt(os.path.join(os.pardir, 'pairs', filename))
    dimension = raw_data.shape[1]
    if size < 1:
        logging.info("Size parameter was <1. Using all available points.")
        points = raw_data
    elif raw_data.shape[0] < size:
        logging.info("Data contains %d < %d points. Using all available.",
                     raw_data.shape[0], size)
        points = raw_data
    else:
        # np.random.seed(0)
        logging.info("Data contains %d > %d points. Sampling %d.",
                     raw_data.shape[0], size, size)
        indices = np.random.randint(0, raw_data.shape[0], size)
        points = raw_data[indices]

    column_blacklist = []
    for i in range(dimension):
        if np.isnan(np.dot(points[:, i], points[:, i])):
            logging.info("Column %d contains Nan(s). Discarding", i)
            column_blacklist.append(i)
    if column_blacklist:
        import numpy.ma as ma
        masked_points = ma.MaskedArray(points)
        for i in column_blacklist:
            masked_points[:, i] = ma.masked

        new_shape = (points.shape[0], points.shape[1]-len(column_blacklist))
        logging.debug("Old shape: %s, New_shape: %s", str(points.shape),
                      str(new_shape))
        logging.debug("Length of compressed points: %d",
                      len(masked_points.compressed()))
        
        points = masked_points.compressed().reshape(new_shape)

    if quant:
        std_points = quantise(points)
    else:
        std_points = points
    np.savetxt(os.path.join(target_dir, 'points.std'), std_points)
    logging.info("Sampling Done!")

if __name__ == '__main__':

    if len(sys.argv) == 3:
        filename = os.getcwd()[-8:] + ".txt"
        size = int(sys.argv[1])
        quant = bool(int(sys.argv[2]))
        workflow(filename, size, quant)
    else:
        print("Usage: points_sampler.py $SIZE $QUANTISATION")
