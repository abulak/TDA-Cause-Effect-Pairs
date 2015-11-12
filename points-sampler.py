import numpy as np
import sys
import os
import logging


def standardise(points):
    """Standardise self.points, i.e.
    mean = 0 and standard deviation = 1 in both dimensions"""
    for i in range(points.shape[1]):
        p = points[:, i]
        mean = np.mean(p)
        std = np.std(p)
        p -= mean
        p /= std
    return points


def digitise(points):
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


def workflow(filename, size=1000):

    logging.basicConfig(filename=filename[:-4]+".log",
                        filemode='w',
                        level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')

    logging.info("Sampling up to %d points from %s", size, filename)
    target_dir = os.path.join(os.getcwd())

    os.chdir('../../')
    raw_data = np.loadtxt(os.path.join(os.getcwd(), 'pairs', filename))
    dimension = raw_data.shape[1]
    if size < 1 or raw_data.shape[0] < size:
        logging.info("Data contains less points than %d. Using all available.",
                     size)
        points = raw_data
    else:
        # np.random.seed(0)
        logging.info("Data contains more points than %d. Sampling.",
                     size)
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

    p1 = standardise(points)
    p2 = digitise(p1)
    std_points = standardise(p2)
    np.savetxt(os.path.join(target_dir, 'std_points'), std_points)
    logging.info("Sampling Done!")



if __name__ == '__main__':

    blacklist = ['pair0023.txt',
                 'pair0033.txt',
                 'pair0037.txt',
                 'pair0047.txt',
                 'pair0070.txt']

    if len(sys.argv) == 3:
        filename = sys.argv[1]
        size = int(sys.argv[2])
        if filename not in blacklist:
            workflow(filename, size)
        else:
            print(filename, "is blacklisted! (it doesn't fit the model?)")
    else:
        print("Usage: points-sampler.py $FILENAME $SIZE")
