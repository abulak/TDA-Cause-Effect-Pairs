import subprocess
import re
import os

import numpy as np

SIZE = 2000

pattern = re.compile('pair[0-9][0-9][0-9][0-9].txt')
all_pair_names = sorted([x for x in os.listdir('pairs') if pattern.match(x)])
BLACKLIST = [ # 'pair0026.txt',  # sampling of 'uneven' density
             # 'pair0027.txt',  # sampling of 'uneven' density
             # 'pair0028.txt',  # sampling of 'uneven' density
             # 'pair0029.txt',  # sampling of 'uneven' density
             # 'pair0032.txt',  # sampling of 'uneven' density
             # 'pair0033.txt',  # sampling of 'uneven' density
             'pair0047.txt',  # Y is binary
             'pair0070.txt',  # Y is binary
             'pair0071.txt']  # Y is binary

SLOW_PAIRS = ['pair0052.txt', 'pair0053.txt']

pair_names = [f for f in all_pair_names if f not in BLACKLIST]
fast_pairs = [f for f in pair_names if f not in SLOW_PAIRS]


def save_names(file_name, pairs=pair_names):
    with open(file_name, 'w') as file:
        for f in pairs:
            file.write(f+'\n')

save_names(file_name="fast_pairs", pairs=fast_pairs)


def find_large_pairs(pairs):
    result = []
    for f in pairs:
        x = np.loadtxt(os.path.join('pairs', f))
        if x.shape[0] > SIZE:
            result.append(f)
    return result

# large_pairs = find_large_pairs(fast_pairs)


# for i in range(1, 21):
#     prefix = '{0:03d}'.format(i)
#     print("Starting ", prefix, end=' ')
#     subprocess.call(["make", "-j3", "knn", "PREFIX="+prefix, "SIZE="+str(SIZE)])
#     print("done", prefix)
