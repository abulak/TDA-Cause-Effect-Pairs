import subprocess

for i in range(10, 20):
    prefix = 'sampled-600_'+'{0:03d}'.format(i)
    print("Starting ", prefix, end=' ')
    subprocess.call(["make", "-j8", "knn", "PREFIX="+prefix, "SIZE=600"])
    print("done", prefix)



