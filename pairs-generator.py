import subprocess

for i in range(1, 21):
    prefix = '{0:03d}'.format(i)
    print("Starting ", prefix, end=' ')
    subprocess.call(["make", "-j3", "knn", "PREFIX="+prefix, "SIZE=2000"])
    print("done", prefix)
