import glob
import h5py as hp
import numpy as np

files = glob.glob('*hdf5')

print('found %d files'%len(files))

for n in files:
    try:
        f = hp.File(n)
        f.close()
    except:
        print('could not open %s'%n)


print('Done')
