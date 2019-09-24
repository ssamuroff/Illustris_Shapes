import numpy as np
import sys
import glob

sim_name = sys.argv[1]
snapshot = sys.argv[2]
shapes = sys.argv[3]
files = glob.glob('%s_%s_%s_galaxy_shapes-*.dat'%(sim_name,snapshot,shapes))

print('Found %d files'%len(files))

base = files[0].split('galaxy_shapes-')[0]+'galaxy_shapes.dat'

out = np.zeros_like(np.genfromtxt(files[0],names=True))
columns = out.dtype.names


for f in files:
	dat = np.genfromtxt(f, names=True)
	mask = (dat['a']!=0)
	for col in columns:
		out[col][mask] = np.copy(dat[col][mask]) 


print('Saving merged catalogue %s'%base)
np.savetxt(base, out)

# add the column names to the top of the file
header = open(files[0]).readline()
with open(base, 'r+') as fp:
	lines = fp.readlines()
	lines.insert(0, header)
	fp.seek(0)
	fp.writelines(lines) 

