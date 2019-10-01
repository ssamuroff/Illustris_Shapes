"""
script to measure galaxy shapes in the Illustris simulations
"""

from __future__ import print_function, division
import numpy as np
from astropy.table import Table, Column
import time
from astropy.io import ascii
import sys
import glob

from illustris_python.snapshot import loadHalo, snapPath, loadSubhalo
from illustris_python.groupcat import gcPath, loadHalos, loadSubhalos
from inertia_tensors import inertia_tensors, reduced_inertia_tensors, iterative_inertia_tensors_3D

from simulation_props import sim_prop_dict

# progress bar
from tqdm import tqdm


def format_particles(center, coords, Lbox):
    """
    center the partivcle coordinates on (0,0,0) and account for PBCs

    Parameters
    ----------
    center : array_like
        array of shape (3,) storing the coordinates of the
        center of the particle distribution

    coords :  array_like
        array of shape (nptcls, 3) storing the coordinates of the
        particles

    Lbox : array_like
        length 3-array giving the box size in each dimension

    Returns
    -------
    coords : numpy.array
        array of shape (nptcls, 3) storing the centered particle coordinates
    """

    dx = coords[:,0] - center[0]
    dy = coords[:,1] - center[1]
    dz = coords[:,2] - center[2]

    # x-coordinate
    mask = (dx > Lbox[0]/2.0)
    dx[mask] = dx[mask] - Lbox[0]
    mask = (dx < -Lbox[0]/2.0)
    dx[mask] = dx[mask] + Lbox[0]

    # y-coordinate
    mask = (dy > Lbox[1]/2.0)
    dy[mask] = dy[mask] - Lbox[1]
    mask = (dy < -Lbox[1]/2.0)
    dy[mask] = dy[mask] + Lbox[1]

    # z-coordinate
    mask = (dz > Lbox[2]/2.0)
    dz[mask] = dz[mask] - Lbox[2]
    mask = (dz < -Lbox[2]/2.0)
    dz[mask] = dz[mask] + Lbox[2]

    # format coordinates
    coords = np.vstack((dx,dy,dz)).T

    return coords


def particle_selection(gal_id, ptcl_coords, galaxy_table, basePath, snapNum, radial_mask=True):
    """
    Apply the selection criteria to galaxy particles

    Returns
    -------
    mask : numpy.array
       boolean array of shape (nptcls,)
    """

    # don't use wind particles
    sf_time = loadSubhalo(basePath, snapNum, gal_id, 4, fields=['GFM_StellarFormationTime'])
    star_mask = (sf_time>=0.0) # don't use wind particles

    # get the half mass radius
    gal_rhalfs = galaxy_table['SubhaloHalfmassRadType'][:,4]/1000.0
    gal_rhalf = gal_rhalfs[gal_id]

    # use only particles within 2 * R_half
    r = np.sqrt(np.sum(ptcl_coords**2, axis=1))/gal_rhalf
    if radial_mask:
        radial_mask = (r<=2.0)
    else:
        radial_mask = np.array([True]*len(star_mask))

    return (radial_mask) & (star_mask)


def galaxy_center(gal_id, galaxy_table):
    """
    Return the coordinates of the center of the galaxy
    """

    # load position of the most bound particle (of any type)
    coords = galaxy_table['SubhaloPos']/1000.0
    coord = coords[gal_id]

    return coord


def galaxy_shape(gal_id, galaxy_table, basePath, snapNum, Lbox, shape_type='reduced'):
    """
    Parameters
    ----------
    gal_id : int

    basepath : string

    snapNum : int

    Lbox : array_like

    reduced : bool

    Returns
    -------
    eig_vals, eig_vecs
    """

    # choose a 'center' for each galaxy
    gal_position = galaxy_center(gal_id, galaxy_table)

    # load stellar particle positions and masses
    try:
        ptcl_coords = loadSubhalo(basePath, snapNum, gal_id, 4, fields=['Coordinates'])/1000.0
        ptcl_masses = loadSubhalo(basePath, snapNum, gal_id, 4, fields=['Masses'])*10.0**10
    except: 
        import pdb ; pdb.set_trace()

    # center and account for PBCs
    ptcl_coords = format_particles(gal_position, ptcl_coords, Lbox)

    # make a selection cut on the particles
    if shape_type=='non-reduced':
        ptcl_mask = particle_selection(gal_id, ptcl_coords, galaxy_table, basePath, snapNum, radial_mask=False)
    else:
        ptcl_mask = particle_selection(gal_id, ptcl_coords, galaxy_table,  basePath, snapNum, radial_mask=True)

    if shape_type == 'reduced':
        I = reduced_inertia_tensors(ptcl_coords[ptcl_mask], ptcl_masses[ptcl_mask])
    elif shape_type == 'non-reduced':
        #I = inertia_tensors(ptcl_coords[ptcl_mask], ptcl_masses[ptcl_mask])
        I = inertia_tensors(ptcl_coords[ptcl_mask], None)
    elif shape_type == 'iterative':
        I = iterative_inertia_tensors_3D(ptcl_coords[ptcl_mask], ptcl_masses[ptcl_mask], rtol=0.01, niter_max=10)
    else:
        msg = ('tensor calculation type not recognized.')
        raise ValueError(msg)

    evals, evecs = np.linalg.eigh(I)
    evals = np.sqrt(evals)

    return evals[0], evecs[0]


def galaxy_selection(min_mstar, min_ndark, basePath, snapNum):
    """
    make a cut on galaxy properties
    """

    # make selection
    galaxy_table = loadSubhalos(basePath, snapNum, fields=['SubhaloGrNr', 'SubhaloMassInRadType', 'SubhaloLenType'])

    gal_ids = np.arange(0,len(galaxy_table['SubhaloGrNr']))

    # mass of stellar particles within 2*R_half
    mstar = galaxy_table['SubhaloMassInRadType'][:,4]
    mstar = mstar*10**10

    ndark = galaxy_table['SubhaloLenType'][:,1]

    mask = (mstar >= min_mstar) & (ndark >= min_ndark)

    return mask, gal_ids[mask]


def reset_arrays(Ngals):
    # create array to store shape properties
    # eigen values
    a = np.zeros(Ngals)
    b = np.zeros(Ngals)
    c = np.zeros(Ngals)
    # eigen vectors
    av = np.zeros((Ngals,3))
    bv = np.zeros((Ngals,3))
    cv = np.zeros((Ngals,3))

    return a,b,c,av,bv,cv

def save_arrays(i, gal_ids, a, b, c, av, bv, cv, sim_name, snapNum, shape_type):
    # save measurements
    fpath = './data/shape_catalogs/'
    fname = sim_name + '_' + str(snapNum) + '_'+ shape_type +'_galaxy_shapes-%d.dat'%i
    ascii.write([gal_ids, a, b, c, av[:,0], av[:,1], av[:,2], bv[:,0], bv[:,1], bv[:,2], cv[:,0], cv[:,1], cv[:,2]], fpath+fname,
        names=['gal_id', 'a', 'b', 'c',
        'av_x', 'av_y','av_z',
        'bv_x', 'bv_y','bv_z',
        'cv_x', 'cv_y','cv_z'],
        overwrite=True)

def check_output_exists(i, sim_name, snapNum, shape_type):
    # check whether the text file this galaxy should be saved in already exists
    fpath = './data/shape_catalogs/'
    fname = sim_name + '_' + str(snapNum) + '_'+ shape_type +'_galaxy_shapes-*.dat'
    files = glob.glob(fpath+fname)

    if len(files)==0:
        exists = False
    else:
        existing_indices = [int(f.split('shapes-')[1].replace('.dat','')) for f in files]
        jmax = np.max(existing_indices)
        if (i<=jmax):
            exists = True
        else:
            exists = False

    #import pdb ; pdb.set_trace()

    return exists



def main():

    if len(sys.argv)>1:
        sim_name = sys.argv[1]
        snapNum = int(sys.argv[2])
        shape_type = sys.argv[3]
    else:
        sim_name = 'Illustris-1' # full physics high-res run
        snapNum = 135  # z=0
        shape_type = 'reduced'  # non-reduced, reduced, iterative

    # get simulation properties
    d = sim_prop_dict[sim_name]
    basePath = d['basePath']
    m_dm = d['m_dm']
    litte_h = d['litte_h']
    Lbox = d['Lbox']

    # make galaxy selection
    min_mstar = litte_h*10.0**8.
    min_ndark = 1000
    mask, gal_ids = galaxy_selection(min_mstar, min_ndark, basePath, snapNum)

    # number of galaxies in selection
    Ngals = len(gal_ids)
    print("number of galaxies in selection: {0}".format(Ngals))

    # load galaxy table
    fields = ['SubhaloGrNr', 'SubhaloMassInRadType', 'SubhaloPos', 'SubhaloHalfmassRadType']
    galaxy_table = loadSubhalos(basePath, snapNum, fields=fields)

    a,b,c,av,bv,cv = reset_arrays(Ngals)
    save_points = np.append(np.arange(1000,Ngals,2000), Ngals-1)
    print('Will save output over %d files'%len(save_points))

    # loop over the list of galaxy IDs
    for i in tqdm(range(Ngals)):
        if check_output_exists(i, sim_name, snapNum, shape_type):
            continue

        gal_id = gal_ids[i]
        evals, evecs = galaxy_shape(gal_id, galaxy_table, basePath, snapNum, Lbox, shape_type=shape_type)
        a[i] = evals[2]
        b[i] = evals[1]
        c[i] = evals[0]
        av[i,:] = evecs[:,2]
        bv[i,:] = evecs[:,1]
        cv[i,:] = evecs[:,0]

        if i in save_points:
            save_arrays(i, gal_ids, a, b, c, av, bv, cv, sim_name, snapNum, shape_type)
            print('%d saved %d galaxies'%(i, len(a[a!=0])))
            a,b,c,av,bv,cv = reset_arrays(Ngals)


if __name__ == '__main__':
    main()
