import numpy as np
import h5py
import sim_tools as st
import yb_utils as yb
import hydrangea_tools as ht
from astropy.io import ascii


def gaussian(x, mu, sigma):
    return np.exp(-(x-mu)**2/2/sigma**2)/np.sqrt(2*np.pi*sigma*sigma)


def weight_function(origin, position, R):
    x = abs(position[0]-origin[0])
    y = abs(position[1]-origin[1])
    z = abs(position[2]-origin[2])
    d = np.sqrt(x*x+y*y+z*z)
    return gaussian(d, 0, R)


def find_CoM(origin, pos, mass, R, ptype):
    numer = np.array((0, 0, 0))
    denom = np.array((0, 0, 0))

    wmax = weight_function(origin, origin, R)
    print(wmax)

    if ptype == 1:
        m_dm = mass[0]
        for i in range(len(pos)):
            numer = numer + np.array(pos[i])*m_dm*weight_function(origin, pos[i], R)
            denom = denom + m_dm * weight_function(origin, pos[i], R)
            # print('{:1.6f}'.format(weight_function(origin, pos[i], R) / wmax))
    else:
        for i in range(len(pos)):
            numer = numer + np.array(pos[i])*mass[i]*weight_function(origin, pos[i], R)
            denom = denom + mass[i]*weight_function(origin, pos[i], R)
            # print(weight_function(origin, pos[i], R)/wmax)

    return numer/denom


def CoM_auto(galID, ptype):
    # ----------------------------------------

    cluster_index = 29
    plot_snap = 29  # Which snapshot to plot
    imsize = 0.01  # (Half-)image size in pMpc

    # ---------------------------------------

    rundir = '/net/quasar/data3/Hydrangea/CE-' + str(cluster_index) + '/HYDRO/'

    posloc = rundir + '/highlev/GalaxyPositionsSnap.hdf5'
    spiderloc = rundir + '/highlev/SpiderwebTables.hdf5'
    fgtloc = rundir + '/highlev/FullGalaxyTables.hdf5'

    snaplistloc = rundir + "sneplists/allsnaps.dat"
    snaplist = ascii.read(snaplistloc)
    aexpSnap = np.array(snaplist['aexp'])

    # Shortcut of loading the positions of all galaxies
    pos_allGal = yb.read_hdf5(posloc, "Centre")[:, plot_snap, :]
    pos_gal = pos_allGal[galID, :]

    # Slightly lazy way to set comoving <-> proper conversion factor
    aexp_factor = aexpSnap[plot_snap]
    h_factor = 1 / 0.6777
    conv_astro_pos = h_factor * aexp_factor

    print("Determined galaxy position as {:.3f}/{:.3f}/{:.3f}..."
          .format(*pos_gal))

    # snapdir = st.form_files(rundir, isnap=plot_snap, types='sub')
    snapdir = st.form_files(rundir, isnap=plot_snap, types='snap')

    readReg = ht.ReadRegion(snapdir, ptype, [*pos_gal / conv_astro_pos, imsize * np.sqrt(3) / conv_astro_pos])

    # Read position and mass of particles in target region
    pos = readReg.read_data("Coordinates", astro=True)

    if ptype != 1:
        mass = readReg.read_data("Mass", astro=True)
        ind = np.loadtxt('output/indexing_array_stars_' + str(galID) + '.txt', unpack=True).astype(int)
    else:
        mass = np.zeros(pos.shape[0]) + st.m_dm(snapdir, issnapdir=True, astro=True)
        ind = np.loadtxt('output/indexing_array_'+str(galID)+'.txt', unpack=True).astype(int)

    # print(pos[ind])
    # print(mass[ind])

    pos = pos[ind]
    mass = mass[ind]

    R = h5py.File(fgtloc, 'r')['StellarHalfMassRad']
    R = R[galID, plot_snap]  # stellar half-mass radius of 2373, pMpc

    CoM = find_CoM(pos_gal, pos, mass, R, ptype)
    print('\n')
    print('CoM \t', CoM)
    print('origin \t', pos_gal)
    print('\n')

    d = abs(pos_gal - CoM)
    displacement = np.sqrt(sum(d**2))

    print('CoM displacement: [pkpc]\t', displacement*1e6)
    print('\n')


CoM_auto(galID=4452, ptype=4)
