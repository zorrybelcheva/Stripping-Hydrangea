import numpy as np
import matplotlib.pyplot as plt
import h5py
import time
import hydrangea_tools as ht
import sim_tools as st
import yb_utils as yb
from astropy.io import ascii


def bootstrap(sample):
    size = len(sample)
    boot = np.zeros(size)
    for i in range(size):
        rand = np.random.randint(0, size-1)
        boot[i] = sample[rand]
    return boot


def gaussian(x, mu, sigma):
    return np.exp(-(x-mu)**2/2/sigma**2)/np.sqrt(2*np.pi*sigma*sigma)


def weight_function(origin, position, R):
    x = abs(position[0]-origin[0])
    y = abs(position[1]-origin[1])
    z = abs(position[2]-origin[2])
    d = np.sqrt(x*x+y*y+z*z)
    return gaussian(d, 0, R)


def find_CoM(origin, pos, mass, R, ind, ptype):
    numer = np.array((0, 0, 0))
    denom = np.array((0, 0, 0))

    wmax = weight_function(origin, origin, R)
    # print(wmax)

    if ptype == 1:
        m_dm = mass[0]
        for i in range(len(ind)):
            numer = numer + np.array(pos[i])*m_dm*weight_function(origin, pos[i], R)
            denom = denom + m_dm * weight_function(origin, pos[i], R)
            # print('{:1.6f}'.format(weight_function(origin, pos[i], R) / wmax))
    else:
        for i in range(len(ind)):
            numer = numer + np.array(pos[i])*mass[i]*weight_function(origin, pos[i], R)
            denom = denom + mass[i]*weight_function(origin, pos[i], R)
            # print(weight_function(origin, pos[i], R)/wmax)

    return numer/denom


def bootstrap_CoM(galID, ptype, size):
    start = time.time()

    # --------------------- PRELIMINARIES -----------------------------------
    cluster_index = 29
    plot_snap = 29
    rundir = '/net/quasar/data3/Hydrangea/CE-' + str(cluster_index) + '/HYDRO/'
    snapdir = st.form_files(rundir, isnap=plot_snap, types='snap')
    posloc = rundir + '/highlev/GalaxyPositionsSnap.hdf5'
    fgtloc = rundir + '/highlev/FullGalaxyTables.hdf5'

    snaplistloc = rundir + "sneplists/allsnaps.dat"
    snaplist = ascii.read(snaplistloc)
    aexpSnap = np.array(snaplist['aexp'])

    # Slightly lazy way to set comoving <-> proper conversion factor
    aexp_factor = aexpSnap[plot_snap]
    h_factor = 1 / 0.6777
    conv_astro_pos = h_factor * aexp_factor

    # ----------------------- SPECIFICS -------------------------------------
    imsize = 0.01   # half image size in pMpc
    R = h5py.File(fgtloc, 'r')['StellarHalfMassRad']
    R = R[galID, plot_snap]  # stellar half-mass radius of 2373, pMpc

    pos_allGal = yb.read_hdf5(posloc, "Centre")[:, plot_snap, :]
    pos_gal = pos_allGal[galID, :]

    readReg = ht.ReadRegion(snapdir, ptype, [*pos_gal / conv_astro_pos, imsize * np.sqrt(3) / conv_astro_pos])
    pos = readReg.read_data("Coordinates", astro=True)

    if ptype != 1:
        mass = readReg.read_data("Mass", astro=True)
        ind = np.loadtxt('output/indexing_array_stars_' + str(galID) + '.txt', unpack=True).astype(int)
    else:
        mass = np.zeros(pos.shape[0]) + st.m_dm(snapdir, issnapdir=True, astro=True)
        ind = np.loadtxt('output/indexing_array_'+str(galID)+'.txt', unpack=True).astype(int)

    pos = pos[ind]
    mass = mass[ind]
    dist = np.zeros(size)
    dx = np.zeros(size)
    dy = np.zeros(size)
    dz = np.zeros(size)

    f = open('output/displacement_'+str(ptype)+'_'+str(galID)+'.txt', 'a')

    timecheck = time.time()
    k = 1

    for i in range(size):

        if i == size/100*k:
            print(str(k)+'%, '+str(time.time()-timecheck))
            timecheck = time.time()
            k += 1

        index = np.arange(len(pos))
        index = bootstrap(index).astype(int)

        CoM = find_CoM(pos_gal, pos[index], mass[index], R, ind, ptype)
        # print('\n')
        # print('CoM \t', CoM)
        # print('origin \t', pos_gal)
        # print('\n')

        d = abs(pos_gal - CoM)
        dist[i] = np.sqrt(sum(d ** 2))
        dx[i] = CoM[0] - pos_gal[0]
        dy[i] = CoM[1] - pos_gal[1]
        dz[i] = CoM[2] - pos_gal[2]

        f.write('{}\t{}\t{}\t{}\n'.format(dx[i], dy[i], dz[i], dist[i]))

    f.close()

    print(np.average(dist)*1000)
    print(np.std(dist)*1000)

    end = time.time()

    print('\nElapsed: ', end-start, '\n')


galIDs = np.loadtxt('output/lessDM.txt', unpack=True)[0].astype(int)

for galID in galIDs[1:]:
    if galID == 2373:
        pass
    else:
        beg = time.time()

        print(galID)

        print('\nBootstrapping DM...\n')
        bootstrap_CoM(galID=galID, ptype=1, size=5000)

        mid = time.time()
        print('DM took ', mid-beg)

        print('\nBootstrapping SM...\n')
        bootstrap_CoM(galID=galID, ptype=4, size=5000)

        end = time.time()

        print('SM took ', end-mid)
        print('Elapsed for '+str(galID)+': ' + str(end-beg))
