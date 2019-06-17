import numpy as np
import matplotlib.pyplot as plt
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


start = time.time()

# --------------------- PRELIMINARIES -----------------------------------
cluster_index = 29
plot_snap = 29
rundir = '/net/quasar/data3/Hydrangea/CE-' + str(cluster_index) + '/HYDRO/'
snapdir = st.form_files(rundir, isnap=plot_snap, types='snap')
posloc = rundir + '/highlev/GalaxyPositionsSnap.hdf5'

snaplistloc = rundir + "sneplists/allsnaps.dat"
snaplist = ascii.read(snaplistloc)
aexpSnap = np.array(snaplist['aexp'])

# Slightly lazy way to set comoving <-> proper conversion factor
aexp_factor = aexpSnap[plot_snap]
h_factor = 1 / 0.6777
conv_astro_pos = h_factor * aexp_factor

# ----------------------- SPECIFICS -------------------------------------
imsize = 0.01   # half image size in pMpc
ptype = 4
galID = 2373
R = 0.0012080314    # stellar half-mass radius of 2373, pMpc

if ptype == 1:
    ind = np.loadtxt('indexing_array.txt', unpack=True).astype(int)
else:
    ind = np.loadtxt('indexing_array_stars.txt', unpack=True).astype(int)

pos_allGal = yb.read_hdf5(posloc, "Centre")[:, plot_snap, :]
pos_gal = pos_allGal[galID, :]

readReg = ht.ReadRegion(snapdir, ptype, [*pos_gal / conv_astro_pos, imsize * np.sqrt(3) / conv_astro_pos])
pos = readReg.read_data("Coordinates", astro=True)

if ptype != 1:
    mass = readReg.read_data("Mass", astro=True)
else:
    mass = np.zeros(pos.shape[0]) + st.m_dm(snapdir, issnapdir=True, astro=True)

size = 1000
pos = pos[ind]
mass = mass[ind]
dist = np.zeros(size)
dx = np.zeros(size)
dy = np.zeros(size)
dz = np.zeros(size)

f = open('displacement_stars_trials.txt', 'a')

for i in range(size):
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
