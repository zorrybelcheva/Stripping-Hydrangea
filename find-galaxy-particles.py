"""
-> Find the particles that belong to a galaxy
-> save their IDs in a text file
-> save an indexing array that corresponds to the indices of the particles among the list of all.
"""

import numpy as np
import example_code.sim_tools as st
import example_code.yb_utils as yb
import example_code.hydrangea_tools as ht
import matplotlib
from astropy.io import ascii
import time

matplotlib.use('pdf')


matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['font.serif'][0] = 'palatino'


def find_particles(galID, ptype):
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

    shi = yb.read_hdf5(spiderloc, 'SubHaloIndex')[:, plot_snap]

    # Shortcut of loading the positions of all galaxies
    pos_allGal = yb.read_hdf5(posloc, "Centre")[:, plot_snap, :]
    pos_gal = pos_allGal[galID, :]

    # Slightly lazy way to set comoving <-> proper conversion factor
    aexp_factor = aexpSnap[plot_snap]
    h_factor = 1 / 0.6777
    conv_astro_pos = h_factor * aexp_factor

    print("Determined galaxy position as {:.3f}/{:.3f}/{:.3f}..."
          .format(*pos_gal))

    snapdir = st.form_files(rundir, isnap=plot_snap, types='snap')

    readReg = ht.ReadRegion(snapdir, ptype, [*pos_gal / conv_astro_pos, imsize * np.sqrt(3) / conv_astro_pos])
    IDs = readReg.read_data('ParticleIDs', astro=False)

    snap = '029_z000p000'
    simdir = '/net/quasar/data3/Hydrangea/CE-' + str(cluster_index) + '/HYDRO/'
    subdir = simdir + '/data/groups_'+snap+'/eagle_subfind_tab_'+snap+'.0.hdf5'
    SubLength = st.eagleread(subdir, 'Subhalo/SubLength', astro=False, silent=False)
    SubOffset = st.eagleread(subdir, 'Subhalo/SubOffset', astro=False, silent=False)
    particleIDs = st.eagleread(subdir, 'IDs/ParticleID', astro=False, silent=False)

    SHI = shi[galID]
    length = SubLength[SHI]
    offset = SubOffset[SHI]
    particles = particleIDs[offset:offset + length]

    np.savetxt('output/particles_'+str(galID)+'.txt', particles, fmt='%d')

    dm_particles = particles[np.where(particles % 2 == 0)[0]]
    star_particles = particles[np.where(particles % 2 == 1)[0]]

    if ptype == 1:
        print('Making index array, DM\n')
        ind = np.zeros(len(IDs))
        for i in range(len(IDs)):
            if IDs[i] in dm_particles:
                ind[i] = 1
        print('Index array DM done')

        ind = np.where(ind == 1)[0]

        index = open('output/indexing_array_'+str(galID)+'.txt', 'a')
        for i in range(len(ind)):
            index.write('{:d}\n'.format(ind[i]))

        index.close()
    else:
        print('Making index array, stars\n')
        ind = np.zeros(len(IDs))
        for i in range(len(IDs)):
            if IDs[i] in star_particles:
                ind[i] = 1
        print('Index array SM done')

        ind = np.where(ind == 1)[0]

        index = open('output/indexing_array_stars_'+str(galID)+'.txt', 'a')
        for i in range(len(ind)):
            index.write('{:d}\n'.format(ind[i]))

        index.close()


beg = time.time()

# ------------------------------------------------------
galIDs = np.loadtxt('output/lessDM.txt', unpack=True)[0]

for galID in galIDs[2:].astype(int):
    print(galID)
    find_particles(galID, ptype=1)
    find_particles(galID, ptype=4)

# ------------------------------------------------------

end = time.time()

print('Elapsed: ', end-beg, '\n')
