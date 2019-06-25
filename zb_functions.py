import numpy as np
import h5py
import sim_tools as st
import matplotlib
from matplotlib import cm
from astropy.io import ascii

import matplotlib.pyplot as plt
import time
from matplotlib.backends.backend_pdf import PdfPages

matplotlib.use('pdf')

matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['font.serif'][0] = 'palatino'

# cluster_index = 0
far = 40
close = 10
list = ['000_z014p003', '001_z006p772', '002_z004p614', '003_z003p512', '004_z002p825', '005_z002p348',
        '006_z001p993', '007_z001p716', '008_z001p493', '009_z001p308', '010_z001p151', '011_z001p017',
        '012_z000p899', '013_z000p795', '014_z000p703', '015_z000p619', '016_z000p543', '017_z000p474',
        '018_z000p411', '019_z000p366', '020_z000p352', '021_z000p297', '022_z000p247', '023_z000p199',
        '024_z000p155', '025_z000p113', '026_z000p101', '027_z000p073', '028_z000p036', '029_z000p000']


# Reads the IDs of the galaxies starting from the FirstSubhaloID
# returns matrix of ranges
def ReadSubhaloIDs(subdir):
    print('\nReading IDs...\n')
    FirstSubhaloID = st.eagleread(subdir, 'FOF/FirstSubhaloID', astro=False, silent=True)

    # Makes a list of Subhalo IDs
    IDs = []
    # print(FirstSubhaloID.shape)
    for i in range(len(FirstSubhaloID)-1):
        IDs.append(range(FirstSubhaloID[i], FirstSubhaloID[i+1]))

    # print(IDs)
    return IDs


# Reads CE-0/HYDRO/highlev/SpiderwebTables.hdf5/Subhalo/Snapshot_029/Galaxy
# = galaxyID corresponding to each subhalo
def read_galaxyIDs(simdir, snap):
    # Reads the GalaxyIDs from Spiderweb tables
    print('\nReading SpiderWebTables to get GalaxyIDs...')
    GalaxyIDs = h5py.File(simdir + 'highlev/SpiderwebTables.hdf5', 'r')
    # print(GalaxyIDs)
    # print(GalaxyIDs['Subhalo/Snapshot_029/Galaxy'])
    Gals = GalaxyIDs['Subhalo/Snapshot_'+snap[:3]+'/Galaxy'][()]
    # print(len(Gals))
    # print(Gals[2])
    return Gals


# Reads highlev/FullGalaxyTables.hdf5/RelRadius for cluster_index
def ReadRelRadius(simdir, snap):
    print('\nReading RelRadius..')
    snapnum = int(snap[1:3])
    RelRadius = h5py.File(simdir+'highlev/FullGalaxyTables.hdf5', 'r')['RelRadius'][:, snapnum]
    # RelRadius = RelRadius_full['RelRadius'][:, cluster_index]
    # print(RelRadius[19])
    return RelRadius


# Reads highlev/SubhaloExtra.hdf5/Snapshot_029/BoundaryFlag
# returns  1) bool indices where BoundaryFlag < 2 i.e. object is not too close to a boundary particle
#          2) the flags themselves (numbers from 0 to 4)
def ReadFlags(simdir, snap):
    print('\nReading Flags...')
    flags = h5py.File(simdir+'highlev/SubhaloExtra.hdf5', 'r')
    flags = flags['Snapshot_'+snap[:3]+'/BoundaryFlag'][()]
    # print(flags.shape)
    # print(flags[653361:653393])
    # print(flags[np.where(flags < 2)])
    npflags = np.array(flags)
    # print(flags)
    # print(npflags)
    # print(npflags[np.where(npflags < 2)])
    return np.where(flags < 2)[0], flags


# Reads subdir/Subhalo/CentreOfMass - 3 components, takes diff with central halo
# and computes d = sqrt(x**2+y**2+z**2);        astro = True
def ReadDist(subdir):
    # if all:
    print('\nReading distance...')
    dist_read = st.eagleread(subdir, 'Subhalo/CentreOfMass', astro=True, silent=True)[0]
    # else:
    #     GalaxyPositions = h5py.File(simdir + 'highlev/GalaxyPositionsSnap.hdf5')['Centre']
    #     dist_read = GalaxyPositions[galID, :]
    #     central = GalaxyPositions[760, :]

    distx = dist_read[:, 0] - dist_read[0, 0]
    disty = dist_read[:, 1] - dist_read[0, 1]
    distz = dist_read[:, 2] - dist_read[0, 2]
    dist = np.sqrt(distx**2+disty**2+distz**2)
    # print(len(dist))
    return dist


# ---------- THE SEMI-DUMB WAY TO SELECT: ----------
def GetInd(close,far,subdir):
    dist = ReadDist(subdir)
    N = len(dist)
    ind, flags = ReadFlags(simdir, snap)

    ind_far = np.where(dist > far)[0]       # indices of far objects
    ind_close = np.where(dist < close)[0]   # indices of close objects

    far_ind_flagged = np.zeros(N)
    close_ind_flagged = np.zeros(N)

    # print('Far, close, all:\n', len(ind_far), len(ind_close), N, '\n')

    for i in ind_far:
        if flags[i] < 2:
            far_ind_flagged[i] = int(1)    # if i in far and flag < 2, take it

    for i in ind_close:
        if flags[i] < 2:
            close_ind_flagged[i] = int(1)

    # Close flags test: 'close' objects should be far from boundary. If not, something is wrong.
    if int(len(np.where(close_ind_flagged == 1)[0])) - int(len(ind_close)) == 0:
        print('\nCLOSE FLAGS TEST PASSED!')
    else:
        print('\nWARNING: close have bad flags\n    ', len(ind_close)-len(np.where(close_ind_flagged == 1)[0]))

    return np.where(close_ind_flagged == 1)[0], np.where(far_ind_flagged == 1)[0]


# Reads subdir/Subhalo/MassType - DM mass and Stellar mass.
# returns log_DM and log_SM;                    astro = True
# take only Mdm > 8;    Mstar > 7
def ReadMasses(subdir):
    print('\nReading masses...')
    mass_types = st.eagleread(subdir, 'Subhalo/MassType', astro=True, silent=True)[0]

    mass_DM = mass_types[:, 1]
    mass_stars = mass_types[:, 4]

    mass_DM_log = np.log10(mass_DM) + 10
    mass_stars_log = np.log10(mass_stars) + 10

    # mass_DM_log = mass_DM_log[np.where(mass_DM_log > 8)]
    # mass_stars_log = mass_stars_log[np.where(mass_stars_log > 7)]

    return mass_DM_log, mass_stars_log


# Given lower boundaries Mstar and Mdm, gives list of gals within
# limits + 0.1 (log space);     returns (ind, len(ind))
def ExtractGalaxy(simdir, subdir, Mstar, Mdm):
    logDM, logSM = ReadMasses(subdir)
    # i, flags = ReadFlags(simdir)
    N = len(logSM)
    ind = np.zeros(N)

    # ----- SEMI-DUMB, CAN BE IMPROVED -----
    ind[np.where(logSM > Mstar)[0]] = 1
    ind[np.where(logSM > (Mstar+0.1))[0]] = 0

    # print('\n\n', len(np.where(ind ==1)[0]))

    ind[np.where(logDM > (Mdm+0.1))[0]] = 0
    ind[np.where(logDM < Mdm)[0]] = 0
    ind[np.where(flags > 1)[0]] = 0

    ind = np.where(ind == 1)[0]
    print('\n\nFound ' + str(len(ind)) + ' galaxies in the interval (Mstar, Mdm) = (' + str(Mstar) + ', ' + str(Mdm) + ') + 0.1.\n')

    # print(logDM[ind], logSM[ind])
    return ind, len(ind)


# Reads highlev/FullGalaxyTables.hdf5/SFR for cluster_index
def ReadSFR(simdir, snap):
    print('\nReading SFR...')
    snapnum = int(snap[1:3])
    SFR = h5py.File(simdir + 'highlev/FullGalaxyTables.hdf5', 'r')['SFR'][:, snapnum]
    return SFR


# Returns indices of galaxies with logDM > logSM
def SelectDMdeprived(logDM, logSM, flags):
    N = len(logDM)
    a = np.zeros(N)
    a[np.where(logSM > logDM)[0]] = 1
    for i in np.where(a == 1)[0]:
        if flags[i] > 1:
            a[i] = 0
    return np.where(a == 1)[0]


def ReadMaxMass(simdir):
    full = h5py.File(simdir + 'highlev/FullGalaxyTables.hdf5', 'r')['Full']
    return full['MDM'], full['Mstar']


def MakeHistOfFracs(filename, cluster_index=18, show=True):
    f = np.loadtxt(filename, unpack=True, delimiter=',')
    plt.figure()
    plt.bar(f[0], f[1]*100)
    plt.xlabel('Snapshot')
    # plt.xlabel('Cluster index')
    plt.ylabel('Fraction of galaxies with $M_{dm} < M_{stars}$, %')
    # plt.title('All clusters, snapshot 29; only BoundaryFlag < 2\n$M_{star}>7$, $M_{dm}>8$')
    plt.title('Cluster index: ' + str(cluster_index))
    if show:
        plt.show()
    plt.savefig('18/fracs_18.png')
    plt.savefig('18/fracs_18.pdf')
    print('\nHistogram done!')


# -------------- FIND PROPERTIES OF THIS SPECIFIC GALAXY ---------------
def write_mass(galID, M_star, M_dm, dist, d):
    m = open('masses-'+str(galID)+'.txt', 'a')
    for i in range(30):
        m.write('{:d}, {:f}, {:f}, {:f}, {:f}\n'.format(i, M_star[i], M_dm[i], dist[i], d[i]))
    m.close()
    print('Masses file written!')


# Calculate distance of galID from central for all snapshots
def extract_dist(simdir, galID):
    positions = h5py.File(simdir+'highlev/GalaxyPositionsSnap.hdf5', 'r')['Centre']
    # c = positions[760]
    galaxy_position = positions[galID]
    dist = np.zeros(30)
    for i in range(30):
        centralID = read_galaxyIDs(simdir, snap=list[i])[0]
        c = positions[centralID]
        print(centralID)
        if galaxy_position[i, 0] is not -np.inf:
            dx = galaxy_position[i, 0] - c[i, 0]
            dy = galaxy_position[i, 1] - c[i, 1]
            dz = galaxy_position[i, 2] - c[i, 2]
            dist[i] = np.sqrt(dx*dx + dy*dy + dz*dz)
    return dist


# Extract the particles of a given galID; must be done by SHI
# -> Add galID - SHI convertion
def extract_particles(galID, SHI, subdir):
    SubLength = st.eagleread(subdir, 'Subhalo/SubLength', astro=False, silent=True)
    SubLengthType = st.eagleread(subdir, 'Subhalo/SubLengthType', astro=False, silent=True)
    SubOffset = st.eagleread(subdir, 'Subhalo/SubOffset', astro=False, silent=True)
    particleIDs = st.eagleread(subdir, 'IDs/ParticleID', astro=False, silent=True)

    length = SubLength[SHI]
    type = SubLengthType[SHI]
    offset = SubOffset[SHI]
    particles = particleIDs[offset:offset+length]

    dm_particles = particles[np.where(particles % 2 == 0)[0]]
    star_particles = particles[np.where(particles % 2 == 1)[0]]
    return type, dm_particles, star_particles


def extract_coords_of_particles(simdir, dm, stars):
    beg = time.time()
    star_IDs = st.eagleread(simdir+'data/snapshot_029_z000p000/snap_029_z000p000.0.hdf5', 'PartType4/ParticleIDs', astro=False, silent=True)
    print('read IDs', len(star_IDs))
    # star_coords = star['Coordinates']
    # star_mass = star['Mass']
    # star_IDs = star['IDs']

    ind = np.zeros(len(star_IDs))
    for i in range(len(star_IDs)):
        if star_IDs[i] in stars:
            ind[i] = 1
        mid = time.time()
        if mid - beg > 60:
            print(i)
            break
    print(ind)
    print(stars)
    print(star_IDs[ind])

    end = time.time()
    print(end-beg)


def ThePlot_all(logDM, logSM, cmap, ind, close_logDM, close_logSM, far_logDM, far_logSM, closecount, farcount,
            Mstar, Mdm, frac, no, conv, extract=True, zoomin=True, show=True):
    plt.figure(figsize=(8, 6))
    # plt.scatter(logDM, logSM, s=1, c='k', label='BoundaryFlag > 1')
    plt.scatter(logDM[ind], logSM[ind], s=4, c=cmap[conv], cmap='Greys')
    # plt.scatter(logDM[ind], logSM[ind], s=4, c=cmap[conv], cmap='coolwarm_r')
    # plt.plot(close_logDM, close_logSM, 'xr', alpha=0.3, label='close, <' + str(close) + ' Mpc')
    # plt.plot(far_logDM, far_logSM, 'xb',  alpha=0.8, label='far, > ' + str(far) + ' Mpc')
    # plt.title('Cluster ' + str(cluster_index) + ', close/far ratio = ' + str(int(closecount/farcount)) + ', no. far = ' + str(farcount))
    plt.title('Cluster ' + str(cluster_index) + ', snap ' + snap[:3] + ', fraction of DM-lacking = {:.3f}%, '
              .format(frac*100) + 'total no. = ' + str(no))
    plt.xlabel('log($M_{dark}/M_{\odot}$)')

    ylim = [7, 12]
    xlim = [8, 14]
    if extract:
        plt.axhline(Mstar, c='k', linewidth=1)
        plt.axhline(Mstar + 0.1, c='k', linewidth=1)
        plt.axvline(Mdm, c='k', linewidth=1)
        plt.axvline(Mdm+0.1, c='k', linewidth=1)

    if zoomin:
        plt.xlim(Mdm-1, Mdm+1)
        plt.ylim(Mstar-1, Mstar+1)
    else:
        plt.xlim(9, 14)
        plt.ylim(7.5, 12)

    plt.plot(ylim, ylim, c='g', linewidth=1, label='y = x')
    plt.ylabel('log($M_{stars}/M_{\odot}$)')
    plt.legend()
    # cbar = plt.colorbar()
    # cbar.ax.set_ylabel('SFR')
    # cbar.ax.set_ylabel('Distance from nearest more massive neighbour, [$R_vir, massive$]')
    # cbar.ax.set_ylabel('Distance from centre, Mpc')
    # plt.tight_layout()
    # plt.savefig('StarsDM_CentrDist'+str(cluster_index)+'_flagged_all_tests.png')

    m_2372 = np.loadtxt('masses-2373.txt', delimiter=',', unpack=True)
    # plt.plot(m_2372[2], m_2372[1], color='navy')
    d = extract_dist(simdir, 2373)
    plt.plot(m_2372[2], m_2372[1], color='k', linewidth=1, linestyle='--')
    plt.scatter(m_2372[2], m_2372[1], marker='*', c=d, cmap='plasma')
    plt.colorbar()

    # l = np.loadtxt('lessDM.txt', unpack=True)
    # logDM = l[1, :]
    # logSM = l[2, :]
    # maxDM = l[3, :]
    # maxSM = l[4, :]
    #
    # for i in range(len(logDM)):
    #     plt.plot([logDM[i], maxDM[i]], [logSM[i], maxSM[i]], '*', c='k')
    #     plt.plot([logDM[i], maxDM[i]], [logSM[i], maxSM[i]], c='maroon', linestyle='--')

    if show:
        plt.show()
    plt.savefig('stripping_dist.png', dpi=300)
    # plt.savefig('stripping.pdf')
    print('\nPLOTTING DONE!')


# RelRadius = ReadRelRadius(simdir, cluster_index)
# SFR = ReadSFR(simdir, cluster_index)
# SFR = SFR[np.where(SFR != -1)[0]]
# print(len(SFR[np.where(SFR == -np.inf)[0]]))
# SFR[np.where(SFR == -np.inf)[0]] = min(SFR[np.where(SFR != -np.inf)[0]])
# print(min(SFR[np.where(SFR != -np.inf)[0]]))
# print(len(SFR[np.where(SFR != -np.inf)[0]]))

# print('\n', len(dist), len(np.where(RelRadius < 1000)[0]), '\n')
# print('\n', len(dist), len(SFR), '\n')

# f = open("18/fracs.txt", "a")


# for cluster_index in (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 18, 21, 22, 24, 25, 28, 29):
# for snap in list:
for cluster_index in [29]:
    # cluster_index = 29
    snap = '029_z000p000'
    simdir = '/net/quasar/data3/Hydrangea/CE-' + str(cluster_index) + '/HYDRO/'
    subdir = simdir + '/data/groups_'+snap+'/eagle_subfind_tab_'+snap+'.0.hdf5'
    dist = ReadDist(subdir)
    logDM, logSM = ReadMasses(subdir)
    # logDM = logDM[np.where(logDM > 8)[0]]
    # logSM = logSM[np.where(logSM > 7)[0]]
    ind, flags = ReadFlags(simdir, snap)
    # print(ind)
    # print(flags)
    close_ind, far_ind = GetInd(close, far, subdir)

    closecount = len(close_ind)
    farcount = len(far_ind)

    close_logDM = logDM[close_ind]
    close_logSM = logSM[close_ind]

    far_logDM = logDM[far_ind]
    far_logSM = logSM[far_ind]

    print('close stars: ', len(dist[np.where(dist < close)]), len(close_ind))

    Mstar = 9.8
    Mdm = 9.30
    SHI, length = ExtractGalaxy(simdir, subdir, Mstar, Mdm)
    GalaxyIDs = read_galaxyIDs(simdir, snap)

    a = SelectDMdeprived(logDM, logSM, flags)
    a = a[np.where(logDM[a] > 8)[0]]
    a = a[np.where(logSM[a] > 7)[0]]
    # print(a)
    a = a[np.where(logDM[a] != -np.inf)[0]]

    print(a)    # SUBHALO INDEX!

    maxMdm, maxMstar = ReadMaxMass(simdir)

    # lessDM = GalaxyIDs[a]
    # print(lessDM)
    # l = open('lessDM.txt', 'a')
    # for i in range(len(a)):
    #     ID = GalaxyIDs[a[i]]
    #     SHI = a[i]
    #     l.write('{:d} {:f} {:f} {:f} {:f}\n'.format(ID, logDM[SHI], logSM[SHI], maxMdm[ID], maxMstar[ID]))
    # l.close()

    no = len(np.where(logDM[a] != -np.inf)[0])
    # print(no)
    # print(ind)
    print('Infinite DM: ', len(np.where(logDM == -np.inf)[0]))
    print('Non-nfinite DM: ', len(np.where(logDM != -np.inf)[0]))
    print('Infinite SM: ', len(np.where(logSM == -np.inf)[0]))
    print('Non-nfinite SM: ', len(np.where(logSM != -np.inf)[0]))
    frac = no/len(ind)
    print('Cluster: ', cluster_index, 'all DM: ', len(logDM), 'all SM: ', len(logSM), 'all ind:', len(ind),
          '\nNon-infinite DM among y > x: ', no, 'Frac of non-inf: ', frac, '\n')

    # RelRadius = ReadRelRadius(simdir, snap)
    # print(len(RelRadius), len(np.where(RelRadius < 1000)[0]))
    # print('Selected galaxy IDs: ', GalaxyIDs[a])
    # RelRadSelected = RelRadius[GalaxyIDs[a]]
    # print(len(RelRadSelected), len(a))
    # print(RelRadSelected[:10])

    print('Subhalo index = ', SHI)
    print('Galaxy index = ', GalaxyIDs[SHI])
    print('Mstar, Mdm = ', logSM[SHI], logDM[SHI])
    print('dist = ', dist[SHI])
    # print('RelRad = ', RelRadius[GalaxyIDs[SHI]])

    # f = open("18/fracs.txt", "a")
    # f.write('{:d}, {:f} \n'.format(cluster_index, frac))
    # f.write('{:d}, {:f} \n'.format(int(snap[:3]), frac))

    # print('\na, len DM deprived ', a, len(a))
    # print('their masses: DM/SM \n', logDM[a], '\n', logSM[a])
    # print('mins: DM/SM\n', min(logDM[a]), '\n', min(logSM[a]), '\nlen(DM != -inf) = ', len(np.where(logDM[a] != -np.inf)[0]))
    # print('frac of non-inf DM-deprived: ', frac)

    # ThePlot(logDM, logSM, dist, ind, close_logDM, close_logSM, far_logDM, far_logSM, closecount, farcount,
    #         Mstar, Mdm, frac, no, extract=True, zoomin=False, show=True)

    # print(a)
    # print(GalaxyIDs, len(GalaxyIDs))
    # print(RelRadius, len(RelRadius))

    ind = ind[np.where(logDM[ind] != -np.inf)[0]]
    # ind = ind[np.where(RelRadius[GalaxyIDs[ind]] < 1000)[0]]

    ThePlot_all(logDM, logSM, dist, ind, close_logDM, close_logSM, far_logDM, far_logSM, closecount, farcount,
        Mstar, Mdm, frac, no, conv=ind, extract=False, zoomin=False, show=True)


    # # ------------- HISTOGRAMS ------------------
    # # plt.figure()
    # plt.hist(logDM[np.where(logDM != -np.inf)[0]], bins=20, color='grey', log=True, cumulative=True)
    # plt.title('Dark matter')
    # plt.ylabel('counts')
    # plt.xlabel('log($M_{dark}/M_{\odot}$)')
    # plt.show()
    #
    # # plt.figure()
    # plt.hist(logSM[np.where(logSM != -np.inf)[0]], log=True, bins=20, color='orange', alpha=0.5, cumulative=True)
    # plt.title('Stellar matter')
    # plt.ylabel('counts')
    # plt.xlabel('log($M_{stars}/M_{\odot}$)')
    # plt.show()

    # m_2372 = np.loadtxt('masses-2373.txt', delimiter=',', unpack=True)
    # plt.plot(m_2372[2], m_2372[1], 'x', color='k')

    # ThePlot(logDM, logSM, RelRadius, ind, close_logDM, close_logSM, far_logDM, far_logSM, closecount, farcount,
    #     Mstar, Mdm, frac, no, conv=GalaxyIDs[ind], extract=False, zoomin=False, show=True)

exit()

# MakeHistOfFracs('18/fracs.txt', show=False)
print('\n')

cluster_index = 29
snap = list[-1]
simdir = '/net/quasar/data3/Hydrangea/CE-' + str(cluster_index) + '/HYDRO/'
subdir = simdir + '/data/groups_'+snap+'/eagle_subfind_tab_'+snap+'.0.hdf5'
dist_2373 = extract_dist(simdir, 2373)
print(dist_2373)

print('\n')

# types, dm_particles_2373, star_particles_2373 = extract_particles(galID=2373, SHI=1066, subdir=subdir)

# particles = open('particles.txt', 'a')
# for i in range(len(dm_particles_2373)):
#     particles.write('{:f}\n'.format(dm_particles_2373[i]))
#
# print('DM written')
#
# for i in range(len(star_particles_2373)):
#     particles.write('{:f}\n'.format(star_particles_2373[i]))
# particles.close()

IDs_2373 = np.loadtxt('particles.txt', unpack=True)
IDs_2373 = IDs_2373.astype(int)

dm = IDs_2373[np.where(IDs_2373 % 2 == 0)[0]]
stars = IDs_2373[np.where(IDs_2373 % 2 == 1)[0]]

extract_coords_of_particles(simdir, dm, stars)

# extract_coords_of_particles(simdir, dm_particles_2373, star_particles_2373)

# subdir = ''
#
# galID = 2373
# SHI = 1066
# print('Reading hdf5 file...')
# file = h5py.File(simdir+'highlev/FullGalaxyTables.hdf5', 'r')
# M_star = file['Mstar'][galID, :]
# M_dm = file['MDM'][galID, :]
# dist1 = [0]*30
# dist2 = [0]*30

# print('Reading galaxy positions...')
# dist = ReadDist(subdir, simdir, gal=galID, all=False)
# print(dist)

# i=0
# for snap in list:
#     subdir = simdir + '/data/groups_' + snap + '/eagle_subfind_tab_' + snap + '.0.hdf5'
#     distances = ReadDist(subdir, simdir, gal=None, all=True)
#     dist1[i] = distances[SHI]
#     dist2[i] = distances[galID]
#     print(dist1[i], dist2[i])
#     i += 1
#
# write_mass(galID, M_star, M_dm, dist1, dist2)
