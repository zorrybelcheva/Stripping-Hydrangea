import numpy as np
import h5py
import sim_tools as st
import matplotlib.pyplot as plt
import time

start = time.time()

global snaplist
snaplist = ['000_z014p003', '001_z006p772', '002_z004p614', '003_z003p512', '004_z002p825', '005_z002p348',
            '006_z001p993', '007_z001p716', '008_z001p493', '009_z001p308', '010_z001p151', '011_z001p017',
            '012_z000p899', '013_z000p795', '014_z000p703', '015_z000p619', '016_z000p543', '017_z000p474',
            '018_z000p411', '019_z000p366', '020_z000p352', '021_z000p297', '022_z000p247', '023_z000p199',
            '024_z000p155', '025_z000p113', '026_z000p101', '027_z000p073', '028_z000p036', '029_z000p000']


# Reads subdir/Subhalo/MassType - DM mass and Stellar mass.
# returns log_DM and log_SM;                    astro = True
# take only Mdm > 8;    Mstar > 7
def read_masses(subdir):
    print('\nReading masses...')
    mass_types = st.eagleread(subdir, 'Subhalo/MassType', astro=True, silent=True)[0]

    mass_DM = mass_types[:, 1]
    mass_stars = mass_types[:, 4]

    mass_DM_log = np.log10(mass_DM) + 10
    mass_stars_log = np.log10(mass_stars) + 10

    return mass_DM_log, mass_stars_log


# Reads subdir/Subhalo/CentreOfMass - 3 components, takes diff with central halo
# and computes d = sqrt(x**2+y**2+z**2);        astro = True
def read_dist(subdir):
    print('\nReading distance...')
    dist_read = st.eagleread(subdir, 'Subhalo/CentreOfMass', astro=True, silent=True)[0]
    distx = dist_read[:, 0] - dist_read[0, 0]
    disty = dist_read[:, 1] - dist_read[0, 1]
    distz = dist_read[:, 2] - dist_read[0, 2]
    dist = np.sqrt(distx**2+disty**2+distz**2)
    return dist


# Reads CE-0/HYDRO/highlev/SpiderwebTables.hdf5/Subhalo/Snapshot_029/Galaxy
# = galaxyID corresponding to each subhalo
def read_galaxyIDs(simdir, snap, central=False):
    # Reads the GalaxyIDs from Spiderweb tables
    # print('\nReading SpiderWebTables to get GalaxyIDs...')
    GalaxyIDs = h5py.File(simdir + 'highlev/SpiderwebTables.hdf5', 'r')

    if central:
        return GalaxyIDs['Subhalo/Snapshot_'+snap[:3]+'/Galaxy'][0]
    else:
        Gals = GalaxyIDs['Subhalo/Snapshot_'+snap[:3]+'/Galaxy'][()]
        return Gals


# Reads highlev/SubhaloExtra.hdf5/Snapshot_029/BoundaryFlag
# returns  1) bool indices where BoundaryFlag < 2 i.e. object is not too close to a boundary particle
#          2) the flags themselves (numbers from 0 to 4)
def read_flags(simdir, snap):
    print('\nReading Flags...')
    flags = h5py.File(simdir+'highlev/SubhaloExtra.hdf5', 'r')
    flags = flags['Snapshot_'+snap[:3]+'/BoundaryFlag'][()]
    return np.where(flags < 2)[0], flags


# Calculate distance of galID from central for all snapshots
def extract_dist(simdir, galID, relativeGalID):
    positions = h5py.File(simdir+'highlev/GalaxyPositionsSnap.hdf5', 'r')['Centre']
    # SHI = h5py.File(simdir+'highlev/FullGalaxyTables.hdf5', 'r')['SHI']
    # spiderweb = h5py.File(simdir+'highlev/SpiderwebTables.hdf5', 'r')['Subhalo']
    # gal_SHI = SHI[galID]
    # rel_SHI = SHI[relativeGalID]

    dist = np.zeros(30)
    for i in range(30):
        gal_pos = positions[galID]
        rel_gal_pos = positions[relativeGalID]
        # print(centralID)
        if gal_pos[i, 0] is not -np.inf:
            dx = gal_pos[i, 0] - rel_gal_pos[i, 0]
            dy = gal_pos[i, 1] - rel_gal_pos[i, 1]
            dz = gal_pos[i, 2] - rel_gal_pos[i, 2]
            dist[i] = np.sqrt(dx*dx + dy*dy + dz*dz)
    return dist


# Returns indices of galaxies with logDM > logSM
def SelectDMdeprived(logDM, logSM, flags):
    N = len(logDM)
    a = np.zeros(N)
    a[np.where(logSM > logDM)[0]] = 1
    for i in np.where(a == 1)[0]:
        if flags[i] > 1:
            a[i] = 0
    return np.where(a == 1)[0]


def ThePlot_stripping(galID, relGalID, cluster_index, snap, simdir, logDM, logSM, cmap, ind, frac, no, conv,
                      filename=None, Mstar=None, Mdm=None, extract=False, zoomin=False, show=True):
    plt.figure(figsize=(7, 6))
    plt.scatter(logDM[ind], logSM[ind], s=4, c=cmap[conv], cmap='Greys')
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
        plt.xlim(xlim)
        plt.ylim(ylim)

    plt.plot(ylim, ylim, c='g', linewidth=1, label='y = x')
    plt.ylabel('log($M_{stars}/M_{\odot}$)')
    plt.legend()

    m_2372 = np.loadtxt('output/masses-2373.txt', delimiter=',', unpack=True)
    d = extract_dist(simdir, galID, relGalID)
    plt.plot(m_2372[2], m_2372[1], color='k', linewidth=1, linestyle='--')
    plt.scatter(m_2372[2], m_2372[1], marker='*', c=d, cmap='plasma')
    cbar = plt.colorbar()
    cbar.ax.set_ylabel('Distance from galaxy ID = '+str(relGalID))
    plt.tight_layout()

    plt.savefig(filename, dpi=300)
    print('\nPLOTTING DONE!')
    print('\nSaved '+filename)

    if show:
        plt.show()


cluster_index = 29
snap = snaplist[-1]
simdir = '/net/quasar/data3/Hydrangea/CE-' + str(cluster_index) + '/HYDRO/'
subdir = simdir + '/data/groups_'+snap+'/eagle_subfind_tab_'+snap+'.0.hdf5'

# ----------------- DATA NEEDED -------------------------------------------
logDM, logSM = read_masses(subdir)
dist = read_dist(subdir)

ind, flags = read_flags(simdir, snap)
ind = ind[np.where(logDM[ind] != -np.inf)[0]]   # take away 0 DM masses

a = SelectDMdeprived(logDM, logSM, flags)
a = a[np.where(logDM[a] > 8)[0]]
a = a[np.where(logSM[a] > 7)[0]]
a = a[np.where(logDM[a] != -np.inf)[0]]

no = len(np.where(logDM[a] != -np.inf)[0])
frac = no/len(ind)


galID = 2373
relGalID = 163379
filename = '/home/belcheva/Desktop/str-relto'+str(relGalID)+'.png'

d = extract_dist(simdir, galID, relGalID)
print(d)
print(np.argmax(d))

ThePlot_stripping(galID, relGalID, cluster_index, snap, simdir, logDM, logSM,
                  cmap=dist, ind=ind, frac=frac, no=no, conv=ind, filename=filename)

end = time.time()

print('Time elapsed: ', end-start)
