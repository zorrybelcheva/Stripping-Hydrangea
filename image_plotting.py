"""
Example code to generate a stellar mass (surface density) image
with SPH smoothing (optional).

Written 16-Nov-2018
"""

import numpy as np
import sim_tools as st
import yb_utils as yb
import hydrangea_tools as ht
import image_routines as ir
import matplotlib
from astropy.io import ascii
import matplotlib.pyplot as plt

matplotlib.use('pdf')


from matplotlib.backends.backend_pdf import PdfPages

matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['font.serif'][0] = 'palatino'

from pdb import set_trace

# ----------------------------------------

cluster_index = 29
mark_galaxies = True  # Optionally overplot location of galaxies
save_maps = False  # Store images as HDF5 as well as picture
desNGB = 32  # Number of neighbours for smoothing calculation
plot_snap = 29  # Which snapshot to plot

ptype = 4  # Particle type to plot (stars = 4)
imsize = 0.1  # (Half-)image size in pMpc
min_mpeak_otherGal = 8.0  # Minimum (peak) stellar mass of labelled galaxies
fixedSmoothingLength = 0  # Set to > 0 to disable variable smoothing length computation
numPix = 1000  # Image side length in pixels
camDir = [0, 0, -1]  # Specify viewing direction, default = down z-axis
vmin, vmax = 1.5, 5.5  # Parameters for image scaling

# ---------------------------------------

rundir = '/net/quasar/data3/Hydrangea/CE-' + str(cluster_index) + '/HYDRO/'

posloc = rundir + '/highlev/GalaxyPositionsSnap.hdf5'
spiderloc = rundir + '/highlev/SpiderwebTables.hdf5'
# plotloc = '/data2/Hydrangea/STUDENTS/Stars_CE-29-BCG_snap'
fgtloc = rundir + '/highlev/FullGalaxyTables.hdf5'

snaplistloc = rundir + "sneplists/allsnaps.dat"
snaplist = ascii.read(snaplistloc)
aexpSnap = np.array(snaplist['aexp'])

galID = 2373
plotloc = '/home/belcheva/Desktop/Project/CE-'+str(cluster_index)+'-snap'+str(plot_snap)+'-'+str(ptype)+'-'+str(galID)
# galID = yb.read_hdf5(spiderloc,'Subhalo/Snapshot_'+str(plot_snap).zfill(3)+'/Galaxy')[5]  # Center on central cluster

shi = yb.read_hdf5(spiderloc, 'SubHaloIndex')[:, plot_snap]

lastsnap = yb.read_hdf5(spiderloc, 'LastSnap')
firstsnap = yb.read_hdf5(spiderloc, 'FirstSnap')
mpeak = yb.read_hdf5(fgtloc, 'Full/Mstar')

snapStr = 'Snapshot_' + str(plot_snap).zfill(3)

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
print(snapdir)
# exit()

# Set up a mask to efficiently read only the required region from the simulation
# The third argument is a four-element array, with the first three elements
# specifying the point around which particles should be loaded, and the
# fourth the search radius (both in units of comoving Mpc h^-1).
# N.B. the division by 'conv_astro_pos' here converts the position to these
# units (since they are stored in proper Mpc in GalaxyPositionsSnap.hdf5).

readReg = ht.ReadRegion(snapdir, ptype, [*pos_gal / conv_astro_pos, imsize * np.sqrt(3) / conv_astro_pos])

# Read position and mass of particles in target region
pos = readReg.read_data("Coordinates", astro=True)

if ptype != 1:
    mass = readReg.read_data("Mass", astro=True)
else:
    mass = np.zeros(pos.shape[0]) + st.m_dm(snapdir, issnapdir=True, astro=True)


# --------------------------------------------------------------------------------
# IDs = readReg.read_data('ParticleIDs', astro=False)
#
# # snap = '029_z000p000'
# # simdir = '/net/quasar/data3/Hydrangea/CE-' + str(cluster_index) + '/HYDRO/'
# # subdir = simdir + '/data/groups_'+snap+'/eagle_subfind_tab_'+snap+'.0.hdf5'
# # SubLength = st.eagleread(subdir, 'Subhalo/SubLength', astro=False, silent=False)
# # SubLengthType = st.eagleread(subdir, 'Subhalo/SubLengthType', astro=False, silent=False)
# # SubOffset = st.eagleread(subdir, 'Subhalo/SubOffset', astro=False, silent=False)
# # particleIDs = st.eagleread(subdir, 'IDs/ParticleID', astro=False, silent=False)
# #
# # SHI = 1066
# # length = SubLength[SHI]
# # type = SubLengthType[SHI]
# # offset = SubOffset[SHI]
# # particles = particleIDs[offset:offset + length]
#
# IDs_2373 = np.loadtxt('particles.txt', unpack=True)
# IDs_2373 = IDs_2373.astype(int)
#
# dm = IDs_2373[np.where(IDs_2373 % 2 == 0)[0]]
# stars = IDs_2373[np.where(IDs_2373 % 2 == 1)[0]]
#
# # dm_particles = particles[np.where(particles % 2 == 0)[0]]
# # star_particles = particles[np.where(particles % 2 == 1)[0]]
#
# # print('Making index array, DM\n')
# # ind = np.zeros(len(IDs))
# # for i in range(len(IDs)):
# #     if IDs[i] in dm:
# #         ind[i] = 1
# # print('Index array done')
# #
# # ind = np.where(ind == 1)[0]
# #
# # index = open('indexing_array.txt', 'a')
# # for i in range(len(ind)):
# #     index.write('{:d}\n'.format(ind[i]))
# #
# # index.close()
#
# # ind = np.loadtxt('indexing_array.txt', unpack=True).astype(int)
#
# # print('Making index array, stars\n')
# # ind = np.zeros(len(IDs))
# # for i in range(len(IDs)):
# #     if IDs[i] in stars:
# #         ind[i] = 1
# # print('Index array done')
# #
# # ind = np.where(ind == 1)[0]
# #
# # index = open('indexing_array_stars.txt', 'a')
# # for i in range(len(ind)):
# #     index.write('{:d}\n'.format(ind[i]))
# #
# # index.close()
#
# ind = np.loadtxt('indexing_array_stars.txt', unpack=True).astype(int)
#
# # print(pos[ind])
# # print(mass[ind])
#
# galaxy_positions = pos[ind]
#
# pos = pos[ind]
# mass = mass[ind]


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
        for i in range(len(ind)):
            numer = numer + np.array(pos[i])*m_dm*weight_function(origin, pos[i], R)
            denom = denom + m_dm * weight_function(origin, pos[i], R)
            # print('{:1.6f}'.format(weight_function(origin, pos[i], R) / wmax))
    else:
        for i in range(len(ind)):
            numer = numer + np.array(pos[i])*mass[i]*weight_function(origin, pos[i], R)
            denom = denom + mass[i]*weight_function(origin, pos[i], R)
            print(weight_function(origin, pos[i], R))

    return numer/denom


# R = 0.0012080314    # stellar half-mass radius of 2373, pMpc
# CoM = find_CoM(pos_gal, pos, mass, R, ptype)
# print('CoM \t', CoM)
# print('origin \t', pos_gal)
#
# d = abs(pos_gal - CoM)
# displacement = np.sqrt(sum(d**2))
#
# print('CoM displacement: [pkpc]\t', displacement*1000)


from mpl_toolkits.mplot3d import Axes3D

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
#
# ax.scatter(galaxy_positions[:, 0], galaxy_positions[:, 1], galaxy_positions[:, 2], marker='x', alpha=0.5)
# plt.show()
# plt.savefig('/home/belcheva/Desktop/3Dgal_DM.png', dpi=300)

# plt.figure(figsize=(8, 8))
# plt.scatter(galaxy_positions[:, 0], galaxy_positions[:, 1], s=2, c='w', alpha=0.3)
# ax = plt.gca()
# ax.set_facecolor('k')
# plt.xlabel('x coordinate')
# plt.ylabel('y coordinate')
# plt.title('Galaxy ID = 2373, stellar matter particles')
# plt.plot(pos_gal[0], pos_gal[1], 'x', color='maroon', label='centre, from simulation')
# plt.plot(CoM[0], CoM[1], 'x', color='royalblue', label='CoM, as calculated')
# plt.legend(fontsize='small')
# plt.savefig('/home/belcheva/Desktop/2Dgal_SM.png', dpi=300)

# mu = pos_gal[0]
# x = np.linspace(mu-0.1, mu+0.1, 1000)
# y = gaussian(x, mu, sigma=R)
# w = weight_function(pos_gal, (x, x, x), R)
#
# plt.figure()
# plt.plot(x, y)
# plt.savefig('gaussian.png')
#
# plt.figure()
# plt.plot(x, w)
# plt.savefig('weight.png')
#
# exit()

# --------------------------------------------------------------------------------

deltaPos = pos - pos_gal[None, :]

# Select only those particles that are actually within target sphere
rad = np.linalg.norm(deltaPos, axis=1)
ind_sphere = np.nonzero(rad < imsize * np.sqrt(3))[0]
mass = mass[ind_sphere]
pos = pos[ind_sphere, :]

# Find galaxies in the field-of-view
ind_in_field = np.nonzero(
    (np.max(np.abs(pos_allGal - pos_gal[None, :]), axis=1) < imsize) & (mpeak >= min_mpeak_otherGal) & (
                aexpSnap[lastsnap] >= aexp_factor) & (aexpSnap[firstsnap] <= aexp_factor))[0]

pos_in_field = (pos_allGal[ind_in_field, :] - pos_gal[None, :])

if fixedSmoothingLength > 0:
    hsml = np.zeros(len(mass), dtype=np.float32) + fixedSmoothingLength
else:
    hsml = None

# Generate actual image
image_weight_all, image_quant = ir.make_sph_image_new_3d(pos, mass, mass, hsml, DesNgb=desNGB, imsize=numPix, zpix=1,
                                                         boxsize=imsize, CamPos=pos_gal, CamDir=camDir,
                                                         CamAngle=[0, 0, 0], CamFOV=[0.0, 0.0], make_deepcopy=True,
                                                         zrange=[-imsize, imsize], tau=1e6, return_hsml=False)
# m = max(np.log10(image_weight_all[:, :, 1] + 1e-5))

# Display image to check things went fine:


# if ptype == 4:
cmap = plt.cm.Greys_r
# else:
#    cmap = plt.cm.

plt.figure(figsize=(6.0, 6.0))
plt.imshow(np.log10(image_weight_all[:, :, 1] + 1e-5), cmap=cmap, origin='lower',
           extent=[-imsize, imsize, -imsize, imsize], vmin=vmin, vmax=vmax,
           interpolation='none')

# Some embellishments on the image
plt.text(-0.045 / 0.05 * imsize, 0.045 / 0.05 * imsize, 'z = {:.3f}'
         .format(1 / aexp_factor - 1), va='center',
         ha='left', color='white')

plt.text(0.045 / 0.05 * imsize, 0.045 / 0.05 * imsize, 'galID = {:d}'
         .format(galID), va='center',
         ha='right', color='white')

if mark_galaxies:
    for iigal, igal in enumerate(ind_in_field):
        plt.scatter(pos_in_field[iigal, 0], pos_in_field[iigal, 1], 30,
                    edgecolor='limegreen', facecolor='none', alpha=0.5)

        plt.text(pos_in_field[iigal, 0] + imsize / 100, pos_in_field[iigal, 1] + imsize / 100,
                 str(shi[igal]), color='limegreen', va='bottom', ha='left', alpha=0.5, label='SHI')

ax = plt.gca()
ax.set_xlabel(r'$\Delta x$ [pMpc]')
ax.set_ylabel(r'$\Delta y$ [pMpc]')

ax.set_xlim((-imsize, imsize))
ax.set_ylim((-imsize, imsize))

plt.subplots_adjust(left=0.15, right=0.95, bottom=0.15, top=0.92)
plt.plot(0, 0, 'x', color='red', markersize=2, label='(0, 0)')
plt.legend(loc='lower left')
# plt.plot(0, 0, 'x', color='maroon', label='centre, from simulation')
# plt.plot(CoM[0]-pos_gal[0], CoM[1]-pos_gal[1], 'x', color='royalblue', label='CoM, as calculated')
# figname = plotloc + '-only-SM-own.png'
figname = '/home/belcheva/Desktop/2373-stars-marked-zoom.png'
# plt.legend(fontsize='small', loc='lower right')
plt.savefig(figname, dpi=200)
plt.close()

if save_maps:
    maploc = plotloc + str(plot_snap).zfill(4) + '.hdf5'
    yb.write_hdf5(image_weight_all, maploc, 'Image', new=True)
    yb.write_hdf5(np.array((-imsize, imsize, -imsize, imsize)), maploc, 'Extent')

print("Done!")
print('Saved '+figname)