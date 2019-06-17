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
matplotlib.use('pdf')

from astropy.io import ascii

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

matplotlib.rcParams['font.family'] = 'serif' 
matplotlib.rcParams['font.serif'][0] = 'palatino'

from pdb import set_trace

# ----------------------------------------

cluster_index = 29
mark_galaxies = False    # Optionally overplot location of galaxies
save_maps = False     # Store images as HDF5 as well as picture
desNGB = 32          # Number of neighbours for smoothing calculation
plot_snap = 29      # Which snapshot to plot

ptype1 = 1            # Particle type to plot (DM = 1)
ptype2 = 4            # Particle type to plot (stars = 4)
imsize = 0.25         # (Half-)image size in pMpc
min_mpeak_otherGal = 8.0  # Minimum (peak) stellar mass of labelled galaxies
fixedSmoothingLength = 0  # Set to > 0 to disable variable smoothing length computation
numPix = 1000        # Image side length in pixels
camDir = [0, 0, -1]  # Specify viewing direction, default = down z-axis
vmin, vmax = 1.5, 5.5  # Parameters for image scaling

# ---------------------------------------

rundir = '/net/quasar/data3/Hydrangea/CE-' + str(cluster_index) + '/HYDRO'

posloc = rundir + '/highlev/GalaxyPositionsSnap.hdf5'
spiderloc = rundir + '/highlev/SpiderwebTables.hdf5'
# ~ plotloc = '/data3/Hydrangea/STUDENTS/Stars_CE-0-BCG_snap'
plotloc = '/home/belcheva/Desktop/Stars_CE-0-BCG_snap'
fgtloc = rundir + '/highlev/FullGalaxyTables.hdf5' 

snaplistloc = rundir + "/sneplists/allsnaps.dat"
snaplist = ascii.read(snaplistloc)
aexpSnap = np.array(snaplist['aexp'])

galID = 2373
# galID = yb.read_hdf5(spiderloc, 'Subhalo/Snapshot_' + str(plot_snap).zfill(3) + '/Galaxy')[10]  # Center on central cluster

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
h_factor = 1/0.6777
conv_astro_pos = h_factor*aexp_factor

print("Determined galaxy position as {:.3f}/{:.3f}/{:.3f}..."
      .format(*pos_gal))

snapdir = st.form_files(rundir, isnap = plot_snap, types = 'snap')

# Set up a mask to efficiently read only the required region from the simulation
readReg_DM = ht.ReadRegion(snapdir, ptype1, [*pos_gal/conv_astro_pos, imsize*np.sqrt(3)/conv_astro_pos])
readReg_stars = ht.ReadRegion(snapdir, ptype2, [*pos_gal/conv_astro_pos, imsize*np.sqrt(3)/conv_astro_pos])

# Read position and mass of particles in target region
pos1 = readReg_DM.read_data("Coordinates", astro = True)
pos2 = readReg_stars.read_data("Coordinates", astro = True)


if ptype2 != 1:
    mass2 = readReg_stars.read_data("Mass", astro = True)
if ptype1 == 1:
    mass1 = np.zeros(pos1.shape[0])+st.m_dm(snapdir, issnapdir=True,astro = True)

deltaPos1 = pos1 - pos_gal[None, :]
deltaPos2 = pos2 - pos_gal[None, :]

# Select only those particles that are actually within target sphere
rad1 = np.linalg.norm(deltaPos1, axis = 1)
rad2 = np.linalg.norm(deltaPos2, axis = 1)
ind_sphere1 = np.nonzero(rad1 < imsize*np.sqrt(3))[0]
ind_sphere2 = np.nonzero(rad2 < imsize*np.sqrt(3))[0]
mass1 = mass1[ind_sphere1]
mass2 = mass2[ind_sphere2]
pos1 = pos1[ind_sphere1, :]
pos2 = pos2[ind_sphere2, :]

# Find galaxies in the field-of-view
ind_in_field = np.nonzero((np.max(np.abs(pos_allGal-pos_gal[None, :]), axis = 1) < imsize) & (mpeak >= min_mpeak_otherGal) & (aexpSnap[lastsnap] >= aexp_factor) & (aexpSnap[firstsnap] <= aexp_factor))[0]

pos_in_field = (pos_allGal[ind_in_field, :]-pos_gal[None, :])

if fixedSmoothingLength > 0:
    hsml = np.zeros(len(mass), dtype = np.float32) + fixedSmoothingLength
else:
    hsml = None
        
# Generate actual image
# ~ image_weight_all_DM, image_quant_DM = ir.make_sph_image_new_3d(pos1, mass1, mass1, hsml, DesNgb=desNGB,imsize=numPix, zpix = 1, boxsize = imsize, CamPos = pos_gal, CamDir = camDir, CamAngle = [0,0,0], CamFOV = [0.0, 0.0], make_deepcopy = True, zrange = [-imsize, imsize], tau = 1e6, return_hsml = False)
image_weight_all_stars, image_quant_stars = ir.make_sph_image_new_3d(pos2, mass2, mass2, hsml, DesNgb=desNGB,imsize=numPix, zpix = 1, boxsize = imsize, CamPos = pos_gal, CamDir = camDir, CamAngle = [0,0,0], CamFOV = [0.0, 0.0], make_deepcopy = True, zrange = [-imsize, imsize], tau = 1e6, return_hsml = False)
    
# Display image to check things went fine:

cmap1 = plt.cm.Purples
cmap2 = plt.cm.Greys_r

'''
if ptype == 4:
	cmap = plt.cm.Greys_r
else:
    cmap = plt.cm.Purples
'''
 
plt.figure(figsize = (6.0, 6.0))

a = np.log10(image_weight_all_stars[:,:,1])+1e-5
a = a - min(a[0])
#print(a, min(a[0]))
b = np.zeros([1000,1000])
b = a[np.where(a[0] > 0.8), np.where(a[1] > 0.8)]
print(b)



plt.imshow(a, cmap = cmap2, origin = 'lower', 
           extent = [-imsize, imsize, -imsize, imsize], vmin = vmin, vmax = vmax, 
           interpolation = 'none')

'''
plt.imshow(np.log10(image_weight_all_DM[:, :, 1]+1e-5), cmap = cmap1, origin = 'lower', 
           extent = [-imsize, imsize, -imsize, imsize], vmin = vmin, vmax = vmax, 
           interpolation = 'none')
'''

# Some embellishments on the image
plt.text(-0.045/0.05*imsize, 0.045/0.05*imsize, 'z = {:.3f}' 
         .format(1/aexp_factor - 1), va = 'center', 
         ha = 'left', color = 'white')

plt.text(0.045/0.05*imsize, 0.045/0.05*imsize, 'galID = {:d}' 
         .format(galID), va = 'center', 
         ha = 'right', color = 'white')

if mark_galaxies:
    for iigal, igal in enumerate(ind_in_field):
        plt.scatter(pos_in_field[iigal, 0], pos_in_field[iigal, 1], 30, 
                    edgecolor = 'limegreen', facecolor = 'none', alpha = 0.5)

        plt.text(pos_in_field[iigal, 0]+imsize/100, pos_in_field[iigal, 1]+imsize/100, 
                 str(shi[igal]), color = 'limegreen', va = 'bottom', ha = 'left', alpha = 0.5)
        
            
ax = plt.gca()
ax.set_xlabel(r'$\Delta x$ [pMpc]')
ax.set_ylabel(r'$\Delta y$ [pMpc]')
    
ax.set_xlim((-imsize, imsize))
ax.set_ylim((-imsize, imsize))

plt.subplots_adjust(left = 0.15, right = 0.95, bottom = 0.15, top = 0.92)
plt.savefig(plotloc + str(plot_snap).zfill(4) + str(ptype1) + '-and-' + str(ptype2) + 'central' + '.png', dpi=200)
plt.close()

if save_maps:
    maploc = plotloc + str(plot_snap).zfill(4) + '.hdf5'
    yb.write_hdf5(image_weight_all, maploc, 'Image', new = True)
    yb.write_hdf5(np.array((-imsize, imsize, -imsize, imsize)), maploc, 'Extent')
            
    
print("Done!")


