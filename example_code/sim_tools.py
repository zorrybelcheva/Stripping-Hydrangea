"""
Collection of tools used for both Eagle and Hydrangea analysis
"""

import numpy as np
from os.path import isfile
import sys
import h5py as h5
import time
import math
import glob
import numexpr as ne
import hydrangea_tools as ht
from astropy.io import ascii
from astropy.cosmology import Planck13
import image_routines as imtools
import yb_utils as yb
from copy import deepcopy
from scipy.interpolate import interp1d

from pdb import set_trace

def seq_file(fileparts, seqnr):
    fileparts[-2] = str(seqnr)
    filename = ".".join(fileparts)
    return filename

# ---------- EAGLEREAD ------------------


def eagleread(file, var, ptype = None, depth = 1, astro = True, nopred = False, dtype = None, sim='Eagle', zoom = 0, merge = None, return_shi = False, nfiles = None, silent = False):
    
    """
    zoom: Set to > 0 if this is a zoom-in simulation, and you want to exclude haloes outside the high-resolution region.
          This requires a file "BoundaryFlag.hdf5" in the directory (n.b.: no effect on particle reading).
          1: exclude haloes <= 8 cMpc from low-res particles
          2: exclude haloes <= 2 cMpc from low-res particles (default)

    merge: If not None, perform on-the-fly post-processing of subfind outputs to deal with spurious subhaloes.
           0: Simply exclude bad subhaloes.
           > 0: Add quantities to their parents
                1: Only add distributive quantities, i.e. f(a+b) = f(a)+f(b)
                2: Also re-compute weighted quantities

    """

    # First: check if the file exists...!
    if isfile(file) == False:
        print("-----------------------------------------------------")
        print("Error [" + file + "]")
        print("This file does not exist. Please supply correct name!")
        print("-----------------------------------------------------")
    
        return -1
        

    # Break down variable into groups and dataset name
    nameparts = var.split("/")
    ngroups = len(nameparts)-1      # Number of groups to desired dataset
    dataset_name = nameparts[-1]    # The 'actual' name of the dataset
    if len(nameparts) > 1:
        basegroup_name = nameparts[0]
    else:
        basegroup_name = ""
    
    if not silent:
        print('Reading "' + dataset_name.upper() + '" for "' + basegroup_name.upper() + '"...')

    if basegroup_name[:8] == 'PartType':
        ptype = int(basegroup_name[8])
        
        if not silent:
            print("Determined particle type {:d}" .format(ptype))
        
    # Break file name into base and running sequence number
    fileparts = file.split(".")
    filebase = fileparts[:-3]
    filecoda = fileparts [-1]

    type_determinant = ((fileparts[-3]).split("/"))[-1]
    type_parts = type_determinant.split("_")

    type_code = 0 # Set as this for default
    
    # ------- Determine type of file to be read -----------

    if (type_parts[0] == 'snip'):
        if not silent: print("SNIPSHOT detected!")
        type_code = 1
    elif (type_parts[0] == 'snap'):
        if not silent: print("SNAPSHOT detected!")
        type_code = 2
    elif(type_parts[0] in ['rockstar', 'Rockstar']):
        if not silent: print("ROCKSTAR detected!")
        type_code = 5
    elif len(type_parts) >= 2:
        if ("_".join(type_parts[0:3]) == 'eagle_subfind_tab'):
            if not silent: print("EAGLE_SUBFIND_TAB detected!")
            type_code = 3
        elif ("_".join(type_parts[0:3]) == 'eagle_subfind_particles'):
            if not silent: print("EAGLE_SUBFIND_PARTICLES detected!")
            type_code = 4



    # ------ Determine number of files to be read ---------

    if nfiles is None:
        f = h5.File(file, "r")
        if sim == 'Eagle':
            nfiles = f['Header'].attrs['NumFilesPerSnapshot']
        elif sim == 'Illustris':
            nfiles = f['Header'].attrs['NumFiles']
        else:
            print("Cannot determine number of files...")
            sys.exit(111)
        f.close()
        
        
     # ----- Determine type of data to be read and set up output -----------

    if (type_code == 1 or type_code == 2 or type_code == 4) and ptype is not None:
        f = h5.File(seq_file(fileparts, 0), 'r')
        hac = False
        header = f["/Header"]
        nelem_full = header.attrs["NumPart_Total"][ptype]
        f.close()
        
    elif (type_code == 3):  # Subfind tab
        f = h5.File(seq_file(fileparts, 0), 'r')
        hac = False
        header = f["/Header"]
        
        if basegroup_name == "FOF":
            nelem_full = header.attrs["TotNgroups"]
        elif basegroup_name == "Subhalo":
            nelem_full = header.attrs["TotNsubgroups"]
        elif basegroup_name == "IDs":
            nelem_full = header.attrs["TotNids"]
        else:
            print("Base group name '" + basegroup_name + "' is not understood, reverting to hac...")
            hac = True
        f.close()

            
    elif (type_code == 5):   # Rockstar output
        f = h5.File(seq_file(fileparts, nfiles-1), "r")
        hac = False
        header = f["/Header"]

        if basegroup_name == 'Halo':
            nelem_full = header.attrs["Tot_NHalo"]
            if dataset_name.upper() == "PL_OFFSET":
                nelem_full += 1

        elif basegroup_name == 'Particles':
            nelem_full = header.attrs["Tot_NumPartList"]

        else:
            print("Base group name '" + basegroup_name + "' is not understood, reverting to hac...")
            hac = True
        f.close()
        
    else:
        print("Type code = {:d}, reverting to hac..." .format(type_code))
        hac = True
        

    if not hac:
        offset = 0
        if not silent: print("   ... (detected {:d} elements) ..." .format(nelem_full), flush=True)

  
    # Loop over individual files to read data

    t_lastupdate = time.time()
    t_readstart = time.time()

    tcat = np.zeros(3)
    output_is_set_up = False
    
    for seqnr in range(nfiles):

        if not silent: print(str(seqnr)+" ", end = "",flush=True)
        
        fileparts[-2] = str(seqnr)
        filename = ".".join(fileparts)
        
        # Test whether file exists:
        if isfile(filename) == False:
            print("\nEnd of files found at " + str(seqnr-1))
            print("\nThis should NOT happen!", flush=True)
            sys.exit(111)

        # Load current (partial) HDF5 file:

        t_s1 = time.time()
        f = h5.File(filename, "r")

        e = var in f
        if not e:
            print("No data found on file {:d}!" .format(seqnr))
            continue

        dataset = f[var]
        data_part = np.empty(f[var].shape, f[var].dtype)
        f[var].read_direct(data_part)
        t_s2 = time.time()
        tcat[0] += t_s2-t_s1

        # Need to set up full output array if this is the first file
        if not output_is_set_up:
            dshape = list(data_part.shape)

            if hac:
                depth = len(f[var].shape)
                dshape[0] = 0
                data_stack = np.empty(dshape,dtype=dtype)
            else:

                if nelem_full is None:
                    print("ERROR: nelem_full not initialized. Please investigate.")
                    sys.exit(77)

                if nelem_full == 0:
                    print("ERROR: expecting zero data elements in total. Please investigate.")
                    sys.exit(78)

                if dtype is None:
                    dtype = data_part.dtype
                
                dshape[0] = nelem_full 
                data_full = np.zeros(dshape, dtype=dtype)
            
            # Initialization of full output is the same with and without hac, just that 'dshape' is different!    
            data_full = np.empty(dshape,dtype=dtype)
            output_is_set_up = True
            
        # Special section for HAC:
        if hac:
            data_stack = np.concatenate((data_stack, data_part), axis = 0)
            t_s3 = time.time()
            tcat[1] += t_s3-t_s2
            
            # Combine to full list if critical size reached:
            if len(data_stack) > 1e9:
                t_upd_start = time.time()
                sys.stdout.write("Update full list [" + "%.2f" % (time.time() - t_lastupdate) + " sec...")            
                sys.stdout.flush()
                data_full = np.concatenate((data_full, data_stack),axis=0)
                data_stack = np.empty(dshape, dtype=dtype)
                t_lastupdate = time.time()
                sys.stdout.write(" / %.2f" % (t_lastupdate-t_upd_start) + " sec.]") 
                sys.stdout.flush()
                
                t_s4 = time.time()
                tcat[2] += t_s4 - t_s3

        # Standard case: NO HAC
        else:
            len_part = (data_part.shape)[0]
            data_full[offset:offset+len_part, ... ] = data_part
            offset += len_part
            
    if hac:
        # Final list concatenation:
        t_s1 = time.time()
        data_full = np.concatenate((data_full, data_stack),axis=0)
        tcat[2] += (time.time()-t_s1)

    if not silent:
        print('Reading "' + dataset_name.upper() + '" took %.2f sec.' % (time.time()-t_readstart))

        if hac:
            print('   Read    = %.2f sec. (%.1f%%)' % (tcat[0], tcat[0]/np.sum(tcat)*100))
            print('   Stack   = %.2f sec. (%.1f%%)' % (tcat[1], tcat[1]/np.sum(tcat)*100))
            print('   Combine = %.2f sec. (%.1f%%)' % (tcat[2], tcat[2]/np.sum(tcat)*100))

    if return_shi:
        ind_good = np.arange(data_full.shape[0])

    # New bit added 16 Jan 2017: deal with spurious subhaloes
    if basegroup_name == "Subhalo":

        """
        There are two different lists for excluding particles, so we need to make a 
        "super-list" of clean subhaloes and then successively mark subhaloes off it.
        """

        goodlist = np.zeros(data_full.shape[0])
        
        # Merge back spuriously cut-off subhaloes, if desired:
        if merge is not None:
            bad_sh = yb.read_hdf5(yb.dir(file) + "SubhaloMergerFlag.hdf5", "Spurious")
            parent_sh = yb.read_hdf5(yb.dir(file) + "SubhaloMergerFlag.hdf5", "Parents")
            goodlist[bad_sh] = 1

            # This here is the fun part: Adding bad subhaloes back to their parents...
            if merge >= 1:
            
                code, refquant = quantcode(var)
                if code <= merge:
                    if code == 1:
                        for ibad in bad_sh:
                            data_full[parent_sh,...] += data_full[bad_sh,...]

                    if code == 2:
                        data_ref = eagleread(file, 'Subhalo/' + refquant, zoom = 0, merge = None)
                        for ibad in bad_sh:
                            data_full[parent_sh,...] = (data_full[parent_sh,...] * data_ref[parent_sh] + data_full[bad_sh,...] * data_ref[bad_sh]) / (data_ref[parent_sh] + data_ref[bad_sh])
                            
                            
                    if code > 2:
                        print("Please recompute this quantity yourself...")


        # Exclude subhaloes outside of high-resolution region
        if zoom > 0:
            cont_flag = yb.read_hdf5(yb.dir(file) + "BoundaryFlag.hdf5", "ContaminationFlag")
            ind_bad = np.nonzero(cont_flag >= zoom)[0]
            goodlist[ind_bad] = 1

        ind_good = np.nonzero(goodlist == 0)[0]
        data_full = data_full[ind_good,...]
        
        
    # end of spurious subhalo section

    retlist = [data_full]

    if sim == 'Eagle':

        if astro == True:

            # Determine code --> physical conversion factors
            hscale_exponent = dataset.attrs["h-scale-exponent"]
            ascale_exponent = dataset.attrs["aexp-scale-exponent"]

            header = f["/Header"]
            
            aexp = header.attrs["ExpansionFactor"]
            h_hubble = header.attrs["HubbleParam"]
            
            conv_astro = aexp**ascale_exponent * h_hubble**hscale_exponent

            data_full *= conv_astro
        
            retlist = [data_full, conv_astro, aexp]
            if return_shi:
                retlist.append(ind_good)

            return retlist

        else:
            retlist = [data_full]
            if return_shi:
                retlist.append(ind_good)
            else:
                retlist = retlist[0]

            return retlist

    else:
        retlist = [data_full]
        if return_shi:
            retlist.append(ind_good)
        else:
            retlist = retlist[0]

        return retlist



def eagleread_attribute(filename, var):
    
    # First: check if the file exists...!
    if isfile(filename) == False:
        print("-----------------------------------------------------")
        print("Error [" + file + "]")
        print("This file does not exist. Please supply correct name!")
        print("-----------------------------------------------------")
    
        return -1
    
    # Break down variable into groups, [dataset,] and attribute name
    
    nameparts = var.split("/")
    
    if len(nameparts) == 0:
        print("You have to specify the dataset/group AND attribute name...")
        1/0
        
    attribute_name = nameparts[-1]    # The 'actual' name of the attribute
    reference_name = "/".join(nameparts[:-1])    # Name of the dataset/group containing the attribute

    # Load current (partial) HDF5 file:
    
    t_s1 = time.time()
    f = h5.File(filename, "r")
    reference = f[reference_name]
    
    attrib_val = reference.attrs[attribute_name]

    return attrib_val


def gal_to_sh(rundir, gal, isnap = 29, file = True, allsh=True):

    """
    Find the subhalo representing a given galaxy in a given snapshot.

    'gal' can also be a list/array of galaxies.
    """

    if hasattr(gal, "__len__"):
        n_gal = len(gal)
    else:
        n_gal = 1
        gal = [gal]

    
    if file:
        parts = rundir.split('/')
        workdir = "/".join(parts[:-2])
    else:
        workdir = rundir
        
    f = h5.File(workdir + "/highlev/TracingTable.hdf5",'r')

    if allsh and isnap > 0:
        coda = "/SubHaloIndexAll"
    else:
        coda = "/SubHaloIndex"

    DSet = f["Snapshot_" + str(isnap).zfill(3) + coda]

    ind_gal_arr = np.array(gal,dtype=int)
    ind_exist = np.nonzero(ind_gal_arr < DSet.shape[0])[0]

    flag = np.zeros(n_gal,dtype=int)-1
    sh = np.zeros(n_gal,dtype=int)

    sh[ind_exist] = (DSet[:])[ind_gal_arr[ind_exist]]
    flag[ind_exist] = 1

    ind_alive = np.nonzero(sh[ind_exist] >= 0)[0]
    flag[ind_exist[ind_alive]] = 0

    return sh, flag


def gal_sh_history(workdir, gal, snaplist = "/u/ybahe/ANALYSIS/snapshots_hydrangea_regular.dat"):
    f = h5.File(workdir + "/TracingTable.hdf5",'r')
    listdata = ascii.read(snaplist, format = 'no_header', guess = False)

    snaps = listdata['col1']
    nsnaps = len(snaps)
    sh = np.zeros(nsnaps, dtype = int)-100

    for ii, isnap in enumerate(snaps):
        DSet = f["Snapshot_" + str(isnap).zfill(3) + "/SubHaloIndex"]

        if (gal >= DSet.shape[0]):
            continue
        
        sh[ii] = (DSet[:])[gal]
        
    return sh
    


def gal_carrier(rundir, gal, isnap):
    """
    Find the 'carrier' of a galaxy.

    If the galaxy is still alive, this is simply itself. 
    If not, it is the galaxy it has merged with.
    """

    f = h5.File(rundir + "/TracingTable.hdf5",'r')
    DSet = f["Snapshot_" + str(isnap).zfill(3) + "/MergeList"]

    ind_gal_arr = np.array(gal,dtype=int)
    ind_exist = np.nonzero(ind_gal_arr < DSet.shape[0])[0]

    n_gal = len(gal)
    flag = np.zeros(n_gal,dtype=int)-1
    carrier = np.zeros(n_gal,dtype=int)

    carrier[ind_exist] = (DSet[:])[ind_gal_arr[ind_exist]]
    flag[ind_exist] = 0

    return carrier, flag
    

def sh_to_gal(rundir, sh, isnap, tracfile = None):

    """
    Find the galaxy represented by a given subhalo in a given snapshot

    'sh' can also be a list/array of subhaloes.
    """

    if tracfile is None:
        tracfile = "TracingTable"
    tracfile += ".hdf5"

    if tracfile == "TracingTable.hdf5":

        if isnap > 0:
            dataSetName = "Snapshot_" + str(isnap).zfill(3) + "/SubHaloIndexAll"
        else:
            dataSetName = "Snapshot_" + str(isnap).zfill(3) + "/SubHaloIndex"


    if hasattr(sh, "__len__"):
        n_sh = len(sh)
        is_scalar = False
    else:
        n_sh = 1
        sh = [sh]
        is_scalar = True
        
    ind_match = np.zeros(n_sh,dtype=int)-1

    shi = yb.read_hdf5(rundir + "/highlev/" + tracfile, dataSetName)
    num_bad = np.count_nonzero(sh > shi.shape[0])
    if num_bad > 0:
        print("Error: {:d} subhaloes out of range!"
              .format(num_bad))
        set_trace()

    # Optimisation: if there are only a few SHIs to look up, just
    # use brute-force
    if n_sh < 100:
        for i,ish in enumerate(sh):
            
            ind_match_curr = np.nonzero(shi == ish)[0]

            if len(ind_match_curr) > 1:
                print("*** ERROR: more than one galaxy ({:d}) matches subhalo {:d} ***" .format(len(ind_match), sh))
                sys.exit(12)
                
            if len(ind_match_curr) == 0:
                print("*** ERROR: subhalo {:d} not associated with any galaxy." .format(sh))
                sys.exit(13)

            ind_match[i] = ind_match_curr
                
    else:
        maxval = np.max((np.max(shi), np.max(sh)))
        revlist = create_reverse_list(shi, maxval = maxval+1)
        ind_match = revlist[np.array(sh,dtype=int)]

    if is_scalar:
        return ind_match[0]
    else:
        return ind_match



def conv_astro_factor(file, DataSetName):
    """
    Calculate conversion factor from simulation to astronomical units
    """

    f = h5.File(file,'r')
    DSet = f[DataSetName]

    hscale_exponent = DSet.attrs["h-scale-exponent"]
    ascale_exponent = DSet.attrs["aexp-scale-exponent"]
    
    header = f["/Header"]
    aexp = header.attrs["ExpansionFactor"]
    h_hubble = header.attrs["HubbleParam"]
    conv_astro = aexp**ascale_exponent * h_hubble**hscale_exponent
    
    return conv_astro


def m_dm(rundir, issnapdir = False, astro = True):
    """
    Function to load the DM particle mass from sim output

    Note that the returned particle mass is in *astronomical*, not code units.
    """

    if issnapdir:
        snapdir = rundir
    else:
        snapdir = form_files(rundir, isnap = 0, types = 'snap', stype = 'snap')

    f = h5.File(snapdir,'r')

    header = f["/Header"]
    m_dm_code = header.attrs["MassTable"][1]

    if astro:
        h_hubble = header.attrs["HubbleParam"]
        return m_dm_code / h_hubble 
    else:
        return m_dm_code

def m_bar(rundir, issnapdir = False, astro = True):

    if issnapdir:
        snapdir = rundir
    else:
        snapdir = form_files(rundir, isnap = 0, types = 'snap', stype = 'snap')

    f = h5.File(snapdir,'r')

    header = f["/Header"]
    m_dm_code = header.attrs["MassTable"][1]
    omega0 = header.attrs["Omega0"]
    omegaBar = header.attrs["OmegaBaryon"]

    f_bar = omegaBar/omega0
    m_bar_code = m_dm_code/(1-f_bar)*f_bar

    if astro:
        h_hubble = header.attrs["HubbleParam"]
        return m_bar_code / h_hubble 
    else:
        return m_bar_code

    

def locate_galaxy(rundir, ind_gal, isnep = None, sneplist = 'default', snep_num = None, snep_type = None, trace = 'Average', astro = False):
    """
    Find the approximate position of a galaxy from sniplocate output.
    """

    #set_trace()

    if hasattr(ind_gal, "__len__"):
        n_gal = len(ind_gal)
        ind_gal_arr = np.array(ind_gal,dtype=int)
    else:
        n_gal = 1
        ind_gal_arr = np.array([ind_gal],dtype=int)


    f = h5.File(rundir + "/SnipLocate.hdf5",'r')

    if isnep is not None:

        # The following is a temporary fix until the sneplist type is incorporated
        # directly in the sniplocate output
        
        if sneplist == 'basic' or sneplist == 'BA':
            listfile = rundir + "/sneplist_for_basic.dat"
        if sneplist == 'default' or sneplist == 'DE':
            listfile = rundir + "/sneplist_for_default_long.dat"

        if sneplist == 'short_movie' or sneplist == 'SM':
            listfile = rundir + "/sneplist_for_short_movie.dat"
        if sneplist == 'full_movie' or sneplist == 'FM':
            listfile = rundir + "/sneplist_for_full_movie.dat"
        if sneplist == 'paper_plot':
            listfile = rundir + "/sneplist_for_paper_plot.dat"
            
        listdata = ascii.read(listfile, format = 'no_header', guess = False)
        snep_type = listdata['col2'][isnep]
        snep_num = listdata['col3'][isnep]

    IndType = f['SnepshotType'][:]
    IndNum = np.array(f['SnepshotNum'][:],dtype=int)

    if snep_type == 'snap':
        ind_iscorrecttype = np.nonzero(IndType == 0)[0]
    elif snep_type == 'snip':
        ind_iscorrecttype = np.nonzero(IndType == 1)[0]

    revlist = np.zeros(np.max(IndNum)+1,dtype=int)-1
    revlist[IndNum[ind_iscorrecttype]] = ind_iscorrecttype
    list_snep_ind = revlist[snep_num]

    print("Snepshot={:d}" .format(list_snep_ind))

    DSet = f["Snepshot_" + str(list_snep_ind).zfill(4) + "/" + trace + "/Coordinates/"]


    ind_exist = np.nonzero(ind_gal_arr < DSet.shape[0])[0]

    pos = np.zeros((n_gal,3),dtype=float)-100000
    flag = np.zeros(n_gal,dtype=int)-1
    
    #set_trace()
    pos[ind_exist,:] = (DSet[:])[ind_gal_arr[ind_exist]]
    flag[ind_exist] = 0

    if astro:
        print("Converting to astronomical units not yet implemented.")

    return pos, flag




def locate_subhalo(rundir, ish, isnap = 29, astro = False):
    """
    Find the position of a given subhalo in a given snapshot

    This is essentially a convenience function, it just uses the standard subfind output
    """

    subdir = form_files(rundir, isnap = isnap, types = 'sub')

    cop = eagleread(subdir, 'Subhalo/CentreOfPotential', depth=3, astro = False, dtype=float, sim='Eagle', zoom = 0)

    if hasattr(ish, "__len__"):
        n_sh = len(ish)
        ind_ish_arr = np.array(ish, dtype = int)
    else:
        n_sh = 1
        ind_ish_arr = np.array([ish], dtype = int)
        
    ind_exist = np.nonzero((ind_ish_arr >= 0) & (ind_ish_arr < cop.shape[0]))[0]

    pos = np.zeros((n_sh,3),dtype=float)-10000
    flag = np.zeros(n_sh,dtype=int)-1

#    set_trace()

    pos[ind_exist,:] = cop[ind_ish_arr[ind_exist],:]
    flag[ind_exist] = 0

    if astro:
        conv_astro = conv_astro_factor(subdir, "Subhalo/CentreOfPotential")
        pos *= conv_astro

    return pos, flag


def subhalo_property(rundir, ish, quant, isnap = 29, astro = False):
    """
    Find the property of a given subhalo in a given snapshot

    This is essentially a convenience function, it just uses the standard subfind output
    Generalisation of 'locate_subhalo' function.
    """

    subdir = form_files(rundir, isnap = isnap, types = 'sub')

    quant_all = eagleread(subdir, 'Subhalo/' + quant, astro = astro, sim='Eagle', zoom = 0)

    ind_exist = np.nonzero((ish >= 0) & (ish < quant_all.shape[0]))[0]

    ind_ish_arr = np.array(ish,dtype=int)
    n_sh = len(ish)

    quant_shape = list(quant_all.shape)
    quant_shape[0] = n_sh

    quant = np.zeros(quant_shape,dtype=float)-10000
    flag = np.zeros(n_sh,dtype=int)-1

    quant[ind_exist,:] = quant_all[ind_ish_arr[ind_exist],:]
    flag[ind_exist] = 0
    
    return quant, flag


def generate_images(snapdir, parttype, cen = None, size = 1.0, sizetype = 'code', centype = 'code', numpix = 500, quantlist = [], rot = (0,0,0)):

    """
    Main (sub-)function that turns particle data into a (FITS) image.

    This function processes *ONE* particle type for *ONE* snepshot. This allows some level
    of parallelisation even for single-snepshot use, because the individual particle types
    can be dealt with independently of each other.

    Parameter explanation:
    ----------------------

    snapdir:  The principle file name of the snepshot to be processed.

    parttype: The particle type to image, as number (gas=0, dm=1, stars=4, bh=5)

    cen: The image centre (x/y/z).  
    
    size: The image half-sidelength in Mpc.

    sizetype: Whether specified size is 'astro'/'physical', or 'comoving'/'code'.
              Note - there is not (yet) a version that does 'comoving-without-h-inverse'...
              (and likewise not 'physical-with-h-inverse')

    centype: Like sizetype, but for the centre position [NOT YET IMPLEMENTED].          

    numpix: Resulting image sidelength in pixels.

    quantlist: [Optional] -- List of quantities to interpolate. If not specified, only
               a mass image is created.

    rot:       A 3-tuple specifying the rotation to be performed (theta/phi/rho).
               *** NOTE *** At present, phi and rho are ignored, until a proper 
               rotation function is written (most likely in C++...)

    """

    sstime = time.time()
    tlist = np.zeros(len(quantlist)+2)



    # -----------------------------------------------------------------------------------
    #                        READING / CENTERING / ROTATING
    # -----------------------------------------------------------------------------------

    print("Begin reading coordinates...")
    rstime = time.time()

    # Need to adjust the image size if we are specifying the size as physical
    if sizetype == 'code' or sizetype == 'com':
        lim = size
    elif sizetype == 'physical' or sizetype == 'astro':
        lim = size/conv_astro_factor(snapdir, "PartType0/Coordinates") 
    else:
        print("Your size type '" + sizetype + "' is not understood. Sorry.")
        sys.exit(100)

    # Can modify this to only load exact size, without sqrt-2 factor for rotation...
    loadrad = lim*np.sqrt(2)

    if loadrad > 7.0:
        pos = eagleread(snapdir, 'PartType' + str(parttype) + '/Coordinates', astro = False)
        f = h5.File(snapdir,'r')
        header = f["/Header"]
        aexp = header.attrs["ExpansionFactor"]
    else:
        curr_region = ht.ReadRegion(snapdir, parttype = parttype, coordinates = [cen[0], cen[1], cen[2], loadrad], shape = 'sphere')
        pos, conv_astro_pos, aexp = curr_region.read_data("Coordinates", astro = False, return_conv=True)

    imstack = np.zeros((len(quantlist)+1, numpix, numpix))
   
    if pos is None:
        
        return imstack, zred
    

    print("Beginning centering/rotation...")

    # The following is slightly cheating, because it ignores periodic wrapping. 
    # But it is totally fine for zooms...
    pos = pos - cen[None, :]
        
    theta = rot[0]
    phi = rot[1]
    rho = rot[2]

    if ((phi != 0) or (rho != 0)):
        print("Arbitrary rotations not yet implemented...")
        sys.exit(14)

    if (theta != 0):
        print("Rotating coordinates by theta={:.2f}..." .format(theta))
        pos = rotatebyaxis(pos, [0,1,0], theta)
        print("...rotation done")
            

    retime = time.time()
    print("Reading, centering, and rotating took {:.2f} sec." .format(retime-rstime))
    tlist[0] = (retime-rstime)
    

    # -----------------------------------------------------------------------
    #                      MAKE IMAGES
    # -----------------------------------------------------------------------

    istime = time.time()

    if parttype == 1:
        mass = np.zeros(pos.shape[0])+m_dm(snapdir, issnapdir = True)
    else:

        if loadrad > 7.0:
            mass = eagleread(snapdir, 'PartType' + str(parttype) + '/Mass', astro = True)[0]
        else:
            mass = curr_region.read_data("Mass", astro = True, return_conv=False)

    hsml = None    # Initialise to this
    ind_sel = None # likewise
    shift = True   # likewise


    iim = 0
    iq = 0

    if len(quantlist) == 0:
        quantlist.append("Mass")


    # =========== LOOP THROUGH QUANTITIES ============

    for quant_name in quantlist:

        qstime = time.time()
        if quant_name != "Mass":
            if loadrad > 7.0:
                quant = eagleread(snapdir, 'PartType' + str(parttype) + '/' + quant_name, astro = True)[0]
            else:
                quant, conv_astro_quant, aexp = curr_region.read_data(quant_name, astro = True, return_conv = True)
        else:
            quant = mass

            
        #set_trace()
        massimage, quantimage = imtools.make_sph_image(pos, mass, quant, DesNgb=32, order=[0,1,2], boxsize=lim, imsize=numpix, hsml = hsml)
        #set_trace()

        hsml = None # Fix to enforce new smoothing length computation always.

        # Next line prevents pos being shifted again in (possible) next quantity
        shift = False

        if iim == 0:
            imstack[0,:,:] = massimage
            iim += 1

        if quant_name != "Mass":
            imstack[iim, :, :] = quantimage
            iim += 1
        
        #set_trace()
        tlist[iq] = time.time()-qstime
        iq += 1


    # -------------  Tidying up at the end ---------------

    zred = 1.0/aexp - 1

    print("Done creating images at z={:.3f} for parttype {:d}" .format(zred, parttype))
    print("")
    print("   Total: {:.2f} sec." .format(np.sum(tlist)))
    print("      --- Read             = %.2f sec. (%.2f%%)" % (tlist[0], tlist[0]/np.sum(tlist)*100))
    
    for iq in range(len(quantlist)):
        print("      --- Q" + str(iq) + " ['" + quantlist[iq] + "'] = %.2f sec. (%.2f%%)" % (tlist[iq+1], tlist[iq+1]/np.sum(tlist)*100))
        

    return imstack, zred

 


    
# ------- Ends definition of function generate_images() --------------

def generate_images_dev(snapdir, parttype, cen = None, size = 1.0, sizetype = 'code', centype = 'code', numpix = 500, quantlist = [], rot = (0,0,0), zpix = None, CamDir = [0,0,-1], CamAngle = [0,0,0], CamFOV = [20.0,20.0], loadHsml = True, zrange = None, tau = None, shift_to_cen = True, interpolateHsml = False, rootdir = None):

    
    """
    Updated main (sub-)function that turns particle data into a (FITS) image.

    This function processes *ONE* particle type for *ONE* snepshot. This allows some level
    of parallelisation even for single-snepshot use, because the individual particle types
    can be dealt with independently of each other.

    Note: This is an experimental (development) version, started 10 Nov 2016. 
    It uses the modified gridding function, optionally the one with 3D support.

    Parameter explanation:
    ----------------------

    snapdir:  The principle file name of the snepshot to be processed.

    parttype: The particle type to image, as number (gas=0, dm=1, stars=4, bh=5)

    cen: The image centre (x/y/z).  
    
    size: The image half-sidelength in Mpc.

    sizetype: Whether specified size is 'astro'/'physical', or 'comoving'/'code'.
              Note - there is not (yet) a version that does 'comoving-without-h-inverse'...
              (and likewise not 'physical-with-h-inverse')

    centype: Like sizetype, but for the centre position [NOT YET IMPLEMENTED].          

    numpix: Resulting image sidelength in pixels.

    quantlist: [Optional] -- List of quantities to interpolate. If not specified, only
               a mass image is created.

    rot:       A 3-tuple specifying the rotation to be performed (theta/phi/rho).
               *** NOTE *** At present, phi and rho are ignored, until a proper 
               rotation function is written (most likely in C++...)
               *** NEW NOTE *** In this version, this is a convenience function that 
               simply sets CamDir.           
               
    """


    sstime = time.time()
    tlist = np.zeros(len(quantlist)+2)


    # -----------------------------------------------------------------------------------
    #                        READING / CENTERING / ROTATING
    # -----------------------------------------------------------------------------------

    print("Begin reading coordinates...")
    rstime = time.time()

    # Need to adjust the image size if we are specifying the size as physical
    if sizetype == 'code' or sizetype == 'com':
        lim = size
    elif sizetype == 'physical' or sizetype == 'astro':
        lim = size/conv_astro_factor(snapdir, "PartType0/Coordinates") 
    else:
        print("Your size type '" + sizetype + "' is not understood. Sorry.")
        sys.exit(100)

    # Can modify this to only load exact size, without sqrt-2 factor for rotation...
    loadrad = lim*np.sqrt(2)

    if CamFOV[0] > 1e-3:
        CP_z = 1.0/(np.tan(CamFOV[0]/180.0*np.pi))*lim
    else:
        CP_z = 0.0
        
    if CP_z > loadrad:
        loadrad = CP_z

    if zrange is not None:
        if 4.0*CP_z/(np.cos(CamFOV[0]/180.0*np.pi)) > loadrad:
            loadrad = 4.0*CP_z/(np.cos(CamFOV[0]/180.0*np.pi)) 


    print("Loading radius determined as {:.2f} cMpc..." .format(loadrad))
    sys.stdout.flush()

    # The following is a crude fix for excessively slow cell region setup when loading
    # large volumes. Eventually, this should be replaced by a secondary, coarser grid...
    #
    # Note: Internally, all positions are handled as comoving (hence astro = False)

    if loadrad > 7.0:
        pos = eagleread(snapdir, 'PartType' + str(parttype) + '/Coordinates', astro = False, dtype = np.float32)
        f = h5.File(snapdir,'r')
        header = f["/Header"]
        aexp = header.attrs["ExpansionFactor"]
    else:
        curr_region = ht.ReadRegion(snapdir, parttype = parttype, coordinates = [cen[0], cen[1], cen[2], loadrad], shape = 'sphere')
        pos, conv_astro_pos, aexp = curr_region.read_data("Coordinates", astro = False, return_conv=True)

    # Have to set up different output shapes, depending on whether 3D is on or off

    if loadHsml is False and parttype != 0:
        hsml = imtools.compute_hsml(pos, DesNgb=48, hmax = lim)
        
    if interpolateHsml is True and parttype == 0:
        hsml = interpolate_hsml(snapdir, rootdir)

    if zpix is None:
        imstack = np.zeros((len(quantlist)+1, numpix, numpix))
    elif tau is None:
        imstack = np.zeros((len(quantlist)+1, numpix, numpix, zpix))
    else:
        imstack = np.zeros((len(quantlist)+1, numpix, numpix, 2))

    if pos is None:
        return imstack, zred
    
    print("Beginning centering/rotation...")
    sys.stdout.flush()


    if shift_to_cen:
        # The following is slightly cheating, because it ignores periodic wrapping. 
        # But it is totally fine for zooms...
        pos = pos - cen[None, :]
        
    theta = rot[0]
    phi = rot[1]
    rho = rot[2]

    if CamDir is None:
        CamDirX = -np.sin(theta*np.pi/180) * np.cos(phi*np.pi/180)
        CamDirY = -np.sin(theta*np.pi/180) * np.sin(phi*np.pi/180)
        CamDirZ = -np.cos(theta*np.pi/180)


    retime = time.time()
    print("Reading, centering, and rotating took {:.2f} sec." .format(retime-rstime))
    tlist[0] = (retime-rstime)
    sys.stdout.flush()

    # -----------------------------------------------------------------------
    #                      MAKE IMAGES
    # -----------------------------------------------------------------------

    istime = time.time()

    if parttype == 1:
        mass = np.zeros(pos.shape[0])+m_dm(snapdir, issnapdir = True)
    else:

        if loadrad > 7.0:
            mass = eagleread(snapdir, 'PartType' + str(parttype) + '/Mass', astro = True, dtype = np.float32)[0]
        else:
            mass = curr_region.read_data("Mass", astro = True, return_conv=False)

    if parttype == 0 and loadHsml:
        if loadrad > 7.0:
            hsml = eagleread(snapdir, 'PartType' + str(parttype) + '/SmoothingLength', astro = False, dtype = np.float32)
        else:
            hsml = curr_region.read_data("SmoothingLength", astro = False, return_conv=False)

    """  Note that hsml computation is now done immediately after coordinate reading, to save memory
    else:
        hsml = None    # Initialise to this
    """

    ind_sel = None # likewise
    shift = True   # likewise
    
    iim = 0
    iq = 0

    if len(quantlist) == 0:
        quantlist.append("Mass")

    sys.stdout.flush()

    # =========== LOOP THROUGH QUANTITIES ============

    for quant_name in quantlist:

        qstime = time.time()
        if quant_name != "Mass":
            if loadrad > 7.0:
                quant = eagleread(snapdir, 'PartType' + str(parttype) + '/' + quant_name, astro = True, dtype = np.float32)[0]
            else:
                quant, conv_astro_quant, aexp = curr_region.read_data(quant_name, astro = True, return_conv = True)
        else:
            quant = mass


        # Now perform core function: make SPH image
        if zpix is None:
            massimage, quantimage = imtools.make_sph_image_new(pos, mass, quant, DesNgb = 32, imsize = numpix, boxsize = lim, CamPos = [0,0,CP_z], CamDir = [0,0,-1], CamAngle = [theta, phi, rho], CamFOV = CamFOV, hsml = hsml, zrange = [0.01, 5*lim], make_deepcopy = False)
        else:
            massimage, quantimage = imtools.make_sph_image_new_3d(pos, mass, quant, DesNgb = 32, imsize = numpix, zpix = zpix, boxsize = lim, CamPos = cen, CamDir = CamDir, CamAngle = [theta,phi,rho], CamFOV = CamFOV, hsml = hsml, zrange = zrange, tau = tau, make_deepcopy = False)

        # Next line prevents pos being shifted again in (possible) next quantity
        shift = False

        if iim == 0:

            #set_trace()

            if len(massimage.shape) == 2:
                imstack[0,:,:] = massimage
            elif len(massimage.shape) == 3:
                imstack[0,:,:,:] = massimage
            else:
                print("Unexpected dimensions of massimage returned from SPH routine ({:d})." .format(len(massimage.shape)))
                sys.exit(400)

            iim += 1

        if quant_name != "Mass":
            if len(quantimage.shape) == 2:
                imstack[iim, :, :] = quantimage
            elif len(quantimage.shape) == 3:
                imstack[iim,:,:,:] = quantimage
            else:
                print("Unexpected dimensions of quantimage returned from SPH routine ({:d})." .format(len(massimage.shape)))
                sys.exit(400)

            iim += 1
        
        tlist[iq] = time.time()-qstime
        iq += 1


    # -------------  Tidying up at the end ---------------

    zred = 1.0/aexp - 1

    print("Done creating images at z={:.3f} for parttype {:d}" .format(zred, parttype))
    print("")
    print("   Total: {:.2f} sec." .format(np.sum(tlist)))
    print("      --- Read             = %.2f sec. (%.2f%%)" % (tlist[0], tlist[0]/np.sum(tlist)*100))
    
    for iq in range(len(quantlist)):
        print("      --- Q" + str(iq) + " ['" + quantlist[iq] + "'] = %.2f sec. (%.2f%%)" % (tlist[iq+1], tlist[iq+1]/np.sum(tlist)*100))
        

        #set_trace()

    return imstack, zred




    
# ------- Ends definition of function generate_images_dev() --------------




# --- ROTATEBYANGLES function --------        
            
def rotatebyangles(r,theta=0,phi=0,rho=0, separate=False):

    stime = time.time()

    r = np.asarray(r)    
    
    x = np.asarray(r[:,0])
    y = np.asarray(r[:,1])
    z = np.asarray(r[:,2])

    thetarad = theta * np.pi/180
    phirad = theta * np.pi/180
    rhorad = rho * np.pi/180

    ## First, some calculations that only need to be done once
    ## (to set up reference vectors)

    # Calculate the NEW north pole in the OLD coordinate system
    nx = math.sin(thetarad)*math.cos(phirad)
    ny = math.sin(thetarad)*math.sin(phirad)
    nz = math.cos(thetarad)

    # Point on new equator with phi = 0 (on great circle containing the old                       
    # north pole, which has phi = 180 deg)                                                     

    ex = np.cos(thetarad)*np.cos(phirad)
    ey = np.cos(thetarad)*np.sin(phirad)
    ez = -np.sin(thetarad)

    # Point on new equator with phi = 90 deg ("pluspoint")                                       
    # Cross product n (North pole) x e (point on equator with phi = 0)                            
    # So that old and new phi values agree if old and new North pole are the same               
  
    hx = ny*ez-nz*ey                                                                            
    hy = nz*ex-nx*ez                                                                            
    hz = nx*ey-ny*ex     
    
    tetime = time.time()
    print("   [RBA: finished with transformation setup (%.2f sec./%.2f sec.)" % (tetime-stime, tetime-stime))
    tstime = tetime

    ## Convert all particle coordinates to spherical polars        
    r_sph = ne.evaluate('sqrt(x**2+y**2+z**2)')
    theta_sph = ne.evaluate('arccos(z/r_sph)')
    phi_sph = ne.evaluate('arctan2(y, x)')

        
    # ... and check for any degeneracies (this will happen!!)
    wa = np.nonzero((x == 0) & (y == 0))[0]
    phi_sph[wa] = 0
    wc = np.nonzero(r_sph == 0)[0]
    theta_sph[wc] = 0

    tetime = time.time()
    print("   [RBA: finished converting to sph (%.2f sec./%.2f sec.)" % (tetime-tstime, tetime-stime))
    tstime = tetime

    
    ## Project all data points onto unit sphere
    px = np.sin(theta_sph)*np.cos(phi_sph)
    py = np.sin(theta_sph)*np.sin(phi_sph)
    pz = np.cos(theta_sph)

    tetime = time.time()
    print("   [RBA: finished projecting to unit sphere (%.2f sec./%.2f sec.)" % (tetime-tstime, tetime-stime))
    tstime = tetime

    
    ## Project all data points onto new equator:                                          
    llambda = -(px*nx + py*ny + pz*nz)
    pprojx = px + llambda * nx
    pprojy = py + llambda * ny
    pprojz = pz + llambda * nz

    tetime = time.time()
    print("   [RBA: finished projecting onto new equator (%.2f sec./%.2f sec.)" % (tetime-tstime, tetime-stime))
    tstime = tetime
    
    
    # Make these projected vectors unit-length:                                                  
    projveclength = np.sqrt(pprojx**2 + pprojy**2 + pprojz**2)
    pprojunitx = pprojx/projveclength
    pprojunity = pprojy/projveclength
    pprojunitz = pprojz/projveclength

    tetime = time.time()
    print("   [RBA: finished unilengthing projected vectors (%.2f sec./%.2f sec.)" % (tetime-tstime, tetime-stime))
    tstime = tetime
    

    # Form new theta values:
    thetadash = np.arccos(px*nx + py*ny + pz*nz)
    wg1 = np.nonzero(((px*nx+py*ny+pz*nz >= 1) | (px*nx+py*ny+pz*nz < -1)))[0]
    thetadash[wg1] = 0.0

    tetime = time.time()
    print("   [RBA: finished forming new theta (%.2f sec./%.2f sec.)" % (tetime-tstime, tetime-stime))
    tstime = tetime
    

    # Form magnitude of new phi angle:                                                            
    cosphistar = pprojunitx*ex + pprojunity*ey + pprojunitz*ez
    wax = np.nonzero(projveclength == 0)[0]
    cosphistar[wax] = 0

    # ... and deal with (numerical) clipping issues...
    wch = np.nonzero(cosphistar > 1)[0]
    cosphistar[wch] = 1
    wcl = np.nonzero(cosphistar < -1)
    cosphistar[wcl] = -1

    tetime = time.clock()
    print("   [RBA: finished forming cosphistar (%.2f sec./%.2f sec.)" % (tetime-tstime, tetime-stime))
    tstime = tetime
    
    
    # Form dot-product of data points with pluspoint
    dotproduct = px*hx + py*hy + pz*hz
    sign = dotproduct / (abs(dotproduct))

    wsz = np.nonzero(dotproduct == 0)[0]
    sign[wsz] = 1

    tetime = time.clock()
    print("   [RBA: finished forming sign (%.2f sec./%.2f sec.)" % (tetime-tstime, tetime-stime))
    tstime = tetime
    

    # Form new phi angle:
    phidash = sign * np.arccos(cosphistar)
    
    # Reduce new phi angle by given rho, to get rotation:
    phidash += rhorad

    tetime = time.clock()
    print("   [RBA: finished calculating phidash (%.2f sec./%.2f sec.)" % (tetime-tstime, tetime-stime))
    tstime = tetime
    

    ## Finally, convert back to Cartesian coordinates (dashed):
    xdash = r_sph * np.sin(thetadash) * np.cos(phidash)
    ydash = r_sph * np.sin(thetadash) * np.sin(phidash)
    zdash = r_sph * np.cos(thetadash)

    tetime = time.clock()
    print("   [RBA: finished converting to Cartesian (%.2f sec./%.2f sec.)" % (tetime-tstime, tetime-stime))
    tstime = tetime
    
                   
    if separate:
        return xdash, ydash, zdash
    else:
        returnarray = np.vstack((xdash, ydash, zdash)).T
        return returnarray
        


def rotation_matrix(axis, theta):
    """
    Return the rotation matrix associated with counterclockwise rotation about
    the given axis by theta radians.
    From http://stackoverflow.com/questions/6802577/python-rotation-of-3d-vector
    """
    axis = np.asarray(axis)
    theta = np.asarray(theta)
    axis = axis/math.sqrt(np.dot(axis, axis))
    a = math.cos(theta/2)
    b, c, d = -axis*math.sin(theta/2)
    aa, bb, cc, dd = a*a, b*b, c*c, d*d
    bc, ad, ac, ab, bd, cd = b*c, a*d, a*c, a*b, b*d, c*d
    return np.array([[aa+bb-cc-dd, 2*(bc+ad), 2*(bd-ac)],
                     [2*(bc-ad), aa+cc-bb-dd, 2*(cd+ab)],
                     [2*(bd+ac), 2*(cd-ab), aa+dd-bb-cc]])


def rotatebyaxis(r, axis, theta):
    """
    Rotate points using matrix / axis formalism
    """
    
    return np.dot(r,rotation_matrix(axis,theta*np.pi/180))


def form_files(rootloc, isnap=29, types = 'sub', stype = 'snap'):
    
    """
    Creates the file names of hydrangea output
    """


    typelist = types.split()

    sstring = '%03d' % isnap

    if stype == 'snap':
        if 'snap' in types:
            snapdir = glob.glob(rootloc + '/data/snapshot_' + sstring + '_*')
        else:
            snapdir = glob.glob(rootloc + '/data/groups_' + sstring + '_*')
    elif stype == 'snip':
        snapdir = glob.glob(rootloc + '/data/snipshot_' + sstring + '_*')
    else:
        print("The snepshot type '" + stype + "' is not understood.")
        sys.exit(10)

    if len(snapdir) == 0:
        retlist = []
        for iout in range(len(typelist)):
            retlist.append(None)

        return retlist

    snapdir = snapdir[0]

    filename = (snapdir.split('/'))[-1]
    zstring = (filename.split('_'))[-1]

    retlist = []

    if 'sublum' in types:
        dirs = rootloc.split('/')
        if '/'.join(dirs[:4]) != '/virgo/simulations/Hydrangea' or dirs[6] != 'HYDRO':
            print("I believe that the simulation in '" + rootloc + "' is not a Hydrangea run - cannot get luminosity data.")
            set_trace()
            sys.exit(2911181147)

        if isnap != 29:
            print("Currently, luminosities are only available for snapshot 29. Please try again later.")
            sys.exit(2911181155)
        
        simdir = dirs[5]

    for type in typelist:
        if type == 'snap':
            if stype == 'snap':
                names = ('snapshot', 'snap')
            elif stype == 'snip':
                names = ('snipshot', 'snip')
            else:
                print("Snepshot type '" + stype + "' is not understood.")
                sys.exit()

        elif type == 'sub':
            names = ('groups', 'eagle_subfind_tab')
        elif type == 'subpart':
            names = ('particledata', 'eagle_subfind_particles')
        elif type == 'fof':
            names = ('groups', 'group_tab')
        elif types == 'sublum':
            names = ('', 'partMags_EMILES_PDXX_DUST_CH')
        else:
            print("The data type '" + types + "' is not understood. Please try another one.")
            sys.exit(2911181157)

        if types == 'sublum':
            filename = '/virgo/scratch/ybahe/PARTICLE_MAGS/' + simdir + '/HYDRO/data/' + names[1] + '_' + sstring + '_' + zstring + '.0.hdf5'
        else:
            filename = rootloc + '/data/' + names[0] + '_' + sstring + '_' + zstring + '/' + names[1] + '_' + sstring + '_' + zstring + '.0.hdf5'

        retlist.append(filename)
    
    if len(typelist) == 1:
        retlist = retlist[0]

    return retlist


def find_next_snap(aexp, dir='previous'):
    snap_list = ascii.read('/ptmp/mpa/ybahe/HYDRANGEA/OutputLists/hydrangea_snapshots.dat', guess = False, format = 'no_header')

    index = np.searchsorted(snap_list['col1'], aexp)

    if dir == 'previous':
        best_snap = index-1
    elif dir == 'next':
        best_snap = index
    else:
        print("Cannot understand '" + dir + "'...")
        sys.exit(100)

    # Account for extra non-in-sync snapshots:
    if best_snap > 18:
        best_snap += 1
    if best_snap > 25:
        best_snap += 1
       
    return best_snap


def snap_age(snapdir, type = 'lbt'):
    f = h5.File(snapdir,'r')
    header = f["/Header"]
    aexp = header.attrs["ExpansionFactor"]
    zred = 1/aexp-1

    lbt = Planck13.lookback_time(zred).value

    if type == 'lbt':
        return lbt

    elif type == 'aexp':
        return aexp
    
    elif type == 'zred':
        return zred

    else:
        print("Don't understand '" + type + "'")
        sys.exit(200)


def create_reverse_list(ids, delete_ids = False, cut = False, maxval = None):

    maxid = ids.max()

    if maxval is not None:
        if maxval > maxid:
            maxid = maxval

    if len(ids) > 2e9:
        dtype = np.int64
    else:
        dtype = np.int32
    
    if cut:
        ind_good = np.nonzero(ids >= 0)[0]
        ngood = len(ind_good)
    else:
        ind_good = np.arange(ids.shape[0], dtype = int)
        ngood = ids.shape[0]

    revlist = np.zeros(np.int64(maxid+1), dtype = dtype)-1
    revlist[ids[ind_good]] = ind_good

    if delete_ids:
        del ids

    return revlist

        


def interpolate_hsml(snipdir, rootdir):

    """
    This function determines the smoothing length of particles 
    """

    aexp_snip = snap_age(snipdir, 'aexp')
    lbt_snip = snap_age(snipdir, 'lbt')

    snap_prev = find_next_snap(aexp_snip, 'previous')
    snap_next = find_next_snap(aexp_snip, 'next')

    snapdir_prev = form_files(rootdir, isnap = snap_prev, types = 'snap')
    snapdir_next = form_files(rootdir, isnap = snap_next, types = 'snap')

    delta_t = snap_age(snapdir_prev, 'lbt')-snap_age(snapdir_next, 'lbt')
    t_next = snap_age(snapdir_next, 'lbt')

    # -----------------------------------
    # Ok. Now let the actual fun start...
    # General outline is as follows:
    # 1: Create A-aligned B-hsml list
    # 2: Create hsml interpolation coefficients
    # 3: Create interpolated hsml values
    # ------------------------------------

    # 1a: Reverse-index list for A
    ids_a = eagleread(snapdir_prev, 'PartType0/ParticleIDs', dtype = np.int64, astro = False)
    revinds_a = create_reverse_list(ids_a, delete_ids = True)

    len_a = len(ids_a)
    del ids_a

    # 1b: Load and align hsml_B
    hsml_b = eagleread(snapdir_next, 'PartType0/SmoothingLength', astro = False, dtype = np.float32)
    ids_b = eagleread(snapdir_next, 'PartType0/ParticleIDs', dtype = np.int64, astro = False)
    hsml_b_aligned_to_a = np.zeros(len_a, dtype = np.float32)-1

    hsml_b_aligned_to_a[revinds_a[ids_b]] = hsml_b

    # 1c: Free RevIndsB and hsmlB
    del ids_b
    del hsml_b
    
    # ---------------------------------------------------
    # ---> 3 units loaded: RevIndsA (x2), hsmlB_alignedA <---
    # ---------------------------------------------------

    # 2a: Compute interpolation coefficients
    hsml_a = eagleread(snapdir_prev, 'PartType0/SmoothingLength', astro = False, dtype = np.float32)
    hsml_alpha = (hsml_a - hsml_b_aligned_to_a)/(delta_t)
    hsml_beta = hsml_b_aligned_to_a - hsml_alpha*t_next

    # 2b: Deal with particles not present in snap B
    ind_gone = np.nonzero(hsml_b_aligned_to_a < 0)[0]

    hsml_alpha[ind_gone] = 0
    hsml_beta[ind_gone] = hsml_a[ind_gone]

    # 2c: Tidy up
    del hsml_a
    del hsml_b_aligned_to_a

    # -----------------------------------------------------

    # 3a: Interpolate to snipshot
    ids_snip = eagleread(snipdir, 'PartType0/ParticleIDs', astro = False, dtype = np.int64)
    
    hsml_snip = hsml_alpha[revinds_a[ids_snip]] * lbt_snip + hsml_beta[revinds_a[ids_snip]]
    
    del ids_snip
    del revinds_a
    del hsml_alpha
    del hsml_beta

    print("Done with interpolation!")

    return hsml_snip



def quantcode(var):

    """
    Helper routine that checks how a subhalo quantity should be treated during mergers of spurious subhaloes.
    """

    # Set up defaults
    code = 3
    refquant = ""

    varparts = var.split("/")
    if len(varparts) <= 1:
        print("Error - expected variable of type 'Subhalo/VAR', but was given '" + var + "'")
    
    var1 = varparts[1]  # The "main" variable (may be a group)

    if var1 in ["CentreOfMass", "CentreOfPotential", "GroupNumber", "IDMostBound", "SubGroupNumber", "Vmax", "VmaxRadius"]:
        code = 0

    if var1 in ["BlackHoleMass", "BlackHoleMassAccretionRate", "KineticEnergy", "Mass", "MassType", "StarFormationRate",
                "StellarInitialMass", "SubLength", "SubLengthType", "ThermalEnergy", "TotalEnergy" ]:
        code = 1

    if var1 in ["InitialMassWeightedBirthZ", "InitialMassWeightedStellarAge"]:
        code = 2
        refquant = "StellarInitialMass"

    if var1 == "NSF" or var1 == "SF" or var1 == "Stars":
        var2 = varparts[2]

        if var2 in ["KineticEnergy", "Mass", "MassFromAGB", "MassFromSNII", "MassFromSNIa", "ThermalEnergy", "TotalEnergy"]:
            code = 1

        if var2 in ["ElementAbundance", "IronFromSNIa", "IronFromSNIaSmoothed", "MassWeightedEntropy", "MassWeightedPotential", "MassWeightedTemperature", "Metallicity", "MetalsFromAGB", "MetalsFromSNII", "MetalsFromSNIa", "SmoothedElementAbundance", "SmoothedMetallicity"]:
            code = 2
            refquant = var1 + "/Mass"

        if var2 in ["SFWeightedMetallicity", "SmoothedSFWeightedMetallicity"]:
            code = 2
            refquant = "StarFormationRate"

    return code, refquant



def periodic_3d_old(pos, boxsize, cen):

    set_trace()

    pos_rel = deepcopy(pos)
    pos_rel -= cen[None,:]

    for idim in range(3):

        offset = cen[idim] - boxsize/2.0
        
        if offset < 0:
            pos_rel[pos_rel[:, idim] > boxsize/2, idim] -= boxsize
        if offset > 0:
            pos_rel[pos_rel[:, idim] < boxsize/2, idim] += boxsize

    return pos_rel


def periodic_3d(pos, boxsize, cen):

    pos_rel = pos - cen[None, :]

    for idim in range(3):

        offset = cen[idim] - boxsize/2.0

        # The following if clauses are not strictly necessary,
        # because if they are not met, no element satisfies the flip
        # condition. But they might speed things up...

        if offset < 0:
            pos_rel[pos_rel[:, idim] > boxsize/2.0, idim] -= boxsize

        elif offset > 0:
            pos_rel[pos_rel[:, idim] < -boxsize/2.0, idim] += boxsize

    return pos_rel

def calculate_hi(rho = None, temp = None, sfr = None, xH = None, mgas = None,
                 nh = None, zred = None, zred_simple = None, Zgas = None,
                 densIsNH = False, calculateH2 = True, uvb = 'HM01'):

    msun_SI = 1.989e30         # m_sun in kg
    mpc_SI = 3.08567758e22     # Mpc in m
    mP_SI = 1.6726e-27         # Proton mass in kg
    
    if rho is None:
        print("Supplying gas density is non-optional.")
        return None

    if temp is None:
        print("Supplying gas temperature is non-optional.")
        return None

    temp_int = np.copy(temp)  # Because we'll modify it internally

    if calculateH2 is not None and sfr is None:
        print("You did not supply star formation rates, but want the molecular fractions to be calculated. This won't work.")
        return None
    
    if calculateH2 == 'GK11' and (mgas is None or rho is None or Zgas is None):
        print("The GK11 calculation requires the gas mass, density, and metallicity.")
        return None

    if not densIsNH:
        
        if xH is None:
            print("If the density is absolute, then the hydrogen fractions must be provided.")
            return None

        nH = rho * xH * (1e10 * msun_SI / mpc_SI**3) / mP_SI / 1e6   # last factor is m**-3 --> cm**-3 
        
    else:
        nH = rho
            
    if nh is not None:
        nH = nh

    # Artificially set temperature of SFR gas to 10^4 K
    if sfr is not None:
        temp_int[sfr > 0] = 1e4

    # Calculate collisional ionisation variables:
    lambdaVal = 315614.0/temp_int
    alphaa = 1.269e-13*lambdaVal**1.503/((1+(lambdaVal/0.522)**0.47)**1.923) # in cm^3 s^-1

    lambdat = 1.17e-10*(np.sqrt(temp_int)*np.exp(-157809.0/temp_int))/(1.0+np.sqrt(temp_int/1e5))


    if zred is None:
        zred = np.zeros_like(nH)

    if zred_simple is not None:
        zred = np.zeros_like(nH) + zred_simple

    # Numbers from Ali (Table A2):
    # Currently only for z = 0!!
    zred_tab = np.array((0.0, 1, 2, 3, 4, 5))
    log_n0_tab = np.array((-2.56, -2.29, -2.06, -2.13, -2.23, -2.35))   #[2.75423e-3
    alpha1_tab = np.array((-1.86, -2.94, -2.22, -1.99, -2.05, -2.63))
    alpha2_tab = np.array((-0.51, -0.90, -1.09, -0.88, -0.75, -0.57))
    beta_tab = np.array((2.83, 1.21, 1.75, 1.72, 1.93, 1.77))
    f_tab = np.array((0.01, 0.03, 0.03, 0.04, 0.02, 0.01))

    # UVB values from Rahmati+13:
    gamma_uvb_hm01_tab = np.array((8.34e-14, 7.39e-13, 1.50e-12, 1.16e-12, 7.92e-13, 5.43e-13))
    gamma_uvb_hm12_tab = np.array((2.27e-14, 3.42e-13, 8.98e-13, 8.74e-13, 6.14e-13, 4.57e-13))
    gamma_uvb_fg09_tab = np.array((3.99e-14, 3.03e-13, 6.00e-13, 5.53e-13, 4.31e-13, 3.52e-13))

    csi_log_n0 = interp1d(zred_tab, log_n0_tab, kind = 'cubic')
    csi_alpha1 = interp1d(zred_tab, alpha1_tab, kind = 'cubic')
    csi_alpha2 = interp1d(zred_tab, alpha2_tab, kind = 'cubic')
    csi_beta = interp1d(zred_tab, beta_tab, kind = 'cubic')
    csi_f = interp1d(zred_tab, f_tab, kind = 'cubic')
    csi_uvb_hm01 = interp1d(zred_tab, gamma_uvb_hm01_tab, kind = 'cubic')
    csi_uvb_hm12 = interp1d(zred_tab, gamma_uvb_hm12_tab, kind = 'cubic')
    csi_uvb_fg09 = interp1d(zred_tab, gamma_uvb_fg09_tab, kind = 'cubic')

    n0 = 10.0**(csi_log_n0(zred))
    alpha1 = csi_alpha1(zred)
    alpha2 = csi_alpha2(zred)
    beta = csi_beta(zred)
    f = csi_f(zred)

    if uvb == 'HM01':
        gamma_uvb = csi_uvb_hm01(zred)   # Haardt & Madau (2001)
    elif uvb == 'HM12':
        gamma_uvb = csi_uvb_hm12(zred)   # Haardt & Madau (2012)
    elif uvb == 'FG09':
        gamma_uvb = csi_uvb_fg09(zred)   # Faucher-Giguere+ (2009)
    else:
        print("I don't understand the UVB setting '" + uvb + "'")
        return None
    
    #print("n0 = ", n0, ", alpha1 = ", alpha1, ", alpha2 = ", alpha2, ", beta = ", beta, ", f = ", f)

    xi = (1.0-f)*(1.0+(nH/n0)**beta)**alpha1 + f*(1.0+(nH/n0))**alpha2
    #print("xi = ", xi)

    gamma_phot_dash = gamma_uvb*xi/nH

    # Now form the quadratic components:                                      
    aquad = alphaa+lambdat
    bquad = 2.0*alphaa+gamma_phot_dash+lambdat
    cquad = alphaa
        
    """
    for ii in range(len(nH)):
        print("DEBUG: nH={:.2E}, T={:.2E} ==> lambdaVal = {:.2E}, alphaa = {:.2E}, lambdat = {:.2E}, gamma_phot_dash={:.2E}, aquad = {:.2E}, bquad = {:.2E}, cquad = {:.2E}"
              .format(nH[ii], temp_int[ii], lambdaVal[ii], alphaa[ii], lambdat[ii], gamma_phot_dash[ii], aquad[ii], bquad[ii], cquad[ii]))
    """

    # And FINALLY form HI fraction:                                          

    f_neutral = (bquad-np.sqrt(bquad**2-4.0*aquad*cquad))/(2.0*aquad)
    if not densIsNH:
        f_neutral *= xH   

    #print("fNeutral", f_neutral)
    
    if calculateH2 == 'BR06':
        
        t_eos_star       = 8e3       # EoS normalisation temperature [K]               
        nH_eos_star      = 0.1       # EoS normalisation hydrogen number density [cm^-3]                                                                              
        XH_prim          = 0.752     # Primordial hydrogen abundance by mass           
        mu_mean_prim     = 1.2285    # Primordial mean molecular mass                  
        gamma_eos        = 5/3       # Slope of imposed equation of state        
        p_eos_star = nH_eos_star * t_eos_star / (XH_prim*mu_mean_prim) # EoS normalization

        print("p_eos_star = {:.3e}" .format(p_eos_star))

        # Now calculate H2 contribution
        # (using Blitz & Rosolowski formula)
        ind_sf = np.nonzero(sfr > 0)[0]
        peos = p_eos_star*(nH[ind_sf]/nH_eos_star)**(gamma_eos)
        rmol = (peos/3.5e4)**(0.92)

        print("nH[0] = {:.3e}, Rmol[0] = {:.3e}" .format(nH[ind_sf[0]], rmol[0]))
    
        fsubh2 = 1.0/(1.0+1.0/rmol)
        fsubh1 = 1.0-fsubh2

        f_h1 = np.copy(f_neutral)
        f_h2 = np.copy(f_neutral)

        f_h1[ind_sf] *= fsubh1
        f_h2[ind_sf] *= fsubh2

    elif calculateH2 == 'GK11':
        
        fsubh2 = fh2_gk11_yb18(sfr, mgas, Zgas, nH*f_neutral, rho, gamma_uvb)
        fsubh1 = 1.0-fsubh2

        f_h1 = np.copy(f_neutral)
        f_h2 = np.copy(f_neutral)

        f_h1 *= fsubh1
        f_h2 *= fsubh2

        
    else:  # If we're skipping H2 computation
        f_h1 = np.copy(f_neutral)
        f_h2 = np.zeros_like(f_neutral)

    return f_neutral, f_h1, f_h2


def fh2_gk11(sfr, mgas, zgas, nH, rho, temp, f_neutral, uvb):
    
    # Define relevant constants
    sfrMW = 3.68e-22   # SFR density in solar neighbourhood (Bonatto+11)     [gr/s/cm^2]
    zFloor = 1e-5      # Arbitrary floor applied to metallicities            [Z_solar]
    nNorm = 25.0   # Normalisation density                                   [cm^-3]

    mSun_g = 1.989e33   # Solar mass                                         [g]
    pc_cm  = 3.086e18   # Parsec                                             [cm]
    
    mu  = 1.2285  # Mean molecular weight [m_p]. Assumed no H2!
    mp = 1.66e-25 # Proton mass [g]
    gNewton = 6.67e-11 * 1e6/ 1e3    # Newton's constant in CGS
    gamma = 5/3
    kBoltz = 1.38e-23*1e3*1e4   # Boltzmann's constant in CGS

    # Get Jeans length
    lJeans = np.sqrt(gamma * kBoltz * temp / (mu * mp * gNewton * rho))

    # Get required values:
    U_MW = sfr / sfrMW * mgas * rho * lJeans
    D_MW = np.clip(zgas, zFloor, None)  # dust-to-gas ratio

    # Do actual calculation
    lambda_dStar = 0.0015 * np.log(1+(3*U_MW)**1.7)
    lambda_alpha = 2.5 * U_MW / (1+(0.5*U_MW)**2)
    lambda_s = 0.05 / (lambda_dStar + D_MW)
    lambda_g = (1+lambda_alpha*lambda_s + lambda_s*lambda_s)/(1+lambda_s)
    lambdaVal = np.log(1+lambda_g*D_MW**(3/7)*(U_MW/15)**(4/7))

    # Critical surface density (eq. 14 of GK11; in Msun pc^-2)
    sigmaCrit = 20 * lambdaVal**(4/7) / (D_MW + np.sqrt(1+U_MW*D_MW*D_MW)) 

    # Convert neutral volume density to surface density:
    sigmaNeutral = nH * mp * f_neutral * lJeans  # nH * mp = rho_H = rho_gas * X_H

    
    fsub_h2 = (1 + SigmaNeutral/SigmaCrit)**(-2)

    return fsub_h2


def fh2_gk11_yb18(sfr, mgas, Zgas, nH, rho, uvb, ZFloor = 1e-5, ZMW = 1.0):
    
    """
    This is a from-scratch new implementation of the GK11 procedure, Nov-2018
    
    It is based on equation (6) of GK11, which is interpreted as yielding the 
    ratio of n_H2 / n_HI [p.4]. The dust-to-gas ratio relative to the MW, D_MW
    is estimated as Z_smooth/Z_solar, as on p. 10 of GK11. The radiation field
    relative to the MW, U_MW, is estimated as rho_SFR / rho_SFR_B11, where the 
    latter is the (3D!) SFR estimate within 1kpc from the Sun of Bonatto+11. 
    In the simulation, rho_SFR_i is taken as SFR_i/(m_i/rho_i) for particle i.

    Note: currently the radiation is taken exclusively from the particle's SFR.
          If SFR=0, a floor corresponding to uvb/2.2e-12 is assumed.

    Input variables:
    
    sfr:  particle SFR, in M_sun/yr
    mgas: particle total gas mass, in 10^10 M_sun
    Zgas: smoothed particle metallicity, in Z_solar
    nH:   neutral hydrogen number density, in cm^-3. This is technically
          itself dependent on f_H2, but for now, keep it at original value.
    rho:  gas particle density, in 10^10 M_sun / kpc^3
    uvb:  UV-background field
    
    """ 
    
    # Define relevant constants
    rhoSFR_B11 = 2.5e-3   # rhoSFR near Sun, in M_sun / yr [Bonatto+11]
    nStar = 25.0  # p. 5 of GK11 [in cm^-3]
    UV_SolarNeighbourhood = 2.2e-12 # Taken from Claudia's code, origin unknown.

    # Calculate base variables U_MW and D_MW
    rhoSFR = sfr / (mgas/(rho*10))  # Converting rho to M_sun / kpc^3
    U_MW = rhoSFR/rhoSFR_B11
    U_MW = np.clip(U_MW, uvb/UV_SolarNeighbourhood, None)

    print("U_MW[0] = {:.3e}" .format(U_MW[0]))

    # dust-to-gas ratio relative to MW. Assumes (I think) that Z_MW = Z_sol and 
    # that the dust-to-gas ratio is proportional to Z (see p. 10 of GK11)
    # ZFloor is just a minimum for the (numerical) case of Z = 0
    # ZMW allows to set another MW metallicity (in Z_solar).
    D_MW = np.clip(ZMW*Zgas, ZFloor, None)  

    print("D_MW[0] = {:.3e}" .format(D_MW[0]))

    # Helper variables used in H2 formula (bottom of p. 5L of GK11)
    
    lambda_dStar = 0.0015 * np.log(1+(3*U_MW)**1.7)
    lambda_alpha = 2.5 * U_MW / (1+(0.5*U_MW)**2)
    lambda_s = 0.05 / (lambda_dStar + D_MW)
    lambda_g = (1+lambda_alpha*lambda_s + lambda_s*lambda_s)/(1+lambda_s)
    lambdaVal = np.log(1+lambda_g * D_MW**(3/7) * (U_MW/15)**(4/7))  # [eq. 8 of GK11]

    print("lambdaVal[0] = {:.3e}" .format(lambdaVal[0]))

    # Equation (6) of Gnedin & Kravtsov (2011)
    xval = lambdaVal**(3/7)*np.log(D_MW * nH / (lambdaVal*nStar))
    fsub_h2 = 1.0/(1+np.exp(-4*xval - 3*xval*xval*xval))

    print("xval[0] = {:.3e}, fsub_h2[0] = {:.3e}" .format(xval[0], fsub_h2[0]))

    # Now convert this to mH2/mNeutral for particle
    # The formula is derived from combining m_Hn = m_HI + m_H2,
    # n_Hn = n_HI + n_H2, m_HI = m_p*n_HI, m_H2 = 2*m_p*n_H2,
    # fsub_h2 = n_H2/n_Hn, and fRhoSub_h2 = m_H2/m_Hn.

    fRhoSub_h2 = 2*fsub_h2/(1+fsub_h2)

    print("fRhoSub_h2[0] = {:.3e}" .format(fRhoSub_h2[0]))

    return fRhoSub_h2

def fh2_gd14(mode = 0):

    """
    Gnedin & Draine (2014) parameterisation of HI/H2 partition.

    mode == 0 uses their equation (6), which gives n_H2/nH (presumably)
    mode == 1 uses their equation (8a), which gives Sigma_H2/Sigma_HI
    """

    fRhoSub_h2 = -1  # Dummy

    return fRhoSub_h2

class Spiderweb:
    
    """
    Collection of functions for using Spiderweb output
    """

    def __init__(self, rundir, filename = None, highlev = False):

        if filename is None:
            if highlev:
                self.filename = rundir + "/SpiderwebTables.hdf5"
            else:
                self.filename = rundir + "/highlev/SpiderwebTables.hdf5"
        else:
            if highlev:
                self.filename = rundir + "/" + filename
            else:
                self.filename = rundir + "/highlev/" + filename

    def sh_to_gal(self, sh, isnap, dealWithOOR = False):

        shi_to_gal = yb.read_hdf5(self.filename, "Subhalo/Snapshot_" + str(isnap).zfill(3) + "/Galaxy")
        
        if len(sh.shape) > 1:
            print("Cannot deal with matrix inputs...")
            set_trace()

        # Deal with possibility that SH input contains out-of-range values
        if np.max(sh) >= len(shi_to_gal) or np.min(sh) < 0:

            if dealWithOOR is False:
                print("Input contains out-of-range values!")
                set_trace()
            else:
                retArr = np.zeros(sh.shape[0], dtype = int)-1
                ind_good = np.nonzero((sh >= 0) & (sh < len(shi_to_gal)))[0]
                retArr[ind_good] = shi_to_gal[sh[ind_good]]

        else:
            retArr = shi_to_gal[sh]

        return retArr


    def gal_to_sh(self, gal, isnap):
        
        shiTable = yb.read_hdf5(self.filename, "SubHaloIndex")
        
        if isnap >= shiTable.shape[1]:
            print("Requested illegal target snapshot!")
            set_trace()

        if np.max(gal) >= shiTable.shape[0]:
            print("At least some input galaxies are out-of-range!")
            set_trace()

        return shiTable[gal, isnap]

    
    def next_shi(self, sh, isnap):
        
        shi_fwd_length = yb.read_hdf5(self.filename, "Subhalo/Snapshot" + str(isnap).zfill(3) + "/Forward/Length")
        shi_fwd_shi = yb.read_hdf5(self.filename, "Subhalo/Snapshot" + str(isnap).zfill(3) + "/Forward/SubHaloIndex")
        
        if np.max(sh) >= len(shi_fwd_length):
            print("Input contains out-of-range values!")
            set_trace()

        return shi_fwd_length[sh], shi_fwd_shi[sh]


    def prev_shi(self, sh, isnap):
        
        shi_rev_length = yb.read_hdf5(self.filename, "Subhalo/Snapshot" + str(isnap).zfill(3) + "/Reverse/Length")
        shi_rev_shi = yb.read_hdf5(self.filename, "Subhalo/Snapshot" + str(isnap).zfill(3) + "/Reverse/SubHaloIndex")
        
        if np.max(sh) >= len(shi_rev_length):
            print("Input contains out-of-range values!")
            set_trace()

        return shi_rev_length[sh], shi_rev_shi[sh]







