"""
Collection of generic python helper programs
"""

import numpy as np
import os
import h5py as h5
from pdb import set_trace
import ctypes as c
import time

def binary_search(a, x, lo=0, hi=None):   # can't use a to specify default for hi
    hi = hi if hi is not None else len(a) # hi defaults to len(a)   
    pos = bisect_left(a,x,lo,hi)          # find insertion position
    return (pos if pos != hi and a[pos] == x else -1) # don't walk off the end

def dir(filename):
    return '/'.join((filename.split('/'))[:-1]) + '/'


def file_to_dir(filename):

    """ Convenience alias for backwards-compatibility """
    return dir(filename)



def write_hdf5(arr, outloc, name, new=False, comment = None, update = False):
    
    if new:
        if os.path.isfile(outloc):
            os.rename(outloc, outloc+'.old')

    f = h5.File(outloc,'a')

    if update and name in f:
        print("Dataset '" + name + "' already exists, updating...")
        write_comment = False
        dset = f[name]
    else:
        dset = f.create_dataset(name, arr.shape, dtype = arr.dtype)
        write_comment = True

    dset[...] = arr

    if comment is not None and write_comment:
        dset.attrs.create('Comment', np.string_(comment))

    f.close()

    return

def write_hdf5_attribute(filename, container, att_name, value, new = False, group = True):

    if new:
        if os.path.isfile(filename):
            os.rename(filename, filename+'.old')

    f = h5.File(filename, 'a')
    if container not in f.keys():

        if group:
            f.create_group(container)
        else:
            f.create_dataset(container)

    if att_name in f[container].attrs.keys():
        f[container].attrs.modify(att_name, value)
    else:
        f[container].attrs.create(att_name, value)

    return

def read_hdf5_attribute(filename, container, att_name):

    f = h5.File(filename, 'r')
    att = f[container].attrs[att_name]
    f.close()

    return att


def read_hdf5(file, var, first=None, stop=None):
    f = h5.File(file, 'r')

    dataset = f[var]
    dtype = f[var].dtype
    dshape = f[var].shape
    
    if first is None:
        first = 0
    if stop is None:
        stop = dshape[0]

    if first == 0 and stop == dshape[0]:
        outshape = f[var].shape
        data_out = np.empty(f[var].shape, f[var].dtype)
    else:
        outshape = np.zeros(len(f[var].shape), dtype = int)
        outshape[0] = stop-first
        outshape[1:] = dshape[1:]
        data_out = np.empty(outshape, dtype)

    if f[var].size:
        if first == 0 and stop == dshape[0]:
            f[var].read_direct(data_out)
        else:
            data_out = f[var][first:stop, ...]

    f.close()

    return data_out


def is_sorted(arr):
    return all(arr[i] <= arr[i+1] for i in range(len(arr)-1))

def match_keys(key1, key2,check_sort=False):
    
    if check_sort:
        if not is_sorted(key1):
            print("Input key1 needs to be sorted!")
            sys.exit()
    
    ind_trial = np.searchsorted(key1, key2)

    ind_clipped = np.nonzero(ind_trial >= len(key1))[0]
    ind_trial[ind_clipped] = len(key1)-1

    matches = key1[ind_trial]
    ind_goodmatch = np.nonzero(matches == key2)[0]

    print("Found {:d} matching keys (= {:.2f} per cent of KEY1, {:.2f} per cent of KEY2)..." .format(len(ind_goodmatch), len(ind_goodmatch)/len(key1)*100.0, len(ind_goodmatch)/len(key2)*100.0))

    return ind_trial[ind_goodmatch], ind_goodmatch


def ckatamaran_search(a, b):
    
    """ 
    Sped-up version of katamaran-search which does the loop in C.
    Assumes that a and b are unique and sorted.
    """

    if len(a) == 0:
        return np.zeros(0, dtype = np.int32)-1
        
    locs_a = np.zeros(len(a), dtype = np.int64)-1

    if len(b) == 0:
        return locs_a
        
    ObjectFile = "/u/ybahe/ANALYSIS/ckat.so"
    lib = c.cdll.LoadLibrary(ObjectFile)
    ckat = lib.ckat
    
    a_for_c = a.astype(np.int64)
    b_for_c = b.astype(np.int64)

    a_p = a_for_c.ctypes.data_as(c.c_void_p)
    b_p = b_for_c.ctypes.data_as(c.c_void_p)
        
    na_c = c.c_long(len(a))
    nb_c = c.c_long(len(b))
    locs_a_p = locs_a.ctypes.data_as(c.c_void_p)

    myargv = c.c_void_p * 5
    argv = myargv(a_p, b_p, 
                  c.addressof(na_c),
                  c.addressof(nb_c),
                  locs_a_p)
    
    succ = ckat(5, argv)
    return locs_a
    

def katamaran_search(a, b):

        """
        Perform a katamaran search to locate
        the index (if any) of the elements of a in b

        This assumes that the elements in a and b are unique, and
        that a and b are sorted
        """

        if len(a) == 0:
            return np.zeros(0, dtype = int)-1

        locs_a = np.zeros(len(a), dtype = int)-1

        if len(b) == 0:
            return locs_a
        
        ind_a = ind_b = 0
        len_a = len(a)
        len_b = len(b)

        val_a = a[ind_a]
        val_b = b[ind_b]
                
        while(True):

            if val_a > val_b:
                ind_b += 1
                if ind_b >= len_b:
                    break
                val_b = b[ind_b]
                continue

            if val_b > val_a:
                ind_a += 1
                if ind_a >= len_a:
                    break
                val_a = a[ind_a]
                continue

            if val_a == val_b:
                locs_a[ind_a] = ind_b
                ind_a += 1
                ind_b += 1
                if ind_a >= len_a or ind_b >= len_b:
                    break
                val_a = a[ind_a]
                val_b = b[ind_b]

                continue

        return locs_a

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


def find_id_indices(ids, reflist, max_direct = 1e10, min_c = 1e5):

    maxid_in = np.max(ids)
    maxid_ref = np.max(reflist)

    if maxid_in > max_direct or maxid_ref > max_direct:
        use_direct = False
    else:
        use_direct = True

    if use_direct:
        revlist = create_reverse_list(reflist, maxval = maxid_in)
        ind = revlist[ids]
        ind_match = np.nonzero(ind >= 0)[0]

    else:
        # Need to identify matching IDs in sh_ids by brute force
        
        tstart = time.time()
        sorter_in = np.argsort(ids)
        sorter_ref = np.argsort(reflist)
        tsort = time.time()-tstart
            
        tstart = time.time()
        if len(ids) > min_c  or len(reflist) >= min_c:
            ind_in_sorted_ref = ckatamaran_search(ids[sorter_in], reflist[sorter_ref])
        else:
            ind_in_sorted_ref = katamaran_search(ids[sorter_in], reflist[sorter_ref])

        ind_prematch_in = np.nonzero(ind_in_sorted_ref >= 0)[0]
        ind = np.zeros(len(ids), dtype = int)-1
        ind[sorter_in[ind_prematch_in]] = sorter_ref[ind_in_sorted_ref[ind_prematch_in]]
        ind_match = sorter_in[ind_prematch_in]

        tkat = time.time()-tstart
        
        print("Sorting took    {:.3f} sec." .format(tsort))
        print("Kat-search took {:.3f} sec." .format(tkat))

    return ind, ind_match
