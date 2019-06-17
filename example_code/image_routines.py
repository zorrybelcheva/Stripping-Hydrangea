"""
Collection of image routines to use in python programs
"""

import ctypes as c
import numpy as np
import sys
from pdb import set_trace
from copy import deepcopy
import time

def make_sph_image_new_3d(pos_external, mass_external, quant_external, hsml_external = None, DesNgb=48, imsize=500, boxsize = 10.0, hmax = None, zpix = 1, CamPos = [0,0,0], CamDir = [0,0,-1], CamAngle = [0,0,0], CamFOV = [0.0, 0.0], make_deepcopy = True, zrange = None, tau = 1e6, return_hsml = False, vel_external = None, vzrange = None):


    """
    **** IMPORTANT - NEED TO ADJUST PATH IN LINES 330 and 358 ******
    
    Function to generate an SPH-smoothed image from a set of particles
    
    It returns both a density map and a map of a mass-weighted quantity,
    if supplied.
    

    Input values:
    -------------

    pos_external: Particle coordinates [Nx3] 
    
    mass_external: Particle masses [N]

    quant_external: Quantity to interpolate [N]. If you only want to generate
                    a mass map, set this to the same value as mass_external.

    hsml_external {None}: Smoothing lengths to use [N]. Set to None to 
                          compute them internally (N.B.: this can be very 
                          slow for large N).
                    
    DesNgb {48}: Desired number of neighbours in smoothing kernel. Only 
                 significant if hsml is None.
                 Note: if there must be at least 4 times DesNgb particles in 
                       selection region, or a blank image is returned.

                       
    imsize {500}: Pixel sidelength of resulting images.

    boxsize {10.0}: HALF-sidelength of the box from which the image will
                    be generated. Uses same coordinate system as pos_external.

    hmax {None}: Maximum smoothing length. If unspecified, the value of 
                 boxsize is used.
   
    CamPos {[0,0,0]}: Position of the camera. Uses same coordinate system
                      as pos_external.

    CamDir {[0,0,-1]}: Viewing direction of the camera 
                       (i.e. default is down the z-axis).

    CamAngle {[0,0,0]}: Alternative way of specifying viewing direction by 
                        providing the angles (theta, phi, rho). Theta is 
                        the angle w.r.t. the z-axis, phi the angle w.r.t. the 
                        x-axis in the xy-plane, and rho the roll angle of the 
                        camera. Note: only used when CamDir = [0, 0, 0]

    CamFOV {[0, 0]}: Camera field of view, in degrees. The two values are for
                     the image x- and y-direction, but they should generally
                     be set to the same value. If this is 0, an orthogonal
                     projection is performed, including both particles 
                     in front of and behind the camera (within the selection 
                     region). Otherwise, the image includes perspectivic
                     projection and *only* shows particles in front of the
                     camera.
                        
    make_deepcopy {True}: Make a copy of the input data to work with. 
                          Generally advisable unless memory is a serious
                          constraint!

    zrange {None}: Two-element array which can be used to make the selection 
                   region non-cubic, by
                   specifying a different (image-)z range. If not specified, 
                   the z-range is the same as x/y for orthogonal projection,
                   and [0.03, 10.0] for perspectivic projection.

    zpix {1}: Number of image slices to construct internally in the (image-)z
              direction (see 'tau' below').

    tau {1e6}: 'Optical depth' coefficient for image generation. If zpix == 1,
               this has no effect. If zpix > 1, then the individual z-slices
               are combined in two different ways. First, they are simply
               added to give a total image in which all particles contribute
               equally. Second, a simplified ray-tracing is performed, in 
               which the density contribution of particles further away 
               from the camera is reduced by the density of pixels closer to
               the camera (i.e. features closer to the camera become more 
               prominent). A higher value of tau leads to stronger extinction,
               i.e. a more effective 'blocking out' of far-away structures.

               Note that this is currently only working correctly when 
               using perspectivic projection, i.e. CamFOV > 0.
    
    return_hsml {False}: If set to True, the smoothing lengths are returned
                         so they can be used again. This can e.g. speed things
                         up when multiple quantities are to be interpolated
                         over the same field.

    vel_external {None}: Gives the option to supply the particle velocities,
                         which will trigger 'velocity-split' mode. 

    vzrange {None}: Two-parameter array specifying the output velocity range,
                    if in velocity-split mode (otherwise ignored).

    

    --------
    RETURNS:
    --------

    result_weight [imsize, imsize, 2]: 
        Array containing the WEIGHT images (i.e. the mass surface density 
        maps). The first index in the third dimension contains the 
        'ray-traced' map, and the second the standard map. The two are 
        identical if zpix == 1. If smoothing lengths were not provided
        and there are too few particles in the selection region, this
        array is filled with zeros.

    result_quant [imsize, imsize, 2]: 
        As result_weight, but containing the mass-weighted interpolated
        quantities. 

    hsml_out [N]: Smoothing length, either from input or computed internally.
                  Only returned if return_hsml is True.

                  
    """

    
    
    print("Starting image preparation...")


    # Make copies of input data, if desired:

    if make_deepcopy:
        pos = deepcopy(pos_external)
        mass = deepcopy(mass_external)
        quant = deepcopy(quant_external)
    
        if vel_external is not None:
            vel = deepcopy(vel_external)
        
    else:
        pos = pos_external
        mass = mass_external
        quant = quant_external
        
        if vel_external is not None:
            vel = vel_external
    
    n_sel = pos.shape[0]
    print("   (input includes {:d} particles...)" .format(n_sel))

    # Force camera parameters to be in numpy.array form, and convert
    # field of view to radians

    CamPos = np.array(CamPos, dtype=np.float32)
    CamDir = np.array(CamDir, dtype=np.float32)
    CamAngle = np.array(CamAngle, dtype=np.float32)
    CamFOV = np.array(CamFOV, dtype=np.float32)/180.0*np.pi

    # Set up image parameters from input
    
    nxpix = imsize
    nypix = imsize
    nzpix = zpix

    if (CamFOV[0] < 1e-3):
        perspective = False
    else:
        perspective = True
        
    xmin, ymin, xmax, ymax = -boxsize, -boxsize, boxsize, boxsize

    if zrange is None:
        if perspective:
            zmin, zmax = 0.03, 10.0
        else:
            zmin, zmax = -boxsize, boxsize

    else:
        zrange = np.array(zrange)
        zmin, zmax = zrange[0], zrange[1]


    # Force position, mass, and quantity to be in correct precision:

    if pos.dtype != np.float32:
        pos = pos.astype(np.float32)

    if mass.dtype != np.float32:
        mass = mass.astype(np.float32)
        
    if quant.dtype != np.float32:
        quant = quant.astype(np.float32)

    if vel_external is not None:
        if vel.dtype != np.float32:
            vel = vel.astype(np.float32)

        # The C-code expects the 6D particle phase space coordinates, 
        # so we need to construct these:
        pos = np.concatenate((pos, vel), axis = 1)


    # Set up smoothing length array, depending on input:

    if hsml_external is None:
        hsml = np.zeros(n_sel, dtype = np.float32)
        flagUseHsml = 0
    else:
        print("Using pre-defined smoothing lengths...")

        if make_deepcopy:
            hsml = deepcopy(hsml_external)
        else:
            hsml = hsml_external

        if hsml.dtype != np.float32:
            hsml = hsml.astype(np.float32)
        flagUseHsml = 1


    # Set up output arrays:

    if vel_external is None:
        nPixOut = 2
    else:
        nPixOut = zpix
        
    resultW_py = np.zeros((nxpix, nypix, nPixOut), dtype = np.float32)
    resultQ_py = np.zeros((nxpix, nypix, nPixOut), dtype = np.float32)
            
    # Check if there are enough particles to compute smoothing lengths. 
    # If not, we can stop:

    if n_sel < 4*DesNgb and flagUseHsml == 0:

        print("   Too few particles for smoothing length determination.\n")
        print("   Either reduce DesNgb (current value: %d), or change code" % DesNgb)
        print("   For now, we are just returning an empty image.")
    
        if return_hsml:
            return resultW_py, resultQ_py, np.zeros(n_sel, dtype = np.float32)
        else:
            return resultW_py, resultQ_py

    if hmax is None:
        hmax = boxsize

    # Now set things up so they can be used by the C-code in the 
    # external library...
        
    flagHsmlOnly = 0

    resultW = resultW_py.ctypes.data_as(c.c_void_p)
    resultQ = resultQ_py.ctypes.data_as(c.c_void_p)
 
    c_float_p = c.POINTER(c.c_float)

    pos_p = pos.ctypes.data_as(c.c_void_p)
    mass_p = mass.ctypes.data_as(c.c_void_p)
    hsml_p = hsml.ctypes.data_as(c.c_void_p)
    quant_p = quant.ctypes.data_as(c.c_void_p)

    CamPos_p = CamPos.ctypes.data_as(c.c_void_p)
    CamDir_p = CamDir.ctypes.data_as(c.c_void_p)
    CamAngle_p = CamAngle.ctypes.data_as(c.c_void_p)
    CamFOV_p = CamFOV.ctypes.data_as(c.c_void_p)
    
    c_n_sel = c.c_int(n_sel)

    c_xmin = c.c_float(xmin)
    c_xmax = c.c_float(xmax)
    c_ymin = c.c_float(ymin)
    c_ymax = c.c_float(ymax)
    c_zmin = c.c_float(zmin)
    c_zmax = c.c_float(zmax)

    c_nxpix = c.c_int(nxpix)
    c_nypix = c.c_int(nypix)
    c_nzpix = c.c_int(nzpix)
    c_DesNgb = c.c_int(DesNgb)
    c_hmax = c.c_float(hmax)
    c_flagUseHsml = c.c_long(flagUseHsml)
    c_flagHsmlOnly = c.c_int(flagHsmlOnly)


    if vel_external is None:
        c_Tau = c.c_float(tau)        
    else:
        c_vzmin = c.c_float(vzrange[0])
        c_vzmax = c.c_float(vzrange[1])
    

    print("Now performing SPH interpolation [nxpix={:d}, nypix={:d}, nzpix={:d}]..." .format(nxpix, nypix, nzpix))

    if vel_external is None:

        nargs = 25
        myargv = c.c_void_p * 25
        argv = myargv(c.addressof(c_n_sel), 
                      pos_p,
                      hsml_p,
                      mass_p,
                      quant_p,
                      c.addressof(c_xmin), c.addressof(c_xmax),
                      c.addressof(c_ymin), c.addressof(c_ymax),
                      c.addressof(c_zmin), c.addressof(c_zmax),
                      c.addressof(c_nxpix),c.addressof(c_nypix),
                      c.addressof(c_nzpix),
                      c.addressof(c_DesNgb), c.addressof(c_hmax), 
                      CamPos_p, CamDir_p, CamAngle_p, CamFOV_p,
                      resultW, resultQ, 
                      c.addressof(c_flagUseHsml), c.addressof(c_Tau), 
                      c.addressof(c_flagHsmlOnly))
        
        # *********** IMPORTANT ********************************
        # This next line needs to be modified to point
        # to the full path of where the library has been copied.
        # *******************************************************

        ObjectFile = "/net/quasar/data3/Hydrangea/EXAMPLE_CODE/HsmlAndProjectYB_Tau.so" 

    else:

        nargs = 26
        myargv = c.c_void_p * 26
        argv = myargv(c.addressof(c_n_sel), 
                      pos_p,
                      hsml_p,
                      mass_p,
                      quant_p,
                      c.addressof(c_xmin), c.addressof(c_xmax),
                      c.addressof(c_ymin), c.addressof(c_ymax),
                      c.addressof(c_zmin), c.addressof(c_zmax),
                      c.addressof(c_vzmin), c.addressof(c_vzmax),
                      c.addressof(c_nxpix),c.addressof(c_nypix),
                      c.addressof(c_nzpix),
                      c.addressof(c_DesNgb), c.addressof(c_hmax), 
                      CamPos_p, CamDir_p, CamAngle_p, CamFOV_p,
                      resultW, resultQ, 
                      c.addressof(c_flagUseHsml), 
                      c.addressof(c_flagHsmlOnly))
        
        
        # *********** IMPORTANT ********************************
        # This next line needs to be modified to point
        # to the full path of where the library has been copied.
        # *******************************************************

        ObjectFile = "/net/quasar/data3/Hydrangea/EXAMPLE_CODE/HsmlAndProjectYB_VelSplit.so"


    lib = c.cdll.LoadLibrary(ObjectFile)
    
    if vel_external is None:
        findHsmlAndProjectYB = lib.findHsmlAndProjectYB_Tau
    else:
        findHsmlAndProjectYB = lib.findHsmlAndProjectYB_VelSplit

    succ = findHsmlAndProjectYB(nargs, argv)

    print("Done with SPH interpolation!")

    resultW_out = np.transpose(resultW_py.astype(np.float), axes=[1,0,2])
    resultQ_out = np.transpose(resultQ_py.astype(np.float), axes=[1,0,2])
    
    if return_hsml:
        return resultW_out, resultQ_out, hsml
    else:
        return resultW_out, resultQ_out



