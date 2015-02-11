#!/usr/bin/python

import numpy as np
import cffi
import traceback
import libcloudphxx as libcl
from libcloudphxx.common import R_v, R_d, c_pd, eps
from setup import params, opts
import diagnostics as dg
import os
import json
import pdb

ptrfname = "/tmp/micro_step-" + str(os.getuid()) + "-" + str(os.getpid()) + ".ptr"

# CFFI stuff
ffi = cffi.FFI()
flib = ffi.dlopen('KiD_SC_2D.so')
clib = ffi.dlopen('ptrutil.so')

# C functions
ffi.cdef("void save_ptr(char*,void*);")

# Fortran functions (_sp_ means single precision)
ffi.cdef("void __main_MOD_main_loop();")

# object storing super-droplet model state (to be initialised)
prtcls = False

arrays = {}
timestep = 0
last_diag = -1

#savings some parameters from setup.py file and libcl revision number
params_write = params.copy()
# converting numpy objects to lists or strings, so json can save them
for key_ar in ["bins_qc_r20um", "bins_qc_r32um"]:
  params_write[key_ar] = params[key_ar].tolist()
for key_str in ["real_t"]:
  params_write[key_str] = str(params[key_str])
params_write["libcloudph_git_rev"] = libcl.git_revision

file_out = open("output/python_setup.txt", "w")
json.dump(params_write, file_out)
file_out.close()

def lognormal(lnr):
  from math import exp, log, sqrt, pi
  return params["n_tot"] * exp(
    -pow((lnr - log(params["meanr"])), 2) / 2 / pow(log(params["gstdv"]),2)
  ) / log(params["gstdv"]) / sqrt(2*pi);

def ptr2np(ptr, size_x, size_z):
  numpy_ar = np.frombuffer(
    ffi.buffer(ptr, size_x*size_z*np.dtype(params["real_t"]).itemsize),
    dtype=params["real_t"]
  ).reshape(size_x, size_z)
  return numpy_ar.squeeze()

def th_kid2dry(th, rv):
  return th * (1 + rv * R_v / R_d)**(R_d/c_pd)

def th_dry2kid(th_d, rv):
  return th_d * (1 + rv * R_v / R_d)**(-R_d/c_pd)

def rho_kid2dry(rho, rv):
  # KiD seems to define rho as (p_v + p_d) / (R_d T)
  return rho / (1 + rv / eps) 

@ffi.callback("bool(int, float, int, int, double*, double*, double*, double*, double*, double*, double*, double*, double*, double*, double*, double*, double*, double*)")
def micro_step(it_diag, dt, size_z, size_x, th_ar, qv_ar, rhof_ar, rhoh_ar, 
               uf_ar, uh_ar, wf_ar, wh_ar, xf_ar, zf_ar, xh_ar, zh_ar, tend_th_ar, tend_qv_ar):
  try:
    # global should be used for all variables defined in "if first_timestep"  
    global prtcls, dx, dz, timestep, last_diag

    print "\n timestep, last_diag", timestep, last_diag

    # superdroplets: initialisation (done only once)
    if timestep == 0:

      # first, removing the no-longer-needed pointer file
      os.unlink(ptrfname)

      arrx = ptr2np(xf_ar, size_x, 1)
      arrz = ptr2np(zf_ar, 1, size_z)

      # checking if grids are equal
      np.testing.assert_almost_equal((arrx[1:]-arrx[:-1]).max(), (arrx[1:]-arrx[:-1]).min(), decimal=7)
      np.testing.assert_almost_equal((arrz[1:]-arrz[:-1]).max(), (arrz[1:]-arrz[:-1]).min(), decimal=7)
      dx = arrx[1] - arrx[0]                            
      dz = arrz[1] - arrz[0]                            

      opts_init = libcl.lgrngn.opts_init_t()
      opts_init.dt = dt
      opts_init.nx, opts_init.nz = size_x - 2, size_z
      opts_init.dx, opts_init.dz = dx, dz 
      opts_init.x1, opts_init.z1 = dx * opts_init.nx, dz * opts_init.nz
      opts_init.sd_conc_mean = params["sd_conc"]
      opts_init.dry_distros = { params["kappa"] : lognormal }
      opts_init.sstp_cond, opts_init.sstp_coal = params["sstp_cond"], params["sstp_coal"]

      try:
        print("Trying with CUDA backend..."),
	prtcls = libcl.lgrngn.factory(libcl.lgrngn.backend_t.CUDA, opts_init)
        print (" OK!")
      except:
        print (" KO!")
        try:
          print("Trying with OpenMP backend..."),
	  prtcls = libcl.lgrngn.factory(libcl.lgrngn.backend_t.OpenMP, opts_init)
          print (" OK!")
        except:
          print (" KO!")
          print("Trying with serial backend..."),
	  prtcls = libcl.lgrngn.factory(libcl.lgrngn.backend_t.serial, opts_init)
          print (" OK!")
    
      # allocating arrays for those variables that are not ready to use
      # (i.e. either different size or value conversion needed)
      for name in ("thetad", "qv"):
	arrays[name] = np.empty((size_x-2, size_z))
      arrays["rhod"] = np.empty((size_z,))
      arrays["rhod_Cx"] = np.empty((size_x-1, size_z))
      arrays["rhod_Cz"] = np.empty((size_x-2, size_z+1))

    # defining qv and thetad (in every timestep) 
    arrays["qv"][:,:] = ptr2np(qv_ar, size_x, size_z)[1:-1, :]
    arrays["thetad"][:,:] = th_kid2dry(ptr2np(th_ar, size_x, size_z)[1:-1, :], arrays["qv"][:,:])
    np.set_printoptions(precision=12)
    print "qv[9,19] , qv[19,9]", ptr2np(qv_ar, size_x, size_z)[9,19], ptr2np(qv_ar, size_x, size_z)[19,9]
    print "qv[8:12,18:22], qv[18:22,8:12] w pythonie \n", ptr2np(qv_ar, size_x, size_z)[8:12,18:22], "\n", ptr2np(qv_ar, size_x, size_z)[18:22,8:12]

    print "SA: qv[19,9] w Pythonie", arrays["qv"][19,9], ptr2np(qv_ar, size_x, size_z)[20,9]

    # finalising initialisation
    if timestep == 0:
      arrays["rhod"][:] = rho_kid2dry(ptr2np(rhof_ar, 1, size_z)[:], arrays["qv"][0,:])
     
      arrays["rhod_Cx"][:,:] = ptr2np(uh_ar, size_x, size_z)[:-1, :]
      assert (arrays["rhod_Cx"][0,:] == arrays["rhod_Cx"][-1,:]).all()
      arrays["rhod_Cx"] *= ptr2np(rhof_ar, 1, size_z)[:] * dt / dx 

      arrays["rhod_Cz"][:, 1:] = ptr2np(wh_ar, size_x, size_z)[1:-1, :] 
      arrays["rhod_Cz"][:, 0 ] = 0
      arrays["rhod_Cz"][:, 1:] *= ptr2np(rhoh_ar, 1, size_z) * dt / dz

      prtcls.init(arrays["thetad"], arrays["qv"], arrays["rhod"], arrays["rhod_Cx"], arrays["rhod_Cz"]) 
      dg.diagnostics(prtcls, arrays, 1, size_x, size_z, timestep == 0) # writing down state at t=0

    div = abs((arrays["rhod_Cx"][0:-1,:]-arrays["rhod_Cx"][1:,:]) + (arrays["rhod_Cz"][:,0:-1]-arrays["rhod_Cz"][:,1:]))
    print "max div", div.max()
    print "mean div", div.mean()
    print "max bez pierwszego poziomu div[:,1:]", div[:,1:].max()
    print "mean bez pierwszego poziomu div[:,1:]", div[:,1:].mean()


    #pdb.set_trace()

    # spinup period logic
    opts.sedi = opts.coal = timestep >= params["spinup"]

    # superdroplets: all what have to be done within a timestep

    prtcls.step_sync(opts, arrays["thetad"], arrays["qv"],  arrays["rhod"]) 


    prtcls.step_async(opts)


    # calculating tendency for theta (first converting back to non-dry theta
    ptr2np(tend_th_ar, size_x, size_z)[1:-1, :] = - (
      ptr2np(th_ar, size_x, size_z)[1:-1, :] -   # old
      th_dry2kid(arrays["thetad"], arrays["qv"]) # new
    ) / dt #TODO: check if dt needed


    # calculating tendency for qv
    ptr2np(tend_qv_ar, size_x, size_z)[1:-1, :] = - (
      ptr2np(qv_ar, size_x, size_z)[1:-1, :] - # old                
      arrays["qv"]                             # new 
    ) / dt #TODO: check if dt needed    


    # diagnostics
    if last_diag < it_diag:
      dg.diagnostics(prtcls, arrays, it_diag, size_x, size_z, timestep == 0)
      last_diag = it_diag

    timestep += 1
  except:
    traceback.print_exc()
    return False
  else:
    return True
    
# storing pointers to Python functions
clib.save_ptr(ptrfname, micro_step)

# running Fortran stuff
# note: not using command line arguments, namelist name hardcoded in
#       kid_a_setup/namelists/SC_2D_input.nml 
flib.__main_MOD_main_loop()
