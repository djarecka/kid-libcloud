import numpy as np
import libcloudphxx as libcl


# dictionary of simulation parameters
params = {
  "real_t" : np.float64,
  "backend" : libcl.lgrngn.backend_t.serial,
  "sd_conc" : 32.,
  "spinup" : 1801, # number of timesteps during which sedimentation and coalescence are off
  "kappa" : .61,
  "meanr" : .04e-6,
  "gstdv" : 1.4,
  "n_tot" : 100e6,
  "n_bins": 34,              # \__ from the TAU example file @ KiD-A website                   
  "bin0_D_upper" : 3.125e-6, # /                                                               
  # bins considered as cloud water (all below is aerosol, all above is rain) 
  "bins_qc_r20um" : np.arange(1, 12), # r=20 um threshold setting (actually D=39.7 um)
  "bins_qc_r32um" : np.arange(1, 14)  # r=32 um threshold setting (actually D=63.0 um)
}

opts = libcl.lgrngn.opts_t()
opts.sstp_cond = 1
opts.sstp_coal = 1
opts.cond = True
opts.coal = True
opts.adve = True
opts.sedi = True
opts.chem = False
opts.kernel = libcl.lgrngn.kernel_t.geometric
