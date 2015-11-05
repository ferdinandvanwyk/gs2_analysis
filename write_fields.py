# Standard
import os
import sys
import gc

# Third Party
import numpy as np
from netCDF4 import Dataset
from scipy import interpolate as interp

# Local
from run import Run
import field_helper as field

def write_v_exb(cdf_file):
    """
    Write the radial ExB velocity to a NetCDF file.
    """
    run = Run(cdf_file)
    run.calculate_v_exb()

    os.system('mkdir -p ' + run.run_dir + 'analysis/write_fields')

    #interpolate radial coordinate to be approx 0.5cm
    interp_fac = int(np.ceil(run.x[int(run.nx/2)+1]/0.005))
    x_nc = np.linspace(min(run.x), max(run.x), interp_fac*run.nx)
    field_interp = np.empty([run.nt, len(x_nc), run.ny],
                            dtype=float)
    for it in range(run.nt):
        for iy in range(run.ny):
                f = interp.interp1d(run.x,
                                    run.v_exb[it,:,iy])
                field_interp[it,:,iy] = f(x_nc)

    nc_file = Dataset(run.run_dir + 'analysis/write_fields/v_exb.cdf', 'w')

    nc_file.createDimension('x', len(x_nc))
    nc_file.createDimension('y', run.ny)
    nc_file.createDimension('t', run.nt)
    nc_file.createDimension('none', 1)
    nc_nref = nc_file.createVariable('nref','d', ('none',))
    nc_tref = nc_file.createVariable('tref','d', ('none',))
    nc_x = nc_file.createVariable('x','d',('x',))
    nc_y = nc_file.createVariable('y','d',('y',))
    nc_t = nc_file.createVariable('t','d',('t',))
    nc_field = nc_file.createVariable('v_exb',
                                  'd',('t', 'x', 'y'))

    nc_field[:,:,:] = field_interp[:,:,:]
    nc_nref[:] = run.nref
    nc_tref[:] = run.tref
    nc_x[:] = x_nc[:] - x_nc[-1]/2
    nc_y[:] = run.y[:] - run.y[-1]/2
    nc_t[:] = run.t[:] - run.t[0]
    nc_file.close()

    run.v_exb = None
    gc.collect()

def write_ntot_i(cdf_file):
    """
    Write the radial ExB velocity to a NetCDF file.
    """
    run = Run(cdf_file)
    run.read_ntot()

    os.system('mkdir -p ' + run.run_dir + 'analysis/write_fields')

    #interpolate radial coordinate to be approx 0.5cm
    interp_fac = int(np.ceil(run.x[int(run.nx/2)+1]/0.005))
    x_nc = np.linspace(min(run.x), max(run.x), interp_fac*run.nx)
    field_interp = np.empty([run.nt, len(x_nc), run.ny],
                            dtype=float)
    for it in range(run.nt):
        for iy in range(run.ny):
                f = interp.interp1d(run.x,
                                    run.ntot_i[it,:,iy])
                field_interp[it,:,iy] = f(x_nc)

    nc_file = Dataset(run.run_dir + 'analysis/write_fields/ntot_i.cdf', 'w')

    nc_file.createDimension('x', len(x_nc))
    nc_file.createDimension('y', run.ny)
    nc_file.createDimension('t', run.nt)
    nc_file.createDimension('none', 1)
    nc_nref = nc_file.createVariable('nref','d', ('none',))
    nc_tref = nc_file.createVariable('tref','d', ('none',))
    nc_x = nc_file.createVariable('x','d',('x',))
    nc_y = nc_file.createVariable('y','d',('y',))
    nc_t = nc_file.createVariable('t','d',('t',))
    nc_field = nc_file.createVariable('ntot_i',
                                  'd',('t', 'x', 'y'))

    nc_field[:,:,:] = field_interp[:,:,:]
    nc_nref[:] = run.nref
    nc_tref[:] = run.tref
    nc_x[:] = x_nc[:] - x_nc[-1]/2
    nc_y[:] = run.y[:] - run.y[-1]/2
    nc_t[:] = run.t[:] - run.t[0]
    nc_file.close() 

    run.ntot_i = None
    run.ntot_e = None
    gc.collect()

run = Run(sys.argv[1])

write_v_exb(run)
write_ntot_i(run)

