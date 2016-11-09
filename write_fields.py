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

def write_v_exb(run, lab_frame):
    """
    Write the radial ExB velocity to a NetCDF file.
    """
    run.calculate_v_exb()

    os.system('mkdir -p ' + run.run_dir + 'analysis/write_fields')

    #interpolate radial coordinate to be approx 0.5cm
    interp_fac = int(np.ceil(run.r[int(run.nx/2)+1]/0.005))
    x_nc = np.linspace(min(run.r), max(run.r), interp_fac*run.nx)
    field_interp = np.empty([run.nt, len(x_nc), run.ny],
                            dtype=float)
    for it in range(run.nt):
        for iy in range(run.ny):
                f = interp.interp1d(run.r,
                                    run.v_exb[it,:,iy])
                field_interp[it,:,iy] = f(x_nc)

    if lab_frame:
        nc_file = Dataset(run.run_dir +
                          'analysis/write_fields/v_exb_lab_frame.cdf', 'w')
    else:
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
    nc_x[:] = x_nc[:]
    nc_y[:] = run.z[:]
    nc_t[:] = run.t[:] - run.t[0]
    nc_file.close()

    run.v_exb = None
    gc.collect()

def write_phi2(run, lab_frame):
    """
    Write the radial ExB velocity to a NetCDF file.
    """
    run.read_phi()

    os.system('mkdir -p ' + run.run_dir + 'analysis/write_fields')

    #interpolate radial coordinate to be approx 0.5cm
    interp_fac = int(np.ceil(run.r[int(run.nx/2)+1]/0.005))
    x_nc = np.linspace(min(run.r), max(run.r), interp_fac*run.nx)
    field_interp = np.empty([run.nt, len(x_nc), run.ny],
                            dtype=float)
    for it in range(run.nt):
        for iy in range(run.ny):
                f = interp.interp1d(run.r,
                                    run.phi[it,:,iy])
                field_interp[it,:,iy] = f(x_nc)

    if lab_frame:
        nc_file = Dataset(run.run_dir +
                          'analysis/write_fields/phi_lab_frame.cdf', 'w')
    else:
        nc_file = Dataset(run.run_dir + 'analysis/write_fields/phi.cdf', 'w')

    nc_file.createDimension('x', len(x_nc))
    nc_file.createDimension('y', run.ny)
    nc_file.createDimension('t', run.nt)
    nc_file.createDimension('none', 1)
    nc_nref = nc_file.createVariable('nref','d', ('none',))
    nc_tref = nc_file.createVariable('tref','d', ('none',))
    nc_x = nc_file.createVariable('x','d',('x',))
    nc_y = nc_file.createVariable('y','d',('y',))
    nc_t = nc_file.createVariable('t','d',('t',))
    nc_field = nc_file.createVariable('phi2',
                                  'd',('t', 'x', 'y'))

    nc_field[:,:,:] = field_interp[:,:,:]
    nc_nref[:] = run.nref
    nc_tref[:] = run.tref
    nc_x[:] = x_nc[:]
    nc_y[:] = run.z[:]
    nc_t[:] = run.t[:] - run.t[0]
    nc_file.close()

    run.phi = None
    gc.collect()

def write_ntot_i(run, lab_frame):
    """
    Write the radial ExB velocity to a NetCDF file.
    """
    run.read_ntot()

    os.system('mkdir -p ' + run.run_dir + 'analysis/write_fields')

    #interpolate radial coordinate to be approx 0.5cm
    interp_fac = int(np.ceil(run.r[int(run.nx/2)+1]/0.005))
    x_nc = np.linspace(min(run.r), max(run.r), interp_fac*run.nx)
    field_interp = np.empty([run.nt, len(x_nc), run.ny],
                            dtype=float)
    for it in range(run.nt):
        for iy in range(run.ny):
                f = interp.interp1d(run.r,
                                    run.ntot_i[it,:,iy])
                field_interp[it,:,iy] = f(x_nc)

    if lab_frame:
        nc_file = Dataset(run.run_dir +
                          'analysis/write_fields/ntot_i_lab_frame.cdf', 'w')
    else:
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
    nc_x[:] = x_nc[:]
    nc_y[:] = run.z[:]
    nc_t[:] = run.t[:] - run.t[0]
    nc_file.close()

    run.ntot_i = None
    run.ntot_e = None
    gc.collect()

def write_ntot_3d(run, lab_frame):
    """
    Write the radial ExB velocity to a NetCDF file.
    """
    run.read_ntot_3d()

    os.system('mkdir -p ' + run.run_dir + 'analysis/write_fields')

    # Use original GS2 x, y, theta since box is not rectangular for any theta
    # besides theta = 0.
    kx = run.kx*run.drho_dpsi
    ky = run.ky*run.drho_dpsi
    x_gs2 = np.linspace(-np.pi/kx[1], np.pi/kx[1], run.nx, endpoint=False)
    y_gs2 = np.linspace(-np.pi/ky[1], np.pi/ky[1], run.ny, endpoint=False)
    x_gs2 -= x_gs2[int(run.nx/2)]
    y_gs2 -= y_gs2[int(run.ny/2)]

    if lab_frame:
        nc_file = Dataset(run.run_dir +
                          'analysis/write_fields/ntot_3d_lab_frame.cdf', 'w')
    else:
        nc_file = Dataset(run.run_dir + 'analysis/write_fields/ntot_3d.cdf', 'w')

    nc_file.createDimension('x', run.nx)
    nc_file.createDimension('y', run.ny)
    nc_file.createDimension('theta', run.nth)
    nc_file.createDimension('t', run.nt)
    nc_file.createDimension('none', 1)
    nc_nref = nc_file.createVariable('nref','d', ('none',))
    nc_tref = nc_file.createVariable('tref','d', ('none',))
    nc_x = nc_file.createVariable('x','d',('x',))
    nc_y = nc_file.createVariable('y','d',('y',))
    nc_theta = nc_file.createVariable('theta','d',('theta',))
    nc_t = nc_file.createVariable('t','d',('t',))
    nc_field = nc_file.createVariable('ntot_i',
                                  'd',('t', 'x', 'y', 'theta'))

    nc_field[:,:,:] = run.ntot_i[:,:,:,:]
    nc_nref[:] = run.nref
    nc_tref[:] = run.tref
    nc_x[:] = x_gs2[:]
    nc_y[:] = y_gs2[:]
    nc_theta[:] = run.theta[:]
    nc_t[:] = run.t[:] - run.t[0]
    nc_file.close()

    run.ntot_i = None
    gc.collect()

run = Run(sys.argv[1])
lab_frame = False

write_v_exb(run)
write_phi2(run)
write_ntot_i(run)
# write_ntot_3d(run)

