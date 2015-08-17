import numpy as np
from scipy.io import netcdf

def field_to_real_space(field):
    """
    Converts field from (kx, ky) to (x, y) and saves as new array attribute.
    """
    shape = field.shape
    nt = shape[0]
    nx = shape[1]
    nky = shape[2]
    ny = 2*(nky - 1)

    field_real_space = np.empty([nt,nx,ny],dtype=float)
    field_real_space = np.fft.irfft2(field, axes=[1,2])
    field_real_space = np.roll(field_real_space, int(nx/2), axis=1)

    return field_real_space*nx*ny

def get_field(file_name, nc_var_name, spec_idx):
    """
    Read field from ncfile and prepare for calculations.

    * Read from NetCDF file
    * Swap axes order to be (t, x, y)
    * Convert to complex form
    * Apply fourier correction
    """
    nc_file = netcdf.netcdf_file(file_name, 'r')
    if spec_idx == None:
        nc_var = np.array(nc_file.variables[nc_var_name][:,:,:,:])
    else:
        nc_var = np.array(nc_file.variables[nc_var_name][:,spec_idx,:,:,:])

    nc_var = np.swapaxes(nc_var, 1, 2)
    nc_var = nc_var[:,:,:,0] + 1j*nc_var[:,:,:,1]

    nc_var[:,:,1:] = nc_var[:,:,1:]/2 

    nc_file.close()
    return(nc_var)
