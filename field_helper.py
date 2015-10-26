import numpy as np
from netCDF4 import Dataset

def get_field(cdf_file, nc_var_name, spec_idx):
    """
    Read field from ncfile and prepare for calculations.

    * Read from NetCDF file
    * Swap axes order to be (t, x, y)
    * Convert to complex form
    * Apply fourier correction
    """

    nc_file = Dataset(cdf_file, 'r')
    if spec_idx == None:
        nc_var = np.array(nc_file.variables[nc_var_name][:,:,:,:])
    else:
        nc_var = np.array(nc_file.variables[nc_var_name][:,spec_idx,:,:,:])

    nc_var = np.swapaxes(nc_var, 1, 2)
    nc_var = nc_var[:,:,:,0] + 1j*nc_var[:,:,:,1]

    nc_var[:,:,1:] = nc_var[:,:,1:]/2 

    nc_file.close()
    return(nc_var)

def get_field_final_timestep(cdf_file, nc_var_name, spec_idx):
    """
    Read field from ncfile and prepare for calculations.

    * Read from NetCDF file
    * Swap axes order to be (t, x, y)
    * Convert to complex form
    * Apply fourier correction
    """

    nc_file = Dataset(cdf_file, 'r')
    if spec_idx == None:
        nc_var = np.array(nc_file.variables[nc_var_name][:,:,:,:])
    else:
        nc_var = np.array(nc_file.variables[nc_var_name][spec_idx,:,:,:,:])

    nc_var = np.swapaxes(nc_var, 0, 1)
    nc_var = nc_var[:,:,:,0] + 1j*nc_var[:,:,:,1]

    nc_var[:,1:,:] = nc_var[:,1:,:]/2 

    nc_file.close()
    return(nc_var)

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

def calculate_contours(field, n_contours=20):
    """
    Calculates symmetric contours around zero.
    """
    f_max = np.max(field)
    f_min = np.min(field)
    if np.abs(f_max) > np.abs(f_min):
        contours = np.around(np.linspace(-f_max, f_max, n_contours), 7)    
    else:                                                               
        contours = np.around(np.linspace(f_min, -f_min, n_contours), 7)    

    return(contours)
