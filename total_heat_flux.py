# Standard
import os
import sys

# Third Party
import numpy as np
from netCDF4 import Dataset
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import seaborn as sns
import pyfilm as pf
plt.rcParams.update({'figure.autolayout': True})
mpl.rcParams['axes.unicode_minus']=False

#local
from run import Run
import plot_style
import field_helper as field
plot_style.white()

def field_to_real_space(field):
    """
    Converts field from (kx, ky) to (x, y) and saves as new array attribute.
    """
    shape = field.shape
    nx = shape[0]
    nky = shape[1]
    ny = 2*(nky - 1)
    nth = shape[2]

    field_real_space = np.empty([nx,ny,nth],dtype=float)
    field_real_space = np.fft.irfft2(field, axes=[0,1])
    field_real_space = np.roll(field_real_space, int(nx/2), axis=0)

    return field_real_space*nx*ny

cdf_file = sys.argv[1]
ncfile = Dataset(cdf_file, 'r')
phi = field.get_field_final_timestep(cdf_file, 'phi', None)
ntot = field.get_field_final_timestep(cdf_file, 'ntot', 0)
tperp = field.get_field_final_timestep(cdf_file, 'tperp', 0)
tpar = field.get_field_final_timestep(cdf_file, 'tpar', 0)
ky = np.array(ncfile.variables['ky'][:])
theta = np.array(ncfile.variables['theta'][:])
gradpar = np.array(ncfile.variables['gradpar'][:])
bmag = np.array(ncfile.variables['bmag'][:])
grho = np.array(ncfile.variables['grho'][:])
q_nc = np.array(ncfile.variables['es_heat_flux'][:, 0])

shape = phi.shape
nx = shape[0]
nky = shape[1]
ny = 2*(nky - 1)
nth = shape[2]

v_exb = field_to_real_space(1j*ky[np.newaxis, :, np.newaxis]*phi)
ntot = field_to_real_space(ntot)
tperp = field_to_real_space(tperp)
tpar = field_to_real_space(tpar)

dth = np.append(0, np.diff(theta))
dnorm = dth/bmag/gradpar
wgt = np.sum(dnorm*grho)

q_xy = ((tperp + tpar/2 + 3/2*ntot)*v_exb).real/2

q = np.empty([nx,ny], dtype=float)

for ix in range(nx):
    for iy in range(ny):
        q[ix,iy] = np.sum(q_xy[ix,iy,:] * dnorm)/wgt

print('GS2 calculation Q_gB =', q_nc[-1])
print('Moment calculation Q_gB = ', np.mean(q))
