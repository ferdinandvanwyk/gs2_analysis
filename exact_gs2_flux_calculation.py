# Standard
import os
import sys
import gc #garbage collector
import configparser
import logging
import operator #enumerate list
import multiprocessing

# Third Party
import numpy as np
from scipy.io import netcdf
import scipy.interpolate as interp
import scipy.optimize as opt
import scipy.signal as sig
import matplotlib.pyplot as plt
plt.rcParams.update({'figure.autolayout': True})
from mpl_toolkits.axes_grid1 import make_axes_locatable
import seaborn as sns
sns.set_context('talk')

def field_to_real_space(field):
    """
    Converts field from (kx, ky) to (x, y) and saves as new array attribute.
    """

    field_real_space = np.empty([nt,nx,ny],dtype=float)
    field_real_space = np.fft.irfft2(field, axes=[1,2])
    field_real_space = np.roll(field_real_space, int(nx/2), axis=1)

    return field_real_space*nx*ny

def field_to_real_space_no_time(field):
    """
    Converts field from (kx, ky) to (x, y) and saves as new array attribute.
    """

    field_real_space = np.empty([nx,ny,nth])
    field_real_space = np.fft.irfft2(field, axes=[0,1])
    #field_real_space = np.roll(field_real_space, int(nx/2), axis=0)

    return field_real_space

# Read NetCDF
in_file = sys.argv[1]
ncfile = netcdf.netcdf_file(in_file, 'r')
kx = np.array(ncfile.variables['kx'][:])
ky = np.array(ncfile.variables['ky'][:])
th = np.array(ncfile.variables['theta'][:])
t = np.array(ncfile.variables['t'][:])
gradpar = np.array(ncfile.variables['gradpar'][:])
grho = np.array(ncfile.variables['grho'][:])
bmag = np.array(ncfile.variables['bmag'][:])
dth = np.append(np.diff(th), 0)

phi = np.array(ncfile.variables['phi_igomega_by_mode'][:])
phi = np.swapaxes(phi, 1, 2)
phi = phi[:,:,:,0] + 1j*phi[:,:,:,1]

dens = np.array(ncfile.variables['ntot_igomega_by_mode'][:,0,:,:,:])
dens = np.swapaxes(dens, 1, 2)
dens = dens[:,:,:,0] + 1j*dens[:,:,:,1]

t_perp = np.array(ncfile.variables['tperp_igomega_by_mode'][:,0,:,:,:])
t_perp = np.swapaxes(t_perp, 1, 2)
t_perp = t_perp[:,:,:,0] + 1j*t_perp[:,:,:,1]

t_par = np.array(ncfile.variables['tpar_igomega_by_mode'][:,0,:,:,:])
t_par = np.swapaxes(t_par, 1, 2)
t_par = t_par[:,:,:,0] + 1j*t_par[:,:,:,1]

q_nc = np.array(ncfile.variables['es_heat_flux'][:])
part_nc = np.array(ncfile.variables['es_part_flux'][:])
q_perp = np.array(ncfile.variables['es_heat_flux_perp'][:])
q_par = np.array(ncfile.variables['es_heat_flux_par'][:])
q_by_ky = np.array(ncfile.variables['total_es_heat_flux_by_ky'][:])

t_perp_final = np.array(ncfile.variables['tperp'][0,:,:,:,:])
t_perp_final = np.swapaxes(t_perp_final, 0, 1)
t_perp_final = t_perp_final[:,:,:,0] + 1j*t_perp_final[:,:,:,1]

t_par_final = np.array(ncfile.variables['tpar'][0,:,:,:,:])
t_par_final = np.swapaxes(t_par_final, 0, 1)
t_par_final = t_par_final[:,:,:,0] + 1j*t_par_final[:,:,:,1]

ntot_final = np.array(ncfile.variables['ntot'][0,:,:,:,:])
ntot_final = np.swapaxes(ntot_final, 0, 1)
ntot_final = ntot_final[:,:,:,0] + 1j*ntot_final[:,:,:,1]

phi_final = np.array(ncfile.variables['phi'][:,:,:,:])
phi_final = np.swapaxes(phi_final, 0, 1)
phi_final = phi_final[:,:,:,0] + 1j*phi_final[:,:,:,1]

# Calculate sizes and real arrays
if 'analysis' not in os.listdir():
    os.system("mkdir analysis")
if 'misc' not in os.listdir('analysis'):
    os.system("mkdir analysis/misc")
nt = len(t)
nkx = len(kx)
nky = len(ky)
nth = len(th)
nx = nkx
ny = 2*(nky - 1)
t = t*amin/vth
x = np.linspace(0, 2*np.pi/kx[1], nx)*rhoref
y = np.linspace(0, 2*np.pi/ky[1], ny)*rhoref \
                     *np.tan(pitch_angle)

##################################
# Repeat GS2 calculation exactly #
##################################

wgt = np.sum(dth*grho/bmag/gradpar)
dnorm = dth/bmag/gradpar

part_gs2 = np.empty([nkx, nky])
for ikx in range(nkx):
    for iky in range(nky):
        part_gs2[ikx, iky] = np.sum((ntot_final[ikx,iky,:]* \
                                np.conj(phi_final[ikx,iky,:])*ky[iky]*dnorm).imag)/wgt
part_gs2 *= 0.5
part_gs2[:,1:] /= 2

print('part calc = ', np.sum(part_gs2))
print('part_nc = ', part_nc[-1,0])

##################################
# Repeat GS2 calculation exactly #
##################################

q_gs2 = np.empty([nkx, nky])
for ikx in range(nkx):
    for iky in range(nky):
        q_gs2[ikx, iky] = np.sum(((t_perp_final[ikx,iky,:] +
                          t_par_final[ikx,iky,:]/2 + 3/2*ntot_final[ikx,iky,:])* \
                                np.conj(phi_final[ikx,iky,:])*ky[iky]*dnorm).imag)/wgt
q_gs2 *= 0.5
q_gs2[:,1:] /= 2

print('Q calc = ', np.sum(q_gs2))
print('q_final_gs2 = ', q_nc[-1,0], q_perp[-1,0], q_par[-1,0]/2,
        q_perp[-1,0]+q_par[-1,0]/2)

ncfile.close()
sys.exit()

