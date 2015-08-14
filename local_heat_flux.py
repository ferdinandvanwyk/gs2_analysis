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
import matplotlib as mpl
import matplotlib.pyplot as plt
plt.rcParams.update({'figure.autolayout': True})
from mpl_toolkits.axes_grid1 import make_axes_locatable
import seaborn as sns
plt.rcParams.update({'figure.autolayout': True})                                
mpl.rcParams['axes.unicode_minus']=False                                        
pal = sns.color_palette('deep')                                                 
sns.set_context('poster', font_scale=1.5, rc={"lines.linewidth": 5.})

#local
import plot_style
plot_style.white()

def field_to_real_space(field):
    """
    Converts field from (kx, ky) to (x, y) and saves as new array attribute.
    """

    field_real_space = np.empty([nt,nx,ny],dtype=float)
    field_real_space = np.fft.irfft2(field, axes=[1,2])
    field_real_space = np.roll(field_real_space, int(nx/2), axis=1)

    return field_real_space*nx*ny

os.system('mkdir analysis')

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

ntot = np.array(ncfile.variables['ntot_igomega_by_mode'][:,0,:,:,:])
ntot = np.swapaxes(ntot, 1, 2)
ntot = ntot[:,:,:,0] + 1j*ntot[:,:,:,1] 

t_perp = np.array(ncfile.variables['tperp_igomega_by_mode'][:,0,:,:,:])
t_perp = np.swapaxes(t_perp, 1, 2)
t_perp = t_perp[:,:,:,0] + 1j*t_perp[:,:,:,1] 

t_par = np.array(ncfile.variables['tpar_igomega_by_mode'][:,0,:,:,:])
t_par = np.swapaxes(t_par, 1, 2)
t_par = t_par[:,:,:,0] + 1j*t_par[:,:,:,1] 

q_nc = np.array(ncfile.variables['es_heat_flux'][:])

# Normalization parameters
# Outer scale in m
amin = 0.58044
# Thermal Velocity of reference species in m/s
vth = 1.4587e+05
# Larmor radius of reference species in m
rhoref = 6.0791e-03
# Expansion parameter
rho_star = rhoref/amin
# Angle between magnetic field lines and the horizontal in radians
pitch_angle = 0.6001
# Major radius at the outboard midplane
rmaj = 1.32574
# Reference density (m^-3)
nref = 1.3180e+19
# Reference temperature (kT)
tref = 2.2054e+02 / 8.6173324e-5 * 1.38e-23

nt = len(t)
nkx = len(kx)
nky = len(ky)
nth = len(th)
nx = nkx
ny = 2*(nky - 1)
t = t*amin/vth
x = np.linspace(-np.pi/kx[1], np.pi/kx[1], nx)*rhoref
y = np.linspace(-np.pi/ky[1], np.pi/ky[1], ny)*rhoref \
                     *np.tan(pitch_angle)

# Fourier correction
phi[:,:,1:] = phi[:,:,1:]/2
ntot[:,:,1:] = ntot[:,:,1:]/2
t_perp[:,:,1:] = t_perp[:,:,1:]/2
t_par[:,:,1:] = t_par[:,:,1:]/2

# Convert to real space
v_exb = field_to_real_space(1j*ky*phi)
ntot = field_to_real_space(ntot)
t_perp = field_to_real_space(t_perp)
t_par = field_to_real_space(t_par)

print(np.max(np.abs(t_perp)), np.max(np.abs(t_par)), np.max(np.abs(ntot)), np.max(np.abs(v_exb)))
q = ((t_perp + t_par/2 + 3/2*ntot)*v_exb).real/2

q_k = np.fft.rfft2(q, axes=[1,2])/nx/ny
print('q_k(0,0) = ', q_k[-1,0,0].real)
print('mean(q) = ', np.mean(q[-1,:,:]))

q_bg = np.empty([41])
for i in range(41):
    clip = i*10
    q_bg[i] = np.mean(np.clip(q[675,:,:], a_min = -clip, a_max=clip))

# Final time step density
ntot = ntot*rho_star
plt.clf()
contours = np.around(np.linspace(-0.09,0.09,41),5)
cbar_ticks = np.around(np.linspace(-0.09,0.09,5),7) 
ax = plt.subplot(111)
im = ax.contourf(x, y, np.transpose(ntot[675,:,:]),levels=contours, cmap='seismic')
plt.xlabel(r'$x (m)$')
plt.ylabel(r'$y (m)$')
ax.set_aspect('equal')
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)

plot_style.ticks_bottom_left(ax)
plt.colorbar(im, cax=cax, label=r'$\delta n / n [-]$', 
             ticks=cbar_ticks, format='%.2f')
plt.savefig('analysis/ntot_final.pdf')

# Final time step q
print(np.min(q[675,:,:]), np.max(q[675,:,:]))
plt.clf()
limit = 350
contours = np.around(np.linspace(-limit,limit,41),5)
cbar_ticks = np.around(np.linspace(-limit,limit,5),7) 
ax = plt.subplot(111)
im = ax.contourf(x, y, np.transpose(q[675,:,:]), levels=contours, cmap='seismic')
plt.xlabel(r'$x (m)$')
plt.ylabel(r'$y (m)$')
ax.set_aspect('equal')
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)

plot_style.ticks_bottom_left(ax)
plt.colorbar(im, cax=cax, label=r'$Q_{i,gB,local} [-]$', 
             ticks=cbar_ticks, format='%2d')
plt.savefig('analysis/qi_final.pdf')

# Contribution to heat flux for different max amplitudes
plt.clf()
fig, ax = plt.subplots(1,1)
plt.plot(np.arange(41)*10, q_bg/q_k[675,0,0])
plot_style.minor_grid(ax)
plot_style.ticks_bottom_left(ax)
plt.xlabel(r'Maximum $|Q_{i,gB,local}|$')
plt.ylabel(r'Relative ion heat flux contribution')
plt.savefig('analysis/qi_contrib.pdf')
plt.close()

# Film of radial heat flux profile
q_radial = np.mean(q, axis=2)
for it in range(nt):
    print('Saving frame %d of %d'%(it,nt))
    plt.clf()
    fig, ax = plt.subplots(1,1)
    plt.plot(x, q_radial[it,:])
    plt.xlabel(r'$x (m)$')
    plt.ylabel(r'Heat flux ($Q_{i,gB}$)')
    plt.ylim(0,20)
    plot_style.ticks_bottom_left(ax)
    plt.savefig("analysis/misc/film_frames/q_vs_x_%04d.png"%it)
    plt.close(fig)

os.system("avconv -threads 2 -y -f image2 -r 30 -i 'analysis/misc/film_frames/q_vs_x_%04d.png' -q 1 analysis/misc/q_vs_x.mp4")






