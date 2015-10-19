# Standard
import os
import sys

# Third Party
import numpy as np
from scipy.io import netcdf
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import seaborn as sns
import pyfilm as pf
plt.rcParams.update({'figure.autolayout': True})
mpl.rcParams['axes.unicode_minus']=False                                        

#local
import plot_style
import field_helper as field
plot_style.white()

# Read NetCDF
in_file = sys.argv[1]
ncfile = netcdf.netcdf_file(in_file, 'r')
kx = np.array(ncfile.variables['kx'][:])
ky = np.array(ncfile.variables['ky'][:])
th = np.array(ncfile.variables['theta'][:])
t = np.array(ncfile.variables['t'][:])
gradpar = np.array(ncfile.variables['gradpar'][:])

print('Reading ntot_i...')
ntot_i = field.get_field(in_file, 'ntot_igomega_by_mode', 0)
print('Reading ntot_e...')
ntot_e = field.get_field(in_file, 'ntot_igomega_by_mode', 1)

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

# Convert to real space
ntot_i = field.field_to_real_space(ntot_i)*rho_star
ntot_e = field.field_to_real_space(ntot_e)*rho_star

# Ion density film
contours = field.calculate_contours(ntot_i)

plot_options = {'levels':contours, 'cmap':'seismic'}
options = {'file_name':'ntot_i',       
           'film_dir':'analysis/moments',
           'frame_dir':'analysis/moments/film_frames',   
           'aspect':'equal',                                            
           'xlabel':r'$x (m)$',                                         
           'ylabel':r'$y (m)$',                                         
           'cbar_ticks':5,                                
           'cbar_label':r'$\delta n_i / n_r$',                          
           'bbox_inches':'tight',
           'fps':30}

options['title'] = []                                                   
for it in range(nt):                                               
    options['title'].append(r'Time = {0:04d} $\mu s$'.format(           
                            int(np.round((t[it]-t[0])*1e6))))

pf.make_film_2d(x, y, ntot_i, plot_options=plot_options, options=options)

#Electron density film
contours = field.calculate_contours(ntot_e)

plot_options = {'levels':contours, 'cmap':'seismic'}
options = {'file_name':'ntot_e',       
           'film_dir':'analysis/moments',
           'frame_dir':'analysis/moments/film_frames',   
           'aspect':'equal',                                            
           'xlabel':r'$x (m)$',                                         
           'ylabel':r'$y (m)$',                                         
           'cbar_ticks':5,                                
           'cbar_label':r'$\delta n_e / n_r$',                          
           'bbox_inches':'tight',
           'fps':30}

options['title'] = []                                                   
for it in range(nt):                                               
    options['title'].append(r'Time = {0:04d} $\mu s$'.format(           
                            int(np.round((t[it]-t[0])*1e6))))

pf.make_film_2d(x, y, ntot_e, plot_options=plot_options, options=options)