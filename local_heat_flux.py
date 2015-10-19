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
import plot_style
import field_helper as field
plot_style.white()

# Read NetCDF
in_file = sys.argv[1]
ncfile = Dataset(in_file, 'r')
kx = np.array(ncfile.variables['kx'][:])
ky = np.array(ncfile.variables['ky'][:])
th = np.array(ncfile.variables['theta'][:])
t = np.array(ncfile.variables['t'][:])
gradpar = np.array(ncfile.variables['gradpar'][:])
grho = np.array(ncfile.variables['grho'][:])
bmag = np.array(ncfile.variables['bmag'][:])
dth = np.append(np.diff(th), 0)

phi = field.get_field(in_file, 'phi_igomega_by_mode', None)
ntot = field.get_field(in_file, 'ntot_igomega_by_mode', 0)
t_perp = field.get_field(in_file, 'tperp_igomega_by_mode', 0)
t_par = field.get_field(in_file, 'tpar_igomega_by_mode', 0)

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

# Convert to real space
v_exb = field.field_to_real_space(1j*ky*phi)
ntot = field.field_to_real_space(ntot)
t_perp = field.field_to_real_space(t_perp)
t_par = field.field_to_real_space(t_par)

q = ((t_perp + t_par/2 + 3/2*ntot)*v_exb).real/2

q_max = np.max(q)
q_min = np.min(q)                          
if np.abs(q_max) > np.abs(q_min):
    contours = np.around(np.linspace(-q_max, q_max, 20), 7)    
else:                                                               
    contours = np.around(np.linspace(q_min, -q_min, 20), 7)    

plot_options = {'levels':contours, 'cmap':'seismic'}
options = {'file_name':'q_vs_x_vs_y',       
           'film_dir':'analysis/local_q',
           'frame_dir':'analysis/local_q/film_frames',   
           'aspect':'equal',                                            
           'xlabel':r'$x (m)$',                                         
           'ylabel':r'$y (m)$',                                         
           'cbar_ticks':5,                                
           'cbar_label':r'$Q_i(x, y) / Q_{gB}$',                          
           'bbox_inches':'tight',
           'fps':30}

options['title'] = []                                                   
for it in range(nt):                                               
    options['title'].append(r'Time = {0:04d} $\mu s$'.format(           
                            int(np.round((t[it]-t[0])*1e6))))

pf.make_film_2d(x, y, q[:,:,:], plot_options=plot_options, options=options)
