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
import f90nml as nml

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
drhodpsi = float(ncfile.variables['drhodpsi'].data)
dth = np.append(np.diff(th), 0)

phi = field.get_field(in_file, 'phi_igomega_by_mode', None)
ntot = field.get_field(in_file, 'ntot_igomega_by_mode', 0)
t_perp = field.get_field(in_file, 'tperp_igomega_by_mode', 0)
t_par = field.get_field(in_file, 'tpar_igomega_by_mode', 0)

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

# Extract input file from NetCDF and write to text
os.system('sh extract_input_file.sh ' + str(in_file) + ' > input_file.in')
gs2_in = nml.read('input_file.in')

# Calculate kxfac
rhoc = float(gs2_in['theta_grid_parameters']['rhoc'])
qinp = float(gs2_in['theta_grid_parameters']['qinp'])

kxfac = abs(qinp)/rhoc/abs(drhodpsi)

# Calculate the zonal flow velocity
zf_shear = 0.5 * kxfac * np.fft.ifft(- phi[:,:,0]*kx[np.newaxis,:]**2, axis=1).real* \
           nx*rho_star

# Make film of v_zf versus x
plot_options = {}
options = {'file_name':'zf_shear_vs_x',
           'film_dir':'analysis/zf',
           'frame_dir':'analysis/zf/film_frames',
           'xlabel':r'$x$ $(m)$',
           'ylabel':r"$v'_{ZF}$ $(v_{th,i}/\rho_i)$",
           'bbox_inches':'tight',
           'fps':30}

options['title'] = []
for it in range(nt):
    options['title'].append(r'Time = {0:04d} $\mu s$'.format(
                            int(np.round((t[it]-t[0])*1e6))))

pf.make_film_1d(x, zf_shear, plot_options=plot_options, options=options)












ncfile.close()
