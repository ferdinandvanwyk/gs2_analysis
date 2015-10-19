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

class Run(object):
    """
    Run object which stores basic simulation information.
    """

    def __init__(self, cdf_file):
        """
        Initialize with NetCDF file information.
        """
        self.cdf_file = cdf_file

        ncfile = Dataset(cdf_file, 'r')
        
        self.kx = np.array(ncfile.variables['kx'][:])
        self.ky = np.array(ncfile.variables['ky'][:])
        self.theta = np.array(ncfile.variables['theta'][:])
        self.t = np.array(ncfile.variables['t'][:])
        self.gradpar = np.array(ncfile.variables['gradpar'][:])

        ncfile.close()

        # Normalization parameters
        # Outer scale in m
        self.amin = 0.58044
        # Thermal Velocity of reference species in m/s
        self.vth = 1.4587e+05
        # Larmor radius of reference species in m
        self.rhoref = 6.0791e-03
        # Expansion parameter
        self.rho_star = self.rhoref/self.amin
        # Angle between magnetic field lines and the horizontal in radians
        self.pitch_angle = 0.6001
        # Major radius at the outboard midplane
        self.rmaj = 1.32574
        # Reference density (m^-3)
        self.nref = 1.3180e+19
        # Reference temperature (kT)
        self.tref = 2.2054e+02 / 8.6173324e-5 * 1.38e-23

        self.nt = len(self.t)
        self.nkx = len(self.kx)
        self.nky = len(self.ky)
        self.nth = len(self.theta)
        self.nx = self.nkx
        self.ny = 2*(self.nky - 1)
        self.t = self.t*self.amin/self.vth
        self.x = np.linspace(-np.pi/self.kx[1], np.pi/self.kx[1], 
                             self.nx)*self.rhoref
        self.y = np.linspace(-np.pi/self.ky[1], np.pi/self.ky[1], 
                             self.ny)*self.rhoref*np.tan(self.pitch_angle)




