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
        self.grho = np.array(ncfile.variables['grho'][:])
        self.bmag = np.array(ncfile.variables['bmag'][:])
        self.dtheta = np.append(np.diff(self.theta), 0)

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

    def read_phi(self):
        """
        Read the electrostatic potenential from the NetCDF file.
        """

        print('Reading phi...')
        self.phi = field.get_field(self.cdf_file, 'phi_igomega_by_mode', None)
        self.phi = field.field_to_real_space(self.phi)*self.rho_star

    def read_ntot(self):
        """
        Read the electrostatic potenential from the NetCDF file.
        """

        print('Reading ntot_i...')
        self.ntot_i = field.get_field(self.cdf_file, 'ntot_igomega_by_mode', 0)
        print('Reading ntot_e...')
        self.ntot_e = field.get_field(self.cdf_file, 'ntot_igomega_by_mode', 1)

        # Convert to real space
        self.ntot_i = field.field_to_real_space(self.ntot_i)*self.rho_star
        self.ntot_e = field.field_to_real_space(self.ntot_e)*self.rho_star

    def read_upar(self):
        """
        Read the parallel velocity.
        """

        print('Reading upar_i...')
        self.upar_i = field.get_field(self.cdf_file, 'upar_igomega_by_mode', 0)
        print('Reading upar_e...')
        self.upar_e = field.get_field(self.cdf_file, 'upar_igomega_by_mode', 1)

        self.upar_i = field.field_to_real_space(self.upar_i)*self.rho_star
        self.upar_e = field.field_to_real_space(self.upar_e)*self.rho_star

    def read_tpar(self):
        """
        Read the parallel temperature.
        """

        print('Reading tpar_i...')
        self.tpar_i = field.get_field(self.cdf_file, 'tpar_igomega_by_mode', 0)
        print('Reading tpar_e...')
        self.tpar_e = field.get_field(self.cdf_file, 'tpar_igomega_by_mode', 1)

        # Convert to real space
        self.tpar_i = field.field_to_real_space(self.tpar_i)*self.rho_star
        self.tpar_e = field.field_to_real_space(self.tpar_e)*self.rho_star

    def read_tperp(self):
        """
        Read the perpendicular temperature.
        """

        print('Reading tperp_i...')
        self.tperp_i = field.get_field(self.cdf_file, 'tperp_igomega_by_mode', 0)
        print('Reading tperp_i...')
        self.tperp_e = field.get_field(self.cdf_file, 'tperp_igomega_by_mode', 1)

        # Convert to real space
        self.tperp_i = field.field_to_real_space(self.tperp_i)*self.rho_star
        self.tperp_e = field.field_to_real_space(self.tperp_e)*self.rho_star

    def calculate_v_exb(self):
        """
        Calculates the radial ExB velocity in real space in units of v_th,i.
        """

        phi_k = field.get_field(self.cdf_file, 'phi_igomega_by_mode', None)

        self.v_exb = field.field_to_real_space(1j*self.ky*phi_k)*\
                     self.rho_star*self.vth

    def calculate_q(self):
        """
        Calculate the local heat flux Q(x, y) for the ion species.
        """

        # Need phi as a function of kx, ky so read directly from netcdf file
        self.phi = field.get_field(self.cdf_file, 'phi_igomega_by_mode', None)
        self.read_ntot()
        self.read_tperp()
        self.read_tpar()

        ncfile = Dataset(self.cdf_file, 'r')
        self.q_nc = np.array(ncfile.variables['es_heat_flux'][:])

        # Convert to real space
        v_exb = field.field_to_real_space(1j*self.ky*self.phi)
        ntot_i = self.ntot_i/self.rho_star
        tperp_i = self.tperp_i/self.rho_star
        tpar_i = self.tpar_i/self.rho_star

        self.q = ((tperp_i + tpar_i/2 + 3/2*ntot_i)*v_exb).real/2

        dnorm = self.dtheta/self.bmag/self.gradpar
        wgt = np.sum(dnorm*self.grho)

        self.q = self.q * dnorm[int(self.nth/2)] / wgt
