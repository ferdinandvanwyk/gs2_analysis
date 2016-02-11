# This script plots the flux tube in all its 3D glory. i
#
###################################
# NEEDS TO BE CLEANED UP LATER!!! #
###################################

import os
import sys
import gc

# Third Party
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mayavi import mlab
from netCDF4 import Dataset
import seaborn as sns
import f90nml as nml
plt.rcParams.update({'figure.autolayout': True})
mpl.rcParams['axes.unicode_minus'] = False
from pyevtk.hl import gridToVTK

# Local
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

        # Normalization parameters
        # Radial location in terms of sqrt(psi_tor_N)
        self.rho_tor = 0.8
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
        # Angular rotation frequency (s^-1)
        self.omega = 4.7144e+04
        # Relates derivatives wrt to psi to those in real space
        self.dpsi_da = 1.09398

        self.cdf_file = cdf_file

        try:
            idx = self.cdf_file.rindex('/')
            self.run_dir = self.cdf_file[:idx] + '/'
        except ValueError:
            self.run_dir = ''

        ncfile = Dataset(cdf_file, 'r')

        self.drho_dpsi = ncfile.variables['drhodpsi'][0]
        self.kx = np.array(ncfile.variables['kx'][:])
        self.ky = np.array(ncfile.variables['ky'][:])
        self.theta = np.array(ncfile.variables['theta'][:])
        self.t = np.array(ncfile.variables['t'][:])
        self.gradpar = np.array(ncfile.variables['gradpar'][:])
        self.grho = np.array(ncfile.variables['grho'][:])
        self.bmag = np.array(ncfile.variables['bmag'][:])
        self.dtheta = np.append(np.diff(self.theta), 0)
        self.r_0 = np.array(ncfile.variables['Rplot'][:])*self.amin
        self.rprime = np.array(ncfile.variables['Rprime'][:])*self.amin
        self.z_0 = np.array(ncfile.variables['Zplot'][:])*self.amin
        self.zprime = np.array(ncfile.variables['Zprime'][:])*self.amin
        self.alpha_0 = np.array(ncfile.variables['aplot'][:])
        self.alpha_prime = np.array(ncfile.variables['aprime'][:])

        ncfile.close()

        # Extract input file from NetCDF and write to text
        os.system('sh extract_input_file.sh ' + str(self.cdf_file) +
                  ' > ' + self.run_dir + 'input_file.in')
        gs2_in = nml.read(self.run_dir + 'input_file.in')

        self.g_exb = float(gs2_in['dist_fn_knobs']['g_exb'])
        self.rhoc = float(gs2_in['theta_grid_parameters']['rhoc'])
        self.qinp = float(gs2_in['theta_grid_parameters']['qinp'])
        self.shat = float(gs2_in['theta_grid_parameters']['shat'])
        self.jtwist = float(gs2_in['kt_grids_box_parameters']['jtwist'])
        self.tprim_1 = float(gs2_in['species_parameters_1']['tprim'])
        self.fprim_1 = float(gs2_in['species_parameters_1']['fprim'])
        self.mass_1 = float(gs2_in['species_parameters_1']['mass'])
        if gs2_in['species_knobs']['nspec'] == 2:
            self.tprim_2 = float(gs2_in['species_parameters_2']['tprim'])
            self.fprim_2 = float(gs2_in['species_parameters_2']['fprim'])
            self.mass_2 = float(gs2_in['species_parameters_2']['mass'])


        # Toroidal mode number
        self.n0 = int(np.around(self.ky[1]/self.drho_dpsi*(self.amin/self.rhoref)))

        self.nt = len(self.t)
        self.nkx = len(self.kx)
        self.nky = len(self.ky)
        self.nth = len(self.theta)
        self.nx = self.nkx
        self.ny = 2*(self.nky - 1)
        self.t = self.t*self.amin/self.vth

    def read_phi(self):
        """
        Read the electrostatic potenential from the NetCDF file.
        """

        self.phi = field.get_field_final_timestep(self.cdf_file, 'phi', None)

        self.phi = field.field_to_real_space_final_timestep(self.phi)*\
                   self.rho_star


if __name__ == '__main__':

    run = Run(sys.argv[1])

    # Recalculate rho_star using new n0:
    run.rho_star = run.ky[1] / run.n0 / run.drho_dpsi

    run.read_phi()

    # Set q to closest rational q
    run.m_mode = int(np.around(run.qinp*run.n0))
    run.q_rational = float(run.m_mode)/float(run.n0)

    # Correct alpha read in from geometry
    run.alpha_0_corrected = run.alpha_0 + (run.qinp - run.q_rational)*run.theta

    # Correct alpha_prime
    run.q_prime = abs((run.alpha_prime[0] - run.alpha_prime[-1]) / (2 * np.pi))
    run.delta_rho = (run.rho_tor/run.q_rational) * (run.jtwist/(run.n0*run.shat))
    run.q_prime_corrected = run.jtwist / (run.n0 * run.delta_rho)
    run.alpha_prime_corrected = run.alpha_prime + (run.q_prime - \
                                    run.q_prime_corrected) * run.theta
    print(run.q_prime, run.q_prime_corrected)
    sys.exit()

    # Redo previous calculation
    run.x = 2*np.pi*np.linspace(0, 1/run.kx[1], run.nx, endpoint=False)
    run.y = 2*np.pi*np.linspace(0, 1/run.ky[1], run.ny, endpoint=False)

    # Calculate (rho - rho_n0) and call it rho_n and alpha
    run.rho_n = run.x * run.rhoc / run.q_rational * run.drho_dpsi * run.rho_star
    run.alpha = run.y * run.drho_dpsi * run.rho_star

    run.R = np.empty([run.nx, run.ny, run.nth], dtype=float)
    run.Z = np.empty([run.nx, run.ny, run.nth], dtype=float)
    run.phi_tor = np.empty([run.nx, run.ny, run.nth], dtype=float)
    for i in range(run.nx):
        for j in range(run.ny):
            for k in range(run.nth):
                run.R[i,j,k] = run.r_0[k] + run.rho_n[i] * run.rprime[k]
                run.Z[i,j,k] = run.z_0[k] + run.rho_n[i] * run.zprime[k]
                run.phi_tor[i,j,k] = run.alpha[j] - run.alpha_0_corrected[k] - \
                                     run.rho_n[i] * run.alpha_prime_corrected[k]

    run.X = run.R * np.cos(run.phi_tor)
    run.Y = run.R * np.sin(run.phi_tor)
    run.Z = run.Z

    mlab.points3d(run.X, run.Y, run.Z, scale_factor=0.05)
    mlab.show()
