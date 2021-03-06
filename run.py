# Standard
import os
import sys

# Third Party
import numpy as np
from netCDF4 import Dataset
import f90nml as nml

#local
import field_helper as field

class Run(object):
    """
    Run object which stores basic simulation information.
    """

    def __init__(self, cdf_file):
        """
        Initialize with NetCDF file information.
        """
        self.cdf_file = cdf_file

        try:
            idx = self.cdf_file.rindex('/')
            self.run_dir = self.cdf_file[:idx] + '/'
        except ValueError:
            self.run_dir = ''

        ncfile = Dataset(cdf_file, 'r')

        self.drho_dpsi = ncfile.variables['drhodpsi'][0]
        self.kx = np.array(ncfile.variables['kx'][:])/self.drho_dpsi
        self.kx_natural_order = np.roll(self.kx, int(len(self.kx)/2))
        self.ky = np.array(ncfile.variables['ky'][:])/self.drho_dpsi
        self.theta = np.array(ncfile.variables['theta'][:])
        self.t_gs2 = np.array(ncfile.variables['t'][:])
        self.gradpar = np.array(ncfile.variables['gradpar'][:])
        self.grho = np.array(ncfile.variables['grho'][:])
        self.gds2 = np.array(ncfile.variables['gds2'][:])
        self.galpha = np.sqrt(self.drho_dpsi**2 * self.gds2)
        self.bmag = np.array(ncfile.variables['bmag'][:])
        self.rprime = np.array(ncfile.variables['Rprime'][:])
        self.dtheta = np.append(np.diff(self.theta), 0)

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
        self.temp_1 = float(gs2_in['species_parameters_1']['temp'])
        self.tprim_1 = float(gs2_in['species_parameters_1']['tprim'])
        self.fprim_1 = float(gs2_in['species_parameters_1']['fprim'])
        self.mass_1 = float(gs2_in['species_parameters_1']['mass'])
        if gs2_in['species_knobs']['nspec'] == 2:
            self.temp_2 = float(gs2_in['species_parameters_2']['temp'])
            self.tprim_2 = float(gs2_in['species_parameters_2']['tprim'])
            self.fprim_2 = float(gs2_in['species_parameters_2']['fprim'])
            self.mass_2 = float(gs2_in['species_parameters_2']['mass'])


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
        # Toroidal mode number
        self.n0 = int(np.around(self.ky[1]*(self.amin/self.rhoref)))

        self.nt = len(self.t_gs2)
        self.nkx = len(self.kx)
        self.nky = len(self.ky)
        self.nth = len(self.theta)
        self.nx = self.nkx
        self.ny = 2*(self.nky - 1)
        self.t = self.t_gs2*self.amin/self.vth
        self.x = np.pi * np.linspace(-1/self.kx[1], 1/self.kx[1], self.nx, endpoint=False) / self.drho_dpsi
        self.y = np.pi * np.linspace(-1/self.ky[1], 1/self.ky[1], self.ny, endpoint=False) / self.drho_dpsi
        delta_rho = (self.rho_tor/self.qinp) * (self.jtwist/(self.n0*self.shat))
        self.r_box_size = self.rprime[int(self.nth/2)]*delta_rho*self.amin
        self.r = np.linspace(-self.r_box_size/2, self.r_box_size/2, self.nx,
                             endpoint=False)
        self.z_tor_box_size = self.rmaj * 2 * np.pi / self.n0
        self.z_pol_box_size = self.z_tor_box_size * np.tan(self.pitch_angle)
        self.z = np.linspace(-self.z_pol_box_size/2, self.z_pol_box_size/2,
                             self.ny, endpoint=False)

    def read_phi(self, lab_frame=False):
        """
        Read the electrostatic potenential from the NetCDF file.
        """

        self.phi = field.get_field(self.cdf_file, 'phi_igomega_by_mode', None)

        if lab_frame:
            for ix in range(self.nkx):
                for iy in range(self.nky):
                    self.phi[:,ix,iy] = self.phi[:,ix,iy]* \
                                        np.exp(1j * self.n0 * iy * \
                                               self.omega * self.t)

        self.phi = field.field_to_real_space(self.phi)*self.rho_star

    def read_ntot(self, lab_frame=False):
        """
        Read the 2D density fluctuations from the NetCDF file.
        """

        self.ntot_i = field.get_field(self.cdf_file, 'ntot_igomega_by_mode', 0)
        self.ntot_e = field.get_field(self.cdf_file, 'ntot_igomega_by_mode', 1)

        if lab_frame:
            for ix in range(self.nkx):
                for iy in range(self.nky):
                    self.ntot_i[:,ix,iy] = self.ntot_i[:,ix,iy]* \
                                           np.exp(1j * self.n0 * iy * \
                                                  self.omega * self.t)
                    self.ntot_e[:,ix,iy] = self.ntot_e[:,ix,iy]* \
                                           np.exp(1j * self.n0 * iy * \
                                                  self.omega * self.t)

        # Convert to real space
        self.ntot_i = field.field_to_real_space(self.ntot_i)*self.rho_star
        self.ntot_e = field.field_to_real_space(self.ntot_e)*self.rho_star

    def read_ntot_3d(self, it=0, isp=0):
        """
        Read the 3D density fluctuations from the NetCDF file.

        Parameters
        ----------

        it : int
            Time index
        isp : int
            Species index
        """

        self.ntot_i = field.get_field_3d(self.cdf_file, 'ntot_t', it, isp)

        self.ntot_i = field.field_to_real_space_3d(self.ntot_i)*self.rho_star

    def read_upar(self, lab_frame=False):
        """
        Read the parallel velocity.
        """

        self.upar_i = field.get_field(self.cdf_file, 'upar_igomega_by_mode', 0)
        self.upar_e = field.get_field(self.cdf_file, 'upar_igomega_by_mode', 1)

        if lab_frame:
            for ix in range(self.nkx):
                for iy in range(self.nky):
                    self.upar_i[:,ix,iy] = self.upar_i[:,ix,iy]* \
                                           np.exp(1j * self.n0 * iy * \
                                                  self.omega * self.t)
                    self.upar_e[:,ix,iy] = self.upar_e[:,ix,iy]* \
                                           np.exp(1j * self.n0 * iy * \
                                                  self.omega * self.t)

        self.upar_i = field.field_to_real_space(self.upar_i)*self.rho_star
        self.upar_e = field.field_to_real_space(self.upar_e)*self.rho_star

    def read_tpar(self, lab_frame=False):
        """
        Read the parallel temperature.
        """

        self.tpar_i = field.get_field(self.cdf_file, 'tpar_igomega_by_mode', 0)
        self.tpar_e = field.get_field(self.cdf_file, 'tpar_igomega_by_mode', 1)

        if lab_frame:
            for ix in range(self.nkx):
                for iy in range(self.nky):
                    self.tpar_i[:,ix,iy] = self.tpar_i[:,ix,iy]* \
                                           np.exp(1j * self.n0 * iy * \
                                                  self.omega * self.t)
                    self.tpar_e[:,ix,iy] = self.tpar_e[:,ix,iy]* \
                                           np.exp(1j * self.n0 * iy * \
                                                  self.omega * self.t)

        # Convert to real space
        self.tpar_i = field.field_to_real_space(self.tpar_i)*self.rho_star
        self.tpar_e = field.field_to_real_space(self.tpar_e)*self.rho_star

    def read_tperp(self, lab_frame=False):
        """
        Read the perpendicular temperature.
        """

        self.tperp_i = field.get_field(self.cdf_file, 'tperp_igomega_by_mode', 0)
        self.tperp_e = field.get_field(self.cdf_file, 'tperp_igomega_by_mode', 1)

        if lab_frame:
            for ix in range(self.nkx):
                for iy in range(self.nky):
                    self.tperp_i[:,ix,iy] = self.tperp_i[:,ix,iy]* \
                                            np.exp(1j * self.n0 * iy * \
                                                   self.omega * self.t)
                    self.tperp_e[:,ix,iy] = self.terp_e[:,ix,iy]* \
                                            np.exp(1j * self.n0 * iy * \
                                                   self.omega * self.t)

        # Convert to real space
        self.tperp_i = field.field_to_real_space(self.tperp_i)*self.rho_star
        self.tperp_e = field.field_to_real_space(self.tperp_e)*self.rho_star

    def read_q(self):
        """
        Read the flux surfaced averaged heat flux as a function of time.
        """

        ncfile = Dataset(self.cdf_file, 'r')
        self.q_i = np.array(ncfile.variables['es_heat_flux'][:,0])
        self.q_e = np.array(ncfile.variables['es_heat_flux'][:,1])

    def calculate_v_exb(self, lab_frame=False):
        """
        Calculates the radial ExB velocity in real space in units of v_th,i.
        """

        phi_k = field.get_field(self.cdf_file, 'phi_igomega_by_mode', None)

        self.v_exb = 1j*self.ky*phi_k

        if lab_frame:
            for ix in range(self.nkx):
                for iy in range(self.nky):
                    self.v_exb[:,ix,iy] = self.v_exb[:,ix,iy]* \
                                          np.exp(1j * self.n0 * iy * \
                                                 self.omega * self.t)

        self.v_exb = (0.5 / self.grho[int(self.nth/2)]) * \
                      field.field_to_real_space(self.v_exb)* \
                      self.rho_star

    def calculate_q(self):
        """
        Calculate the local heat flux Q(x, y) for the ion species.
        """

        # Need phi as a function of kx, ky so read directly from netcdf file
        self.phi = field.get_field(self.cdf_file, 'phi_igomega_by_mode', None)
        self.read_ntot()
        self.read_tperp()
        self.read_tpar()

        # Convert to real space
        v_exb = field.field_to_real_space(1j*self.ky*self.phi)
        ntot_i = self.ntot_i/self.rho_star
        tperp_i = self.tperp_i/self.rho_star
        tpar_i = self.tpar_i/self.rho_star

        self.q = ((tperp_i + tpar_i/2 + 3/2*ntot_i)*v_exb).real/2

        dnorm = self.dtheta/self.bmag/self.gradpar
        wgt = np.sum(dnorm*self.grho)

        self.q = self.q * dnorm[int(self.nth/2)] / wgt

    def calculate_v_zf(self, add_mean_flow=False):
        """
        Calculate the zonal flow velocity as a function of radius and time.

        The velocity is in units of vth.
        """

        phi_k = field.get_field(self.cdf_file, 'phi_igomega_by_mode', None)

        self.norm_fac = 0.5 * abs(self.qinp)/self.rhoc/ \
                        abs(self.galpha[int(self.nth/2)])

        self.v_zf = 0.5 * self.norm_fac * \
                    np.fft.ifft(-phi_k[:, :, 0] * self.kx[np.newaxis, :],
                                axis=1).imag * self.nx * self.rho_star

        if add_mean_flow:
            self.v_zf += self.r * self.g_exb / self.amin

    def calculate_zf_shear(self):
        """
        Calculate the shear in the  zonal flow velocity as a function of radius
        and time.
        """

        phi_k = field.get_field(self.cdf_file, 'phi_igomega_by_mode', None)

        self.norm_fac = 0.5 * abs(self.qinp)/self.rhoc/ \
                        abs(self.galpha[int(self.nth/2)])

        self.zf_shear = 0.5 * self.norm_fac * \
                        np.fft.ifft(- phi_k[:, :, 0]*self.kx[np.newaxis, :]**2,
                                    axis=1).real * \
                        self.nx * self.rho_star / self.rhoref

        self.zf_shear_max = np.sqrt(np.mean(np.max(np.abs(self.zf_shear), axis=1)**2))
        self.zf_shear_rms = np.sqrt(np.mean(self.zf_shear**2))

    def calc_phi_spectra(self):
        """Read phi2 values, calc spectra and average over time."""
        with Dataset(self.cdf_file, 'r') as ncfile:
            self.phi2_by_kx = np.array(ncfile.variables['phi2_by_kx'][:])
            self.phi2_by_ky = np.array(ncfile.variables['phi2_by_ky'][:])

        # average over time
        self.phi2_by_kx = np.mean(self.phi2_by_kx, axis=0)
        self.phi2_by_ky = np.mean(self.phi2_by_ky, axis=0)

        # change kx order
        self.phi2_by_kx = np.roll(self.phi2_by_kx, int(self.nkx/2))
