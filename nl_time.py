# This script calculates the nonlinear time using the formula:
# tau_nl ~ k_x k_y rho_i v_th (T_e/T_i) (dn/n)_rms
# where k_x and k_y represent the peak values of the phi spectra.

# Standard
import os
import sys
import json

# Third Party
import numpy as np
from netCDF4 import Dataset
import matplotlib as mpl
# mpl.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import seaborn as sns
plt.rcParams.update({'figure.autolayout': True})
mpl.rcParams['axes.unicode_minus'] = False

#local
from run import Run
import plot_style
import field_helper as field
plot_style.white()
pal = sns.color_palette('deep')

def zero_zonal_component(v):
    """Set zonal components (smallest kx values) of kx spectrum to zero."""

    centre_idx = int(len(v)/2)

    v[centre_idx-1:centre_idx+2] = 0

    return v

def find_spectrum_peak(k, spectrum):
    """Find the k which corresponds to the peak of the spectrum."""
    max_idx = np.argmax(spectrum)

    return k[max_idx]

if __name__ == '__main__':

    res = {}

    run = Run(sys.argv[1])

    os.system('mkdir -p ' + run.run_dir + 'analysis/nl_time')

    run.read_ntot()
    rms_i = np.mean(np.sqrt(np.mean(run.ntot_i**2, axis=0)))

    run.calc_phi_spectra()
    kx_spectrum = zero_zonal_component(run.phi2_by_kx)*run.kx_natural_order**2
    ky_spectrum = run.phi2_by_ky*run.ky**2

    kx_peak = np.abs(find_spectrum_peak(run.kx_natural_order, kx_spectrum))
    ky_peak = np.abs(find_spectrum_peak(run.ky, ky_spectrum))

    te_over_ti = run.temp_2/run.temp_1

    res['tau_nl'] = 1/(kx_peak/run.rhoref * ky_peak/run.rhoref * run.rhoref *
                run.vth * te_over_ti * rms_i)
    res['kx_peak'] = kx_peak
    res['ky_peak'] = ky_peak

    json.dump(res, open(run.run_dir + 'analysis/nl_time/nl_time.json', 'w'),
              indent=2)


