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
mpl.rcParams['axes.unicode_minus']=False

#local
from run import Run
import plot_style
import field_helper as field
plot_style.white()
pal = sns.color_palette('deep')

def write_zf_shear(run):
    """
    Calculate max (in kx) rms (in time) shear and write to csv file.
    """
    run.calculate_zf_shear()

    os.system('mkdir -p ' + run.run_dir + 'analysis/zonal_flows')

    res = {}
    res['zf_shear_max'] = run.zf_shear_max
    res['zf_shear_rms'] = run.zf_shear_rms

    json.dump(res, open(run.run_dir + 'analysis/zonal_flows/results.json', 'w'),
              indent=2)

def check_zf_calcs(run):
    """
    Manually calculate gradients of phi to check fourier space calculation of
    zonal flows.
    """
    run.calculate_v_zf()
    run.calculate_zf_shear()

    phi_k = field.get_field(run.cdf_file, 'phi_igomega_by_mode', None)

    # Show zonal flow velocity is equivalent
    dx = np.abs(run.x[1] - run.x[0])

    phi_zf = 0.5 * run.kxfac * \
             np.fft.ifft(phi_k[0, :, 0]).real * run.nx * run.rho_star
    v_zf = np.gradient(phi_zf, dx)

    plt.plot(run.x, run.v_zf[0,:])
    plt.plot(run.x, v_zf)
    plt.show()

    # Show zonal shears are equivalent
    zf_shear = np.gradient(v_zf, dx)

    plt.plot(run.x, run.zf_shear[0,:])
    plt.plot(run.x, zf_shear/run.rhoref)
    plt.show()

if __name__ == '__main__':
    run = Run(sys.argv[1])

    write_zf_shear(run)

