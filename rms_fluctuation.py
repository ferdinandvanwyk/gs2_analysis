# This script calculates the maximum amplitude of the turbulent density
# fluctuations and outputs it as a function of time step

# Standard
import os
import sys
import json

# Third Party
import numpy as np
from netCDF4 import Dataset
import matplotlib as mpl
mpl.use('Agg')
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

def average_time_windows(n):
    """
    Calculate a mean over 100 time step windows and return array of means.
    """

    nt = n.shape[0]
    nx = n.shape[1]
    ny = n.shape[2]
    time_window_size = 100

    ntime_windows = int(nt/time_window_size)

    rms_t = np.empty([ntime_windows], dtype=float)
    for i in range(ntime_windows):
        rms_t[i] = np.sqrt(np.mean(n[i*100:(i+1)*100, :, :]**2))

    return rms_t

if __name__ == '__main__':
    run = Run(sys.argv[1])

    run.read_ntot()

    os.system('mkdir -p ' + run.run_dir + 'analysis/amplitude')

    res = {}

    rms_i_t = average_time_windows(run.ntot_i)
    res['ntot_i_rms'] = np.mean(rms_i_t)
    res['ntot_i_err'] = np.std(rms_i_t)

    rms_e_t = average_time_windows(run.ntot_e)
    res['ntot_e_rms'] = np.mean(rms_e_t)
    res['ntot_e_err'] = np.std(rms_e_t)

    json.dump(res, open(run.run_dir + 'analysis/amplitude/rms.json', 'w'),
              indent=2)

