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

def zf_shear_max(run):
    """
    Calculate max (in kx) rms (in time) shear and write to csv file.
    """
    run.calculate_zf_shear_rms()

    os.system('mkdir -p ' + run.run_dir + 'analysis/zonal_flows')

    res = {}
    res['zf_shear_rms'] = run.zf_shear_rms

    json.dump(res, open(run.run_dir + 'analysis/zonal_flows/results.json', 'w'),
              indent=2)

def v_zf(run):
    run.calculate_v_zf(add_mean_flow=False)

    plt.plot(run.r, np.mean(run.v_zf, axis=0))
    plt.show()

if __name__ == '__main__':
    run = Run(sys.argv[1])

    zf_shear_max(run)
    # v_zf(run)

