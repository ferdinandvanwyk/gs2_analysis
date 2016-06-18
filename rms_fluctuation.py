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

run = Run(sys.argv[1])

run.read_ntot()

os.system('mkdir -p ' + run.run_dir + 'analysis/amplitude')

res = {}

rms_i = np.sqrt(np.mean(run.ntot_i**2, axis=0))
res['ntot_i_rms'] = np.mean(rms_i)
res['ntot_i_std'] = np.std(rms_i)

rms_e = np.sqrt(np.mean(run.ntot_e**2, axis=0))
res['ntot_e_rms'] = np.mean(rms_e)
res['ntot_e_std'] = np.std(rms_e)

json.dump(res, open(run.run_dir + 'analysis/amplitude/rms.json', 'w'),
          indent=2)

