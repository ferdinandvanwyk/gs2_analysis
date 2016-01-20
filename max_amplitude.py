# This script calculates the maximum amplitude of the turbulent density
# fluctuations and outputs it as a function of time step

# Standard
import os
import sys

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

max_amp_i = np.empty(run.nt, dtype=float)
max_amp_e = np.empty(run.nt, dtype=float)
for it in range(run.nt):
    max_amp_i[it] = np.max(run.ntot_i[it, :, :])
    max_amp_e[it] = np.max(run.ntot_e[it, :, :])

np.savetxt(run.run_dir + 'analysis/amplitude/max_amp.csv', 
           np.transpose((range(run.nt), max_amp_i, max_amp_e)), 
           delimiter=',', fmt=['%d', '%.5f', '%.5f'], 
           header='t_index,max_amp_i,max_amp_e')

plt.clf()
plt.plot(max_amp_i)
plt.plot(np.arange(run.nt), [np.median(max_amp_i)]*run.nt, c=pal[2])
plt.xlabel('Time index')
plt.ylabel('Max ion amplitude')
plt.ylim(0)
plt.savefig(run.run_dir + 'analysis/amplitude//max_amp_i.pdf')

plt.clf()
plt.plot(max_amp_e)
plt.plot(np.arange(run.nt), [np.median(max_amp_e)]*run.nt, c=pal[2])
plt.xlabel('Time index')
plt.ylabel('Max electron amplitude')
plt.ylim(0)
plt.savefig(run.run_dir + 'analysis/amplitude//max_amp_e.pdf')
