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
from run import Run
import plot_style
import field_helper as field
plot_style.white()

run = Run(sys.argv[1])

run.calculate_q()

print(np.mean(run.q[-1,:,:]), run.q_nc[-1,0])
sys.exit()

q_max = np.max(run.q)
q_min = np.min(run.q)
if np.abs(q_max) > np.abs(q_min):
    contours = np.around(np.linspace(-q_max, q_max, 20), 7)
else:
    contours = np.around(np.linspace(q_min, -q_min, 20), 7)

plot_options = {'levels':contours, 'cmap':'seismic'}
options = {'file_name':'q_vs_x_vs_y',
           'film_dir':'analysis/local_q',
           'frame_dir':'analysis/local_q/film_frames',
           'aspect':'equal',
           'xlabel':r'$x (m)$',
           'ylabel':r'$y (m)$',
           'cbar_ticks':5,
           'cbar_label':r'$Q_i(x, y) / Q_{gB}$',
           'bbox_inches':'tight',
           'fps':30}

options['title'] = []
for it in range(run.nt):
    options['title'].append(r'Time = {0:04d} $\mu s$'.format(
                            int(np.round((run.t[it]-run.t[0])*1e6))))

pf.make_film_2d(run.x, run.y, run.q[:,:,:], plot_options=plot_options, options=options)
