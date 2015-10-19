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

q_clip = np.empty([run.nt, run.nx, run.ny], dtype=float)
for it in range(run.nt):
    mask = run.q[it,:,:] < np.median(run.q[it,:,:])*500
    q_clip[it,:,:] = run.q[it,:,:]
    q_clip[it,mask] = 0

plt.plot(np.mean(q_clip[0,:,:], axis=1))
plt.show()
