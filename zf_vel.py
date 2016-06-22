# Standard
import os
import sys

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

if __name__ == '__main__':
    run = Run(sys.argv[1])

    run.calculate_v_zf(add_mean_flow=True)

    plt.plot(run.x, np.mean(run.v_zf, axis=0))
    plt.show()

