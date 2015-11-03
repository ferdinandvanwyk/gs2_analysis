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
from skimage.segmentation import clear_border
from skimage.measure import label
from skimage import filters
from skimage.morphology import disk
plt.rcParams.update({'figure.autolayout': True})
mpl.rcParams['axes.unicode_minus']=False

#local
from run import Run
import plot_style
import field_helper as field
plot_style.white()
pal = sns.color_palette('deep')                                                 

os.system('mkdir -p analysis/structures')

run = Run(sys.argv[1])

run.calculate_q()

nblobs = np.zeros(run.nt, dtype=int)
for it in range(run.nt):
    tmp = run.q[it,:,:]

    # Apply Gaussian filter
    tmp = filters.gaussian(tmp, sigma=1)

    cut_off = np.percentile(tmp, 80, interpolation='nearest')
    tmp /= np.max(tmp)

    tmp[tmp <= cut_off] = 0.0
    tmp[tmp > cut_off] = 1.0

    # Clear areas around the borders
    tmp = clear_border(tmp) 

    # Label the resulting structures
    label_image, nlabels = label(tmp, return_num=True)

    # Now remove any structures which are too small
    hist = np.histogram(np.ravel(label_image), bins=range(1,nlabels+1))[0]
    smallest_struc = np.mean(hist)*0.1 
    hist = hist[hist >  smallest_struc]

    nblobs[it] = len(hist)

plt.clf()
plt.plot(nblobs)
plt.xlabel('time index')
plt.ylabel('Number of blobs')
plt.ylim(0)
plt.savefig('analysis/structures/nblobs.pdf')
print('Median no. of blobs = ', int(np.median(nblobs)))
np.savetxt('analysis/structures/nblobs.csv', 
            np.transpose((range(run.nt), nblobs)), delimiter=',', fmt='%d',
            header='t_index,nblobs')
