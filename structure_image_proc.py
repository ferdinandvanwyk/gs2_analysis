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
from skimage.feature import blob_dog, blob_log, blob_doh, peak_local_max
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

# Test case
test = np.copy(run.q[0,:,:])

cut_off = np.percentile(test, 75, interpolation='nearest')
test[test <= cut_off] = 0

# normalize
test /= np.max(test)

blobs = np.array(blob_doh(test, min_sigma = 1, max_sigma=10, threshold=0.005))

fig, ax = plt.subplots(1, 1)
plt.contourf(np.transpose(test), 40, interpolation='nearest')
plt.colorbar()
plt.xlabel('x index')
plt.ylabel('y index')
for blob in blobs:                                                          
        y, x, r = blob                                                          
        c = plt.Circle((y, x), r, color=pal[2], linewidth=2, fill=False)
        ax.add_patch(c)
plt.savefig('analysis/structures/typical_frame.pdf')

# Full loop over time
nblobs = np.empty(run.nt, dtype=int)
for it in range(run.nt):
    tmp = run.q[it,:,:]
    cut_off = np.percentile(tmp, 75, interpolation='nearest')
    tmp[tmp <= cut_off] = 0

    # normalize
    tmp /= np.max(tmp)

    blobs = np.array(blob_doh(tmp, min_sigma = 1, max_sigma=10, threshold=0.005))
    nblobs[it] = len(blobs[:,0]) 

plt.clf()
plt.plot(nblobs)
plt.xlabel('time index')
plt.ylabel('Number of blobs')
plt.ylim(0)
plt.savefig('analysis/structures/nblobs.pdf')
print('Median no. of blobs = ', int(np.median(nblobs)))
