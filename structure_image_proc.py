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
    
run = Run(sys.argv[1])

run.calculate_q()

# Test case
test = np.copy(run.q[0,:,:])

cut_off = np.percentile(test, 75, interpolation='nearest')
test[test <= cut_off] = 0

# normalize
test /= np.max(test)

blobs = np.array(blob_doh(test, min_sigma = 1, max_sigma=10, threshold=0.003))
print(blobs)

fig, ax = plt.subplots(1, 1)
ax.imshow(test, interpolation='nearest')
for blob in blobs:                                                          
        y, x, r = blob                                                          
        c = plt.Circle((x, y), r, color='red', linewidth=2, fill=False)         
        ax.add_patch(c)
plt.show()

# Full loop over time
nblobs = np.empty(run.nt, dtype=int)
for it in range(run.nt):
    tmp = run.q[it,:,:]
    cut_off = np.percentile(tmp, 75, interpolation='nearest')
    tmp[tmp <= cut_off] = 0

    # normalize
    tmp /= np.max(tmp)

    blobs = np.array(blob_doh(tmp, min_sigma = 1, max_sigma=10, threshold=0.003))
    nblobs[it] = len(blobs[:,0]) 

plt.plot(nblobs)
plt.show()
print(np.mean(nblobs))
