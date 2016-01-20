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

def n_structures(run, perc_thresh, create_film=False):
    """
    Calculates the number of structures as a function of time for a given
    percentile cut-off. Writes results and plots to an appropriate directory.

    Parameters
    ----------

    run : object
        Run object calculated by the Run class.
    perc_thresh : int
        Percentile threshold at which to cut off fluctuations.
    create_film : bool
        Determines whether a film of the labelled structures is produced.
    """

    perc_thresh = int(perc_thresh)

    run.read_ntot()

    os.system('mkdir -p ' + run.run_dir + 'analysis/structures_' +
              str(perc_thresh))

    # Determine the appropriate threshold to apply at each time later
    thresh_t = np.zeros(run.nt, dtype=float)
    for it in range(run.nt):
        thresh_t[it] = np.percentile(run.ntot_i[it,:,:], perc_thresh,
                                      interpolation='nearest')

    thresh = np.median(thresh_t)

    nblobs = np.empty(run.nt, dtype=int)
    nlabel = np.empty(run.nt, dtype=int)
    label_image = np.empty([run.nt, run.nx, run.ny], dtype=int)
    for it in range(run.nt):
        tmp = run.ntot_i[it,:,:]

        # Apply Gaussian filter
        tmp = filters.gaussian(tmp, sigma=1)

        tmp_max = np.max(tmp)
        tmp_thresh = thresh/tmp_max
        tmp /= tmp_max

        tmp[tmp <= tmp_thresh] = 0
        tmp[tmp > tmp_thresh] = 1

        # Label the resulting structures
        label_image[it,:,:], nlabel[it] = label(tmp, return_num=True,
                                                background=0)

        # Now remove any structures which are too small
        hist = np.histogram(np.ravel(label_image), bins=range(1,nlabel[it]+1))[0]
        smallest_struc = np.mean(hist)*0.1
        hist = hist[hist >  smallest_struc]

        nblobs[it] = len(hist)

    plt.clf()
    plt.plot(nblobs)
    plt.xlabel('time index')
    plt.ylabel('Number of blobs')
    plt.ylim(0)
    plt.savefig(run.run_dir + 'analysis/structures_' + str(perc_thresh) +
                '/nblobs.pdf')
    np.savetxt(run.run_dir + 'analysis/structures_' + str(perc_thresh) +
               '/nblobs.csv', np.transpose((range(run.nt), nblobs)),
               delimiter=',', fmt='%d', header='t_index,nblobs')

    if create_film:
        # Create film labelled image (could do with sorting out bbox)
        titles = []
        for it in range(run.nt):
            titles.append('No. of structures = {}'.format(nlabel[it]))
        plot_options = {'cmap':'gist_rainbow', 
                        'levels':range(-1,np.max(label_image))}
        options = {'file_name':'structures',
                   'film_dir':run.run_dir + 'analysis/structures_' + 
                              str(perc_thresh) ,
                   'frame_dir':run.run_dir + 'analysis/structures_' + 
                               str(perc_thresh) + '/film_frames',
                   'aspect':'equal',
                   'xlabel':r'$x (m)$',
                   'ylabel':r'$y (m)$',
                   'cbar_ticks':5,
                   'cbar_label':r'$Label$',
                   'bbox_inches':'tight',
                   'fps':10,
                   'title':titles}

        pf.make_film_2d(run.x, run.y, label_image, plot_options=plot_options, 
                        options=options)

run = Run(sys.argv[1])

n_structures(run, 75, create_film=False)
n_structures(run, 95, create_film=False)

