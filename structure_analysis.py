# Standard
import os
import sys

# Third Party
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pyfilm as pf
from skimage.measure import label
from skimage import filters
plt.rcParams.update({'figure.autolayout': True})
mpl.rcParams['axes.unicode_minus'] = False

#local
from run import Run
import plot_style
plot_style.white()
pal = sns.color_palette('deep')

def structure_analysis(run, perc_thresh, create_film=False):
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

    run.read_ntot()

    make_results_dir(run, perc_thresh)
    labelled_image, nlabels = label_structures(run, perc_thresh)
    no_structures = count_structures(run, labelled_image, nlabels)
    plot_no_structures(run, no_structures, perc_thresh)
    save_results(run, no_structures, perc_thresh)

    if create_film:
        make_film(run, no_structures, labelled_image, perc_thresh)

def make_results_dir(run, perc_thresh):
    os.system('mkdir -p ' + run.run_dir + 'analysis/structures_' +
              str(perc_thresh))

def label_structures(run, perc_thresh):
    nlabels = np.empty(run.nt, dtype=int)
    labelled_image = np.empty([run.nt, run.nx, run.ny], dtype=int)
    for it in range(run.nt):
        tmp = run.ntot_i[it,:,:].copy()

        # Apply Gaussian filter
        tmp = filters.gaussian(tmp, sigma=1)

        thresh = np.percentile(tmp, perc_thresh,
                               interpolation='nearest')
        tmp_max = np.max(tmp)
        tmp_thresh = thresh/tmp_max
        tmp /= tmp_max

        tmp[tmp <= tmp_thresh] = 0
        tmp[tmp > tmp_thresh] = 1

        # Label the resulting structures
        labelled_image[it,:,:], nlabels[it] = label(tmp, return_num=True,
                                                   background=0)

    return(labelled_image, nlabels)

def count_structures(run, labelled_image, nlabels):
    """
    Remove any structures which are too small and count structures.
    """
    nblobs = np.empty(run.nt, dtype=int)
    for it in range(run.nt):
        hist = np.histogram(np.ravel(labelled_image[it]),
                            bins=range(1,nlabels[it]+1))[0]
        smallest_struc = np.mean(hist)*0.1
        hist = hist[hist >  smallest_struc]

        nblobs[it] = len(hist)

    return(nblobs)

def plot_no_structures(run, no_structures, perc_thresh):
    """
    Plot number of structures as a function of time.
    """
    plt.clf()
    plt.plot(no_structures)
    plt.xlabel('Time index')
    plt.ylabel('Number of structures')
    plt.ylim(0)
    plt.savefig(run.run_dir + 'analysis/structures_' + str(perc_thresh) +
                '/nblobs.pdf')

def save_results(run, no_structures, perc_thresh):
    """
    Save the number of structures as a function of time in a file.
    """
    np.savetxt(run.run_dir + 'analysis/structures_' + str(perc_thresh) +
               '/nblobs.csv', np.transpose((range(run.nt), no_structures)),
               delimiter=',', fmt='%d', header='t_index,nblobs')

def make_film(run, no_structures, labelled_image, perc_thresh):
    titles = []
    for it in range(run.nt):
        titles.append('No. of structures = {}'.format(no_structures[it]))
    plot_options = {'cmap':'gist_rainbow',
                    'levels':np.arange(-1,np.max(labelled_image))
                    }
    options = {'file_name':'structures',
               'film_dir':run.run_dir + 'analysis/structures_' +
                          str(perc_thresh) ,
               'frame_dir':run.run_dir + 'analysis/structures_' +
                           str(perc_thresh) + '/film_frames',
               'nprocs':None,
               'aspect':'equal',
               'xlabel':r'$x$ (m)',
               'ylabel':r'$y$ (m)',
               'cbar_ticks':np.arange(-1,np.max(labelled_image),2),
               'cbar_label':r'Label',
               'fps':10,
               'bbox_inches':'tight',
               'title':titles
               }

    pf.make_film_2d(run.r, run.z, labelled_image,
                    plot_options=plot_options, options=options)

if __name__ == '__main__':
    run = Run(sys.argv[1])

    structure_analysis(run, 75, create_film=False)
    structure_analysis(run, 95, create_film=False)

