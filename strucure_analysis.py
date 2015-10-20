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

def count_bands(array):
    """
    Count the number of bands of structures above the 75th percentile.

    Parameters
    ----------

    y : array_like
        One dimensional array for which to count bands
    """

    y = np.copy(array)

    y[y < 0] = 0
    y[y > 0] = 1

    bands = remove_duplicates(y)

    # take care of case where start and end reperesent same band
    if bands[0] == 1 and bands[-1] == 1:
        nbands = np.sum(bands[:-1])
    else:
        nbands = np.sum(bands)

    return(nbands)

def remove_duplicates(y):
    """
    Given an array of 0's and 1's, remove adjacent duplicate entries.

    y : array_like
        1D array of 0's and 1's
    """
    res = np.array([y[0]], dtype=int)
    for i in range(1,len(y)):
        if int(y[i]) == res[-1]:
            pass
        else:
            res = np.append(res, int(y[i]))

    return(res) 

def get_edges(array):
    """
    Get the indices of the edges of the bands.

    Example
    -------
    
    If function is [1,1,1,0,0,0,1,1,0,0,1] the function will return [3,5,8,9].
    In other words, it will return the indices on either side of the structure
    where the function is zero.
    """
    
    y = np.copy(array)
    x = np.arange(len(y))

    y[y < 0] = 0
    y[y > 0] = 1
    
    res = np.array([y[0]], dtype=int)
    edges = np.array([], dtype=int)
    for i in range(1,len(y)):
        if int(y[i]) == res[-1]:
            pass
        else:
            res = np.append(res, int(y[i]))
            if y[i] == 0:
                edges = np.append(edges, i)
            elif y[i] == 1:
                edges = np.append(edges, i-1)

    return(edges)
    
run = Run(sys.argv[1])

run.calculate_q()

run.q_x = np.mean(run.q, axis=2)

for it in range(run.nt):
    tmp = run.q_x[it,:]
    mask = tmp < np.percentile(tmp, 75, interpolation='nearest')
    tmp[mask] = 0
    run.q_x[it,:] = tmp

nbands = count_bands(run.q_x[0,:])
edges = get_edges(run.q_x[0,:])

