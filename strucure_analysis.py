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

##############################
# Count number of bands in x #
##############################
run.q_x = np.mean(run.q, axis=2)

for it in range(run.nt):
    tmp = run.q_x[it,:]
    mask = tmp <= np.percentile(tmp, 75, interpolation='nearest')
    tmp[mask] = 0
    run.q_x[it,:] = tmp

x_bands = np.empty(run.nt, dtype=int)
for it in range(run.nt):
    x_bands[it] = count_bands(run.q_x[it,:])

#########################
# Count structures in y #
#########################
# Cannot just repeat analysis above since we're interested in the no. of 
# structures in each band. Therefore need to consider each band individually,
# integrate in the x direction and then determine no of structures. 
# One important compliciation involves cases when the band of structures is 
# split across the boundary of the box. Fix this by 'rolling' the function
# in x by an appropriate amount to ensure that function (already cut off at the
# 75th percentile) is 0 at edges.

# Find rolling distance for each time step
i_roll = np.zeros(run.nt, dtype=int)
for it in range(run.nt):
    if run.q_x[it,0] == 0 and run.q_x[it,-1] == 0:
        pass
    else:
        while (run.q_x[it,0] != 0 or run.q_x[it,-1] != 0):
            run.q_x[it,:] = np.roll(run.q_x[it,:],1)
            i_roll[it] += 1
        
# Now integrate over each band and count structures
run.q_y = np.empty([run.nt, max(x_bands), run.ny], dtype=float)
run.y_structures = np.empty([run.nt, max(x_bands)], dtype=int)
run.y_structures = np.empty([run.nt, max(x_bands)], dtype=int)
run.q_band = np.empty([run.nt, max(x_bands)], dtype=int)
run.q_per_structure = np.empty([run.nt, max(x_bands)], dtype=int)
for it in range(run.nt):
    edges = get_edges(run.q_x[it,:])
    for i in range(int(len(edges)/2)):
        run.q_y[it,i,:] = np.mean(run.q[it,edges[i*2]:edges[i*2+1],:], axis=0)

        tmp = run.q_y[it,i,:]
        mask = tmp <= np.percentile(tmp, 75, interpolation='nearest')
        tmp[mask] = 0
        run.q_y[it,i,:] = tmp

        # counting bands is exactly like counting structures
        run.y_structures[it,i] = count_bands(run.q_y[it,i,:])
    
        # Calculate the total Q_rad per band in x
        run.q_band[it,i] = np.mean(run.q_x[it,edges[i*2]:edges[i*2+1]])

        run.q_per_structure[it,i] = run.q_band[it,i] / run.y_structures[it,i]

print(np.mean(run.q_per_structure))
         


















