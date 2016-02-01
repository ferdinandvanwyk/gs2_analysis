import os
import sys
import gc

# Third Party
import numpy as np
from netCDF4 import Dataset
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import seaborn as sns
import pyfilm as pf
plt.rcParams.update({'figure.autolayout': True})
mpl.rcParams['axes.unicode_minus'] = False

# Local
from run import Run
import plot_style
import field_helper as field
plot_style.white()


def normalize(field):
    """
    Normalizes the field which is assumed to be of the form f(t, x, y).

    Parameters
    ----------
    field : array-like
        3D array of the form f(t, x, y)
    """

    for it in range(field.shape[0]):
        field[it,:,:] /= np.max(np.abs(field[it,:,:]))


def phi_film(run):
    """
    Create film of electrostatic potential.

    Parameters
    ----------
    run : object
        Instance of the Run class describing a given simulation
    """
    run.read_phi()

    print('Normalize the field? y or n')
    norm = None
    while norm != 'y' and norm != 'n':
        norm = str(input())
        if norm == 'y':
            normalize(run.phi)
        elif norm == 'n':
            pass
        else:
            print('Input not recognized, try again: y or n')

    contours = field.calculate_contours(run.phi)

    plot_options = {'levels':contours, 'cmap':'seismic'}
    options = {'file_name':'phi',
               'film_dir':run.run_dir + 'analysis/moments',
               'frame_dir':run.run_dir + 'analysis/moments/film_frames',
               'aspect':'equal',
               'xlabel':r'$x (m)$',
               'ylabel':r'$y (m)$',
               'cbar_ticks':5,
               'cbar_label':r'$\varphi$',
               'bbox_inches':'tight',
               'fps':30}

    options['title'] = []
    for it in range(run.nt):
        options['title'].append(r'Time = {0:04d} $\mu s$'.format(
                                int(np.round((run.t[it]-run.t[0])*1e6))))

    pf.make_film_2d(run.x, run.y, run.phi, plot_options=plot_options,
                    options=options)

    run.phi = None
    gc.collect()


def ntot_film(run):
    """
    Create film of density fluctuations.
    """

    run.read_ntot()

    print('Normalize the field? y or n')
    norm = None
    while norm != 'y' and norm != 'n':
        norm = str(input())
        if norm == 'y':
            normalize(run.ntot_i)
            normalize(run.ntot_e)
        elif norm == 'n':
            pass
        else:
            print('Input not recognized, try again: y or n')

    # Ion density film
    contours = field.calculate_contours(run.ntot_i)

    plot_options = {'levels':contours, 'cmap':'seismic'}
    options = {'file_name':'ntot_i',
               'film_dir':run.run_dir + 'analysis/moments',
               'frame_dir':run.run_dir + 'analysis/moments/film_frames',
               'aspect':'equal',
               'xlabel':r'$x (m)$',
               'ylabel':r'$y (m)$',
               'cbar_ticks':5,
               'cbar_label':r'$\delta n_i / n_r$',
               'bbox_inches':'tight',
               'fps':30}

    options['title'] = []
    for it in range(run.nt):
        options['title'].append(r'Time = {0:04d} $\mu s$'.format(
                                int(np.round((run.t[it]-run.t[0])*1e6))))

    pf.make_film_2d(run.x, run.y, run.ntot_i, plot_options=plot_options,
                    options=options)

    run.ntot_i = None
    gc.collect()

    # Electron density film
    contours = field.calculate_contours(run.ntot_e)

    plot_options = {'levels':contours, 'cmap':'seismic'}
    options = {'file_name':'ntot_e',
               'film_dir':run.run_dir + 'analysis/moments',
               'frame_dir':run.run_dir + 'analysis/moments/film_frames',
               'aspect':'equal',
               'xlabel':r'$x (m)$',
               'ylabel':r'$y (m)$',
               'cbar_ticks':5,
               'cbar_label':r'$\delta n_e / n_r$',
               'bbox_inches':'tight',
               'fps':30}

    options['title'] = []
    for it in range(run.nt):
        options['title'].append(r'Time = {0:04d} $\mu s$'.format(
                                int(np.round((run.t[it]-run.t[0])*1e6))))

    pf.make_film_2d(run.x, run.y, run.ntot_e, plot_options=plot_options,
                    options=options)

    run.ntot_e = None
    gc.collect()


def upar_film(run):
    """
    Make film of parallel velocity.
    """

    run.read_upar()

    print('Normalize the field? y or n')
    norm = None
    while norm != 'y' and norm != 'n':
        norm = str(input())
        if norm == 'y':
            normalize(run.upar_i)
            normalize(run.upar_e)
        elif norm == 'n':
            pass
        else:
            print('Input not recognized, try again: y or n')

    # Ion upar film
    contours = field.calculate_contours(run.upar_i)

    plot_options = {'levels':contours, 'cmap':'seismic'}
    options = {'file_name':'upar_i',
               'film_dir':run.run_dir + 'analysis/moments',
               'frame_dir':run.run_dir + 'analysis/moments/film_frames',
               'aspect':'equal',
               'xlabel':r'$x (m)$',
               'ylabel':r'$y (m)$',
               'cbar_ticks':5,
               'cbar_label':r'$u_{i, \parallel}$',
               'bbox_inches':'tight',
               'fps':30}

    options['title'] = []
    for it in range(run.nt):
        options['title'].append(r'Time = {0:04d} $\mu s$'.format(
                                int(np.round((run.t[it]-run.t[0])*1e6))))

    pf.make_film_2d(run.x, run.y, run.upar_i, plot_options=plot_options,
                    options=options)

    run.upar_i = None
    gc.collect()

    # Electron upar film
    contours = field.calculate_contours(run.upar_e)

    plot_options = {'levels':contours, 'cmap':'seismic'}
    options = {'file_name':'upar_e',
               'film_dir':run.run_dir + 'analysis/moments',
               'frame_dir':run.run_dir + 'analysis/moments/film_frames',
               'aspect':'equal',
               'xlabel':r'$x (m)$',
               'ylabel':r'$y (m)$',
               'cbar_ticks':5,
               'cbar_label':r'$u_{e, \parallel}$',
               'bbox_inches':'tight',
               'fps':30}

    options['title'] = []
    for it in range(run.nt):
        options['title'].append(r'Time = {0:04d} $\mu s$'.format(
                                int(np.round((run.t[it]-run.t[0])*1e6))))

    pf.make_film_2d(run.x, run.y, run.upar_e, plot_options=plot_options,
                    options=options)

    run.upar_e = None
    gc.collect()


def v_exb_film(run):
    """
    Make film of parallel velocity.
    """

    run.calculate_v_exb()

    print('Normalize the field? y or n')
    norm = None
    while norm != 'y' and norm != 'n':
        norm = str(input())
        if norm == 'y':
            normalize(run.v_exb)
        elif norm == 'n':
            pass
        else:
            print('Input not recognized, try again: y or n')

    # Ion upar film
    contours = field.calculate_contours(run.v_exb)

    plot_options = {'levels':contours, 'cmap':'seismic'}
    options = {'file_name':'v_exb',
               'film_dir':run.run_dir + 'analysis/moments',
               'frame_dir':run.run_dir + 'analysis/moments/film_frames',
               'aspect':'equal',
               'xlabel':r'$x (m)$',
               'ylabel':r'$y (m)$',
               'cbar_ticks':5,
               'cbar_label':r'$v_{E \times B}$ (m/s)',
               'bbox_inches':'tight',
               'fps':30}

    options['title'] = []
    for it in range(run.nt):
        options['title'].append(r'Time = {0:04d} $\mu s$'.format(
                                int(np.round((run.t[it]-run.t[0])*1e6))))

    pf.make_film_2d(run.x, run.y, run.v_exb, plot_options=plot_options,
                    options=options)

    run.v_exb = None
    gc.collect()


def tpar_film(run):
    """
    Make film of parallel temperature.
    """

    run.read_tpar()

    print('Normalize the field? y or n')
    norm = None
    while norm != 'y' and norm != 'n':
        norm = str(input())
        if norm == 'y':
            normalize(run.tpar_i)
            normalize(run.tpar_e)
        elif norm == 'n':
            pass
        else:
            print('Input not recognized, try again: y or n')

    contours = field.calculate_contours(run.tpar_i)

    plot_options = {'levels':contours, 'cmap':'seismic'}
    options = {'file_name':'tpar_i',
               'film_dir':run.run_dir + 'analysis/moments',
               'frame_dir':run.run_dir + 'analysis/moments/film_frames',
               'aspect':'equal',
               'xlabel':r'$x (m)$',
               'ylabel':r'$y (m)$',
               'cbar_ticks':5,
               'cbar_label':r'$\delta T_{i, \parallel} / T_r$',
               'bbox_inches':'tight',
               'fps':30}

    options['title'] = []
    for it in range(run.nt):
        options['title'].append(r'Time = {0:04d} $\mu s$'.format(
                                int(np.round((run.t[it]-run.t[0])*1e6))))

    pf.make_film_2d(run.x, run.y, run.tpar_i, plot_options=plot_options,
                    options=options)

    run.tpar_i = None
    gc.collect()

    contours = field.calculate_contours(run.tpar_e)

    plot_options = {'levels':contours, 'cmap':'seismic'}
    options = {'file_name':'tpar_e',
               'film_dir':run.run_dir + 'analysis/moments',
               'frame_dir':run.run_dir + 'analysis/moments/film_frames',
               'aspect':'equal',
               'xlabel':r'$x (m)$',
               'ylabel':r'$y (m)$',
               'cbar_ticks':5,
               'cbar_label':r'$\delta T_{e, \parallel} / T_r$',
               'bbox_inches':'tight',
               'fps':30}

    options['title'] = []
    for it in range(run.nt):
        options['title'].append(r'Time = {0:04d} $\mu s$'.format(
                                int(np.round((run.t[it]-run.t[0])*1e6))))

    pf.make_film_2d(run.x, run.y, run.tpar_e, plot_options=plot_options,
                    options=options)

    run.tpar_e = None
    gc.collect()


def tperp_film(run):
    """
    Make film of perpendicular temperature.
    """

    run.read_tperp()

    print('Normalize the field? y or n')
    norm = None
    while norm != 'y' and norm != 'n':
        norm = str(input())
        if norm == 'y':
            normalize(run.tperp_i)
            normalize(run.tperp_e)
        elif norm == 'n':
            pass
        else:
            print('Input not recognized, try again: y or n')

    contours = field.calculate_contours(run.tperp_i)

    plot_options = {'levels':contours, 'cmap':'seismic'}
    options = {'file_name':'tperp_i',
               'film_dir':run.run_dir + 'analysis/moments',
               'frame_dir':run.run_dir + 'analysis/moments/film_frames',
               'aspect':'equal',
               'xlabel':r'$x (m)$',
               'ylabel':r'$y (m)$',
               'cbar_ticks':5,
               'cbar_label':r'$\delta T_{i, \perp} / T_r$',
               'bbox_inches':'tight',
               'fps':30}

    options['title'] = []
    for it in range(run.nt):
        options['title'].append(r'Time = {0:04d} $\mu s$'.format(
                                int(np.round((run.t[it]-run.t[0])*1e6))))

    pf.make_film_2d(run.x, run.y, run.tperp_i, plot_options=plot_options,
                    options=options)

    run.tperp_i = None
    gc.collect()

    contours = field.calculate_contours(run.tperp_e)

    plot_options = {'levels':contours, 'cmap':'seismic'}
    options = {'file_name':'tperp_e',
               'film_dir':run.run_dir + 'analysis/moments',
               'frame_dir':run.run_dir + 'analysis/moments/film_frames',
               'aspect':'equal',
               'xlabel':r'$x (m)$',
               'ylabel':r'$y (m)$',
               'cbar_ticks':5,
               'cbar_label':r'$\delta T_{e, \perp} / T_r$',
               'bbox_inches':'tight',
               'fps':30}

    options['title'] = []
    for it in range(run.nt):
        options['title'].append(r'Time = {0:04d} $\mu s$'.format(
                                int(np.round((run.t[it]-run.t[0])*1e6))))

    pf.make_film_2d(run.x, run.y, run.tperp_e, plot_options=plot_options,
                    options=options)

    run.tperp_e = None
    gc.collect()


def heat_flux_film(run):
    """
    Make film of local heat flux as a function of x and y.
    """

    run.calculate_q()

    print('Normalize the field? y or n')
    norm = None
    while norm != 'y' and norm != 'n':
        norm = str(input())
        if norm == 'y':
            normalize(run.q)
        elif norm == 'n':
            pass
        else:
            print('Input not recognized, try again: y or n')

    contours = field.calculate_contours(run.q)

    plot_options = {'levels':contours, 'cmap':'seismic'}
    options = {'file_name':'q_i',
               'film_dir':run.run_dir + 'analysis/moments',
               'frame_dir':run.run_dir + 'analysis/moments/film_frames',
               'aspect':'equal',
               'xlabel':r'$x (m)$',
               'ylabel':r'$y (m)$',
               'cbar_ticks':5,
               'cbar_label':r'$Q_{i}(x, y) / Q_{gB}$',
               'bbox_inches':'tight',
               'fps':30}

    options['title'] = []
    for it in range(run.nt):
        options['title'].append(r'Time = {0:04d} $\mu s$'.format(
                                int(np.round((run.t[it]-run.t[0])*1e6))))

    pf.make_film_2d(run.x, run.y, run.q, plot_options=plot_options,
                    options=options)

    run.q = None
    gc.collect()


def radial_heat_flux_film(run):
    """
    Make film of the radial heat flux.
    """

    run.calculate_q()
    run.q_rad = np.mean(run.q, axis=2)

    plot_options = {}
    options = {'file_name':'q_i_rad',
               'film_dir':run.run_dir + 'analysis/moments',
               'frame_dir':run.run_dir + 'analysis/moments/film_frames',
               'xlabel':r'$x (m)$',
               'ylabel':r'$\left<Q_{i}(x)\right>_y / Q_{gB}$',
               'ylim':0,
               'bbox_inches':'tight',
               'fps':30}

    options['title'] = []
    for it in range(run.nt):
        options['title'].append(r'Time = {0:04d} $\mu s$'.format(
                                int(np.round((run.t[it]-run.t[0])*1e6))))

    pf.make_film_1d(run.x, run.q_rad, plot_options=plot_options,
                    options=options)

    run.q = None
    run.q_rad = None
    gc.collect()


def v_zf_film(run):
    """
    Make film of zonal flow velocity as a function of x and t.
    """

    run.calculate_v_zf()

    plot_options = {}
    options = {'file_name':'v_zf',
               'film_dir':run.run_dir + 'analysis/moments',
               'frame_dir':run.run_dir + 'analysis/moments/film_frames',
               'xlabel':r'$x (m)$',
               'ylabel':r'$v_{ZF} / v_{th,i}$',
               'bbox_inches':'tight',
               'fps':30}

    options['title'] = []
    for it in range(run.nt):
        options['title'].append(r'Time = {0:04d} $\mu s$'.format(
                                int(np.round((run.t[it]-run.t[0])*1e6))))

    pf.make_film_1d(run.x, run.v_zf, plot_options=plot_options,
                    options=options)

    run.v_zf = None
    gc.collect()


def zf_shear_film(run):
    """
    Make film of zonal flow velocity as a function of x and t.
    """

    run.calculate_zf_shear()

    plot_options = {}
    options = {'file_name':'zf_shear',
               'film_dir':run.run_dir + 'analysis/moments',
               'frame_dir':run.run_dir + 'analysis/moments/film_frames',
               'xlabel':r'$x (m)$',
               'ylabel':r"$v'_{ZF} / v_{th,i}$",
               'bbox_inches':'tight',
               'fps':30}

    options['title'] = []
    for it in range(run.nt):
        options['title'].append(r'Time = {0:04d} $\mu s$'.format(
                                int(np.round((run.t[it]-run.t[0])*1e6))))

    pf.make_film_1d(run.x, run.zf_shear, plot_options=plot_options,
                    options=options)

    run.v_zf = None
    gc.collect()

if __name__ == '__main__':

    run = Run(sys.argv[1])
    run.lab_frame = False

    print('Which field do you want to make a film of?')
    print('1 : phi')
    print('2 : ntot')
    print('3 : upar')
    print('4 : v_exb')
    print('5 : zonal flow velocity')
    print('6 : zf velocity shear')
    print('7 : tpar')
    print('8 : tperp')
    print('9 : heat flux')
    print('10 : radial heat flux')
    print('all : all moments')
    case_id = str(input())

    if case_id == '1':
        phi_film(run)
    elif case_id == '2':
        ntot_film(run)
    elif case_id == '3':
        upar_film(run)
    elif case_id == '4':
        v_exb_film(run)
    elif case_id == '5':
        v_zf_film(run)
    elif case_id == '6':
        zf_shear_film(run)
    elif case_id == '7':
        tpar_film(run)
    elif case_id == '8':
        tperp_film(run)
    elif case_id == '9':
        heat_flux_film(run)
    elif case_id == '10':
        radial_heat_flux_film(run)
    elif case_id == 'all':
        phi_film(run)
        ntot_film(run)
        upar_film(run)
        v_exb_film(run)
        v_zf_film(run)
        zf_shear_film(run)
        tpar_film(run)
        tperp_film(run)
        heat_flux_film(run)
        radial_heat_flux_film(run)
