import os
import sys
import gc

# Third Party
import numpy as np
from netCDF4 import Dataset
import matplotlib as mpl
mpl.use('Agg')
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


def phi_film(run, should_normalize):
    """
    Create film of electrostatic potential.

    Parameters
    ----------
    run : object
        Instance of the Run class describing a given simulation
    """
    run.read_phi()

    if should_normalize:
        field.normalize(run.phi)

    contours = field.calculate_contours(run.phi)

    plot_options = {'levels':contours, 'cmap':'seismic'}
    options = {'file_name':'phi',
               'film_dir':run.run_dir + 'analysis/moments',
               'frame_dir':run.run_dir + 'analysis/moments/film_frames',
               'aspect':'equal',
               'xlabel':r'$R (m)$',
               'ylabel':r'$Z (m)$',
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


def ntot_film(run, should_normalize):
    """
    Create film of density fluctuations.
    """

    run.read_ntot()

    if should_normalize:
        field.normalize(run.ntot_i)
        field.normalize(run.ntot_e)

    # Ion density film
    contours = field.calculate_contours(run.ntot_i)

    plot_options = {'levels':contours, 'cmap':'seismic'}
    options = {'file_name':'ntot_i',
               'film_dir':run.run_dir + 'analysis/moments',
               'frame_dir':run.run_dir + 'analysis/moments/film_frames',
               'aspect':'equal',
               'xlabel':r'$R (m)$',
               'ylabel':r'$Z (m)$',
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
               'xlabel':r'$R (m)$',
               'ylabel':r'$Z (m)$',
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


def upar_film(run, should_normalize):
    """
    Make film of parallel velocity.
    """

    run.read_upar()

    if should_normalize:
        field.normalize(run.upar_i)
        field.normalize(run.upar_e)

    # Ion upar film
    contours = field.calculate_contours(run.upar_i)

    plot_options = {'levels':contours, 'cmap':'seismic'}
    options = {'file_name':'upar_i',
               'film_dir':run.run_dir + 'analysis/moments',
               'frame_dir':run.run_dir + 'analysis/moments/film_frames',
               'aspect':'equal',
               'xlabel':r'$R (m)$',
               'ylabel':r'$Z (m)$',
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
               'xlabel':r'$R (m)$',
               'ylabel':r'$Z (m)$',
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


def v_exb_film(run, should_normalize):
    """
    Make film of parallel velocity.
    """

    run.calculate_v_exb()

    if should_normalize:
        field.normalize(run.v_exb)

    # Ion upar film
    contours = field.calculate_contours(run.v_exb)

    plot_options = {'levels':contours, 'cmap':'seismic'}
    options = {'file_name':'v_exb',
               'film_dir':run.run_dir + 'analysis/moments',
               'frame_dir':run.run_dir + 'analysis/moments/film_frames',
               'aspect':'equal',
               'xlabel':r'$R (m)$',
               'ylabel':r'$Z (m)$',
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


def tpar_film(run, should_normalize):
    """
    Make film of parallel temperature.
    """

    run.read_tpar()

    if should_normalize:
        field.normalize(run.tpar_i)
        field.normalize(run.tpar_e)

    contours = field.calculate_contours(run.tpar_i)

    plot_options = {'levels':contours, 'cmap':'seismic'}
    options = {'file_name':'tpar_i',
               'film_dir':run.run_dir + 'analysis/moments',
               'frame_dir':run.run_dir + 'analysis/moments/film_frames',
               'aspect':'equal',
               'xlabel':r'$R (m)$',
               'ylabel':r'$Z (m)$',
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
               'xlabel':r'$R (m)$',
               'ylabel':r'$Z (m)$',
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


def tperp_film(run, should_normalize):
    """
    Make film of perpendicular temperature.
    """

    run.read_tperp()

    if should_normalize:
        field.normalize(run.tperp_i)
        field.normalize(run.tperp_e)

    contours = field.calculate_contours(run.tperp_i)

    plot_options = {'levels':contours, 'cmap':'seismic'}
    options = {'file_name':'tperp_i',
               'film_dir':run.run_dir + 'analysis/moments',
               'frame_dir':run.run_dir + 'analysis/moments/film_frames',
               'aspect':'equal',
               'xlabel':r'$R (m)$',
               'ylabel':r'$Z (m)$',
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
               'xlabel':r'$R (m)$',
               'ylabel':r'$Z (m)$',
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


def heat_flux_film(run, should_normalize):
    """
    Make film of local heat flux as a function of x and y.
    """

    run.calculate_q()

    if should_normalize:
        field.normalize(run.q)

    contours = field.calculate_contours(run.q)

    plot_options = {'levels':contours, 'cmap':'seismic'}
    options = {'file_name':'q_i',
               'film_dir':run.run_dir + 'analysis/moments',
               'frame_dir':run.run_dir + 'analysis/moments/film_frames',
               'aspect':'equal',
               'xlabel':r'$R (m)$',
               'ylabel':r'$Z (m)$',
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


def radial_heat_flux_film(run, should_normalize):
    """
    Make film of the radial heat flux.
    """

    run.calculate_q()
    run.q_rad = np.mean(run.q, axis=2)

    plot_options = {}
    options = {'file_name':'q_i_rad',
               'film_dir':run.run_dir + 'analysis/moments',
               'frame_dir':run.run_dir + 'analysis/moments/film_frames',
               'xlabel':r'$R (m)$',
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


def v_zf_film(run, should_normalize):
    """
    Make film of zonal flow velocity as a function of x and t.
    """

    run.calculate_v_zf()

    plot_options = {}
    options = {'file_name':'v_zf',
               'film_dir':run.run_dir + 'analysis/moments',
               'frame_dir':run.run_dir + 'analysis/moments/film_frames',
               'xlabel':r'$R (m)$',
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


def zf_shear_film(run, should_normalize):
    """
    Make film of zonal flow velocity as a function of x and t.
    """

    run.calculate_zf_shear()

    plot_options = {}
    options = {'file_name':'zf_shear',
               'film_dir':run.run_dir + 'analysis/moments',
               'frame_dir':run.run_dir + 'analysis/moments/film_frames',
               'xlabel':r'$R (m)$',
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

    try:
        case_id = str(sys.argv[2])
    except IndexError:
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

    try:
        user_answer = str(sys.argv[3])
    except IndexError:
        print('Normalize field? y/n')
        user_answer = str(input())

    if user_answer == 'y' or user_answer == 'Y':
        should_normalize = True
    elif user_answer == 'n' or user_answer == 'N':
        should_normalize = False
    else:
        sys.exit('Wrong option.')

    if case_id == '1':
        phi_film(run, should_normalize)
    elif case_id == '2':
        ntot_film(run, should_normalize)
    elif case_id == '3':
        upar_film(run, should_normalize)
    elif case_id == '4':
        v_exb_film(run, should_normalize)
    elif case_id == '5':
        v_zf_film(run, should_normalize)
    elif case_id == '6':
        zf_shear_film(run, should_normalize)
    elif case_id == '7':
        tpar_film(run, should_normalize)
    elif case_id == '8':
        tperp_film(run, should_normalize)
    elif case_id == '9':
        heat_flux_film(run, should_normalize)
    elif case_id == '10':
        radial_heat_flux_film(run, should_normalize)
    elif case_id == 'all':
        phi_film(run, should_normalize)
        ntot_film(run, should_normalize)
        upar_film(run, should_normalize)
        v_exb_film(run, should_normalize)
        v_zf_film(run, should_normalize)
        zf_shear_film(run, should_normalize)
        tpar_film(run, should_normalize)
        tperp_film(run, should_normalize)
        heat_flux_film(run, should_normalize)
        radial_heat_flux_film(run, should_normalize)
