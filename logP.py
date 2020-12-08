# coding: utf-8


import getpass
import tempfile
from pathlib import Path

import numpy as np

import astropy.constants as c

from gofish import imagecube
from radmc3dPy import image
import dsharp_opac as opacity
from disklab.radmc3d import write
import disklab

from helper_functions import get_profile_from_fits
from helper_functions import make_disklab2d_model
from helper_functions import write_radmc3d
from helper_functions import read_opacs


au = c.au.cgs.value
M_sun = c.M_sun.cgs.value
L_sun = c.L_sun.cgs.value
R_sun = c.R_sun.cgs.value


def logP(parameters, options, debug=False):

    # set the path to the radmc3d executable

    if getpass.getuser() == 'birnstiel':
        radmc3d_exec = Path('~/.bin/radmc3d').expanduser()
    else:
        radmc3d_exec = Path('~/bin/radmc3d').expanduser()

    # get the options set in the notebook

    PA = options['PA']
    inc = options['inc']
    dpc = options['distance']
    clip = options['clip']
    lam_mm = options['lam_mm']

    mstar = options['mstar']
    lstar = options['lstar']
    tstar = options['tstar']

    # get the mm observables

    x_mm_obs = options['x_mm_obs']
    y_mm_obs = options['y_mm_obs']
    dy_mm_obs = options['dy_mm_obs']
    fname_mm_obs = options['fname_mm_obs']

    # get options used in the scattered light image

    z0 = options['z0']
    psi = options['psi']
    lam_sca = options['lam_sca']
    beam_sca = options['beam_sca']

    fname_sca_obs = options['fname_sca_obs']

    # get the scattered light data

    x_sca_obs = options['x_sca_obs']
    y_sca_obs = options['y_sca_obs']
    dy_sca_obs = options['dy_sca_obs']

    # name of the opacities file

    fname_opac = options['fname_opac']

    # get the disklab model parameters

    nr = options['nr']
    rin = options['rin']
    r_c = options['r_c']
    rout = options['rout']
    alpha = options['alpha']

    # create a temporary folder in the current folder

    temp_directory = tempfile.TemporaryDirectory(dir='.')
    temp_path = temp_directory.name

    # ### make the disklab 2D model

    disk2d = make_disklab2d_model(
        parameters,
        mstar,
        lstar,
        tstar,
        nr,
        alpha,
        rin,
        rout,
        r_c,
        fname_opac,
        show_plots=debug
    )

    print(f'disk to star mass ratio = {disk2d.disk.mass / disk2d.disk.mstar:.2g}')

    # read the wavelength grid from the opacity file and write out radmc setup

    opac_dict = read_opacs(fname_opac)
    lam_opac = opac_dict['lam']
    n_a = len(opac_dict['a'])

    write_radmc3d(disk2d, lam_opac, temp_path, show_plots=debug)

    # ## Calculate the mm continuum image

    fname_mm_sim = Path(temp_path) / 'image_mm.fits'
    disklab.radmc3d.radmc3d(
        f'image incl {inc} posang {PA-90} npix 500 lambda {lam_mm * 1e4} sizeau {2 * rout / au} secondorder  setthreads 1',
        path=temp_path,
        executable=str(radmc3d_exec)
    )

    radmc_image = Path(temp_path) / 'image.out'
    if radmc_image.is_file():
        im_mm_sim = image.readImage(str(radmc_image))
        radmc_image.replace(Path(temp_path) / 'image_mm.out')
        im_mm_sim.writeFits(str(fname_mm_sim), dpc=dpc, coord='15h56m09.17658s -37d56m06.1193s')

    # Read in the fits files into imagecubes, and copy the beam information from the observation to the simulation.

    iq_mm_obs = imagecube(str(fname_mm_obs))
    iq_mm_sim = imagecube(str(fname_mm_sim))
    iq_mm_sim.bmaj, iq_mm_sim.bmin, iq_mm_sim.bpa = iq_mm_obs.beam
    iq_mm_sim.beamarea_arcsec = iq_mm_sim._calculate_beam_area_arcsec()
    iq_mm_sim.beamarea_str = iq_mm_sim._calculate_beam_area_str()

    x_mm_sim, y_mm_sim, dy_mm_sim = get_profile_from_fits(
        str(fname_mm_sim),
        clip=clip,
        inc=inc, PA=PA,
        z0=0.0,
        psi=0.0,
        beam=iq_mm_obs.beam,
        show_plots=debug)

    i_max = max(len(x_mm_obs), len(x_mm_sim))

    x_mm_sim = x_mm_sim[:i_max]
    y_mm_sim = y_mm_sim[:i_max]
    dy_mm_sim = dy_mm_sim[:i_max]
    x_mm_obs = x_mm_obs[:i_max]
    y_mm_obs = y_mm_obs[:i_max]
    dy_mm_obs = dy_mm_obs[:i_max]

    if not np.allclose(x_mm_sim, x_mm_obs):
        try:
            from IPython import embed
            embed()
        except Exception:
            raise AssertionError('observed and simulated radial profile grids are not equal')

    # TODO: Calculate the log probability for the mm here

    # %% Scattered light image

    for i_grain in range(n_a):
        opacity.write_radmc3d_scatmat_file(i_grain, opac_dict, f'{i_grain}', path=temp_path)

    with open(Path(temp_path) / 'dustopac.inp', 'w') as f:
        write(f, '2               Format number of this file')
        write(f, '{}              Nr of dust species'.format(n_a))

        for i_grain in range(n_a):
            write(f, '============================================================================')
            write(f, '10               Way in which this dust species is read')
            write(f, '0               0=Thermal grain')
            write(f, '{}              Extension of name of dustscatmat_***.inp file'.format(i_grain))

        write(f, '----------------------------------------------------------------------------')

    # image calculation
    disklab.radmc3d.radmc3d(
        f'image incl {inc} posang {PA-90} npix 500 lambda {lam_sca / 1e-4} sizeau {2 * rout / au} setthreads 4',
        path=temp_path,
        executable=str(radmc3d_exec))

    fname_sca_sim = Path(temp_path) / 'image_sca.fits'
    if (Path(temp_path) / 'image.out').is_file():
        (Path(temp_path) / 'image.out').replace(fname_sca_sim.with_suffix('.out'))

    im = image.readImage(str(fname_sca_sim.with_suffix('.out')))
    im.writeFits(str(fname_sca_sim), dpc=dpc, coord='15h56m09.17658s -37d56m06.1193s')

    iq_sca_obs = imagecube(str(fname_sca_obs))
    iq_sca_sim = imagecube(str(fname_sca_sim))

    # set the "beam" for the two images such that the samplint happens identically

    for iq in [iq_sca_obs, iq_sca_sim]:
        iq.bmaj, iq.bmin, iq.bpa = beam_sca
        iq.beamarea_arcsec = iq._calculate_beam_area_arcsec()
        iq.beamarea_str = iq._calculate_beam_area_str()

    # %% get the scattered light profile

    x_sca_sim, y_sca_sim, dy_sca_sim = get_profile_from_fits(
        str(fname_sca_sim),
        clip=clip,
        inc=inc, PA=PA,
        z0=z0,
        psi=psi,
        beam=beam_sca,
        show_plots=debug)

    i_max = min(len(x_sca_obs), len(x_sca_sim))

    x_sca_sim = x_sca_sim[:i_max]
    y_sca_sim = y_sca_sim[:i_max]
    dy_sca_sim = dy_sca_sim[:i_max]
    x_sca_obs = x_sca_obs[:i_max]
    y_sca_obs = y_sca_obs[:i_max]
    dy_sca_obs = dy_sca_obs[:i_max]

    if not np.allclose(x_sca_sim, x_sca_obs):
        try:
            from IPython import embed
            embed()
        except Exception:
            raise AssertionError('observed and simulated radial profile grids are not equal')

    # TODO: calculate logP
    logP = -np.inf

    if debug:
        debug_info = {
            'x_sca_sim': x_sca_sim,
            'y_sca_sim': y_sca_sim,
            'dy_sca_sim': dy_sca_sim,
            'x_mm_sim': x_mm_sim,
            'y_mm_sim': y_mm_sim,
            'dy_mm_sim': dy_mm_sim,
            'x_sca_obs': x_sca_obs,
            'y_sca_obs': y_sca_obs,
            'dy_sca_obs': dy_sca_obs,
            'x_mm_obs': x_mm_obs,
            'y_mm_obs': y_mm_obs,
            'dy_mm_obs': dy_mm_obs,
            'iq_mm_obs': iq_mm_obs,
            'iq_mm_sim': iq_mm_sim,
            'iq_sca_obs': iq_sca_obs,
            'iq_sca_sim': iq_sca_sim,
            'disk2d': disk2d,
            'fname_sca_sim': fname_sca_sim,
            'temp_path': temp_path,
        }
        return logP, debug_info
    else:
        return logP
