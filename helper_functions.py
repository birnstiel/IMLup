import tempfile
import warnings
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import astropy.constants as c
import astropy.units as u

from imgcube import imagecube
import dsharp_helper as dh
import dsharp_opac as opacity
import disklab

from dipsy.utils import get_interfaces_from_log_cell_centers
from dipsy import get_powerlaw_dust_distribution
from radmc3dPy import image

au = c.au.cgs.value
M_sun = c.M_sun.cgs.value
L_sun = c.L_sun.cgs.value
R_sun = c.R_sun.cgs.value


def movingaverage(interval, window_size):
    window = np.ones(int(window_size)) / float(window_size)
    return np.convolve(interval, window, 'same')


def make_disklab2d_model(
    parameters,
    mstar,
    lstar,
    tstar,
    nr,
    alpha,
    rin,
    rout,
    r_c,
    opac_fname,
    show_plots=False
):

    # The different indices in the parameters list correspond to different physical paramters

    sigma_coeff = parameters[0]
    sigma_exp = parameters[1]
    size_exp = parameters[2]
    amax_coeff = parameters[3]
    amax_exp = parameters[4]
    d2g_coeff = parameters[5]
    d2g_exp = parameters[6]

    # read some values from the parameters file

    with np.load(opac_fname) as fid:
        a_opac = fid['a']
        rho_s = fid['rho_s']
        n_a = len(a_opac)

    # start with the 1D model

    d = disklab.DiskRadialModel(mstar=mstar, lstar=lstar, tstar=tstar, nr=nr, alpha=alpha, rin=rin, rout=rout)
    d.make_disk_from_simplified_lbp(sigma_coeff, r_c, sigma_exp)

    if d.mass / mstar > 0.2:
        warnings.warn('Disk mass is unreasonably high: M_disk / Mstar = {d.mass/mstar:.2g}')

    # add the dust, based on the dust-to-gas parameters

    d2g = d2g_coeff * ((d.r / au)**d2g_exp)
    a_max = amax_coeff * (d.r / au)**(-amax_exp)

    a_i = get_interfaces_from_log_cell_centers(a_opac)
    a, a_i, sig_da = get_powerlaw_dust_distribution(d.sigma * d2g, np.minimum(a_opac[-1], a_max), q=4 - size_exp, na=n_a, a0=a_i[0], a1=a_i[-1])

    for _sig, _a in zip(np.transpose(sig_da), a_opac):
        d.add_dust(agrain=_a, xigrain=rho_s, dtg=_sig / d.sigma)

    if show_plots:

        f, ax = plt.subplots()

        ax.contourf(d.r / au, a_opac, np.log10(sig_da.T))

        ax.loglog(d.r / au, a_max, label='a_max')
        ax.loglog(d.r / au, d2g, label='d2g')

        ax.set_ylim(1e-5, 1e0)
        ax.legend()

    # load the opacity from the previously calculated opacity table
    for dust in d.dust:
        dust.grain.read_opacity(str(opac_fname))

    # compute the mean opacities
    d.meanopacitymodel = ['dustcomponents', {'method': 'simplemixing'}]
    d.compute_mean_opacity()

    if show_plots:

        f, ax = plt.subplots()
        ax.loglog(d.r / au, d.mean_opacity_planck)
        ax.loglog(d.r / au, d.mean_opacity_rosseland)

    # smooth the mean opacities

    d.mean_opacity_planck[7:-7] = movingaverage(d.mean_opacity_planck, 10)[7:-7]
    d.mean_opacity_rosseland[7:-7] = movingaverage(d.mean_opacity_rosseland, 10)[7:-7]

    if show_plots:
        ax.loglog(d.r / au, d.mean_opacity_planck, 'C0--')
        ax.loglog(d.r / au, d.mean_opacity_rosseland, 'C1--')

        f, ax = plt.subplots()
        ax.loglog(d.r / au, d.tmid)

    # iterate the temperature

    for iter in range(100):
        d.compute_hsurf()
        d.compute_flareindex()
        d.compute_flareangle_from_flareindex(inclrstar=True)
        d.compute_disktmid(keeptvisc=False)
        d.compute_cs_and_hp()

    # ---- Make a 2D model out of it ----

    disk2d = disklab.Disk2D(
        disk=d,
        meanopacitymodel=d.meanopacitymodel,
        nz=100)

    # taken from snippet vertstruc 2d_1
    for vert in disk2d.verts:
        vert.iterate_vertical_structure()
    disk2d.radial_raytrace()
    for vert in disk2d.verts:
        vert.solve_vert_rad_diffusion()
        vert.tgas = (vert.tgas**4 + 15**4)**(1 / 4)
        for dust in vert.dust:
            dust.compute_settling_mixing_equilibrium()

    # --- done setting up the radmc3d model ---
    return disk2d


def get_profile_from_fits(fname, clip=2.5, show_plots=False, inc=0, PA=0, z0=0.0, psi=0.0):
    """Get radial profile from fits file.

    Reads a fits file and determines a radial profile with `imagecube`

    fname : str | path
        path to fits file

    clip : float
        clip the image at that many image units (usually arcsec)

    show_plots : bool
        if true: produce some plots for sanity checking

    inc, PA : float
        inclination and position angle used in the radial profile

    z0, psi : float
        the scale height at 1 arcse and the radial exponent used in the deprojection

    Returns:
    x, y, dy: arrays
        radial grid, intensity (cgs), error (cgs)
    """

    data = imagecube(fname, clip=clip)

    x, y, dy = data.radial_profile(inc=inc, PA=PA)

    y = (y * u.Jy / data.beam_area_str).cgs.value
    dy = (dy * u.Jy / data.beam_area_str).cgs.value

    if show_plots:
        f, ax = plt.subplots()
        ax.semilogy(x, y)
        ax.fill_between(x, y - dy, y + dy, alpha=0.5)
        ax.set_ylim(bottom=1e-16)

    return x, y, dy


def make_opacs(a, lam, fname='dustkappa_IMLUP', constants=None, n_theta=101):
    "make optical constants file"

    if n_theta // 2 == n_theta / 2:
        n_theta += 1
        print(f'n_theta needs to be odd, will set it to {n_theta}')

    n_a = len(a)
    n_lam = len(lam)
    opac_fname = Path(fname).with_suffix('.npz')

    if constants is None:
        constants = opacity.get_dsharp_mix()

    diel_const, rho_s = constants

    run_opac = True

    if opac_fname.is_file():
        with np.load(opac_fname) as fid:
            opac_dict = {k: v for k, v in fid.items()}
        if (
            (len(opac_dict['a']) == n_a) and
            np.allclose(opac_dict['a'], a) and
            (len(opac_dict['lam']) == n_lam) and
            np.allclose(opac_dict['lam'], lam) and
            (len(opac_dict['theta']) == n_theta) and
            (opac_dict['rho_s'] == rho_s)
        ):
            print(f'reading from file {opac_fname}')
            run_opac = False

    if run_opac:
        # call the Mie calculation & store the opacity in a npz file
        opac_dict = opacity.get_smooth_opacities(
            a,
            lam,
            rho_s=rho_s,
            diel_const=diel_const,
            extrapolate_large_grains=False,
            n_angle=(n_theta + 1) // 2)

        print(f'writing opacity to {opac_fname} ... ', end='', flush=True)
        opacity.write_disklab_opacity(opac_fname, opac_dict)
        print('Done!')

    return opac_dict


def chop_forward_scattering(opac_dict, chopforward=3):
    """Chop the forward scattering.

    This part chops the very-forward scattering part of the phase function.
    This very-forward scattering part is basically the same as no scattering,
    but is treated by the code as a scattering event. By cutting this part out
    of the phase function, we avoid those non-scattering scattering events.
    This needs to recalculate the scattering opacity kappa_sca and asymmetry
    factor g.

    Parameters
    ----------
    opac_dict : dict
        opacity dictionary as produced by dsharp_opac

    chopforward : float
        up to which angle to we truncate the forward scattering
    """

    k_sca = opac_dict['k_sca']
    S1 = opac_dict['S1']
    S2 = opac_dict['S2']
    theta = opac_dict['theta']
    g = opac_dict['g']
    rho_s = opac_dict['rho_s']
    lam = opac_dict['lam']
    a = opac_dict['a']
    m = 4 * np.pi / 3 * rho_s * a**3

    n_a = len(a)
    n_lam = len(lam)
    n_theta = len(theta)

    zscat = opacity.calculate_mueller_matrix(lam, m, S1, S2, theta=theta, k_sca=k_sca)['zscat']

    zscat_nochop = zscat.copy()

    for grain in range(n_a):
        for i in range(n_lam):
            #
            # Now loop over the grain sizes
            #
            if chopforward > 0:
                iang = np.where(theta < chopforward)
                if theta[0] == 0.0:
                    iiang = np.max(iang) + 1
                else:
                    iiang = np.min(iang) - 1
                zscat[grain, i, iang, :] = zscat[grain, i, iiang, :]
                mu = np.cos(theta * np.pi / 180.)
                dmu = np.abs(mu[1:n_theta] - mu[0:(n_theta - 1)])
                zav = 0.5 * (zscat[grain, i, 1:n_theta, 0] + zscat[grain, i, 0:n_theta - 1, 0])
                dum = 0.5 * zav * dmu
                sum = dum.sum() * 4 * np.pi
                k_sca[grain, i] = sum

                mu_2 = 0.5 * (np.cos(theta[1:n_theta] * np.pi / 180.) + np.cos(theta[0:n_theta - 1] * np.pi / 180.))
                P_mu = 0.5 * ((2 * np.pi * zscat[grain, i, 1:n_theta, 0] / k_sca[grain, i]) + (2 * np.pi * zscat[grain, i, 0:n_theta - 1, 0] / k_sca[grain, i]))
                g[grain, i] = np.sum(P_mu * mu_2 * dmu)

    return zscat, zscat_nochop, k_sca, g


def write_radmc3d(disk2d, lam, path, show_plots=False, nphot=10000000):
    """
    convert the disk2d object to radmc3d format and write the radmc3d input files.

    disk2d : disklab.Disk2D instance
        the disk

    lam : array
        wavelength grid [cm]

    path : str | path
        the path into which to write the output

    show_plots : bool
        if true: produce some plots for checking

    nphot : int
        number of photons to send
    """

    rmcd = disklab.radmc3d.get_radmc3d_arrays(disk2d, showplots=show_plots)

    # Assign the radmc3d data

    ri = rmcd['ri']
    thetai = rmcd['thetai']
    phii = rmcd['phii']
    rho = rmcd['rho']
    n_a = rho.shape[-1]

    # we need to tile this for each species

    rmcd_temp = rmcd['temp'][:, :, None] * np.ones(n_a)[None, None, :]

    # Define the wavelength grid for the radiative transfer

    lam_mic = lam * 1e4

    # Write the `RADMC3D` input

    disklab.radmc3d.write_stars_input(disk2d.disk, lam_mic, path=path)
    disklab.radmc3d.write_grid(ri, thetai, phii, mirror=False, path=path)
    disklab.radmc3d.write_dust_density(rmcd_temp, fname='dust_temperature.dat', path=path, mirror=False)
    disklab.radmc3d.write_dust_density(rho, mirror=False, path=path)
    disklab.radmc3d.write_wavelength_micron(lam_mic, path=path)
    disklab.radmc3d.write_opacity(disk2d, path=path)
    disklab.radmc3d.write_radmc3d_input(
        {'scattering_mode': 5, 'scattering_mode_max': 5, 'nphot': nphot},
        path=path)