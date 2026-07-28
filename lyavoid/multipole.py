"""Multipole decomposition and plotting for Lyman-alpha forest x void cross-correlations.

Given a cross-correlation xi(r, mu) computed from `lelantos`/`picca`
outputs (r in h^-1 Mpc, mu = r_parallel / r), this module decomposes it
into Legendre multipoles (monopole ell=0, dipole ell=1, quadrupole
ell=2, hexadecapole ell=4) via `lyavoid.xcorr_objects.Multipole`, and
provides plotting utilities for single, averaged, and multi-dataset
multipole comparisons, error-bar estimation from per-sub-sample data,
a linear-RSD beta fit from the monopole/quadrupole shapes, 2D xi maps,
and mu-wedge plots (via `picca.wedgize`).

Integration methods for the multipole projection (passed as
`multipole_method`) include: "rect", "nbody", "trap", "simps",
"rebin_trap", "rebin_simps", "quad", "romb" -- see
`lyavoid.xcorr_objects.Multipole` for their implementation.
"""
import matplotlib.pyplot as plt
import numpy as np
from iminuit import Minuit
from scipy import integrate
from scipy.interpolate import interp1d
from scipy.stats import sem

from lyavoid import utils, xcorr_objects


def get_poles(mu, da, method):
    """Decompose a 2D correlation function xi(mu) into Legendre multipoles.

    Builds a `xcorr_objects.Multipole` for ell = [0, 1, 2, 4] (monopole,
    dipole, quadrupole, hexadecapole) from the input mu/xi arrays.

    Args:
        mu (numpy.ndarray): Array of mu = r_parallel / r values.
        da (numpy.ndarray): Correlation function xi(r, mu) values, same
            shape as mu.
        method (str): Integration method used to project onto Legendre
            polynomials (e.g. "rect", "nbody", "trap", "simps",
            "rebin_trap", "rebin_simps", "quad", "romb").

    Returns:
        tuple: (monopole, dipole, quadrupole, hexadecapole) arrays, the
            ell=0,1,2,4 multipoles of xi.
    """
    ell = [0, 1, 2, 4]
    multipole = xcorr_objects.Multipole.init_from_xcorr(
        ell, mu, da, method, name=None, r_array=None, extrapolate=True
    )
    monopole = multipole.poles[0]
    dipole = multipole.poles[1]
    quadrupole = multipole.poles[2]
    hexadecapole = multipole.poles[4]
    return (monopole, dipole, quadrupole, hexadecapole)


def get_error_bars_from_no_export(
    file_xi_no_export, multipole_method, supress_first_pixels=0
):
    """Estimate multipole error bars from per-sub-sample (non-exported) correlation data.

    Loads a CrossCorr FITS file whose xi_array holds one row per
    sub-sample (not yet averaged/exported), computes the multipoles for
    each row, and takes the standard error of the mean (SEM) across rows.

    Args:
        file_xi_no_export (str): Path to the FITS file with the
            un-exported, per-sub-sample cross-correlation.
        multipole_method (str): Integration method passed to `get_poles`.
        supress_first_pixels (int): Number of leading pixels to drop when
            loading the CrossCorr. Defaults to 0.

    Returns:
        tuple: (error_monopole, error_dipole, error_quadrupole,
            error_hexadecapole) arrays, the SEM of each multipole across
            sub-samples.
    """
    xcorr = xcorr_objects.CrossCorr.init_from_fits(
        file_xi_no_export, supress_first_pixels=supress_first_pixels
    )
    mu, da = xcorr.mu_array, xcorr.xi_array
    monopole, dipole, quadrupole, hexadecapole = [], [], [], []
    for i in range(len(da)):
        (mono, di, quad, hexa) = get_poles(mu, da[i], multipole_method)
        monopole.append(mono)
        dipole.append(di)
        quadrupole.append(quad)
        hexadecapole.append(hexa)
    error_monopole = sem(np.array(monopole))
    error_dipole = sem(np.array(dipole))
    error_quadrupole = sem(np.array(quadrupole))
    error_hexadecapole = sem(np.array(hexadecapole))
    return (error_monopole, error_dipole, error_quadrupole, error_hexadecapole)


def get_mean_multipoles(
    file_xi, supress_first_pixels=0, multipole_method="rect", error_bar_file_unique=None
):
    """Compute mean multipoles (and optional errors) from one or several xi(r, mu) FITS files.

    If `file_xi` is a list, each file is treated as an independent
    realization: multipoles are computed per file and then averaged,
    with the SEM across files used as the error. If `file_xi` is a
    single path, multipoles are computed once from that file; error
    bars are then either taken from `error_bar_file_unique` (via
    `get_error_bars_from_no_export`) or left as None.

    Args:
        file_xi (str or list): Path, or list of paths, to CrossCorr FITS
            file(s).
        supress_first_pixels (int): Number of leading pixels to drop
            when loading each CrossCorr. Defaults to 0.
        multipole_method (str): Integration method passed to
            `get_poles`. Defaults to "rect".
        error_bar_file_unique (str, optional): Path to a per-sub-sample
            FITS file used to estimate error bars when `file_xi` is a
            single file.

    Returns:
        tuple: (r_array, mean_monopole, mean_dipole, mean_quadrupole,
            mean_hexadecapole, error_mean_monopole, error_mean_dipole,
            error_mean_quadrupole, error_mean_hexadecapole).
    """

    if type(file_xi) == list:
        (mean_monopole, mean_dipole, mean_quadrupole, mean_hexadecapole) = (
            [],
            [],
            [],
            [],
        )
        for i in range(len(file_xi)):
            xcorr = xcorr_objects.CrossCorr.init_from_fits(
                file_xi[i], supress_first_pixels=supress_first_pixels
            )
            (r, mu, da) = xcorr.r_array, xcorr.mu_array, xcorr.xi_array
            r[r == 0] = np.nan
            r_array = np.nanmean(r, axis=1)
            (monopole, dipole, quadrupole, hexadecapole) = get_poles(
                mu, da, multipole_method
            )
            mean_monopole.append(monopole)
            mean_dipole.append(dipole)
            mean_quadrupole.append(quadrupole)
            mean_hexadecapole.append(hexadecapole)
        mean_monopole, error_mean_monopole = np.mean(mean_monopole, axis=0), sem(
            mean_monopole, axis=0
        )
        mean_dipole, error_mean_dipole = np.mean(mean_dipole, axis=0), sem(
            mean_dipole, axis=0
        )
        mean_quadrupole, error_mean_quadrupole = np.mean(mean_quadrupole, axis=0), sem(
            mean_quadrupole, axis=0
        )
        mean_hexadecapole, error_mean_hexadecapole = np.mean(
            mean_hexadecapole, axis=0
        ), sem(mean_hexadecapole, axis=0)
    else:
        xcorr = xcorr_objects.CrossCorr.init_from_fits(
            file_xi, supress_first_pixels=supress_first_pixels
        )
        (r, mu, da) = xcorr.r_array, xcorr.mu_array, xcorr.xi_array
        r[r == 0] = np.nan
        r_array = np.nanmean(r, axis=1)
        (mean_monopole, mean_dipole, mean_quadrupole, mean_hexadecapole) = get_poles(
            mu, da, multipole_method
        )

        if error_bar_file_unique is not None:
            (
                error_mean_monopole,
                error_mean_dipole,
                error_mean_quadrupole,
                error_mean_hexadecapole,
            ) = get_error_bars_from_no_export(
                error_bar_file_unique,
                multipole_method,
                supress_first_pixels=supress_first_pixels,
            )
        else:
            (
                error_mean_monopole,
                error_mean_dipole,
                error_mean_quadrupole,
                error_mean_hexadecapole,
            ) = (None, None, None, None)

    return (
        r_array,
        mean_monopole,
        mean_dipole,
        mean_quadrupole,
        mean_hexadecapole,
        error_mean_monopole,
        error_mean_dipole,
        error_mean_quadrupole,
        error_mean_hexadecapole,
    )


def plot_multipole(
    ax,
    r_array,
    monopole,
    dipole,
    quadrupole,
    hexadecapole,
    error_monopole,
    error_dipole,
    error_quadrupole,
    error_hexadecapole,
    monopole_division=False,
    set_label=True,
    **kwargs,
):
    """Plot precomputed multipoles (monopole/dipole/quadrupole/hexadecapole) on given axes.

    One subplot per multipole in `ax`, optionally with error bars (via
    `errorbar`) or a plain line (`plot`). Supports scaling by a power of
    r, and dividing each pole by the monopole.

    Args:
        ax (list): Sequence of matplotlib Axes, one per multipole to
            plot (ordered as in `poles_to_plot`).
        r_array (numpy.ndarray): Radial bin centers, h^-1 Mpc.
        monopole, dipole, quadrupole, hexadecapole (numpy.ndarray or
            None): ell=0,1,2,4 multipole values; None entries are
            skipped for that pole.
        error_monopole, error_dipole, error_quadrupole,
            error_hexadecapole (numpy.ndarray or None): Corresponding
            1-sigma errors. If `error_monopole` is None, plain `plot`
            is used instead of `errorbar` for all poles.
        monopole_division (bool): If True, divide dipole/quadrupole/
            hexadecapole by the monopole before plotting (and adjust
            labels accordingly). Defaults to False.
        set_label (bool): If True, set y-axis labels (and x-axis label
            on the last axis). Defaults to True.
        **kwargs: Plotting options read via `utils.return_key`,
            including `style`, `poles_to_plot` (list of ell values to
            plot, default [0, 1, 2, 4]), `color`, `alpha`,
            `radius_multiplication` (power of r to multiply poles by),
            `ylabel`, `fontsize`, `labelsize_x`, `labelsize_y`,
            `linestyle`, `marker`, `marker_size`.
    """
    style = utils.return_key(kwargs, "style", None)
    poles_to_plot = utils.return_key(kwargs, "poles_to_plot", [0, 1, 2, 4])

    color = utils.return_key(kwargs, "color", None)
    alpha = utils.return_key(kwargs, "alpha", None)
    radius_multiplication_power = utils.return_key(
        kwargs, "radius_multiplication", None
    )
    radius_multiplication = (
        r_array**radius_multiplication_power
        if radius_multiplication_power is not None
        else 1
    )

    title_add = ""
    if radius_multiplication_power is not None:
        if radius_multiplication_power == 1:
            title_add = r" $r \times$"
        else:
            title_add = r" $r^{" + str(radius_multiplication_power) + "} \times$"

    ylabel = utils.return_key(kwargs, "ylabel", r"$\xi_{\mathrm{v}\alpha,")
    if monopole_division:
        poles_label = {
            0: title_add + ylabel + "0}$",
            1: title_add + ylabel + "1}$" + "/" + ylabel + "0}$",
            2: title_add + ylabel + "2}$" + "/" + ylabel + "0}$",
            4: title_add + ylabel + "4}$" + "/" + ylabel + "0}$",
        }
        poles = {
            0: monopole * radius_multiplication if monopole is not None else None,
            1: (
                dipole * radius_multiplication / monopole
                if dipole is not None
                else None
            ),
            2: (
                quadrupole * radius_multiplication / monopole
                if quadrupole is not None
                else None
            ),
            4: (
                hexadecapole * radius_multiplication / monopole
                if hexadecapole is not None
                else None
            ),
        }
        if error_monopole is not None:
            error_poles = {
                0: (
                    error_monopole * radius_multiplication
                    if error_monopole is not None
                    else None
                ),
                1: (
                    error_dipole * radius_multiplication / monopole
                    if error_dipole is not None
                    else None
                ),
                2: (
                    error_quadrupole * radius_multiplication / monopole
                    if error_quadrupole is not None
                    else None
                ),
                4: (
                    error_hexadecapole * radius_multiplication / monopole
                    if error_hexadecapole is not None
                    else None
                ),
            }
    else:
        poles = {
            0: monopole * radius_multiplication if monopole is not None else None,
            1: dipole * radius_multiplication if dipole is not None else None,
            2: quadrupole * radius_multiplication if quadrupole is not None else None,
            4: (
                hexadecapole * radius_multiplication
                if hexadecapole is not None
                else None
            ),
        }
        poles_label = {
            0: title_add + ylabel + "0}$",
            1: title_add + ylabel + "1}$",
            2: title_add + ylabel + "2}$",
            4: title_add + ylabel + "4}$",
        }
        if error_monopole is not None:
            error_poles = {
                0: (
                    error_monopole * radius_multiplication
                    if error_monopole is not None
                    else None
                ),
                1: (
                    error_dipole * radius_multiplication
                    if error_dipole is not None
                    else None
                ),
                2: (
                    error_quadrupole * radius_multiplication
                    if error_quadrupole is not None
                    else None
                ),
                4: (
                    error_hexadecapole * radius_multiplication
                    if error_hexadecapole is not None
                    else None
                ),
            }

    fontsize = utils.return_key(kwargs, "fontsize", 13)
    labelsize_x = utils.return_key(kwargs, "labelsize_x", 13)
    labelsize_y = utils.return_key(kwargs, "labelsize_y", 13)
    linestyle = utils.return_key(kwargs, "linestyle", None)
    marker = utils.return_key(kwargs, "marker", None)
    marker_size = utils.return_key(kwargs, "marker_size", None)
    for i in range(len(poles_to_plot)):
        if poles_to_plot[i] is not None:
            if style is None:
                ax[i].grid()
            if error_monopole is not None:
                ax[i].errorbar(
                    r_array,
                    poles[poles_to_plot[i]],
                    error_poles[poles_to_plot[i]],
                    color=color,
                    alpha=alpha,
                    linestyle=linestyle,
                    marker=marker,
                    markersize=marker_size,
                )
            else:
                ax[i].plot(
                    r_array,
                    poles[poles_to_plot[i]],
                    color=color,
                    alpha=alpha,
                    linestyle=linestyle,
                    marker=marker,
                    markersize=marker_size,
                )
            if set_label:
                ax[i].set_ylabel(poles_label[poles_to_plot[i]], fontsize=fontsize)
                ax[i].tick_params(axis="y", labelsize=labelsize_y)
                if i == len(poles_to_plot) - 1:
                    ax[i].set_xlabel(r"$r~[\mathrm{h^{-1} Mpc}]$", fontsize=fontsize)
                    ax[i].tick_params(axis="x", labelsize=labelsize_x)


def compute_and_plot_multipole(
    file_xi,
    save_plot,
    supress_first_pixels=0,
    error_bar=None,
    multipole_method="rect",
    monopole_division=False,
    second_plot=None,
    set_label=True,
    factor_monopole=None,
    **kwargs,
):
    """Compute multipoles from a xi(r, mu) FITS file, plot them, and optionally save.

    Loads the CrossCorr, ensures it is in (r, mu) representation,
    computes the ell=[0,1,2,4] multipoles, optionally applies a
    monopole-proportional correction to dipole/quadrupole/hexadecapole,
    plots them via `plot_multipole`, and (if `save_plot` is given) saves
    the figure and writes a `Multipole` FITS file.

    Args:
        file_xi (str): Path to the CrossCorr FITS file.
        save_plot (str or None): Output basename (without extension); if
            not None, saves "{save_plot}.pdf" and "{save_plot}.fits".
        supress_first_pixels (int): Number of leading pixels to drop.
            Defaults to 0.
        error_bar (str, optional): Path to a per-sub-sample FITS file
            used to compute error bars via
            `get_error_bars_from_no_export`.
        multipole_method (str): Integration method passed to
            `get_poles`. Defaults to "rect".
        monopole_division (bool): Passed to `plot_multipole`; divide
            higher poles by the monopole. Defaults to False.
        second_plot (tuple, optional): (fig, ax) to plot onto instead of
            creating a new figure.
        set_label (bool): Passed to `plot_multipole`. Defaults to True.
        factor_monopole (tuple, optional): (f1, f2, f3) factors added as
            `dipole += f1*monopole`, `quadrupole += f2*monopole`,
            `hexadecapole += f3*monopole`.
        **kwargs: Extra plotting options forwarded to `plot_multipole`
            (e.g. `poles_to_plot`, `figsize`).

    Returns:
        tuple: (fig, ax, r_array, monopole, dipole, quadrupole,
            hexadecapole).
    """
    style = utils.return_key(kwargs, "style", None)
    if style is not None:
        plt.style.use(style)
    xcorr = xcorr_objects.CrossCorr.init_from_fits(
        file_xi, supress_first_pixels=supress_first_pixels
    )
    if xcorr.rmu == False:
        xcorr.switch()
    (r, mu, da) = xcorr.r_array, xcorr.mu_array, xcorr.xi_array
    r[r == 0] = np.nan
    r_array = np.nanmean(r, axis=1)
    (monopole, dipole, quadrupole, hexadecapole) = get_poles(mu, da, multipole_method)
    if factor_monopole is not None:
        dipole = dipole + factor_monopole[0] * monopole
        quadrupole = quadrupole + factor_monopole[1] * monopole
        hexadecapole = hexadecapole + factor_monopole[2] * monopole
    poles_to_plot = utils.return_key(kwargs, "poles_to_plot", [0, 1, 2, 4])
    figsize = utils.return_key(kwargs, "figsize", (8, 10))
    if second_plot is not None:
        fig = second_plot[0]
        ax = second_plot[1]
    else:
        fig, ax = plt.subplots(len(poles_to_plot), 1, figsize=figsize, sharex=True)

    if error_bar is not None:
        (error_monopole, error_dipole, error_quadrupole, error_hexadecapole) = (
            get_error_bars_from_no_export(
                error_bar, multipole_method, supress_first_pixels=supress_first_pixels
            )
        )
    else:
        (error_monopole, error_dipole, error_quadrupole, error_hexadecapole) = (
            None,
            None,
            None,
            None,
        )
    plot_multipole(
        ax,
        r_array,
        monopole,
        dipole,
        quadrupole,
        hexadecapole,
        error_monopole,
        error_dipole,
        error_quadrupole,
        error_hexadecapole,
        monopole_division=monopole_division,
        set_label=set_label,
        **kwargs,
    )
    if save_plot is not None:
        plt.tight_layout()
        fig.savefig(f"{save_plot}.pdf", format="pdf")
        if error_bar is None:
            error_poles = None
        else:
            error_poles = {
                0: error_monopole,
                1: error_dipole,
                2: error_quadrupole,
                4: error_hexadecapole,
            }
        mult = xcorr_objects.Multipole(
            name=f"{save_plot}.fits",
            r_array=r_array,
            ell=[0, 1, 2, 4],
            poles={0: monopole, 1: dipole, 2: quadrupole, 4: hexadecapole},
            error_poles=error_poles,
        )
        mult.write_fits()
    return (fig, ax, r_array, monopole, dipole, quadrupole, hexadecapole)


def compute_and_plot_beta_mean(
    file_xi,
    save_plot,
    supress_first_pixels_fit=3,
    file_xi_substract=None,
    error_bar_file_unique=None,
    multipole_method="rect",
    minuit_parameters=None,
    legend=None,
    ax=None,
    substract_monopole_factor=None,
    **kwargs,
):
    """Fit the linear-RSD beta parameter from mean monopole/quadrupole and plot the fit ingredients.

    Computes mean multipoles (optionally subtracting a second dataset's
    multipoles), builds the volume-averaged monopole
    `mean_monopole_bar`, forms `diff_mono = monopole - mean_monopole_bar`
    and the (possibly background-subtracted) quadrupole, then fits
    `quadrupole = diff_mono * (2*beta)/(3+beta)` with iminuit. Plots the
    monopole, diff_mono, quadrupole and beta-rescaled quadrupole curves
    on a single axis.

    Args:
        file_xi (str or list): Path(s) to the main CrossCorr FITS
            file(s), forwarded to `get_mean_multipoles`.
        save_plot (str or None): Output basename (without extension) for
            the saved PDF; if None, the figure is not saved.
        supress_first_pixels_fit (int): Number of leading radial bins
            excluded from the beta fit. Defaults to 3.
        file_xi_substract (str or list, optional): Path(s) to a
            background dataset whose multipoles are subtracted from the
            main ones.
        error_bar_file_unique (str, optional): Path to a per-sub-sample
            FITS file for error bars, forwarded to `get_mean_multipoles`.
        multipole_method (str): Integration method passed to
            `get_mean_multipoles`. Defaults to "rect".
        minuit_parameters (dict): Keyword arguments forwarded to
            `iminuit.Minuit` to initialize the `beta` cost-function fit.
        legend (list, optional): Legend labels for the plotted curves.
        ax (matplotlib.axes.Axes, optional): Axis to plot onto; if None,
            a new figure/axis is created.
        substract_monopole_factor (float, optional): If set (together
            with `file_xi_substract`), subtract
            `substract_monopole_factor * mean_quadrupole2` from the main
            mean monopole before the beta fit.
        **kwargs: Extra options forwarded to `plot_multipole` and read
            via `utils.return_key` (e.g. `style`, `fontsize_legend`).
    """

    (
        r_array,
        mean_monopole,
        mean_dipole,
        mean_quadrupole,
        mean_hexadecapole,
        error_mean_monopole,
        error_mean_dipole,
        error_mean_quadrupole,
        error_mean_hexadecapole,
    ) = get_mean_multipoles(
        file_xi,
        multipole_method=multipole_method,
        error_bar_file_unique=error_bar_file_unique,
    )
    if file_xi_substract is not None:
        (
            r_array2,
            mean_monopole2,
            mean_dipole2,
            mean_quadrupole2,
            mean_hexadecapole2,
            error_mean_monopole2,
            error_mean_dipole2,
            error_mean_quadrupole2,
            error_mean_hexadecapole2,
        ) = get_mean_multipoles(
            file_xi_substract,
            multipole_method=multipole_method,
            error_bar_file_unique=error_bar_file_unique,
        )

        if substract_monopole_factor is not None:
            mean_monopole = mean_monopole - substract_monopole_factor * mean_quadrupole2

    mean_monopole_bar = np.zeros(mean_monopole.shape)
    for i in range(len(r_array)):
        mask = (r_array <= r_array[i]) & (r_array > 0)
        integrand = mean_monopole[mask] * r_array[mask] ** 2
        integral = integrate.simpson(integrand, r_array[mask], axis=0)
        mean_monopole_bar[i] = integral * 3 / (r_array[i] ** 3)

    if file_xi_substract is not None:
        corrected_quad = (
            mean_quadrupole[supress_first_pixels_fit:]
            - mean_quadrupole2[supress_first_pixels_fit:]
        )
        error_corrected_quad = np.sqrt(
            error_mean_quadrupole[supress_first_pixels_fit:] ** 2
            + error_mean_quadrupole2[supress_first_pixels_fit:] ** 2
        )
    else:
        corrected_quad = mean_quadrupole[supress_first_pixels_fit:]
        error_corrected_quad = error_mean_quadrupole[supress_first_pixels_fit:]

    diff_mono = (
        mean_monopole[supress_first_pixels_fit:]
        - mean_monopole_bar[supress_first_pixels_fit:]
    )
    error_diff_mono = np.sqrt(2) * error_mean_monopole[supress_first_pixels_fit:]

    mean_monopole = mean_monopole[supress_first_pixels_fit:]
    error_mean_monopole = error_mean_monopole[supress_first_pixels_fit:]

    r_array = r_array[supress_first_pixels_fit:]

    data_y = corrected_quad
    data_yerr = error_corrected_quad
    model = lambda beta: diff_mono * ((2 * beta) / (3 + beta))

    cost_function = lambda beta: np.nansum(((data_y - model(beta)) / data_yerr) ** 2)
    minuit = Minuit(cost_function, **minuit_parameters)
    minuit.migrad(10000, resume=True)
    beta = dict(minuit.values)["beta"]
    print("beta: ", dict(minuit.values)["beta"])
    print("beta error: ", dict(minuit.errors)["beta"])

    if ax is None:
        style = utils.return_key(kwargs, "style", None)
        if style is not None:
            plt.style.use(style)
        plt.figure(figsize=(10, 7))
        ax = plt.gca()

    plot_multipole(
        [ax],
        r_array,
        mean_monopole,
        None,
        None,
        None,
        error_mean_monopole,
        None,
        None,
        None,
        set_label=False,
        poles_to_plot=[0],
        **kwargs,
    )
    plot_multipole(
        [ax],
        r_array,
        diff_mono,
        None,
        None,
        None,
        error_diff_mono,
        None,
        None,
        None,
        set_label=False,
        poles_to_plot=[0],
        **kwargs,
    )
    plot_multipole(
        [ax],
        r_array,
        corrected_quad,
        None,
        None,
        None,
        error_corrected_quad,
        None,
        None,
        None,
        set_label=False,
        poles_to_plot=[0],
        **kwargs,
    )
    plot_multipole(
        [ax],
        r_array,
        ((3 + beta) / (2 * beta)) * corrected_quad,
        None,
        None,
        None,
        ((3 + beta) / (2 * beta)) * error_corrected_quad,
        None,
        None,
        None,
        set_label=False,
        poles_to_plot=[0],
        **kwargs,
    )

    fontsize_legend = utils.return_key(kwargs, "fontsize_legend", 12)
    if legend is not None:
        ax.legend(legend, fontsize=fontsize_legend)

    if save_plot is not None:
        plt.savefig(f"{save_plot}.pdf", format="pdf")


def compute_and_plot_poles(
    file_xi,
    ell_to_plot,
    save_plot,
    file_xi_substract=None,
    supress_first_pixels=0,
    error_bar_file_unique=None,
    error_bar_file_unique_substract=None,
    multipole_method="rect",
    legend=None,
    ax=None,
    plot_ratio_first=None,
    divide_pole_by=None,
    **kwargs,
):
    """Plot a single multipole (ell=0, 2 or 4) for several datasets, with optional ratio/difference sub-panel.

    For each entry in `file_xi`, computes mean multipoles via
    `get_mean_multipoles`, optionally subtracts the corresponding entry
    in `file_xi_substract` (quadrupole only), selects the requested
    `ell_to_plot` pole, optionally normalizes it by a monopole-based
    volume average (`divide_pole_by="monopole_diff"`), and plots it. If
    `plot_ratio_first` is set, a second axis shows the
    difference/relative-difference/ratio of each curve with respect to
    the first one.

    Args:
        file_xi (list): List of CrossCorr FITS file paths (or lists
            thereof), one entry per dataset/curve.
        ell_to_plot (int): Multipole order to plot; one of 0, 2, 4.
        save_plot (str or None): Output basename (without extension); if
            not None, saves "{save_plot}.pdf".
        file_xi_substract (list, optional): Background dataset(s) whose
            quadrupole is subtracted from each corresponding `file_xi`
            entry.
        supress_first_pixels (int): Number of leading pixels to drop.
            Defaults to 0.
        error_bar_file_unique (list, optional): Per-dataset paths for
            error bar estimation, forwarded to `get_mean_multipoles`.
        error_bar_file_unique_substract (list, optional): Per-dataset
            paths for error bars of the substracted dataset.
        multipole_method (str): Integration method passed to
            `get_mean_multipoles`. Defaults to "rect".
        legend (list, optional): Legend labels.
        ax (list of matplotlib.axes.Axes, optional): Axes to plot onto;
            if None, new axes are created (2 rows if `plot_ratio_first`,
            else 1).
        plot_ratio_first (str, optional): One of "diff", "absdiff",
            "reldiff", "ratio"; if set, plots the comparison of each
            curve to the first curve on a second axis.
        divide_pole_by (str, optional): If "monopole_diff", divides the
            selected pole by `monopole - monopole_bar` (volume-averaged
            monopole subtraction).
        **kwargs: Extra options forwarded to `plot_multipole` and read
            via `utils.return_key` (e.g. `style`, `figsize`,
            `dotted_line`, `dotted_line_color`, `ylabel`, `ylabel2`,
            `plot_x_label`, `fontsize`, `fontsize_legend`).
    """

    if ax is None:
        style = utils.return_key(kwargs, "style", None)
        if style is not None:
            plt.style.use(style)
        figsize = utils.return_key(kwargs, "figsize", None)
        if plot_ratio_first:
            fig, ax = plt.subplots(2, 1, figsize=figsize, sharex=True)
        else:
            plt.figure(figsize=figsize)
            ax = [plt.gca()]
    dotted_line = utils.return_key(kwargs, "dotted_line", None)
    dotted_line_color = utils.return_key(kwargs, "dotted_line_color", None)
    if dotted_line is not None:
        linestyle_save = utils.return_key(kwargs, "linestyle", "-")
        color_save = utils.return_key(kwargs, "color", None)
    for i in range(len(file_xi)):
        if dotted_line is not None:
            if i == dotted_line:
                kwargs["linestyle"] = "--"
                kwargs["color"] = dotted_line_color
            else:
                kwargs["linestyle"] = linestyle_save
                kwargs["color"] = color_save
        (
            r_array,
            monopole,
            dipole,
            quadrupole,
            hexadecapole,
            error_monopole,
            error_dipole,
            error_quadrupole,
            error_hexadecapole,
        ) = get_mean_multipoles(
            file_xi[i],
            supress_first_pixels=supress_first_pixels,
            multipole_method=multipole_method,
            error_bar_file_unique=error_bar_file_unique[i],
        )
        if file_xi_substract is not None:
            (
                r_array_substract,
                monopole_substract,
                dipole_substract,
                quadrupole_substract,
                hexadecapole_substract,
                error_monopole_substract,
                error_dipole_substract,
                error_quadrupole_substract,
                error_hexadecapole_substract,
            ) = get_mean_multipoles(
                file_xi_substract[i],
                supress_first_pixels=supress_first_pixels,
                multipole_method=multipole_method,
                error_bar_file_unique=error_bar_file_unique_substract[i],
            )

            error_quadrupole = np.sqrt(
                error_quadrupole**2 + error_quadrupole_substract**2
            )
            quadrupole = quadrupole - quadrupole_substract

        if i != len(file_xi) - 1:
            error_quadrupole = None
        if ell_to_plot == 0:
            poles_to_plot = monopole
            error_poles_to_plot = error_monopole
        elif ell_to_plot == 2:
            poles_to_plot = quadrupole
            error_poles_to_plot = error_quadrupole
        elif ell_to_plot == 4:
            poles_to_plot = hexadecapole
            error_poles_to_plot = error_hexadecapole

        if divide_pole_by is not None:
            if divide_pole_by == "monopole_diff":
                monopole_bar = np.zeros(monopole.shape)
                for i in range(len(r_array)):
                    mask = (r_array <= r_array[i]) & (r_array > 0)
                    integrand = monopole[mask] * r_array[mask] ** 2
                    integral = integrate.simpson(integrand, r_array[mask], axis=0)
                    monopole_bar[i] = integral * 3 / (r_array[i] ** 3)
                poles_to_plot = poles_to_plot / (monopole - monopole_bar)
                if error_poles_to_plot is not None:
                    error_poles_to_plot = error_poles_to_plot / (
                        monopole - monopole_bar
                    )

        if i == 0:
            poles_to_plot_save = poles_to_plot
        plot_multipole(
            [ax[0]],
            r_array,
            poles_to_plot,
            None,
            None,
            None,
            error_poles_to_plot,
            None,
            None,
            None,
            set_label=False,
            poles_to_plot=[0],
            **kwargs,
        )
        if plot_ratio_first is not None:
            if plot_ratio_first == "diff":
                plot_multipole(
                    [ax[1]],
                    r_array,
                    (poles_to_plot - poles_to_plot_save),
                    None,
                    None,
                    None,
                    error_poles_to_plot,
                    None,
                    None,
                    None,
                    set_label=False,
                    poles_to_plot=[0],
                    **kwargs,
                )
            elif plot_ratio_first == "absdiff":
                plot_multipole(
                    [ax[1]],
                    r_array,
                    np.abs(poles_to_plot - poles_to_plot_save),
                    None,
                    None,
                    None,
                    error_poles_to_plot,
                    None,
                    None,
                    None,
                    set_label=False,
                    poles_to_plot=[0],
                    **kwargs,
                )
            if plot_ratio_first == "reldiff":
                plot_multipole(
                    [ax[1]],
                    r_array,
                    (poles_to_plot - poles_to_plot_save) / poles_to_plot_save,
                    None,
                    None,
                    None,
                    error_poles_to_plot / poles_to_plot_save,
                    None,
                    None,
                    None,
                    set_label=False,
                    poles_to_plot=[0],
                    **kwargs,
                )
            if plot_ratio_first == "ratio":
                plot_multipole(
                    [ax[1]],
                    r_array,
                    poles_to_plot / poles_to_plot_save,
                    None,
                    None,
                    None,
                    error_poles_to_plot / poles_to_plot_save,
                    None,
                    None,
                    None,
                    set_label=False,
                    poles_to_plot=[0],
                    **kwargs,
                )

    fontsize = utils.return_key(kwargs, "fontsize", 13)
    labelsize_y = utils.return_key(kwargs, "labelsize_y", 13)
    labelsize_x = utils.return_key(kwargs, "labelsize_y", 13)
    ylabel = utils.return_key(kwargs, "ylabel", None)
    if ylabel is not None:
        ax[0].set_ylabel(ylabel, fontsize=fontsize)
        ax[0].tick_params(axis="y", labelsize=labelsize_y)

    if plot_ratio_first is not None:
        ylabel2 = utils.return_key(kwargs, "ylabel2", None)
        if ylabel is not None:
            ax[1].set_ylabel(ylabel2, fontsize=fontsize)
            ax[1].tick_params(axis="y", labelsize=labelsize_y)
    plot_x_label = utils.return_key(kwargs, "plot_x_label", True)
    if plot_x_label:
        ax[-1].set_xlabel(r"$r~[\mathrm{h^{-1} Mpc}]$", fontsize=fontsize)
        ax[-1].tick_params(axis="x", labelsize=labelsize_x)

    fontsize_legend = utils.return_key(kwargs, "fontsize_legend", 12)
    if legend is not None:
        ax[0].legend(legend, fontsize=fontsize_legend)

    if save_plot is not None:
        plt.tight_layout()
        plt.savefig(f"{save_plot}.pdf", format="pdf")


def compute_and_plot_beta_mean_several(
    file_xi,
    save_plot,
    file_xi_substract,
    error_bar_file_unique,
    substract_monopole_factor,
    legend,
    supress_first_pixels_fit=3,
    multipole_method="rect",
    minuit_parameters=None,
    **kwargs,
):
    """Run `compute_and_plot_beta_mean` for several datasets side by side and save a combined figure.

    Creates a row of subplots (one per entry in `file_xi`) sharing the y
    axis, and calls `compute_and_plot_beta_mean` on each with its
    matching `file_xi_substract`, `error_bar_file_unique`,
    `substract_monopole_factor` and `legend` entry.

    Args:
        file_xi (list): List of main CrossCorr FITS file paths/lists,
            one per subplot.
        save_plot (str): Output basename (without extension) for the
            saved PDF.
        file_xi_substract (list): Background dataset(s) per subplot,
            forwarded to `compute_and_plot_beta_mean`.
        error_bar_file_unique (list): Per-subplot error-bar file paths.
        substract_monopole_factor (list): Per-subplot monopole
            subtraction factors.
        legend (list): Per-subplot legend labels.
        supress_first_pixels_fit (int): Leading radial bins excluded
            from each beta fit. Defaults to 3.
        multipole_method (str): Integration method. Defaults to "rect".
        minuit_parameters (dict, optional): Forwarded to
            `iminuit.Minuit` for each beta fit.
        **kwargs: Extra options forwarded to `compute_and_plot_beta_mean`
            and read via `utils.return_key` (e.g. `style`, `figsize`).
    """
    style = utils.return_key(kwargs, "style", None)
    if style is not None:
        plt.style.use(style)

    figsize = utils.return_key(kwargs, "figsize", None)
    fig, ax = plt.subplots(1, len(file_xi), figsize=figsize, sharey=True)

    for i in range(len(file_xi)):
        compute_and_plot_beta_mean(
            file_xi[i],
            None,
            supress_first_pixels_fit=supress_first_pixels_fit,
            file_xi_substract=file_xi_substract[i],
            error_bar_file_unique=error_bar_file_unique[i],
            substract_monopole_factor=substract_monopole_factor[i],
            multipole_method=multipole_method,
            minuit_parameters=minuit_parameters,
            legend=legend[i],
            ax=ax[i],
            **kwargs,
        )
    plt.tight_layout()
    plt.savefig(f"{save_plot}.pdf", format="pdf")


def compute_and_plot_multipole_several(
    file_xi,
    save_plot,
    supress_first_pixels=0,
    error_bar=None,
    multipole_method="rect",
    monopole_division=False,
    second_plot=None,
    set_label=True,
    factor_monopole=None,
    name_mean_multipole=None,
    **kwargs,
):
    """Compute and plot the mean multipoles across several xi(r, mu) FITS files.

    Calls `compute_and_plot_multipole` for each file in `file_xi`
    (stacking onto the same figure after the first), collects the
    resulting multipoles, averages them across files (SEM as error),
    optionally writes the averaged multipoles to a `Multipole` FITS
    file, then plots the mean curves via `plot_multipole`.

    Args:
        file_xi (list): List of CrossCorr FITS file paths.
        save_plot (str or None): Forwarded to `compute_and_plot_multipole`
            for each individual file's own save (typically None here
            since the mean is saved separately via `name_mean_multipole`).
        supress_first_pixels (int): Number of leading pixels to drop.
            Defaults to 0.
        error_bar (list, optional): Per-file error-bar FITS paths,
            forwarded to `compute_and_plot_multipole`.
        multipole_method (str): Integration method. Defaults to "rect".
        monopole_division (bool): Passed to `plot_multipole`. Defaults
            to False.
        second_plot (tuple, optional): (fig, ax) to plot onto instead of
            creating a new figure.
        set_label (bool): Whether to set axis labels on the mean-curve
            plot. Defaults to True.
        factor_monopole (tuple, optional): Forwarded to
            `compute_and_plot_multipole` for each file.
        name_mean_multipole (str, optional): If given, basename used to
            save the averaged multipoles as a `Multipole` FITS file
            ("{name_mean_multipole}.fits").
        **kwargs: Extra options forwarded to `compute_and_plot_multipole`
            / `plot_multipole`, read via `utils.return_key` (e.g.
            `alpha_mean`, `plot_mean_error`).

    Returns:
        tuple: (fig, ax, r_array, mean_monopole, mean_dipole,
            mean_quadrupole, mean_hexadecapole).
    """

    (mean_monopole, mean_dipole, mean_quadrupole, mean_hexadecapole) = [], [], [], []
    for i in range(len(file_xi)):
        (fig, ax, r_array, monopole, dipole, quadrupole, hexadecapole) = (
            compute_and_plot_multipole(
                file_xi[i],
                save_plot,
                supress_first_pixels=supress_first_pixels,
                error_bar=error_bar[i] if error_bar is not None else None,
                multipole_method=multipole_method,
                monopole_division=monopole_division,
                second_plot=second_plot,
                set_label=False,
                factor_monopole=factor_monopole,
                **kwargs,
            )
        )

        if i == 0:
            second_plot = (fig, ax)
        mean_monopole.append(monopole)
        mean_dipole.append(dipole)
        mean_quadrupole.append(quadrupole)
        mean_hexadecapole.append(hexadecapole)

    error_mean_monopole = sem(mean_monopole, axis=0)
    error_mean_dipole = sem(mean_dipole, axis=0)
    error_mean_quadrupole = sem(mean_quadrupole, axis=0)
    error_mean_hexadecapole = sem(mean_hexadecapole, axis=0)
    mean_monopole = np.mean(mean_monopole, axis=0)
    mean_dipole = np.mean(mean_dipole, axis=0)
    mean_quadrupole = np.mean(mean_quadrupole, axis=0)
    mean_hexadecapole = np.mean(mean_hexadecapole, axis=0)

    alpha_mean = utils.return_key(kwargs, "alpha_mean", None)
    kwargs["alpha"] = alpha_mean
    if name_mean_multipole is not None:
        if error_mean_monopole is not None:
            error_poles = {
                0: error_mean_monopole,
                1: error_mean_dipole,
                2: error_mean_quadrupole,
                4: error_mean_hexadecapole,
            }
        else:
            error_poles = None

        mult = xcorr_objects.Multipole(
            name=f"{name_mean_multipole}.fits",
            r_array=r_array,
            ell=[0, 1, 2, 4],
            poles={
                0: mean_monopole,
                1: mean_dipole,
                2: mean_quadrupole,
                4: mean_hexadecapole,
            },
            error_poles=error_poles,
        )
        mult.write_fits()

    plot_error = utils.return_key(kwargs, "plot_mean_error", False)
    if not (plot_error):
        (
            error_mean_monopole,
            error_mean_dipole,
            error_mean_quadrupole,
            error_mean_hexadecapole,
        ) = (None, None, None, None)
    plot_multipole(
        ax,
        r_array,
        mean_monopole,
        mean_dipole,
        mean_quadrupole,
        mean_hexadecapole,
        error_mean_monopole,
        error_mean_dipole,
        error_mean_quadrupole,
        error_mean_hexadecapole,
        monopole_division=monopole_division,
        set_label=set_label,
        **kwargs,
    )
    return (
        fig,
        ax,
        r_array,
        mean_monopole,
        mean_dipole,
        mean_quadrupole,
        mean_hexadecapole,
    )


def compute_and_plot_multipole_comparison(
    name_in,
    name_in2,
    nameout,
    supress_first_pixels=0,
    error_bar=None,
    error_bar2=None,
    file_xi_optional=None,
    error_bar_optional=None,
    legend=None,
    multipole_method="rect",
    monopole_division=False,
    monopole_normalization=False,
    **kwargs,
):
    """Plot multipoles from two (or more) datasets on the same axes for comparison.

    Plots `name_in` and `name_in2` via `compute_and_plot_multipole` on a
    shared figure, then optionally overlays any number of additional
    datasets from `file_xi_optional`, each optionally rescaled so its
    monopole matches the first dataset's monopole
    (`monopole_normalization`), before saving the combined figure.

    Args:
        name_in (str): Path to the first CrossCorr FITS file.
        name_in2 (str): Path to the second CrossCorr FITS file.
        nameout (str): Output basename (without extension) for the saved
            PDF.
        supress_first_pixels (int): Number of leading pixels to drop.
            Defaults to 0.
        error_bar (str, optional): Error-bar FITS path for `name_in`.
        error_bar2 (str, optional): Error-bar FITS path for `name_in2`.
        file_xi_optional (list, optional): Additional CrossCorr FITS
            file paths to overlay.
        error_bar_optional (list, optional): Error-bar FITS paths
            matching `file_xi_optional`.
        legend (list, optional): Legend labels for the first axis.
        multipole_method (str): Integration method. Defaults to "rect".
        monopole_division (bool): Passed to `plot_multipole`. Defaults
            to False.
        monopole_normalization (bool): If True, rescale each optional
            dataset's xi so its monopole matches `name_in`'s monopole
            (interpolated onto its own r grid) before computing its
            multipoles. Defaults to False.
        **kwargs: Extra options forwarded to `compute_and_plot_multipole`
            and `plot_multipole`.
    """
    (fig, ax, r_array1, monopole1, dipole1, quadrupole1, hexadecapole1) = (
        compute_and_plot_multipole(
            name_in,
            None,
            supress_first_pixels=supress_first_pixels,
            error_bar=error_bar,
            multipole_method=multipole_method,
            monopole_division=monopole_division,
            set_label=True,
            **kwargs,
        )
    )
    (fig, ax, r_array2, monopole2, dipole2, quadrupole2, hexadecapole2) = (
        compute_and_plot_multipole(
            name_in2,
            None,
            supress_first_pixels=supress_first_pixels,
            error_bar=error_bar2,
            multipole_method=multipole_method,
            second_plot=(fig, ax),
            monopole_division=monopole_division,
            set_label=False,
            **kwargs,
        )
    )
    if file_xi_optional is not None:
        for i in range(len(file_xi_optional)):

            xcorr = xcorr_objects.CrossCorr.init_from_fits(
                file_xi_optional[i], supress_first_pixels=supress_first_pixels
            )
            (r, mu, da) = xcorr.r_array, xcorr.mu_array, xcorr.xi_array
            r_array = np.mean(r, axis=1)
            if monopole_normalization:
                (monopole, dipole, quadrupole, hexadecapole) = get_poles(
                    mu, da, multipole_method
                )
                monopole_ratio = (
                    interp1d(r_array1, monopole1, bounds_error=False, fill_value=1.0)(
                        r_array
                    )
                    / monopole
                )
                for j in range(da.shape[1]):
                    da[:, j] = da[:, j] * monopole_ratio
            (monopole, dipole, quadrupole, hexadecapole) = get_poles(
                mu, da, multipole_method
            )
            if error_bar_optional is not None:
                (error_monopole, error_dipole, error_quadrupole, error_hexadecapole) = (
                    get_error_bars_from_no_export(
                        error_bar_optional[i],
                        multipole_method,
                        supress_first_pixels=supress_first_pixels,
                    )
                )
            else:
                (error_monopole, error_dipole, error_quadrupole, error_hexadecapole) = (
                    None,
                    None,
                    None,
                    None,
                )

            plot_multipole(
                ax,
                r_array,
                monopole,
                dipole,
                quadrupole,
                hexadecapole,
                error_monopole,
                error_dipole,
                error_quadrupole,
                error_hexadecapole,
                monopole_division=monopole_division,
                set_label=False,
                **kwargs,
            )
    ax[0].legend(legend)
    plt.tight_layout()
    fig.savefig(f"{nameout}.pdf", format="pdf")


def compute_and_plot_multipole_several_comparison(
    names_in,
    nameout,
    error_bar=None,
    supress_first_pixels=0,
    file_xi_optional=None,
    error_bar_optional=None,
    legend=None,
    legend_elements=None,
    factor_monopole=None,
    multipole_method="rect",
    monopole_division=False,
    alpha=0.4,
    name_multipole=None,
    name_multipole_optional=None,
    dotted_line_optional=None,
    fit_file=None,
    **kwargs,
):
    """Plot a main multipole curve (single file or mean of several) plus overlaid comparison datasets and an optional fit curve.

    Plots `names_in` (a single file via `compute_and_plot_multipole`, or
    a list of files averaged via `compute_and_plot_multipole_several`),
    then overlays each entry of `file_xi_optional` (single file or list)
    with a distinct color, optionally highlighting one as a dashed
    reference curve via `dotted_line_optional`. If `fit_file` is given,
    overplots a theoretical curve loaded with `numpy.loadtxt`. Saves the
    combined figure.

    Args:
        names_in (str or list): Main dataset: single CrossCorr FITS
            path, or list of paths averaged together.
        nameout (str): Output basename (without extension) for the saved
            PDF.
        error_bar (str or list, optional): Error-bar source(s) for the
            main dataset.
        supress_first_pixels (int): Number of leading pixels to drop.
            Defaults to 0.
        file_xi_optional (list, optional): Additional datasets to
            overlay; each entry is a single path or a list of paths
            (averaged).
        error_bar_optional (list, optional): Error-bar source(s)
            matching `file_xi_optional`.
        legend (list, optional): Legend labels (used if
            `legend_elements` is None).
        legend_elements (list, optional): Explicit matplotlib legend
            handles; takes precedence over `legend`.
        factor_monopole (tuple, optional): Forwarded to the underlying
            `compute_and_plot_multipole(_several)` calls.
        multipole_method (str): Integration method. Defaults to "rect".
        monopole_division (bool): Passed to `plot_multipole`. Defaults
            to False.
        alpha (float): Alpha (transparency) used for mean curves of
            multi-file datasets. Defaults to 0.4.
        name_multipole (str, optional): Basename to save the main
            dataset's multipoles as a FITS file.
        name_multipole_optional (list, optional): Basenames to save each
            optional dataset's multipoles as FITS files.
        dotted_line_optional (int, optional): Index into
            `file_xi_optional` to render as a dashed black reference
            curve instead of its normal style.
        fit_file (str, optional): Path to a text file (columns r, xi)
            with a theoretical curve to overplot via `numpy.loadtxt`.
        **kwargs: Extra options read via `utils.return_key` (e.g.
            `style`, `linestyle_alone`, `linestyle_mean`,
            `marker_alone`, `marker_alone_color`, `marker_mean`,
            `fit_file_color`, `fit_file_axis`, `fontsize_legend`,
            `radius_multiplication`, `poles_to_plot`) and forwarded to
            the underlying plotting calls.
    """
    style = utils.return_key(kwargs, "style", None)
    if style is not None:
        plt.style.use(style)

    linestyle_alone = utils.return_key(kwargs, "linestyle_alone", "None")
    linestyle_mean = utils.return_key(kwargs, "linestyle_mean", "-")
    marker_alone = utils.return_key(kwargs, "marker_alone", ".")
    marker_alone_color = utils.return_key(kwargs, "marker_alone_color", None)
    marker_mean = utils.return_key(kwargs, "marker_mean", None)
    fit_file_color = utils.return_key(kwargs, "fit_file_color", "k")
    fit_file_axis = utils.return_key(kwargs, "fit_file_axis", 1)
    fontsize_legend = utils.return_key(kwargs, "fontsize_legend", None)

    if type(names_in) == list:
        (
            fig,
            ax,
            r_array,
            mean_monopole,
            mean_dipole,
            mean_quadrupole,
            mean_hexadecapole,
        ) = compute_and_plot_multipole_several(
            names_in,
            None,
            supress_first_pixels=supress_first_pixels,
            error_bar=error_bar,
            multipole_method=multipole_method,
            monopole_division=monopole_division,
            second_plot=None,
            set_label=True,
            factor_monopole=factor_monopole,
            alpha=alpha,
            color="C0",
            name_mean_multipole=name_multipole,
            linestyle=linestyle_mean,
            marker=marker_mean,
            **kwargs,
        )

    else:
        if marker_alone_color is not None:
            color = marker_alone_color
        (fig, ax, r_array, monopole, dipole, quadrupole, hexadecapole) = (
            compute_and_plot_multipole(
                names_in,
                name_multipole,
                supress_first_pixels=supress_first_pixels,
                error_bar=error_bar,
                multipole_method=multipole_method,
                second_plot=None,
                monopole_division=monopole_division,
                factor_monopole=factor_monopole,
                set_label=True,
                color="C0",
                linestyle=linestyle_alone,
                marker=marker_alone,
                **kwargs,
            )
        )

    second_plot = (fig, ax)

    if file_xi_optional is not None:
        for i in range(len(file_xi_optional)):
            color = f"C{i+1}"
            if dotted_line_optional is not None:
                if dotted_line_optional == i:
                    color = "k"
                    linestyle_mean = "--"
                    alpha = 0.0
                    kwargs["alpha_mean"] = 0.6
                    kwargs["poles_to_plot"] = [0, 2, 4]

            if error_bar_optional is None:
                error_bar = None
            else:
                error_bar = error_bar_optional[i]
            if type(file_xi_optional[i]) == list:
                if name_multipole_optional is None:
                    name_multipole_optional_i = None
                else:
                    name_multipole_optional_i = name_multipole_optional[i]
                (
                    fig,
                    ax,
                    r_array,
                    mean_monopole,
                    mean_dipole,
                    mean_quadrupole,
                    mean_hexadecapole,
                ) = compute_and_plot_multipole_several(
                    file_xi_optional[i],
                    None,
                    supress_first_pixels=supress_first_pixels,
                    error_bar=error_bar,
                    multipole_method=multipole_method,
                    monopole_division=monopole_division,
                    second_plot=second_plot,
                    set_label=False,
                    factor_monopole=factor_monopole,
                    alpha=alpha,
                    color=color,
                    linestyle=linestyle_mean,
                    marker=marker_mean,
                    name_mean_multipole=name_multipole_optional_i,
                    **kwargs,
                )

            else:
                if marker_alone_color is not None:
                    color = marker_alone_color
                (fig, ax, r_array, monopole, dipole, quadrupole, hexadecapole) = (
                    compute_and_plot_multipole(
                        file_xi_optional[i],
                        name_multipole_optional[i],
                        supress_first_pixels=supress_first_pixels,
                        error_bar=error_bar,
                        multipole_method=multipole_method,
                        second_plot=second_plot,
                        monopole_division=monopole_division,
                        factor_monopole=factor_monopole,
                        set_label=False,
                        color=color,
                        linestyle=linestyle_alone,
                        marker=marker_alone,
                        **kwargs,
                    )
                )

    if fit_file is not None:
        xi2_fit = np.loadtxt(fit_file)
        radius_multiplication_power = utils.return_key(
            kwargs, "radius_multiplication", None
        )
        radius_multiplication = (
            xi2_fit[:, 0] ** radius_multiplication_power
            if radius_multiplication_power is not None
            else 1
        )

        ax[fit_file_axis].plot(
            xi2_fit[:, 0], radius_multiplication * xi2_fit[:, 1], color=fit_file_color
        )

    if legend_elements is not None:
        ax[0].legend(handles=legend_elements, fontsize=fontsize_legend)
    else:
        ax[0].legend(legend, fontsize=fontsize_legend)
    plt.tight_layout()
    fig.savefig(f"{nameout}.pdf", format="pdf")


def plot_2d(name_in, name_out, supress_first_pixels=0, **kwargs):
    """Load a CrossCorr FITS file and plot its 2D xi(r_parallel, r_transverse) map.

    Args:
        name_in (str): Path to the input CrossCorr FITS file.
        name_out (str): Output name/basename passed to
            `xcorr_objects.CrossCorr.plot_2d`.
        supress_first_pixels (int): Number of leading pixels to drop
            when loading the CrossCorr. Defaults to 0.
        **kwargs: Extra options forwarded to
            `xcorr_objects.CrossCorr.plot_2d`.
    """
    xcorr = xcorr_objects.CrossCorr.init_from_fits(
        name_in, supress_first_pixels=supress_first_pixels
    )
    xcorr.plot_2d(name=name_out, **kwargs)


def compute_and_plot_wedge_comparison(file_xi, nameout, comparison, legend=None):
    """Plot wedge-averaged (mu-binned) correlation functions for two datasets side by side.

    Draws 4 mu-wedges (edges [0.0, 0.5, 0.8, 0.95, 1.0]) for `file_xi`
    and `comparison` on the same 4-row figure via `add_wedge`, and saves
    it.

    Args:
        file_xi (str): Path to the first `picca`-format FITS correlation
            file (DA/CO format read by `add_wedge`).
        nameout (str): Output basename (without extension) for the saved
            PDF.
        comparison (str): Path to the second FITS correlation file to
            overlay.
        legend (list, optional): Legend labels for the top axis.
    """
    fig, axes = plt.subplots(4, 1, figsize=(8, 10), sharex=True)
    mus = [0.0, 0.5, 0.8, 0.95, 1.0]
    add_wedge(file_xi, mus, axes)
    add_wedge(comparison, mus, axes)
    for i in range(len(axes)):
        axes[i].grid()
        axes[i].set_ylabel(f"{mus[i]}" + r"$< \mu <$" + f"{mus[i+1]}")
    axes[-1].set_xlabel(r"$r~[\mathrm{Mpc/h}]$")
    axes[0].legend(legend)
    axes[0].set_title(r"$r^2\xi(r)$")
    plt.savefig(f"{nameout}.pdf", format="pdf")


def compute_and_plot_wedge(file_xi, nameout):
    """Plot wedge-averaged (mu-binned) correlation function for a single dataset.

    Draws 4 mu-wedges (edges [0.0, 0.5, 0.8, 0.95, 1.0]) for `file_xi`
    via `add_wedge` and saves the figure.

    Args:
        file_xi (str): Path to a `picca`-format FITS correlation file
            (DA/CO format read by `add_wedge`).
        nameout (str): Output basename (without extension) for the saved
            PDF.
    """
    fig, axes = plt.subplots(4, 1, figsize=(8, 10), sharex=True)
    mus = [0.0, 0.5, 0.8, 0.95, 1.0]
    add_wedge(file_xi, mus, axes)
    for i in range(len(axes)):
        axes[i].grid()
        axes[i].set_ylabel(f"{mus[i]}" + r"$< \mu <$" + f"{mus[i+1]}")
    axes[0].set_title(r"$r^2\xi(r)$")
    axes[-1].set_xlabel(r"$r~[\mathrm{Mpc/h}]$")
    plt.savefig(f"{nameout}.pdf", format="pdf")


def add_wedge(file_xi, mus, axes):
    """Compute and plot mu-wedge-averaged correlation functions from a picca-format FITS file.

    For each consecutive pair of mu edges in `mus`, reads DA/CO and the
    (RPMIN, RPMAX, RTMAX, NP, NT) header geometry from `file_xi`, builds
    a `picca.wedgize.wedge` and computes the wedge-averaged correlation
    and its diagonal-covariance error bars, then plots them with
    `errorbar` on the corresponding axis.

    Args:
        file_xi (str): Path to a `picca`-format FITS correlation file
            containing DA (data) and CO (covariance) HDUs plus
            RPMIN/RPMAX/RTMAX/NP/NT header keywords.
        mus (list): Monotonic list of mu bin edges (e.g. [0.0, 0.5, 0.8,
            0.95, 1.0]); wedges are formed from consecutive pairs.
        axes (list): Sequence of matplotlib Axes, one per wedge (mu
            interval), to plot onto.
    """
    import fitsio
    from picca import wedgize

    ax = 0
    for mumin, mumax in zip(mus[:-1], mus[1:]):
        h = fitsio.FITS(file_xi)
        # ff = h5py.File(fitefile,'r')
        # fit = ff[fitpath+'/fit'][...]
        # ff.close()
        da = h[1]["DA"][:]
        co = h[1]["CO"][:]
        rpmin = h[1].read_header()["RPMIN"]
        rpmax = h[1].read_header()["RPMAX"]
        rtmax = h[1].read_header()["RTMAX"]
        nrp = h[1].read_header()["NP"]
        nt = h[1].read_header()["NT"]
        h.close()
        # ff.close()
        b = wedgize.wedge(
            mumin=mumin,
            mumax=mumax,
            rtmin=0,
            rtmax=rtmax,
            rpmin=rpmin,
            rpmax=rpmax,
            rmax=50,
            nrp=nrp,
            nrt=nt,
            absoluteMu=True,
        )
        r, d, c = b.wedge(da, co)
        # r,f,_ = b.wedge(fit,co)
        # axes[ax].errorbar(r,d*r**2,yerr=np.sqrt(c.diagonal())*r**2,fmt="o")
        axes[ax].errorbar(r, d, yerr=np.sqrt(c.diagonal()), fmt="o")
        ax = ax + 1
        # plt.plot(r,f*r**2)
