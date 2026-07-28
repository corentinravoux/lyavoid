"""Cross-correlation of Lyman-alpha forest deltas with void catalogs.

This module computes the cross-correlation xi(r, mu) (or xi(r_parallel,
r_transverse)) between void catalogs (as produced by `lelantos`) and
Lyman-alpha forest delta files (picca-style FITS deltas), by binning
weighted pixel-void pairs. It offers both a pure-Python/numpy
implementation and thin wrappers around the `picca_xcf.py` /
`picca_export.py` command-line executables. Results are packaged into
`lyavoid.xcorr_objects.CrossCorr` objects (attributes `r_array`,
`mu_array`, `xi_array`, `z_array`, `exported`, `rmu`) which can be
written to disk.

Coordinate conventions: `r` (and `r_parallel`/`r_transverse`) are in
h^-1 Mpc; `mu` is defined as `r_parallel / r`.
"""

import glob
import os

import fitsio
import numpy as np

from lelantos import tomographic_objects
from lyavoid import xcorr_objects

lambdaLy = 1215.673123130217
import subprocess


def run_cross_corr_picca(dict_picca, rmu=True):
    """Run picca's `picca_xcf.py` executable as a subprocess.

    Builds a command line from `dict_picca` (each key/value pair becomes
    a `--key value` argument) and optionally appends the `--rmu` flag,
    then executes it via a shell subprocess.

    Args:
        dict_picca (dict): Mapping of `picca_xcf.py` CLI argument names
            (without leading dashes) to their values.
        rmu (bool, optional): If True, append the `--rmu` flag to bin
            the correlation in (r, mu) instead of (r_parallel,
            r_transverse). Defaults to True.

    Returns:
        None
    """
    cmd = " picca_xcf.py"
    for key, value in dict_picca.items():
        cmd += f" --{key} {value}"
    if rmu:
        cmd += " --rmu"
    subprocess.call(cmd, shell=True)


def run_export_picca(input, output, smooth=False):
    """Run picca's `picca_export.py` executable as a subprocess.

    Args:
        input (str): Path passed as `--data` (input correlation file).
        output (str): Path passed as `--out` (exported output file).
        smooth (bool, optional): If False, appends the
            `--do-not-smooth-cov` flag to disable covariance smoothing.
            Defaults to False.

    Returns:
        None
    """
    cmd = " picca_export.py"
    cmd += f" --data {input}"
    cmd += f" --out {output}"
    if not smooth:
        cmd += " --do-not-smooth-cov"
    subprocess.call(cmd, shell=True)


def create_xcf(delta_path, void_catalog, xcorr_options, rtrp=False):
    """Build the cross-correlation configuration/data dict ("xcf").

    Loads voids and deltas into `xcorr_options` and populates it with
    the neighbor lists needed for the pair-counting step.

    Args:
        delta_path (str): Directory containing picca-style
            `delta-*.fits` files.
        void_catalog (str): Path to the void catalog FITS file, loaded
            via `lelantos.tomographic_objects.VoidCatalog`.
        xcorr_options (dict): Dictionary of cross-correlation
            parameters (e.g. `r_min`, `r_max`, `mu_min`, `mu_max`,
            `nbins_r`, `nbins_mu`, `z_ref`, `z_evol_obj`, `z_evol_del`,
            `z_min_obj`, `z_max_obj`, and, if `rtrp` is True,
            `rp_min`, `rp_max`, `rt_max`); mutated in place and
            returned as `xcf`.
        rtrp (bool, optional): If True, fill neighbors using
            (r_parallel, r_transverse) binning via `fill_neighs_rtrp`;
            otherwise use (r, mu) binning via `fill_neighs`. Defaults
            to False.

    Returns:
        dict: The populated `xcf` dictionary, including `"voids"`,
        `"deltas"`, and per-delta `"neighbors"` entries.
    """
    xcf = xcorr_options
    read_voids(xcf, void_catalog)
    read_deltas(xcf, delta_path)
    if rtrp:
        fill_neighs_rtrp(xcf)
    else:
        fill_neighs(xcf)
    return xcf


def read_voids(xcf, void_catalog):
    """Load a void catalog, apply redshift cuts, and store it in `xcf`.

    Reads the void catalog from FITS, optionally cuts it to the
    `[z_min_obj, z_max_obj]` redshift range, computes cross-correlation
    parameters (x, y, z, weight, redshift columns), applies a redshift
    evolution weighting correction, and stores the result under
    `xcf["voids"]`.

    Args:
        xcf (dict): Cross-correlation parameter dictionary; must
            contain `"z_max_obj"`, `"z_min_obj"`, `"z_ref"`, and
            `"z_evol_obj"`. Mutated in place (adds `"voids"`).
        void_catalog (str): Path to the void catalog FITS file.

    Returns:
        None
    """
    void = tomographic_objects.VoidCatalog.init_from_fits(void_catalog)
    if xcf["z_max_obj"] is not None:
        void.cut_catalog(coord_min=(-np.inf, -np.inf, xcf["z_min_obj"]))
    if xcf["z_max_obj"] is not None:
        void.cut_catalog(coord_max=(np.inf, np.inf, xcf["z_max_obj"]))
    void_coords = void.compute_cross_corr_parameters()
    void_coords[:, 3] = void_coords[:, 3] * (
        ((1 + void_coords[:, 4]) / (1 + xcf["z_ref"])) ** (xcf["z_evol_obj"] - 1)
    )
    xcf["voids"] = void_coords
    print("..done. Number of voids:", void_coords.shape[0])


def read_deltas(xcf, delta_path):
    """Load picca-style delta FITS files and store them in `xcf`.

    Globs `delta-*.fits` files under `delta_path`, and for every HDU
    (forest) in every file, extracts the sightline position (`X`, `Y`
    header keywords), pixel redshift (`Z`), delta field (`DELTA`), and
    computes redshift-evolution-weighted pixel weights from `WEIGHT`
    and `LOGLAM`.

    Args:
        xcf (dict): Cross-correlation parameter dictionary; must
            contain `"z_ref"` and `"z_evol_del"`. Mutated in place
            (adds `"deltas"`).
        delta_path (str): Directory containing picca-style
            `delta-*.fits` files.

    Returns:
        None
    """
    # CR - To replace when lelantos is used
    print("Reading deltas from ", delta_path)
    deltas = []
    delta_names = glob.glob(os.path.join(delta_path, "delta-*.fits"))
    for delta_name in delta_names:
        delta = fitsio.FITS(delta_name)
        for i in range(1, len(delta)):
            delta_dict = {}
            header = delta[i].read_header()
            delta_dict["x"] = header["X"]
            delta_dict["y"] = header["Y"]
            delta_dict["z"] = delta[i]["Z"][:]
            delta_dict["delta"] = delta[i]["DELTA"][:]
            redshift = (10 ** delta[i]["LOGLAM"][:] / lambdaLy) - 1
            delta_dict["weights"] = delta[i]["WEIGHT"][:] * (
                ((1 + redshift) / (1 + xcf["z_ref"])) ** (xcf["z_evol_del"] - 1)
            )
            deltas.append(delta_dict)
    xcf["deltas"] = deltas
    print(
        "..done. Number of deltas:",
        len(deltas),
        "Number of pixels:",
        len(deltas) * len(redshift),
    )


def fill_neighs(xcf):
    """Attach nearby voids to each delta, using an isotropic radius cut.

    For each delta (forest), selects voids whose transverse distance
    from the sightline is at most `r_max` and whose z-coordinate falls
    within `r_max` of the delta's pixel z-range, and stores them under
    `d["neighbors"]`.

    Args:
        xcf (dict): Cross-correlation parameter dictionary; must
            contain `"deltas"`, `"voids"`, and `"r_max"`. Mutated in
            place (each delta dict gains a `"neighbors"` entry).

    Returns:
        None
    """
    print("Filling neighbors..")
    deltas = xcf["deltas"]
    voids = xcf["voids"]
    r_max = xcf["r_max"]
    for d in deltas:
        x, y, z = d["x"], d["y"], d["z"]
        mask = np.sqrt((voids[:, 0] - x) ** 2 + (voids[:, 1] - y) ** 2) <= r_max
        mask &= voids[:, 2] <= np.max(z) + r_max
        mask &= voids[:, 2] >= np.min(z) - r_max
        d["neighbors"] = voids[mask]
    print("..done")


def fill_neighs_rtrp(xcf):
    """Attach nearby voids to each delta, using separate rp/rt cuts.

    For each delta (forest), selects voids whose transverse distance
    from the sightline is at most `rt_max` and whose z-coordinate lies
    within `[rp_min, rp_max]` of the delta's pixel z-range, and stores
    them under `d["neighbors"]`.

    Args:
        xcf (dict): Cross-correlation parameter dictionary; must
            contain `"deltas"`, `"voids"`, `"rp_max"`, `"rp_min"`, and
            `"rt_max"`. Mutated in place (each delta dict gains a
            `"neighbors"` entry).

    Returns:
        None
    """
    print("Filling neighbors..")
    deltas = xcf["deltas"]
    voids = xcf["voids"]
    rp_max = xcf["rp_max"]
    rp_min = xcf["rp_min"]
    rt_max = xcf["rt_max"]
    for d in deltas:
        x, y, z = d["x"], d["y"], d["z"]
        mask = np.sqrt((voids[:, 0] - x) ** 2 + (voids[:, 1] - y) ** 2) <= rt_max
        mask &= voids[:, 2] <= np.max(z) + rp_max
        mask &= voids[:, 2] >= np.min(z) + rp_min
        d["neighbors"] = voids[mask]
    print("..done")


def compute_xi_forest_pairs(xcf, x1, y1, z1, weights1, delta1, x2, y2, z2, weights2):
    """Computes the contribution of a given pair of forests to the correlation
    function.
    Args:
        z1: array of float
            Redshift of pixel 1
        r_comov1: array of float
            Comoving distance for forest 1 (in Mpc/h)
        dist_m1: array of float
            Comoving angular distance for forest 1 (in Mpc/h)
        weights1: array of float
            Pixel weights for forest 1
        delta1: array of float
            Delta field for forest 1
        z2: array of float
            Redshift of pixel 2
        r_comov2: array of float
            Comoving distance for forest 2 (in Mpc/h)
        dist_m2: array of float
            Comoving angular distance for forest 2 (in Mpc/h)
        weights2: array of float
            Pixel weights for forest 2
        ang: array of float
            Angular separation between pixels in forests 1 and 2
    Returns:
        The following variables:
            rebin_weight: The weight of the correlation function pixels
                properly rebinned
            rebin_xi: The correlation function properly rebinned
            rebin_r_par: The parallel distance of the correlation function
                pixels properly rebinned
            rebin_r_trans: The transverse distance of the correlation function
                pixels properly rebinned
            rebin_z: The redshift of the correlation function pixels properly
                rebinned
            rebin_num_pairs: The number of pairs of the correlation function
                pixels properly rebinned
    """

    z = (z1[:, None] + z2) / 2

    x1_array, y1_array = np.full(z1.shape, x1), np.full(z1.shape, y1)
    rt = np.sqrt((x1_array[:, None] - x2) ** 2 + (y1_array[:, None] - y2) ** 2)
    rp = z2 - z1[:, None]
    r = np.sqrt(rp**2 + rt**2)
    mu = rp / r

    weights12 = weights1[:, None] * weights2
    delta_times_weight = (weights1 * delta1)[:, None] * weights2

    w = (
        (r > xcf["r_min"])
        & (r < xcf["r_max"])
        & (mu > xcf["mu_min"])
        & (mu < xcf["mu_max"])
    )

    r = r[w]
    mu = mu[w]
    z = z[w]
    weights12 = weights12[w]
    delta_times_weight = delta_times_weight[w]

    bins_r = (
        (r - xcf["r_min"]) / (xcf["r_max"] - xcf["r_min"]) * xcf["nbins_r"]
    ).astype(int)
    bins_mu = (
        (mu - xcf["mu_min"]) / (xcf["mu_max"] - xcf["mu_min"]) * xcf["nbins_mu"]
    ).astype(int)

    bins = bins_mu + xcf["nbins_mu"] * bins_r

    rebin_xi = np.bincount(bins, weights=delta_times_weight)
    rebin_weight = np.bincount(bins, weights=weights12)
    rebin_r = np.bincount(bins, weights=r * weights12)
    rebin_mu = np.bincount(bins, weights=mu * weights12)
    rebin_z = np.bincount(bins, weights=z * weights12)
    rebin_num_pairs = np.bincount(bins, weights=(weights12 > 0.0))

    return (rebin_weight, rebin_xi, rebin_r, rebin_mu, rebin_z, rebin_num_pairs)


def compute_xi_forest_pairs_rprt(
    xcf, x1, y1, z1, weights1, delta1, x2, y2, z2, weights2
):
    """Compute a forest-void pair's contribution, binned in (rp, rt).

    Analogous to `compute_xi_forest_pairs` but bins pairs on the
    parallel distance `rp` and transverse distance `rt` separately
    instead of (r, mu).

    Args:
        xcf (dict): Cross-correlation parameter dictionary; must
            contain `"rp_min"`, `"rp_max"`, `"rt_max"`, `"nbins_rp"`,
            and `"nbins_rt"`.
        x1 (float): x-coordinate of the delta (forest) sightline.
        y1 (float): y-coordinate of the delta (forest) sightline.
        z1 (numpy.ndarray): z-coordinates (redshift proxy) of forest
            pixels.
        weights1 (numpy.ndarray): Pixel weights for the forest.
        delta1 (numpy.ndarray): Delta field values for the forest.
        x2 (numpy.ndarray): x-coordinates of the neighboring voids.
        y2 (numpy.ndarray): y-coordinates of the neighboring voids.
        z2 (numpy.ndarray): z-coordinates of the neighboring voids.
        weights2 (numpy.ndarray): Weights of the neighboring voids.

    Returns:
        tuple: `(rebin_weight, rebin_xi, rebin_rp, rebin_rt, rebin_z,
        rebin_num_pairs)`, each a 1D `numpy.ndarray` of length
        `nbins_rp * nbins_rt` giving, per (rp, rt) bin, the summed
        pair weight, weighted correlation contribution, weighted `rp`,
        weighted `rt`, weighted redshift, and pair count respectively.
    """
    z = (z1[:, None] + z2) / 2

    x1_array, y1_array = np.full(z1.shape, x1), np.full(z1.shape, y1)
    rt = np.sqrt((x1_array[:, None] - x2) ** 2 + (y1_array[:, None] - y2) ** 2)
    rp = z2 - z1[:, None]

    weights12 = weights1[:, None] * weights2
    delta_times_weight = (weights1 * delta1)[:, None] * weights2

    w = (rp > xcf["rp_min"]) & (rp < xcf["rp_max"]) & (rt < xcf["rt_max"])

    rp = rp[w]
    rt = rt[w]
    z = z[w]
    weights12 = weights12[w]
    delta_times_weight = delta_times_weight[w]

    bins_rp = (
        (rp - xcf["rp_min"]) / (xcf["rp_max"] - xcf["rp_min"]) * xcf["nbins_rp"]
    ).astype(int)
    bins_rt = ((rt / xcf["rt_max"]) * xcf["nbins_rt"]).astype(int)

    bins = bins_rt + xcf["nbins_rt"] * bins_rp

    rebin_xi = np.bincount(bins, weights=delta_times_weight)
    rebin_weight = np.bincount(bins, weights=weights12)
    rebin_rp = np.bincount(bins, weights=rp * weights12)
    rebin_rt = np.bincount(bins, weights=rt * weights12)
    rebin_z = np.bincount(bins, weights=z * weights12)
    rebin_num_pairs = np.bincount(bins, weights=(weights12 > 0.0))

    return (rebin_weight, rebin_xi, rebin_rp, rebin_rt, rebin_z, rebin_num_pairs)


def xcorr_cartesian_mine(xcf, save_corr=None):
    """Compute xi(r, mu) via direct binned_statistic_2d (unweighted mean).

    Alternative implementation of the (r, mu) cross-correlation: for
    every delta with neighboring voids, computes raw pair separations
    (r, mu) and weighted delta*void products, then bins them with
    `scipy.stats.binned_statistic_2d` (mean statistic) rather than the
    manual `np.bincount` rebinning used in `xcorr_cartesian`.

    Note:
        This function references an undefined `r_max` mask expression
        (`r < xcf["r_max"]` on a Python list) and an undefined `r`
        inside `save_corr`'s `z_array=z`; it is a work-in-progress /
        alternative code path.

    Args:
        xcf (dict): Cross-correlation parameter dictionary; must
            contain `"nbins_r"`, `"nbins_mu"`, `"r_max"`, and
            `"deltas"` (each delta dict expected to carry a
            `"neighbors"` array).
        save_corr (str, optional): If provided, name used to build and
            write out a `lyavoid.xcorr_objects.CrossCorr` object.
            Defaults to None (no export).

    Returns:
        tuple: `(r, mu, xi, z)` — the (masked/binned) separations,
        angular coefficients, correlation values, and redshifts.
    """
    nr = xcf["nbins_r"]
    nmu = xcf["nbins_mu"]
    xi = np.zeros(nr * nmu)
    r = np.zeros(nr * nmu)
    mu = np.zeros(nr * nmu)
    z = np.zeros(nr * nmu)
    deltas = xcf["deltas"]

    xi = []
    r = []
    mu = []
    for d in deltas:
        voids = d["neighbors"]
        if len(voids) != 0:
            delta_values = d["delta"]
            x_del, y_del, z_del = d["x"], d["y"], d["z"]
            weights_deltas = d["weights"]
            x_voids = voids[:, 0]
            y_voids = voids[:, 1]
            z_voids = voids[:, 2]
            weights_voids = voids[:, 3]

            rt = np.sqrt(
                (x_del[:, None] - x_voids) ** 2 + (y_del[:, None] - y_voids) ** 2
            )
            rp = -(z_del[:, None] - z_voids)
            r.append(np.ravel(np.sqrt(rp**2 + rt**2)))
            mu.append(np.ravel(rp / r))
            xi.append(
                np.ravel((weights_deltas * delta_values)[:, None] * weights_voids)
            )

    mask = r < xcf["r_max"]
    r = r[mask]
    mu = mu[mask]
    xi = xi[mask]

    from scipy.stats import binned_statistic_2d

    xi_array, r_array, mu_array, bc = binned_statistic_2d(
        r, mu, xi, "mean", bins=[nr, nmu]
    )

    if save_corr is not None:
        xcorr = xcorr_objects.CrossCorr(
            name=save_corr,
            mu_array=mu,
            r_array=r,
            xi_array=xi,
            z_array=z,
            exported=True,
        )
        xcorr.write(xcf_param=xcf)

    return (r, mu, xi, z)


def xcorr_cartesian(xcf, save_corr=None):
    """Compute the (r, mu) cross-correlation xi(r, mu) between deltas and voids.

    For every delta with neighboring voids, accumulates rebinned pair
    weights, correlation contributions, separations, angular
    coefficients, redshifts, and pair counts (via
    `compute_xi_forest_pairs`) into flat `nbins_r * nbins_mu` arrays,
    normalizes by the total weight per bin, and reshapes the result to
    `(nbins_r, nbins_mu)`.

    Args:
        xcf (dict): Cross-correlation parameter dictionary; must
            contain `"nbins_r"`, `"nbins_mu"`, `"r_min"`, `"r_max"`,
            `"mu_min"`, `"mu_max"`, and `"deltas"` (each delta dict
            expected to carry a `"neighbors"` array).
        save_corr (str, optional): If provided, name used to build and
            write out a `lyavoid.xcorr_objects.CrossCorr` object.
            Defaults to None (no export).

    Returns:
        tuple: `(r, mu, xi, z)`, each a 2D `numpy.ndarray` of shape
        `(nbins_r, nbins_mu)` giving the weighted-mean separation,
        angular coefficient, correlation value, and redshift per bin.
    """
    nr = xcf["nbins_r"]
    nmu = xcf["nbins_mu"]
    xi = np.zeros(nr * nmu)
    weights = np.zeros(nr * nmu)
    r = np.zeros(nr * nmu)
    mu = np.zeros(nr * nmu)
    z = np.zeros(nr * nmu)
    num_pairs = np.zeros(nr * nmu, dtype=np.int64)
    deltas = xcf["deltas"]
    for d in deltas:
        voids = d["neighbors"]
        if len(voids) != 0:
            delta_values = d["delta"]
            x_del, y_del, z_del = d["x"], d["y"], d["z"]
            weights_deltas = d["weights"]
            x_voids = voids[:, 0]
            y_voids = voids[:, 1]
            z_voids = voids[:, 2]
            weights_voids = voids[:, 3]
            (
                rebin_weight,
                rebin_xi,
                rebin_r,
                rebin_mu,
                rebin_z,
                rebin_num_pairs,
            ) = compute_xi_forest_pairs(
                xcf,
                x_del,
                y_del,
                z_del,
                weights_deltas,
                delta_values,
                x_voids,
                y_voids,
                z_voids,
                weights_voids,
            )
            xi[: len(rebin_xi)] += rebin_xi
            weights[: len(rebin_weight)] += rebin_weight
            r[: len(rebin_r)] += rebin_r
            mu[: len(rebin_mu)] += rebin_mu
            z[: len(rebin_z)] += rebin_z
            num_pairs[: len(rebin_num_pairs)] += rebin_num_pairs.astype(int)

    w = weights > 0
    xi[w] /= weights[w]
    r[w] /= weights[w]
    mu[w] /= weights[w]
    z[w] /= weights[w]

    r = r.reshape(nr, nmu)
    mu = mu.reshape(nr, nmu)
    xi = xi.reshape(nr, nmu)
    z = z.reshape(nr, nmu)

    if save_corr is not None:
        xcorr = xcorr_objects.CrossCorr(
            name=save_corr,
            mu_array=mu,
            r_array=r,
            xi_array=xi,
            z_array=z,
            exported=True,
        )
        xcorr.write(xcf_param=xcf)

    return (r, mu, xi, z)


def xcorr_cartesian_rprt(xcf, save_corr=None):
    """Compute the (rp, rt) cross-correlation xi(rp, rt) between deltas and voids.

    For every delta with neighboring voids, accumulates rebinned pair
    weights, correlation contributions, parallel/transverse
    separations, redshifts, and pair counts (via
    `compute_xi_forest_pairs_rprt`) into flat `nbins_rp * nbins_rt`
    arrays, normalizes by the total weight per bin, and reshapes the
    result to `(nbins_rp, nbins_rt)`.

    Args:
        xcf (dict): Cross-correlation parameter dictionary; must
            contain `"nbins_rp"`, `"nbins_rt"`, `"rp_min"`, `"rp_max"`,
            `"rt_max"`, and `"deltas"` (each delta dict expected to
            carry a `"neighbors"` array).
        save_corr (str, optional): If provided, name used to build and
            write out a `lyavoid.xcorr_objects.CrossCorr` object (with
            `rmu=False`). Defaults to None (no export).

    Returns:
        tuple: `(rp, rt, xi, z)`, each a 2D `numpy.ndarray` of shape
        `(nbins_rp, nbins_rt)` giving the weighted-mean parallel
        separation, transverse separation, correlation value, and
        redshift per bin.
    """
    nrp = xcf["nbins_rp"]
    nrt = xcf["nbins_rt"]
    xi = np.zeros(nrp * nrt)
    weights = np.zeros(nrp * nrt)
    rp = np.zeros(nrp * nrt)
    rt = np.zeros(nrp * nrt)
    z = np.zeros(nrp * nrt)
    num_pairs = np.zeros(nrp * nrt, dtype=np.int64)
    deltas = xcf["deltas"]
    for d in deltas:
        voids = d["neighbors"]
        if len(voids) != 0:
            delta_values = d["delta"]
            x_del, y_del, z_del = d["x"], d["y"], d["z"]
            weights_deltas = d["weights"]
            x_voids = voids[:, 0]
            y_voids = voids[:, 1]
            z_voids = voids[:, 2]
            weights_voids = voids[:, 3]
            (
                rebin_weight,
                rebin_xi,
                rebin_rp,
                rebin_rt,
                rebin_z,
                rebin_num_pairs,
            ) = compute_xi_forest_pairs_rprt(
                xcf,
                x_del,
                y_del,
                z_del,
                weights_deltas,
                delta_values,
                x_voids,
                y_voids,
                z_voids,
                weights_voids,
            )
            xi[: len(rebin_xi)] += rebin_xi
            weights[: len(rebin_weight)] += rebin_weight
            rp[: len(rebin_rt)] += rebin_rt
            rt[: len(rebin_rp)] += rebin_rp
            z[: len(rebin_z)] += rebin_z
            num_pairs[: len(rebin_num_pairs)] += rebin_num_pairs.astype(int)

    w = weights > 0
    xi[w] /= weights[w]
    rp[w] /= weights[w]
    rt[w] /= weights[w]
    z[w] /= weights[w]

    rp = rp.reshape(nrp, nrt)
    rt = rt.reshape(nrp, nrt)
    xi = xi.reshape(nrp, nrt)
    z = z.reshape(nrp, nrt)

    if save_corr is not None:
        xcorr = xcorr_objects.CrossCorr(
            name=save_corr,
            mu_array=rt,
            r_array=rp,
            xi_array=xi,
            z_array=z,
            exported=True,
            rmu=False,
        )
        xcorr.write()

    return (rp, rt, xi, z)


# def xcorr_cartesian_error(xcf,nb_bins=5,save_corr=None):
#     (r,mu,xi,z) = xcorr_cartesian(xcf,save_corr=save_corr)
#     deltas = xcf["deltas"]
#     sub_deltas = int(round(len(deltas)/(nb_bins-1),0))
#
#     xcorr = xcorr_objects.CrossCorr.init_from_fits(file_xi_no_export,exported=False,supress_first_pixels=supress_first_pixels)
#     mu,da =  xcorr.mu_array,xcorr.xi_array
#     monopole,dipole,quadrupole,hexadecapole = [],[],[],[]
#     for i in range(len(da)):
#         (mono,di,quad,hexa) = get_poles(mu,da[i])
#         monopole.append(mono)
#         dipole.append(di)
#         quadrupole.append(quad)
#         hexadecapole.append(hexa)
#
#     xcorr = xcorr_objects.CrossCorr(name=save_corr,mu_array=mu,r_array=r,xi_array=xi,z_array=z,exported=False)
#     xcorr.write(xcf,weights)
#     error_monopole = sem(np.array(monopole))
#     error_dipole = sem(np.array(dipole))
#     error_quadrupole = sem(np.array(quadrupole))
#     error_hexadecapole = sem(np.array(hexadecapole))


def xcorr(
    delta_path,
    void_catalog,
    xcorr_options,
    corr_name,
    error_calculation=None,
    rtrp=False,
):
    """End-to-end pipeline to compute and export a void-delta cross-correlation.

    Builds the cross-correlation data structure via `create_xcf`, then
    dispatches to `xcorr_cartesian_rprt` (if `rtrp` is True) or
    `xcorr_cartesian` (otherwise) to compute and write the correlation.

    Args:
        delta_path (str): Directory containing picca-style
            `delta-*.fits` files.
        void_catalog (str): Path to the void catalog FITS file.
        xcorr_options (dict): Cross-correlation parameter dictionary
            (see `create_xcf`, `xcorr_cartesian`,
            `xcorr_cartesian_rprt`).
        corr_name (str): Name/path used when writing out the resulting
            `lyavoid.xcorr_objects.CrossCorr` object.
        error_calculation (optional): If not None and `rtrp` is False,
            short-circuits and returns an empty tuple instead of
            computing the correlation (error-estimation path is not
            implemented here). Defaults to None.
        rtrp (bool, optional): If True, compute the correlation in
            (r_parallel, r_transverse) binning; otherwise use (r, mu)
            binning. Defaults to False.

    Returns:
        tuple or None: An empty tuple `()` if `error_calculation` is
        not None and `rtrp` is False; otherwise None (the underlying
        `xcorr_cartesian*` call's return value is discarded).
    """
    xcf = create_xcf(delta_path, void_catalog, xcorr_options, rtrp=rtrp)
    if rtrp:
        xcorr_cartesian_rprt(xcf, save_corr=corr_name)
    else:
        if error_calculation is not None:
            return ()
        else:
            xcorr_cartesian(xcf, save_corr=corr_name)

    ###
