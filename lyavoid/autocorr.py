"""Lyman-alpha forest auto-correlation in cartesian ``(r_par, r_trans)`` bins.

Reads picca delta line-of-sight files, pairs them up (optionally within a
transverse-neighbour radius) and accumulates the weighted delta x delta
auto-correlation with a numba-accelerated pair loop, returning a
:class:`lyavoid.xcorr_objects.CrossCorr`. Separations are in h^-1 Mpc.
"""

import glob
import os

import fitsio
import numpy as np
from numba import int32, njit

from lyavoid import xcorr_objects

lambdaLy = 1215.673123130217


def corr(
    delta_path,
    corr_options,
    corr_name,
):
    """Compute the forest auto-correlation and write it to a FITS file.

    Args:
        delta_path (str): Directory of picca ``delta-*.fits`` files.
        corr_options (dict): Correlation configuration (binning, redshift
            evolution, ...). See :func:`create_cf`.
        corr_name (str): Output cross-correlation FITS file.
    """
    cf = create_cf(delta_path, corr_options)
    corr_cartesian_rprt(cf, save_corr=corr_name)


def create_cf(delta_path, corr_options):
    """Load the deltas and their neighbours into the correlation config.

    Args:
        delta_path (str): Directory of picca ``delta-*.fits`` files.
        corr_options (dict): Correlation configuration, updated in place.

    Returns:
        dict: The configuration with ``deltas`` and neighbour lists filled.
    """
    cf = corr_options
    read_deltas(cf, delta_path)
    fill_neighs(cf)
    return cf


def read_deltas(cf, delta_path):
    """Read the delta line-of-sight files into the config dict.

    Populates ``cf["deltas"]`` with per-LOS dicts (``x``, ``y``, ``z``,
    ``delta``, redshift-evolved ``weights``).

    Args:
        cf (dict): Correlation config (needs ``z_ref`` and ``z_evol_del``).
        delta_path (str): Directory of picca ``delta-*.fits`` files.
    """
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
                ((1 + redshift) / (1 + cf["z_ref"])) ** (cf["z_evol_del"] - 1)
            )
            deltas.append(delta_dict)
    cf["deltas"] = deltas
    print(
        "..done. Number of deltas:",
        len(deltas),
        "Number of pixels:",
        len(deltas) * len(redshift),
    )


def fill_neighs(cf):
    """Attach to each delta its transverse neighbours within ``rt_max``.

    Args:
        cf (dict): Correlation config (needs ``deltas`` and ``rt_max``); each
            delta gains a ``neighbors`` array.
    """
    print("Filling neighbors..")
    deltas = cf["deltas"]
    rt_max = cf["rt_max"]
    delta_x = np.array([d["x"] for d in deltas])
    delta_y = np.array([d["y"] for d in deltas])
    for d in deltas:
        x, y = d["x"], d["y"]
        mask = np.sqrt((delta_x - x) ** 2 + (delta_y - y) ** 2) <= rt_max
        d["neighbors"] = np.asarray(deltas)[mask]
    print("..done")


def corr_cartesian_rprt(cf, save_corr=None, use_neighbors=False):
    """Accumulate the forest auto-correlation over all delta pairs.

    Loops over delta pairs (all pairs, or only precomputed neighbours), bins the
    weighted product in ``(r_par, r_trans)`` and normalises by the weights.

    Args:
        cf (dict): Correlation config with loaded ``deltas`` and binning
            (``nbins_rp``/``nbins_rt``, ``rp_min``/``rp_max``, ``rt_max``).
        save_corr (str, optional): If set, output FITS file to write.
        use_neighbors (bool, optional): Iterate only over each delta's
            precomputed neighbours.

    Returns:
        tuple: ``(rp, rt, xi, z, corr_obj)`` — the binned grids and the
        resulting :class:`CrossCorr`.
    """
    nrp = cf["nbins_rp"]
    nrt = cf["nbins_rt"]
    xi = np.zeros(nrp * nrt)
    weights = np.zeros(nrp * nrt)
    rp = np.zeros(nrp * nrt)
    rt = np.zeros(nrp * nrt)
    z = np.zeros(nrp * nrt)

    num_pairs = np.zeros(nrp * nrt, dtype=np.int64)
    deltas = cf["deltas"]
    for _, delta1 in enumerate(deltas):
        if use_neighbors:
            delta_to_iterate = delta1["neighbors"]
        else:
            delta_to_iterate = deltas
        for _, delta2 in enumerate(delta_to_iterate):
            if delta1["index"] != delta2["index"]:
                x1 = delta1["x"]
                y1 = delta1["y"]
                z1 = delta1["z"]
                weights1 = delta1["weights"]
                delta_array1 = delta1["delta"]

                x2 = delta2["x"]
                y2 = delta2["y"]
                z2 = delta2["z"]
                weights2 = delta2["weights"]
                delta_array2 = delta2["delta"]

                (
                    weights,
                    xi,
                    rp,
                    rt,
                    z,
                    num_pairs,
                ) = compute_xi_forest_pairs_fast(
                    x1,
                    y1,
                    z1,
                    x2,
                    y2,
                    z2,
                    weights1,
                    delta_array1,
                    weights2,
                    delta_array2,
                    weights,
                    xi,
                    rp,
                    rt,
                    z,
                    num_pairs,
                    cf["rp_min"],
                    cf["rp_max"],
                    cf["rt_max"],
                    cf["nbins_rp"],
                    cf["nbins_rt"],
                )

    w = weights > 0
    xi[w] /= weights[w]
    rp[w] /= weights[w]
    rt[w] /= weights[w]
    z[w] /= weights[w]

    rp = rp.reshape(nrp, nrt)
    rt = rt.reshape(nrp, nrt)
    xi = xi.reshape(nrp, nrt)
    z = z.reshape(nrp, nrt)

    corr_obj = xcorr_objects.CrossCorr(
        mu_array=rt,
        r_array=rp,
        xi_array=xi,
        z_array=z,
        exported=True,
        rmu=False,
    )
    if save_corr is not None:
        corr_obj.name = save_corr
        corr_obj.write()

    return (rp, rt, xi, z, corr_obj)


@njit
def compute_xi_forest_pairs_fast(
    x1,
    y1,
    z1,
    x2,
    y2,
    z2,
    weights1,
    delta1,
    weights2,
    delta2,
    rebin_weight,
    rebin_xi,
    rebin_rp,
    rebin_rt,
    rebin_z,
    rebin_num_pairs,
    rp_min,
    rp_max,
    rt_max,
    nbins_rp,
    nbins_rt,
):
    """Adapted from picca"""
    r_trans = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
    for i in range(z1.size):
        if weights1[i] == 0:
            continue

        for j in range(z2.size):
            if weights2[j] == 0:
                continue

            z = (z1[i] + z2[j]) / 2

            r_par = z2[j] - z1[i]
            r_par = np.abs(r_par)
            if r_par >= rp_max or r_trans >= rt_max or r_par < rp_min:
                continue

            delta_times_weight1 = delta1[i] * weights1[i]
            delta_times_weight2 = delta2[j] * weights2[j]
            delta_times_weight12 = delta_times_weight1 * delta_times_weight2
            weights12 = weights1[i] * weights2[j]
            z = (z1[i] + z2[j]) / 2

            bins_r_par = np.floor((r_par - rp_min) / (rp_max - rp_min) * nbins_rp)
            bins_r_trans = np.floor(r_trans / rt_max * nbins_rt)
            bins = int(bins_r_trans + nbins_rt * bins_r_par)

            rebin_xi[bins] += delta_times_weight12
            rebin_weight[bins] += weights12
            rebin_rp[bins] += r_par * weights12
            rebin_rt[bins] += r_trans * weights12
            rebin_z[bins] += z * weights12
            rebin_num_pairs[bins] += 1
    return (
        rebin_weight,
        rebin_xi,
        rebin_rp,
        rebin_rt,
        rebin_z,
        rebin_num_pairs,
    )
