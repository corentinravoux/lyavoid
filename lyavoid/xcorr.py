import glob
import os

import fitsio
import numpy as np

from lelantos import tomographic_objects
from lyavoid import xcorr_objects

lambdaLy = 1215.673123130217
import subprocess


def run_cross_corr_picca(dict_picca, rmu=True):
    cmd = " picca_xcf.py"
    for key, value in dict_picca.items():
        cmd += f" --{key} {value}"
    if rmu:
        cmd += " --rmu"
    subprocess.call(cmd, shell=True)


def run_export_picca(input, output, smooth=False):
    cmd = " picca_export.py"
    cmd += f" --data {input}"
    cmd += f" --out {output}"
    if not smooth:
        cmd += " --do-not-smooth-cov"
    subprocess.call(cmd, shell=True)


def create_xcf(delta_path, void_catalog, xcorr_options, rtrp=False):
    xcf = xcorr_options
    read_voids(xcf, void_catalog)
    read_deltas(xcf, delta_path)
    if rtrp:
        fill_neighs_rtrp(xcf)
    else:
        fill_neighs(xcf)
    return xcf


def read_voids(xcf, void_catalog):
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
    xcf = create_xcf(delta_path, void_catalog, xcorr_options, rtrp=rtrp)
    if rtrp:
        xcorr_cartesian_rprt(xcf, save_corr=corr_name)
    else:
        if error_calculation is not None:
            return ()
        else:
            xcorr_cartesian(xcf, save_corr=corr_name)

    ###
