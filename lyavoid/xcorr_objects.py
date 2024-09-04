import fitsio
import matplotlib.pyplot as plt
import numpy as np
from scipy import integrate
from scipy.interpolate import griddata, interp1d
from scipy.special import legendre
from scipy.stats import sem

from lyavoid import utils


class CrossCorr(object):

    def __init__(
        self,
        name=None,
        mu_array=None,
        r_array=None,
        xi_array=None,
        z_array=None,
        xi_error_array=None,
        exported=True,
        rmu=True,
    ):

        self.name = name
        self.mu_array = mu_array
        self.r_array = r_array
        self.xi_array = xi_array
        self.z_array = z_array
        self.xi_error_array = xi_error_array
        self.exported = exported
        self.rmu = rmu  # If True, r_array contains r and mu_array mu. If False, r_array contains rp (r_parallel) and mu_array rt (r_transverse)

    @classmethod
    def init_from_fits(cls, name, supress_first_pixels=0):
        with fitsio.FITS(name) as h:
            xi_array = h["COR"]["DA"][:]
            xi_error_array = None
            exported = True
            attribut_name = "COR"
            if "WE" in h["COR"].get_colnames():
                exported = False
                attribut_name = "ATTRI"
            if exported:
                if "CO" in h["COR"].get_colnames():
                    xi_error_array = np.sqrt(np.diag(h["COR"]["CO"][:]))
            z_array = h[attribut_name]["Z"][:]
            hh = h[attribut_name].read_header()
            if "R" in h[attribut_name].get_colnames():
                rmu = True
                r_array = h[attribut_name]["R"][:]
                mu_array = h[attribut_name]["MU"][:]
                nr = hh["NR"]
                nmu = hh["NMU"]
            elif "RT" in h[attribut_name].get_colnames():
                rmu = False
                mu_array = h[attribut_name]["RP"][:]
                r_array = h[attribut_name]["RT"][:]
                nr = hh["NP"]
                nmu = hh["NT"]
        if exported:
            xi_array = xi_array.reshape(nr, nmu)[supress_first_pixels:, :]
            if xi_error_array is not None:
                xi_error_array = xi_error_array.reshape(nr, nmu)[
                    supress_first_pixels:, :
                ]
        else:
            xi_array = xi_array.reshape(len(xi_array), nr, nmu)[
                :, supress_first_pixels:, :
            ]
            xi_error_array = sem(xi_array, axis=0)
        r_array = r_array.reshape(nr, nmu)[supress_first_pixels:, :]
        mu_array = mu_array.reshape(nr, nmu)[supress_first_pixels:, :]
        z_array = z_array.reshape(nr, nmu)[supress_first_pixels:, :]
        return cls(
            name=name,
            mu_array=mu_array,
            r_array=r_array,
            xi_array=xi_array,
            z_array=z_array,
            exported=exported,
            xi_error_array=xi_error_array,
            rmu=rmu,
        )

    def write(self, xcf_param=None):

        out = fitsio.FITS(self.name, "rw", clobber=True)
        if self.exported:
            nbins_r = self.xi_array.shape[0]
            nbins_mu = self.xi_array.shape[1]
        else:
            nbins_r = self.xi_array.shape[1]
            nbins_mu = self.xi_array.shape[2]

        if self.rmu:
            head = [
                {"name": "NR", "value": nbins_r, "comment": "Number of bins in r"},
                {"name": "NMU", "value": nbins_mu, "comment": "Number of bins in mu"},
            ]
            if xcf_param is not None:
                head = head + [
                    {
                        "name": "RMIN",
                        "value": xcf_param["r_min"],
                        "comment": "Minimum r [h^-1 Mpc]",
                    },
                    {
                        "name": "RMAX",
                        "value": xcf_param["r_max"],
                        "comment": "Maximum r [h^-1 Mpc]",
                    },
                    {
                        "name": "MUMIN",
                        "value": xcf_param["mu_min"],
                        "comment": "Minimum mu = r_para/r",
                    },
                    {
                        "name": "MUMAX",
                        "value": xcf_param["mu_max"],
                        "comment": "Maximum mu = r_para/r",
                    },
                ]
            name_out1 = "R"
            name_out2 = "MU"
            comment = ["r", "mu = r_para/r", "xi", "redshift"]
            unit = ["h^-1 Mpc", "", "", ""]

        else:
            head = [
                {"name": "NP", "value": nbins_r, "comment": "Number of bins in r"},
                {"name": "NT", "value": nbins_mu, "comment": "Number of bins in mu"},
            ]
            if xcf_param is not None:
                head = head + [
                    {
                        "name": "RPMIN",
                        "value": xcf_param["rp_min"],
                        "comment": "Minimum rp [h^-1 Mpc]",
                    },
                    {
                        "name": "RPMAX",
                        "value": xcf_param["rp_max"],
                        "comment": "Maximum rp [h^-1 Mpc]",
                    },
                    {
                        "name": "RTMAX",
                        "value": xcf_param["rt_max"],
                        "comment": "Maximum rt [h^-1 Mpc]",
                    },
                ]
            name_out1 = "RT"
            name_out2 = "RP"
            comment = ["rp", "rt", "xi", "redshift"]
            unit = ["h^-1 Mpc", "h^-1 Mpc", "", ""]

        out.write(
            [self.r_array, self.mu_array, self.xi_array, self.z_array],
            names=[name_out1, name_out2, "DA", "Z"],
            comment=comment,
            units=unit,
            header=head,
            extname="COR",
        )
        out.close()

    def write_no_export(self, xcf_param=None, weights=None):

        out = fitsio.FITS(self.name, "rw", clobber=True)

        head = [
            {
                "name": "RMIN",
                "value": xcf_param["r_min"],
                "comment": "Minimum r [h^-1 Mpc]",
            },
            {
                "name": "RMAX",
                "value": xcf_param["r_max"],
                "comment": "Maximum r [h^-1 Mpc]",
            },
            {
                "name": "MUMIN",
                "value": xcf_param["mu_min"],
                "comment": "Minimum mu = r_para/r",
            },
            {
                "name": "MUMAX",
                "value": xcf_param["mu_max"],
                "comment": "Maximum mu = r_para/r",
            },
            {
                "name": "NR",
                "value": xcf_param["nbins_r"],
                "comment": "Number of bins in r",
            },
            {
                "name": "NMU",
                "value": xcf_param["nbins_mu"],
                "comment": "Number of bins in mu",
            },
        ]
        out.write(
            [self.r_array, self.mu_array, self.z_array],
            names=["R", "MU", "Z"],
            comment=["r", "mu = r_para/r", "redshift"],
            units=["h^-1 Mpc", "", ""],
            header=head,
            extname="ATTRI",
        )

        head2 = [{"name": "HLPXSCHM", "value": "RING", "comment": "Healpix scheme"}]
        out.write(
            [weights, self.xi_array],
            names=["WE", "DA"],
            comment=["Sum of weight", "Correlation"],
            header=head2,
            extname="COR",
        )
        out.close()

    def plot_2d(self, **kwargs):
        rmu = utils.return_key(kwargs, "rmu", True)
        style = utils.return_key(kwargs, "style", None)
        if style is not None:
            plt.style.use(style)
        if ((rmu == False) & self.rmu) | ((self.rmu == False) & rmu):
            self.switch()
        vmax = utils.return_key(kwargs, "vmax", None)
        vmin = utils.return_key(kwargs, "vmin", None)
        colormap = utils.return_key(kwargs, "colormap", "bwr")
        multiplicative_factor = utils.return_key(kwargs, "multiplicative_factor", None)
        if multiplicative_factor is not None:
            self.xi_array = self.xi_array * multiplicative_factor
        radius_multiplication_power = utils.return_key(kwargs, "r_power", 0)
        title_add = ""
        if radius_multiplication_power != 0:
            if radius_multiplication_power == 1:
                title_add = r"$\times r$"
            else:
                title_add = r"$\times r^{" + str(radius_multiplication_power) + "}$"
        colobar_legend = (
            utils.return_key(
                kwargs, "cbar", r"Ly$\alpha\times$voids cross-correlation $\xi$"
            )
            + title_add
        )
        name = utils.return_key(kwargs, "name", "2d_plot_cross_corr")
        if rmu == False:
            rp_array = self.mu_array
            rt_array = self.r_array
            xi_to_plot = self.xi_array * (
                np.sqrt(rp_array**2 + rt_array**2) ** radius_multiplication_power
            )
            xlabel = r"$r_{\bot}$"
            ylabel = r"$r_{\parallel}$"
            self.plot_2d_regrid(
                name,
                rt_array,
                rp_array,
                xi_to_plot,
                xlabel,
                ylabel,
                vmin,
                vmax,
                colobar_legend,
                colormap,
            )
            self.plot_2d_original_binning(
                name,
                rt_array,
                rp_array,
                xi_to_plot,
                xlabel,
                ylabel,
                vmin,
                vmax,
                colobar_legend,
                colormap,
            )
        else:
            xi_to_plot = self.xi_array * (self.r_array**radius_multiplication_power)
            xlabel = r"$\mu$"
            ylabel = r"$r$"
            self.plot_2d_regrid(
                name,
                self.mu_array,
                self.r_array,
                xi_to_plot,
                xlabel,
                ylabel,
                vmin,
                vmax,
                colobar_legend,
                colormap,
            )
            self.plot_2d_original_binning(
                name,
                self.mu_array,
                self.r_array,
                xi_to_plot,
                xlabel,
                ylabel,
                vmin,
                vmax,
                colobar_legend,
                colormap,
            )

        if ((rmu == False) & self.rmu) | ((self.rmu == False) & rmu):
            self.switch()

    def plot_2d_regrid(
        self,
        name,
        x_array,
        y_array,
        xi_array,
        xlabel,
        ylabel,
        vmin,
        vmax,
        colobar_legend,
        colormap,
    ):
        plt.figure()
        extent = (np.min(x_array), np.max(x_array), np.min(y_array), np.max(y_array))
        xx, yy = np.meshgrid(
            np.linspace(extent[0], extent[1], x_array.shape[1]),
            np.linspace(extent[2], extent[3], y_array.shape[0]),
        )
        grid = griddata(
            np.array([np.ravel(x_array), np.ravel(y_array)]).T,
            np.ravel(xi_array),
            (xx, yy),
            method="nearest",
        )
        plt.imshow(grid, extent=extent, vmin=vmin, vmax=vmax, cmap=colormap)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        cbar = plt.colorbar()
        cbar.set_label(colobar_legend)
        plt.tight_layout()
        plt.savefig(f"{name}_regrid.pdf", format="pdf")

    def plot_2d_original_binning(
        self,
        name,
        x_array,
        y_array,
        xi_array,
        xlabel,
        ylabel,
        vmin,
        vmax,
        colobar_legend,
        colormap,
    ):
        plt.figure()
        plt.pcolor(x_array, y_array, xi_array, vmin=vmin, vmax=vmax, cmap=colormap)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        cbar = plt.colorbar()
        cbar.set_label(colobar_legend)
        plt.tight_layout()
        plt.savefig(f"{name}.pdf", format="pdf")

    def switch(self):
        if self.rmu:
            self.mu_array, self.r_array = (
                self.mu_array * self.r_array,
                self.r_array * np.sqrt(1 - self.mu_array**2),
            )
            self.rmu = False
        else:
            self.r_array, self.mu_array = np.sqrt(
                self.mu_array**2 + self.r_array**2
            ), self.mu_array / np.sqrt(self.mu_array**2 + self.r_array**2)
            self.rmu = True


class Multipole(object):

    def __init__(self, name=None, r_array=None, ell=None, poles=None, error_poles=None):

        self.name = name
        self.r_array = r_array
        self.ell = ell
        self.poles = poles
        self.error_poles = error_poles

    @classmethod
    def init_from_fits(cls, name):
        input = fitsio.FITS(name)["MULT"]
        poles = {}
        error_poles = {}
        ells = []
        for col in input.get_colnames():
            if col == "R":
                r_array = input[col][:]
            else:
                ell = int(col[-1])
                if col[:6] == "ERROR_":
                    error_poles[ell] = input[col][:]
                else:
                    poles[ell] = input[col][:]
                ells.append(ell)
        if error_poles == {}:
            error_poles = None
        return cls(
            name=name,
            r_array=r_array,
            ell=np.unique(ells),
            poles=poles,
            error_poles=error_poles,
        )

    @classmethod
    def init_from_xcorr(
        cls, ell, mu, xi, method, r_array=None, name=None, extrapolate=True
    ):
        poles = {}
        for i in range(len(ell)):
            poles[ell[i]] = Multipole.get_multipole_from_array(
                xi, mu, ell[i], method, extrapolate_mu=extrapolate
            )
        return cls(name=name, r_array=r_array, ell=ell, poles=poles)

    @staticmethod
    def get_multipole_from_array(xi, mu, order, method, extrapolate_mu=True):
        if extrapolate_mu:
            xi, mu = Multipole.extrapolate_xi_mu(xi, mu)
        if method == "rect":
            pole_l = Multipole.get_multipole_from_array_rect(xi, mu, order)
        elif method == "nbody":
            pole_l = Multipole.get_multipole_from_array_rect_nbody(xi, mu, order)
        elif method == "trap":
            integrand = (
                xi * (1 + 2 * order) * legendre(order)(mu)
            ) / 2  # divided by two because integration is between -1 and 1
            pole_l = integrate.trapz(integrand, mu, axis=1)
        elif method == "simps":
            integrand = (
                xi * (1 + 2 * order) * legendre(order)(mu)
            ) / 2  # divided by two because integration is between -1 and 1
            pole_l = integrate.simpson(integrand, mu, axis=1)
        elif method.split("_")[0] == "rebin":
            pole_l = Multipole.get_multipole_from_array_rebin_sample(
                xi, mu, order, method.split("_")[1]
            )
        elif (method == "quad") | (method == "romb"):
            pole_l = Multipole.get_multipole_from_array_fct(xi, mu, order, method)
        else:
            raise KeyError(f"{method} integration method is not implemented")
        return pole_l

    @staticmethod
    def extrapolate_xi_mu(xi, mu):
        xi_extrapolate = np.zeros((xi.shape[0], xi.shape[1] + 2))
        mu_extrapolate = np.zeros((mu.shape[0], mu.shape[1] + 2))
        xi_extrapolate[:, 0] = xi[:, 0] + (-1.0 - mu[:, 0]) * (
            (xi[:, 1] - xi[:, 0]) / (mu[:, 1] - mu[:, 0])
        )
        xi_extrapolate[:, -1] = xi[:, -1] + (1.0 - mu[:, -1]) * (
            (xi[:, -1] - xi[:, -2]) / (mu[:, -1] - mu[:, -2])
        )
        xi_extrapolate[:, 1:-1] = xi[:, :]
        mu_extrapolate[:, 0] = -1.0
        mu_extrapolate[:, -1] = 1.0
        mu_extrapolate[:, -1] = 1.0
        mu_extrapolate[:, 1:-1] = mu[:, :]
        return (xi_extrapolate, mu_extrapolate)

    @staticmethod
    def get_multipole_from_array_rebin_sample(xi, mu, order, method):
        pole_l = []
        for i in range(len(xi)):
            mu_min, mu_max = np.min(mu[i]), np.max(mu[i])
            xi_r = interp1d(mu[i], xi[i], kind="linear")
            mu2 = np.linspace(mu_min, mu_max, 200)
            integrand = (xi_r(mu2) * (1 + 2 * order) * legendre(order)(mu2)) / 2
            if method == "trap":
                pole_l.append(integrate.trapz(integrand, mu2))
            elif method == "simps":
                pole_l.append(integrate.simpson(integrand, mu2))
            else:
                raise KeyError(f"rebin_{method} integration method is not implemented")

        return np.array(pole_l)

    @staticmethod
    def get_multipole_from_array_fct(xi, mu, order, method):
        pole_l = []
        error_pole_l = []
        for i in range(len(xi)):
            xi_r = interp1d(mu[i], xi[i], kind="linear")

            def func_integrand(x):
                return (xi_r(x) * (1 + 2 * order) * legendre(order)(x)) / 2

            mu_min, mu_max = np.min(mu[i]), np.max(mu[i])
            if method == "quad":
                int = integrate.quad(func_integrand, mu_min, mu_max)
                pole_l.append(int[0])
                error_pole_l.append(int[1])
            if method == "romb":
                pole_l.append(integrate.romberg(func_integrand, mu_min, mu_max))
        return np.array(pole_l)

    @staticmethod
    def get_multipole_from_array_rect_nbody(xi, mu, order):

        mu_bins = np.diff(mu)
        mu_mid = (mu[:, 1:] + mu[:, :-1]) / 2.0
        xi_mid = (xi[:, 1:] + xi[:, :-1]) / 2.0
        legendrePolynomial = (2.0 * order + 1.0) * legendre(order)(mu_mid)
        pole = np.sum(xi_mid * legendrePolynomial * mu_bins, axis=-1) / 2
        return pole

    @staticmethod
    def get_multipole_from_array_rect(xi, mu, order):
        pole = []
        for i in range(len(xi)):
            dmu = np.zeros(mu[i].shape)
            dmu[1:-1] = (mu[i][2:] - mu[i][0:-2]) / 2
            dmu[0] = mu[i][1] - mu[i][0]
            dmu[-1] = mu[i][-1] - mu[i][-2]
            legendrePolynomial = (2.0 * order + 1.0) * legendre(order)(mu[i])
            pole.append(np.nansum(xi[i] * legendrePolynomial * dmu) / 2)
        return np.array(pole)

    def write_fits(self):
        out = fitsio.FITS(self.name, "rw", clobber=True)
        head = []
        out_array = [self.r_array]
        out_names = ["R"]
        for i in range(len(self.ell)):
            out_names.append(f"POLE_{self.ell[i]}")
            out_array.append(self.poles[self.ell[i]])
            if self.error_poles is not None:
                out_names.append(f"ERROR_POLE_{self.ell[i]}")
                out_array.append(self.error_poles[self.ell[i]])
        out.write(out_array, names=out_names, header=head, extname="MULT")

    def xcorr_from_pole(self, xcorr_name, nbins_mu, ell):
        mu_array = np.linspace(-1.0, 1.0, nbins_mu)
        coord_xcorr = np.moveaxis(
            np.array(np.meshgrid(self.r_array, mu_array, indexing="ij")), 0, -1
        )
        mu_array = coord_xcorr[:, :, 1]
        r_array = coord_xcorr[:, :, 0]
        xi_array = np.array([self.poles[ell] for i in range(nbins_mu)]).transpose()
        xi_array = xi_array * legendre(ell)(mu_array)
        z_array = np.zeros(xi_array.shape)
        xcorr = CrossCorr(
            name=xcorr_name,
            mu_array=mu_array,
            r_array=r_array,
            xi_array=xi_array,
            z_array=z_array,
            exported=True,
            rmu=True,
        )
        xcorr.write()
        return xcorr

    def xcorr_from_monopole(self, xcorr_name, nbins_mu):
        mu_array = np.linspace(-1.0, 1.0, nbins_mu)
        coord_xcorr = np.moveaxis(
            np.array(np.meshgrid(self.r_array, mu_array, indexing="ij")), 0, -1
        )
        xi_array = np.array([self.poles[0] for i in range(nbins_mu)]).transpose()
        z_array = np.zeros(xi_array.shape)
        xcorr = CrossCorr(
            name=xcorr_name,
            mu_array=coord_xcorr[:, :, 1],
            r_array=coord_xcorr[:, :, 0],
            xi_array=xi_array,
            z_array=z_array,
            exported=True,
            rmu=True,
        )
        xcorr.write()
        return xcorr
