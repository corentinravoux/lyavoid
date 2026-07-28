"""Linear models of the void x Lyman-alpha cross-correlation and their fitting.

Provides simple redshift-space-distortion (linear bias / Kaiser-like) models of
the cross-correlation ``xi(r, mu)`` built from an input correlation, plus a
:class:`Fitter` that fits the model multipoles to a measured cross-correlation
with iminuit.
"""

import numpy as np
from iminuit import Minuit
from scipy import integrate

from lyavoid import multipole, xcorr_objects


class CrossCorrModel(object):
    """A named cross-correlation model built from an input measurement."""

    def __init__(self, model_name=None, xcorr_in=None, model=None):
        """Store the model name, the input cross-correlation and the model callable.

        Args:
            model_name (str, optional): Model identifier (e.g. ``"linear"``).
            xcorr_in (CrossCorr, optional): Input cross-correlation.
            model (callable, optional): The parameterised model function.
        """
        self.model_name = model_name
        self.xcorr_in = xcorr_in
        self.model = model

    @classmethod
    def init_by_xcorr(
        cls,
        model_name,
        xcorr_in_name,
        xcorr_in_type,
        nbins_mu=None,
        xcorr_in_class_name=None,
    ):
        """Build a model from an input cross-correlation or multipole file.

        Args:
            model_name (str): Model identifier.
            xcorr_in_name (str): Input FITS file.
            xcorr_in_type (str): ``"multipole"`` (rebuilt from the monopole) or
                ``"xcorr"`` (read directly).
            nbins_mu (int, optional): Number of mu bins when rebuilding from a
                multipole.
            xcorr_in_class_name (str, optional): Output name for the rebuilt
                cross-correlation.

        Returns:
            CrossCorrModel: The initialised model.
        """
        if xcorr_in_type == "multipole":
            multipole_object = xcorr_objects.Multipole.init_from_fits(xcorr_in_name)
            xcorr_in = multipole_object.xcorr_from_monopole(
                xcorr_in_class_name, nbins_mu
            )
        elif xcorr_in_type == "xcorr":
            xcorr_in = xcorr_objects.CrossCorr.init_from_fits(
                xcorr_in_name, supress_first_pixels=0
            )

        r, mu = xcorr_in.r_array, xcorr_in.mu_array
        xi_array = xcorr_in.xi_array
        xcorr_class = cls(model_name=model_name, xcorr_in=xcorr_in)
        xcorr_class.model = xcorr_class.return_model(r, mu, xi_array)
        return xcorr_class

    def return_model(self, r, mu, xi_array):
        """Return the parameterised model function for the configured model name.

        Args:
            r (numpy.ndarray): Separation grid (h^-1 Mpc).
            mu (numpy.ndarray): ``mu = r_parallel / r`` grid.
            xi_array (numpy.ndarray): Input correlation to scale.

        Returns:
            callable: A model function mapping bias/RSD parameters to a
            :class:`CrossCorr`.
        """
        if self.model_name == "linear":
            xcorr_model = SimpleLinearModel()
        elif self.model_name == "linear2":
            xcorr_model = SimpleLinearModel2()
        elif self.model_name == "linear3":
            xcorr_model = SimpleLinearModel3()
        return xcorr_model.model(r, mu, xi_array)

    def write_model(self, output_name, parameters):
        """Evaluate the model at ``parameters`` and write it to a FITS file.

        Args:
            output_name (str): Output cross-correlation FITS file.
            parameters (dict): Model parameter values.
        """
        xcorr_out = self.model(**parameters)
        xcorr_out.name = output_name
        xcorr_out.z_array = np.zeros(xcorr_out.xi_array.shape)
        xcorr_out.write()


class SimpleLinearModel(object):
    """Linear model with separate void and absorber bias / RSD parameters."""

    def __init__(self):
        """Set the model name."""
        self.model_name = "SimpleLinearModel"

    def model(self, r, mu, xi_array):
        """Return a model function of ``(b_v, b_a, beta_v, beta_a)``.

        Args:
            r (numpy.ndarray): Separation grid (h^-1 Mpc).
            mu (numpy.ndarray): ``mu`` grid.
            xi_array (numpy.ndarray): Input correlation.

        Returns:
            callable: ``model_function(b_v, b_a, beta_v, beta_a) -> CrossCorr``.
        """
        def model_function(b_v=1, b_a=1, beta_v=0, beta_a=0):
            xi = b_v * b_a * (1 + beta_v * mu**2) * (1 + beta_a * mu**2) * xi_array
            xcorr = xcorr_objects.CrossCorr(mu_array=mu, r_array=r, xi_array=xi)
            return xcorr

        return model_function


class SimpleLinearModel2(object):
    """Linear Kaiser model with a single bias ``b`` and RSD ``beta``."""

    def __init__(self):
        """Set the model name."""
        self.model_name = "SimpleLinearModel2"

    def model(self, r, mu, xi_array):
        """Return a model function of a ``{"b", "beta"}`` parameter dict.

        Args:
            r (numpy.ndarray): Separation grid (h^-1 Mpc).
            mu (numpy.ndarray): ``mu`` grid.
            xi_array (numpy.ndarray): Input correlation.

        Returns:
            callable: ``model_function(parameters) -> CrossCorr``.
        """
        def model_function(parameters):
            b = parameters["b"]
            beta = parameters["beta"]
            xi = b**2 * (1 + beta * mu**2) ** 2 * xi_array
            xcorr = xcorr_objects.CrossCorr(mu_array=mu, r_array=r, xi_array=xi)
            return xcorr

        return model_function


class SimpleLinearModel3(object):
    """Linear void model using the integrated correlation (Xi_bar) term."""

    def __init__(self):
        """Set the model name."""
        self.model_name = "SimpleLinearModel3"

    def model(self, r, mu, xi_array):
        """Return a model function using the volume-averaged correlation.

        Precomputes ``Xi_bar(r)``, the correlation averaged within ``r``, then
        returns a function of a ``{"beta"}`` parameter dict implementing the
        linear-theory void RSD relation.

        Args:
            r (numpy.ndarray): Separation grid (h^-1 Mpc).
            mu (numpy.ndarray): ``mu`` grid.
            xi_array (numpy.ndarray): Input correlation.

        Returns:
            callable: ``model_function(parameters) -> CrossCorr``.
        """
        Xi_bar = np.zeros(xi_array.shape)
        for i in range(len(r)):
            mask = (r <= r[i]) & (r > 0)
            integrand = xi_array[mask] * r[mask] ** 2
            integral = integrate.simpson(integrand, r[mask], axis=0)
            Xi_bar[i, :] = integral * 3 / (r[i] ** 3)

        def model_function(parameters):
            beta = parameters["beta"]
            xi = xi_array + (beta / 3) * Xi_bar + beta * mu**2 * (xi_array - Xi_bar)
            xcorr = xcorr_objects.CrossCorr(mu_array=mu, r_array=r, xi_array=xi)
            return xcorr

        return model_function


class Fitter(object):
    """Fit a cross-correlation model to measured multipoles with iminuit."""

    def __init__(self, pwd, minuit_parameters):
        """Store the working directory and the initial Minuit parameters.

        Args:
            pwd (str): Output directory.
            minuit_parameters (dict): Initial parameter values for Minuit.
        """
        self.pwd = pwd
        self.minuit_parameters = minuit_parameters

    # def custom_least_squares_multipole(
    #     self, model, data_x, data_y, data_yerr, multipole_method
    # ):
    #     def cost(b_v=1, b_a=1, beta_v=0, beta_a=0):
    #         xcorr_model = model(b_v=b_v, b_a=b_a, beta_v=beta_v, beta_a=beta_a)
    #         (monopole, _, quadrupole, _) = multipole.get_poles(
    #             xcorr_model.mu_array, xcorr_model.xi_array, multipole_method
    #         )
    #         r_array_model = np.nanmean(xcorr_model.r_array, axis=1)
    #         monopole = np.interp(data_x, r_array_model, monopole)
    #         quadrupole = np.interp(data_x, r_array_model, quadrupole)
    #         ym = np.array([monopole, quadrupole])
    #         z = (data_y - ym) / data_yerr
    #         return np.nansum(z**2)

    #     return cost

    def custom_least_squares_multipole(
        self, model, data_x, data_y, data_yerr, multipole_method
    ):
        """Build a Minuit cost function comparing model and data multipoles.

        Args:
            model (callable): Model function returning a :class:`CrossCorr`.
            data_x (numpy.ndarray): Separation grid of the data (h^-1 Mpc).
            data_y (numpy.ndarray): Stacked measured ``[monopole, quadrupole]``.
            data_yerr (numpy.ndarray): Errors on ``data_y``.
            multipole_method (str): Integration method for the model poles.

        Returns:
            callable: A ``cost(beta)`` returning the chi-squared.
        """
        def cost(beta=0.5):
            xcorr_model = model({"beta": beta})
            (monopole, _, quadrupole, _) = multipole.get_poles(
                xcorr_model.mu_array, xcorr_model.xi_array, multipole_method
            )
            r_array_model = np.nanmean(xcorr_model.r_array, axis=1)
            monopole = np.interp(data_x, r_array_model, monopole)
            quadrupole = np.interp(data_x, r_array_model, quadrupole)
            ym = np.array([monopole, quadrupole])
            z = (data_y - ym) / data_yerr
            return np.nansum(z**2)

        return cost

    def run_migrad(self, minuit, ncall=1000):
        """Run Minuit's MIGRAD minimiser.

        Args:
            minuit (iminuit.Minuit): The Minuit instance.
            ncall (int, optional): Maximum number of calls.

        Returns:
            The MIGRAD result.
        """
        return minuit.migrad(ncall)

    def run_minos(self, minuit, sigma, ncall=1000, var_minos=None):
        """Run Minuit's MINOS error analysis.

        Args:
            minuit (iminuit.Minuit): The Minuit instance.
            sigma (float): Confidence level in sigma.
            ncall (int, optional): Maximum number of calls.
            var_minos (optional): Variable(s) to run MINOS on.

        Returns:
            The MINOS result.
        """
        return minuit.minos(var=var_minos, sigma=sigma, ncall=ncall)

    def run_hesse(self, minuit):
        """Run Minuit's HESSE error estimation.

        Args:
            minuit (iminuit.Minuit): The Minuit instance.

        Returns:
            The HESSE result.
        """
        return minuit.hesse()

    def fit_multipole(
        self,
        xcorr_model,
        xcorr_fit_name,
        xcorr_error_name,
        output_name,
        supress_first_pixels=0,
        multipole_method="simps",
        fix_args=None,
    ):
        """Fit the model monopole+quadrupole to a measured cross-correlation.

        Reads the measured cross-correlation and its error realisations, forms
        the monopole and quadrupole, minimises the chi-squared with Minuit and
        writes the best-fit model.

        Args:
            xcorr_model (CrossCorrModel): Model to fit.
            xcorr_fit_name (str): Measured cross-correlation FITS file.
            xcorr_error_name (str): File with the error realisations.
            output_name (str): Output best-fit model file.
            supress_first_pixels (int, optional): Number of first r bins to drop.
            multipole_method (str, optional): Multipole integration method.
            fix_args (list[str], optional): Parameter names to fix in the fit.

        Returns:
            iminuit.Minuit: The minimiser after MIGRAD.
        """
        xcorr_fit_class = xcorr_objects.CrossCorr.init_from_fits(
            xcorr_fit_name, supress_first_pixels=supress_first_pixels
        )
        (monopole, _, quadrupole, _) = multipole.get_poles(
            xcorr_fit_class.mu_array, xcorr_fit_class.xi_array, multipole_method
        )

        (
            error_monopole,
            _,
            error_quadrupole,
            _,
        ) = multipole.get_error_bars_from_no_export(
            xcorr_error_name,
            multipole_method,
            supress_first_pixels=supress_first_pixels,
        )
        data_x = np.nanmean(xcorr_fit_class.r_array, axis=1)
        data_y = np.array([monopole, quadrupole])
        data_yerr = np.array([error_monopole, error_quadrupole])
        cost_function = self.custom_least_squares_multipole(
            xcorr_model.model, data_x, data_y, data_yerr, multipole_method
        )

        minuit = Minuit(cost_function, **self.minuit_parameters)
        if fix_args is not None:
            for i in range(len(fix_args)):
                minuit.fixed[fix_args[i]] = True

        print(self.run_migrad(minuit))
        # self.run_hesse(minuit)
        # run_minos(minuit,sigma,ncall=ncall,var_minos=var_minos)
        xcorr_model.write_model(output_name, dict(minuit.values))
        return minuit
