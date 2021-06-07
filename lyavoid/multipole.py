import numpy as np
import matplotlib.pyplot as plt
from scipy.special import legendre
from scipy import integrate
from scipy.stats import sem
from lyavoid import xcorr_objects,utils
from scipy.interpolate import interp1d
from iminuit import Minuit






def get_multipole_from_array(xi,mu,order,method):
    if(method=="rect"):
        raise KeyError("Pb with rect method")
        # pole_l = get_multipole_from_array_rect(xi,mu,order)
    if(method=="nbody"):
        pole_l = get_multipole_from_array_rect_nbody(xi,mu,order)
    else:
        poly = legendre(order)
        integrand = (xi*(1+2*order)*poly(mu))/2 # divided by two because integration is between -1 and 1
        if(method=="trap"):
            pole_l = integrate.trapz(integrand,mu,axis=1)
        elif(method=="simps"):
            pole_l = integrate.simps(integrand,mu,axis=1)
    return(pole_l)


    # CR - investigate non-null non-zero multipoles for DM box

    # # new data array
    # x = str(self.dims[0])
    # dtype = numpy.dtype([(x, 'f8')] + [('corr_%d' %ell, 'f8') for ell in poles])
    # data = numpy.zeros((self.shape[0]), dtype=dtype)
    # dims = [x]
    # edges = [self.edges[x]]
    #
    #     # FIXME: use something fancier than the central point.
    # mu_bins = numpy.diff(self.edges['mu'])
    # mu_mid = (self.edges['mu'][1:] + self.edges['mu'][:-1])/2.
    #
    # for ell in poles:
    #     legendrePolynomial = (2.*ell+1.)*legendre(ell)(mu_mid)
    #     data['corr_%d' %ell] = numpy.sum(self['corr']*legendrePolynomial*mu_bins,axis=-1)/numpy.sum(mu_bins)
    #
    # data[x] = numpy.mean(self[x],axis=-1)
    # return BinnedStatistic(dims=dims, edges=edges, data=data, poles=poles)


# def get_multipole_from_array_rect_julian(xi,mu,order):
#     mu1d = np.unique(mu)
#     dmu = np.gradient(mu1d)[0]
#     legendrePolynomial = (2.*order+1.)*legendre(order)(mu)
#     pole = np.sum(xi*legendrePolynomial*dmu, axis=1)/2
#     return pole


def get_multipole_from_array_rect_nbody(xi,mu,order):

    mu_bins= np.diff(mu)
    mu_mid = (mu[:,1:] + mu[:,:-1])/2.
    xi_mid = (xi[:,1:] + xi[:,:-1])/2.
    legendrePolynomial = (2.*order+1.)*legendre(order)(mu_mid)
    pole = np.sum(xi_mid*legendrePolynomial*mu_bins,axis=-1)/2
    return pole

def get_multipole_from_array_rect(xi,mu,order):
    dmu = np.zeros(mu.shape)
    dmu[:,1:-1] = (mu[:,2:] - mu[:,0:-2])/2
    dmu[:,0] = mu[:,1]-mu[:,0]
    dmu[:,-1] = mu[:,-1]-mu[:,-2]
    legendrePolynomial = (2.*order+1.)*legendre(order)(mu)
    pole = np.nansum(xi*legendrePolynomial*dmu, axis=1)/2
    return(pole)


def get_poles(mu,da,method):
    monopole = get_multipole_from_array(da,mu,0,method)
    dipole = get_multipole_from_array(da,mu,1,method)
    quadrupole = get_multipole_from_array(da,mu,2,method)
    hexadecapole = get_multipole_from_array(da,mu,4,method)
    return(monopole,dipole,quadrupole,hexadecapole)




def get_error_bars_from_no_export(file_xi_no_export,multipole_method,supress_first_pixels=0):
    xcorr = xcorr_objects.CrossCorr.init_from_fits(file_xi_no_export,supress_first_pixels=supress_first_pixels)
    mu,da =  xcorr.mu_array,xcorr.xi_array
    monopole,dipole,quadrupole,hexadecapole = [],[],[],[]
    for i in range(len(da)):
        (mono,di,quad,hexa) = get_poles(mu,da[i],multipole_method)
        monopole.append(mono)
        dipole.append(di)
        quadrupole.append(quad)
        hexadecapole.append(hexa)
    error_monopole = sem(np.array(monopole))
    error_dipole = sem(np.array(dipole))
    error_quadrupole = sem(np.array(quadrupole))
    error_hexadecapole = sem(np.array(hexadecapole))
    return(error_monopole,error_dipole,error_quadrupole,error_hexadecapole)



def plot_multipole(ax,
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
                   **kwargs):
    color = utils.return_key(kwargs,"color",None)
    alpha = utils.return_key(kwargs,"alpha",None)
    radius_multiplication_power = utils.return_key(kwargs,"radius_multiplication",None)
    radius_multiplication = r_array ** radius_multiplication_power if radius_multiplication_power is not None else 1
    if(monopole_division):
        if(error_monopole is None):
            ax[0].plot(r_array,monopole*radius_multiplication,color=color,alpha=alpha)
            ax[1].plot(r_array,dipole/monopole,color=color,alpha=alpha)
            ax[2].plot(r_array,quadrupole/monopole,color=color,alpha=alpha)
            ax[3].plot(r_array,hexadecapole/monopole,color=color,alpha=alpha)
        else:
            ax[0].errorbar(r_array,monopole*radius_multiplication,error_monopole*radius_multiplication,color=color,alpha=alpha)
            ax[1].errorbar(r_array,dipole/monopole,error_dipole/monopole,color=color,alpha=alpha)
            ax[2].errorbar(r_array,quadrupole/monopole,error_quadrupole/monopole,color=color,alpha=alpha)
            ax[3].errorbar(r_array,hexadecapole/monopole,error_hexadecapole/monopole,color=color,alpha=alpha)
    else:
        if(error_monopole is None):
            ax[0].plot(r_array,monopole*radius_multiplication,color=color,alpha=alpha)
            ax[1].plot(r_array,dipole,color=color,alpha=alpha)
            ax[2].plot(r_array,quadrupole,color=color,alpha=alpha)
            ax[3].plot(r_array,hexadecapole,color=color,alpha=alpha)
        else:
            ax[0].errorbar(r_array,monopole*radius_multiplication,error_monopole*radius_multiplication,color=color,alpha=alpha)
            ax[1].errorbar(r_array,dipole,error_dipole,color=color,alpha=alpha)
            ax[2].errorbar(r_array,quadrupole,error_quadrupole,color=color,alpha=alpha)
            ax[3].errorbar(r_array,hexadecapole,error_hexadecapole,color=color,alpha=alpha)
    if(set_label):
        if(monopole_division):
            ax[1].set_ylim(top=0.2, bottom=-0.2)
            ax[2].set_ylim(top=0.0, bottom=-0.5)
            ax[3].set_ylim(top=0.0, bottom=-0.5)
            ax[1].set_ylabel(r"$\xi^{vg}_{1}$"+"/"+r"$\xi^{vg}_{0}$", fontsize=13)
            ax[2].set_ylabel(r"$\xi^{vg}_{2}$"+"/"+r"$\xi^{vg}_{0}$", fontsize=13)
            ax[3].set_ylabel(r"$\xi^{vg}_{4}$"+"/"+r"$\xi^{vg}_{0}$", fontsize=13)
        else:
            ax[1].set_ylabel("dipole " + r"$\xi^{vg}_{1}$", fontsize=15)
            ax[2].set_ylabel("quadrupole " + r"$\xi^{vg}_{2}$", fontsize=15)
            ax[3].set_ylabel("hexadecapole " + r"$\xi^{vg}_{4}$", fontsize=15)

        title_add = ""
        if(radius_multiplication_power is not None):
            title_add = r" $\times r^{" + str(radius_multiplication_power) + "}$"
        ax[0].set_ylabel("monopole " + r"$\xi^{vg}_{0}$" + title_add, fontsize=15)
        ax[0].grid()
        ax[0].tick_params(axis='y', labelsize=13)
        ax[1].grid()
        ax[1].tick_params(axis='y', labelsize=13)
        ax[2].grid()
        ax[2].tick_params(axis='y', labelsize=13)
        ax[3].grid()
        ax[3].tick_params(axis='y', labelsize=13)
        ax[3].set_xlabel("r [" + r"$\mathrm{Mpc\cdot h^{-1}}$" + "]", fontsize=15)
        ax[3].tick_params(axis='x', labelsize=13)



def get_mean_multipoles(file_xi,
                        supress_first_pixels=0,
                        multipole_method="rect"):

    if(type(file_xi) == list):
        (mean_monopole,
        mean_dipole,
        mean_quadrupole,
        mean_hexadecapole) = [],[],[],[]
        for i in range(len(file_xi)):
            xcorr = xcorr_objects.CrossCorr.init_from_fits(file_xi[i],
                                                           supress_first_pixels=supress_first_pixels)
            (r,mu,da) =  xcorr.r_array,xcorr.mu_array,xcorr.xi_array
            r[r==0] = np.nan
            r_array = np.nanmean(r,axis=1)
            (monopole,dipole,quadrupole,hexadecapole) = get_poles(mu,da,multipole_method)
            mean_monopole.append(monopole)
            mean_dipole.append(dipole)
            mean_quadrupole.append(quadrupole)
            mean_hexadecapole.append(hexadecapole)
        mean_monopole,error_mean_monopole = np.mean(mean_monopole,axis=0),sem(mean_monopole,axis=0)
        mean_dipole,error_mean_dipole = np.mean(mean_dipole,axis=0),sem(mean_dipole,axis=0)
        mean_quadrupole,error_mean_quadrupole = np.mean(mean_quadrupole,axis=0),sem(mean_quadrupole,axis=0)
        mean_hexadecapole,error_mean_hexadecapole = np.mean(mean_hexadecapole,axis=0),sem(mean_hexadecapole,axis=0)
    else:
        xcorr = xcorr_objects.CrossCorr.init_from_fits(file_xi,
                                                       supress_first_pixels=supress_first_pixels)
        (r,mu,da) =  xcorr.r_array,xcorr.mu_array,xcorr.xi_array
        r[r==0] = np.nan
        r_array = np.nanmean(r,axis=1)
        (mean_monopole,mean_dipole,mean_quadrupole,mean_hexadecapole) = get_poles(mu,da,multipole_method)
        (error_mean_monopole,error_mean_dipole,error_mean_quadrupole,error_mean_hexadecapole) = None, None,None,None

    return(r_array,
           mean_monopole,
           mean_dipole,
           mean_quadrupole,
           mean_hexadecapole,
           error_mean_monopole,
           error_mean_dipole,
           error_mean_quadrupole,
           error_mean_hexadecapole)



def compute_and_plot_multipole(file_xi,
                               save_plot,
                               supress_first_pixels=0,
                               error_bar=None,
                               multipole_method="rect",
                               monopole_division=False,
                               second_plot=None,
                               set_label=True,
                               factor_monopole = None,
                               **kwargs):
    xcorr = xcorr_objects.CrossCorr.init_from_fits(file_xi,
                                                   supress_first_pixels=supress_first_pixels)
    (r,mu,da) =  xcorr.r_array,xcorr.mu_array,xcorr.xi_array
    r[r==0] = np.nan
    r_array = np.nanmean(r,axis=1)
    (monopole,dipole,quadrupole,hexadecapole) = get_poles(mu,da,multipole_method)
    if(factor_monopole is not None):
        dipole = dipole +  factor_monopole[0] * monopole
        quadrupole = quadrupole +  factor_monopole[1] * monopole
        hexadecapole = hexadecapole +  factor_monopole[2] * monopole
    if(second_plot is not None):
        fig = second_plot[0]
        ax = second_plot[1]
    else:
        fig,ax=plt.subplots(4,1,figsize=(8,10),sharex=True)

    if(error_bar is not None):
        (error_monopole,error_dipole,error_quadrupole,error_hexadecapole)=get_error_bars_from_no_export(error_bar,multipole_method,supress_first_pixels=supress_first_pixels)
    else:
        (error_monopole,error_dipole,error_quadrupole,error_hexadecapole)= None,None,None,None
    plot_multipole(ax,
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
                   **kwargs)
    if(save_plot is not None):
        fig.savefig(f"{save_plot}.pdf",format="pdf")
        if(error_bar is None):
            header = "r [Mpc.h-1]     monopole xi0    dipole xi1    quadrupole xi2    hexadecapole xi4"
            txt = np.transpose(np.stack([r_array,monopole,dipole,quadrupole,hexadecapole]))
        else:
            header = "r [Mpc.h-1]     monopole xi0    monopole error sigma_xi0    dipole xi1    dipole error sigma_xi1    quadrupole xi2    quadrupole error sigma_xi2    hexadecapole xi4    hexadecapole error sigma_xi4"
            txt = np.transpose(np.stack([r_array,monopole,error_monopole,dipole,error_dipole,quadrupole,error_quadrupole,hexadecapole,error_hexadecapole]))
        np.savetxt(f"{save_plot}.txt",txt,header=header,delimiter="    ")

    return(fig,ax,r_array,monopole,dipole,quadrupole,hexadecapole)



def compute_and_plot_beta(file_xi,
                          save_plot,
                          supress_first_pixels=0,
                          multipole_method="rect",
                          monopole_division=False,
                          second_plot=None,
                          set_label=True,
                          factor_monopole = None,
                          **kwargs):

    xcorr = xcorr_objects.CrossCorr.init_from_fits(file_xi,
                                                   supress_first_pixels=supress_first_pixels)
    (r,mu,da) =  xcorr.r_array,xcorr.mu_array,xcorr.xi_array
    r[r==0] = np.nan
    r_array = np.nanmean(r,axis=1)
    (monopole,dipole,quadrupole,hexadecapole) = get_poles(mu,da,multipole_method)

    monopole_bar = np.zeros(monopole.shape)
    for i in range(len(r_array)):
        mask = (r_array <= r_array[i]) & (r_array > 0)
        integrand = monopole[mask] * r_array[mask]**2
        integral = integrate.simps(integrand,r_array[mask],axis=0)
        monopole_bar[i] = integral * 3 / (r_array[i]**3)


    if(second_plot is not None):
        fig = second_plot[0]
        ax = second_plot[1]
    else:
        fig,ax=plt.subplots(4,1,figsize=(8,10),sharex=True)

    color = "C0"
    ax[0].plot(r_array[1:],monopole[1:],color=color)
    ax[1].plot(r_array[1:],monopole_bar[1:],color=color)
    ax[2].plot(r_array[1:],monopole[1:] - monopole_bar[1:],color=color)
    ax[3].plot(r_array[1:],quadrupole[1:],color=color)


    if(save_plot is not None):
        fig.savefig(f"{save_plot}.pdf",format="pdf")


def compute_and_plot_beta_mean(file_xi,
                               save_plot,
                               supress_first_pixels=0,
                               pixel_to_plot = 3,
                               file_xi_substract=None,
                               multipole_method="rect",
                               monopole_division=False,
                               second_plot=None,
                               set_label=True,
                               factor_monopole = None,
                               minuit_parameters = None,
                               **kwargs):


    (r_array,
     mean_monopole,
     mean_dipole,
     mean_quadrupole,
     mean_hexadecapole,
     error_mean_monopole,
     error_mean_dipole,
     error_mean_quadrupole,
     error_mean_hexadecapole) = get_mean_multipoles(file_xi,
                                                    supress_first_pixels=supress_first_pixels,
                                                    multipole_method=multipole_method)
    if(file_xi_substract is not None):
        (r_array2,
         mean_monopole2,
         mean_dipole2,
         mean_quadrupole2,
         mean_hexadecapole2,
         error_mean_monopole2,
         error_mean_dipole2,
         error_mean_quadrupole2,
         error_mean_hexadecapole2) = get_mean_multipoles(file_xi_substract,
                                                        supress_first_pixels=supress_first_pixels,
                                                        multipole_method=multipole_method)

    mean_monopole_bar = np.zeros(mean_monopole.shape)
    for i in range(len(r_array)):
        mask = (r_array <= r_array[i]) & (r_array > 0)
        integrand = mean_monopole[mask] * r_array[mask]**2
        integral = integrate.simps(integrand,r_array[mask],axis=0)
        mean_monopole_bar[i] = integral * 3 / (r_array[i]**3)

    plt.figure(figsize=(10,7))

    corrected_quad = mean_quadrupole[pixel_to_plot:] - mean_quadrupole2[pixel_to_plot:]
    diff_mono = mean_monopole[pixel_to_plot:] - mean_monopole_bar[pixel_to_plot:]
    error_corrected_quad = np.sqrt(error_mean_quadrupole[pixel_to_plot:]**2 + error_mean_quadrupole2[pixel_to_plot:]**2 )
    error_diff_mono = np.sqrt(2) * error_mean_monopole[pixel_to_plot:]

    # data_y = diff_mono/corrected_quad
    # data_yerr = data_y * np.sqrt((error_diff_mono/diff_mono)**2 + (error_corrected_quad/corrected_quad)**2)
    # model = lambda beta : (3+beta)/(2*beta)


    # data_y = corrected_quad
    # data_yerr = error_corrected_quad
    # model = lambda beta : diff_mono * ((2*beta)/(3+beta))


    data_y = mean_quadrupole[pixel_to_plot:]
    data_yerr = error_mean_quadrupole[pixel_to_plot:]
    model = lambda beta : diff_mono * ((2*beta)/(3+beta))

    cost_function = lambda beta : np.nansum(((data_y - model(beta))/data_yerr)**2)
    minuit = Minuit(cost_function,**minuit_parameters)
    minuit.migrad(1000,resume=True)
    beta = dict(minuit.values)["beta"]
    print("beta: ", beta)


    # plt.errorbar(r_array[pixel_to_plot:],mean_monopole[pixel_to_plot:],error_mean_monopole[pixel_to_plot:])
    # plt.errorbar(r_array[pixel_to_plot:],diff_mono,error_diff_mono)
    # plt.errorbar(r_array[pixel_to_plot:],corrected_quad,error_corrected_quad)
    # plt.errorbar(r_array[pixel_to_plot:],((3+beta)/(2*beta))*corrected_quad, ((3+beta)/(2*beta))*error_corrected_quad)
    #
    #
    # plt.legend([r"$\xi^{vg}_{0}$",
    #             r"$\xi^{vg}_{0}$" + " - " +  r"$\overline{\xi}^{vg}_{0}$",
    #             r"$\xi^{vg}_{2}(RSD)$" + " - " + r"$\xi^{vg}_{2}(noRSD)$",
    #             r"$\frac{3+\beta}{2\beta}[\xi^{vg}_{2}(RSD)$" + " - " + r"$\xi^{vg}_{2}(noRSD)]$"])
    #

    plt.errorbar(r_array[pixel_to_plot:],mean_monopole[pixel_to_plot:],error_mean_monopole[pixel_to_plot:])
    plt.errorbar(r_array[pixel_to_plot:],diff_mono,error_diff_mono)
    plt.errorbar(r_array[pixel_to_plot:],data_y,data_yerr)
    plt.errorbar(r_array[pixel_to_plot:],((3+beta)/(2*beta))*data_y, ((3+beta)/(2*beta))*data_yerr)


    plt.legend([r"$\xi^{vg}_{0}$",
                r"$\xi^{vg}_{0}$" + " - " +  r"$\overline{\xi}^{vg}_{0}$",
                r"$\xi^{vg}_{2}(RSD)$",
                r"$\frac{3+\beta}{2\beta}[\xi^{vg}_{2}(RSD)]$"])




    if(save_plot is not None):
        plt.savefig(f"{save_plot}.pdf",format="pdf")


def compute_and_plot_multipole_several(file_xi,
                                       save_plot,
                                       supress_first_pixels=0,
                                       error_bar=None,
                                       multipole_method="rect",
                                       monopole_division=False,
                                       second_plot=None,
                                       set_label=True,
                                       factor_monopole = None,
                                       **kwargs):

    (mean_monopole,
    mean_dipole,
    mean_quadrupole,
    mean_hexadecapole) = [],[],[],[]
    for i in range(len(file_xi)):
        (fig,
         ax,
         r_array,
         monopole,
         dipole,
         quadrupole,
         hexadecapole) = compute_and_plot_multipole(file_xi[i],
                                                    save_plot,
                                                    supress_first_pixels=supress_first_pixels,
                                                    error_bar=error_bar[i] if error_bar is not None else None,
                                                    multipole_method=multipole_method,
                                                    monopole_division=monopole_division,
                                                    second_plot=second_plot,
                                                    set_label=False,
                                                    factor_monopole = factor_monopole,
                                                    **kwargs)


        if i == 0 : second_plot = (fig,ax)
        mean_monopole.append(monopole)
        mean_dipole.append(dipole)
        mean_quadrupole.append(quadrupole)
        mean_hexadecapole.append(hexadecapole)
    mean_monopole = np.mean(mean_monopole,axis=0)
    mean_dipole = np.mean(mean_dipole,axis=0)
    mean_quadrupole = np.mean(mean_quadrupole,axis=0)
    mean_hexadecapole = np.mean(mean_hexadecapole,axis=0)
    plot_error = utils.return_key(kwargs,"plot_mean_error",False)
    (error_mean_monopole,
     error_mean_dipole,
     error_mean_quadrupole,
     error_mean_hexadecapole) = (None,None,None,None)
    if(plot_error):
        error_mean_monopole = sem(mean_monopole,axis=0)
        error_mean_dipole = sem(mean_dipole,axis=0)
        error_mean_quadrupole = sem(mean_quadrupole,axis=0)
        error_mean_hexadecapole = sem(mean_hexadecapole,axis=0)
    kwargs["alpha"] = None
    plot_multipole(ax,
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
                   **kwargs)
    return(fig,ax,r_array,mean_monopole,mean_dipole,mean_quadrupole,mean_hexadecapole)






def compute_and_plot_multipole_comparison(name_in,
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
                                          **kwargs):
    (fig,
     ax,
     r_array1,
     monopole1,
     dipole1,
     quadrupole1,
     hexadecapole1) = compute_and_plot_multipole(name_in,
                                                 None,
                                                 supress_first_pixels=supress_first_pixels,
                                                 error_bar=error_bar,
                                                 multipole_method=multipole_method,
                                                 monopole_division=monopole_division,
                                                 set_label=True,
                                                 **kwargs)
    (fig,
     ax,
     r_array2,
     monopole2,
     dipole2,
     quadrupole2,
     hexadecapole2) = compute_and_plot_multipole(name_in2,
                                                 None,
                                                 supress_first_pixels=supress_first_pixels,
                                                 error_bar=error_bar2,
                                                 multipole_method=multipole_method,
                                                 second_plot=(fig,ax),
                                                 monopole_division=monopole_division,
                                                 set_label=False,
                                                 **kwargs)
    if(file_xi_optional is not None):
        for i in range(len(file_xi_optional)):

            xcorr = xcorr_objects.CrossCorr.init_from_fits(file_xi_optional[i],supress_first_pixels=supress_first_pixels)
            (r,mu,da) =  xcorr.r_array,xcorr.mu_array,xcorr.xi_array
            r_array = np.mean(r,axis=1)
            if(monopole_normalization):
                (monopole,dipole,quadrupole,hexadecapole) = get_poles(mu,da,multipole_method)
                monopole_ratio = interp1d(r_array1,monopole1,bounds_error=False,fill_value=1.0)(r_array)/monopole
                for j in range(da.shape[1]):
                    da[:,j] = da[:,j] * monopole_ratio
            (monopole,dipole,quadrupole,hexadecapole) = get_poles(mu,da,multipole_method)
            if(error_bar_optional is not None):
                (error_monopole,error_dipole,error_quadrupole,error_hexadecapole)=get_error_bars_from_no_export(error_bar_optional[i],multipole_method,supress_first_pixels=supress_first_pixels)
            else:
                (error_monopole,error_dipole,error_quadrupole,error_hexadecapole)= None,None,None,None

            plot_multipole(ax,
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
                           **kwargs)
    ax[0].legend(legend)
    fig.savefig(f"{nameout}.pdf",format="pdf")








def compute_and_plot_multipole_several_comparison(names_in,
                                                  nameout,
                                                  error_bar=None,
                                                  supress_first_pixels=0,
                                                  file_xi_optional = None,
                                                  error_bar_optional = None,
                                                  legend=None,
                                                  legend_elements=None,
                                                  factor_monopole = None,
                                                  multipole_method="rect",
                                                  monopole_division=False,
                                                  alpha=0.4,
                                                  **kwargs):
    if(type(names_in) == list):
        (fig,
         ax,
         r_array,
         mean_monopole,
         mean_dipole,
         mean_quadrupole,
         mean_hexadecapole) = compute_and_plot_multipole_several(names_in,
                                                                 None,
                                                                 supress_first_pixels=supress_first_pixels,
                                                                 error_bar=error_bar,
                                                                 multipole_method=multipole_method,
                                                                 monopole_division=monopole_division,
                                                                 second_plot=None,
                                                                 set_label=True,
                                                                 factor_monopole = factor_monopole,
                                                                 alpha=alpha,
                                                                 color="C0",
                                                                 **kwargs)

    else:
        (fig,
        ax,
        r_array,
        monopole,
        dipole,
        quadrupole,
        hexadecapole) = compute_and_plot_multipole(names_in,
                                                    None,
                                                    supress_first_pixels=supress_first_pixels,
                                                    error_bar=error_bar,
                                                    multipole_method=multipole_method,
                                                    second_plot=None,
                                                    monopole_division=monopole_division,
                                                    factor_monopole = factor_monopole,
                                                    set_label=True,
                                                    color="C0",
                                                    **kwargs)

    second_plot = (fig,ax)

    if(file_xi_optional is not None):
        for i in range(len(file_xi_optional)):
            if(error_bar_optional is None):
                error_bar = None
            else:
                error_bar = error_bar_optional[i]
            if(type(file_xi_optional[i]) == list):
                (fig,
                 ax,
                 r_array,
                 mean_monopole,
                 mean_dipole,
                 mean_quadrupole,
                 mean_hexadecapole) = compute_and_plot_multipole_several(file_xi_optional[i],
                                                                         None,
                                                                         supress_first_pixels=supress_first_pixels,
                                                                         error_bar=error_bar,
                                                                         multipole_method=multipole_method,
                                                                         monopole_division=monopole_division,
                                                                         second_plot=second_plot,
                                                                         set_label=False,
                                                                         factor_monopole = factor_monopole,
                                                                         alpha=alpha,
                                                                         color=f"C{i+1}",
                                                                         **kwargs)

            else:
                (fig,
                 ax,
                 r_array,
                 monopole,
                 dipole,
                 quadrupole,
                 hexadecapole) = compute_and_plot_multipole(file_xi_optional[i],
                                                             None,
                                                             supress_first_pixels=supress_first_pixels,
                                                             error_bar=error_bar,
                                                             multipole_method=multipole_method,
                                                             second_plot=second_plot,
                                                             monopole_division=monopole_division,
                                                             factor_monopole = factor_monopole,
                                                             set_label=False,
                                                             color=f"C{i+1}",
                                                             **kwargs)

    if(legend_elements is not None):
        ax[0].legend(handles=legend_elements)
    else:
        ax[0].legend(legend)
    fig.savefig(f"{nameout}.pdf",format="pdf")




def plot_2d(name_in,
            name_out,
            supress_first_pixels=0,
            **kwargs):
    xcorr = xcorr_objects.CrossCorr.init_from_fits(name_in,
                                                   supress_first_pixels=supress_first_pixels)
    xcorr.plot_2d(name=name_out,**kwargs)




    # CR - Rewrite wedge plots with Julian routines


def compute_and_plot_wedge_comparison(file_xi,nameout,comparison,legend=None):
    fig,axes=plt.subplots(4,1,figsize=(8,10),sharex=True)
    mus=[0., 0.5, 0.8, 0.95, 1.]
    add_wedge(file_xi,mus,axes)
    add_wedge(comparison,mus,axes)
    for i in range(len(axes)):
        axes[i].grid()
        axes[i].set_ylabel(f"{mus[i]}"+r"$< \mu <$"+f"{mus[i+1]}")
    axes[-1].set_xlabel(r"$r~[\mathrm{Mpc/h}]$")
    axes[0].legend(legend)
    axes[0].set_title(r"$r^2\xi(r)$")
    plt.savefig(f"{nameout}.pdf",format = "pdf")


def compute_and_plot_wedge(file_xi,nameout):
    fig,axes=plt.subplots(4,1,figsize=(8,10),sharex=True)
    mus=[0., 0.5, 0.8, 0.95, 1.]
    add_wedge(file_xi,mus,axes)
    for i in range(len(axes)):
        axes[i].grid()
        axes[i].set_ylabel(f"{mus[i]}"+r"$< \mu <$"+f"{mus[i+1]}")
    axes[0].set_title(r"$r^2\xi(r)$")
    axes[-1].set_xlabel(r"$r~[\mathrm{Mpc/h}]$")
    plt.savefig(f"{nameout}.pdf",format = "pdf")


def add_wedge(file_xi,mus,axes):
    from picca import wedgize
    import fitsio
    ax = 0
    for mumin,mumax in zip(mus[:-1],mus[1:]):
        h = fitsio.FITS(file_xi)
        # ff = h5py.File(fitefile,'r')
        # fit = ff[fitpath+'/fit'][...]
        # ff.close()
        da = h[1]['DA'][:]
        co = h[1]['CO'][:]
        rpmin = h[1].read_header()['RPMIN']
        rpmax = h[1].read_header()['RPMAX']
        rtmax = h[1].read_header()['RTMAX']
        nrp = h[1].read_header()['NP']
        nt = h[1].read_header()['NT']
        h.close()
        # ff.close()
        b = wedgize.wedge(mumin=mumin,mumax=mumax,rtmin=0,rtmax=rtmax,
                          rpmin =rpmin,rpmax=rpmax,rmax=50,
                          nrp=nrp,nrt=nt,absoluteMu=True)
        r,d,c = b.wedge(da,co)
        # r,f,_ = b.wedge(fit,co)
        # axes[ax].errorbar(r,d*r**2,yerr=np.sqrt(c.diagonal())*r**2,fmt="o")
        axes[ax].errorbar(r,d,yerr=np.sqrt(c.diagonal()),fmt="o")
        ax = ax + 1
        # plt.plot(r,f*r**2)
