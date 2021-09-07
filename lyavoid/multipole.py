import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
from scipy.stats import sem
from lyavoid import xcorr_objects,utils
from scipy.interpolate import interp1d
from iminuit import Minuit




def get_poles(mu,da,method):
    ell = [0,1,2,4]
    multipole = xcorr_objects.Multipole.init_from_xcorr(ell,mu,da,method,name=None,r_array=None,extrapolate=True)
    monopole = multipole.poles[0]
    dipole =  multipole.poles[1]
    quadrupole =  multipole.poles[2]
    hexadecapole = multipole.poles[4]
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




def get_mean_multipoles(file_xi,
                        supress_first_pixels=0,
                        multipole_method="rect",
                        error_bar_file_unique=None):

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

        if(error_bar_file_unique is not None):
            (error_mean_monopole,
             error_mean_dipole,
             error_mean_quadrupole,
             error_mean_hexadecapole)=get_error_bars_from_no_export(error_bar_file_unique,
                                                                    multipole_method,
                                                                    supress_first_pixels=supress_first_pixels)
        else:
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
    style = utils.return_key(kwargs,"style",[0,1,2,4])
    poles_to_plot = utils.return_key(kwargs,"poles_to_plot",[0,1,2,4])

    color = utils.return_key(kwargs,"color",None)
    alpha = utils.return_key(kwargs,"alpha",None)
    radius_multiplication_power = utils.return_key(kwargs,"radius_multiplication",None)
    radius_multiplication = r_array ** radius_multiplication_power if radius_multiplication_power is not None else 1

    title_add = ""
    if(radius_multiplication_power is not None):
        if(radius_multiplication_power ==1):
            title_add = r" $\times r$"
        else:
            title_add = r" $\times r^{" + str(radius_multiplication_power) + "}$"


    if(monopole_division):
        poles_label = {0 : r"$\xi^{vl}_{0}$" + title_add,
                       1 : r"$\xi^{vl}_{1}$"+"/"+r"$\xi^{vl}_{0}$",
                       2 : r"$\xi^{vl}_{2}$"+"/"+r"$\xi^{vl}_{0}$",
                       4 : r"$\xi^{vl}_{4}$"+"/"+r"$\xi^{vl}_{0}$"}
        poles = {0 : monopole*radius_multiplication,
                 1 : dipole/monopole,
                 2 : quadrupole/monopole,
                 4 : hexadecapole/monopole}
        if(error_monopole is not None):
            error_poles = {0 : error_monopole*radius_multiplication,
                           1 : error_dipole/monopole,
                           2 : error_quadrupole/monopole,
                           4 : error_hexadecapole/monopole}
    else:
        poles = {0 : monopole*radius_multiplication,
                 1 : dipole,
                 2 : quadrupole,
                 4 : hexadecapole}
        poles_label = {0 : r"$\xi^{vl}_{0}$" + title_add,
                       1 : r"$\xi^{vl}_{1}$",
                       2 : r"$\xi^{vl}_{2}$",
                       4 : r"$\xi^{vl}_{4}$"}
        if(error_monopole is not None):
            error_poles = {0 : error_monopole*radius_multiplication,
                           1 : error_dipole,
                           2 : error_quadrupole,
                           4 : error_hexadecapole}



    fontsize = utils.return_key(kwargs,"fontsize",13)
    labelsize_x = utils.return_key(kwargs,"labelsize_x",13)
    labelsize_y = utils.return_key(kwargs,"labelsize_y",13)
    linestyle = utils.return_key(kwargs,"linestyle",None)
    marker = utils.return_key(kwargs,"marker",None)
    for i in range(len(poles_to_plot)):
        if(style is None):
            ax[i].grid()
        if(error_monopole is not None):
            ax[i].errorbar(r_array,poles[poles_to_plot[i]],error_poles[poles_to_plot[i]],color=color,alpha=alpha,linestyle=linestyle,marker=marker)
        else:
            ax[i].plot(r_array,poles[poles_to_plot[i]],color=color,alpha=alpha,linestyle=linestyle,marker=marker)
        if(set_label):
            ax[i].set_ylabel(poles_label[poles_to_plot[i]], fontsize=fontsize)
            ax[i].tick_params(axis='y', labelsize=labelsize_y)
        if(i == len(poles_to_plot) - 1 ):
            ax[i].set_xlabel(r"$r~[\mathrm{h^{-1} Mpc}]$", fontsize=fontsize)
            ax[i].tick_params(axis='x', labelsize=labelsize_x)






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
    if(xcorr.rmu == False):
        xcorr.switch()
    (r,mu,da) =  xcorr.r_array,xcorr.mu_array,xcorr.xi_array
    r[r==0] = np.nan
    r_array = np.nanmean(r,axis=1)
    (monopole,dipole,quadrupole,hexadecapole) = get_poles(mu,da,multipole_method)
    if(factor_monopole is not None):
        dipole = dipole +  factor_monopole[0] * monopole
        quadrupole = quadrupole +  factor_monopole[1] * monopole
        hexadecapole = hexadecapole +  factor_monopole[2] * monopole
    poles_to_plot = utils.return_key(kwargs,"poles_to_plot",[0,1,2,4])
    figsize = utils.return_key(kwargs,"figsize",(8,10))
    if(second_plot is not None):
        fig = second_plot[0]
        ax = second_plot[1]
    else:
        fig,ax=plt.subplots(len(poles_to_plot),1,figsize=figsize,sharex=True)

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
            error_poles = None
        else:
            error_poles = {0:error_monopole,1:error_dipole,2:error_quadrupole,4:error_hexadecapole}
        mult = xcorr_objects.Multipole(name=f"{save_plot}.fits",
                                       r_array=r_array,
                                       ell=[0,1,2,4],
                                       poles={0:monopole,1:dipole,2:quadrupole,4:hexadecapole},
                                       error_poles=error_poles)
        mult.write_fits()
    return(fig,ax,r_array,monopole,dipole,quadrupole,hexadecapole)




def compute_and_plot_beta_mean(file_xi,
                               save_plot,
                               supress_first_pixels_fit = 3,
                               file_xi_substract=None,
                               error_bar_file_unique=None,
                               multipole_method="rect",
                               minuit_parameters = None,
                               legend = None,
                               ax=None,
                               substract_monopole_factor=None,
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
                                                    multipole_method=multipole_method,
                                                    error_bar_file_unique=error_bar_file_unique)
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
                                                         multipole_method=multipole_method,
                                                         error_bar_file_unique=error_bar_file_unique)

        if(substract_monopole_factor is not None):
            mean_monopole = mean_monopole - substract_monopole_factor * mean_quadrupole2

    mean_monopole_bar = np.zeros(mean_monopole.shape)
    for i in range(len(r_array)):
        mask = (r_array <= r_array[i]) & (r_array > 0)
        integrand = mean_monopole[mask] * r_array[mask]**2
        integral = integrate.simps(integrand,r_array[mask],axis=0)
        mean_monopole_bar[i] = integral * 3 / (r_array[i]**3)


    if(file_xi_substract is not None):
        corrected_quad = mean_quadrupole[supress_first_pixels_fit:] - mean_quadrupole2[supress_first_pixels_fit:]
        error_corrected_quad = np.sqrt(error_mean_quadrupole[supress_first_pixels_fit:]**2 + error_mean_quadrupole2[supress_first_pixels_fit:]**2 )
    else:
        corrected_quad = mean_quadrupole[supress_first_pixels_fit:]
        error_corrected_quad = error_mean_quadrupole[supress_first_pixels_fit:]

    diff_mono = mean_monopole[supress_first_pixels_fit:] - mean_monopole_bar[supress_first_pixels_fit:]
    error_diff_mono = np.sqrt(2) * error_mean_monopole[supress_first_pixels_fit:]

    mean_monopole = mean_monopole[supress_first_pixels_fit:]
    error_mean_monopole = error_mean_monopole[supress_first_pixels_fit:]

    r_array = r_array[supress_first_pixels_fit:]


    data_y = corrected_quad
    data_yerr = error_corrected_quad
    model = lambda beta : diff_mono * ((2*beta)/(3+beta))

    cost_function = lambda beta : np.nansum(((data_y - model(beta))/data_yerr)**2)
    minuit = Minuit(cost_function,**minuit_parameters)
    minuit.migrad(10000,resume=True)
    beta = dict(minuit.values)["beta"]
    print("beta: ", dict(minuit.values)["beta"])
    print("beta error: ", dict(minuit.errors)["beta"])

    if(ax is None):
        style = utils.return_key(kwargs,"style",None)
        if(style is not None):
            plt.style.use(style)
        plt.figure(figsize=(10,7))
        ax = plt.gca()



    plot_multipole([ax],
                   r_array,
                   mean_monopole,None,None,None,
                   error_mean_monopole,None,None,None,
                   set_label=False,
                   poles_to_plot = [0],
                   **kwargs)
    plot_multipole([ax],
                   r_array,
                   diff_mono,None,None,None,
                   error_diff_mono,None,None,None,
                   set_label=False,
                   poles_to_plot = [0],
                   **kwargs)
    plot_multipole([ax],
                   r_array,
                   corrected_quad,None,None,None,
                   error_corrected_quad,None,None,None,
                   set_label=False,
                   poles_to_plot = [0],
                   **kwargs)
    plot_multipole([ax],
                   r_array,
                   ((3+beta)/(2*beta))*corrected_quad,None,None,None,
                   ((3+beta)/(2*beta))*error_corrected_quad,None,None,None,
                   set_label=False,
                   poles_to_plot = [0],
                   **kwargs)

    fontsize_legend = utils.return_key(kwargs,"fontsize_legend",12)
    if(legend is not None):
        ax.legend(legend,fontsize=fontsize_legend)


    if(save_plot is not None):
        plt.savefig(f"{save_plot}.pdf",format="pdf")




def compute_and_plot_quadrupoles(file_xi,
                                 save_plot,
                                 file_xi_substract=None,
                                 supress_first_pixels=0,
                                 error_bar_file_unique=None,
                                 error_bar_file_unique_substract=None,
                                 multipole_method="rect",
                                 legend = None,
                                 ax=None,
                                 **kwargs):

    if(ax is None):
        style = utils.return_key(kwargs,"style",None)
        if(style is not None):
            plt.style.use(style)
        figsize = utils.return_key(kwargs,"figsize",None)
        plt.figure(figsize=figsize)
        ax = plt.gca()

    for i in range(len(file_xi)):
        (r_array,
         monopole,
         dipole,
         quadrupole,
         hexadecapole,
         error_monopole,
         error_dipole,
         error_quadrupole,
         error_hexadecapole) = get_mean_multipoles(file_xi[i],
                                                   supress_first_pixels=supress_first_pixels,
                                                   multipole_method=multipole_method,
                                                   error_bar_file_unique=error_bar_file_unique[i])
        if(file_xi_substract is not None):
            (r_array_substract,
             monopole_substract,
             dipole_substract,
             quadrupole_substract,
             hexadecapole_substract,
             error_monopole_substract,
             error_dipole_substract,
             error_quadrupole_substract,
             error_hexadecapole_substract) = get_mean_multipoles(file_xi_substract[i],
                                                                 supress_first_pixels=supress_first_pixels,
                                                                 multipole_method=multipole_method,
                                                                 error_bar_file_unique=error_bar_file_unique_substract[i])


            error_quadrupole = np.sqrt(error_quadrupole**2 + error_quadrupole_substract**2)
            quadrupole = quadrupole - quadrupole_substract

        if(i!=len(file_xi)-1):
            error_quadrupole = None
        plot_multipole([ax],
                       r_array,
                       quadrupole,None,None,None,
                       error_quadrupole,None,None,None,
                       set_label=False,
                       poles_to_plot = [0],
                       **kwargs)

    fontsize = utils.return_key(kwargs,"fontsize",13)
    labelsize_y = utils.return_key(kwargs,"labelsize_y",13)
    ylabel = utils.return_key(kwargs,"ylabel",None)
    if(ylabel is not None):
        ax.set_ylabel(ylabel, fontsize=fontsize)
        ax.tick_params(axis='y', labelsize=labelsize_y)

    fontsize_legend = utils.return_key(kwargs,"fontsize_legend",12)
    if(legend is not None):
        ax.legend(legend,fontsize=fontsize_legend)

    if(save_plot is not None):
        plt.tight_layout()
        plt.savefig(f"{save_plot}.pdf",format="pdf")





def compute_and_plot_beta_mean_several(file_xi,
                                       save_plot,
                                       file_xi_substract,
                                       error_bar_file_unique,
                                       substract_monopole_factor,
                                       legend,
                                       supress_first_pixels_fit = 3,
                                       multipole_method="rect",
                                       minuit_parameters = None,
                                       **kwargs):
    style = utils.return_key(kwargs,"style",None)
    if(style is not None):
        plt.style.use(style)

    figsize = utils.return_key(kwargs,"figsize",None)
    fig,ax=plt.subplots(1,len(file_xi),figsize=figsize,sharey=True)

    for i in range(len(file_xi)):
        compute_and_plot_beta_mean(file_xi[i],
                                   None,
                                   supress_first_pixels_fit = supress_first_pixels_fit,
                                   file_xi_substract=file_xi_substract[i],
                                   error_bar_file_unique=error_bar_file_unique[i],
                                   substract_monopole_factor=substract_monopole_factor[i],
                                   multipole_method=multipole_method,
                                   minuit_parameters = minuit_parameters,
                                   legend = legend[i],
                                   ax=ax[i],
                                   **kwargs)
    plt.tight_layout()
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
                                       name_mean_multipole = None,
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


    error_mean_monopole = sem(mean_monopole,axis=0)
    error_mean_dipole = sem(mean_dipole,axis=0)
    error_mean_quadrupole = sem(mean_quadrupole,axis=0)
    error_mean_hexadecapole = sem(mean_hexadecapole,axis=0)
    mean_monopole = np.mean(mean_monopole,axis=0)
    mean_dipole = np.mean(mean_dipole,axis=0)
    mean_quadrupole = np.mean(mean_quadrupole,axis=0)
    mean_hexadecapole = np.mean(mean_hexadecapole,axis=0)

    kwargs["alpha"] = None
    if(name_mean_multipole is not None):
        if(error_mean_monopole is not None):
            error_poles = {0:error_mean_monopole,1:error_mean_dipole,2:error_mean_quadrupole,4:error_mean_hexadecapole}
        else:
            error_poles = None

        mult = xcorr_objects.Multipole(name=f"{name_mean_multipole}.fits",
                                       r_array=r_array,
                                       ell=[0,1,2,4],
                                       poles={0:mean_monopole,1:mean_dipole,2:mean_quadrupole,4:mean_hexadecapole},
                                       error_poles=error_poles)
        mult.write_fits()

    plot_error = utils.return_key(kwargs,"plot_mean_error",False)
    if(not(plot_error)):
        (error_mean_monopole,
         error_mean_dipole,
         error_mean_quadrupole,
         error_mean_hexadecapole) = (None,None,None,None)
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
                                                  name_multipole = None,
                                                  name_multipole_optional = None,
                                                  **kwargs):
    style = utils.return_key(kwargs,"style",None)
    if(style is not None):
        plt.style.use(style)

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
                                                                 name_mean_multipole = name_multipole,
                                                                 **kwargs)

    else:
        (fig,
        ax,
        r_array,
        monopole,
        dipole,
        quadrupole,
        hexadecapole) = compute_and_plot_multipole(names_in,
                                                    name_multipole,
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
                if(name_multipole_optional is None):
                    name_multipole_optional_i = None
                else:
                    name_multipole_optional_i = name_multipole_optional[i]
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
                                                                         name_mean_multipole = name_multipole_optional_i,
                                                                         **kwargs)

            else:
                (fig,
                 ax,
                 r_array,
                 monopole,
                 dipole,
                 quadrupole,
                 hexadecapole) = compute_and_plot_multipole(file_xi_optional[i],
                                                             name_multipole_optional[i],
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
    plt.tight_layout()
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
