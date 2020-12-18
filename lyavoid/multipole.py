import numpy as np
import matplotlib.pyplot as plt
from scipy.special import legendre
from scipy import integrate
from scipy.stats import sem
from lyavoid import xcorr_objects






def get_multipole_from_array(xi,mu,order,method="simps"):
    poly = legendre(order)
    integrand = (xi*(1+2*order)*poly(mu))/2 # divided by two because integration is between -1 and 1
    if(method=="trap"):
        pole_l = integrate.trapz(integrand,mu,axis=1)
    elif(method=="simps"):
        pole_l = integrate.simps(integrand,mu,axis=1)
    return(pole_l)



def get_poles(mu,da,method="simps"):
    if(method == "rect"):
        dmu = np.zeros(mu.shape)
        dmu[:,1:-1] = (mu[:,2:] - mu[:,0:-2])/2
        dmu[:,0] = mu[:,1]-mu[:,0]
        dmu[:,-1] = mu[:,-1]-mu[:,-2]
        monopole = np.nansum(da*dmu, axis=1)/2
        dipole = np.nansum(da*mu*dmu, axis=1)*3./2
        quadrupole = np.nansum(da*0.5*(3*mu**2-1)*dmu, axis=1)*5./2
        hexadecapole = np.nansum(da*1/8*(35*mu**4-30*mu**2+3)*dmu, axis=1)*9./2
    elif((method == "trap")|(method == "simps")):
        # da = da[~np.isnan(da)]
        # mu = mu[~np.isnan(mu)]
        monopole = get_multipole_from_array(da,mu,0,method=method)
        dipole = get_multipole_from_array(da,mu,1,method=method)
        quadrupole = get_multipole_from_array(da,mu,2,method=method)
        hexadecapole = get_multipole_from_array(da,mu,4,method=method)
    return(monopole,dipole,quadrupole,hexadecapole)



def get_error_bars_from_no_export(file_xi_no_export,supress_first_pixels=0):
    xcorr = xcorr_objects.CrossCorr.init_from_fits(file_xi_no_export,exported=False,supress_first_pixels=supress_first_pixels)
    mu,da =  xcorr.mu_array,xcorr.xi_array
    monopole,dipole,quadrupole,hexadecapole = [],[],[],[]
    for i in range(len(da)):
        (mono,di,quad,hexa) = get_poles(mu,da[i])
        monopole.append(mono)
        dipole.append(di)
        quadrupole.append(quad)
        hexadecapole.append(hexa)
    error_monopole = sem(np.array(monopole))
    error_dipole = sem(np.array(dipole))
    error_quadrupole = sem(np.array(quadrupole))
    error_hexadecapole = sem(np.array(hexadecapole))
    return(error_monopole,error_dipole,error_quadrupole,error_hexadecapole)





def compute_and_plot_multipole(file_xi,save_plot,supress_first_pixels=0,
                               save_txt=None,error_bar=None,multipole_method="rect",second_plot=None):
    xcorr = xcorr_objects.CrossCorr.init_from_fits(file_xi,exported=True,
                                                   supress_first_pixels=supress_first_pixels)
    (r,mu,da) =  xcorr.r_array,xcorr.mu_array,xcorr.xi_array
    r[r==0] = np.nan
    r_array = np.nanmean(r,axis=1)
    (monopole,dipole,quadrupole,hexadecapole) = get_poles(mu,da,method=multipole_method)
    if(error_bar is not None):
        (error_monopole,error_dipole,error_quadrupole,error_hexadecapole)=get_error_bars_from_no_export(error_bar,supress_first_pixels=supress_first_pixels)

    if(save_txt is not None):
        if(error_bar is None):
            header = "r [Mpc.h-1]     monopole xi0    dipole xi1    quadrupole xi2    hexadecapole xi4"
            txt = np.transpose(np.stack([r_array,monopole,dipole,quadrupole,hexadecapole]))
        else:
            header = "r [Mpc.h-1]     monopole xi0    monopole error sigma_xi0    dipole xi1    dipole error sigma_xi1    quadrupole xi2    quadrupole error sigma_xi2    hexadecapole xi4    hexadecapole error sigma_xi4"
            txt = np.transpose(np.stack([r_array,monopole,error_monopole,dipole,error_dipole,quadrupole,error_quadrupole,hexadecapole,error_hexadecapole]))
        np.savetxt(save_txt.split(".")[0] + ".txt",txt,header=header,delimiter="    ")

    if(second_plot is not None):
        fig = second_plot[0]
        ax = second_plot[1]
    else:
        fig,ax=plt.subplots(4,1,figsize=(8,10),sharex=True)
    if(error_bar is None):
        ax[0].plot(r_array,monopole)
        ax[1].plot(r_array,dipole)
        ax[2].plot(r_array,quadrupole)
        ax[3].plot(r_array,hexadecapole)
    else:
        ax[0].errorbar(r_array,monopole,error_monopole)
        ax[1].errorbar(r_array,dipole,error_dipole)
        ax[2].errorbar(r_array,quadrupole,error_quadrupole)
        ax[3].errorbar(r_array,hexadecapole,error_hexadecapole)
    ax[0].grid()
    ax[0].set_ylabel("monopole " + r"$\xi^{vg}_{0}$", fontsize=15)
    ax[0].tick_params(axis='y', labelsize=13)
    ax[1].grid()
    ax[1].set_ylabel("dipole " + r"$\xi^{vg}_{1}$", fontsize=15)
    ax[1].tick_params(axis='y', labelsize=13)
    ax[2].grid()
    ax[2].set_ylabel("quadrupole " + r"$\xi^{vg}_{2}$", fontsize=15)
    ax[2].tick_params(axis='y', labelsize=13)
    ax[3].grid()
    ax[3].set_ylabel("hexadecapole " + r"$\xi^{vg}_{4}$", fontsize=15)
    ax[3].tick_params(axis='y', labelsize=13)
    ax[3].set_xlabel("r [" + r"$\mathrm{Mpc\cdot h^{-1}}$" + "]", fontsize=15)
    ax[3].tick_params(axis='x', labelsize=13)
    if(save_plot is not None):
        fig.savefig(f"{save_plot}.pdf",format="pdf")
    return(fig,ax)





def compute_and_plot_multipole_comparison(file_xi1,file_xi2,nameout,supress_first_pixels=0,
                                          error_bar1=None,error_bar2=None,save_txt1=None,
                                          save_txt2=None,file_xi_optional=None,
                                          error_bar_optional=None,legend=None,
                                          multipole_method="rect"):
    (fig,ax) = compute_and_plot_multipole(file_xi1,None,supress_first_pixels=supress_first_pixels,
                                   save_txt=save_txt1,error_bar=error_bar1,multipole_method=multipole_method)
    (fig,ax) = compute_and_plot_multipole(file_xi2,None,supress_first_pixels=supress_first_pixels,
                               save_txt=save_txt2,error_bar=error_bar2,multipole_method=multipole_method,
                               second_plot=(fig,ax))
    if(file_xi_optional is not None):
        for i in range(len(file_xi_optional)):
            xcorr = xcorr_objects.CrossCorr.init_from_fits(file_xi_optional[i],exported=True,supress_first_pixels=supress_first_pixels)
            (r,mu,da) =  xcorr.r_array,xcorr.mu_array,xcorr.xi_array
            r_array = np.mean(r,axis=1)
            (monopole,dipole,quadrupole,hexadecapole) = get_poles(mu,da)
            if(error_bar_optional is None):
                ax[0].plot(r_array,monopole)
                ax[1].plot(r_array,dipole)
                ax[2].plot(r_array,quadrupole)
                ax[3].plot(r_array,hexadecapole)
            else:
                (error_monopole,error_dipole,error_quadrupole,error_hexadecapole)=get_error_bars_from_no_export(error_bar_optional[i],supress_first_pixels=supress_first_pixels)
                ax[0].errorbar(r_array,monopole,error_monopole)
                ax[1].errorbar(r_array,dipole,error_dipole)
                ax[2].errorbar(r_array,quadrupole,error_quadrupole)
                ax[3].errorbar(r_array,hexadecapole,error_hexadecapole)
    ax[0].legend(legend)
    fig.savefig(f"{nameout}.pdf",format="pdf")



def plot_2d(file_xi,supress_first_pixels=0,**kwargs):
    xcorr = xcorr_objects.CrossCorr.init_from_fits(file_xi,exported=True,
                                                   supress_first_pixels=supress_first_pixels)
    xcorr.plot_2d(**kwargs)
