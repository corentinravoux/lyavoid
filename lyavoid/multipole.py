import numpy as np
import matplotlib.pyplot as plt
from scipy.special import legendre
from scipy import integrate
from scipy.stats import sem
from lyavoid import xcorr_objects
from scipy.interpolate import interp1d






def get_multipole_from_array(xi,mu,order,method):
    if(method=="rect"):
        pole_l = get_multipole_from_array_rect(xi,mu,order)
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



def get_multipole_from_array_rect_nbody(xi,mu,order):
    mu_bins = np.diff(mu)
    mu_mid = (mu[:,1:] + mu[:,:-1])/2.
    xi_mid = (xi[:,1:] + xi[:,:-1])/2.
    legendrePolynomial = (2.*order+1.)*legendre(order)(mu_mid)
    pole = np.sum(xi_mid*legendrePolynomial*mu_bins,axis=-1)/np.sum(mu_bins)
    return pole

def get_multipole_from_array_rect(xi,mu,order):
    dmu = np.zeros(mu.shape)
    dmu[:,1:-1] = (mu[:,2:] - mu[:,0:-2])/2
    dmu[:,0] = mu[:,1]-mu[:,0]
    dmu[:,-1] = mu[:,-1]-mu[:,-2]
    legendrePolynomial = (2.*order+1.)*legendre(order)(mu)
    pole = np.nansum(xi*legendrePolynomial*dmu, axis=1)/2
    return(pole)


# def get_multipole_from_array_rect_julian(xi,mu,order):
#     mu1d = np.unique(mu)
#     dmu = np.gradient(mu1d)[0]
#     legendrePolynomial = (2.*order+1.)*legendre(order)(mu)
#     pole = np.sum(xi*legendrePolynomial*dmu, axis=1)/2
#     return pole


def get_poles(mu,da,method):
    monopole = get_multipole_from_array(da,mu,0,method)
    dipole = get_multipole_from_array(da,mu,1,method)
    quadrupole = get_multipole_from_array(da,mu,2,method)
    hexadecapole = get_multipole_from_array(da,mu,4,method)
    return(monopole,dipole,quadrupole,hexadecapole)

def get_error_bars_from_no_export(file_xi_no_export,multipole_method,supress_first_pixels=0):
    xcorr = xcorr_objects.CrossCorr.init_from_fits(file_xi_no_export,exported=False,supress_first_pixels=supress_first_pixels)
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





def compute_and_plot_multipole(file_xi,save_plot,supress_first_pixels=0,
                               save_txt=None,error_bar=None,multipole_method="rect",
                               second_plot=None,monopole_division=False,set_label=True):
    xcorr = xcorr_objects.CrossCorr.init_from_fits(file_xi,exported=True,
                                                   supress_first_pixels=supress_first_pixels)
    (r,mu,da) =  xcorr.r_array,xcorr.mu_array,xcorr.xi_array
    r[r==0] = np.nan
    r_array = np.nanmean(r,axis=1)
    (monopole,dipole,quadrupole,hexadecapole) = get_poles(mu,da,multipole_method)
    if(second_plot is not None):
        fig = second_plot[0]
        ax = second_plot[1]
    else:
        fig,ax=plt.subplots(4,1,figsize=(8,10),sharex=True)

    if(error_bar is not None):
        (error_monopole,error_dipole,error_quadrupole,error_hexadecapole)=get_error_bars_from_no_export(error_bar,multipole_method,supress_first_pixels=supress_first_pixels)
    else:
        (error_monopole,error_dipole,error_quadrupole,error_hexadecapole)= None,None,None,None
    plot_multipole(ax,r_array,monopole,dipole,quadrupole,hexadecapole,
                   error_monopole,error_dipole,error_quadrupole,
                   error_hexadecapole,monopole_division=monopole_division,
                   set_label=set_label)
    if(save_plot is not None):
        fig.savefig(f"{save_plot}.pdf",format="pdf")

    if(save_txt is not None):
        if(error_bar is None):
            header = "r [Mpc.h-1]     monopole xi0    dipole xi1    quadrupole xi2    hexadecapole xi4"
            txt = np.transpose(np.stack([r_array,monopole,dipole,quadrupole,hexadecapole]))
        else:
            header = "r [Mpc.h-1]     monopole xi0    monopole error sigma_xi0    dipole xi1    dipole error sigma_xi1    quadrupole xi2    quadrupole error sigma_xi2    hexadecapole xi4    hexadecapole error sigma_xi4"
            txt = np.transpose(np.stack([r_array,monopole,error_monopole,dipole,error_dipole,quadrupole,error_quadrupole,hexadecapole,error_hexadecapole]))
        np.savetxt(save_txt,txt,header=header,delimiter="    ")

    return(fig,ax,r_array,monopole,dipole,quadrupole,hexadecapole)



def plot_multipole(ax,r_array,monopole,dipole,quadrupole,hexadecapole,
                   error_monopole,error_dipole,error_quadrupole,
                   error_hexadecapole,monopole_division=False,set_label=True):
    if(monopole_division):
        if(error_monopole is None):
            ax[0].plot(r_array,monopole)
            ax[1].plot(r_array,dipole/monopole)
            ax[2].plot(r_array,quadrupole/monopole)
            ax[3].plot(r_array,hexadecapole/monopole)
        else:
            ax[0].errorbar(r_array,monopole/monopole,error_monopole/monopole)
            ax[1].errorbar(r_array,dipole/monopole,error_dipole/monopole)
            ax[2].errorbar(r_array,quadrupole/monopole,error_quadrupole/monopole)
            ax[3].errorbar(r_array,hexadecapole/monopole,error_hexadecapole/monopole)
    else:
        if(error_monopole is None):
            ax[0].plot(r_array,monopole)
            ax[1].plot(r_array,dipole)
            ax[2].plot(r_array,quadrupole)
            ax[3].plot(r_array,hexadecapole)
        else:
            ax[0].errorbar(r_array,monopole,error_monopole)
            ax[1].errorbar(r_array,dipole,error_dipole)
            ax[2].errorbar(r_array,quadrupole,error_quadrupole)
            ax[3].errorbar(r_array,hexadecapole,error_hexadecapole)
    if(set_label):
        if(monopole_division):
            ax[1].set_ylabel(r"$\xi^{vg}_{1}$"+"/"+r"$\xi^{vg}_{0}$", fontsize=13)
            ax[2].set_ylabel(r"$\xi^{vg}_{2}$"+"/"+r"$\xi^{vg}_{0}$", fontsize=13)
            ax[3].set_ylabel(r"$\xi^{vg}_{4}$"+"/"+r"$\xi^{vg}_{0}$", fontsize=13)
        else:
            ax[1].set_ylabel("dipole " + r"$\xi^{vg}_{1}$", fontsize=15)
            ax[2].set_ylabel("quadrupole " + r"$\xi^{vg}_{2}$", fontsize=15)
            ax[3].set_ylabel("hexadecapole " + r"$\xi^{vg}_{4}$", fontsize=15)

        ax[0].set_ylabel("monopole " + r"$\xi^{vg}_{0}$", fontsize=15)
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




def compute_and_plot_multipole_comparison(file_xi1,file_xi2,nameout,supress_first_pixels=0,
                                          error_bar1=None,error_bar2=None,save_txt1=None,
                                          save_txt2=None,file_xi_optional=None,
                                          error_bar_optional=None,legend=None,
                                          multipole_method="rect",monopole_division=False,
                                          optional_monopole_normalization=True):
    (fig,ax,r_array1,monopole1,dipole1,quadrupole1,hexadecapole1) = compute_and_plot_multipole(file_xi1,None,supress_first_pixels=supress_first_pixels,
                                          save_txt=save_txt1,error_bar=error_bar1,
                                          multipole_method=multipole_method,
                                          monopole_division=monopole_division,
                                          set_label=True)
    (fig,ax,r_array2,monopole2,dipole2,quadrupole2,hexadecapole2) = compute_and_plot_multipole(file_xi2,None,supress_first_pixels=supress_first_pixels,
                                          save_txt=save_txt2,error_bar=error_bar2,
                                          multipole_method=multipole_method,
                                          second_plot=(fig,ax),
                                          monopole_division=monopole_division,
                                          set_label=False)
    if(file_xi_optional is not None):
        for i in range(len(file_xi_optional)):

            xcorr = xcorr_objects.CrossCorr.init_from_fits(file_xi_optional[i],exported=True,supress_first_pixels=supress_first_pixels)
            (r,mu,da) =  xcorr.r_array,xcorr.mu_array,xcorr.xi_array
            r_array = np.mean(r,axis=1)
            if(optional_monopole_normalization):
                (monopole,dipole,quadrupole,hexadecapole) = get_poles(mu,da,multipole_method)
                monopole_ratio = interp1d(r_array1,monopole1)(r_array) - monopole
                # mask = np.abs(monopole) <= 10**-3
                # monopole_ratio[mask] = 1.0
                for j in range(da.shape[1]):
                    da[:,j] = da[:,j] + monopole_ratio
            (monopole,dipole,quadrupole,hexadecapole) = get_poles(mu,da,multipole_method)
            if(error_bar_optional is not None):
                (error_monopole,error_dipole,error_quadrupole,error_hexadecapole)=get_error_bars_from_no_export(error_bar_optional[i],multipole_method,supress_first_pixels=supress_first_pixels)
            else:
                (error_monopole,error_dipole,error_quadrupole,error_hexadecapole)= None,None,None,None

            plot_multipole(ax,r_array,monopole,dipole,quadrupole,hexadecapole,
                           error_monopole,error_dipole,error_quadrupole,
                           error_hexadecapole,monopole_division=monopole_division,
                           set_label=False)
    ax[0].legend(legend)
    fig.savefig(f"{nameout}.pdf",format="pdf")



def plot_2d(file_xi,supress_first_pixels=0,**kwargs):
    xcorr = xcorr_objects.CrossCorr.init_from_fits(file_xi,exported=True,
                                                   supress_first_pixels=supress_first_pixels)
    xcorr.plot_2d(**kwargs)





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
