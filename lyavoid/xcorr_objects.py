import fitsio
import matplotlib.pyplot as plt
from lyavoid import utils
import numpy as np
from scipy.interpolate import griddata
from scipy import integrate



class CrossCorr(object):

    def __init__(self,name=None,mu_array=None,r_array=None,xi_array=None,z_array=None,exported=True,rmu=True):

        self.name = name
        self.mu_array = mu_array
        self.r_array = r_array
        self.xi_array = xi_array
        self.z_array = z_array
        self.exported = exported
        self.rmu = rmu


    @classmethod
    def init_from_fits(cls,name,exported=True,supress_first_pixels=0):
        #- Read correlation function and covariance matrix
        h = fitsio.FITS(name)
        if(exported):
            xi_array = h["COR"]['DA'][:]
            r_array = h["COR"]['R'][:]
            mu_array = h["COR"]['MU'][:]
            z_array = h["COR"]['Z'][:]
            hh = h["COR"].read_header()
        else:
            xi_array = h["COR"]['DA'][:]
            r_array = h["ATTRI"]['R'][:]
            mu_array = h["ATTRI"]['MU'][:]
            z_array = h["ATTRI"]['Z'][:]
            hh = h["ATTRI"].read_header()
        nr = hh['NR']
        nmu = hh['NMU']
        h.close()
        r_array = r_array.reshape(nr, nmu)[supress_first_pixels:,:]
        mu_array = mu_array.reshape(nr, nmu)[supress_first_pixels:,:]
        z_array = z_array.reshape(nr, nmu)[supress_first_pixels:,:]
        if(exported):
            xi_array = xi_array.reshape(nr, nmu)[supress_first_pixels:,:]
        else:
            xi_array = xi_array.reshape(len(xi_array),nr, nmu)[:,supress_first_pixels:,:]
        return(cls(name=name,mu_array=mu_array,r_array=r_array,xi_array=xi_array,z_array=z_array,exported=exported))


    def write(self,xcf_param=None):

        out = fitsio.FITS(self.name,'rw',clobber=True)
        if(self.exported):
            nbins_r = self.xi_array.shape[0]
            nbins_mu = self.xi_array.shape[1]
        else:
            nbins_r = self.xi_array.shape[1]
            nbins_mu = self.xi_array.shape[2]
        head =[ {'name':'NR','value': nbins_r,'comment':'Number of bins in r'},
                {'name':'NMU','value': nbins_mu,'comment':'Number of bins in mu'}
                ]
        if(xcf_param is not None):
            head = head + [ {'name':'RMIN','value':xcf_param["r_min"],'comment':'Minimum r [h^-1 Mpc]'},
                    {'name':'RMAX','value':xcf_param["r_max"],'comment':'Maximum r [h^-1 Mpc]'},
                    {'name':'MUMIN','value':xcf_param["mu_min"],'comment':'Minimum mu = r_para/r'},
                    {'name':'MUMAX','value':xcf_param["mu_max"],'comment':'Maximum mu = r_para/r'},
                    # {'name':'NR','value': xcf_param["nbins_r"],'comment':'Number of bins in r'},
                    # {'name':'NMU','value':xcf_param["nbins_mu"],'comment':'Number of bins in mu'},
                ]
        out.write([self.r_array,self.mu_array,self.xi_array,self.z_array],names=['R','MU','DA','Z'],
                comment=['r','mu = r_para/r','xi','redshift'],
                units=['h^-1 Mpc','','',''],
                header=head,extname='COR')
        out.close()


    def write_no_export(self,xcf_param=None,weights=None):

        out = fitsio.FITS(self.name,'rw',clobber=True)
        head = [ {'name':'RMIN','value':xcf_param["r_min"],'comment':'Minimum r [h^-1 Mpc]'},
                {'name':'RMAX','value':xcf_param["r_max"],'comment':'Maximum r [h^-1 Mpc]'},
                {'name':'MUMIN','value':xcf_param["mu_min"],'comment':'Minimum mu = r_para/r'},
                {'name':'MUMAX','value':xcf_param["mu_max"],'comment':'Maximum mu = r_para/r'},
                {'name':'NR','value': xcf_param["nbins_r"],'comment':'Number of bins in r'},
                {'name':'NMU','value':xcf_param["nbins_mu"],'comment':'Number of bins in mu'},
            ]
        out.write([self.r_array,self.mu_array,self.z_array],names=['R','MU','Z'],
                comment=['r','mu = r_para/r','redshift'],
                units=['h^-1 Mpc','',''],
                header=head,extname='ATTRI')

        head2 = [{'name':'HLPXSCHM','value':'RING','comment':'Healpix scheme'}]
        out.write([weights,self.xi_array],names=['WE','DA'],
            comment=['Sum of weight', 'Correlation'],
            header=head2,extname='COR')
        out.close()


    def plot_2d(self,**kwargs):
        rmu = utils.return_key(kwargs,"rmu",True)
        if(not(rmu)):
            self.switch()
        vmax = utils.return_key(kwargs,"vmax",None)
        vmin = utils.return_key(kwargs,"vmin",None)
        radius_multiplication_power = utils.return_key(kwargs,"r_power",0)
        title_add = ""
        if(radius_multiplication_power !=0):
            title_add = r"$\times r^{" + str(radius_multiplication_power) + "}$"
        colobar_legend = utils.return_key(kwargs,"cbar",r"Cross-correlation void-lya $\xi_{vl}$" + title_add )
        # plt.figure(figsize=(10,16))
        plt.figure()
        if(not(rmu)):
            extent = (np.min(self.mu_array),np.max(self.mu_array),np.min(self.r_array),np.max(self.r_array))
            xv, yv = np.meshgrid(np.linspace(extent[0],extent[1],self.mu_array.shape[1]),np.linspace(extent[2],extent[3],self.r_array.shape[0]))
            grid = griddata(np.array([np.ravel(self.mu_array),np.ravel(self.r_array)]).T, np.ravel(self.xi_array), (xv, yv), method='nearest')
            plt.imshow(grid, extent=extent,vmin=vmin,vmax=vmax)

            plt.xlabel(r"$r_{\bot}$")
            plt.ylabel(r"$r_{\parallel}$")
        else:
            plt.pcolor(self.mu_array , self.r_array, self.xi_array * (self.r_array**radius_multiplication_power),vmin=vmin,vmax=vmax)
            plt.xlabel(r"$\mu$")
            plt.ylabel(r"$r$")
        cbar = plt.colorbar()
        cbar.set_label(colobar_legend)
        if(not(rmu)):
            self.switch()
        name = utils.return_key(kwargs,"name","2d_plot_cross_corr")
        plt.savefig(f"{name}.pdf",format="pdf")





    def switch(self):
        if(self.rmu):
            self.r_array, self.mu_array = self.mu_array * self.r_array, self.r_array * np.sqrt(1 - self.mu_array**2)
            self.rmu = False
        else:
            self.r_array, self.mu_array = np.sqrt(self.mu_array**2 + self.r_array**2), self.r_array / np.sqrt(self.mu_array**2 + self.r_array**2)
            self.rmu = True


# def test_switch(r,mu,xi):
#     corr = CrossCorr(mu_array=mu,r_array=r,xi_array=xi)
#     corr.switch().switch()
#     assert(np.mean(corr.r_array - r) < 10**-10)
#     assert(np.mean(corr.mu_array - mu) < 10**-10)






class CrossCorrModel(object):

    def __init__(self,model_name=None,exported=True,supress_first_pixels=0):

        self.model_name = model_name
        self.exported = exported
        self.supress_first_pixels = supress_first_pixels



    def return_model(self,r,mu,real_space_correlation_function):
        if(self.model_name == "linear"):
            xcorr_model = SimpleLinearModel()
        elif(self.model_name == "linear2"):
            xcorr_model = SimpleLinearModel2()
        elif(self.model_name == "linear3"):
            xcorr_model = SimpleLinearModel3()
        return(xcorr_model.model(r,mu,real_space_correlation_function))

    def compute_model(self,real_space_file,output_name,parameters):

        xcorr = CrossCorr.init_from_fits(real_space_file,exported=True,supress_first_pixels=0)

        r,mu = xcorr.r_array,xcorr.mu_array
        real_space_correlation_function = xcorr.xi_array
        model = self.return_model(r,mu,real_space_correlation_function)
        redshift_space_correlation_function = model(parameters)
        xcorr.xi_array = redshift_space_correlation_function
        xcorr.name = output_name

        xcorr.write()




class SimpleLinearModel(object):
    def __init__(self):
        self.model_name = "SimpleLinearModel"

    def model(self,r,mu,real_space_correlation_function):

        def model_function(parameters):
            b = parameters["b"]
            beta = parameters["beta"]
            Xi = b**2 * (1 + beta * mu**2)**2 * real_space_correlation_function
            return(Xi)

        return(model_function)


class SimpleLinearModel2(object):
    def __init__(self):
        self.model_name = "SimpleLinearModel2"

    def model(self,r,mu,real_space_correlation_function):

        def model_function(parameters):
            bv = parameters["bv"]
            ba = parameters["ba"]
            betav = parameters["betav"]
            betaa = parameters["betaa"]
            Xi = bv * ba * (1 + betav * mu**2) * (1 + betaa * mu**2) * real_space_correlation_function
            return(Xi)

        return(model_function)




class SimpleLinearModel3(object):
    def __init__(self):
        self.model_name = "SimpleLinearModel2"

    def model(self,r,mu,real_space_correlation_function):

        Xi_bar = np.zeros(real_space_correlation_function.shape)
        for i in range(len(r)):
            mask = (r <= r[i]) & (r > 0)
            integrand = real_space_correlation_function[mask] * r[mask]**2
            integral = integrate.simps(integrand,r[mask],axis=0)
            Xi_bar[i,:] = integral * 3 / (r[i]**3)

        def model_function(parameters):
            beta = parameters["beta"]
            Xi = real_space_correlation_function + (beta/3) * Xi_bar + beta*mu**2*(real_space_correlation_function -Xi_bar)
            return(Xi)


        return(model_function)
