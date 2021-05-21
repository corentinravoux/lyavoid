import fitsio
import matplotlib.pyplot as plt
from lyavoid import utils
import numpy as np
from scipy.stats import sem
from scipy.interpolate import griddata



class CrossCorr(object):

    def __init__(self,
                 name=None,
                 mu_array=None,
                 r_array=None,
                 xi_array=None,
                 z_array=None,
                 xi_error_array=None,
                 exported=True,
                 rmu=True):

        self.name = name
        self.mu_array = mu_array
        self.r_array = r_array
        self.xi_array = xi_array
        self.z_array = z_array
        self.xi_error_array = xi_error_array
        self.exported = exported
        self.rmu = rmu          # If True, r_array contains r and mu_array mu. If False, r_array contains rp (r_parallel) and mu_array rt (r_transverse)



    @classmethod
    def init_from_fits(cls,name,supress_first_pixels=0):
        with fitsio.FITS(name) as h:
            xi_array = h["COR"]['DA'][:]
            xi_error_array = None
            exported = True
            attribut_name = "COR"
            if("WE" in h["COR"].get_colnames()):
                exported=False
                attribut_name = "ATTRI"
            if(exported):
                if("CO" in h["COR"].get_colnames()):
                    xi_error_array = np.sqrt(np.diag(h["COR"]['CO'][:]))
            z_array = h[attribut_name]['Z'][:]
            hh = h[attribut_name].read_header()
            if('R' in h[attribut_name].get_colnames()):
                rmu = True
                r_array = h[attribut_name]['R'][:]
                mu_array = h[attribut_name]['MU'][:]
                nr = hh['NR']
                nmu = hh['NMU']
            elif('RT' in h[attribut_name].get_colnames()):
                rmu = False
                r_array = h[attribut_name]['RP'][:]
                mu_array = h[attribut_name]['RT'][:]
                nr = hh['NP']
                nmu = hh['NT']
        if(exported):
            xi_array = xi_array.reshape(nr, nmu)[supress_first_pixels:,:]
            if(xi_error_array is not None):
                xi_error_array = xi_error_array.reshape(nr, nmu)[supress_first_pixels:,:]
        else:
            xi_array = xi_array.reshape(len(xi_array),nr, nmu)[:,supress_first_pixels:,:]
            xi_error_array = sem(xi_array,axis=0)
        r_array = r_array.reshape(nr, nmu)[supress_first_pixels:,:]
        mu_array = mu_array.reshape(nr, nmu)[supress_first_pixels:,:]
        z_array = z_array.reshape(nr, nmu)[supress_first_pixels:,:]
        return(cls(name=name,mu_array=mu_array,r_array=r_array,xi_array=xi_array,z_array=z_array,exported=exported,xi_error_array=xi_error_array,rmu=rmu))


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
        if((not(rmu)&self.rmu)|(not(self.rmu)&rmu)):
            self.switch()
        vmax = utils.return_key(kwargs,"vmax",None)
        vmin = utils.return_key(kwargs,"vmin",None)
        radius_multiplication_power = utils.return_key(kwargs,"r_power",0)
        title_add = ""
        if(radius_multiplication_power !=0):
            title_add = r"$\times r^{" + str(radius_multiplication_power) + "}$"
        colobar_legend = utils.return_key(kwargs,"cbar",r"Cross-correlation void-lya $\xi_{vl}$" + title_add )
        plt.figure()
        if(not(rmu)):
            extent = (np.min(self.mu_array),np.max(self.mu_array),np.min(self.r_array),np.max(self.r_array))
            xv, yv = np.meshgrid(np.linspace(extent[0],extent[1],self.mu_array.shape[1]),np.linspace(extent[2],extent[3],self.r_array.shape[0]))
            xi_to_plot = self.xi_array * (np.sqrt(self.mu_array**2 + self.r_array**2)**radius_multiplication_power)
            grid = griddata(np.array([np.ravel(self.mu_array),np.ravel(self.r_array)]).T, np.ravel(xi_to_plot), (xv, yv), method='nearest')
            plt.imshow(grid, extent=extent,vmin=vmin,vmax=vmax)

            plt.xlabel(r"$r_{\bot}$")
            plt.ylabel(r"$r_{\parallel}$")
        else:
            plt.pcolor(self.mu_array , self.r_array, self.xi_array * (self.r_array**radius_multiplication_power),vmin=vmin,vmax=vmax)
            plt.xlabel(r"$\mu$")
            plt.ylabel(r"$r$")
        cbar = plt.colorbar()
        cbar.set_label(colobar_legend)
        if((not(rmu)&self.rmu)|(not(self.rmu)&rmu)):
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


class Multipole(object):

    def __init__(self,name=None,r_array=None,monopole=None,dipole=None,quadrupole=None,hexadecapole=None):

        self.name = name
        self.r_array = r_array
        self.monopole = monopole
        self.dipole = dipole
        self.quadrupole = quadrupole
        self.hexadecapole = hexadecapole


    @classmethod
    def init_from_txt(cls,name):

        multipole = np.loadtxt(name)
        r_array = multipole[:,0]
        monopole = multipole[:,1]
        dipole = multipole[:,2]
        quadrupole = multipole[:,3]
        hexadecapole = multipole[:,4]
        return(cls(name=name,r_array=r_array,monopole=monopole,dipole=dipole,quadrupole=quadrupole,hexadecapole=hexadecapole))


    def write_txt(self):
        header = "r [Mpc.h-1]     monopole xi0    dipole xi1    quadrupole xi2    hexadecapole xi4"
        txt = np.transpose(np.stack([self.r_array,self.monopole,self.dipole,self.quadrupole,self.hexadecapole]))
        np.savetxt(self.name,txt,header=header,delimiter="    ")




    def xcorr_from_monopole(self,xcorr_name,nbins_mu):
        mu_array = np.linspace(-1.0,1.0,nbins_mu)
        coord_xcorr =np.moveaxis(np.array(np.meshgrid(self.r_array,mu_array,indexing='ij')),0,-1)
        xi_array = np.array([self.monopole for i in range(nbins_mu)]).transpose()
        z_array = np.zeros(xi_array.shape)
        xcorr = CrossCorr(name=xcorr_name,mu_array=coord_xcorr[:,:,1],r_array=coord_xcorr[:,:,0],xi_array=xi_array,z_array=z_array,exported=True,rmu=True)
        xcorr.write()
        return(xcorr)
