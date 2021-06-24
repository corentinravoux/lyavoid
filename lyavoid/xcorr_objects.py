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
                mu_array = h[attribut_name]['RP'][:]
                r_array = h[attribut_name]['RT'][:]
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

        if(self.rmu):
            head =[ {'name':'NR','value': nbins_r,'comment':'Number of bins in r'},
                    {'name':'NMU','value': nbins_mu,'comment':'Number of bins in mu'}
                    ]
            if(xcf_param is not None):
                head = head + [ {'name':'RMIN','value':xcf_param["r_min"],'comment':'Minimum r [h^-1 Mpc]'},
                        {'name':'RMAX','value':xcf_param["r_max"],'comment':'Maximum r [h^-1 Mpc]'},
                        {'name':'MUMIN','value':xcf_param["mu_min"],'comment':'Minimum mu = r_para/r'},
                        {'name':'MUMAX','value':xcf_param["mu_max"],'comment':'Maximum mu = r_para/r'},
                    ]
            name_out1 = 'R'
            name_out2 = 'MU'
            comment = ['r','mu = r_para/r','xi','redshift']
            unit = ['h^-1 Mpc','','','']

        else:
            head =[ {'name':'NP','value': nbins_r,'comment':'Number of bins in r'},
                    {'name':'NT','value': nbins_mu,'comment':'Number of bins in mu'}
                    ]
            if(xcf_param is not None):
                head = head + [ {'name':'RPMIN','value':xcf_param["rp_min"],'comment':'Minimum rp [h^-1 Mpc]'},
                        {'name':'RPMAX','value':xcf_param["rp_max"],'comment':'Maximum rp [h^-1 Mpc]'},
                        {'name':'RTMAX','value':xcf_param["rt_max"],'comment':'Maximum rt [h^-1 Mpc]'},
                    ]
            name_out1 = 'RT'
            name_out2 = 'RP'
            comment = ['rp','rt','xi','redshift']
            unit = ['h^-1 Mpc','h^-1 Mpc','','']

        out.write([self.r_array,self.mu_array,self.xi_array,self.z_array],names=[name_out1,name_out2,'DA','Z'],
                comment=comment,
                units=unit,
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
        if(((rmu==False)&self.rmu)|((self.rmu == False)&rmu)):
            self.switch()
        vmax = utils.return_key(kwargs,"vmax",None)
        vmin = utils.return_key(kwargs,"vmin",None)
        multiplicative_factor = utils.return_key(kwargs,"multiplicative_factor",None)
        if(multiplicative_factor is not None):
            self.xi_array = self.xi_array * multiplicative_factor
        radius_multiplication_power = utils.return_key(kwargs,"r_power",0)
        title_add = ""
        if(radius_multiplication_power !=0):
            title_add = r"$\times r^{" + str(radius_multiplication_power) + "}$"
        colobar_legend = utils.return_key(kwargs,"cbar",r"Cross-correlation void-lya $\xi_{vl}$" + title_add )
        plt.figure()
        if(rmu == False):
            rp_array = self.mu_array
            rt_array = self.r_array
            extent = (np.min(rt_array),np.max(rt_array),np.min(rp_array),np.max(rp_array))

            rtrt, rprp = np.meshgrid(np.linspace(extent[0],extent[1],rt_array.shape[1]),
                                     np.linspace(extent[2],extent[3],rp_array.shape[0]))
            xi_to_plot = self.xi_array * (np.sqrt(rp_array**2 + rt_array**2)**radius_multiplication_power)
            grid = griddata(np.array([np.ravel(rt_array),np.ravel(rp_array)]).T,
                            np.ravel(xi_to_plot),
                            (rtrt, rprp),
                            method='nearest')
            plt.imshow(grid, extent=extent,vmin=vmin,vmax=vmax)
            plt.xlabel(r"$r_{\bot}$")
            plt.ylabel(r"$r_{\parallel}$")
        else:

            # extent = (np.min(self.mu_array),np.max(self.mu_array),np.min(self.r_array),np.max(self.r_array))
            #
            # mumu, rr = np.meshgrid(np.linspace(extent[0],extent[1],self.mu_array.shape[1]),
            #                          np.linspace(extent[2],extent[3],self.r_array.shape[0]))
            # xi_to_plot = self.xi_array * (self.r_array**radius_multiplication_power)
            # grid = griddata(np.array([np.ravel(self.mu_array),np.ravel(self.r_array)]).T,
            #                 np.ravel(xi_to_plot),
            #                 (mumu, rr),
            #                 method='nearest')
            # plt.imshow(grid,vmin=vmin,vmax=vmax)
            #

            plt.pcolor(self.mu_array , self.r_array, self.xi_array * (self.r_array**radius_multiplication_power),vmin=vmin,vmax=vmax)
            plt.xlabel(r"$\mu$")
            plt.ylabel(r"$r$")
        cbar = plt.colorbar()
        cbar.set_label(colobar_legend)
        if(((rmu==False)&self.rmu)|((self.rmu == False)&rmu)):
            self.switch()
        name = utils.return_key(kwargs,"name","2d_plot_cross_corr")
        plt.savefig(f"{name}.pdf",format="pdf")





    def switch(self):
        if(self.rmu):
            self.mu_array, self.r_array = self.mu_array * self.r_array, self.r_array * np.sqrt(1 - self.mu_array**2)
            self.rmu = False
        else:
            self.r_array, self.mu_array = np.sqrt(self.mu_array**2 + self.r_array**2), self.mu_array / np.sqrt(self.mu_array**2 + self.r_array**2)
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
