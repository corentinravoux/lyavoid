from scipy import integrate
from lyavoid import xcorr_objects,multipole
import numpy as np
from iminuit import Minuit

class CrossCorrModel(object):

    def __init__(self,model_name=None,xcorr_in=None,model=None):

        self.model_name = model_name
        self.xcorr_in = xcorr_in
        self.model = model


    @classmethod
    def init_by_xcorr(cls,model_name,xcorr_in_name,xcorr_in_type,
                      nbins_mu=None,xcorr_in_class_name=None):
        if(xcorr_in_type == "multipole"):
            multipole_object = xcorr_objects.Multipole.init_from_fits(xcorr_in_name)
            xcorr_in = multipole_object.xcorr_from_monopole(xcorr_in_class_name,nbins_mu)
        elif(xcorr_in_type == "xcorr"):
            xcorr_in = xcorr_objects.CrossCorr.init_from_fits(xcorr_in_name,supress_first_pixels=0)

        r,mu = xcorr_in.r_array,xcorr_in.mu_array
        xi_array = xcorr_in.xi_array
        xcorr_class = cls(model_name=model_name,xcorr_in=xcorr_in)
        xcorr_class.model = xcorr_class.return_model(r,mu,xi_array)
        return(xcorr_class)



    def return_model(self,r,mu,xi_array):
        if(self.model_name == "linear"):
            xcorr_model = SimpleLinearModel()
        elif(self.model_name == "linear2"):
            xcorr_model = SimpleLinearModel2()
        elif(self.model_name == "linear3"):
            xcorr_model = SimpleLinearModel3()
        return(xcorr_model.model(r,mu,xi_array))



    def write_model(self,output_name,parameters):
        xcorr_out = self.model(**parameters)
        xcorr_out.name = output_name
        xcorr_out.z_array = np.zeros(xcorr_out.xi_array.shape)
        xcorr_out.write()


class SimpleLinearModel(object):
    def __init__(self):
        self.model_name = "SimpleLinearModel"

    def model(self,r,mu,xi_array):

        def model_function(b_v=1,b_a=1,beta_v=0,beta_a=0):
            xi = b_v * b_a * (1 + beta_v * mu**2) * (1 + beta_a * mu**2) * xi_array
            xcorr = xcorr_objects.CrossCorr(mu_array=mu,r_array=r,xi_array=xi)
            return(xcorr)

        return(model_function)



class SimpleLinearModel2(object):
    def __init__(self):
        self.model_name = "SimpleLinearModel2"

    def model(self,r,mu,xi_array):

        def model_function(parameters):
            b = parameters["b"]
            beta = parameters["beta"]
            xi = b**2 * (1 + beta * mu**2)**2 * xi_array
            xcorr = xcorr_objects.CrossCorr(mu_array=mu,r_array=r,xi_array=xi)
            return(xcorr)

        return(model_function)






class SimpleLinearModel3(object):
    def __init__(self):
        self.model_name = "SimpleLinearModel3"

    def model(self,r,mu,xi_array):

        Xi_bar = np.zeros(xi_array.shape)
        for i in range(len(r)):
            mask = (r <= r[i]) & (r > 0)
            integrand = xi_array[mask] * r[mask]**2
            integral = integrate.simps(integrand,r[mask],axis=0)
            Xi_bar[i,:] = integral * 3 / (r[i]**3)

        def model_function(parameters):
            beta = parameters["beta"]
            xi = xi_array + (beta/3) * Xi_bar + beta*mu**2*(xi_array -Xi_bar)
            xcorr = xcorr_objects.CrossCorr(mu_array=mu,r_array=r,xi_array=xi)
            return(xcorr)


        return(model_function)







class Fitter(object):
    def __init__(self,pwd,minuit_parameters):
        self.pwd = pwd
        self.minuit_parameters = minuit_parameters


    # def custom_least_squares(self,model,data_x,data_y,data_yerr,multipole_method):
    #
    #
    #     def cost(b_v=1,b_a=1,beta_v=0,beta_a=0):
    #         xcorr_model = model(b_v=b_v,b_a=b_a,beta_v=beta_v,beta_a=beta_a)
    #
    #         r_array_model = np.nanmean(xcorr_model.r_array,axis=1)
    #         monopole = np.interp(data_x,r_array_model,monopole)
    #         quadrupole = np.interp(data_x,r_array_model,quadrupole)
    #         ym = np.array([monopole,quadrupole])
    #         z = (data_y - ym) / data_yerr
    #         return np.nansum(z ** 2)
    #     return(cost)



    def custom_least_squares_multipole(self,model,data_x,data_y,data_yerr,multipole_method):


        def cost(b_v=1,b_a=1,beta_v=0,beta_a=0):
            xcorr_model = model(b_v=b_v,b_a=b_a,beta_v=beta_v,beta_a=beta_a)
            (monopole,
             dipole,
             quadrupole,
             hexadecapole) = multipole.get_poles(xcorr_model.mu_array,
                                                 xcorr_model.xi_array,
                                                 multipole_method)
            r_array_model = np.nanmean(xcorr_model.r_array,axis=1)
            monopole = np.interp(data_x,r_array_model,monopole)
            quadrupole = np.interp(data_x,r_array_model,quadrupole)
            ym = np.array([monopole,quadrupole])
            z = (data_y - ym) / data_yerr
            return np.nansum(z ** 2)
        return(cost)



    def run_migrad(self,minuit,ncall=1000):
        return(minuit.migrad(ncall,resume=True))

    def run_minos(self,minuit,sigma,ncall=1000,var_minos=None):
        return(minuit.minos(var=var_minos,sigma=sigma,ncall=ncall))

    def run_hesse(self,minuit):
        return(minuit.hesse())



    def fit_multipole(self,xcorr_model,
                      xcorr_fit_name,
                      xcorr_error_name,
                      output_name,
                      supress_first_pixels=0,
                      multipole_method="simps",
                      fix_args=None):

        xcorr_fit_class = xcorr_objects.CrossCorr.init_from_fits(xcorr_fit_name,
                                                                 supress_first_pixels=supress_first_pixels)
        (monopole,dipole,quadrupole,hexadecapole) = multipole.get_poles(xcorr_fit_class.mu_array,
                                                                        xcorr_fit_class.xi_array,
                                                                        multipole_method)

        (error_monopole,
         error_dipole,
         error_quadrupole,
         error_hexadecapole)=multipole.get_error_bars_from_no_export(xcorr_error_name,
                                                                     multipole_method,
                                                                     supress_first_pixels=supress_first_pixels)
        data_x = np.nanmean(xcorr_fit_class.r_array,axis=1)
        data_y = np.array([monopole,quadrupole])
        data_yerr = np.array([error_monopole,error_quadrupole])
        cost_function = self.custom_least_squares_multipole(xcorr_model.model,data_x,data_y,data_yerr,multipole_method)

        minuit = Minuit(cost_function,**self.minuit_parameters)
        if(fix_args is not None):
            for i in range(len(fix_args)):
                minuit.fixed[fix_args[i]] = True


        print(self.run_migrad(minuit))
        # self.run_hesse(minuit)
        #run_minos(minuit,sigma,ncall=ncall,var_minos=var_minos)
        xcorr_model.write_model(output_name,dict(minuit.values))
        return(minuit)
