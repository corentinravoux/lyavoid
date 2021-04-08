from scipy import integrate
from lyavoid import xcorr_objects
import numpy as np

class CrossCorrModel(object):

    def __init__(self,model_name=None):

        self.model_name = model_name



    def return_model(self,r,mu,real_space_correlation_function):
        if(self.model_name == "linear"):
            xcorr_model = SimpleLinearModel()
        elif(self.model_name == "linear2"):
            xcorr_model = SimpleLinearModel2()
        elif(self.model_name == "linear3"):
            xcorr_model = SimpleLinearModel3()
        return(xcorr_model.model(r,mu,real_space_correlation_function))

    def compute_model(self,real_space_file,output_name,parameters,monopole_only=False,nbins_mu=None,xcorr_name=None):
        if(monopole_only):
            multipole = xcorr_objects.Multipole.init_from_txt(real_space_file)
            xcorr = multipole.xcorr_from_monopole(xcorr_name,nbins_mu)
        else:
            xcorr = xcorr_objects.CrossCorr.init_from_fits(real_space_file,exported=True,supress_first_pixels=0)

        r,mu = xcorr.r_array,xcorr.mu_array
        xi_array = xcorr.xi_array
        model = self.return_model(r,mu,xi_array)
        xi_array_out = model(parameters)
        xcorr.xi_array = xi_array_out
        xcorr.name = output_name
        xcorr.write()




class SimpleLinearModel(object):
    def __init__(self):
        self.model_name = "SimpleLinearModel"

    def model(self,r,mu,xi_array):

        def model_function(parameters):
            b = parameters["b"]
            beta = parameters["beta"]
            Xi = b**2 * (1 + beta * mu**2)**2 * xi_array
            return(Xi)

        return(model_function)


class SimpleLinearModel2(object):
    def __init__(self):
        self.model_name = "SimpleLinearModel2"

    def model(self,r,mu,xi_array):

        def model_function(parameters):
            bv = parameters["bv"]
            ba = parameters["ba"]
            betav = parameters["betav"]
            betaa = parameters["betaa"]
            Xi = bv * ba * (1 + betav * mu**2) * (1 + betaa * mu**2) * xi_array
            return(Xi)

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
            Xi = xi_array + (beta/3) * Xi_bar + beta*mu**2*(xi_array -Xi_bar)
            return(Xi)


        return(model_function)



# class Fitter()
