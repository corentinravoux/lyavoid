from lyavoid import xcorr
import os 


delta_path  = "/local/home/cravoux/Documents/Crosscorr_void_lya/Saclay_Mocks/deltas/box_cartesian/linear_growth"
void_name = "catalog_voids_mocks_Clusters_SPHERICAL_-0.8threshold_-0.7average_4rmin_ITERATION_deletion_cartesian_RADEC_fakeqso.fits"
void_path = "/local/home/cravoux/Documents/Crosscorr_void_lya/Saclay_Mocks/tomo/box/linear_growth/delta_m_smooth5/voids"
void_catalog = os.path.join(void_path,void_name)




z_evol_del = 2.9
z_ref = 2.25
z_min_obj = None
z_max_obj = None
z_evol_obj = 1.
mu_max = 1.
mu_min = -1.
r_min =0.
r_max = 50.
nbins_mu = 30
nbins_r = 15
weights_voids = 1.0
multipole_method = "trap"

xcorr_options = {"z_evol_del" : z_evol_del,"z_ref" : z_ref,"z_min_obj" : z_min_obj,"z_min_obj" : z_min_obj,
                "z_evol_obj" : z_evol_obj,"mu_max" : mu_max,"mu_min" : mu_min,"r_min" :r_min,"r_max" : r_max,
                "nbins_mu" : nbins_mu,"nbins_r" : nbins_r,"weights_voids" : weights_voids}



save_corr = "correlation.fits"
nameout = "multipoles.pdf"
nametxt = "multipoles.txt"



if __name__ == "__main__":
    xcf = xcorr.create_xcf(delta_path,void_catalog,xcorr_options)
    (r,mu,xi) = xcorr.xcorr_cartesian(xcf,save_corr=save_corr)
    xcorr.compute_and_plot_multipole_cartesian(mu,r,xi,save_plot = nameout,save_txt=nametxt,multipole_method = multipole_method)
