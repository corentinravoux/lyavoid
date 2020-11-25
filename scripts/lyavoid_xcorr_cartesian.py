from lyavoid import xcorr,multipole
import os 



void_method = "S"
par1 = -0.8
par2 = -0.7

if(void_method == "S"):
    void = f"Catalog_Clusters_SPHERICAL_{par1}threshold_{par2}average_4rmin_ITERATION_deletion.fits_cor"
    suffixe = f"_SPHERICAL_{par1}threshold_{par2}average_4rmin_ITERATION_deletion"
elif(void_method == "W"):
    void = f"Catalog_Clusters_WATERSHED_{par1}threshold_1.5dist_clusters_4rmin_CLUSTERS_deletion.fits_cor"
    suffixe = f"_WATERSHED_{par1}threshold_1.5dist_clusters_4rmin_CLUSTERS_deletion"



delta_path  = "/local/home/cravoux/Documents/Crosscorr_void_lya/Saclay_Mocks/deltas/box_cartesian/no_linear_growth/delta_m_smooth5_iso"
void_path = "/local/home/cravoux/Documents/Crosscorr_void_lya/Saclay_Mocks/tomo/box/no_linear_growth/delta_m_smoothed5mpc_iso"
void_catalog = os.path.join(void_path,void)




z_evol_del = 2.9
z_ref = 2.25
z_min_obj = None
z_max_obj = None
z_evol_obj = 1.
mu_max = 1.
mu_min = -1.
r_min =2
r_max = 50.
nbins_mu = 50
nbins_r = 25
weights_voids = 1.0
multipole_method = "rect"

xcorr_options = {"z_evol_del" : z_evol_del,"z_ref" : z_ref,"z_min_obj" : z_min_obj,"z_max_obj" : z_max_obj,
                "z_evol_obj" : z_evol_obj,"mu_max" : mu_max,"mu_min" : mu_min,"r_min" :r_min,"r_max" : r_max,
                "nbins_mu" : nbins_mu,"nbins_r" : nbins_r,"weights_voids" : weights_voids}



namecorr = f"correlation{suffixe}.fits"
nameout = f"multipoles{suffixe}.pdf"
nametxt = f"multipoles{suffixe}.txt"
name_2d = f"2d_plot{suffixe}"

plt_options_2d = {"rmu" : False,"name" : name_2d}

skip_calculation = False


if __name__ == "__main__":
    (r,mu,xi) = xcorr.xcorr(delta_path,void_catalog,xcorr_options,namecorr,skip_calculation=skip_calculation)
    multipole.compute_and_plot_multipole_cartesian(mu,r,xi,save_plot = nameout,save_txt=nametxt,multipole_method = multipole_method)
    multipole.plot_2d(mu,r,xi,**plt_options_2d)

