import fitsio
import numpy as np
import glob,os
from lyavoid import xcorr_objects
from lslyatomo import tomographic_objects
lambdaLy = 1215.673123130217



# def run_cross_corr_picca(picca_path,dict_picca):
    # import subprocess
#     exec_command = os.path.join(picca_path,"picca_xcf.py")
#     list_exec = [f"--{key} {value}" for (key,value) in dict_picca.items()]
#     print([exec_command] + list_exec)
#     subprocess.call([exec_command] + list_exec)



def create_xcf(delta_path,void_catalog,xcorr_options):
    xcf = xcorr_options
    read_voids(xcf,void_catalog)
    read_deltas(xcf,delta_path)
    fill_neighs(xcf)
    return(xcf)


def read_voids(xcf,void_catalog):
    void = tomographic_objects.VoidCatalog.init_from_fits(void_catalog)
    if(xcf["z_max_obj"] is not None):
        void.cut_catalog(coord_min=(-np.inf,-np.inf,xcf["z_min_obj"]))
    if(xcf["z_max_obj"] is not None):
        void.cut_catalog(coord_max=(np.inf,np.inf,xcf["z_max_obj"]))
    # void.weights = void.weights * (((1 + void_coords[:,4])/(1+xcf["z_ref"]))**(xcf["z_evol_obj"]-1))
    xcf["voids"]=void
    print('..done. Number of voids:', void.coord.shape[0])



def read_deltas(xcf,delta_path):
    # CR - To replace when lslyatomo is used
    print('Reading deltas from ', delta_path)
    deltas = []
    delta_names = glob.glob(os.path.join(delta_path,"delta-*.fits"))
    for delta_name in delta_names:
        delta = fitsio.FITS(delta_name)
        for i in range(1,len(delta)):
            delta_dict = {}
            header = delta[i].read_header()
            delta_dict["x"] = header["X"]
            delta_dict["y"] = header["Y"]
            delta_dict["z"] = delta[i]["Z"][:]
            delta_dict["delta"] = delta[i]["DELTA"][:]
            redshift = ((10**delta[i]["LOGLAM"][:] / lambdaLy)-1)
            delta_dict["weights"] = delta[i]["WEIGHT"][:] *(((1+redshift)/(1+xcf["z_ref"]))**(xcf["z_evol_del"]-1))
            deltas.append(delta_dict)
    xcf["deltas"] = deltas
    print('..done. Number of deltas:', len(deltas), 'Number of pixels:', len(deltas)*len(redshift))



def fill_neighs(xcf):
    print('Filling neighbors..')
    deltas = xcf["deltas"]
    void = xcf["voids"]
    r_max = xcf["r_max"]
    for d in deltas:
        x,y,z = d["x"],d["y"],d["z"]
        mask = np.sqrt((void.coord[:,0]-x)**2 + (void.coord[:,1]-y)**2) <= r_max
        mask &= void.coord[:,2]  <= np.max(z) + r_max
        mask &= void.coord[:,2]  >= np.min(z) + r_max
        voids = np.transpose(np.stack([void.coord[:,0][mask],void.coord[:,1][mask],void.coord[:,2][mask],void.weights[:][mask]]))
        d["neighbors"] = voids
    print('..done')




def fast_xcf(xcf,x1,y1,z1,w1,d1,x2,y2,z2,w2):
    x1_array,y1_array = np.full(z1.shape,x1),np.full(z1.shape,y1)
    rt = np.sqrt((x1_array[:,None] - x2)**2 + (y1_array[:,None] - y2)**2)
    rp = (z2 - z1[:,None])
    r = np.sqrt(rp**2+rt**2)
#    mask = r == 0
#    mu = np.zeros(r.shape)
#    mu[mask] = np.nan
#    mu[~mask] = rp[~mask]/r[~mask]
    mu = rp/r

#    z = (z1[:,None]+z2)/2

    we = w1[:,None]*w2
    wde = (w1*d1)[:,None]*w2
    w = (r>xcf["r_min"]) & (r<xcf["r_max"]) & (mu>xcf["mu_min"]) & (mu<xcf["mu_max"])
    r = r[w]
    mu = mu[w]
#    z  = z[w]
    we = we[w]
    wde = wde[w]


    br = ((r-xcf["r_min"])/(xcf["r_max"]-xcf["r_min"])*xcf["nbins_r"]).astype(int)
    bmu = ((mu-xcf["mu_min"])/(xcf["mu_max"]-xcf["mu_min"])*xcf["nbins_mu"]).astype(int)

    bins = bmu + xcf["nbins_mu"]*br

    rebin_xi = np.bincount(bins,weights=wde)
    rebin_weight = np.bincount(bins,weights=we)
    rebin_r = np.bincount(bins,weights=r*we)
    rebin_mu = np.bincount(bins,weights=mu*we)
#    rebin_z = np.bincount(bins,weights=z*we)
    rebin_num_pairs = np.bincount(bins,weights=(we>0.))

    return (rebin_weight, rebin_xi, rebin_r, rebin_mu, rebin_num_pairs)




def xcorr_cartesian(xcf,save_corr=None):
    nr = xcf["nbins_r"]
    nmu = xcf["nbins_mu"]
    xi = np.zeros(nr*nmu)
    weights = np.zeros(nr*nmu)
    r = np.zeros(nr*nmu)
    mu = np.zeros(nr*nmu)
#    z = np.zeros(nr*nmu)
    num_pairs = np.zeros(nr*nmu, dtype=np.int64)
    deltas = xcf["deltas"]
    for d in deltas:
        voids = d["neighbors"]
        if (len(voids) != 0):
            delta_values = d["delta"]
            x,y,z = d["x"],d["y"],d["z"]
            weights_deltas =  d["weights"]
            x_voids = voids[:,0]
            y_voids = voids[:,1]
            z_voids = voids[:,2]
            weights_voids = voids[:,3]
            (rebin_weight, rebin_xi, rebin_r, rebin_mu, rebin_num_pairs) =  fast_xcf(xcf,x,y,z,weights_deltas,delta_values,x_voids,y_voids,z_voids,weights_voids)
            xi[:len(rebin_xi)]+=rebin_xi
            weights[:len(rebin_weight)]+=rebin_weight
            r[:len(rebin_r)]+=rebin_r
            mu[:len(rebin_mu)]+=rebin_mu
#           z[:len(rebin_z)]+=rebin_z
            num_pairs[:len(rebin_num_pairs)]+=rebin_num_pairs.astype(int)

    w = weights>0
    xi[w]/=weights[w]
    r[w]/=weights[w]
    mu[w]/=weights[w]

    if(save_corr is not None):
        xcorr = xcorr_objects.CrossCorr(name=save_corr,mu_array=mu,r_array=r,xi_array=xi,exported=True)
        xcorr.write(xcf,weights)

#    z[w]/=weights[w]
    r = r.reshape(nr, nmu)
    mu = mu.reshape(nr, nmu)
    xi = xi.reshape(nr, nmu)
    return(r,mu,xi)





def xcorr(delta_path,void_catalog,xcorr_options,save_corr=None):
    if(save_corr is not None):
        if(os.path.isfile(save_corr)):
            print("Cross-correlation already computed")
            corr = xcorr_objects.CrossCorr.init_from_fits(save_corr,exported=True)
            return(corr.r_array,corr.mu_array,corr.xi_array)
    xcf = create_xcf(delta_path,void_catalog,xcorr_options)
    return(xcorr_cartesian(xcf,save_corr=save_corr))











    ###