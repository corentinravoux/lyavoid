from lyavoid import delta_creation
import os
pwd = os.getcwd()



box_DM_name = "/local/home/cravoux/Documents/Crosscorr_void_lya/Saclay_Mocks/tomo/box/linear_growth/delta_m_smooth5/dm_map_from_mocks.bindelta_m_smooth5"
shape_map = (6360, 180, 834)
los_selection = {"method":"equidistant", "rebin" : 4}
nb_files = 100
max_list_name = "/local/home/cravoux/Documents/Crosscorr_void_lya/Saclay_Mocks/tomo/box/linear_growth/delta_m_smooth5/list_of_maximums_of_data_cube.pickle"
Omega_m =  0.3147
map_size = (6355.584501234485, 180.6003008627381, 834.0181249831885)

mode="cartesian"

if __name__ == "__main__":
    delta_generator = delta_creation.DeltaGenerator(pwd,shape_map,box_DM_name,los_selection,nb_files,mode=mode)
    delta_generator.create_deltas_from_cube(map_size,max_list_name,Omega_m)
