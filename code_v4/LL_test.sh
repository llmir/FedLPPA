# # LL
# python -u test_LL.py --client client5 --num_classes 3 --in_chns 3  --root_path ../data/ODOC_h5 --img_class odoc --exp LL_bs_12/odoc_LL_block_0.01 --sup_type block &
# python -u test_LL.py --client client4 --num_classes 3 --in_chns 3 --root_path ../data/ODOC_h5 --img_class odoc --exp LL_bs_12/odoc_LL_keypoint_0.005 --sup_type keypoint &
# python -u test_LL.py --client client3 --num_classes 3 --in_chns 3 --root_path ../data/ODOC_h5 --img_class odoc --exp LL_bs_12/odoc_LL_scr_n_0.005 --sup_type scribble_noisy
# python -u test_LL.py --client client2 --num_classes 3 --in_chns 3 --root_path ../data/ODOC_h5 --img_class odoc --exp LL_bs_12/odoc_LL_scr_n_0.005 --sup_type scribble_noisy &
# python -u test_LL.py --client client1 --num_classes 3 --in_chns 3 --root_path ../data/ODOC_h5 --img_class odoc --exp LL_bs_12/odoc_LL_scr_0.01 --sup_type scribble &

# python -u test_LL.py --client client2 --num_classes 2 --in_chns 1 --root_path ../data/FAZ_h5 --img_class faz --exp LL_bs_12/faz_LL_keypoint_0.01 --sup_type keypoint
# python -u test_LL.py --client client3 --num_classes 2 --in_chns 1 --root_path ../data/FAZ_h5 --img_class faz --exp LL_bs_12/faz_LL_block_0.01 --sup_type block &
# python -u test_LL.py --client client4 --num_classes 2 --in_chns 1 --root_path ../data/FAZ_h5 --img_class faz --exp LL_bs_12/faz_LL_box_0.001 --sup_type box &
# python -u test_LL.py --client client5 --num_classes 2 --in_chns 1 --root_path ../data/FAZ_h5 --img_class faz --exp LL_bs_12/faz_LL_scr_0.003 --sup_type scribble&
# python -u test_LL.py --client client1 --num_classes 2 --in_chns 1 --root_path ../data/FAZ_h5 --img_class faz --exp LL_bs_12/faz_LL_scr_n_0.003 --sup_type scribble_noisy
# # FL
# python -u test_LL.py --client client1 --num_classes 3 --in_chns 3 --root_path ../data/ODOC_h5 --img_class odoc --exp FULL_bs_12/odoc_pce --sup_type mask
# python -u test_LL.py --client client2 --num_classes 3 --in_chns 3 --root_path ../data/ODOC_h5 --img_class odoc --exp FULL_bs_12/odoc_pce --sup_type mask&
# python -u test_LL.py --client client3 --num_classes 3 --in_chns 3 --root_path ../data/ODOC_h5 --img_class odoc --exp FULL_bs_12/odoc_pce --sup_type mask
# python -u test_LL.py --client client4 --num_classes 3 --in_chns 3 --root_path ../data/ODOC_h5 --img_class odoc --exp FULL_bs_12/odoc_pce --sup_type mask&
# python -u test_LL.py --client client5 --num_classes 3 --in_chns 3 --root_path ../data/ODOC_h5 --img_class odoc --exp FULL_bs_12/odoc_pce --sup_type mask

# python -u test_LL.py --client client1 --num_classes 2 --in_chns 1 --root_path ../data/FAZ_h5 --img_class faz --exp FULL_bs_12/faz_pce --sup_type mask
# python -u test_LL.py --client client2 --num_classes 2 --in_chns 1 --root_path ../data/FAZ_h5 --img_class faz --exp FULL_bs_12/faz_pce --sup_type mask&
# python -u test_LL.py --client client3 --num_classes 2 --in_chns 1 --root_path ../data/FAZ_h5 --img_class faz --exp FULL_bs_12/faz_pce --sup_type mask
# python -u test_LL.py --client client4 --num_classes 2 --in_chns 1 --root_path ../data/FAZ_h5 --img_class faz --exp FULL_bs_12/faz_pce --sup_type mask&
# python -u test_LL.py --client client5 --num_classes 2 --in_chns 1 --root_path ../data/FAZ_h5 --img_class faz --exp FULL_bs_12/faz_pce --sup_type mask

# CL
# python -u test_client4onemod.py --client client1 --num_classes 2 --in_chns 1 --root_path ../data/FAZ_h5 --img_class faz --exp CL_augmenzation_nomask/faz_pce_0.01 &
# python -u test_client4onemod.py --client client2 --num_classes 2 --in_chns 1 --root_path ../data/FAZ_h5 --img_class faz --exp CL_augmenzation_nomask/faz_pce_0.01 
# python -u test_client4onemod.py --client client3 --num_classes 2 --in_chns 1 --root_path ../data/FAZ_h5 --img_class faz --exp CL_augmenzation_nomask/faz_pce_0.01 &
# python -u test_client4onemod.py --client client4 --num_classes 2 --in_chns 1 --root_path ../data/FAZ_h5 --img_class faz --exp CL_augmenzation_nomask/faz_pce_0.01 
# python -u test_client4onemod.py --client client5 --num_classes 2 --in_chns 1 --root_path ../data/FAZ_h5 --img_class faz --exp CL_augmenzation_nomask/faz_pce_0.01 &

# python -u test_client4onemod.py --client client1 --num_classes 3 --in_chns 3 --root_path ../data/ODOC_h5 --img_class odoc --exp CL_augmenzation_nomask/odoc_pce_0.01  &
# python -u test_client4onemod.py --client client2 --num_classes 3 --in_chns 3 --root_path ../data/ODOC_h5 --img_class odoc --exp CL_augmenzation_nomask/odoc_pce_0.01  
# python -u test_client4onemod.py --client client3 --num_classes 3 --in_chns 3 --root_path ../data/ODOC_h5 --img_class odoc --exp CL_augmenzation_nomask/odoc_pce_0.01  &
# python -u test_client4onemod.py --client client4 --num_classes 3 --in_chns 3 --root_path ../data/ODOC_h5 --img_class odoc --exp CL_augmenzation_nomask/odoc_pce_0.01  
# python -u test_client4onemod.py --client client5 --num_classes 3 --in_chns 3 --root_path ../data/ODOC_h5 --img_class odoc --exp CL_augmenzation_nomask/odoc_pce_0.01 


# # CFL
# python -u test_client4onemod.py --client client1 --num_classes 2 --in_chns 1 --root_path ../data/FAZ_h5 --img_class faz --exp CFL_bs_12/faz_pce --sup_type mask&
# python -u test_client4onemod.py --client client2 --num_classes 2 --in_chns 1 --root_path ../data/FAZ_h5 --img_class faz --exp CFL_bs_12/faz_pce --sup_type mask
# python -u test_client4onemod.py --client client3 --num_classes 2 --in_chns 1 --root_path ../data/FAZ_h5 --img_class faz --exp CFL_bs_12/faz_pce --sup_type mask&
# python -u test_client4onemod.py --client client4 --num_classes 2 --in_chns 1 --root_path ../data/FAZ_h5 --img_class faz --exp CFL_bs_12/faz_pce --sup_type mask
# python -u test_client4onemod.py --client client5 --num_classes 2 --in_chns 1 --root_path ../data/FAZ_h5 --img_class faz --exp CFL_bs_12/faz_pce --sup_type mask&

# python -u test_client4onemod.py --client client1 --num_classes 3 --in_chns 3 --root_path ../data/ODOC_h5 --img_class odoc --exp CFL_bs_12/odoc_pce --sup_type mask&
# python -u test_client4onemod.py --client client2 --num_classes 3 --in_chns 3 --root_path ../data/ODOC_h5 --img_class odoc --exp CFL_bs_12/odoc_pce --sup_type mask
# python -u test_client4onemod.py --client client3 --num_classes 3 --in_chns 3 --root_path ../data/ODOC_h5 --img_class odoc --exp CFL_bs_12/odoc_pce --sup_type mask&
# python -u test_client4onemod.py --client client4 --num_classes 3 --in_chns 3 --root_path ../data/ODOC_h5 --img_class odoc --exp CFL_bs_12/odoc_pce --sup_type mask
# python -u test_client4onemod.py --client client5 --num_classes 3 --in_chns 3 --root_path ../data/ODOC_h5 --img_class odoc --exp CFL_bs_12/odoc_pce --sup_type mask


# # FL
# python -u test_client4onemod_FL.py --client client1 --num_classes 2 --in_chns 1 --root_path ../data/FAZ_h5 --img_class faz --exp faz/FL_WeaklySeg_pCE_EM_FedAvg &
# python -u test_client4onemod_FL.py --client client1 --num_classes 2 --in_chns 1 --root_path ../data/FAZ_h5 --img_class faz --exp faz/FL_WeaklySeg_pCE_FedAvg & 
# python -u test_client4onemod_FL.py --client client1 --num_classes 2 --in_chns 1 --root_path ../data/FAZ_h5 --img_class faz --exp faz/FL_WeaklySeg_pCE_GatedCRFLoss_FedAvg
# python -u test_client4onemod_FL.py --client client1 --num_classes 2 --in_chns 1 --root_path ../data/FAZ_h5 --img_class faz --exp faz/FL_WeaklySeg_pCE_S2L_FedAvg_6000 & 
# python -u test_client4onemod_FL.py --client client1 --num_classes 2 --in_chns 1 --root_path ../data/FAZ_h5 --img_class faz --exp faz/FL_WeaklySeg_pCE_TV_FedAvg &
# python -u test_client4onemod_FL.py --client client1 --num_classes 2 --in_chns 1 --root_path ../data/FAZ_h5 --img_class faz --exp faz/FL_WeaklySeg_pCE_USTM_FedAvg

# python -u test_client4onemod_FL.py --client client1 --num_classes 3 --in_chns 3 --root_path ../data/ODOC_h5 --img_class odoc --exp odoc/FL_WeaklySeg_pCE_EM_FedAvg &
# python -u test_client4onemod_FL.py --client client1 --num_classes 3 --in_chns 3 --root_path ../data/ODOC_h5 --img_class odoc --exp odoc/FL_WeaklySeg_pCE_S2L_FedAvg &
# python -u test_client4onemod_FL.py --client client1 --num_classes 3 --in_chns 3 --root_path ../data/ODOC_h5 --img_class odoc --exp odoc/FL_WeaklySeg_pCE_TV_FedAvg
# python -u test_client4onemod_FL.py --client client1 --num_classes 3 --in_chns 3 --root_path ../data/ODOC_h5 --img_class odoc --exp odoc/WeaklySeg_pCE &  
# python -u test_client4onemod_FL.py --client client1 --num_classes 3 --in_chns 3 --root_path ../data/ODOC_h5 --img_class odoc --exp odoc/WeaklySeg_pCE_0.005 


# python -u test_client4onemod_FL.py --client client2 --num_classes 2 --in_chns 1 --root_path ../data/FAZ_h5 --img_class faz --exp faz/FL_WeaklySeg_pCE_EM_FedAvg &
# python -u test_client4onemod_FL.py --client client2 --num_classes 2 --in_chns 1 --root_path ../data/FAZ_h5 --img_class faz --exp faz/FL_WeaklySeg_pCE_FedAvg & 
# python -u test_client4onemod_FL.py --client client2 --num_classes 2 --in_chns 1 --root_path ../data/FAZ_h5 --img_class faz --exp faz/FL_WeaklySeg_pCE_GatedCRFLoss_FedAvg
# python -u test_client4onemod_FL.py --client client2 --num_classes 2 --in_chns 1 --root_path ../data/FAZ_h5 --img_class faz --exp faz/FL_WeaklySeg_pCE_S2L_FedAvg_6000 & 
# python -u test_client4onemod_FL.py --client client2 --num_classes 2 --in_chns 1 --root_path ../data/FAZ_h5 --img_class faz --exp faz/FL_WeaklySeg_pCE_TV_FedAvg &
# python -u test_client4onemod_FL.py --client client2 --num_classes 2 --in_chns 1 --root_path ../data/FAZ_h5 --img_class faz --exp faz/FL_WeaklySeg_pCE_USTM_FedAvg

# python -u test_client4onemod_FL.py --client client2 --num_classes 3 --in_chns 3 --root_path ../data/ODOC_h5 --img_class odoc --exp odoc/FL_WeaklySeg_pCE_EM_FedAvg &
# python -u test_client4onemod_FL.py --client client2 --num_classes 3 --in_chns 3 --root_path ../data/ODOC_h5 --img_class odoc --exp odoc/FL_WeaklySeg_pCE_S2L_FedAvg &  
# python -u test_client4onemod_FL.py --client client2 --num_classes 3 --in_chns 3 --root_path ../data/ODOC_h5 --img_class odoc --exp odoc/FL_WeaklySeg_pCE_TV_FedAvg
# python -u test_client4onemod_FL.py --client client2 --num_classes 3 --in_chns 3 --root_path ../data/ODOC_h5 --img_class odoc --exp odoc/WeaklySeg_pCE &  
# python -u test_client4onemod_FL.py --client client2 --num_classes 3 --in_chns 3 --root_path ../data/ODOC_h5 --img_class odoc --exp odoc/WeaklySeg_pCE_0.005 

# python -u test_client4onemod_FL.py --client client3 --num_classes 2 --in_chns 1 --root_path ../data/FAZ_h5 --img_class faz --exp faz/FL_WeaklySeg_pCE_EM_FedAvg &
# python -u test_client4onemod_FL.py --client client3 --num_classes 2 --in_chns 1 --root_path ../data/FAZ_h5 --img_class faz --exp faz/FL_WeaklySeg_pCE_FedAvg & 
# python -u test_client4onemod_FL.py --client client3 --num_classes 2 --in_chns 1 --root_path ../data/FAZ_h5 --img_class faz --exp faz/FL_WeaklySeg_pCE_GatedCRFLoss_FedAvg
# python -u test_client4onemod_FL.py --client client3 --num_classes 2 --in_chns 1 --root_path ../data/FAZ_h5 --img_class faz --exp faz/FL_WeaklySeg_pCE_S2L_FedAvg_6000 & 
# python -u test_client4onemod_FL.py --client client3 --num_classes 2 --in_chns 1 --root_path ../data/FAZ_h5 --img_class faz --exp faz/FL_WeaklySeg_pCE_TV_FedAvg &
# python -u test_client4onemod_FL.py --client client3 --num_classes 2 --in_chns 1 --root_path ../data/FAZ_h5 --img_class faz --exp faz/FL_WeaklySeg_pCE_USTM_FedAvg

# python -u test_client4onemod_FL.py --client client3 --num_classes 3 --in_chns 3 --root_path ../data/ODOC_h5 --img_class odoc --exp odoc/FL_WeaklySeg_pCE_EM_FedAvg &
# python -u test_client4onemod_FL.py --client client3 --num_classes 3 --in_chns 3 --root_path ../data/ODOC_h5 --img_class odoc --exp odoc/FL_WeaklySeg_pCE_S2L_FedAvg &  
# python -u test_client4onemod_FL.py --client client3 --num_classes 3 --in_chns 3 --root_path ../data/ODOC_h5 --img_class odoc --exp odoc/FL_WeaklySeg_pCE_TV_FedAvg
# python -u test_client4onemod_FL.py --client client3 --num_classes 3 --in_chns 3 --root_path ../data/ODOC_h5 --img_class odoc --exp odoc/WeaklySeg_pCE &  
# python -u test_client4onemod_FL.py --client client3 --num_classes 3 --in_chns 3 --root_path ../data/ODOC_h5 --img_class odoc --exp odoc/WeaklySeg_pCE_0.005 

# python -u test_client4onemod_FL.py --client client4 --num_classes 2 --in_chns 1 --root_path ../data/FAZ_h5 --img_class faz --exp faz/FL_WeaklySeg_pCE_EM_FedAvg &
# python -u test_client4onemod_FL.py --client client4 --num_classes 2 --in_chns 1 --root_path ../data/FAZ_h5 --img_class faz --exp faz/FL_WeaklySeg_pCE_FedAvg & 
# python -u test_client4onemod_FL.py --client client4 --num_classes 2 --in_chns 1 --root_path ../data/FAZ_h5 --img_class faz --exp faz/FL_WeaklySeg_pCE_GatedCRFLoss_FedAvg
# python -u test_client4onemod_FL.py --client client4 --num_classes 2 --in_chns 1 --root_path ../data/FAZ_h5 --img_class faz --exp faz/FL_WeaklySeg_pCE_S2L_FedAvg_6000 & 
# python -u test_client4onemod_FL.py --client client4 --num_classes 2 --in_chns 1 --root_path ../data/FAZ_h5 --img_class faz --exp faz/FL_WeaklySeg_pCE_TV_FedAvg &
# python -u test_client4onemod_FL.py --client client4 --num_classes 2 --in_chns 1 --root_path ../data/FAZ_h5 --img_class faz --exp faz/FL_WeaklySeg_pCE_USTM_FedAvg

# python -u test_client4onemod_FL.py --client client4 --num_classes 3 --in_chns 3 --root_path ../data/ODOC_h5 --img_class odoc --exp odoc/FL_WeaklySeg_pCE_EM_FedAvg &
# python -u test_client4onemod_FL.py --client client4 --num_classes 3 --in_chns 3 --root_path ../data/ODOC_h5 --img_class odoc --exp odoc/FL_WeaklySeg_pCE_S2L_FedAvg &  
# python -u test_client4onemod_FL.py --client client4 --num_classes 3 --in_chns 3 --root_path ../data/ODOC_h5 --img_class odoc --exp odoc/FL_WeaklySeg_pCE_TV_FedAvg
# python -u test_client4onemod_FL.py --client client4 --num_classes 3 --in_chns 3 --root_path ../data/ODOC_h5 --img_class odoc --exp odoc/WeaklySeg_pCE &  
# python -u test_client4onemod_FL.py --client client4 --num_classes 3 --in_chns 3 --root_path ../data/ODOC_h5 --img_class odoc --exp odoc/WeaklySeg_pCE_0.005 

# python -u test_client4onemod_FL.py --client client5 --num_classes 2 --in_chns 1 --root_path ../data/FAZ_h5 --img_class faz --exp faz/FL_WeaklySeg_pCE_EM_FedAvg &
# python -u test_client4onemod_FL.py --client client5 --num_classes 2 --in_chns 1 --root_path ../data/FAZ_h5 --img_class faz --exp faz/FL_WeaklySeg_pCE_FedAvg & 
# python -u test_client4onemod_FL.py --client client5 --num_classes 2 --in_chns 1 --root_path ../data/FAZ_h5 --img_class faz --exp faz/FL_WeaklySeg_pCE_GatedCRFLoss_FedAvg
# python -u test_client4onemod_FL.py --client client5 --num_classes 2 --in_chns 1 --root_path ../data/FAZ_h5 --img_class faz --exp faz/FL_WeaklySeg_pCE_S2L_FedAvg_6000 & 
# python -u test_client4onemod_FL.py --client client5 --num_classes 2 --in_chns 1 --root_path ../data/FAZ_h5 --img_class faz --exp faz/FL_WeaklySeg_pCE_TV_FedAvg &
# python -u test_client4onemod_FL.py --client client5 --num_classes 2 --in_chns 1 --root_path ../data/FAZ_h5 --img_class faz --exp faz/FL_WeaklySeg_pCE_USTM_FedAvg

# python -u test_client4onemod_FL.py --client client5 --num_classes 3 --in_chns 3 --root_path ../data/ODOC_h5 --img_class odoc --exp odoc/FL_WeaklySeg_pCE_EM_FedAvg &
# python -u test_client4onemod_FL.py --client client5 --num_classes 3 --in_chns 3 --root_path ../data/ODOC_h5 --img_class odoc --exp odoc/FL_WeaklySeg_pCE_S2L_FedAvg &  
# python -u test_client4onemod_FL.py --client client5 --num_classes 3 --in_chns 3 --root_path ../data/ODOC_h5 --img_class odoc --exp odoc/FL_WeaklySeg_pCE_TV_FedAvg
# python -u test_client4onemod_FL.py --client client5 --num_classes 3 --in_chns 3 --root_path ../data/ODOC_h5 --img_class odoc --exp odoc/WeaklySeg_pCE &  
# python -u test_client4onemod_FL.py --client client5 --num_classes 3 --in_chns 3 --root_path ../data/ODOC_h5 --img_class odoc --exp odoc/WeaklySeg_pCE_0.005 



# python -u test_client4onemod_FL.py --client client1 --num_classes 2 --in_chns 1 --root_path ../data/FAZ_h5 --img_class faz --exp faz/FL_WeaklySeg_pCE_GatedCRFTreeEnergyLoss_FedAvg &
# python -u test_client4onemod_FL.py --client client1 --num_classes 2 --in_chns 1 --root_path ../data/FAZ_h5 --img_class faz --exp faz/FL_WeaklySeg_pCE_TreeEnergyLoss_FedAvg

# python -u test_client4onemod_FL.py --client client2 --num_classes 2 --in_chns 1 --root_path ../data/FAZ_h5 --img_class faz --exp faz/FL_WeaklySeg_pCE_GatedCRFTreeEnergyLoss_FedAvg &
# python -u test_client4onemod_FL.py --client client2 --num_classes 2 --in_chns 1 --root_path ../data/FAZ_h5 --img_class faz --exp faz/FL_WeaklySeg_pCE_TreeEnergyLoss_FedAvg
# python -u test_client4onemod_FL.py --client client3 --num_classes 2 --in_chns 1 --root_path ../data/FAZ_h5 --img_class faz --exp faz/FL_WeaklySeg_pCE_GatedCRFTreeEnergyLoss_FedAvg &
# python -u test_client4onemod_FL.py --client client3 --num_classes 2 --in_chns 1 --root_path ../data/FAZ_h5 --img_class faz --exp faz/FL_WeaklySeg_pCE_TreeEnergyLoss_FedAvg
# python -u test_client4onemod_FL.py --client client4 --num_classes 2 --in_chns 1 --root_path ../data/FAZ_h5 --img_class faz --exp faz/FL_WeaklySeg_pCE_GatedCRFTreeEnergyLoss_FedAvg &
# python -u test_client4onemod_FL.py --client client4 --num_classes 2 --in_chns 1 --root_path ../data/FAZ_h5 --img_class faz --exp faz/FL_WeaklySeg_pCE_TreeEnergyLoss_FedAvg
# python -u test_client4onemod_FL.py --client client5 --num_classes 2 --in_chns 1 --root_path ../data/FAZ_h5 --img_class faz --exp faz/FL_WeaklySeg_pCE_GatedCRFTreeEnergyLoss_FedAvg &
# python -u test_client4onemod_FL.py --client client5 --num_classes 2 --in_chns 1 --root_path ../data/FAZ_h5 --img_class faz --exp faz/FL_WeaklySeg_pCE_TreeEnergyLoss_FedAvg

# python -u test_client4onemod_FL.py --client client1 --num_classes 2 --in_chns 1 --root_path ../data/FAZ_h5 --img_class faz --exp faz/FL_WeaklySeg_pCE_MLoss_FedAvg &
# # python -u test_client4onemod_FL.py --client client1 --num_classes 2 --in_chns 1 --root_path ../data/FAZ_h5 --img_class faz --exp faz/FL_WeaklySeg_pCE_WSL4MIS_FedAvg &

# python -u test_client4onemod_FL.py --client client2 --num_classes 2 --in_chns 1 --root_path ../data/FAZ_h5 --img_class faz --exp faz/FL_WeaklySeg_pCE_MLoss_FedAvg
# # python -u test_client4onemod_FL.py --client client2 --num_classes 2 --in_chns 1 --root_path ../data/FAZ_h5 --img_class faz --exp faz/FL_WeaklySeg_pCE_WSL4MIS_FedAvg &
# python -u test_client4onemod_FL.py --client client3 --num_classes 2 --in_chns 1 --root_path ../data/FAZ_h5 --img_class faz --exp faz/FL_WeaklySeg_pCE_MLoss_FedAvg &
# # python -u test_client4onemod_FL.py --client client3 --num_classes 2 --in_chns 1 --root_path ../data/FAZ_h5 --img_class faz --exp faz/FL_WeaklySeg_pCE_WSL4MIS_FedAvg
# python -u test_client4onemod_FL.py --client client4 --num_classes 2 --in_chns 1 --root_path ../data/FAZ_h5 --img_class faz --exp faz/FL_WeaklySeg_pCE_MLoss_FedAvg
# # python -u test_client4onemod_FL.py --client client4 --num_classes 2 --in_chns 1 --root_path ../data/FAZ_h5 --img_class faz --exp faz/FL_WeaklySeg_pCE_WSL4MIS_FedAvg &
# python -u test_client4onemod_FL.py --client client5 --num_classes 2 --in_chns 1 --root_path ../data/FAZ_h5 --img_class faz --exp faz/FL_WeaklySeg_pCE_MLoss_FedAvg
# # python -u test_client4onemod_FL.py --client client5 --num_classes 2 --in_chns 1 --root_path ../data/FAZ_h5 --img_class faz --exp faz/FL_WeaklySeg_pCE_WSL4MIS_FedAvg
# python -u test_client4onemod_FL.py --client client1 --num_classes 3 --in_chns 3 --root_path ../data/ODOC_h5 --img_class odoc --exp odoc/FL_WeaklySeg_pCE_EM_FedAvg --model unet &
# python -u test_client4onemod_FL.py --client client1 --num_classes 3 --in_chns 3 --root_path ../data/ODOC_h5 --img_class odoc --exp odoc/FL_WeaklySeg_pCE_GatedCRFTreeEnergyLoss_FedAvg_AdamW_new --model unet_head &
# python -u test_client4onemod_FL.py --client client2 --num_classes 3 --in_chns 3 --root_path ../data/ODOC_h5 --img_class odoc --exp odoc/FL_WeaklySeg_pCE_EM_FedAvg --model unet
# python -u test_client4onemod_FL.py --client client2 --num_classes 3 --in_chns 3 --root_path ../data/ODOC_h5 --img_class odoc --exp odoc/FL_WeaklySeg_pCE_GatedCRFTreeEnergyLoss_FedAvg_AdamW_new --model unet_head &
# python -u test_client4onemod_FL.py --client client3 --num_classes 3 --in_chns 3 --root_path ../data/ODOC_h5 --img_class odoc --exp odoc/FL_WeaklySeg_pCE_EM_FedAvg --model unet &
# python -u test_client4onemod_FL.py --client client3 --num_classes 3 --in_chns 3 --root_path ../data/ODOC_h5 --img_class odoc --exp odoc/FL_WeaklySeg_pCE_GatedCRFTreeEnergyLoss_FedAvg_AdamW_new --model unet_head
# python -u test_client4onemod_FL.py --client client4 --num_classes 3 --in_chns 3 --root_path ../data/ODOC_h5 --img_class odoc --exp odoc/FL_WeaklySeg_pCE_EM_FedAvg --model unet &
# python -u test_client4onemod_FL.py --client client4 --num_classes 3 --in_chns 3 --root_path ../data/ODOC_h5 --img_class odoc --exp odoc/FL_WeaklySeg_pCE_GatedCRFTreeEnergyLoss_FedAvg_AdamW_new --model unet_head &
# python -u test_client4onemod_FL.py --client client5 --num_classes 3 --in_chns 3 --root_path ../data/ODOC_h5 --img_class odoc --exp odoc/FL_WeaklySeg_pCE_EM_FedAvg --model unet
# python -u test_client4onemod_FL.py --client client5 --num_classes 3 --in_chns 3 --root_path ../data/ODOC_h5 --img_class odoc --exp odoc/FL_WeaklySeg_pCE_GatedCRFTreeEnergyLoss_FedAvg_AdamW_new --model unet_head &

# python -u test_client4onemod_FL.py --client client1 --num_classes 3 --in_chns 3 --root_path ../data/ODOC_h5 --img_class odoc --exp odoc/FL_WeaklySeg_pCE_WSL4MIS_FedAvg --model unet_cct &
# python -u test_client4onemod_FL.py --client client2 --num_classes 3 --in_chns 3 --root_path ../data/ODOC_h5 --img_class odoc --exp odoc/FL_WeaklySeg_pCE_WSL4MIS_FedAvg --model unet_cct
# python -u test_client4onemod_FL.py --client client3 --num_classes 3 --in_chns 3 --root_path ../data/ODOC_h5 --img_class odoc --exp odoc/FL_WeaklySeg_pCE_WSL4MIS_FedAvg --model unet_cct &
# python -u test_client4onemod_FL.py --client client4 --num_classes 3 --in_chns 3 --root_path ../data/ODOC_h5 --img_class odoc --exp odoc/FL_WeaklySeg_pCE_WSL4MIS_FedAvg --model unet_cct &
# python -u test_client4onemod_FL.py --client client5 --num_classes 3 --in_chns 3 --root_path ../data/ODOC_h5 --img_class odoc --exp odoc/FL_WeaklySeg_pCE_WSL4MIS_FedAvg --model unet_cct

# python -u test_client4onemod_FL.py --client client1 --num_classes 3 --in_chns 3 --root_path ../data/ODOC_h5 --img_class odoc --exp odoc/FL_WeaklySeg_pCE_TreeEnergyLoss_FedAvg --model unet_head &
# python -u test_client4onemod_FL.py --client client2 --num_classes 3 --in_chns 3 --root_path ../data/ODOC_h5 --img_class odoc --exp odoc/FL_WeaklySeg_pCE_TreeEnergyLoss_FedAvg --model unet_head
# python -u test_client4onemod_FL.py --client client3 --num_classes 3 --in_chns 3 --root_path ../data/ODOC_h5 --img_class odoc --exp odoc/FL_WeaklySeg_pCE_TreeEnergyLoss_FedAvg --model unet_head &
# python -u test_client4onemod_FL.py --client client4 --num_classes 3 --in_chns 3 --root_path ../data/ODOC_h5 --img_class odoc --exp odoc/FL_WeaklySeg_pCE_TreeEnergyLoss_FedAvg --model unet_head &
# python -u test_client4onemod_FL.py --client client5 --num_classes 3 --in_chns 3 --root_path ../data/ODOC_h5 --img_class odoc --exp odoc/FL_WeaklySeg_pCE_TreeEnergyLoss_FedAvg --model unet_head


# python -u test_client4onemod_FL.py --client client1 --num_classes 3 --in_chns 3 --root_path ../data/ODOC_h5 --img_class odoc --exp odoc/FL_WeaklySeg_pCE_S2L_FedAvg --model unet &
# python -u test_client4onemod_FL.py --client client2 --num_classes 3 --in_chns 3 --root_path ../data/ODOC_h5 --img_class odoc --exp odoc/FL_WeaklySeg_pCE_S2L_FedAvg --model unet
# python -u test_client4onemod_FL.py --client client3 --num_classes 3 --in_chns 3 --root_path ../data/ODOC_h5 --img_class odoc --exp odoc/FL_WeaklySeg_pCE_S2L_FedAvg --model unet &
# python -u test_client4onemod_FL.py --client client4 --num_classes 3 --in_chns 3 --root_path ../data/ODOC_h5 --img_class odoc --exp odoc/FL_WeaklySeg_pCE_S2L_FedAvg --model unet &
# python -u test_client4onemod_FL.py --client client5 --num_classes 3 --in_chns 3 --root_path ../data/ODOC_h5 --img_class odoc --exp odoc/FL_WeaklySeg_pCE_S2L_FedAvg --model unet

# python -u test_client4onemod_FL.py --client client1 --num_classes 3 --in_chns 3 --root_path ../data/ODOC_h5 --img_class odoc --exp odoc/FL_WeaklySeg_pCE_USTM_FedAvg --model unet &
# python -u test_client4onemod_FL.py --client client2 --num_classes 3 --in_chns 3 --root_path ../data/ODOC_h5 --img_class odoc --exp odoc/FL_WeaklySeg_pCE_USTM_FedAvg --model unet
# python -u test_client4onemod_FL.py --client client3 --num_classes 3 --in_chns 3 --root_path ../data/ODOC_h5 --img_class odoc --exp odoc/FL_WeaklySeg_pCE_USTM_FedAvg --model unet &
# python -u test_client4onemod_FL.py --client client4 --num_classes 3 --in_chns 3 --root_path ../data/ODOC_h5 --img_class odoc --exp odoc/FL_WeaklySeg_pCE_USTM_FedAvg --model unet &
# python -u test_client4onemod_FL.py --client client5 --num_classes 3 --in_chns 3 --root_path ../data/ODOC_h5 --img_class odoc --exp odoc/FL_WeaklySeg_pCE_USTM_FedAvg --model unet


# python -u test_client4onemod_FL.py --client client1 --num_classes 2 --in_chns 1 --root_path ../data/FAZ_h5 --img_class faz --exp faz/FL_WeaklySeg_pCE_FedBN_0.001 &
# # python -u test_client4onemod_FL.py --client client1 --num_classes 2 --in_chns 1 --root_path ../data/FAZ_h5 --img_class faz --exp faz/FL_WeaklySeg_pCE_WSL4MIS_FedAvg &

# python -u test_client4onemod_FL.py --client client2 --num_classes 2 --in_chns 1 --root_path ../data/FAZ_h5 --img_class faz --exp faz/FL_WeaklySeg_pCE_FedBN_0.001
# # python -u test_client4onemod_FL.py --client client2 --num_classes 2 --in_chns 1 --root_path ../data/FAZ_h5 --img_class faz --exp faz/FL_WeaklySeg_pCE_WSL4MIS_FedAvg &
# python -u test_client4onemod_FL.py --client client3 --num_classes 2 --in_chns 1 --root_path ../data/FAZ_h5 --img_class faz --exp faz/FL_WeaklySeg_pCE_FedBN_0.001 &
# # python -u test_client4onemod_FL.py --client client3 --num_classes 2 --in_chns 1 --root_path ../data/FAZ_h5 --img_class faz --exp faz/FL_WeaklySeg_pCE_WSL4MIS_FedAvg
# python -u test_client4onemod_FL.py --client client4 --num_classes 2 --in_chns 1 --root_path ../data/FAZ_h5 --img_class faz --exp faz/FL_WeaklySeg_pCE_FedBN_0.001
# # python -u test_client4onemod_FL.py --client client4 --num_classes 2 --in_chns 1 --root_path ../data/FAZ_h5 --img_class faz --exp faz/FL_WeaklySeg_pCE_WSL4MIS_FedAvg &
# python -u test_client4onemod_FL.py --client client5 --num_classes 2 --in_chns 1 --root_path ../data/FAZ_h5 --img_class faz --exp faz/FL_WeaklySeg_pCE_FedBN_0.001
# # python -u test_client4onemod_FL.py --client client5 --num_classes 2 --in_chns 1 --root_path ../data/FAZ_h5 --img_class faz --exp faz/FL_WeaklySeg_pCE_WSL4MIS_FedAvg

# python -u test_client4onemod_FL_Personalize.py --client client0 --num_classes 2 --in_chns 1 --root_path ../data/FAZ_h5 --img_class faz --exp faz/FL_WeaklySeg_pCE_FedRep_0.001 &
# # python -u test_client4onemod_FL_Personalize.py --client client1 --num_classes 2 --in_chns 1 --root_path ../data/FAZ_h5 --img_class faz --exp faz/FL_WeaklySeg_pCE_WSL4MIS_FedAvg &

# python -u test_client4onemod_FL_Personalize.py --client client1 --num_classes 2 --in_chns 1 --root_path ../data/FAZ_h5 --img_class faz --exp faz/FL_WeaklySeg_pCE_FedRep_0.001
# # python -u test_client4onemod_FL_Personalize.py --client client2 --num_classes 2 --in_chns 1 --root_path ../data/FAZ_h5 --img_class faz --exp faz/FL_WeaklySeg_pCE_WSL4MIS_FedAvg &
# python -u test_client4onemod_FL_Personalize.py --client client2 --num_classes 2 --in_chns 1 --root_path ../data/FAZ_h5 --img_class faz --exp faz/FL_WeaklySeg_pCE_FedRep_0.001 &
# # python -u test_client4onemod_FL_Personalize.py --client client3 --num_classes 2 --in_chns 1 --root_path ../data/FAZ_h5 --img_class faz --exp faz/FL_WeaklySeg_pCE_WSL4MIS_FedAvg
# python -u test_client4onemod_FL_Personalize.py --client client3 --num_classes 2 --in_chns 1 --root_path ../data/FAZ_h5 --img_class faz --exp faz/FL_WeaklySeg_pCE_FedRep_0.001
# # python -u test_client4onemod_FL_Personalize.py --client client4 --num_classes 2 --in_chns 1 --root_path ../data/FAZ_h5 --img_class faz --exp faz/FL_WeaklySeg_pCE_WSL4MIS_FedAvg &
# python -u test_client4onemod_FL_Personalize.py --client client4 --num_classes 2 --in_chns 1 --root_path ../data/FAZ_h5 --img_class faz --exp faz/FL_WeaklySeg_pCE_FedRep_0.001
# # python -u test_client4onemod_FL_Personalize.py --client client5 --num_classes 2 --in_chns 1 --root_path ../data/FAZ_h5 --img_class faz --exp faz/FL_WeaklySeg_pCE_WSL4MIS_FedAvg

# python -u test_client4onemod_FL.py --client client1 --num_classes 3 --in_chns 3 --root_path ../data/ODOC_h5 --img_class odoc --exp odoc/FL_WeaklySeg_pCE_RW_FedAvg &
# # python -u test_client4onemod_FL.py --client client1 --num_classes 3 --in_chns 3 --root_path ../data/ODOC_h5 --img_class odoc --exp odoc/FL_WeaklySeg_pCE_WSL4MIS_FedAvg &
# python -u test_client4onemod_FL.py --client client2 --num_classes 3 --in_chns 3 --root_path ../data/ODOC_h5 --img_class odoc --exp odoc/FL_WeaklySeg_pCE_RW_FedAvg
# # python -u test_client4onemod_FL.py --client client2 --num_classes 3 --in_chns 3 --root_path ../data/ODOC_h5 --img_class odoc --exp odoc/FL_WeaklySeg_pCE_WSL4MIS_FedAvg &
# python -u test_client4onemod_FL.py --client client3 --num_classes 3 --in_chns 3 --root_path ../data/ODOC_h5 --img_class odoc --exp odoc/FL_WeaklySeg_pCE_RW_FedAvg &
# # python -u test_client4onemod_FL.py --client client3 --num_classes 3 --in_chns 3 --root_path ../data/ODOC_h5 --img_class odoc --exp odoc/FL_WeaklySeg_pCE_WSL4MIS_FedAvg
# python -u test_client4onemod_FL.py --client client4 --num_classes 3 --in_chns 3 --root_path ../data/ODOC_h5 --img_class odoc --exp odoc/FL_WeaklySeg_pCE_RW_FedAvg
# # python -u test_client4onemod_FL.py --client client4 --num_classes 3 --in_chns 3 --root_path ../data/ODOC_h5 --img_class odoc --exp odoc/FL_WeaklySeg_pCE_WSL4MIS_FedAvg &
# python -u test_client4onemod_FL.py --client client5 --num_classes 3 --in_chns 3 --root_path ../data/ODOC_h5 --img_class odoc --exp odoc/FL_WeaklySeg_pCE_RW_FedAvg
# # python -u test_client4onemod_FL.py --client client5 --num_classes 3 --in_chns 3 --root_path ../data/ODOC_h5 --img_class odoc --exp odoc/FL_WeaklySeg_pCE_WSL4MIS_FedAvg

# python -u test_client4onemod_FL_Personalize.py --client client0 --num_classes 2 --in_chns 1 --root_path ../data/FAZ_h5 --img_class faz --exp faz/FL_WeaklySeg_pCE_FedRep_0.01_rep_iters_3 &
# # python -u test_client4onemod_FL_Personalize.py --client client1 --num_classes 2 --in_chns 1 --root_path ../data/FAZ_h5 --img_class faz --exp faz/FL_WeaklySeg_pCE_WSL4MIS_FedAvg &
# python -u test_client4onemod_FL_Personalize.py --client client1 --num_classes 2 --in_chns 1 --root_path ../data/FAZ_h5 --img_class faz --exp faz/FL_WeaklySeg_pCE_FedRep_0.01_rep_iters_3
# # python -u test_client4onemod_FL_Personalize.py --client client2 --num_classes 2 --in_chns 1 --root_path ../data/FAZ_h5 --img_class faz --exp faz/FL_WeaklySeg_pCE_WSL4MIS_FedAvg &
# python -u test_client4onemod_FL_Personalize.py --client client2 --num_classes 2 --in_chns 1 --root_path ../data/FAZ_h5 --img_class faz --exp faz/FL_WeaklySeg_pCE_FedRep_0.01_rep_iters_3 &
# # python -u test_client4onemod_FL_Personalize.py --client client3 --num_classes 2 --in_chns 1 --root_path ../data/FAZ_h5 --img_class faz --exp faz/FL_WeaklySeg_pCE_WSL4MIS_FedAvg
# python -u test_client4onemod_FL_Personalize.py --client client3 --num_classes 2 --in_chns 1 --root_path ../data/FAZ_h5 --img_class faz --exp faz/FL_WeaklySeg_pCE_FedRep_0.01_rep_iters_3
# # python -u test_client4onemod_FL_Personalize.py --client client4 --num_classes 2 --in_chns 1 --root_path ../data/FAZ_h5 --img_class faz --exp faz/FL_WeaklySeg_pCE_WSL4MIS_FedAvg &
# python -u test_client4onemod_FL_Personalize.py --client client4 --num_classes 2 --in_chns 1 --root_path ../data/FAZ_h5 --img_class faz --exp faz/FL_WeaklySeg_pCE_FedRep_0.01_rep_iters_3
# # python -u test_client4onemod_FL_Personalize.py --client client5 --num_classes 2 --in_chns 1 --root_path ../data/FAZ_h5 --img_class faz --exp faz/FL_WeaklySeg_pCE_WSL4MIS_FedAvg

# python -u test_client4onemod_FL_Personalize.py --client client0 --num_classes 2 --in_chns 1 --root_path ../data/FAZ_h5 --img_class faz --exp faz/FL_WeaklySeg_pCE_FedBN &
# # python -u test_client4onemod_FL_Personalize.py --client client1 --num_classes 2 --in_chns 1 --root_path ../data/FAZ_h5 --img_class faz --exp faz/FL_WeaklySeg_pCE_WSL4MIS_FedAvg &

# python -u test_client4onemod_FL_Personalize.py --client client1 --num_classes 2 --in_chns 1 --root_path ../data/FAZ_h5 --img_class faz --exp faz/FL_WeaklySeg_pCE_FedBN
# # python -u test_client4onemod_FL_Personalize.py --client client2 --num_classes 2 --in_chns 1 --root_path ../data/FAZ_h5 --img_class faz --exp faz/FL_WeaklySeg_pCE_WSL4MIS_FedAvg &
# python -u test_client4onemod_FL_Personalize.py --client client2 --num_classes 2 --in_chns 1 --root_path ../data/FAZ_h5 --img_class faz --exp faz/FL_WeaklySeg_pCE_FedBN_Newevaluate &
# # python -u test_client4onemod_FL_Personalize.py --client client3 --num_classes 2 --in_chns 1 --root_path ../data/FAZ_h5 --img_class faz --exp faz/FL_WeaklySeg_pCE_WSL4MIS_FedAvg
# python -u test_client4onemod_FL_Personalize.py --client client3 --num_classes 2 --in_chns 1 --root_path ../data/FAZ_h5 --img_class faz --exp faz/FL_WeaklySeg_pCE_FedBN
# # python -u test_client4onemod_FL_Personalize.py --client client4 --num_classes 2 --in_chns 1 --root_path ../data/FAZ_h5 --img_class faz --exp faz/FL_WeaklySeg_pCE_WSL4MIS_FedAvg &
# python -u test_client4onemod_FL_Personalize.py --client client4 --num_classes 2 --in_chns 1 --root_path ../data/FAZ_h5 --img_class faz --exp faz/FL_WeaklySeg_pCE_FedBN
# # python -u test_client4onemod_FL_Personalize.py --client client5 --num_classes 2 --in_chns 1 --root_path ../data/FAZ_h5 --img_class faz --exp faz/FL_WeaklySeg_pCE_WSL4MIS_FedAvg

# python -u test_client4onemod_FL_Personalize.py --client client0 --num_classes 2 --in_chns 1 --root_path ../data/FAZ_h5 --img_class faz --exp faz/FL_WeaklySeg_pCE_FedBN_0.001 &
# python -u test_client4onemod_FL_Personalize.py --client client1 --num_classes 2 --in_chns 1 --root_path ../data/FAZ_h5 --img_class faz --exp faz/FL_WeaklySeg_pCE_FedBN_0.001
# python -u test_client4onemod_FL_Personalize.py --client client2 --num_classes 2 --in_chns 1 --root_path ../data/FAZ_h5 --img_class faz --exp faz/FL_WeaklySeg_pCE_FedBN_0.001 &
# python -u test_client4onemod_FL_Personalize.py --client client3 --num_classes 2 --in_chns 1 --root_path ../data/FAZ_h5 --img_class faz --exp faz/FL_WeaklySeg_pCE_FedBN_0.001
# python -u test_client4onemod_FL_Personalize.py --client client4 --num_classes 2 --in_chns 1 --root_path ../data/FAZ_h5 --img_class faz --exp faz/FL_WeaklySeg_pCE_FedBN_0.001




# python -u test_client4onemod_FL_Personalize.py --client client0 --num_classes 3 --in_chns 3 --root_path ../data/ODOC_h5 --img_class odoc --exp odoc/FL_WeaklySeg_pCE_FedBN &
# python -u test_client4onemod_FL_Personalize.py --client client1 --num_classes 3 --in_chns 3 --root_path ../data/ODOC_h5 --img_class odoc --exp odoc/FL_WeaklySeg_pCE_FedBN
# python -u test_client4onemod_FL_Personalize.py --client client2 --num_classes 3 --in_chns 3 --root_path ../data/ODOC_h5 --img_class odoc --exp odoc/FL_WeaklySeg_pCE_FedBN &
# python -u test_client4onemod_FL_Personalize.py --client client3 --num_classes 3 --in_chns 3 --root_path ../data/ODOC_h5 --img_class odoc --exp odoc/FL_WeaklySeg_pCE_FedBN
# python -u test_client4onemod_FL_Personalize.py --client client4 --num_classes 3 --in_chns 3 --root_path ../data/ODOC_h5 --img_class odoc --exp odoc/FL_WeaklySeg_pCE_FedBN


# python -u test_client4onemod_FL_Personalize.py --client client0 --num_classes 2 --in_chns 1 --root_path ../data/FAZ_h5 --img_class faz --exp faz/FL_WeaklySeg_pCE_2D_metaFed_firststage_dynamic_2D_GateCRFTreeloss_threshold_0.6_all_2_gate_0.05_FT &
# python -u test_client4onemod_FL_Personalize.py --client client1 --num_classes 2 --in_chns 1 --root_path ../data/FAZ_h5 --img_class faz --exp faz/FL_WeaklySeg_pCE_2D_metaFed_firststage_dynamic_2D_GateCRFTreeloss_threshold_0.6_all_2_gate_0.05_FT
# python -u test_client4onemod_FL_Personalize.py --client client2 --num_classes 2 --in_chns 1 --root_path ../data/FAZ_h5 --img_class faz --exp faz/FL_WeaklySeg_pCE_2D_metaFed_firststage_dynamic_2D_GateCRFTreeloss_threshold_0.6_all_2_gate_0.05_FT &
# python -u test_client4onemod_FL_Personalize.py --client client3 --num_classes 2 --in_chns 1 --root_path ../data/FAZ_h5 --img_class faz --exp faz/FL_WeaklySeg_pCE_2D_metaFed_firststage_dynamic_2D_GateCRFTreeloss_threshold_0.6_all_2_gate_0.05_FT
# python -u test_client4onemod_FL_Personalize.py --client client4 --num_classes 2 --in_chns 1 --root_path ../data/FAZ_h5 --img_class faz --exp faz/FL_WeaklySeg_pCE_2D_metaFed_firststage_dynamic_2D_GateCRFTreeloss_threshold_0.6_all_2_gate_0.05_FT

# python -u test_client4onemod_FL_Personalize.py --client client0 --num_classes 3 --in_chns 3 --root_path ../data/ODOC_h5 --img_class odoc --exp odoc/FL_WeaklySeg_pCE_GatedCRFTreeEnergyLoss_FedAvg_AdamW_new --model unet_head &
# python -u test_client4onemod_FL_Personalize.py --client client1 --num_classes 3 --in_chns 3 --root_path ../data/ODOC_h5 --img_class odoc --exp odoc/FL_WeaklySeg_pCE_GatedCRFTreeEnergyLoss_FedAvg_AdamW_new --model unet_head
# python -u test_client4onemod_FL_Personalize.py --client client2 --num_classes 3 --in_chns 3 --root_path ../data/ODOC_h5 --img_class odoc --exp odoc/FL_WeaklySeg_pCE_GatedCRFTreeEnergyLoss_FedAvg_AdamW_new --model unet_head &
# python -u test_client4onemod_FL_Personalize.py --client client3 --num_classes 3 --in_chns 3 --root_path ../data/ODOC_h5 --img_class odoc --exp odoc/FL_WeaklySeg_pCE_GatedCRFTreeEnergyLoss_FedAvg_AdamW_new --model unet_head &
# python -u test_client4onemod_FL_Personalize.py --client client4 --num_classes 3 --in_chns 3 --root_path ../data/ODOC_h5 --img_class odoc --exp odoc/FL_WeaklySeg_pCE_GatedCRFTreeEnergyLoss_FedAvg_AdamW_new --model unet_head

# python -u test_client4onemod_FL_Personalize.py --client client0 --num_classes 2 --in_chns 1 --root_path ../data/FAZ_h5 --img_class faz --exp faz/FL_WeaklySeg_pCE_FedAP &
# python -u test_client4onemod_FL_Personalize.py --client client1 --num_classes 2 --in_chns 1 --root_path ../data/FAZ_h5 --img_class faz --exp faz/FL_WeaklySeg_pCE_FedAP
# python -u test_client4onemod_FL_Personalize.py --client client2 --num_classes 2 --in_chns 1 --root_path ../data/FAZ_h5 --img_class faz --exp faz/FL_WeaklySeg_pCE_FedAP &
# python -u test_client4onemod_FL_Personalize.py --client client3 --num_classes 2 --in_chns 1 --root_path ../data/FAZ_h5 --img_class faz --exp faz/FL_WeaklySeg_pCE_FedAP &
# python -u test_client4onemod_FL_Personalize.py --client client4 --num_classes 2 --in_chns 1 --root_path ../data/FAZ_h5 --img_class faz --exp faz/FL_WeaklySeg_pCE_FedAP






# python -u test_client4onemod_FL_Personalize.py --client client4 --num_classes 3 --in_chns 3 --root_path ../data/ODOC_h5 --img_class odoc --exp odoc/FL_WeaklySeg_pCE_USTM_FedAvg --model unet


# python -u test_client4onemod_FL_Personalize.py --client client0 --num_classes 2 --in_chns 1 --root_path ../data/FAZ_h5 --img_class faz --exp faz/FL_WeaklySeg_pCE_FedAvg --model unet


# # polyp FedALALC GatedCRFTreeEnergyLoss
# python -u test_client4onemod_FL_Personalize.py --client client0 --num_classes 2 --in_chns 3 --root_path ../data/Polypdata_h5 --img_class polyp --exp polyp/FL_WeaklySeg_GatedCRFTreeEnergyLoss_flower_FedALALC_alpha1 --min_num_clients 4 --cid 0 --model unet_lc &

# python -u test_client4onemod_FL_Personalize.py --client client1 --num_classes 2 --in_chns 3 --root_path ../data/Polypdata_h5 --img_class polyp --exp polyp/FL_WeaklySeg_GatedCRFTreeEnergyLoss_flower_FedALALC_alpha1 --min_num_clients 4 --cid 1 --model unet_lc &

# python -u test_client4onemod_FL_Personalize.py --client client2 --num_classes 2 --in_chns 3 --root_path ../data/Polypdata_h5 --img_class polyp --exp polyp/FL_WeaklySeg_GatedCRFTreeEnergyLoss_flower_FedALALC_alpha1 --min_num_clients 4 --cid 2 --model unet_lc &

# python -u test_client4onemod_FL_Personalize.py --client client3 --num_classes 2 --in_chns 3 --root_path ../data/Polypdata_h5 --img_class polyp --exp polyp/FL_WeaklySeg_GatedCRFTreeEnergyLoss_flower_FedALALC_alpha1 --min_num_clients 4 --cid 3 --model unet_lc


# polyp FedAPLC GatedCRFTreeEnergyLoss
# python -u test_client4onemod_FL_Personalize.py --client client0 --num_classes 2 --in_chns 3 --root_path ../data/Polypdata_h5 --img_class polyp --exp polyp/FL_WeaklySeg_GatedCRFTreeEnergyLoss_flower_FedAPLC_alpha1 --min_num_clients 4 --cid 0 --model unet_lc &

# python -u test_client4onemod_FL_Personalize.py --client client1 --num_classes 2 --in_chns 3 --root_path ../data/Polypdata_h5 --img_class polyp --exp polyp/FL_WeaklySeg_GatedCRFTreeEnergyLoss_flower_FedAPLC_alpha1 --min_num_clients 4 --cid 1 --model unet_lc &

# python -u test_client4onemod_FL_Personalize.py --client client2 --num_classes 2 --in_chns 3 --root_path ../data/Polypdata_h5 --img_class polyp --exp polyp/FL_WeaklySeg_GatedCRFTreeEnergyLoss_flower_FedAPLC_alpha1 --min_num_clients 4 --cid 2 --model unet_lc &

# python -u test_client4onemod_FL_Personalize.py --client client3 --num_classes 2 --in_chns 3 --root_path ../data/Polypdata_h5 --img_class polyp --exp polyp/FL_WeaklySeg_GatedCRFTreeEnergyLoss_flower_FedAPLC_alpha1 --min_num_clients 4 --cid 3 --model unet_lc





# tSNE
# python -u test_client4onemod_FL_Personalize_tsNE.py --client client0 --num_classes 2 --in_chns 1 --root_path ../data/FAZ_h5 --img_class faz --exp faz/FL_WeaklySeg_GateCRFMsacleTreeEnergyLoss_flower_FedALALC_alpha5_up --model unet_lc_multihead &
# python -u test_client4onemod_FL_Personalize_tsNE.py --client client1 --num_classes 2 --in_chns 1 --root_path ../data/FAZ_h5 --img_class faz --exp faz/FL_WeaklySeg_GateCRFMsacleTreeEnergyLoss_flower_FedALALC_alpha5_up --model unet_lc_multihead
# python -u test_client4onemod_FL_Personalize_tsNE.py --client client2 --num_classes 2 --in_chns 1 --root_path ../data/FAZ_h5 --img_class faz --exp faz/FL_WeaklySeg_GateCRFMsacleTreeEnergyLoss_flower_FedALALC_alpha5_up --model unet_lc_multihead &
# python -u test_client4onemod_FL_Personalize_tsNE.py --client client3 --num_classes 2 --in_chns 1 --root_path ../data/FAZ_h5 --img_class faz --exp faz/FL_WeaklySeg_GateCRFMsacleTreeEnergyLoss_flower_FedALALC_alpha5_up --model unet_lc_multihead &
# python -u test_client4onemod_FL_Personalize_tsNE.py --client client4 --num_classes 2 --in_chns 1 --root_path ../data/FAZ_h5 --img_class faz --exp faz/FL_WeaklySeg_GateCRFMsacleTreeEnergyLoss_flower_FedALALC_alpha5_up --model unet_lc_multihead





# python -u test_client4onemod_FL_Personalize.py --client client0 --num_classes 3 --in_chns 3 --root_path ../data/ODOC_h5 --img_class odoc --exp odoc/FL_WeaklySeg_GateCRFLoss_flower_FedALALC_alpha1_up --model unet_lc --cid 0 --min_num_clients 5 &

# python -u test_client4onemod_FL_Personalize.py --client client1 --num_classes 3 --in_chns 3 --root_path ../data/ODOC_h5 --img_class odoc --exp odoc/FL_WeaklySeg_GateCRFLoss_flower_FedALALC_alpha1_up --model unet_lc --cid 1 --min_num_clients 5 &

# python -u test_client4onemod_FL_Personalize.py --client client2 --num_classes 3 --in_chns 3 --root_path ../data/ODOC_h5 --img_class odoc --exp odoc/FL_WeaklySeg_GateCRFLoss_flower_FedALALC_alpha1_up --model unet_lc --cid 2 --min_num_clients 5 &

# python -u test_client4onemod_FL_Personalize.py --client client3 --num_classes 3 --in_chns 3 --root_path ../data/ODOC_h5 --img_class odoc --exp odoc/FL_WeaklySeg_GateCRFLoss_flower_FedALALC_alpha1_up --model unet_lc --cid 3 --min_num_clients 5 &

# python -u test_client4onemod_FL_Personalize.py --client client4 --num_classes 3 --in_chns 3 --root_path ../data/ODOC_h5 --img_class odoc --exp odoc/FL_WeaklySeg_GateCRFLoss_flower_FedALALC_alpha1_up --model unet_lc --cid 4 --min_num_clients 5



# python -u test_client4onemod_FL_Personalize.py --client client0 --num_classes 2 --in_chns 1 --root_path ../data/FAZ_h5 --img_class faz --exp faz/WeaklySeg_pCE_2D_v5_auxpCE_detach_GatedCRFLoss_TwoLayersMLPBN_GAP_DisUni_2 --model unet_univ5 --min_num_clients 5 --img_size 256 &
# python -u test_client4onemod_FL_Personalize.py --client client1 --num_classes 2 --in_chns 1 --root_path ../data/FAZ_h5 --img_class faz --exp faz/WeaklySeg_pCE_2D_v5_auxpCE_detach_GatedCRFLoss_TwoLayersMLPBN_GAP_DisUni_2 --model unet_univ5 --min_num_clients 5 --img_size 256 &
# python -u test_client4onemod_FL_Personalize.py --client client2 --num_classes 2 --in_chns 1 --root_path ../data/FAZ_h5 --img_class faz --exp faz/WeaklySeg_pCE_2D_v5_auxpCE_detach_GatedCRFLoss_TwoLayersMLPBN_GAP_DisUni_2 --model unet_univ5 --min_num_clients 5 --img_size 256 --sup_type block&
# python -u test_client4onemod_FL_Personalize.py --client client3 --num_classes 2 --in_chns 1 --root_path ../data/FAZ_h5 --img_class faz --exp faz/WeaklySeg_pCE_2D_v5_auxpCE_detach_GatedCRFLoss_TwoLayersMLPBN_GAP_DisUni_2 --model unet_univ5 --min_num_clients 5 --img_size 256 --sup_type box&
# python -u test_client4onemod_FL_Personalize.py --client client4 --num_classes 2 --in_chns 1 --root_path ../data/FAZ_h5 --img_class faz --exp faz/WeaklySeg_pCE_2D_v5_auxpCE_detach_GatedCRFLoss_TwoLayersMLPBN_GAP_DisUni_2 --model unet_univ5 --min_num_clients 5 --img_size 256




# polyp FedAPLC GatedCRFTreeEnergyLoss
# python -u test_client4onemod_FL_Personalize.py --client client0 --num_classes 2 --in_chns 3 --root_path ../data/Polypdata_h5 --img_class polyp --exp polyp/FL_WeaklySeg_pCE_flower_FedLC_alpha0.1 --min_num_clients 4 --cid 0 --model unet_lc

# python -u test_client4onemod_FL_Personalize.py --client client1 --num_classes 2 --in_chns 3 --root_path ../data/Polypdata_h5 --img_class polyp --exp polyp/FL_WeaklySeg_pCE_flower_FedLC_alpha0.1 --min_num_clients 4 --cid 1 --model unet_lc

# python -u test_client4onemod_FL_Personalize.py --client client2 --num_classes 2 --in_chns 3 --root_path ../data/Polypdata_h5 --img_class polyp --exp polyp/FL_WeaklySeg_pCE_flower_FedLC_alpha0.1 --min_num_clients 4 --cid 2 --model unet_lc

# python -u test_client4onemod_FL_Personalize.py --client client3 --num_classes 2 --in_chns 3 --root_path ../data/Polypdata_h5 --img_class polyp --exp polyp/FL_WeaklySeg_pCE_flower_FedLC_alpha0.1 --min_num_clients 4 --cid 3 --model unet_lc



# # MainAblation2_Prompt_DA_Dual_NoALA
# python -u test_client4onemod_FL_Personalize.py --client client0 --num_classes 3 --in_chns 3 --root_path ../data/ODOC_h5 --img_class odoc --exp odoc/MainAblation2_Prompt_DA_Dual_NoALA --min_num_clients 5 --cid 0 --model unet_univ5 --img_size 384 --sup_type scribble --label_prompt 0 &

# python -u test_client4onemod_FL_Personalize.py --client client1 --num_classes 3 --in_chns 3 --root_path ../data/ODOC_h5 --img_class odoc --exp odoc/MainAblation2_Prompt_DA_Dual_NoALA --min_num_clients 5 --cid 1 --model unet_univ5 --img_size 384 --sup_type scribble_noisy --label_prompt 0 &

# python -u test_client4onemod_FL_Personalize.py --client client2 --num_classes 3 --in_chns 3 --root_path ../data/ODOC_h5 --img_class odoc --exp odoc/MainAblation2_Prompt_DA_Dual_NoALA --min_num_clients 5 --cid 2 --model unet_univ5 --img_size 384 --sup_type scribble_noisy --label_prompt 0 &

# python -u test_client4onemod_FL_Personalize.py --client client3 --num_classes 3 --in_chns 3 --root_path ../data/ODOC_h5 --img_class odoc --exp odoc/MainAblation2_Prompt_DA_Dual_NoALA --min_num_clients 5 --cid 3 --model unet_univ5 --img_size 384 --sup_type keypoint --label_prompt 0 &

# python -u test_client4onemod_FL_Personalize.py --client client4 --num_classes 3 --in_chns 3 --root_path ../data/ODOC_h5 --img_class odoc --exp odoc/MainAblation2_Prompt_DA_Dual_NoALA --min_num_clients 5 --cid 4 --model unet_univ5 --img_size 384 --sup_type block --label_prompt 0

# # MainAblation3_Prompt_DA_ALA_NoDual
# python -u test_client4onemod_FL_Personalize.py --client client0 --num_classes 3 --in_chns 3 --root_path ../data/ODOC_h5 --img_class odoc --exp odoc/MainAblation3_Prompt_DA_ALA_NoDual --min_num_clients 5 --cid 0 --model unet_univ5 --img_size 384 --sup_type scribble --label_prompt 0 &

# python -u test_client4onemod_FL_Personalize.py --client client1 --num_classes 3 --in_chns 3 --root_path ../data/ODOC_h5 --img_class odoc --exp odoc/MainAblation3_Prompt_DA_ALA_NoDual --min_num_clients 5 --cid 1 --model unet_univ5 --img_size 384 --sup_type scribble_noisy --label_prompt 0 &

# python -u test_client4onemod_FL_Personalize.py --client client2 --num_classes 3 --in_chns 3 --root_path ../data/ODOC_h5 --img_class odoc --exp odoc/MainAblation3_Prompt_DA_ALA_NoDual --min_num_clients 5 --cid 2 --model unet_univ5 --img_size 384 --sup_type scribble_noisy --label_prompt 0 &

# python -u test_client4onemod_FL_Personalize.py --client client3 --num_classes 3 --in_chns 3 --root_path ../data/ODOC_h5 --img_class odoc --exp odoc/MainAblation3_Prompt_DA_ALA_NoDual --min_num_clients 5 --cid 3 --model unet_univ5 --img_size 384 --sup_type keypoint --label_prompt 0 &

# python -u test_client4onemod_FL_Personalize.py --client client4 --num_classes 3 --in_chns 3 --root_path ../data/ODOC_h5 --img_class odoc --exp odoc/MainAblation3_Prompt_DA_ALA_NoDual --min_num_clients 5 --cid 4 --model unet_univ5 --img_size 384 --sup_type block --label_prompt 0


# # MainAblation4_Prompt_ALA_Dual_NoDA
# python -u test_client4onemod_FL_Personalize.py --client client0 --num_classes 3 --in_chns 3 --root_path ../data/ODOC_h5 --img_class odoc --exp odoc/MainAblation4_Prompt_ALA_Dual_NoDA --min_num_clients 5 --cid 0 --model unet_univ5_ablation --img_size 384 --sup_type scribble --label_prompt 0 --cid 0 &

# python -u test_client4onemod_FL_Personalize.py --client client1 --num_classes 3 --in_chns 3 --root_path ../data/ODOC_h5 --img_class odoc --exp odoc/MainAblation4_Prompt_ALA_Dual_NoDA --min_num_clients 5 --cid 1 --model unet_univ5_ablation --img_size 384 --sup_type scribble_noisy --label_prompt 0 --cid 1&

# python -u test_client4onemod_FL_Personalize.py --client client2 --num_classes 3 --in_chns 3 --root_path ../data/ODOC_h5 --img_class odoc --exp odoc/MainAblation4_Prompt_ALA_Dual_NoDA --min_num_clients 5 --cid 2 --model unet_univ5_ablation --img_size 384 --sup_type scribble_noisy --label_prompt 0 --cid 2&

# python -u test_client4onemod_FL_Personalize.py --client client3 --num_classes 3 --in_chns 3 --root_path ../data/ODOC_h5 --img_class odoc --exp odoc/MainAblation4_Prompt_ALA_Dual_NoDA --min_num_clients 5 --cid 3 --model unet_univ5_ablation --img_size 384 --sup_type keypoint --label_prompt 0 --cid 3&

# python -u test_client4onemod_FL_Personalize.py --client client4 --num_classes 3 --in_chns 3 --root_path ../data/ODOC_h5 --img_class odoc --exp odoc/MainAblation4_Prompt_ALA_Dual_NoDA --min_num_clients 5 --cid 4 --model unet_univ5_ablation --img_size 384 --sup_type block --label_prompt 0 --cid 4

# # DA_Ablation_Cab
# python -u test_client4onemod_FL_Personalize.py --client client0 --num_classes 3 --in_chns 3 --root_path ../data/ODOC_h5 --img_class odoc --exp odoc/DA_Ablation_Cab --min_num_clients 5 --cid 0 --model unet_univ5 --img_size 384 --sup_type scribble --label_prompt 0 --attention cab &

# python -u test_client4onemod_FL_Personalize.py --client client1 --num_classes 3 --in_chns 3 --root_path ../data/ODOC_h5 --img_class odoc --exp odoc/DA_Ablation_Cab --min_num_clients 5 --cid 1 --model unet_univ5 --img_size 384 --sup_type scribble_noisy --label_prompt 0 --attention cab &

# python -u test_client4onemod_FL_Personalize.py --client client2 --num_classes 3 --in_chns 3 --root_path ../data/ODOC_h5 --img_class odoc --exp odoc/DA_Ablation_Cab --min_num_clients 5 --cid 2 --model unet_univ5 --img_size 384 --sup_type scribble_noisy --label_prompt 0 --attention cab &

# python -u test_client4onemod_FL_Personalize.py --client client3 --num_classes 3 --in_chns 3 --root_path ../data/ODOC_h5 --img_class odoc --exp odoc/DA_Ablation_Cab --min_num_clients 5 --cid 3 --model unet_univ5 --img_size 384 --sup_type keypoint --label_prompt 0 --attention cab &

# python -u test_client4onemod_FL_Personalize.py --client client4 --num_classes 3 --in_chns 3 --root_path ../data/ODOC_h5 --img_class odoc --exp odoc/DA_Ablation_Cab --min_num_clients 5 --cid 4 --model unet_univ5 --img_size 384 --sup_type block --label_prompt 0 --attention cab

# DA_Ablation_Sab
# python -u test_client4onemod_FL_Personalize.py --client client0 --num_classes 3 --in_chns 3 --root_path ../data/ODOC_h5 --img_class odoc --exp odoc/DA_Ablation_Sab --min_num_clients 5 --cid 0 --model unet_univ5 --img_size 384 --sup_type scribble --label_prompt 0 --attention sab &

# python -u test_client4onemod_FL_Personalize.py --client client1 --num_classes 3 --in_chns 3 --root_path ../data/ODOC_h5 --img_class odoc --exp odoc/DA_Ablation_Sab --min_num_clients 5 --cid 1 --model unet_univ5 --img_size 384 --sup_type scribble_noisy --label_prompt 0 --attention sab &

# python -u test_client4onemod_FL_Personalize.py --client client2 --num_classes 3 --in_chns 3 --root_path ../data/ODOC_h5 --img_class odoc --exp odoc/DA_Ablation_Sab --min_num_clients 5 --cid 2 --model unet_univ5 --img_size 384 --sup_type scribble_noisy --label_prompt 0 --attention sab &

# python -u test_client4onemod_FL_Personalize.py --client client3 --num_classes 3 --in_chns 3 --root_path ../data/ODOC_h5 --img_class odoc --exp odoc/DA_Ablation_Sab --min_num_clients 5 --cid 3 --model unet_univ5 --img_size 384 --sup_type keypoint --label_prompt 0 --attention sab &

# python -u test_client4onemod_FL_Personalize.py --client client4 --num_classes 3 --in_chns 3 --root_path ../data/ODOC_h5 --img_class odoc --exp odoc/DA_Ablation_Sab --min_num_clients 5 --cid 4 --model unet_univ5 --img_size 384 --sup_type block --label_prompt 0 --attention sab



# # Prompt_Ablation_JustDist
# python -u test_client4onemod_FL_Personalize.py --client client0 --num_classes 3 --in_chns 3 --root_path ../data/ODOC_h5 --img_class odoc --exp odoc/Prompt_Ablation_JustDist --min_num_clients 5 --cid 0 --model unet_univ5 --img_size 384 --sup_type scribble --label_prompt 0 &

# python -u test_client4onemod_FL_Personalize.py --client client1 --num_classes 3 --in_chns 3 --root_path ../data/ODOC_h5 --img_class odoc --exp odoc/Prompt_Ablation_JustDist --min_num_clients 5 --cid 1 --model unet_univ5 --img_size 384 --sup_type scribble_noisy --label_prompt 0 &

# python -u test_client4onemod_FL_Personalize.py --client client2 --num_classes 3 --in_chns 3 --root_path ../data/ODOC_h5 --img_class odoc --exp odoc/Prompt_Ablation_JustDist --min_num_clients 5 --cid 2 --model unet_univ5 --img_size 384 --sup_type scribble_noisy --label_prompt 0 &

# python -u test_client4onemod_FL_Personalize.py --client client3 --num_classes 3 --in_chns 3 --root_path ../data/ODOC_h5 --img_class odoc --exp odoc/Prompt_Ablation_JustDist --min_num_clients 5 --cid 3 --model unet_univ5 --img_size 384 --sup_type keypoint --label_prompt 0 &

# python -u test_client4onemod_FL_Personalize.py --client client4 --num_classes 3 --in_chns 3 --root_path ../data/ODOC_h5 --img_class odoc --exp odoc/Prompt_Ablation_JustDist --min_num_clients 5 --cid 4 --model unet_univ5 --img_size 384 --sup_type block --label_prompt 0


# # Prompt_Ablation_NoLabelPrompt
# python -u test_client4onemod_FL_Personalize.py --client client0 --num_classes 3 --in_chns 3 --root_path ../data/ODOC_h5 --img_class odoc --exp odoc/Prompt_Ablation_NoLabelPrompt --min_num_clients 5 --cid 0 --model unet_univ5 --img_size 384 --sup_type scribble --label_prompt 0 &

# python -u test_client4onemod_FL_Personalize.py --client client1 --num_classes 3 --in_chns 3 --root_path ../data/ODOC_h5 --img_class odoc --exp odoc/Prompt_Ablation_NoLabelPrompt --min_num_clients 5 --cid 1 --model unet_univ5 --img_size 384 --sup_type scribble_noisy --label_prompt 0 &

# python -u test_client4onemod_FL_Personalize.py --client client2 --num_classes 3 --in_chns 3 --root_path ../data/ODOC_h5 --img_class odoc --exp odoc/Prompt_Ablation_NoLabelPrompt --min_num_clients 5 --cid 2 --model unet_univ5 --img_size 384 --sup_type scribble_noisy --label_prompt 0 &

# python -u test_client4onemod_FL_Personalize.py --client client3 --num_classes 3 --in_chns 3 --root_path ../data/ODOC_h5 --img_class odoc --exp odoc/Prompt_Ablation_NoLabelPrompt --min_num_clients 5 --cid 3 --model unet_univ5 --img_size 384 --sup_type keypoint --label_prompt 0 &

# python -u test_client4onemod_FL_Personalize.py --client client4 --num_classes 3 --in_chns 3 --root_path ../data/ODOC_h5 --img_class odoc --exp odoc/Prompt_Ablation_NoLabelPrompt --min_num_clients 5 --cid 4 --model unet_univ5 --img_size 384 --sup_type block --label_prompt 0

# # faz MainAblation1_Prompt_DA_NoDecoder2
# python -u test_client4onemod_FL_Personalize.py --client client0 --num_classes 2 --in_chns 1 --root_path ../data/FAZ_h5 --img_class faz --exp faz/MainAblation1_Prompt_DA_NoDecoder2 --model unet_univ5 --min_num_clients 5 --img_size 256 --sup_type scribble_noisy --label_prompt 0 --cid 0&
# python -u test_client4onemod_FL_Personalize.py --client client1 --num_classes 2 --in_chns 1 --root_path ../data/FAZ_h5 --img_class faz --exp faz/MainAblation1_Prompt_DA_NoDecoder2 --model unet_univ5 --min_num_clients 5 --img_size 256 --sup_type keypoint --label_prompt 0 --cid 1&
# python -u test_client4onemod_FL_Personalize.py --client client2 --num_classes 2 --in_chns 1 --root_path ../data/FAZ_h5 --img_class faz --exp faz/MainAblation1_Prompt_DA_NoDecoder2 --model unet_univ5 --min_num_clients 5 --img_size 256 --sup_type block --label_prompt 0 --cid 2&
# python -u test_client4onemod_FL_Personalize.py --client client3 --num_classes 2 --in_chns 1 --root_path ../data/FAZ_h5 --img_class faz --exp faz/MainAblation1_Prompt_DA_NoDecoder2 --model unet_univ5 --min_num_clients 5 --img_size 256 --sup_type box --label_prompt 0 --cid 3&
# python -u test_client4onemod_FL_Personalize.py --client client4 --num_classes 2 --in_chns 1 --root_path ../data/FAZ_h5 --img_class faz --exp faz/MainAblation1_Prompt_DA_NoDecoder2 --model unet_univ5 --min_num_clients 5 --img_size 256 --sup_type scribble --label_prompt 0 --cid 4


# # faz MainAblation2_Prompt_DA_Dual_NoALA
# python -u test_client4onemod_FL_Personalize.py --client client0 --num_classes 2 --in_chns 1 --root_path ../data/FAZ_h5 --img_class faz --exp faz/MainAblation2_Prompt_DA_Dual_NoALA --model unet_univ5 --min_num_clients 5 --img_size 256 --sup_type scribble_noisy --label_prompt 0 --cid 0 &
# python -u test_client4onemod_FL_Personalize.py --client client1 --num_classes 2 --in_chns 1 --root_path ../data/FAZ_h5 --img_class faz --exp faz/MainAblation2_Prompt_DA_Dual_NoALA --model unet_univ5 --min_num_clients 5 --img_size 256 --sup_type keypoint --label_prompt 0 --cid 1&
# python -u test_client4onemod_FL_Personalize.py --client client2 --num_classes 2 --in_chns 1 --root_path ../data/FAZ_h5 --img_class faz --exp faz/MainAblation2_Prompt_DA_Dual_NoALA --model unet_univ5 --min_num_clients 5 --img_size 256 --sup_type block --label_prompt 0 --cid 2&
# python -u test_client4onemod_FL_Personalize.py --client client3 --num_classes 2 --in_chns 1 --root_path ../data/FAZ_h5 --img_class faz --exp faz/MainAblation2_Prompt_DA_Dual_NoALA --model unet_univ5 --min_num_clients 5 --img_size 256 --sup_type box --label_prompt 0 --cid 3&
# python -u test_client4onemod_FL_Personalize.py --client client4 --num_classes 2 --in_chns 1 --root_path ../data/FAZ_h5 --img_class faz --exp faz/MainAblation2_Prompt_DA_Dual_NoALA --model unet_univ5 --min_num_clients 5 --img_size 256 --sup_type scribble --label_prompt 0 --cid 4



# python -u test_client4onemod_FL_Personalize.py --client client0 --num_classes 3 --in_chns 3 --root_path ../data/ODOC_h5 --img_class odoc --exp odoc/Dual_Ablation_Nearest --min_num_clients 5 --cid 0 --model unet_univ5 --img_size 384 --sup_type scribble --label_prompt 0 --label_prompt 0 --cid 0&

# python -u test_client4onemod_FL_Personalize.py --client client1 --num_classes 3 --in_chns 3 --root_path ../data/ODOC_h5 --img_class odoc --exp odoc/Dual_Ablation_Nearest --min_num_clients 5 --cid 1 --model unet_univ5 --img_size 384 --sup_type scribble_noisy --label_prompt 0 --label_prompt 0 --cid 1&

# python -u test_client4onemod_FL_Personalize.py --client client2 --num_classes 3 --in_chns 3 --root_path ../data/ODOC_h5 --img_class odoc --exp odoc/Dual_Ablation_Nearest --min_num_clients 5 --cid 2 --model unet_univ5 --img_size 384 --sup_type scribble_noisy --label_prompt 0 --label_prompt 0 --cid 2&

# python -u test_client4onemod_FL_Personalize.py --client client3 --num_classes 3 --in_chns 3 --root_path ../data/ODOC_h5 --img_class odoc --exp odoc/Dual_Ablation_Nearest --min_num_clients 5 --cid 3 --model unet_univ5 --img_size 384 --sup_type keypoint --label_prompt 0 --label_prompt 0 --cid 3&

# python -u test_client4onemod_FL_Personalize.py --client client4 --num_classes 3 --in_chns 3 --root_path ../data/ODOC_h5 --img_class odoc --exp odoc/Dual_Ablation_Nearest --min_num_clients 5 --cid 4 --model unet_univ5 --img_size 384 --sup_type block --label_prompt 0 --label_prompt 0 --cid 4



# python -u test_client4onemod_FL_Personalize.py --client client0 --num_classes 3 --in_chns 3 --root_path ../data/ODOC_h5 --img_class odoc --exp odoc/Dual_Ablation_Random --min_num_clients 5 --cid 0 --model unet_univ5 --img_size 384 --sup_type scribble --label_prompt 0 --label_prompt 0 --attention cab --cid 0&

# python -u test_client4onemod_FL_Personalize.py --client client1 --num_classes 3 --in_chns 3 --root_path ../data/ODOC_h5 --img_class odoc --exp odoc/Dual_Ablation_Random --min_num_clients 5 --cid 1 --model unet_univ5 --img_size 384 --sup_type scribble_noisy --label_prompt 0 --label_prompt 0 --attention cab --cid 1&

# python -u test_client4onemod_FL_Personalize.py --client client2 --num_classes 3 --in_chns 3 --root_path ../data/ODOC_h5 --img_class odoc --exp odoc/Dual_Ablation_Random --min_num_clients 5 --cid 2 --model unet_univ5 --img_size 384 --sup_type scribble_noisy --label_prompt 0 --label_prompt 0 --attention cab --cid 2&

# python -u test_client4onemod_FL_Personalize.py --client client3 --num_classes 3 --in_chns 3 --root_path ../data/ODOC_h5 --img_class odoc --exp odoc/Dual_Ablation_Random --min_num_clients 5 --cid 3 --model unet_univ5 --img_size 384 --sup_type keypoint --label_prompt 0 --label_prompt 0 --attention cab --cid 3&

# python -u test_client4onemod_FL_Personalize.py --client client4 --num_classes 3 --in_chns 3 --root_path ../data/ODOC_h5 --img_class odoc --exp odoc/Dual_Ablation_Random --min_num_clients 5 --cid 4 --model unet_univ5 --img_size 384 --sup_type block --label_prompt 0 --label_prompt 0 --attention cab --cid 4



# Rebuttal_WSL_Setting_FedAvg_lr0.01
# # FAZ WSL setting FedAvg
# python -u test_client4onemod_FL_Personalize.py --client client0 --num_classes 2 --in_chns 1 --root_path ../data/FAZ_h5 --img_class faz --exp faz/Rebuttal_WSL_Setting_FedAvg --model unet_univ5 --min_num_clients 5 --img_size 256 --sup_type scribble_noisy --label_prompt 0 --cid 0 &
# python -u test_client4onemod_FL_Personalize.py --client client1 --num_classes 2 --in_chns 1 --root_path ../data/FAZ_h5 --img_class faz --exp faz/Rebuttal_WSL_Setting_FedAvg --model unet_univ5 --min_num_clients 5 --img_size 256 --sup_type keypoint --label_prompt 0 --cid 1&
# python -u test_client4onemod_FL_Personalize.py --client client2 --num_classes 2 --in_chns 1 --root_path ../data/FAZ_h5 --img_class faz --exp faz/Rebuttal_WSL_Setting_FedAvg --model unet_univ5 --min_num_clients 5 --img_size 256 --sup_type block --label_prompt 0 --cid 2&
# python -u test_client4onemod_FL_Personalize.py --client client3 --num_classes 2 --in_chns 1 --root_path ../data/FAZ_h5 --img_class faz --exp faz/Rebuttal_WSL_Setting_FedAvg --model unet_univ5 --min_num_clients 5 --img_size 256 --sup_type box --label_prompt 0 --cid 3&
# python -u test_client4onemod_FL_Personalize.py --client client4 --num_classes 2 --in_chns 1 --root_path ../data/FAZ_h5 --img_class faz --exp faz/Rebuttal_WSL_Setting_FedAvg --model unet_univ5 --min_num_clients 5 --img_size 256 --sup_type scribble --label_prompt 0 --cid 4

# # FAZ WSL setting FedALA
# python -u test_client4onemod_FL_Personalize.py --client client0 --num_classes 2 --in_chns 1 --root_path ../data/FAZ_h5 --img_class faz --exp faz/Rebuttal_WSL_Setting_FedALA_evl_iter10 --model unet_univ5 --min_num_clients 5 --img_size 256 --sup_type scribble_noisy --prompt universal --attention dual --dual_init aggregated --label_prompt 0 --cid 0 &

# python -u test_client4onemod_FL_Personalize.py --client client1 --num_classes 2 --in_chns 1 --root_path ../data/FAZ_h5 --img_class faz --exp faz/Rebuttal_WSL_Setting_FedALA_evl_iter10 --model unet_univ5 --min_num_clients 5 --img_size 256 --sup_type keypoint --prompt universal --attention dual --dual_init aggregated --label_prompt 0 --cid 1 &

# python -u test_client4onemod_FL_Personalize.py --client client2 --num_classes 2 --in_chns 1 --root_path ../data/FAZ_h5 --img_class faz --exp faz/Rebuttal_WSL_Setting_FedALA_evl_iter10 --model unet_univ5 --min_num_clients 5 --img_size 256 --sup_type block --prompt universal --attention dual --dual_init aggregated --label_prompt 0 --cid 2 &

# python -u test_client4onemod_FL_Personalize.py --client client3 --num_classes 2 --in_chns 1 --root_path ../data/FAZ_h5 --img_class faz --exp faz/Rebuttal_WSL_Setting_FedALA_evl_iter10 --model unet_univ5 --min_num_clients 5 --img_size 256 --sup_type box --prompt universal --attention dual --dual_init aggregated --label_prompt 0 --cid 3 &

# python -u test_client4onemod_FL_Personalize.py --client client4 --num_classes 2 --in_chns 1 --root_path ../data/FAZ_h5 --img_class faz --exp faz/Rebuttal_WSL_Setting_FedALA_evl_iter10 --model unet_univ5 --min_num_clients 5 --img_size 256 --sup_type scribble --prompt universal --attention dual --dual_init aggregated --label_prompt 0 --cid 4



# # Polyp
# python -u test_client4onemod_FL_Personalize.py --client client0 --num_classes 2 --in_chns 3  --root_path ../data/Polypdata_h5 --img_class polyp --exp polyp/Rebuttal_WSL_Setting_FedAvg_lr0.001 --sup_type keypoint --model unet_univ5 --img_size 384 --min_num_clients 4 --sup_type keypoint --label_prompt 0 --cid 0&

# python -u test_client4onemod_FL_Personalize.py --client client1 --num_classes 2 --in_chns 3  --root_path ../data/Polypdata_h5 --img_class polyp --exp polyp/Rebuttal_WSL_Setting_FedAvg_lr0.001 --sup_type scribble --model unet_univ5 --img_size 384 --min_num_clients 4 --sup_type scribble --label_prompt 0 --cid 1& 

# python -u test_client4onemod_FL_Personalize.py --client client2 --num_classes 2 --in_chns 3  --root_path ../data/Polypdata_h5 --img_class polyp --exp polyp/Rebuttal_WSL_Setting_FedAvg_lr0.001 --sup_type box --model unet_univ5 --img_size 384 --min_num_clients 4 --sup_type box --label_prompt 0 --cid 2&

# python -u test_client4onemod_FL_Personalize.py --client client3 --num_classes 2 --in_chns 3  --root_path ../data/Polypdata_h5 --img_class polyp --exp polyp/Rebuttal_WSL_Setting_FedAvg_lr0.001 --sup_type block --model unet_univ5 --img_size 384 --min_num_clients 4 --sup_type block --label_prompt 0 --cid 3




# # CFL
# python -u test_client4onemod_FL_Personalize.py --client client0 --num_classes 2 --in_chns 1  --root_path ../data/Prostate_h5 --img_class prostate --exp prostate/Unet_pCE_AdamW_0.05_CFL_client_all/mask --sup_type block --model unet --img_size 384 --min_num_clients 6 --label_prompt 0 --cid 0 &

# python -u test_client4onemod_FL_Personalize.py --client client1 --num_classes 2 --in_chns 1  --root_path ../data/Prostate_h5 --img_class prostate --exp prostate/Unet_pCE_AdamW_0.05_CFL_client_all/mask --sup_type keypoint --model unet --img_size 384 --min_num_clients 6 --label_prompt 0 --cid 1 &

# python -u test_client4onemod_FL_Personalize.py --client client2 --num_classes 2 --in_chns 1  --root_path ../data/Prostate_h5 --img_class prostate --exp prostate/Unet_pCE_AdamW_0.05_CFL_client_all/mask --sup_type scribble --model unet --img_size 384 --min_num_clients 6 --label_prompt 0 --cid 2 &

# python -u test_client4onemod_FL_Personalize.py --client client3 --num_classes 2 --in_chns 1  --root_path ../data/Prostate_h5 --img_class prostate --exp prostate/Unet_pCE_AdamW_0.05_CFL_client_all/mask --sup_type keypoint --model unet --img_size 384 --min_num_clients 6 --label_prompt 0 --cid 3 &

# python -u test_client4onemod_FL_Personalize.py --client client4 --num_classes 2 --in_chns 1  --root_path ../data/Prostate_h5 --img_class prostate --exp prostate/Unet_pCE_AdamW_0.05_CFL_client_all/mask --sup_type scribble --model unet --img_size 384 --min_num_clients 6 --label_prompt 0 --cid 4 &

# python -u test_client4onemod_FL_Personalize.py --client client5 --num_classes 2 --in_chns 1  --root_path ../data/Prostate_h5 --img_class prostate --exp prostate/Unet_pCE_AdamW_0.05_CFL_client_all/mask --sup_type box --model unet --img_size 384 --min_num_clients 6 --label_prompt 0 --cid 5


# CL
# python -u test_client4onemod_FL_Personalize.py --client client0 --num_classes 2 --in_chns 1  --root_path ../data/Prostate_h5 --img_class prostate --exp prostate/Unet_pCE_AdamW_0.001_CL_client_all/prostate --sup_type block --model unet --img_size 384 --min_num_clients 6 --label_prompt 0 --cid 0 &

# python -u test_client4onemod_FL_Personalize.py --client client1 --num_classes 2 --in_chns 1  --root_path ../data/Prostate_h5 --img_class prostate --exp prostate/Unet_pCE_AdamW_0.001_CL_client_all/prostate --sup_type keypoint --model unet --img_size 384 --min_num_clients 6 --label_prompt 0 --cid 1 &

# python -u test_client4onemod_FL_Personalize.py --client client2 --num_classes 2 --in_chns 1  --root_path ../data/Prostate_h5 --img_class prostate --exp prostate/Unet_pCE_AdamW_0.001_CL_client_all/prostate --sup_type scribble --model unet --img_size 384 --min_num_clients 6 --label_prompt 0 --cid 2 &

# python -u test_client4onemod_FL_Personalize.py --client client3 --num_classes 2 --in_chns 1  --root_path ../data/Prostate_h5 --img_class prostate --exp prostate/Unet_pCE_AdamW_0.001_CL_client_all/prostate --sup_type keypoint --model unet --img_size 384 --min_num_clients 6 --label_prompt 0 --cid 3 &

# python -u test_client4onemod_FL_Personalize.py --client client4 --num_classes 2 --in_chns 1  --root_path ../data/Prostate_h5 --img_class prostate --exp prostate/Unet_pCE_AdamW_0.001_CL_client_all/prostate --sup_type scribble --model unet --img_size 384 --min_num_clients 6 --label_prompt 0 --cid 4 &

# python -u test_client4onemod_FL_Personalize.py --client client5 --num_classes 2 --in_chns 1  --root_path ../data/Prostate_h5 --img_class prostate --exp prostate/Unet_pCE_AdamW_0.001_CL_client_all/prostate --sup_type box --model unet --img_size 384 --min_num_clients 6 --label_prompt 0 --cid 5



# FedAVG
# python -u test_client4onemod_FL_Personalize.py --client client0 --num_classes 2 --in_chns 1  --root_path ../data/Prostate_h5 --img_class prostate --exp prostate/Rebuttal_WSL_Setting_FedAvg_AdamW_lr0.001 --sup_type block --model unet --img_size 384 --min_num_clients 6 --label_prompt 0 --cid 0 &

# python -u test_client4onemod_FL_Personalize.py --client client1 --num_classes 2 --in_chns 1  --root_path ../data/Prostate_h5 --img_class prostate --exp prostate/Rebuttal_WSL_Setting_FedAvg_AdamW_lr0.001 --sup_type keypoint --model unet --img_size 384 --min_num_clients 6 --label_prompt 0 --cid 1 &

# python -u test_client4onemod_FL_Personalize.py --client client2 --num_classes 2 --in_chns 1  --root_path ../data/Prostate_h5 --img_class prostate --exp prostate/Rebuttal_WSL_Setting_FedAvg_AdamW_lr0.001 --sup_type scribble --model unet --img_size 384 --min_num_clients 6 --label_prompt 0 --cid 2 &

# python -u test_client4onemod_FL_Personalize.py --client client3 --num_classes 2 --in_chns 1  --root_path ../data/Prostate_h5 --img_class prostate --exp prostate/Rebuttal_WSL_Setting_FedAvg_AdamW_lr0.001 --sup_type keypoint --model unet --img_size 384 --min_num_clients 6 --label_prompt 0 --cid 3 &

# python -u test_client4onemod_FL_Personalize.py --client client4 --num_classes 2 --in_chns 1  --root_path ../data/Prostate_h5 --img_class prostate --exp prostate/Rebuttal_WSL_Setting_FedAvg_AdamW_lr0.001 --sup_type scribble --model unet --img_size 384 --min_num_clients 6 --label_prompt 0 --cid 4 &

# python -u test_client4onemod_FL_Personalize.py --client client5 --num_classes 2 --in_chns 1  --root_path ../data/Prostate_h5 --img_class prostate --exp prostate/Rebuttal_WSL_Setting_FedAvg_AdamW_lr0.001 --sup_type box --model unet --img_size 384 --min_num_clients 6 --label_prompt 0 --cid 5



# # FedProx
# python -u test_client4onemod_FL_Personalize.py --client client0 --num_classes 2 --in_chns 1  --root_path ../data/Prostate_h5 --img_class prostate --exp prostate/Rebuttal_WSL_Setting_FedProx_AdamW_lr0.001 --sup_type block --model unet --img_size 384 --min_num_clients 6 --label_prompt 0 --cid 0 &

# python -u test_client4onemod_FL_Personalize.py --client client1 --num_classes 2 --in_chns 1  --root_path ../data/Prostate_h5 --img_class prostate --exp prostate/Rebuttal_WSL_Setting_FedProx_AdamW_lr0.001 --sup_type keypoint --model unet --img_size 384 --min_num_clients 6 --label_prompt 0 --cid 1 &

# python -u test_client4onemod_FL_Personalize.py --client client2 --num_classes 2 --in_chns 1  --root_path ../data/Prostate_h5 --img_class prostate --exp prostate/Rebuttal_WSL_Setting_FedProx_AdamW_lr0.001 --sup_type scribble --model unet --img_size 384 --min_num_clients 6 --label_prompt 0 --cid 2 &

# python -u test_client4onemod_FL_Personalize.py --client client3 --num_classes 2 --in_chns 1  --root_path ../data/Prostate_h5 --img_class prostate --exp prostate/Rebuttal_WSL_Setting_FedProx_AdamW_lr0.001 --sup_type keypoint --model unet --img_size 384 --min_num_clients 6 --label_prompt 0 --cid 3 &

# python -u test_client4onemod_FL_Personalize.py --client client4 --num_classes 2 --in_chns 1  --root_path ../data/Prostate_h5 --img_class prostate --exp prostate/Rebuttal_WSL_Setting_FedProx_AdamW_lr0.001 --sup_type scribble --model unet --img_size 384 --min_num_clients 6 --label_prompt 0 --cid 4 &

# python -u test_client4onemod_FL_Personalize.py --client client5 --num_classes 2 --in_chns 1  --root_path ../data/Prostate_h5 --img_class prostate --exp prostate/Rebuttal_WSL_Setting_FedProx_AdamW_lr0.001 --sup_type box --model unet --img_size 384 --min_num_clients 6 --label_prompt 0 --cid 5




# # FedLC
# python -u test_client4onemod_FL_Personalize.py --client client0 --num_classes 2 --in_chns 1  --root_path ../data/Prostate_h5 --img_class prostate --exp prostate/Rebuttal_WSL_Setting_FedLC_AdamW_lr0.01 --sup_type block --model unet_lc --img_size 384 --min_num_clients 6 --label_prompt 0 --cid 0 &

# python -u test_client4onemod_FL_Personalize.py --client client1 --num_classes 2 --in_chns 1  --root_path ../data/Prostate_h5 --img_class prostate --exp prostate/Rebuttal_WSL_Setting_FedLC_AdamW_lr0.01 --sup_type keypoint --model unet_lc --img_size 384 --min_num_clients 6 --label_prompt 0 --cid 1 &

# python -u test_client4onemod_FL_Personalize.py --client client2 --num_classes 2 --in_chns 1  --root_path ../data/Prostate_h5 --img_class prostate --exp prostate/Rebuttal_WSL_Setting_FedLC_AdamW_lr0.01 --sup_type scribble --model unet_lc --img_size 384 --min_num_clients 6 --label_prompt 0 --cid 2 &

# python -u test_client4onemod_FL_Personalize.py --client client3 --num_classes 2 --in_chns 1  --root_path ../data/Prostate_h5 --img_class prostate --exp prostate/Rebuttal_WSL_Setting_FedLC_AdamW_lr0.01 --sup_type keypoint --model unet_lc --img_size 384 --min_num_clients 6 --label_prompt 0 --cid 3 &

# python -u test_client4onemod_FL_Personalize.py --client client4 --num_classes 2 --in_chns 1  --root_path ../data/Prostate_h5 --img_class prostate --exp prostate/Rebuttal_WSL_Setting_FedLC_AdamW_lr0.01 --sup_type scribble --model unet_lc --img_size 384 --min_num_clients 6 --label_prompt 0 --cid 4 &

# python -u test_client4onemod_FL_Personalize.py --client client5 --num_classes 2 --in_chns 1  --root_path ../data/Prostate_h5 --img_class prostate --exp prostate/Rebuttal_WSL_Setting_FedLC_AdamW_lr0.01 --sup_type box --model unet_lc --img_size 384 --min_num_clients 6 --label_prompt 0 --cid 5


# # FedALA
# python -u test_client4onemod_FL_Personalize.py --client client0 --num_classes 2 --in_chns 1  --root_path ../data/Prostate_h5 --img_class prostate --exp prostate/FedALA --sup_type block --model unet --img_size 384 --min_num_clients 6 --label_prompt 0 --cid 0 &

# python -u test_client4onemod_FL_Personalize.py --client client1 --num_classes 2 --in_chns 1  --root_path ../data/Prostate_h5 --img_class prostate --exp prostate/FedALA --sup_type keypoint --model unet --img_size 384 --min_num_clients 6 --label_prompt 0 --cid 1 &

# python -u test_client4onemod_FL_Personalize.py --client client2 --num_classes 2 --in_chns 1  --root_path ../data/Prostate_h5 --img_class prostate --exp prostate/FedALA --sup_type scribble --model unet --img_size 384 --min_num_clients 6 --label_prompt 0 --cid 2 &

# python -u test_client4onemod_FL_Personalize.py --client client3 --num_classes 2 --in_chns 1  --root_path ../data/Prostate_h5 --img_class prostate --exp prostate/FedALA --sup_type keypoint --model unet --img_size 384 --min_num_clients 6 --label_prompt 0 --cid 3 &

# python -u test_client4onemod_FL_Personalize.py --client client4 --num_classes 2 --in_chns 1  --root_path ../data/Prostate_h5 --img_class prostate --exp prostate/FedALA --sup_type scribble --model unet --img_size 384 --min_num_clients 6 --label_prompt 0 --cid 4 &

# python -u test_client4onemod_FL_Personalize.py --client client5 --num_classes 2 --in_chns 1  --root_path ../data/Prostate_h5 --img_class prostate --exp prostate/FedALA --sup_type box --model unet --img_size 384 --min_num_clients 6 --label_prompt 0 --cid 5


# FedICRA
# python -u test_client4onemod_FL_Personalize.py --client client0 --num_classes 2 --in_chns 1  --root_path ../data/Prostate_h5 --img_class prostate --exp prostate/FedICRA --sup_type block --model unet_lc_multihead --img_size 384 --min_num_clients 6 --label_prompt 0 --cid 0 &

# python -u test_client4onemod_FL_Personalize.py --client client1 --num_classes 2 --in_chns 1  --root_path ../data/Prostate_h5 --img_class prostate --exp prostate/FedICRA --sup_type keypoint --model unet_lc_multihead --img_size 384 --min_num_clients 6 --label_prompt 0 --cid 1 &

# python -u test_client4onemod_FL_Personalize.py --client client2 --num_classes 2 --in_chns 1  --root_path ../data/Prostate_h5 --img_class prostate --exp prostate/FedICRA --sup_type scribble --model unet_lc_multihead --img_size 384 --min_num_clients 6 --label_prompt 0 --cid 2 &

# python -u test_client4onemod_FL_Personalize.py --client client3 --num_classes 2 --in_chns 1  --root_path ../data/Prostate_h5 --img_class prostate --exp prostate/FedICRA --sup_type keypoint --model unet_lc_multihead --img_size 384 --min_num_clients 6 --label_prompt 0 --cid 3 &

# python -u test_client4onemod_FL_Personalize.py --client client4 --num_classes 2 --in_chns 1  --root_path ../data/Prostate_h5 --img_class prostate --exp prostate/FedICRA --sup_type scribble --model unet_lc_multihead --img_size 384 --min_num_clients 6 --label_prompt 0 --cid 4 &

# python -u test_client4onemod_FL_Personalize.py --client client5 --num_classes 2 --in_chns 1  --root_path ../data/Prostate_h5 --img_class prostate --exp prostate/FedICRA --sup_type box --model unet_lc_multihead --img_size 384 --min_num_clients 6 --label_prompt 0 --cid 5




# python -u test_client4onemod_FL_Personalize.py --client client3 --num_classes 2 --in_chns 1 --root_path ../data/FAZ_h5 --img_class faz --exp faz/FedLC --min_num_clients 5 --cid 3 --model unet_lc




# python -u test_client4onemod_FL_Personalize.py --client client0 --num_classes 2 --in_chns 1 --root_path ../data/FAZ_h5 --img_class faz --exp faz/Rebuttal_WSL_Setting_FedLC --model unet_lc_auxi --min_num_clients 5 --img_size 256 --sup_type scribble_noisy --prompt universal --attention dual --dual_init aggregated --label_prompt 0 --cid 0 &

# python -u test_client4onemod_FL_Personalize.py --client client1 --num_classes 2 --in_chns 1 --root_path ../data/FAZ_h5 --img_class faz --exp faz/Rebuttal_WSL_Setting_FedLC --model unet_lc_auxi --min_num_clients 5 --img_size 256 --sup_type keypoint --prompt universal --attention dual --dual_init aggregated --label_prompt 0 --cid 1 &

# python -u test_client4onemod_FL_Personalize.py --client client2 --num_classes 2 --in_chns 1 --root_path ../data/FAZ_h5 --img_class faz --exp faz/Rebuttal_WSL_Setting_FedLC --model unet_lc_auxi --min_num_clients 5 --img_size 256 --sup_type block --prompt universal --attention dual --dual_init aggregated --label_prompt 0 --cid 2 &

# python -u test_client4onemod_FL_Personalize.py --client client3 --num_classes 2 --in_chns 1 --root_path ../data/FAZ_h5 --img_class faz --exp faz/Rebuttal_WSL_Setting_FedLC --model unet_lc_auxi --min_num_clients 5 --img_size 256 --sup_type box --prompt universal --attention dual --dual_init aggregated --label_prompt 0 --cid 3 &

# python -u test_client4onemod_FL_Personalize.py --client client4 --num_classes 2 --in_chns 1 --root_path ../data/FAZ_h5 --img_class faz --exp faz/Rebuttal_WSL_Setting_FedLC --model unet_lc_auxi --min_num_clients 5 --img_size 256 --sup_type scribble --prompt universal --attention dual --dual_init aggregated --label_prompt 0 --cid 4



# 

# python -u test_WSL.py --client client0 --num_classes 2 --in_chns 1  --root_path ../data/Prostate_h5 --img_class prostate --exp prostate_LFL/Unet_pCE_test_AdamW_0.001_client1 --sup_type block --model unet --img_size 384 --min_num_clients 6 --label_prompt 0 --cid 0 &

# python -u test_WSL.py --client client1 --num_classes 2 --in_chns 1  --root_path ../data/Prostate_h5 --img_class prostate --exp prostate_LFL/Unet_pCE_test_AdamW_0.001_client2 --sup_type keypoint --model unet --img_size 384 --min_num_clients 6 --label_prompt 0 --cid 1 &

# python -u test_WSL.py --client client2 --num_classes 2 --in_chns 1  --root_path ../data/Prostate_h5 --img_class prostate --exp prostate_LFL/Unet_pCE_test_AdamW_0.001_client3 --sup_type scribble --model unet --img_size 384 --min_num_clients 6 --label_prompt 0 --cid 2


# python -u test_WSL.py --client client0 --num_classes 2 --in_chns 1  --root_path ../data/Prostate_h5 --img_class prostate --exp prostate_LFL/Unet_pCE_test_AdamW_0.001_client1 --sup_type mask --model unet --img_size 384 --min_num_clients 6 --label_prompt 0 --cid 0 &

# python -u test_WSL.py --client client1 --num_classes 2 --in_chns 1  --root_path ../data/Prostate_h5 --img_class prostate --exp prostate_LFL/Unet_pCE_test_AdamW_0.001_client2 --sup_type mask --model unet --img_size 384 --min_num_clients 6 --label_prompt 0 --cid 1 &

# python -u test_WSL.py --client client2 --num_classes 2 --in_chns 1  --root_path ../data/Prostate_h5 --img_class prostate --exp prostate_LFL/Unet_pCE_test_AdamW_0.001_client3 --sup_type mask --model unet --img_size 384 --min_num_clients 6 --label_prompt 0 --cid 2 &




# # CFL
python -u test_client4onemod_FL_Personalize.py --client client0 --num_classes 2 --in_chns 1  --root_path ../data/Prostate_h5 --img_class prostate --exp prostate/Unet_pCE_AdamW_0.05_CFL_client_all/mask --sup_type block --model unet --img_size 384 --min_num_clients 6 --label_prompt 0 --cid 0 &

python -u test_client4onemod_FL_Personalize.py --client client1 --num_classes 2 --in_chns 1  --root_path ../data/Prostate_h5 --img_class prostate --exp prostate/Unet_pCE_AdamW_0.05_CFL_client_all/mask --sup_type keypoint --model unet --img_size 384 --min_num_clients 6 --label_prompt 0 --cid 1 &

python -u test_client4onemod_FL_Personalize.py --client client2 --num_classes 2 --in_chns 1  --root_path ../data/Prostate_h5 --img_class prostate --exp prostate/Unet_pCE_AdamW_0.05_CFL_client_all/mask --sup_type scribble --model unet --img_size 384 --min_num_clients 6 --label_prompt 0 --cid 2 &

python -u test_client4onemod_FL_Personalize.py --client client3 --num_classes 2 --in_chns 1  --root_path ../data/Prostate_h5 --img_class prostate --exp prostate/Unet_pCE_AdamW_0.05_CFL_client_all/mask --sup_type keypoint --model unet --img_size 384 --min_num_clients 6 --label_prompt 0 --cid 3 &

python -u test_client4onemod_FL_Personalize.py --client client4 --num_classes 2 --in_chns 1  --root_path ../data/Prostate_h5 --img_class prostate --exp prostate/Unet_pCE_AdamW_0.05_CFL_client_all/mask --sup_type scribble --model unet --img_size 384 --min_num_clients 6 --label_prompt 0 --cid 4 &

python -u test_client4onemod_FL_Personalize.py --client client5 --num_classes 2 --in_chns 1  --root_path ../data/Prostate_h5 --img_class prostate --exp prostate/Unet_pCE_AdamW_0.05_CFL_client_all/mask --sup_type box --model unet --img_size 384 --min_num_clients 6 --label_prompt 0 --cid 5