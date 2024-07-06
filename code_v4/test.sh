# Ours test
python -u test_client4onemod_FL_Personalize.py --client client0 --num_classes 3 --in_chns 3 --root_path ../data/ODOC_h5 --img_class odoc --exp odoc/FL_WeaklySeg_GatedCRFMscaleTreeEnergyLoss_FedALALC_alpha1_up_full_FT_0.001 --min_num_clients 5 --cid 0 --model unet_lc_multihead &

python -u test_client4onemod_FL_Personalize.py --client client1 --num_classes 3 --in_chns 3 --root_path ../data/ODOC_h5 --img_class odoc --exp odoc/FL_WeaklySeg_GatedCRFMscaleTreeEnergyLoss_FedALALC_alpha1_up_full_FT_0.001 --min_num_clients 5 --cid 1 --model unet_lc_multihead &

python -u test_client4onemod_FL_Personalize.py --client client2 --num_classes 3 --in_chns 3 --root_path ../data/ODOC_h5 --img_class odoc --exp odoc/FL_WeaklySeg_GatedCRFMscaleTreeEnergyLoss_FedALALC_alpha1_up_full_FT_0.001 --min_num_clients 5 --cid 2 --model unet_lc_multihead &

python -u test_client4onemod_FL_Personalize.py --client client3 --num_classes 3 --in_chns 3 --root_path ../data/ODOC_h5 --img_class odoc --exp odoc/FL_WeaklySeg_GatedCRFMscaleTreeEnergyLoss_FedALALC_alpha1_up_full_FT_0.001 --min_num_clients 5 --cid 3 --model unet_lc_multihead &

python -u test_client4onemod_FL_Personalize.py --client client4 --num_classes 3 --in_chns 3 --root_path ../data/ODOC_h5 --img_class odoc --exp odoc/FL_WeaklySeg_GatedCRFMscaleTreeEnergyLoss_FedALALC_alpha1_up_full_FT_0.001 --min_num_clients 5 --cid 4 --model unet_lc_multihead
