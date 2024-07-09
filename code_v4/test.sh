## FAZ 
# python -u test_client4onemod_FL_Personalize.py --client client0 --num_classes 2 --in_chns 1 --root_path ../data/FAZ_h5 --img_class faz --exp faz/FedLPPA --model unet_univ5 --min_num_clients 5 --img_size 256 --sup_type scribble_noisy --label_prompt 1 &
# python -u test_client4onemod_FL_Personalize.py --client client1 --num_classes 2 --in_chns 1 --root_path ../data/FAZ_h5 --img_class faz --exp faz/FedLPPA --model unet_univ5 --min_num_clients 5 --img_size 256 --sup_type keypoint --label_prompt 1 &
# python -u test_client4onemod_FL_Personalize.py --client client2 --num_classes 2 --in_chns 1 --root_path ../data/FAZ_h5 --img_class faz --exp faz/FedLPPA --model unet_univ5 --min_num_clients 5 --img_size 256 --sup_type block --label_prompt 1 &
# python -u test_client4onemod_FL_Personalize.py --client client3 --num_classes 2 --in_chns 1 --root_path ../data/FAZ_h5 --img_class faz --exp faz/FedLPPA --model unet_univ5 --min_num_clients 5 --img_size 256 --sup_type box --label_prompt 1 &
# python -u test_client4onemod_FL_Personalize.py --client client4 --num_classes 2 --in_chns 1 --root_path ../data/FAZ_h5 --img_class faz --exp faz/FedLPPA --model unet_univ5 --min_num_clients 5 --img_size 256 --sup_type scribble --label_prompt 1

## ODOC

# python -u test_client4onemod_FL_Personalize.py --client client0 --num_classes 3 --in_chns 3 --root_path ../data/ODOC_h5 --img_class odoc --exp odoc/FedLPPA --cid 0 --model unet_univ5 --img_size 384 --sup_type scribble --label_prompt 1 &
# python -u test_client4onemod_FL_Personalize.py --client client1 --num_classes 3 --in_chns 3 --root_path ../data/ODOC_h5 --img_class odoc --exp odoc/FedLPPA --cid 1 --model unet_univ5 --img_size 384 --sup_type scribble_noisy --label_prompt 1 &
# python -u test_client4onemod_FL_Personalize.py --client client2 --num_classes 3 --in_chns 3 --root_path ../data/ODOC_h5 --img_class odoc --exp odoc/FedLPPA --cid 2 --model unet_univ5 --img_size 384 --sup_type scribble_noisy --label_prompt 1 &
# python -u test_client4onemod_FL_Personalize.py --client client3 --num_classes 3 --in_chns 3 --root_path ../data/ODOC_h5 --img_class odoc --exp odoc/FedLPPA --cid 3 --model unet_univ5 --img_size 384 --sup_type keypoint --label_prompt 1 &
# python -u test_client4onemod_FL_Personalize.py --client client4 --num_classes 3 --in_chns 3 --root_path ../data/ODOC_h5 --img_class odoc --exp odoc/FedLPPA --cid 4 --model unet_univ5 --img_size 384 --sup_type block --label_prompt 1

