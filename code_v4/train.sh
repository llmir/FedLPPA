python flower_runner.py --port 8094 --procedure flower_pCE_2D_GateCRFMsacleTreeEnergyLoss_Ours --exp faz/FL_WeaklySeg_pCE_flower_FedALALC_Ours --base_lr 0.01 --img_class faz --model unet_lc_multihead --gpus 0 0 0 0 0 0 --iters 5 --eval_iters 5 --rep_iters 2 --alpha 0.1 --img_size 256 --strategy FedALALC




# #prostate
# #pCE Mask(Full)
python Unet_pCE.py --root_path ../data/Prostate_h5 --exp prostate_LFL/Unet_pCE_test_SGD_0.01 --client client1 --sup_type mask --model unet --max_iterations 30000 --batch_size 12 --in_chns 1 --num_classes 2 --base_lr 0.01 --gpus 0 --img_class prostate

python Unet_pCE.py --root_path ../data/Prostate_h5 --exp prostate_LFL/Unet_pCE_test_SGD_0.01 --client client2 --sup_type mask --model unet --max_iterations 30000 --batch_size 12 --in_chns 1 --num_classes 2 --base_lr 0.01 --gpus 0 --img_class prostate

python Unet_pCE.py --root_path ../data/Prostate_h5 --exp prostate_LFL/Unet_pCE_test_SGD_0.01 --client client3 --sup_type mask --model unet --max_iterations 30000 --batch_size 12 --in_chns 1 --num_classes 2 --base_lr 0.01 --gpus 1 --img_class prostate

python Unet_pCE.py --root_path ../data/Prostate_h5 --exp prostate_LFL/Unet_pCE_test_SGD_0.01 --client client4 --sup_type mask --model unet --max_iterations 30000 --batch_size 12 --in_chns 1 --num_classes 2 --base_lr 0.01 --gpus 1 --img_class prostate

python Unet_pCE.py --root_path ../data/Prostate_h5 --exp prostate_LFL/Unet_pCE_test_SGD_0.01 --client client5 --sup_type mask --model unet --max_iterations 30000 --batch_size 12 --in_chns 1 --num_classes 2 --base_lr 0.01 --gpus 2 --img_class prostate

python Unet_pCE.py --root_path ../data/Prostate_h5 --exp prostate_LFL/Unet_pCE_test_SGD_0.01 --client client6 --sup_type mask --model unet --max_iterations 30000 --batch_size 12 --in_chns 1 --num_classes 2 --base_lr 0.01 --gpus 2 --img_class prostate



# #prostate
# #pCE Mask(WSL)
python Unet_pCE.py --root_path ../data/Prostate_h5 --exp prostate_LFL/Unet_pCE_test_AdamW_0.001 --client client1 --sup_type block --model unet --max_iterations 30000 --batch_size 12 --in_chns 1 --num_classes 2 --base_lr 0.001 --gpus 4 --img_class prostate

python Unet_pCE.py --root_path ../data/Prostate_h5 --exp prostate_LFL/Unet_pCE_test_AdamW_0.001 --client client2 --sup_type keypoint --model unet --max_iterations 30000 --batch_size 12 --in_chns 1 --num_classes 2 --base_lr 0.001 --gpus 4 --img_class prostate

python Unet_pCE.py --root_path ../data/Prostate_h5 --exp prostate_LFL/Unet_pCE_test_AdamW_0.001 --client client3 --sup_type scribble --model unet --max_iterations 30000 --batch_size 12 --in_chns 1 --num_classes 2 --base_lr 0.001 --gpus 5 --img_class prostate

python Unet_pCE.py --root_path ../data/Prostate_h5 --exp prostate_LFL/Unet_pCE_test_AdamW_0.001 --client client4 --sup_type keypoint --model unet --max_iterations 30000 --batch_size 12 --in_chns 1 --num_classes 2 --base_lr 0.001 --gpus 5 --img_class prostate

python Unet_pCE.py --root_path ../data/Prostate_h5 --exp prostate_LFL/Unet_pCE_test_AdamW_0.001 --client client5 --sup_type scribble --model unet --max_iterations 30000 --batch_size 12 --in_chns 1 --num_classes 2 --base_lr 0.001 --gpus 6 --img_class prostate

python Unet_pCE.py --root_path ../data/Prostate_h5 --exp prostate_LFL/Unet_pCE_test_AdamW_0.001 --client client6 --sup_type box --model unet --max_iterations 30000 --batch_size 12 --in_chns 1 --num_classes 2 --base_lr 0.001 --gpus 6 --img_class prostate
