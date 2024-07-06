# python flower_runner.py --port 8092 --procedure flower_pCE_TreeEnergyLoss_2D --exp faz/WeaklySeg_pCE_TreeEnergyLoss_unet_head_0.01_1 --base_lr 0.01 --img_class faz --model unet_head --gpus 1 1 2 2 3 3 --iters 10 --eval_iters 20 --tree_loss_weight 1

# python flower_runner.py --port 8092 --procedure flower_pCE_TreeEnergyLoss_2D --exp faz/WeaklySeg_pCE_TreeEnergyLoss_unet_head_0.01_0.5 --base_lr 0.01 --img_class faz --model unet_head --gpus 1 1 2 2 3 3 --iters 10 --eval_iters 20 --tree_loss_weight 0.5


# python flower_runner.py --port 8092 --procedure flower_pCE_2D --exp faz/WeaklySeg_pCE_unet_0.001 --base_lr 0.001 --img_class faz --model unet --gpus 0 0 0 1 1 1 --iters 10 --eval_iters 10 --strategy FedAvg

# python flower_runner.py --port 8093 --procedure flower_pCE_2D --exp faz/WeaklySeg_pCE_unet_0.001_FedBN --base_lr 0.001 --img_class faz --model unet --gpus 2 2 2 3 3 3 --iters 10 --eval_iters 10 --strategy FedBN

# python flower_runner.py --port 8094 --procedure flower_pCE_2D --exp faz/WeaklySeg_pCE_unet_0.001_FedProx --base_lr 0.001 --img_class faz --model unet --gpus 4 4 4 5 5 5 --iters 10 --eval_iters 10 --strategy FedProx

# python flower_runner.py --port 8095 --procedure flower_pCE_2D --exp faz/WeaklySeg_pCE_unet_0.001_FedRep --base_lr 0.001 --img_class faz --model unet --gpus 4 4 4 5 5 5 --iters 10 --eval_iters 10 --strategy FedRep --rep_iters 2

# python flower_runner.py --port 8096 --procedure flower_pCE_2D --exp faz/WeaklySeg_pCE_unet_0.001_FedRep_4 --base_lr 0.001 --img_class faz --model unet --gpus 2 2 2 3 3 3 --iters 10 --eval_iters 10 --strategy FedRep --rep_iters 4

# python flower_runner.py --port 8097 --procedure flower_pCE_2D --exp faz/WeaklySeg_pCE_unet_0.001_FedRep_6 --base_lr 0.001 --img_class faz --model unet --gpus 0 0 0 1 1 1 --iters 10 --eval_iters 10 --strategy FedRep --rep_iters 6


# python flower_runner.py --port 8098 --procedure flower_pCE_2D --exp faz/WeaklySeg_pCE_unet_0.001_FedAdam --base_lr 0.001 --img_class faz --model unet --gpus 0 0 0 1 1 1 --iters 10 --eval_iters 10 --strategy FedAdam

# python flower_runner.py --port 8099 --procedure flower_pCE_2D --exp faz/WeaklySeg_pCE_unet_0.001_FedProx_0.1 --base_lr 0.001 --img_class faz --model unet --gpus 2 2 2 3 3 3 --iters 10 --eval_iters 10 --strategy FedProx --mu 0.1

# python flower_runner.py --port 8100 --procedure flower_pCE_2D --exp faz/WeaklySeg_pCE_unet_0.001_FedAdagrad --base_lr 0.001 --img_class faz --model unet --gpus 2 2 2 3 3 3 --iters 10 --eval_iters 10 --strategy FedAdagrad

# python flower_runner.py --port 8101 --procedure flower_pCE_2D --exp faz/WeaklySeg_pCE_unet_0.001_FedYogi --base_lr 0.001 --img_class faz --model unet --gpus 4 4 4 5 5 5 --iters 10 --eval_iters 10 --strategy FedYogi

# python flower_runner.py --port 8102 --procedure flower_pCE_2D --exp faz/WeaklySeg_pCE_unet_0.001_QFedAvg --base_lr 0.001 --img_class faz --model unet --gpus 2 2 2 3 3 3 --iters 10 --eval_iters 10 --strategy QFedAvg



# python flower_runner.py --port 8103 --procedure flower_pCE_2D --exp odoc/WeaklySeg_pCE_unet_0.01_test --base_lr 0.01 --img_class odoc --model unet --gpus 0 0 1 2 3 4 --iters 10 --eval_iters 10 --strategy FedAvg

# python flower_runner.py --port 8104 --procedure flower_pCE_2D --exp odoc/WeaklySeg_pCE_unet_0.01_FedRep --base_lr 0.01 --img_class odoc --model unet --gpus 0 0 1 2 3 4 --iters 10 --eval_iters 10 --strategy FedRep --rep_iters 6

# python flower_runner.py --port 8105 --procedure flower_pCE_2D --exp odoc/WeaklySeg_pCE_unet_0.01_FedBN --base_lr 0.01 --img_class odoc --model unet --gpus 0 0 1 2 3 4 --iters 10 --eval_iters 10 --strategy FedBN

# python flower_runner.py --port 8106 --procedure flower_pCE_2D --exp odoc/WeaklySeg_pCE_unet_0.01_FedProx_0.1 --base_lr 0.01 --img_class odoc --model unet --gpus 0 0 1 2 3 4 --iters 10 --eval_iters 10 --strategy FedProx --mu 0.1


#python flower_runner.py --port 8103 --procedure flower_pCE_2D --exp faz/WeaklySeg_pCE_unet_0.01_test --base_lr 0.01 --img_class faz --model unet --gpus 2 2 2 3 3 3 --iters 10 --eval_iters 10 --strategy MetaFed --init_iters 600 --common_iters 1000 --lam 1

# python flower_runner.py --port 8104 --procedure flower_pCE_2D --exp faz/WeaklySeg_pCE_unet_0.01_test2 --base_lr 0.01 --img_class faz --model unet --gpus 0 0 2 3 4 5 --iters 10 --eval_iters 10 --strategy FedAP



# python flower_runner.py --port 8105 --procedure flower_pCE_2D --exp faz/WeaklySeg_pCE_unet_test_fed --base_lr 0.01 --img_class faz --model unet --gpus 0 0 2 3 4 5 --iters 5 --eval_iters 5 --strategy FedLC

# python flower_runner.py --port 8105 --procedure flower_pCE_2D --exp faz/WeaklySeg_pCE_unet_test_fed --base_lr 0.01 --img_class faz --model unet_lc --gpus 0 0 2 3 4 5 --iters 5 --eval_iters 5 --rep_iters 2 --alpha 5 --strategy FedLC

# python flower_runner.py --port 8105 --procedure flower_pCE_2D --exp faz/WeaklySeg_pCE_unet_test_fed --base_lr 0.01 --img_class faz --model unet_lc --gpus 0 0 2 3 4 5 --iters 5 --eval_iters 5 --rep_iters 2 --alpha 5 --strategy FedALALC



# python flower_runner.py --port 8099 --procedure flower_pCE_2D --exp faz/WeaklySeg_pCE_debug_feduni_kl_0.1 --base_lr 0.01 --img_class faz --model unet_uni --gpus 5 1 1 1 1 1  --iters 5 --eval_iters 5 --rep_iters 3 --alpha 0.1 --strategy FedUni --debug 0 --tsne_iters 200

python flower_runner.py --port 8194 --procedure flower_pCE_2D --exp faz/WeaklySeg_pCE_debug_feduni_info_lr_0.005 --base_lr 0.005 --img_class faz --model unet_uni --gpus 1 5 5 5 5 5  --iters 5 --eval_iters 5 --rep_iters 3 --alpha 0.5 --strategy FedUni --debug 0 --tsne_iters 200

python flower_runner.py --port 8199 --procedure flower_pCE_2D --exp faz/WeaklySeg_pCE_debug_feduni_info_lr_0.01 --base_lr 0.005 --img_class faz --model unet_uni --gpus 1 5 5 5 5 5  --iters 5 --eval_iters 5 --rep_iters 3 --alpha 0.5 --strategy FedUni --debug 0 --tsne_iters 200

python flower_runner.py --port 8199 --procedure flower_pCE_2D --exp faz/WeaklySeg_pCE_debug_feduniv2 --base_lr 0.01 --img_class faz --model unet_univ2 --gpus 3 3 1 1 2 2  --iters 5 --eval_iters 5 --rep_iters 3 --alpha 0.5 --beta 0.5 --prompt onehot --attention dual --label_prompt  --strategy FedUniV2 --debug 0 --tsne_iters 

python flower_runner.py --port 8199 --procedure flower_pCE_2D --exp faz/WeaklySeg_pCE_debug_feduniv2 --base_lr 0.01 --img_class faz --model unet_univ2 --gpus 3 3 1 1 2 2  --iters 5 --eval_iters 5 --rep_iters 3 --alpha 0.5 --beta 0.5 --prompt universal --attention dual --dual_init adjacent --label_prompt 0 --strategy FedUniV2 --debug 0 --tsne_iters 200

python flower_runner.py --port 8199 --procedure flower_pCE_2D --exp faz/WeaklySeg_pCE_debug_feduniv2.1 --base_lr 0.01 --img_class faz --model unet_univ2 --gpus 3 3 1 1 2 2  --iters 5 --eval_iters 5 --rep_iters 3 --alpha 0.5 --beta 0.5 --prompt universal --attention dual --dual_init aggregated --label_prompt 0 --strategy FedUniV2.1 --debug 0 --tsne_iters 200

# gatedcrf faz
python flower_runner_v2.1.py --port 8199 --procedure flower_pCE_2D_GatedCRFLoss_v2.1 --exp faz/WeaklySeg_pCE_GatedCRFLoss_feduniv2.1 --base_lr 0.01 --img_class faz --model unet_univ2 --gpus 0 1 2 3 4 5  --iters 5 --eval_iters 5 --rep_iters 3 --alpha 0.5 --beta 0.5 --prompt universal --attention dual --dual_init aggregated --label_prompt 1 --strategy FedUniV2.1 --debug 0 --tsne_iters 200 --img_size 256

# gatedcrf odoc
python flower_runner_v2.1.py --port 8199 --procedure flower_pCE_2D_GatedCRFLoss_v2.1 --exp odoc/WeaklySeg_pCE_GatedCRFLoss_feduniv2.1_pseudo_detach --base_lr 0.01 --img_class odoc --model unet_univ2 --gpus 0 1 2 3 4 5  --iters 5 --eval_iters 5 --rep_iters 3 --alpha 0.5 --beta 0.5 --prompt universal --attention dual --dual_init aggregated --label_prompt 1 --strategy FedUniV2.1 --debug 0 --tsne_iters 200 --img_size 384

# gatedcrf polyp
python flower_runner_v2.1.py --port 8198 --procedure flower_pCE_2D_GatedCRFLoss_v2.1 --exp polyp/WeaklySeg_pCE_GatedCRFLoss_feduniv2.1 --base_lr 0.01 --img_class polyp --model unet_univ2 --gpus 0 1 2 3 4 5  --iters 5 --eval_iters 5 --rep_iters 3 --alpha 0.5 --beta 0.5 --prompt universal --attention dual --dual_init aggregated --label_prompt 1 --strategy FedUniV2.1 --debug 0 --tsne_iters 200 --img_size 384



# gatedcrf odoc uni v1
python flower_runner_v2.1.py --port 8199 --procedure flower_pCE_2D_GatedCRFLoss_v2.1 --exp odoc/WeaklySeg_pCE_GatedCRFLoss_feduniv1 --base_lr 0.01 --img_class odoc --model unet_uni --gpus 2 3 4 5 6 7  --iters 5 --eval_iters 5 --rep_iters 3 --alpha 0.5 --strategy FedUni --debug 0 --tsne_iters 200 --img_size 384

python flower_pCE_2D_GatedCRFLoss_v1.py --root_path ../data/ODOC_h5 --num_classes 3 --in_chns 3 --img_class odoc --exp odoc/WeaklySeg_pCE_GatedCRFLoss_feduniv1 --model unet_uni --max_iterations 30000 --iters 5 --eval_iters 5 --tsne_iters 200 --batch_size 12 --base_lr 0.01 --amp 0 --server_address [::]:8199 --strategy FedUni --min_num_clients 5 --img_size 384 --alpha 0.5 --rep_iters 3 --role server --client client_all --sup_type mask --gpu 2

python flower_pCE_2D_GatedCRFLoss_v1.py --root_path ../data/ODOC_h5 --num_classes 3 --in_chns 3 --img_class odoc --exp odoc/WeaklySeg_pCE_GatedCRFLoss_feduniv1 --model unet_uni --max_iterations 30000 --iters 5 --eval_iters 5 --tsne_iters 200 --batch_size 12 --base_lr 0.01 --amp 0 --server_address [::]:8199 --strategy FedUni --min_num_clients 5 --img_size 384 --alpha 0.5 --rep_iters 3 --role client --cid 0 --client client1 --sup_type scribble --gpu 3

python flower_pCE_2D_GatedCRFLoss_v1.py --root_path ../data/ODOC_h5 --num_classes 3 --in_chns 3 --img_class odoc --exp odoc/WeaklySeg_pCE_GatedCRFLoss_feduniv1 --model unet_uni --max_iterations 30000 --iters 5 --eval_iters 5 --tsne_iters 200 --batch_size 12 --base_lr 0.01 --amp 0 --server_address [::]:8199 --strategy FedUni --min_num_clients 5 --img_size 384 --alpha 0.5 --rep_iters 3 --role client --cid 1 --client client2 --sup_type scribble_noisy --gpu 4

python flower_pCE_2D_GatedCRFLoss_v1.py --root_path ../data/ODOC_h5 --num_classes 3 --in_chns 3 --img_class odoc --exp odoc/WeaklySeg_pCE_GatedCRFLoss_feduniv1 --model unet_uni --max_iterations 30000 --iters 5 --eval_iters 5 --tsne_iters 200 --batch_size 12 --base_lr 0.01 --amp 0 --server_address [::]:8199 --strategy FedUni --min_num_clients 5 --img_size 384 --alpha 0.5 --rep_iters 3 --role client --cid 2 --client client3 --sup_type scribble_noisy --gpu 5

python flower_pCE_2D_GatedCRFLoss_v1.py --root_path ../data/ODOC_h5 --num_classes 3 --in_chns 3 --img_class odoc --exp odoc/WeaklySeg_pCE_GatedCRFLoss_feduniv1 --model unet_uni --max_iterations 30000 --iters 5 --eval_iters 5 --tsne_iters 200 --batch_size 12 --base_lr 0.01 --amp 0 --server_address [::]:8199 --strategy FedUni --min_num_clients 5 --img_size 384 --alpha 0.5 --rep_iters 3 --role client --cid 3 --client client4 --sup_type keypoint --gpu 6

python flower_pCE_2D_GatedCRFLoss_v1.py --root_path ../data/ODOC_h5 --num_classes 3 --in_chns 3 --img_class odoc --exp odoc/WeaklySeg_pCE_GatedCRFLoss_feduniv1 --model unet_uni --max_iterations 30000 --iters 5 --eval_iters 5 --tsne_iters 200 --batch_size 12 --base_lr 0.01 --amp 0 --server_address [::]:8199 --strategy FedUni --min_num_clients 5 --img_size 384 --alpha 0.5 --rep_iters 3 --role client --cid 4 --client client5 --sup_type block --gpu 7



# gatedcrf polyp uni v1
python flower_runner_v1.py --port 8199 --procedure flower_pCE_2D_GatedCRFLoss_v1 --exp odoc/WeaklySeg_pCE_GatedCRFLoss_feduniv1 --base_lr 0.01 --img_class odoc --model unet_uni --gpus 2 3 4 5 6 7  --iters 5 --eval_iters 5 --rep_iters 3 --alpha 0.5 --strategy FedUni --debug 0 --tsne_iters 200 --img_size 384


# 
python flower_runner_v2.1.py --port 8198 --procedure flower_pCE_2D_GatedCRFLoss_Pseudo_v2.1 --exp faz/WeaklySeg_pCE_PseudoGatedCRFLoss_feduniv2.1_detach --base_lr 0.01 --img_class faz --model unet_univ2 --gpus 0 0 0 6 6 7 --iters 5 --eval_iters 5 --rep_iters 3 --alpha 0.5 --beta 0.5 --prompt universal --attention dual --dual_init aggregated --label_prompt 1 --strategy FedUniV2.1 --debug 0 --tsne_iters 200 --img_size 256


# odoc ft
python flower_FedAvg_pCE_2D_GatedCRFLoss_FT_Unet.py --root_path ../data/ODOC_h5 --exp odoc/WeaklySeg_pCE_GatedCRFLoss_feduniv2.1_FT --client client0 --sup_type scribble --model unet_univ2 --in_chns 3 --num_classes 3 --in_chns 3 --base_lr 0.0001 --gpus 2 --img_class odoc --amp 0 --img_size 384 --batch_size 12  --cid 0 --prompt universal --attention dual --dual_init aggregated --label_prompt 1 --min_num_clients 5


python flower_FedAvg_pCE_2D_GatedCRFLoss_FT.py --root_path ../data/ODOC_h5 --exp odoc/WeaklySeg_pCE_GatedCRFLoss_feduniv2.1_FT --client client0 --sup_type scribble --model unet_univ2 --in_chns 3 --num_classes 3 --in_chns 3 --base_lr 0.0001 --gpus 0 --img_class odoc --amp 0 --img_size 384 --batch_size 12  --cid 0 --prompt universal --attention dual --dual_init aggregated --label_prompt 1 --min_num_clients 5

python flower_FedAvg_pCE_2D_GatedCRFLoss_FT_Unet.py --root_path ../data/ODOC_h5 --exp odoc/WeaklySeg_pCE_GatedCRFLoss_feduniv2.1_FT --client client0 --sup_type scribble --model unet_320 --in_chns 3 --num_classes 3 --in_chns 3 --base_lr 0.0001 --gpus 5 --img_class odoc --amp 0 --img_size 384 --batch_size 12  --cid 0

# ft gatedcrf
# ft pCE
python flower_FedAvg_pCE_2D_GatedCRFLoss_FT.py --root_path ../data/ODOC_h5 --exp odoc/WeaklySeg_pCE_GatedCRFLoss_feduniv2.1_FT --client client0 --sup_type scribble --model unet_univ2 --in_chns 3 --num_classes 3 --in_chns 3 --base_lr 0.0001 --gpus 0 --img_class odoc --amp 0 --img_size 384 --batch_size 12  --cid 0 --prompt universal --attention dual --dual_init aggregated --label_prompt 1 --min_num_clients 5

python flower_FedAvg_pCE_2D_GatedCRFLoss_FT.py --root_path ../data/ODOC_h5 --exp odoc/WeaklySeg_pCE_GatedCRFLoss_feduniv2.1_FT --client client1 --sup_type scribble_noisy --model unet_univ2 --in_chns 3 --num_classes 3 --in_chns 3 --base_lr 0.0001 --gpus 1 --img_class odoc --amp 0 --img_size 384 --batch_size 12  --cid 1 --prompt universal --attention dual --dual_init aggregated --label_prompt 1 --min_num_clients 5

python flower_FedAvg_pCE_2D_GatedCRFLoss_FT.py --root_path ../data/ODOC_h5 --exp odoc/WeaklySeg_pCE_GatedCRFLoss_feduniv2.1_FT --client client2 --sup_type scribble_noisy --model unet_univ2 --in_chns 3 --num_classes 3 --in_chns 3 --base_lr 0.0001 --gpus 2 --img_class odoc --amp 0 --img_size 384 --batch_size 12  --cid 2 --prompt universal --attention dual --dual_init aggregated --label_prompt 1 --min_num_clients 5

python flower_FedAvg_pCE_2D_GatedCRFLoss_FT.py --root_path ../data/ODOC_h5 --exp odoc/WeaklySeg_pCE_GatedCRFLoss_feduniv2.1_FT --client client3 --sup_type keypoint --model unet_univ2 --in_chns 3 --num_classes 3 --in_chns 3 --base_lr 0.0001 --gpus 3 --img_class odoc --amp 0 --img_size 384 --batch_size 12  --cid 3 --prompt universal --attention dual --dual_init aggregated --label_prompt 1 --min_num_clients 5

python flower_FedAvg_pCE_2D_GatedCRFLoss_FT.py --root_path ../data/ODOC_h5 --exp odoc/WeaklySeg_pCE_GatedCRFLoss_feduniv2.1_FT --client client4 --sup_type block --model unet_univ2 --in_chns 3 --num_classes 3 --in_chns 3 --base_lr 0.0001 --gpus 4 --img_class odoc --amp 0 --img_size 384 --batch_size 12  --cid 4 --prompt universal --attention dual --dual_init aggregated --label_prompt 1 --min_num_clients 5


# odoc
python flower_pCE_2D_GatedCRFLoss_v2.1_auxpCE.py --root_path ../data/ODOC_h5 --num_classes 3 --in_chns 3 --img_class odoc --exp odoc/WeaklySeg_pCE_2D_GatedCRFLoss_v2.1_auxpCE_detach_UniPrompt_Selection --model unet_univ2 --max_iterations 30000 --iters 5 --eval_iters 5 --tsne_iters 200 --batch_size 12 --base_lr 0.01 --amp 0 --server_address 127.0.0.1:8198 --strategy FedUniV2.1 --min_num_clients 5 --img_size 384 --alpha 0.5 --beta 0.5 --prompt universal --attention dual --dual_init aggregated --label_prompt 1 --role server --client client_all --sup_type mask --gpu 7

python flower_pCE_2D_GatedCRFLoss_v2.1_auxpCE.py --root_path ../data/ODOC_h5 --num_classes 3 --in_chns 3 --img_class odoc --exp odoc/WeaklySeg_pCE_2D_GatedCRFLoss_v2.1_auxpCE_detach_UniPrompt_Selection --model unet_univ2 --max_iterations 30000 --iters 5 --eval_iters 5 --tsne_iters 200 --batch_size 12 --base_lr 0.01 --amp 0 --server_address 127.0.0.1:8198 --strategy FedUniV2.1 --min_num_clients 5 --img_size 384 --alpha 0.5 --beta 0.5 --prompt universal --attention dual --dual_init aggregated --label_prompt 1 --role client --cid 0 --client client1 --sup_type scribble --gpu 2

python flower_pCE_2D_GatedCRFLoss_v2.1_auxpCE.py --root_path ../data/ODOC_h5 --num_classes 3 --in_chns 3 --img_class odoc --exp odoc/WeaklySeg_pCE_2D_GatedCRFLoss_v2.1_auxpCE_detach_UniPrompt_Selection --model unet_univ2 --max_iterations 30000 --iters 5 --eval_iters 5 --tsne_iters 200 --batch_size 12 --base_lr 0.01 --amp 0 --server_address 127.0.0.1:8198 --strategy FedUniV2.1 --min_num_clients 5 --img_size 384 --alpha 0.5 --beta 0.5 --prompt universal --attention dual --dual_init aggregated --label_prompt 1 --role client --cid 1 --client client2 --sup_type scribble_noisy --gpu 3

python flower_pCE_2D_GatedCRFLoss_v2.1_auxpCE.py --root_path ../data/ODOC_h5 --num_classes 3 --in_chns 3 --img_class odoc --exp odoc/WeaklySeg_pCE_2D_GatedCRFLoss_v2.1_auxpCE_detach_UniPrompt_Selection --model unet_univ2 --max_iterations 30000 --iters 5 --eval_iters 5 --tsne_iters 200 --batch_size 12 --base_lr 0.01 --amp 0 --server_address 127.0.0.1:8198 --strategy FedUniV2.1 --min_num_clients 5 --img_size 384 --alpha 0.5 --beta 0.5 --prompt universal --attention dual --dual_init aggregated --label_prompt 1 --role client --cid 2 --client client3 --sup_type scribble_noisy --gpu 4

python flower_pCE_2D_GatedCRFLoss_v2.1_auxpCE.py --root_path ../data/ODOC_h5 --num_classes 3 --in_chns 3 --img_class odoc --exp odoc/WeaklySeg_pCE_2D_GatedCRFLoss_v2.1_auxpCE_detach_UniPrompt_Selection --model unet_univ2 --max_iterations 30000 --iters 5 --eval_iters 5 --tsne_iters 200 --batch_size 12 --base_lr 0.01 --amp 0 --server_address 127.0.0.1:8198 --strategy FedUniV2.1 --min_num_clients 5 --img_size 384 --alpha 0.5 --beta 0.5 --prompt universal --attention dual --dual_init aggregated --label_prompt 1 --role client --cid 3 --client client4 --sup_type keypoint --gpu 5

python flower_pCE_2D_GatedCRFLoss_v2.1_auxpCE.py --root_path ../data/ODOC_h5 --num_classes 3 --in_chns 3 --img_class odoc --exp odoc/WeaklySeg_pCE_2D_GatedCRFLoss_v2.1_auxpCE_detach_UniPrompt_Selection --model unet_univ2 --max_iterations 30000 --iters 5 --eval_iters 5 --tsne_iters 200 --batch_size 12 --base_lr 0.01 --amp 0 --server_address 127.0.0.1:8198 --strategy FedUniV2.1 --min_num_clients 5 --img_size 384 --alpha 0.5 --beta 0.5 --prompt universal --attention dual --dual_init aggregated --label_prompt 1 --role client --cid 4 --client client5 --sup_type block --gpu 6


# gatedcrf polyp auxi detach 
python flower_pCE_2D_GatedCRFLoss_v2.1_auxpCE.py --root_path ../data/Polypdata_h5 --num_classes 2 --in_chns 3 --img_class polyp --exp polyp/WeaklySeg_pCE_2D_GatedCRFLoss_v3_auxpCE_Prompt_Distance --model unet_univ3 --max_iterations 30000 --iters 5 --eval_iters 5 --tsne_iters 200 --batch_size 12 --base_lr 0.001 --amp 0 --server_address 127.0.0.1:8199 --strategy FedUniV2.1 --min_num_clients 4 --img_size 384 --alpha 0.5 --beta 0.1 --prompt universal --attention dual --dual_init aggregated --label_prompt 1 --role server --client client_all --sup_type mask --gpu 0

python flower_pCE_2D_GatedCRFLoss_v2.1_auxpCE.py --root_path ../data/Polypdata_h5 --num_classes 2 --in_chns 3 --img_class polyp --exp polyp/WeaklySeg_pCE_2D_GatedCRFLoss_v3_auxpCE_Prompt_Distance --model unet_univ3 --max_iterations 30000 --iters 5 --eval_iters 5 --tsne_iters 200 --batch_size 12 --base_lr 0.001 --amp 0 --server_address 127.0.0.1:8199 --strategy FedUniV2.1 --min_num_clients 4 --img_size 384 --alpha 0.5 --beta 0.1 --prompt universal --attention dual --dual_init aggregated --label_prompt 1 --role client --cid 0 --client client1 --sup_type keypoint --gpu 2

python flower_pCE_2D_GatedCRFLoss_v2.1_auxpCE.py --root_path ../data/Polypdata_h5 --num_classes 2 --in_chns 3 --img_class polyp --exp polyp/WeaklySeg_pCE_2D_GatedCRFLoss_v3_auxpCE_Prompt_Distance --model unet_univ3 --max_iterations 30000 --iters 5 --eval_iters 5 --tsne_iters 200 --batch_size 12 --base_lr 0.001 --amp 0 --server_address 127.0.0.1:8199 --strategy FedUniV2.1 --min_num_clients 4 --img_size 384 --alpha 0.5 --beta 0.1 --prompt universal --attention dual --dual_init aggregated --label_prompt 1 --role client --cid 1 --client client2 --sup_type scribble --gpu 3

python flower_pCE_2D_GatedCRFLoss_v2.1_auxpCE.py --root_path ../data/Polypdata_h5 --num_classes 2 --in_chns 3 --img_class polyp --exp polyp/WeaklySeg_pCE_2D_GatedCRFLoss_v3_auxpCE_Prompt_Distance --model unet_univ3 --max_iterations 30000 --iters 5 --eval_iters 5 --tsne_iters 200 --batch_size 12 --base_lr 0.001 --amp 0 --server_address 127.0.0.1:8199 --strategy FedUniV2.1 --min_num_clients 4 --img_size 384 --alpha 0.5 --beta 0.1 --prompt universal --attention dual --dual_init aggregated --label_prompt 1 --role client --cid 2 --client client3 --sup_type box --gpu 5

python flower_pCE_2D_GatedCRFLoss_v2.1_auxpCE.py --root_path ../data/Polypdata_h5 --num_classes 2 --in_chns 3 --img_class polyp --exp polyp/WeaklySeg_pCE_2D_GatedCRFLoss_v3_auxpCE_Prompt_Distance --model unet_univ3 --max_iterations 30000 --iters 5 --eval_iters 5 --tsne_iters 200 --batch_size 12 --base_lr 0.001 --amp 0 --server_address 127.0.0.1:8199 --strategy FedUniV2.1 --min_num_clients 4 --img_size 384 --alpha 0.5 --beta 0.1 --prompt universal --attention dual --dual_init aggregated --label_prompt 1 --role client --cid 3 --client client4 --sup_type block --gpu 6

# odoc
python flower_pCE_2D_GatedCRFLoss_v2.1_auxpCE.py --root_path ../data/ODOC_h5 --num_classes 3 --in_chns 3 --img_class odoc --exp odoc/WeaklySeg_pCE_2D_GatedCRFLoss_v2.1_auxpCE_Prompt_Distance --model unet_univ3 --max_iterations 30000 --iters 5 --eval_iters 5 --tsne_iters 200 --batch_size 12 --base_lr 0.01 --amp 0 --server_address 127.0.0.1:8091 --strategy FedUniV2.1 --min_num_clients 5 --img_size 384 --alpha 0.5 --beta 0.5 --prompt universal --attention dual --dual_init aggregated --label_prompt 1 --role server --client client_all --sup_type mask --gpu 0

python flower_pCE_2D_GatedCRFLoss_v2.1_auxpCE.py --root_path ../data/ODOC_h5 --num_classes 3 --in_chns 3 --img_class odoc --exp odoc/WeaklySeg_pCE_2D_GatedCRFLoss_v2.1_auxpCE_Prompt_Distance --model unet_univ3 --max_iterations 30000 --iters 5 --eval_iters 5 --tsne_iters 200 --batch_size 12 --base_lr 0.01 --amp 0 --server_address 127.0.0.1:8091 --strategy FedUniV2.1 --min_num_clients 5 --img_size 384 --alpha 0.5 --beta 0.5 --prompt universal --attention dual --dual_init aggregated --label_prompt 1 --role client --cid 0 --client client1 --sup_type scribble --gpu 0

python flower_pCE_2D_GatedCRFLoss_v2.1_auxpCE.py --root_path ../data/ODOC_h5 --num_classes 3 --in_chns 3 --img_class odoc --exp odoc/WeaklySeg_pCE_2D_GatedCRFLoss_v2.1_auxpCE_Prompt_Distance --model unet_univ3 --max_iterations 30000 --iters 5 --eval_iters 5 --tsne_iters 200 --batch_size 12 --base_lr 0.01 --amp 0 --server_address 127.0.0.1:8091 --strategy FedUniV2.1 --min_num_clients 5 --img_size 384 --alpha 0.5 --beta 0.5 --prompt universal --attention dual --dual_init aggregated --label_prompt 1 --role client --cid 1 --client client2 --sup_type scribble_noisy --gpu 1

python flower_pCE_2D_GatedCRFLoss_v2.1_auxpCE.py --root_path ../data/ODOC_h5 --num_classes 3 --in_chns 3 --img_class odoc --exp odoc/WeaklySeg_pCE_2D_GatedCRFLoss_v2.1_auxpCE_Prompt_Distance --model unet_univ3 --max_iterations 30000 --iters 5 --eval_iters 5 --tsne_iters 200 --batch_size 12 --base_lr 0.01 --amp 0 --server_address 127.0.0.1:8091 --strategy FedUniV2.1 --min_num_clients 5 --img_size 384 --alpha 0.5 --beta 0.5 --prompt universal --attention dual --dual_init aggregated --label_prompt 1 --role client --cid 2 --client client3 --sup_type scribble_noisy --gpu 2

python flower_pCE_2D_GatedCRFLoss_v2.1_auxpCE.py --root_path ../data/ODOC_h5 --num_classes 3 --in_chns 3 --img_class odoc --exp odoc/WeaklySeg_pCE_2D_GatedCRFLoss_v2.1_auxpCE_Prompt_Distance --model unet_univ3 --max_iterations 30000 --iters 5 --eval_iters 5 --tsne_iters 200 --batch_size 12 --base_lr 0.01 --amp 0 --server_address 127.0.0.1:8091 --strategy FedUniV2.1 --min_num_clients 5 --img_size 384 --alpha 0.5 --beta 0.5 --prompt universal --attention dual --dual_init aggregated --label_prompt 1 --role client --cid 3 --client client4 --sup_type keypoint --gpu 3

python flower_pCE_2D_GatedCRFLoss_v2.1_auxpCE.py --root_path ../data/ODOC_h5 --num_classes 3 --in_chns 3 --img_class odoc --exp odoc/WeaklySeg_pCE_2D_GatedCRFLoss_v2.1_auxpCE_Prompt_Distance --model unet_univ3 --max_iterations 30000 --iters 5 --eval_iters 5 --tsne_iters 200 --batch_size 12 --base_lr 0.01 --amp 0 --server_address 127.0.0.1:8091 --strategy FedUniV2.1 --min_num_clients 5 --img_size 384 --alpha 0.5 --beta 0.5 --prompt universal --attention dual --dual_init aggregated --label_prompt 1 --role client --cid 4 --client client5 --sup_type block --gpu 5


# faz
python flower_runner_v2.1.py --port 8199 --procedure flower_pCE_2D_GatedCRFLoss_v2.1_auxpCE --exp faz/WeaklySeg_pCE_2D_GatedCRFLoss_v3_auxpCE_detach_Prompt_Distance --base_lr 0.01 --img_class faz --model unet_univ3 --gpus 0 0 0 1 1 1 --iters 5 --eval_iters 5 --rep_iters 3 --alpha 0.5 --beta 0.5 --prompt universal --attention dual --dual_init aggregated --label_prompt 1 --strategy FedUniV2.1 --debug 1 --tsne_iters 200 --img_size 256

python flower_pCE_2D_GatedCRFLoss_v2.1_auxpCE.py --root_path ../data/FAZ_h5 --num_classes 2 --in_chns 1 --img_class faz --exp faz/WeaklySeg_pCE_2D_GatedCRFLoss_v3_auxpCE_detach_Prompt_Distance --model unet_univ3 --max_iterations 30000 --iters 5 --eval_iters 5 --tsne_iters 200 --batch_size 12 --base_lr 0.01 --amp 0 --server_address [::]:8199 --strategy FedUniV2.1 --min_num_clients 5 --img_size 256 --alpha 0.5 --beta 0.5 --prompt universal --attention dual --dual_init aggregated --label_prompt 1 --role server --client client_all --sup_type mask --gpu 0

python flower_pCE_2D_GatedCRFLoss_v2.1_auxpCE.py --root_path ../data/FAZ_h5 --num_classes 2 --in_chns 1 --img_class faz --exp faz/WeaklySeg_pCE_2D_GatedCRFLoss_v3_auxpCE_detach_Prompt_Distance --model unet_univ3 --max_iterations 30000 --iters 5 --eval_iters 5 --tsne_iters 200 --batch_size 12 --base_lr 0.01 --amp 0 --server_address [::]:8199 --strategy FedUniV2.1 --min_num_clients 5 --img_size 256 --alpha 0.5 --beta 0.5 --prompt universal --attention dual --dual_init aggregated --label_prompt 1 --role client --cid 0 --client client1 --sup_type scribble_noisy --gpu 0

python flower_pCE_2D_GatedCRFLoss_v2.1_auxpCE.py --root_path ../data/FAZ_h5 --num_classes 2 --in_chns 1 --img_class faz --exp faz/WeaklySeg_pCE_2D_GatedCRFLoss_v3_auxpCE_detach_Prompt_Distance --model unet_univ3 --max_iterations 30000 --iters 5 --eval_iters 5 --tsne_iters 200 --batch_size 12 --base_lr 0.01 --amp 0 --server_address [::]:8199 --strategy FedUniV2.1 --min_num_clients 5 --img_size 256 --alpha 0.5 --beta 0.5 --prompt universal --attention dual --dual_init aggregated --label_prompt 1 --role client --cid 1 --client client2 --sup_type keypoint --gpu 0

python flower_pCE_2D_GatedCRFLoss_v2.1_auxpCE.py --root_path ../data/FAZ_h5 --num_classes 2 --in_chns 1 --img_class faz --exp faz/WeaklySeg_pCE_2D_GatedCRFLoss_v3_auxpCE_detach_Prompt_Distance --model unet_univ3 --max_iterations 30000 --iters 5 --eval_iters 5 --tsne_iters 200 --batch_size 12 --base_lr 0.01 --amp 0 --server_address [::]:8199 --strategy FedUniV2.1 --min_num_clients 5 --img_size 256 --alpha 0.5 --beta 0.5 --prompt universal --attention dual --dual_init aggregated --label_prompt 1 --role client --cid 2 --client client3 --sup_type block --gpu 1

python flower_pCE_2D_GatedCRFLoss_v2.1_auxpCE.py --root_path ../data/FAZ_h5 --num_classes 2 --in_chns 1 --img_class faz --exp faz/WeaklySeg_pCE_2D_GatedCRFLoss_v3_auxpCE_detach_Prompt_Distance --model unet_univ3 --max_iterations 30000 --iters 5 --eval_iters 5 --tsne_iters 200 --batch_size 12 --base_lr 0.01 --amp 0 --server_address [::]:8199 --strategy FedUniV2.1 --min_num_clients 5 --img_size 256 --alpha 0.5 --beta 0.5 --prompt universal --attention dual --dual_init aggregated --label_prompt 1 --role client --cid 3 --client client4 --sup_type box --gpu 1

python flower_pCE_2D_GatedCRFLoss_v2.1_auxpCE.py --root_path ../data/FAZ_h5 --num_classes 2 --in_chns 1 --img_class faz --exp faz/WeaklySeg_pCE_2D_GatedCRFLoss_v3_auxpCE_detach_Prompt_Distance --model unet_univ3 --max_iterations 30000 --iters 5 --eval_iters 5 --tsne_iters 200 --batch_size 12 --base_lr 0.01 --amp 0 --server_address [::]:8199 --strategy FedUniV2.1 --min_num_clients 5 --img_size 256 --alpha 0.5 --beta 0.5 --prompt universal --attention dual --dual_init aggregated --label_prompt 1 --role client --cid 4 --client client5 --sup_type scribble --gpu 1




# odoc no Detach
python flower_pCE_2D_GatedCRFLoss_v2.1_auxpCE_noDetach.py --root_path ../data/ODOC_h5 --num_classes 3 --in_chns 3 --img_class odoc --exp odoc/WeaklySeg_pCE_2D_GatedCRFLoss_v2.1_auxpCE_Prompt_Distance_noDetach --model unet_univ3 --max_iterations 30000 --iters 5 --eval_iters 5 --tsne_iters 200 --batch_size 12 --base_lr 0.01 --amp 0 --server_address 127.0.0.1:8091 --strategy FedUniV2.1 --min_num_clients 5 --img_size 384 --alpha 0.5 --beta 0.5 --prompt universal --attention dual --dual_init aggregated --label_prompt 1 --role server --client client_all --sup_type mask --gpu 0

python flower_pCE_2D_GatedCRFLoss_v2.1_auxpCE_noDetach.py --root_path ../data/ODOC_h5 --num_classes 3 --in_chns 3 --img_class odoc --exp odoc/WeaklySeg_pCE_2D_GatedCRFLoss_v2.1_auxpCE_Prompt_Distance_noDetach --model unet_univ3 --max_iterations 30000 --iters 5 --eval_iters 5 --tsne_iters 200 --batch_size 12 --base_lr 0.01 --amp 0 --server_address 127.0.0.1:8091 --strategy FedUniV2.1 --min_num_clients 5 --img_size 384 --alpha 0.5 --beta 0.5 --prompt universal --attention dual --dual_init aggregated --label_prompt 1 --role client --cid 0 --client client1 --sup_type scribble --gpu 0

python flower_pCE_2D_GatedCRFLoss_v2.1_auxpCE_noDetach.py --root_path ../data/ODOC_h5 --num_classes 3 --in_chns 3 --img_class odoc --exp odoc/WeaklySeg_pCE_2D_GatedCRFLoss_v2.1_auxpCE_Prompt_Distance_noDetach --model unet_univ3 --max_iterations 30000 --iters 5 --eval_iters 5 --tsne_iters 200 --batch_size 12 --base_lr 0.01 --amp 0 --server_address 127.0.0.1:8091 --strategy FedUniV2.1 --min_num_clients 5 --img_size 384 --alpha 0.5 --beta 0.5 --prompt universal --attention dual --dual_init aggregated --label_prompt 1 --role client --cid 1 --client client2 --sup_type scribble_noisy --gpu 1

python flower_pCE_2D_GatedCRFLoss_v2.1_auxpCE_noDetach.py --root_path ../data/ODOC_h5 --num_classes 3 --in_chns 3 --img_class odoc --exp odoc/WeaklySeg_pCE_2D_GatedCRFLoss_v2.1_auxpCE_Prompt_Distance_noDetach --model unet_univ3 --max_iterations 30000 --iters 5 --eval_iters 5 --tsne_iters 200 --batch_size 12 --base_lr 0.01 --amp 0 --server_address 127.0.0.1:8091 --strategy FedUniV2.1 --min_num_clients 5 --img_size 384 --alpha 0.5 --beta 0.5 --prompt universal --attention dual --dual_init aggregated --label_prompt 1 --role client --cid 2 --client client3 --sup_type scribble_noisy --gpu 2

python flower_pCE_2D_GatedCRFLoss_v2.1_auxpCE_noDetach.py --root_path ../data/ODOC_h5 --num_classes 3 --in_chns 3 --img_class odoc --exp odoc/WeaklySeg_pCE_2D_GatedCRFLoss_v2.1_auxpCE_Prompt_Distance_noDetach --model unet_univ3 --max_iterations 30000 --iters 5 --eval_iters 5 --tsne_iters 200 --batch_size 12 --base_lr 0.01 --amp 0 --server_address 127.0.0.1:8091 --strategy FedUniV2.1 --min_num_clients 5 --img_size 384 --alpha 0.5 --beta 0.5 --prompt universal --attention dual --dual_init aggregated --label_prompt 1 --role client --cid 3 --client client4 --sup_type keypoint --gpu 3

python flower_pCE_2D_GatedCRFLoss_v2.1_auxpCE_noDetach.py --root_path ../data/ODOC_h5 --num_classes 3 --in_chns 3 --img_class odoc --exp odoc/WeaklySeg_pCE_2D_GatedCRFLoss_v2.1_auxpCE_Prompt_Distance_noDetach --model unet_univ3 --max_iterations 30000 --iters 5 --eval_iters 5 --tsne_iters 200 --batch_size 12 --base_lr 0.01 --amp 0 --server_address 127.0.0.1:8091 --strategy FedUniV2.1 --min_num_clients 5 --img_size 384 --alpha 0.5 --beta 0.5 --prompt universal --attention dual --dual_init aggregated --label_prompt 1 --role client --cid 4 --client client5 --sup_type block --gpu 5


# faz nearest

python flower_pCE_2D_GatedCRFLoss_v2.1_auxpCE.py --root_path ../data/FAZ_h5 --num_classes 2 --in_chns 1 --img_class faz --exp faz/WeaklySeg_pCE_2D_GatedCRFLoss_v3_auxpCE_detach_Prompt_Distance_Nearest --model unet_univ3 --max_iterations 30000 --iters 5 --eval_iters 5 --tsne_iters 200 --batch_size 12 --base_lr 0.01 --amp 0 --server_address 127.0.0.1:8092 --strategy FedUniV2.1 --min_num_clients 5 --img_size 256 --alpha 0.5 --beta 0.5 --prompt universal --attention dual --dual_init aggregated --label_prompt 1 --role server --client client_all --sup_type mask --gpu 0

python flower_pCE_2D_GatedCRFLoss_v2.1_auxpCE.py --root_path ../data/FAZ_h5 --num_classes 2 --in_chns 1 --img_class faz --exp faz/WeaklySeg_pCE_2D_GatedCRFLoss_v3_auxpCE_detach_Prompt_Distance_Nearest --model unet_univ3 --max_iterations 30000 --iters 5 --eval_iters 5 --tsne_iters 200 --batch_size 12 --base_lr 0.01 --amp 0 --server_address 127.0.0.1:8092 --strategy FedUniV2.1 --min_num_clients 5 --img_size 256 --alpha 0.5 --beta 0.5 --prompt universal --attention dual --dual_init aggregated --label_prompt 1 --role client --cid 0 --client client1 --sup_type scribble_noisy --gpu 0

python flower_pCE_2D_GatedCRFLoss_v2.1_auxpCE.py --root_path ../data/FAZ_h5 --num_classes 2 --in_chns 1 --img_class faz --exp faz/WeaklySeg_pCE_2D_GatedCRFLoss_v3_auxpCE_detach_Prompt_Distance_Nearest --model unet_univ3 --max_iterations 30000 --iters 5 --eval_iters 5 --tsne_iters 200 --batch_size 12 --base_lr 0.01 --amp 0 --server_address 127.0.0.1:8092 --strategy FedUniV2.1 --min_num_clients 5 --img_size 256 --alpha 0.5 --beta 0.5 --prompt universal --attention dual --dual_init aggregated --label_prompt 1 --role client --cid 1 --client client2 --sup_type keypoint --gpu 1

python flower_pCE_2D_GatedCRFLoss_v2.1_auxpCE.py --root_path ../data/FAZ_h5 --num_classes 2 --in_chns 1 --img_class faz --exp faz/WeaklySeg_pCE_2D_GatedCRFLoss_v3_auxpCE_detach_Prompt_Distance_Nearest --model unet_univ3 --max_iterations 30000 --iters 5 --eval_iters 5 --tsne_iters 200 --batch_size 12 --base_lr 0.01 --amp 0 --server_address 127.0.0.1:8092 --strategy FedUniV2.1 --min_num_clients 5 --img_size 256 --alpha 0.5 --beta 0.5 --prompt universal --attention dual --dual_init aggregated --label_prompt 1 --role client --cid 2 --client client3 --sup_type block --gpu 2

python flower_pCE_2D_GatedCRFLoss_v2.1_auxpCE.py --root_path ../data/FAZ_h5 --num_classes 2 --in_chns 1 --img_class faz --exp faz/WeaklySeg_pCE_2D_GatedCRFLoss_v3_auxpCE_detach_Prompt_Distance_Nearest --model unet_univ3 --max_iterations 30000 --iters 5 --eval_iters 5 --tsne_iters 200 --batch_size 12 --base_lr 0.01 --amp 0 --server_address 127.0.0.1:8092 --strategy FedUniV2.1 --min_num_clients 5 --img_size 256 --alpha 0.5 --beta 0.5 --prompt universal --attention dual --dual_init aggregated --label_prompt 1 --role client --cid 3 --client client4 --sup_type box --gpu 3

python flower_pCE_2D_GatedCRFLoss_v2.1_auxpCE.py --root_path ../data/FAZ_h5 --num_classes 2 --in_chns 1 --img_class faz --exp faz/WeaklySeg_pCE_2D_GatedCRFLoss_v3_auxpCE_detach_Prompt_Distance_Nearest --model unet_univ3 --max_iterations 30000 --iters 5 --eval_iters 5 --tsne_iters 200 --batch_size 12 --base_lr 0.01 --amp 0 --server_address 127.0.0.1:8092 --strategy FedUniV2.1 --min_num_clients 5 --img_size 256 --alpha 0.5 --beta 0.5 --prompt universal --attention dual --dual_init aggregated --label_prompt 1 --role client --cid 4 --client client5 --sup_type scribble --gpu 5




# odoc no Detach GatedPseudo
python flower_pCE_2D_GatedCRFLoss_v3_auxpCE_GatedPseudo.py --root_path ../data/ODOC_h5 --num_classes 3 --in_chns 3 --img_class odoc --exp odoc/WeaklySeg_pCE_2D_GatedCRFLoss_v2.1_auxpCE_Prompt_Distance_GatedPseudo --model unet_univ3 --max_iterations 30000 --iters 5 --eval_iters 5 --tsne_iters 200 --batch_size 12 --base_lr 0.01 --amp 0 --server_address 127.0.0.1:8091 --strategy FedUniV2.1 --min_num_clients 5 --img_size 384 --alpha 0.5 --beta 0.5 --prompt universal --attention dual --dual_init aggregated --label_prompt 1 --role server --client client_all --sup_type mask --gpu 0

python flower_pCE_2D_GatedCRFLoss_v3_auxpCE_GatedPseudo.py --root_path ../data/ODOC_h5 --num_classes 3 --in_chns 3 --img_class odoc --exp odoc/WeaklySeg_pCE_2D_GatedCRFLoss_v2.1_auxpCE_Prompt_Distance_GatedPseudo --model unet_univ3 --max_iterations 30000 --iters 5 --eval_iters 5 --tsne_iters 200 --batch_size 12 --base_lr 0.01 --amp 0 --server_address 127.0.0.1:8091 --strategy FedUniV2.1 --min_num_clients 5 --img_size 384 --alpha 0.5 --beta 0.5 --prompt universal --attention dual --dual_init aggregated --label_prompt 1 --role client --cid 0 --client client1 --sup_type scribble --gpu 0

python flower_pCE_2D_GatedCRFLoss_v3_auxpCE_GatedPseudo.py --root_path ../data/ODOC_h5 --num_classes 3 --in_chns 3 --img_class odoc --exp odoc/WeaklySeg_pCE_2D_GatedCRFLoss_v2.1_auxpCE_Prompt_Distance_GatedPseudo --model unet_univ3 --max_iterations 30000 --iters 5 --eval_iters 5 --tsne_iters 200 --batch_size 12 --base_lr 0.01 --amp 0 --server_address 127.0.0.1:8091 --strategy FedUniV2.1 --min_num_clients 5 --img_size 384 --alpha 0.5 --beta 0.5 --prompt universal --attention dual --dual_init aggregated --label_prompt 1 --role client --cid 1 --client client2 --sup_type scribble_noisy --gpu 1

python flower_pCE_2D_GatedCRFLoss_v3_auxpCE_GatedPseudo.py --root_path ../data/ODOC_h5 --num_classes 3 --in_chns 3 --img_class odoc --exp odoc/WeaklySeg_pCE_2D_GatedCRFLoss_v2.1_auxpCE_Prompt_Distance_GatedPseudo --model unet_univ3 --max_iterations 30000 --iters 5 --eval_iters 5 --tsne_iters 200 --batch_size 12 --base_lr 0.01 --amp 0 --server_address 127.0.0.1:8091 --strategy FedUniV2.1 --min_num_clients 5 --img_size 384 --alpha 0.5 --beta 0.5 --prompt universal --attention dual --dual_init aggregated --label_prompt 1 --role client --cid 2 --client client3 --sup_type scribble_noisy --gpu 2

python flower_pCE_2D_GatedCRFLoss_v3_auxpCE_GatedPseudo.py --root_path ../data/ODOC_h5 --num_classes 3 --in_chns 3 --img_class odoc --exp odoc/WeaklySeg_pCE_2D_GatedCRFLoss_v2.1_auxpCE_Prompt_Distance_GatedPseudo --model unet_univ3 --max_iterations 30000 --iters 5 --eval_iters 5 --tsne_iters 200 --batch_size 12 --base_lr 0.01 --amp 0 --server_address 127.0.0.1:8091 --strategy FedUniV2.1 --min_num_clients 5 --img_size 384 --alpha 0.5 --beta 0.5 --prompt universal --attention dual --dual_init aggregated --label_prompt 1 --role client --cid 3 --client client4 --sup_type keypoint --gpu 3

python flower_pCE_2D_GatedCRFLoss_v3_auxpCE_GatedPseudo.py --root_path ../data/ODOC_h5 --num_classes 3 --in_chns 3 --img_class odoc --exp odoc/WeaklySeg_pCE_2D_GatedCRFLoss_v2.1_auxpCE_Prompt_Distance_GatedPseudo --model unet_univ3 --max_iterations 30000 --iters 5 --eval_iters 5 --tsne_iters 200 --batch_size 12 --base_lr 0.01 --amp 0 --server_address 127.0.0.1:8091 --strategy FedUniV2.1 --min_num_clients 5 --img_size 384 --alpha 0.5 --beta 0.5 --prompt universal --attention dual --dual_init aggregated --label_prompt 1 --role client --cid 4 --client client5 --sup_type block --gpu 5


# faz GatedPseudo
python flower_pCE_2D_GatedCRFLoss_v3_auxpCE_GatedPseudo.py --root_path ../data/FAZ_h5 --num_classes 2 --in_chns 1 --img_class faz --exp faz/WeaklySeg_pCE_2D_GatedCRFLoss_v3_auxpCE_detach_Prompt_Distance_GatedPseudo --model unet_univ3 --max_iterations 30000 --iters 5 --eval_iters 5 --tsne_iters 200 --batch_size 12 --base_lr 0.01 --amp 0 --server_address 127.0.0.1:8092 --strategy FedUniV2.1 --min_num_clients 5 --img_size 256 --alpha 0.5 --beta 0.5 --prompt universal --attention dual --dual_init aggregated --label_prompt 1 --role server --client client_all --sup_type mask --gpu 0

python flower_pCE_2D_GatedCRFLoss_v3_auxpCE_GatedPseudo.py --root_path ../data/FAZ_h5 --num_classes 2 --in_chns 1 --img_class faz --exp faz/WeaklySeg_pCE_2D_GatedCRFLoss_v3_auxpCE_detach_Prompt_Distance_GatedPseudo --model unet_univ3 --max_iterations 30000 --iters 5 --eval_iters 5 --tsne_iters 200 --batch_size 12 --base_lr 0.01 --amp 0 --server_address 127.0.0.1:8092 --strategy FedUniV2.1 --min_num_clients 5 --img_size 256 --alpha 0.5 --beta 0.5 --prompt universal --attention dual --dual_init aggregated --label_prompt 1 --role client --cid 0 --client client1 --sup_type scribble_noisy --gpu 0

python flower_pCE_2D_GatedCRFLoss_v3_auxpCE_GatedPseudo.py --root_path ../data/FAZ_h5 --num_classes 2 --in_chns 1 --img_class faz --exp faz/WeaklySeg_pCE_2D_GatedCRFLoss_v3_auxpCE_detach_Prompt_Distance_GatedPseudo --model unet_univ3 --max_iterations 30000 --iters 5 --eval_iters 5 --tsne_iters 200 --batch_size 12 --base_lr 0.01 --amp 0 --server_address 127.0.0.1:8092 --strategy FedUniV2.1 --min_num_clients 5 --img_size 256 --alpha 0.5 --beta 0.5 --prompt universal --attention dual --dual_init aggregated --label_prompt 1 --role client --cid 1 --client client2 --sup_type keypoint --gpu 1

python flower_pCE_2D_GatedCRFLoss_v3_auxpCE_GatedPseudo.py --root_path ../data/FAZ_h5 --num_classes 2 --in_chns 1 --img_class faz --exp faz/WeaklySeg_pCE_2D_GatedCRFLoss_v3_auxpCE_detach_Prompt_Distance_GatedPseudo --model unet_univ3 --max_iterations 30000 --iters 5 --eval_iters 5 --tsne_iters 200 --batch_size 12 --base_lr 0.01 --amp 0 --server_address 127.0.0.1:8092 --strategy FedUniV2.1 --min_num_clients 5 --img_size 256 --alpha 0.5 --beta 0.5 --prompt universal --attention dual --dual_init aggregated --label_prompt 1 --role client --cid 2 --client client3 --sup_type block --gpu 2

python flower_pCE_2D_GatedCRFLoss_v3_auxpCE_GatedPseudo.py --root_path ../data/FAZ_h5 --num_classes 2 --in_chns 1 --img_class faz --exp faz/WeaklySeg_pCE_2D_GatedCRFLoss_v3_auxpCE_detach_Prompt_Distance_GatedPseudo --model unet_univ3 --max_iterations 30000 --iters 5 --eval_iters 5 --tsne_iters 200 --batch_size 12 --base_lr 0.01 --amp 0 --server_address 127.0.0.1:8092 --strategy FedUniV2.1 --min_num_clients 5 --img_size 256 --alpha 0.5 --beta 0.5 --prompt universal --attention dual --dual_init aggregated --label_prompt 1 --role client --cid 3 --client client4 --sup_type box --gpu 3

python flower_pCE_2D_GatedCRFLoss_v3_auxpCE_GatedPseudo.py --root_path ../data/FAZ_h5 --num_classes 2 --in_chns 1 --img_class faz --exp faz/WeaklySeg_pCE_2D_GatedCRFLoss_v3_auxpCE_detach_Prompt_Distance_GatedPseudo --model unet_univ3 --max_iterations 30000 --iters 5 --eval_iters 5 --tsne_iters 200 --batch_size 12 --base_lr 0.01 --amp 0 --server_address 127.0.0.1:8092 --strategy FedUniV2.1 --min_num_clients 5 --img_size 256 --alpha 0.5 --beta 0.5 --prompt universal --attention dual --dual_init aggregated --label_prompt 1 --role client --cid 4 --client client5 --sup_type scribble --gpu 5



# odoc no GatedCrf
python flower_pCE_2D_v3_auxpCE.py --root_path ../data/ODOC_h5 --num_classes 3 --in_chns 3 --img_class odoc --exp odoc/WeaklySeg_pCE_2D_v3_auxpCE_detach --model unet_univ3 --max_iterations 30000 --iters 5 --eval_iters 5 --tsne_iters 200 --batch_size 12 --base_lr 0.01 --amp 0 --server_address 127.0.0.1:8198 --strategy FedUniV2.1 --min_num_clients 5 --img_size 384 --alpha 0.5 --beta 0.5 --prompt universal --attention dual --dual_init aggregated --label_prompt 1 --role server --client client_all --sup_type mask --gpu 0

python flower_pCE_2D_v3_auxpCE.py --root_path ../data/ODOC_h5 --num_classes 3 --in_chns 3 --img_class odoc --exp odoc/WeaklySeg_pCE_2D_v3_auxpCE_detach --model unet_univ3 --max_iterations 30000 --iters 5 --eval_iters 5 --tsne_iters 200 --batch_size 12 --base_lr 0.01 --amp 0 --server_address 127.0.0.1:8198 --strategy FedUniV2.1 --min_num_clients 5 --img_size 384 --alpha 0.5 --beta 0.5 --prompt universal --attention dual --dual_init aggregated --label_prompt 1 --role client --cid 0 --client client1 --sup_type scribble --gpu 0

python flower_pCE_2D_v3_auxpCE.py --root_path ../data/ODOC_h5 --num_classes 3 --in_chns 3 --img_class odoc --exp odoc/WeaklySeg_pCE_2D_v3_auxpCE_detach --model unet_univ3 --max_iterations 30000 --iters 5 --eval_iters 5 --tsne_iters 200 --batch_size 12 --base_lr 0.01 --amp 0 --server_address 127.0.0.1:8198 --strategy FedUniV2.1 --min_num_clients 5 --img_size 384 --alpha 0.5 --beta 0.5 --prompt universal --attention dual --dual_init aggregated --label_prompt 1 --role client --cid 1 --client client2 --sup_type scribble_noisy --gpu 0

python flower_pCE_2D_v3_auxpCE.py --root_path ../data/ODOC_h5 --num_classes 3 --in_chns 3 --img_class odoc --exp odoc/WeaklySeg_pCE_2D_v3_auxpCE_detach --model unet_univ3 --max_iterations 30000 --iters 5 --eval_iters 5 --tsne_iters 200 --batch_size 12 --base_lr 0.01 --amp 0 --server_address 127.0.0.1:8198 --strategy FedUniV2.1 --min_num_clients 5 --img_size 384 --alpha 0.5 --beta 0.5 --prompt universal --attention dual --dual_init aggregated --label_prompt 1 --role client --cid 2 --client client3 --sup_type scribble_noisy --gpu 0

python flower_pCE_2D_v3_auxpCE.py --root_path ../data/ODOC_h5 --num_classes 3 --in_chns 3 --img_class odoc --exp odoc/WeaklySeg_pCE_2D_v3_auxpCE_detach --model unet_univ3 --max_iterations 30000 --iters 5 --eval_iters 5 --tsne_iters 200 --batch_size 12 --base_lr 0.01 --amp 0 --server_address 127.0.0.1:8198 --strategy FedUniV2.1 --min_num_clients 5 --img_size 384 --alpha 0.5 --beta 0.5 --prompt universal --attention dual --dual_init aggregated --label_prompt 1 --role client --cid 3 --client client4 --sup_type keypoint --gpu 1

python flower_pCE_2D_v3_auxpCE.py --root_path ../data/ODOC_h5 --num_classes 3 --in_chns 3 --img_class odoc --exp odoc/WeaklySeg_pCE_2D_v3_auxpCE_detach --model unet_univ3 --max_iterations 30000 --iters 5 --eval_iters 5 --tsne_iters 200 --batch_size 12 --base_lr 0.01 --amp 0 --server_address 127.0.0.1:8198 --strategy FedUniV2.1 --min_num_clients 5 --img_size 384 --alpha 0.5 --beta 0.5 --prompt universal --attention dual --dual_init aggregated --label_prompt 1 --role client --cid 4 --client client5 --sup_type block --gpu 1


# polyp no label prompt
python flower_pCE_2D_v3_auxpCE.py --root_path ../data/Polypdata_h5 --num_classes 2 --in_chns 3 --img_class polyp --exp polyp/WeaklySeg_pCE_2D_v3_auxpCE_detach --model unet_univ3 --max_iterations 30000 --iters 5 --eval_iters 5 --tsne_iters 200 --batch_size 12 --base_lr 0.001 --amp 0 --server_address 127.0.0.1:8199 --strategy FedUniV2.1 --min_num_clients 4 --img_size 384 --alpha 0.5 --beta 0.1 --prompt universal --attention dual --dual_init aggregated --label_prompt 1 --role server --client client_all --sup_type mask --gpu 1

python flower_pCE_2D_v3_auxpCE.py --root_path ../data/Polypdata_h5 --num_classes 2 --in_chns 3 --img_class polyp --exp polyp/WeaklySeg_pCE_2D_v3_auxpCE_detach --model unet_univ3 --max_iterations 30000 --iters 5 --eval_iters 5 --tsne_iters 200 --batch_size 12 --base_lr 0.001 --amp 0 --server_address 127.0.0.1:8199 --strategy FedUniV2.1 --min_num_clients 4 --img_size 384 --alpha 0.5 --beta 0.1 --prompt universal --attention dual --dual_init aggregated --label_prompt 1 --role client --cid 0 --client client1 --sup_type keypoint --gpu 1

python flower_pCE_2D_v3_auxpCE.py --root_path ../data/Polypdata_h5 --num_classes 2 --in_chns 3 --img_class polyp --exp polyp/WeaklySeg_pCE_2D_v3_auxpCE_detach --model unet_univ3 --max_iterations 30000 --iters 5 --eval_iters 5 --tsne_iters 200 --batch_size 12 --base_lr 0.001 --amp 0 --server_address 127.0.0.1:8199 --strategy FedUniV2.1 --min_num_clients 4 --img_size 384 --alpha 0.5 --beta 0.1 --prompt universal --attention dual --dual_init aggregated --label_prompt 1 --role client --cid 1 --client client2 --sup_type scribble --gpu 1

python flower_pCE_2D_v3_auxpCE.py --root_path ../data/Polypdata_h5 --num_classes 2 --in_chns 3 --img_class polyp --exp polyp/WeaklySeg_pCE_2D_v3_auxpCE_detach --model unet_univ3 --max_iterations 30000 --iters 5 --eval_iters 5 --tsne_iters 200 --batch_size 12 --base_lr 0.001 --amp 0 --server_address 127.0.0.1:8199 --strategy FedUniV2.1 --min_num_clients 4 --img_size 384 --alpha 0.5 --beta 0.1 --prompt universal --attention dual --dual_init aggregated --label_prompt 1 --role client --cid 2 --client client3 --sup_type box --gpu 5

python flower_pCE_2D_v3_auxpCE.py --root_path ../data/Polypdata_h5 --num_classes 2 --in_chns 3 --img_class polyp --exp polyp/WeaklySeg_pCE_2D_v3_auxpCE_detach --model unet_univ3 --max_iterations 30000 --iters 5 --eval_iters 5 --tsne_iters 200 --batch_size 12 --base_lr 0.001 --amp 0 --server_address 127.0.0.1:8199 --strategy FedUniV2.1 --min_num_clients 4 --img_size 384 --alpha 0.5 --beta 0.1 --prompt universal --attention dual --dual_init aggregated --label_prompt 1 --role client --cid 3 --client client4 --sup_type block --gpu 5

# faz AGenergyLoss
python flower_pCE_2D_v3_auxpCE_AGEnergyLoss.py --root_path ../data/FAZ_h5 --num_classes 2 --in_chns 1 --img_class faz --exp faz/WeaklySeg_pCE_2D_v3_auxpCE_detach_AGenergyLoss --model unet_univ3 --max_iterations 30000 --iters 5 --eval_iters 5 --tsne_iters 200 --batch_size 12 --base_lr 0.01 --amp 0 --server_address 127.0.0.1:8092 --strategy FedUniV2.1 --min_num_clients 5 --img_size 256 --alpha 0.5 --beta 0.5 --prompt universal --attention dual --dual_init aggregated --label_prompt 1 --role server --client client_all --sup_type mask --gpu 0

python flower_pCE_2D_v3_auxpCE_AGEnergyLoss.py --root_path ../data/FAZ_h5 --num_classes 2 --in_chns 1 --img_class faz --exp faz/WeaklySeg_pCE_2D_v3_auxpCE_detach_AGenergyLoss --model unet_univ3 --max_iterations 30000 --iters 5 --eval_iters 5 --tsne_iters 200 --batch_size 12 --base_lr 0.01 --amp 0 --server_address 127.0.0.1:8092 --strategy FedUniV2.1 --min_num_clients 5 --img_size 256 --alpha 0.5 --beta 0.5 --prompt universal --attention dual --dual_init aggregated --label_prompt 1 --role client --cid 0 --client client1 --sup_type scribble_noisy --gpu 1

python flower_pCE_2D_v3_auxpCE_AGEnergyLoss.py --root_path ../data/FAZ_h5 --num_classes 2 --in_chns 1 --img_class faz --exp faz/WeaklySeg_pCE_2D_v3_auxpCE_detach_AGenergyLoss --model unet_univ3 --max_iterations 30000 --iters 5 --eval_iters 5 --tsne_iters 200 --batch_size 12 --base_lr 0.01 --amp 0 --server_address 127.0.0.1:8092 --strategy FedUniV2.1 --min_num_clients 5 --img_size 256 --alpha 0.5 --beta 0.5 --prompt universal --attention dual --dual_init aggregated --label_prompt 1 --role client --cid 1 --client client2 --sup_type keypoint --gpu 2

python flower_pCE_2D_v3_auxpCE_AGEnergyLoss.py --root_path ../data/FAZ_h5 --num_classes 2 --in_chns 1 --img_class faz --exp faz/WeaklySeg_pCE_2D_v3_auxpCE_detach_AGenergyLoss --model unet_univ3 --max_iterations 30000 --iters 5 --eval_iters 5 --tsne_iters 200 --batch_size 12 --base_lr 0.01 --amp 0 --server_address 127.0.0.1:8092 --strategy FedUniV2.1 --min_num_clients 5 --img_size 256 --alpha 0.5 --beta 0.5 --prompt universal --attention dual --dual_init aggregated --label_prompt 1 --role client --cid 2 --client client3 --sup_type block --gpu 3

python flower_pCE_2D_v3_auxpCE_AGEnergyLoss.py --root_path ../data/FAZ_h5 --num_classes 2 --in_chns 1 --img_class faz --exp faz/WeaklySeg_pCE_2D_v3_auxpCE_detach_AGenergyLoss --model unet_univ3 --max_iterations 30000 --iters 5 --eval_iters 5 --tsne_iters 200 --batch_size 12 --base_lr 0.01 --amp 0 --server_address 127.0.0.1:8092 --strategy FedUniV2.1 --min_num_clients 5 --img_size 256 --alpha 0.5 --beta 0.5 --prompt universal --attention dual --dual_init aggregated --label_prompt 1 --role client --cid 3 --client client4 --sup_type box --gpu 4

python flower_pCE_2D_v3_auxpCE_AGEnergyLoss.py --root_path ../data/FAZ_h5 --num_classes 2 --in_chns 1 --img_class faz --exp faz/WeaklySeg_pCE_2D_v3_auxpCE_detach_AGenergyLoss --model unet_univ3 --max_iterations 30000 --iters 5 --eval_iters 5 --tsne_iters 200 --batch_size 12 --base_lr 0.01 --amp 0 --server_address 127.0.0.1:8092 --strategy FedUniV2.1 --min_num_clients 5 --img_size 256 --alpha 0.5 --beta 0.5 --prompt universal --attention dual --dual_init aggregated --label_prompt 1 --role client --cid 4 --client client5 --sup_type scribble --gpu 7

# odoc AGenergyLoss
python flower_pCE_2D_v3_auxpCE_AGEnergyLoss.py --root_path ../data/ODOC_h5 --num_classes 3 --in_chns 3 --img_class odoc --exp odoc/WeaklySeg_pCE_2D_v3_auxpCE_detach_AGenergyLoss --model unet_univ3 --max_iterations 30000 --iters 5 --eval_iters 5 --tsne_iters 200 --batch_size 12 --base_lr 0.01 --amp 0 --server_address 127.0.0.1:8091 --strategy FedUniV2.1 --min_num_clients 5 --img_size 384 --alpha 0.5 --beta 0.5 --prompt universal --attention dual --dual_init aggregated --label_prompt 1 --role server --client client_all --sup_type mask --gpu 0

python flower_pCE_2D_v3_auxpCE_AGEnergyLoss.py --root_path ../data/ODOC_h5 --num_classes 3 --in_chns 3 --img_class odoc --exp odoc/WeaklySeg_pCE_2D_v3_auxpCE_detach_AGenergyLoss --model unet_univ3 --max_iterations 30000 --iters 5 --eval_iters 5 --tsne_iters 200 --batch_size 12 --base_lr 0.01 --amp 0 --server_address 127.0.0.1:8091 --strategy FedUniV2.1 --min_num_clients 5 --img_size 384 --alpha 0.5 --beta 0.5 --prompt universal --attention dual --dual_init aggregated --label_prompt 1 --role client --cid 0 --client client1 --sup_type scribble --gpu 0

python flower_pCE_2D_v3_auxpCE_AGEnergyLoss.py --root_path ../data/ODOC_h5 --num_classes 3 --in_chns 3 --img_class odoc --exp odoc/WeaklySeg_pCE_2D_v3_auxpCE_detach_AGenergyLoss --model unet_univ3 --max_iterations 30000 --iters 5 --eval_iters 5 --tsne_iters 200 --batch_size 12 --base_lr 0.01 --amp 0 --server_address 127.0.0.1:8091 --strategy FedUniV2.1 --min_num_clients 5 --img_size 384 --alpha 0.5 --beta 0.5 --prompt universal --attention dual --dual_init aggregated --label_prompt 1 --role client --cid 1 --client client2 --sup_type scribble_noisy --gpu 1

python flower_pCE_2D_v3_auxpCE_AGEnergyLoss.py --root_path ../data/ODOC_h5 --num_classes 3 --in_chns 3 --img_class odoc --exp odoc/WeaklySeg_pCE_2D_v3_auxpCE_detach_AGenergyLoss --model unet_univ3 --max_iterations 30000 --iters 5 --eval_iters 5 --tsne_iters 200 --batch_size 12 --base_lr 0.01 --amp 0 --server_address 127.0.0.1:8091 --strategy FedUniV2.1 --min_num_clients 5 --img_size 384 --alpha 0.5 --beta 0.5 --prompt universal --attention dual --dual_init aggregated --label_prompt 1 --role client --cid 2 --client client3 --sup_type scribble_noisy --gpu 2

python flower_pCE_2D_v3_auxpCE_AGEnergyLoss.py --root_path ../data/ODOC_h5 --num_classes 3 --in_chns 3 --img_class odoc --exp odoc/WeaklySeg_pCE_2D_v3_auxpCE_detach_AGenergyLoss --model unet_univ3 --max_iterations 30000 --iters 5 --eval_iters 5 --tsne_iters 200 --batch_size 12 --base_lr 0.01 --amp 0 --server_address 127.0.0.1:8091 --strategy FedUniV2.1 --min_num_clients 5 --img_size 384 --alpha 0.5 --beta 0.5 --prompt universal --attention dual --dual_init aggregated --label_prompt 1 --role client --cid 3 --client client4 --sup_type keypoint --gpu 5

python flower_pCE_2D_v3_auxpCE_AGEnergyLoss.py --root_path ../data/ODOC_h5 --num_classes 3 --in_chns 3 --img_class odoc --exp odoc/WeaklySeg_pCE_2D_v3_auxpCE_detach_AGenergyLoss --model unet_univ3 --max_iterations 30000 --iters 5 --eval_iters 5 --tsne_iters 200 --batch_size 12 --base_lr 0.01 --amp 0 --server_address 127.0.0.1:8091 --strategy FedUniV2.1 --min_num_clients 5 --img_size 384 --alpha 0.5 --beta 0.5 --prompt universal --attention dual --dual_init aggregated --label_prompt 1 --role client --cid 4 --client client5 --sup_type block --gpu 6


# polyp AGenergyLoss
python flower_pCE_2D_v3_auxpCE_AGEnergyLoss.py --root_path ../data/Polypdata_h5 --num_classes 2 --in_chns 3 --img_class polyp --exp polyp/WeaklySeg_pCE_2D_v3_auxpCE_detach_AGenergyLoss --model unet_univ3 --max_iterations 30000 --iters 5 --eval_iters 5 --tsne_iters 200 --batch_size 12 --base_lr 0.001 --amp 0 --server_address 127.0.0.1:8093 --strategy FedUniV2.1 --min_num_clients 4 --img_size 384 --alpha 0.5 --beta 0.1 --prompt universal --attention dual --dual_init aggregated --label_prompt 1 --role server --client client_all --sup_type mask --gpu 0

python flower_pCE_2D_v3_auxpCE_AGEnergyLoss.py --root_path ../data/Polypdata_h5 --num_classes 2 --in_chns 3 --img_class polyp --exp polyp/WeaklySeg_pCE_2D_v3_auxpCE_detach_AGenergyLoss --model unet_univ3 --max_iterations 30000 --iters 5 --eval_iters 5 --tsne_iters 200 --batch_size 12 --base_lr 0.001 --amp 0 --server_address 127.0.0.1:8093 --strategy FedUniV2.1 --min_num_clients 4 --img_size 384 --alpha 0.5 --beta 0.1 --prompt universal --attention dual --dual_init aggregated --label_prompt 1 --role client --cid 0 --client client1 --sup_type keypoint --gpu 0

python flower_pCE_2D_v3_auxpCE_AGEnergyLoss.py --root_path ../data/Polypdata_h5 --num_classes 2 --in_chns 3 --img_class polyp --exp polyp/WeaklySeg_pCE_2D_v3_auxpCE_detach_AGenergyLoss --model unet_univ3 --max_iterations 30000 --iters 5 --eval_iters 5 --tsne_iters 200 --batch_size 12 --base_lr 0.001 --amp 0 --server_address 127.0.0.1:8093 --strategy FedUniV2.1 --min_num_clients 4 --img_size 384 --alpha 0.5 --beta 0.1 --prompt universal --attention dual --dual_init aggregated --label_prompt 1 --role client --cid 1 --client client2 --sup_type scribble --gpu 1

python flower_pCE_2D_v3_auxpCE_AGEnergyLoss.py --root_path ../data/Polypdata_h5 --num_classes 2 --in_chns 3 --img_class polyp --exp polyp/WeaklySeg_pCE_2D_v3_auxpCE_detach_AGenergyLoss --model unet_univ3 --max_iterations 30000 --iters 5 --eval_iters 5 --tsne_iters 200 --batch_size 12 --base_lr 0.001 --amp 0 --server_address 127.0.0.1:8093 --strategy FedUniV2.1 --min_num_clients 4 --img_size 384 --alpha 0.5 --beta 0.1 --prompt universal --attention dual --dual_init aggregated --label_prompt 1 --role client --cid 2 --client client3 --sup_type box --gpu 5

python flower_pCE_2D_v3_auxpCE_AGEnergyLoss.py --root_path ../data/Polypdata_h5 --num_classes 2 --in_chns 3 --img_class polyp --exp polyp/WeaklySeg_pCE_2D_v3_auxpCE_detach_AGenergyLoss --model unet_univ3 --max_iterations 30000 --iters 5 --eval_iters 5 --tsne_iters 200 --batch_size 12 --base_lr 0.001 --amp 0 --server_address 127.0.0.1:8093 --strategy FedUniV2.1 --min_num_clients 4 --img_size 384 --alpha 0.5 --beta 0.1 --prompt universal --attention dual --dual_init aggregated --label_prompt 1 --role client --cid 3 --client client4 --sup_type block --gpu 6




# faz MLP
python flower_pCE_2D_v4_auxpCE_GatedCRFLoss.py --root_path ../data/FAZ_h5 --num_classes 2 --in_chns 1 --img_class faz --exp faz/WeaklySeg_pCE_2D_v4_auxpCE_detach_GatedCRFLoss_MLPBN --model unet_univ4 --max_iterations 30000 --iters 5 --eval_iters 5 --tsne_iters 200 --batch_size 12 --base_lr 0.01 --amp 0 --server_address 172.18.5.181:8091 --strategy FedUniV2.1 --min_num_clients 5 --img_size 256 --alpha 0.5 --beta 0.5 --prompt universal --attention dual --dual_init aggregated --label_prompt 1 --role server --client client_all --sup_type mask --gpu 1

python flower_pCE_2D_v4_auxpCE_GatedCRFLoss.py --root_path ../data/FAZ_h5 --num_classes 2 --in_chns 1 --img_class faz --exp faz/WeaklySeg_pCE_2D_v4_auxpCE_detach_GatedCRFLoss_MLPBN --model unet_univ4 --max_iterations 30000 --iters 5 --eval_iters 5 --tsne_iters 200 --batch_size 12 --base_lr 0.01 --amp 0 --server_address 172.18.5.181:8091 --strategy FedUniV2.1 --min_num_clients 5 --img_size 256 --alpha 0.5 --beta 0.5 --prompt universal --attention dual --dual_init aggregated --label_prompt 1 --role client --cid 0 --client client1 --sup_type scribble_noisy --gpu 0

python flower_pCE_2D_v4_auxpCE_GatedCRFLoss.py --root_path ../data/FAZ_h5 --num_classes 2 --in_chns 1 --img_class faz --exp faz/WeaklySeg_pCE_2D_v4_auxpCE_detach_GatedCRFLoss_MLPBN --model unet_univ4 --max_iterations 30000 --iters 5 --eval_iters 5 --tsne_iters 200 --batch_size 12 --base_lr 0.01 --amp 0 --server_address 172.18.5.181:8091 --strategy FedUniV2.1 --min_num_clients 5 --img_size 256 --alpha 0.5 --beta 0.5 --prompt universal --attention dual --dual_init aggregated --label_prompt 1 --role client --cid 1 --client client2 --sup_type keypoint --gpu 2

python flower_pCE_2D_v4_auxpCE_GatedCRFLoss.py --root_path ../data/FAZ_h5 --num_classes 2 --in_chns 1 --img_class faz --exp faz/WeaklySeg_pCE_2D_v4_auxpCE_detach_GatedCRFLoss_MLPBN --model unet_univ4 --max_iterations 30000 --iters 5 --eval_iters 5 --tsne_iters 200 --batch_size 12 --base_lr 0.01 --amp 0 --server_address 172.18.5.181:8091 --strategy FedUniV2.1 --min_num_clients 5 --img_size 256 --alpha 0.5 --beta 0.5 --prompt universal --attention dual --dual_init aggregated --label_prompt 1 --role client --cid 2 --client client3 --sup_type block --gpu 5

python flower_pCE_2D_v4_auxpCE_GatedCRFLoss.py --root_path ../data/FAZ_h5 --num_classes 2 --in_chns 1 --img_class faz --exp faz/WeaklySeg_pCE_2D_v4_auxpCE_detach_GatedCRFLoss_MLPBN --model unet_univ4 --max_iterations 30000 --iters 5 --eval_iters 5 --tsne_iters 200 --batch_size 12 --base_lr 0.01 --amp 0 --server_address 172.18.5.181:8091 --strategy FedUniV2.1 --min_num_clients 5 --img_size 256 --alpha 0.5 --beta 0.5 --prompt universal --attention dual --dual_init aggregated --label_prompt 1 --role client --cid 3 --client client4 --sup_type box --gpu 6

python flower_pCE_2D_v4_auxpCE_GatedCRFLoss.py --root_path ../data/FAZ_h5 --num_classes 2 --in_chns 1 --img_class faz --exp faz/WeaklySeg_pCE_2D_v4_auxpCE_detach_GatedCRFLoss_MLPBN --model unet_univ4 --max_iterations 30000 --iters 5 --eval_iters 5 --tsne_iters 200 --batch_size 12 --base_lr 0.01 --amp 0 --server_address 172.18.5.181:8091 --strategy FedUniV2.1 --min_num_clients 5 --img_size 256 --alpha 0.5 --beta 0.5 --prompt universal --attention dual --dual_init aggregated --label_prompt 1 --role client --cid 4 --client client5 --sup_type scribble --gpu 7

# odoc MLP
python flower_pCE_2D_v4_auxpCE_GatedCRFLoss.py --root_path ../data/ODOC_h5 --num_classes 3 --in_chns 3 --img_class odoc --exp odoc/WeaklySeg_pCE_2D_v4_auxpCE_detach_GatedCRFLoss_MLP --model unet_univ4 --max_iterations 30000 --iters 5 --eval_iters 5 --tsne_iters 200 --batch_size 12 --base_lr 0.01 --amp 0 --server_address 127.0.0.1:8091 --strategy FedUniV2.1 --min_num_clients 5 --img_size 384 --alpha 0.5 --beta 0.5 --prompt universal --attention dual --dual_init aggregated --label_prompt 1 --role server --client client_all --sup_type mask --gpu 0

python flower_pCE_2D_v4_auxpCE_GatedCRFLoss.py --root_path ../data/ODOC_h5 --num_classes 3 --in_chns 3 --img_class odoc --exp odoc/WeaklySeg_pCE_2D_v4_auxpCE_detach_GatedCRFLoss_MLP --model unet_univ4 --max_iterations 30000 --iters 5 --eval_iters 5 --tsne_iters 200 --batch_size 12 --base_lr 0.01 --amp 0 --server_address 127.0.0.1:8091 --strategy FedUniV2.1 --min_num_clients 5 --img_size 384 --alpha 0.5 --beta 0.5 --prompt universal --attention dual --dual_init aggregated --label_prompt 1 --role client --cid 0 --client client1 --sup_type scribble --gpu 0

python flower_pCE_2D_v4_auxpCE_GatedCRFLoss.py --root_path ../data/ODOC_h5 --num_classes 3 --in_chns 3 --img_class odoc --exp odoc/WeaklySeg_pCE_2D_v4_auxpCE_detach_GatedCRFLoss_MLP --model unet_univ4 --max_iterations 30000 --iters 5 --eval_iters 5 --tsne_iters 200 --batch_size 12 --base_lr 0.01 --amp 0 --server_address 127.0.0.1:8091 --strategy FedUniV2.1 --min_num_clients 5 --img_size 384 --alpha 0.5 --beta 0.5 --prompt universal --attention dual --dual_init aggregated --label_prompt 1 --role client --cid 1 --client client2 --sup_type scribble_noisy --gpu 1

python flower_pCE_2D_v4_auxpCE_GatedCRFLoss.py --root_path ../data/ODOC_h5 --num_classes 3 --in_chns 3 --img_class odoc --exp odoc/WeaklySeg_pCE_2D_v4_auxpCE_detach_GatedCRFLoss_MLP --model unet_univ4 --max_iterations 30000 --iters 5 --eval_iters 5 --tsne_iters 200 --batch_size 12 --base_lr 0.01 --amp 0 --server_address 127.0.0.1:8091 --strategy FedUniV2.1 --min_num_clients 5 --img_size 384 --alpha 0.5 --beta 0.5 --prompt universal --attention dual --dual_init aggregated --label_prompt 1 --role client --cid 2 --client client3 --sup_type scribble_noisy --gpu 2

python flower_pCE_2D_v4_auxpCE_GatedCRFLoss.py --root_path ../data/ODOC_h5 --num_classes 3 --in_chns 3 --img_class odoc --exp odoc/WeaklySeg_pCE_2D_v4_auxpCE_detach_GatedCRFLoss_MLP --model unet_univ4 --max_iterations 30000 --iters 5 --eval_iters 5 --tsne_iters 200 --batch_size 12 --base_lr 0.01 --amp 0 --server_address 127.0.0.1:8091 --strategy FedUniV2.1 --min_num_clients 5 --img_size 384 --alpha 0.5 --beta 0.5 --prompt universal --attention dual --dual_init aggregated --label_prompt 1 --role client --cid 3 --client client4 --sup_type keypoint --gpu 5

python flower_pCE_2D_v4_auxpCE_GatedCRFLoss.py --root_path ../data/ODOC_h5 --num_classes 3 --in_chns 3 --img_class odoc --exp odoc/WeaklySeg_pCE_2D_v4_auxpCE_detach_GatedCRFLoss_MLP --model unet_univ4 --max_iterations 30000 --iters 5 --eval_iters 5 --tsne_iters 200 --batch_size 12 --base_lr 0.01 --amp 0 --server_address 127.0.0.1:8091 --strategy FedUniV2.1 --min_num_clients 5 --img_size 384 --alpha 0.5 --beta 0.5 --prompt universal --attention dual --dual_init aggregated --label_prompt 1 --role client --cid 4 --client client5 --sup_type block --gpu 6


# polyp MLP
python flower_pCE_2D_v4_auxpCE_GatedCRFLoss.py --root_path ../data/Polypdata_h5 --num_classes 2 --in_chns 3 --img_class polyp --exp polyp/WeaklySeg_pCE_2D_v4_auxpCE_detach_GatedCRFLoss_MLP --model unet_univ4 --max_iterations 30000 --iters 5 --eval_iters 5 --tsne_iters 200 --batch_size 12 --base_lr 0.001 --amp 0 --server_address 127.0.0.1:8093 --strategy FedUniV2.1 --min_num_clients 4 --img_size 384 --alpha 0.5 --beta 0.1 --prompt universal --attention dual --dual_init aggregated --label_prompt 1 --role server --client client_all --sup_type mask --gpu 0

python flower_pCE_2D_v4_auxpCE_GatedCRFLoss.py --root_path ../data/Polypdata_h5 --num_classes 2 --in_chns 3 --img_class polyp --exp polyp/WeaklySeg_pCE_2D_v4_auxpCE_detach_GatedCRFLoss_MLP --model unet_univ4 --max_iterations 30000 --iters 5 --eval_iters 5 --tsne_iters 200 --batch_size 12 --base_lr 0.001 --amp 0 --server_address 127.0.0.1:8093 --strategy FedUniV2.1 --min_num_clients 4 --img_size 384 --alpha 0.5 --beta 0.1 --prompt universal --attention dual --dual_init aggregated --label_prompt 1 --role client --cid 0 --client client1 --sup_type keypoint --gpu 0

python flower_pCE_2D_v4_auxpCE_GatedCRFLoss.py --root_path ../data/Polypdata_h5 --num_classes 2 --in_chns 3 --img_class polyp --exp polyp/WeaklySeg_pCE_2D_v4_auxpCE_detach_GatedCRFLoss_MLP --model unet_univ4 --max_iterations 30000 --iters 5 --eval_iters 5 --tsne_iters 200 --batch_size 12 --base_lr 0.001 --amp 0 --server_address 127.0.0.1:8093 --strategy FedUniV2.1 --min_num_clients 4 --img_size 384 --alpha 0.5 --beta 0.1 --prompt universal --attention dual --dual_init aggregated --label_prompt 1 --role client --cid 1 --client client2 --sup_type scribble --gpu 1

python flower_pCE_2D_v4_auxpCE_GatedCRFLoss.py --root_path ../data/Polypdata_h5 --num_classes 2 --in_chns 3 --img_class polyp --exp polyp/WeaklySeg_pCE_2D_v4_auxpCE_detach_GatedCRFLoss_MLP --model unet_univ4 --max_iterations 30000 --iters 5 --eval_iters 5 --tsne_iters 200 --batch_size 12 --base_lr 0.001 --amp 0 --server_address 127.0.0.1:8093 --strategy FedUniV2.1 --min_num_clients 4 --img_size 384 --alpha 0.5 --beta 0.1 --prompt universal --attention dual --dual_init aggregated --label_prompt 1 --role client --cid 2 --client client3 --sup_type box --gpu 5

python flower_pCE_2D_v4_auxpCE_GatedCRFLoss.py --root_path ../data/Polypdata_h5 --num_classes 2 --in_chns 3 --img_class polyp --exp polyp/WeaklySeg_pCE_2D_v4_auxpCE_detach_GatedCRFLoss_MLP --model unet_univ4 --max_iterations 30000 --iters 5 --eval_iters 5 --tsne_iters 200 --batch_size 12 --base_lr 0.001 --amp 0 --server_address 127.0.0.1:8093 --strategy FedUniV2.1 --min_num_clients 4 --img_size 384 --alpha 0.5 --beta 0.1 --prompt universal --attention dual --dual_init aggregated --label_prompt 1 --role client --cid 3 --client client4 --sup_type block --gpu 6




# faz simple mlp
python flower_pCE_2D_v4_auxpCE_GatedCRFLoss.py --root_path ../data/FAZ_h5 --num_classes 2 --in_chns 1 --img_class faz --exp faz/WeaklySeg_pCE_2D_v4_auxpCE_detach_GatedCRFLoss_TwoLayersMLPBN --model unet_univ4 --max_iterations 30000 --iters 5 --eval_iters 5 --tsne_iters 200 --batch_size 12 --base_lr 0.01 --amp 0 --server_address 172.18.5.181:8091 --strategy FedUniV2.1 --min_num_clients 5 --img_size 256 --alpha 0.5 --beta 0.5 --prompt universal --attention dual --dual_init aggregated --label_prompt 1 --role server --client client_all --sup_type mask --gpu 1

python flower_pCE_2D_v4_auxpCE_GatedCRFLoss.py --root_path ../data/FAZ_h5 --num_classes 2 --in_chns 1 --img_class faz --exp faz/WeaklySeg_pCE_2D_v4_auxpCE_detach_GatedCRFLoss_TwoLayersMLPBN --model unet_univ4 --max_iterations 30000 --iters 5 --eval_iters 5 --tsne_iters 200 --batch_size 12 --base_lr 0.01 --amp 0 --server_address 172.18.5.181:8091 --strategy FedUniV2.1 --min_num_clients 5 --img_size 256 --alpha 0.5 --beta 0.5 --prompt universal --attention dual --dual_init aggregated --label_prompt 1 --role client --cid 0 --client client1 --sup_type scribble_noisy --gpu 0

python flower_pCE_2D_v4_auxpCE_GatedCRFLoss.py --root_path ../data/FAZ_h5 --num_classes 2 --in_chns 1 --img_class faz --exp faz/WeaklySeg_pCE_2D_v4_auxpCE_detach_GatedCRFLoss_TwoLayersMLPBN --model unet_univ4 --max_iterations 30000 --iters 5 --eval_iters 5 --tsne_iters 200 --batch_size 12 --base_lr 0.01 --amp 0 --server_address 172.18.5.181:8091 --strategy FedUniV2.1 --min_num_clients 5 --img_size 256 --alpha 0.5 --beta 0.5 --prompt universal --attention dual --dual_init aggregated --label_prompt 1 --role client --cid 1 --client client2 --sup_type keypoint --gpu 2

python flower_pCE_2D_v4_auxpCE_GatedCRFLoss.py --root_path ../data/FAZ_h5 --num_classes 2 --in_chns 1 --img_class faz --exp faz/WeaklySeg_pCE_2D_v4_auxpCE_detach_GatedCRFLoss_TwoLayersMLPBN --model unet_univ4 --max_iterations 30000 --iters 5 --eval_iters 5 --tsne_iters 200 --batch_size 12 --base_lr 0.01 --amp 0 --server_address 172.18.5.181:8091 --strategy FedUniV2.1 --min_num_clients 5 --img_size 256 --alpha 0.5 --beta 0.5 --prompt universal --attention dual --dual_init aggregated --label_prompt 1 --role client --cid 2 --client client3 --sup_type block --gpu 5

python flower_pCE_2D_v4_auxpCE_GatedCRFLoss.py --root_path ../data/FAZ_h5 --num_classes 2 --in_chns 1 --img_class faz --exp faz/WeaklySeg_pCE_2D_v4_auxpCE_detach_GatedCRFLoss_TwoLayersMLPBN --model unet_univ4 --max_iterations 30000 --iters 5 --eval_iters 5 --tsne_iters 200 --batch_size 12 --base_lr 0.01 --amp 0 --server_address 172.18.5.181:8091 --strategy FedUniV2.1 --min_num_clients 5 --img_size 256 --alpha 0.5 --beta 0.5 --prompt universal --attention dual --dual_init aggregated --label_prompt 1 --role client --cid 3 --client client4 --sup_type box --gpu 6

python flower_pCE_2D_v4_auxpCE_GatedCRFLoss.py --root_path ../data/FAZ_h5 --num_classes 2 --in_chns 1 --img_class faz --exp faz/WeaklySeg_pCE_2D_v4_auxpCE_detach_GatedCRFLoss_TwoLayersMLPBN --model unet_univ4 --max_iterations 30000 --iters 5 --eval_iters 5 --tsne_iters 200 --batch_size 12 --base_lr 0.01 --amp 0 --server_address 172.18.5.181:8091 --strategy FedUniV2.1 --min_num_clients 5 --img_size 256 --alpha 0.5 --beta 0.5 --prompt universal --attention dual --dual_init aggregated --label_prompt 1 --role client --cid 4 --client client5 --sup_type scribble --gpu 7


# faz simple mlp + GAP
python flower_pCE_2D_v4_auxpCE_GatedCRFLoss.py --root_path ../data/FAZ_h5 --num_classes 2 --in_chns 1 --img_class faz --exp faz/WeaklySeg_pCE_2D_v4_auxpCE_detach_GatedCRFLoss_TwoLayersMLPBN_GAP --model unet_univ4 --max_iterations 30000 --iters 5 --eval_iters 5 --tsne_iters 200 --batch_size 12 --base_lr 0.01 --amp 0 --server_address 172.18.5.181:8092 --strategy FedUniV2.1 --min_num_clients 5 --img_size 256 --alpha 0.5 --beta 0.5 --prompt universal --attention dual --dual_init aggregated --label_prompt 1 --role server --client client_all --sup_type mask --gpu 4

python flower_pCE_2D_v4_auxpCE_GatedCRFLoss.py --root_path ../data/FAZ_h5 --num_classes 2 --in_chns 1 --img_class faz --exp faz/WeaklySeg_pCE_2D_v4_auxpCE_detach_GatedCRFLoss_TwoLayersMLPBN_GAP --model unet_univ4 --max_iterations 30000 --iters 5 --eval_iters 5 --tsne_iters 200 --batch_size 12 --base_lr 0.01 --amp 0 --server_address 172.18.5.181:8092 --strategy FedUniV2.1 --min_num_clients 5 --img_size 256 --alpha 0.5 --beta 0.5 --prompt universal --attention dual --dual_init aggregated --label_prompt 1 --role client --cid 0 --client client1 --sup_type scribble_noisy --gpu 0

python flower_pCE_2D_v4_auxpCE_GatedCRFLoss.py --root_path ../data/FAZ_h5 --num_classes 2 --in_chns 1 --img_class faz --exp faz/WeaklySeg_pCE_2D_v4_auxpCE_detach_GatedCRFLoss_TwoLayersMLPBN_GAP --model unet_univ4 --max_iterations 30000 --iters 5 --eval_iters 5 --tsne_iters 200 --batch_size 12 --base_lr 0.01 --amp 0 --server_address 172.18.5.181:8092 --strategy FedUniV2.1 --min_num_clients 5 --img_size 256 --alpha 0.5 --beta 0.5 --prompt universal --attention dual --dual_init aggregated --label_prompt 1 --role client --cid 1 --client client2 --sup_type keypoint --gpu 1

python flower_pCE_2D_v4_auxpCE_GatedCRFLoss.py --root_path ../data/FAZ_h5 --num_classes 2 --in_chns 1 --img_class faz --exp faz/WeaklySeg_pCE_2D_v4_auxpCE_detach_GatedCRFLoss_TwoLayersMLPBN_GAP --model unet_univ4 --max_iterations 30000 --iters 5 --eval_iters 5 --tsne_iters 200 --batch_size 12 --base_lr 0.01 --amp 0 --server_address 172.18.5.181:8092 --strategy FedUniV2.1 --min_num_clients 5 --img_size 256 --alpha 0.5 --beta 0.5 --prompt universal --attention dual --dual_init aggregated --label_prompt 1 --role client --cid 2 --client client3 --sup_type block --gpu 2

python flower_pCE_2D_v4_auxpCE_GatedCRFLoss.py --root_path ../data/FAZ_h5 --num_classes 2 --in_chns 1 --img_class faz --exp faz/WeaklySeg_pCE_2D_v4_auxpCE_detach_GatedCRFLoss_TwoLayersMLPBN_GAP --model unet_univ4 --max_iterations 30000 --iters 5 --eval_iters 5 --tsne_iters 200 --batch_size 12 --base_lr 0.01 --amp 0 --server_address 172.18.5.181:8092 --strategy FedUniV2.1 --min_num_clients 5 --img_size 256 --alpha 0.5 --beta 0.5 --prompt universal --attention dual --dual_init aggregated --label_prompt 1 --role client --cid 3 --client client4 --sup_type box --gpu 5

python flower_pCE_2D_v4_auxpCE_GatedCRFLoss.py --root_path ../data/FAZ_h5 --num_classes 2 --in_chns 1 --img_class faz --exp faz/WeaklySeg_pCE_2D_v4_auxpCE_detach_GatedCRFLoss_TwoLayersMLPBN_GAP --model unet_univ4 --max_iterations 30000 --iters 5 --eval_iters 5 --tsne_iters 200 --batch_size 12 --base_lr 0.01 --amp 0 --server_address 172.18.5.181:8092 --strategy FedUniV2.1 --min_num_clients 5 --img_size 256 --alpha 0.5 --beta 0.5 --prompt universal --attention dual --dual_init aggregated --label_prompt 1 --role client --cid 4 --client client5 --sup_type scribble --gpu 6


# faz simple mlp + 后插值
python flower_pCE_2D_v4_auxpCE_GatedCRFLoss.py --root_path ../data/FAZ_h5 --num_classes 2 --in_chns 1 --img_class faz --exp faz/WeaklySeg_pCE_2D_v4_auxpCE_detach_GatedCRFLoss_TwoLayersMLPBN_LaterInter --model unet_univ4 --max_iterations 30000 --iters 5 --eval_iters 5 --tsne_iters 200 --batch_size 12 --base_lr 0.01 --amp 0 --server_address 172.18.5.181:8091 --strategy FedUniV2.1 --min_num_clients 5 --img_size 256 --alpha 0.5 --beta 0.5 --prompt universal --attention dual --dual_init aggregated --label_prompt 1 --role server --client client_all --sup_type mask --gpu 7

python flower_pCE_2D_v4_auxpCE_GatedCRFLoss.py --root_path ../data/FAZ_h5 --num_classes 2 --in_chns 1 --img_class faz --exp faz/WeaklySeg_pCE_2D_v4_auxpCE_detach_GatedCRFLoss_TwoLayersMLPBN_LaterInter --model unet_univ4 --max_iterations 30000 --iters 5 --eval_iters 5 --tsne_iters 200 --batch_size 12 --base_lr 0.01 --amp 0 --server_address 172.18.5.181:8091 --strategy FedUniV2.1 --min_num_clients 5 --img_size 256 --alpha 0.5 --beta 0.5 --prompt universal --attention dual --dual_init aggregated --label_prompt 1 --role client --cid 0 --client client1 --sup_type scribble_noisy --gpu 0

python flower_pCE_2D_v4_auxpCE_GatedCRFLoss.py --root_path ../data/FAZ_h5 --num_classes 2 --in_chns 1 --img_class faz --exp faz/WeaklySeg_pCE_2D_v4_auxpCE_detach_GatedCRFLoss_TwoLayersMLPBN_LaterInter --model unet_univ4 --max_iterations 30000 --iters 5 --eval_iters 5 --tsne_iters 200 --batch_size 12 --base_lr 0.01 --amp 0 --server_address 172.18.5.181:8091 --strategy FedUniV2.1 --min_num_clients 5 --img_size 256 --alpha 0.5 --beta 0.5 --prompt universal --attention dual --dual_init aggregated --label_prompt 1 --role client --cid 1 --client client2 --sup_type keypoint --gpu 1

python flower_pCE_2D_v4_auxpCE_GatedCRFLoss.py --root_path ../data/FAZ_h5 --num_classes 2 --in_chns 1 --img_class faz --exp faz/WeaklySeg_pCE_2D_v4_auxpCE_detach_GatedCRFLoss_TwoLayersMLPBN_LaterInter --model unet_univ4 --max_iterations 30000 --iters 5 --eval_iters 5 --tsne_iters 200 --batch_size 12 --base_lr 0.01 --amp 0 --server_address 172.18.5.181:8091 --strategy FedUniV2.1 --min_num_clients 5 --img_size 256 --alpha 0.5 --beta 0.5 --prompt universal --attention dual --dual_init aggregated --label_prompt 1 --role client --cid 2 --client client3 --sup_type block --gpu 2

python flower_pCE_2D_v4_auxpCE_GatedCRFLoss.py --root_path ../data/FAZ_h5 --num_classes 2 --in_chns 1 --img_class faz --exp faz/WeaklySeg_pCE_2D_v4_auxpCE_detach_GatedCRFLoss_TwoLayersMLPBN_LaterInter --model unet_univ4 --max_iterations 30000 --iters 5 --eval_iters 5 --tsne_iters 200 --batch_size 12 --base_lr 0.01 --amp 0 --server_address 172.18.5.181:8091 --strategy FedUniV2.1 --min_num_clients 5 --img_size 256 --alpha 0.5 --beta 0.5 --prompt universal --attention dual --dual_init aggregated --label_prompt 1 --role client --cid 3 --client client4 --sup_type box --gpu 5

python flower_pCE_2D_v4_auxpCE_GatedCRFLoss.py --root_path ../data/FAZ_h5 --num_classes 2 --in_chns 1 --img_class faz --exp faz/WeaklySeg_pCE_2D_v4_auxpCE_detach_GatedCRFLoss_TwoLayersMLPBN_LaterInter --model unet_univ4 --max_iterations 30000 --iters 5 --eval_iters 5 --tsne_iters 200 --batch_size 12 --base_lr 0.01 --amp 0 --server_address 172.18.5.181:8091 --strategy FedUniV2.1 --min_num_clients 5 --img_size 256 --alpha 0.5 --beta 0.5 --prompt universal --attention dual --dual_init aggregated --label_prompt 1 --role client --cid 4 --client client5 --sup_type scribble --gpu 6



# faz simple mlp + GAP + fuse personalization error
python flower_pCE_2D_v4_auxpCE_GatedCRFLoss.py --root_path ../data/FAZ_h5 --num_classes 2 --in_chns 1 --img_class faz --exp faz/WeaklySeg_pCE_2D_v4_auxpCE_detach_GatedCRFLoss_TwoLayersMLPBN_GAP_PrFuse --model unet_univ4 --max_iterations 30000 --iters 5 --eval_iters 5 --tsne_iters 200 --batch_size 12 --base_lr 0.01 --amp 0 --server_address 172.18.5.181:8093 --strategy FedUniV2.1 --min_num_clients 5 --img_size 256 --alpha 0.5 --beta 0.5 --prompt universal --attention dual --dual_init aggregated --label_prompt 1 --role server --client client_all --sup_type mask --gpu 0

python flower_pCE_2D_v4_auxpCE_GatedCRFLoss.py --root_path ../data/FAZ_h5 --num_classes 2 --in_chns 1 --img_class faz --exp faz/WeaklySeg_pCE_2D_v4_auxpCE_detach_GatedCRFLoss_TwoLayersMLPBN_GAP_PrFuse --model unet_univ4 --max_iterations 30000 --iters 5 --eval_iters 5 --tsne_iters 200 --batch_size 12 --base_lr 0.01 --amp 0 --server_address 172.18.5.181:8093 --strategy FedUniV2.1 --min_num_clients 5 --img_size 256 --alpha 0.5 --beta 0.5 --prompt universal --attention dual --dual_init aggregated --label_prompt 1 --role client --cid 0 --client client1 --sup_type scribble_noisy --gpu 0

python flower_pCE_2D_v4_auxpCE_GatedCRFLoss.py --root_path ../data/FAZ_h5 --num_classes 2 --in_chns 1 --img_class faz --exp faz/WeaklySeg_pCE_2D_v4_auxpCE_detach_GatedCRFLoss_TwoLayersMLPBN_GAP_PrFuse --model unet_univ4 --max_iterations 30000 --iters 5 --eval_iters 5 --tsne_iters 200 --batch_size 12 --base_lr 0.01 --amp 0 --server_address 172.18.5.181:8093 --strategy FedUniV2.1 --min_num_clients 5 --img_size 256 --alpha 0.5 --beta 0.5 --prompt universal --attention dual --dual_init aggregated --label_prompt 1 --role client --cid 1 --client client2 --sup_type keypoint --gpu 1

python flower_pCE_2D_v4_auxpCE_GatedCRFLoss.py --root_path ../data/FAZ_h5 --num_classes 2 --in_chns 1 --img_class faz --exp faz/WeaklySeg_pCE_2D_v4_auxpCE_detach_GatedCRFLoss_TwoLayersMLPBN_GAP_PrFuse --model unet_univ4 --max_iterations 30000 --iters 5 --eval_iters 5 --tsne_iters 200 --batch_size 12 --base_lr 0.01 --amp 0 --server_address 172.18.5.181:8093 --strategy FedUniV2.1 --min_num_clients 5 --img_size 256 --alpha 0.5 --beta 0.5 --prompt universal --attention dual --dual_init aggregated --label_prompt 1 --role client --cid 2 --client client3 --sup_type block --gpu 2

python flower_pCE_2D_v4_auxpCE_GatedCRFLoss.py --root_path ../data/FAZ_h5 --num_classes 2 --in_chns 1 --img_class faz --exp faz/WeaklySeg_pCE_2D_v4_auxpCE_detach_GatedCRFLoss_TwoLayersMLPBN_GAP_PrFuse --model unet_univ4 --max_iterations 30000 --iters 5 --eval_iters 5 --tsne_iters 200 --batch_size 12 --base_lr 0.01 --amp 0 --server_address 172.18.5.181:8093 --strategy FedUniV2.1 --min_num_clients 5 --img_size 256 --alpha 0.5 --beta 0.5 --prompt universal --attention dual --dual_init aggregated --label_prompt 1 --role client --cid 3 --client client4 --sup_type box --gpu 5

python flower_pCE_2D_v4_auxpCE_GatedCRFLoss.py --root_path ../data/FAZ_h5 --num_classes 2 --in_chns 1 --img_class faz --exp faz/WeaklySeg_pCE_2D_v4_auxpCE_detach_GatedCRFLoss_TwoLayersMLPBN_GAP_PrFuse --model unet_univ4 --max_iterations 30000 --iters 5 --eval_iters 5 --tsne_iters 200 --batch_size 12 --base_lr 0.01 --amp 0 --server_address 172.18.5.181:8093 --strategy FedUniV2.1 --min_num_clients 5 --img_size 256 --alpha 0.5 --beta 0.5 --prompt universal --attention dual --dual_init aggregated --label_prompt 1 --role client --cid 4 --client client5 --sup_type scribble --gpu 6



# faz simple mlp + GAP + distribution + uni
python flower_pCE_2D_v4_auxpCE_GatedCRFLoss.py --root_path ../data/FAZ_h5 --num_classes 2 --in_chns 1 --img_class faz --exp faz/WeaklySeg_pCE_2D_v5_auxpCE_detach_GatedCRFLoss_TwoLayersMLPBN_GAP_DisUni_2 --model unet_univ5 --max_iterations 30000 --iters 5 --eval_iters 5 --tsne_iters 200 --batch_size 12 --base_lr 0.01 --amp 0 --server_address 172.18.5.181:8093 --strategy FedUniV2.1 --min_num_clients 5 --img_size 256 --alpha 0.5 --beta 0.5 --prompt universal --attention dual --dual_init aggregated --label_prompt 1 --role server --client client_all --sup_type mask --gpu 0

python flower_pCE_2D_v4_auxpCE_GatedCRFLoss.py --root_path ../data/FAZ_h5 --num_classes 2 --in_chns 1 --img_class faz --exp faz/WeaklySeg_pCE_2D_v5_auxpCE_detach_GatedCRFLoss_TwoLayersMLPBN_GAP_DisUni_2 --model unet_univ5 --max_iterations 30000 --iters 5 --eval_iters 5 --tsne_iters 200 --batch_size 12 --base_lr 0.01 --amp 0 --server_address 172.18.5.181:8093 --strategy FedUniV2.1 --min_num_clients 5 --img_size 256 --alpha 0.5 --beta 0.5 --prompt universal --attention dual --dual_init aggregated --label_prompt 1 --role client --cid 0 --client client1 --sup_type scribble_noisy --gpu 0

python flower_pCE_2D_v4_auxpCE_GatedCRFLoss.py --root_path ../data/FAZ_h5 --num_classes 2 --in_chns 1 --img_class faz --exp faz/WeaklySeg_pCE_2D_v5_auxpCE_detach_GatedCRFLoss_TwoLayersMLPBN_GAP_DisUni_2 --model unet_univ5 --max_iterations 30000 --iters 5 --eval_iters 5 --tsne_iters 200 --batch_size 12 --base_lr 0.01 --amp 0 --server_address 172.18.5.181:8093 --strategy FedUniV2.1 --min_num_clients 5 --img_size 256 --alpha 0.5 --beta 0.5 --prompt universal --attention dual --dual_init aggregated --label_prompt 1 --role client --cid 1 --client client2 --sup_type keypoint --gpu 1

python flower_pCE_2D_v4_auxpCE_GatedCRFLoss.py --root_path ../data/FAZ_h5 --num_classes 2 --in_chns 1 --img_class faz --exp faz/WeaklySeg_pCE_2D_v5_auxpCE_detach_GatedCRFLoss_TwoLayersMLPBN_GAP_DisUni_2 --model unet_univ5 --max_iterations 30000 --iters 5 --eval_iters 5 --tsne_iters 200 --batch_size 12 --base_lr 0.01 --amp 0 --server_address 172.18.5.181:8093 --strategy FedUniV2.1 --min_num_clients 5 --img_size 256 --alpha 0.5 --beta 0.5 --prompt universal --attention dual --dual_init aggregated --label_prompt 1 --role client --cid 2 --client client3 --sup_type block --gpu 2

python flower_pCE_2D_v4_auxpCE_GatedCRFLoss.py --root_path ../data/FAZ_h5 --num_classes 2 --in_chns 1 --img_class faz --exp faz/WeaklySeg_pCE_2D_v5_auxpCE_detach_GatedCRFLoss_TwoLayersMLPBN_GAP_DisUni_2 --model unet_univ5 --max_iterations 30000 --iters 5 --eval_iters 5 --tsne_iters 200 --batch_size 12 --base_lr 0.01 --amp 0 --server_address 172.18.5.181:8093 --strategy FedUniV2.1 --min_num_clients 5 --img_size 256 --alpha 0.5 --beta 0.5 --prompt universal --attention dual --dual_init aggregated --label_prompt 1 --role client --cid 3 --client client4 --sup_type box --gpu 3

python flower_pCE_2D_v4_auxpCE_GatedCRFLoss.py --root_path ../data/FAZ_h5 --num_classes 2 --in_chns 1 --img_class faz --exp faz/WeaklySeg_pCE_2D_v5_auxpCE_detach_GatedCRFLoss_TwoLayersMLPBN_GAP_DisUni_2 --model unet_univ5 --max_iterations 30000 --iters 5 --eval_iters 5 --tsne_iters 200 --batch_size 12 --base_lr 0.01 --amp 0 --server_address 172.18.5.181:8093 --strategy FedUniV2.1 --min_num_clients 5 --img_size 256 --alpha 0.5 --beta 0.5 --prompt universal --attention dual --dual_init aggregated --label_prompt 1 --role client --cid 4 --client client5 --sup_type scribble --gpu 5




# faz simple mlp + GAP + distribution + uni
python flower_pCE_2D_v4_auxpCE_GatedCRFLoss.py --root_path ../data/FAZ_h5 --num_classes 2 --in_chns 1 --img_class faz --exp faz/WeaklySeg_pCE_2D_v5_auxpCE_detach_GatedCRFLoss_TwoLayersMLPBN_GAP_JustDistriibution_2 --model unet_univ5 --max_iterations 30000 --iters 5 --eval_iters 5 --tsne_iters 200 --batch_size 12 --base_lr 0.01 --amp 0 --server_address 172.18.5.181:8092 --strategy FedUniV2.1 --min_num_clients 5 --img_size 256 --alpha 0.5 --beta 0.5 --prompt universal --attention dual --dual_init aggregated --label_prompt 1 --role server --client client_all --sup_type mask --gpu 0

python flower_pCE_2D_v4_auxpCE_GatedCRFLoss.py --root_path ../data/FAZ_h5 --num_classes 2 --in_chns 1 --img_class faz --exp faz/WeaklySeg_pCE_2D_v5_auxpCE_detach_GatedCRFLoss_TwoLayersMLPBN_GAP_JustDistriibution_2 --model unet_univ5 --max_iterations 30000 --iters 5 --eval_iters 5 --tsne_iters 200 --batch_size 12 --base_lr 0.01 --amp 0 --server_address 172.18.5.181:8092 --strategy FedUniV2.1 --min_num_clients 5 --img_size 256 --alpha 0.5 --beta 0.5 --prompt universal --attention dual --dual_init aggregated --label_prompt 1 --role client --cid 0 --client client1 --sup_type scribble_noisy --gpu 0

python flower_pCE_2D_v4_auxpCE_GatedCRFLoss.py --root_path ../data/FAZ_h5 --num_classes 2 --in_chns 1 --img_class faz --exp faz/WeaklySeg_pCE_2D_v5_auxpCE_detach_GatedCRFLoss_TwoLayersMLPBN_GAP_JustDistriibution_2 --model unet_univ5 --max_iterations 30000 --iters 5 --eval_iters 5 --tsne_iters 200 --batch_size 12 --base_lr 0.01 --amp 0 --server_address 172.18.5.181:8092 --strategy FedUniV2.1 --min_num_clients 5 --img_size 256 --alpha 0.5 --beta 0.5 --prompt universal --attention dual --dual_init aggregated --label_prompt 1 --role client --cid 1 --client client2 --sup_type keypoint --gpu 1

python flower_pCE_2D_v4_auxpCE_GatedCRFLoss.py --root_path ../data/FAZ_h5 --num_classes 2 --in_chns 1 --img_class faz --exp faz/WeaklySeg_pCE_2D_v5_auxpCE_detach_GatedCRFLoss_TwoLayersMLPBN_GAP_JustDistriibution_2 --model unet_univ5 --max_iterations 30000 --iters 5 --eval_iters 5 --tsne_iters 200 --batch_size 12 --base_lr 0.01 --amp 0 --server_address 172.18.5.181:8092 --strategy FedUniV2.1 --min_num_clients 5 --img_size 256 --alpha 0.5 --beta 0.5 --prompt universal --attention dual --dual_init aggregated --label_prompt 1 --role client --cid 2 --client client3 --sup_type block --gpu 2

python flower_pCE_2D_v4_auxpCE_GatedCRFLoss.py --root_path ../data/FAZ_h5 --num_classes 2 --in_chns 1 --img_class faz --exp faz/WeaklySeg_pCE_2D_v5_auxpCE_detach_GatedCRFLoss_TwoLayersMLPBN_GAP_JustDistriibution_2 --model unet_univ5 --max_iterations 30000 --iters 5 --eval_iters 5 --tsne_iters 200 --batch_size 12 --base_lr 0.01 --amp 0 --server_address 172.18.5.181:8092 --strategy FedUniV2.1 --min_num_clients 5 --img_size 256 --alpha 0.5 --beta 0.5 --prompt universal --attention dual --dual_init aggregated --label_prompt 1 --role client --cid 3 --client client4 --sup_type box --gpu 3

python flower_pCE_2D_v4_auxpCE_GatedCRFLoss.py --root_path ../data/FAZ_h5 --num_classes 2 --in_chns 1 --img_class faz --exp faz/WeaklySeg_pCE_2D_v5_auxpCE_detach_GatedCRFLoss_TwoLayersMLPBN_GAP_JustDistriibution_2 --model unet_univ5 --max_iterations 30000 --iters 5 --eval_iters 5 --tsne_iters 200 --batch_size 12 --base_lr 0.01 --amp 0 --server_address 172.18.5.181:8092 --strategy FedUniV2.1 --min_num_clients 5 --img_size 256 --alpha 0.5 --beta 0.5 --prompt universal --attention dual --dual_init aggregated --label_prompt 1 --role client --cid 4 --client client5 --sup_type scribble --gpu 5





# faz simple mlp(ALA) + GAP + distribution + uni
python flower_pCE_2D_v4_auxpCE_GatedCRFLoss.py --root_path ../data/FAZ_h5 --num_classes 2 --in_chns 1 --img_class faz --exp faz/WeaklySeg_pCE_2D_v5_auxpCE_detach_GatedCRFLoss_TwoLayersMLPBN_ALA_GAP_DisUni_2 --model unet_univ5 --max_iterations 30000 --iters 5 --eval_iters 5 --tsne_iters 200 --batch_size 12 --base_lr 0.01 --amp 0 --server_address 172.18.5.181:8093 --strategy FedUniV2.1 --min_num_clients 5 --img_size 256 --alpha 0.5 --beta 0.5 --prompt universal --attention dual --dual_init aggregated --label_prompt 1 --role server --client client_all --sup_type mask --gpu 0

python flower_pCE_2D_v4_auxpCE_GatedCRFLoss.py --root_path ../data/FAZ_h5 --num_classes 2 --in_chns 1 --img_class faz --exp faz/WeaklySeg_pCE_2D_v5_auxpCE_detach_GatedCRFLoss_TwoLayersMLPBN_ALA_GAP_DisUni_2 --model unet_univ5 --max_iterations 30000 --iters 5 --eval_iters 5 --tsne_iters 200 --batch_size 12 --base_lr 0.01 --amp 0 --server_address 172.18.5.181:8093 --strategy FedUniV2.1 --min_num_clients 5 --img_size 256 --alpha 0.5 --beta 0.5 --prompt universal --attention dual --dual_init aggregated --label_prompt 1 --role client --cid 0 --client client1 --sup_type scribble_noisy --gpu 0

python flower_pCE_2D_v4_auxpCE_GatedCRFLoss.py --root_path ../data/FAZ_h5 --num_classes 2 --in_chns 1 --img_class faz --exp faz/WeaklySeg_pCE_2D_v5_auxpCE_detach_GatedCRFLoss_TwoLayersMLPBN_ALA_GAP_DisUni_2 --model unet_univ5 --max_iterations 30000 --iters 5 --eval_iters 5 --tsne_iters 200 --batch_size 12 --base_lr 0.01 --amp 0 --server_address 172.18.5.181:8093 --strategy FedUniV2.1 --min_num_clients 5 --img_size 256 --alpha 0.5 --beta 0.5 --prompt universal --attention dual --dual_init aggregated --label_prompt 1 --role client --cid 1 --client client2 --sup_type keypoint --gpu 0

python flower_pCE_2D_v4_auxpCE_GatedCRFLoss.py --root_path ../data/FAZ_h5 --num_classes 2 --in_chns 1 --img_class faz --exp faz/WeaklySeg_pCE_2D_v5_auxpCE_detach_GatedCRFLoss_TwoLayersMLPBN_ALA_GAP_DisUni_2 --model unet_univ5 --max_iterations 30000 --iters 5 --eval_iters 5 --tsne_iters 200 --batch_size 12 --base_lr 0.01 --amp 0 --server_address 172.18.5.181:8093 --strategy FedUniV2.1 --min_num_clients 5 --img_size 256 --alpha 0.5 --beta 0.5 --prompt universal --attention dual --dual_init aggregated --label_prompt 1 --role client --cid 2 --client client3 --sup_type block --gpu 1

python flower_pCE_2D_v4_auxpCE_GatedCRFLoss.py --root_path ../data/FAZ_h5 --num_classes 2 --in_chns 1 --img_class faz --exp faz/WeaklySeg_pCE_2D_v5_auxpCE_detach_GatedCRFLoss_TwoLayersMLPBN_ALA_GAP_DisUni_2 --model unet_univ5 --max_iterations 30000 --iters 5 --eval_iters 5 --tsne_iters 200 --batch_size 12 --base_lr 0.01 --amp 0 --server_address 172.18.5.181:8093 --strategy FedUniV2.1 --min_num_clients 5 --img_size 256 --alpha 0.5 --beta 0.5 --prompt universal --attention dual --dual_init aggregated --label_prompt 1 --role client --cid 3 --client client4 --sup_type box --gpu 1

python flower_pCE_2D_v4_auxpCE_GatedCRFLoss.py --root_path ../data/FAZ_h5 --num_classes 2 --in_chns 1 --img_class faz --exp faz/WeaklySeg_pCE_2D_v5_auxpCE_detach_GatedCRFLoss_TwoLayersMLPBN_ALA_GAP_DisUni_2 --model unet_univ5 --max_iterations 30000 --iters 5 --eval_iters 5 --tsne_iters 200 --batch_size 12 --base_lr 0.01 --amp 0 --server_address 172.18.5.181:8093 --strategy FedUniV2.1 --min_num_clients 5 --img_size 256 --alpha 0.5 --beta 0.5 --prompt universal --attention dual --dual_init aggregated --label_prompt 1 --role client --cid 4 --client client5 --sup_type scribble --gpu 3



# ablation faz

# faz simple mlp(no ALA) + GAP + distribution + uni w/o kl SOTA
python flower_pCE_2D_v4_auxpCE_GatedCRFLoss.py --root_path ../data/FAZ_h5 --num_classes 2 --in_chns 1 --img_class faz --exp faz/WeaklySeg_pCE_2D_v5_auxpCE_detach_GatedCRFLoss_TwoLayersMLPBN_GAP_DisUni_2_w/o_kl --model unet_univ5 --max_iterations 30000 --iters 5 --eval_iters 5 --tsne_iters 200 --batch_size 12 --base_lr 0.01 --amp 0 --server_address 172.18.5.181:8093 --strategy FedUniV2.1 --min_num_clients 5 --img_size 256 --alpha 0.0 --beta 0.5 --prompt universal --attention dual --dual_init aggregated --label_prompt 1 --role server --client client_all --sup_type mask --gpu 0

python flower_pCE_2D_v4_auxpCE_GatedCRFLoss.py --root_path ../data/FAZ_h5 --num_classes 2 --in_chns 1 --img_class faz --exp faz/WeaklySeg_pCE_2D_v5_auxpCE_detach_GatedCRFLoss_TwoLayersMLPBN_GAP_DisUni_2_w/o_kl --model unet_univ5 --max_iterations 30000 --iters 5 --eval_iters 5 --tsne_iters 200 --batch_size 12 --base_lr 0.01 --amp 0 --server_address 172.18.5.181:8093 --strategy FedUniV2.1 --min_num_clients 5 --img_size 256 --alpha 0.0 --beta 0.5 --prompt universal --attention dual --dual_init aggregated --label_prompt 1 --role client --cid 0 --client client1 --sup_type scribble_noisy --gpu 0

python flower_pCE_2D_v4_auxpCE_GatedCRFLoss.py --root_path ../data/FAZ_h5 --num_classes 2 --in_chns 1 --img_class faz --exp faz/WeaklySeg_pCE_2D_v5_auxpCE_detach_GatedCRFLoss_TwoLayersMLPBN_GAP_DisUni_2_w/o_kl --model unet_univ5 --max_iterations 30000 --iters 5 --eval_iters 5 --tsne_iters 200 --batch_size 12 --base_lr 0.01 --amp 0 --server_address 172.18.5.181:8093 --strategy FedUniV2.1 --min_num_clients 5 --img_size 256 --alpha 0.0 --beta 0.5 --prompt universal --attention dual --dual_init aggregated --label_prompt 1 --role client --cid 1 --client client2 --sup_type keypoint --gpu 1

python flower_pCE_2D_v4_auxpCE_GatedCRFLoss.py --root_path ../data/FAZ_h5 --num_classes 2 --in_chns 1 --img_class faz --exp faz/WeaklySeg_pCE_2D_v5_auxpCE_detach_GatedCRFLoss_TwoLayersMLPBN_GAP_DisUni_2_w/o_kl --model unet_univ5 --max_iterations 30000 --iters 5 --eval_iters 5 --tsne_iters 200 --batch_size 12 --base_lr 0.01 --amp 0 --server_address 172.18.5.181:8093 --strategy FedUniV2.1 --min_num_clients 5 --img_size 256 --alpha 0.0 --beta 0.5 --prompt universal --attention dual --dual_init aggregated --label_prompt 1 --role client --cid 2 --client client3 --sup_type block --gpu 1

python flower_pCE_2D_v4_auxpCE_GatedCRFLoss.py --root_path ../data/FAZ_h5 --num_classes 2 --in_chns 1 --img_class faz --exp faz/WeaklySeg_pCE_2D_v5_auxpCE_detach_GatedCRFLoss_TwoLayersMLPBN_GAP_DisUni_2_w/o_kl --model unet_univ5 --max_iterations 30000 --iters 5 --eval_iters 5 --tsne_iters 200 --batch_size 12 --base_lr 0.01 --amp 0 --server_address 172.18.5.181:8093 --strategy FedUniV2.1 --min_num_clients 5 --img_size 256 --alpha 0.0 --beta 0.5 --prompt universal --attention dual --dual_init aggregated --label_prompt 1 --role client --cid 3 --client client4 --sup_type box --gpu 2

python flower_pCE_2D_v4_auxpCE_GatedCRFLoss.py --root_path ../data/FAZ_h5 --num_classes 2 --in_chns 1 --img_class faz --exp faz/WeaklySeg_pCE_2D_v5_auxpCE_detach_GatedCRFLoss_TwoLayersMLPBN_GAP_DisUni_2_w/o_kl --model unet_univ5 --max_iterations 30000 --iters 5 --eval_iters 5 --tsne_iters 200 --batch_size 12 --base_lr 0.01 --amp 0 --server_address 172.18.5.181:8093 --strategy FedUniV2.1 --min_num_clients 5 --img_size 256 --alpha 0.0 --beta 0.5 --prompt universal --attention dual --dual_init aggregated --label_prompt 1 --role client --cid 4 --client client5 --sup_type scribble --gpu 2



# faz simple mlp(no ALA) + GAP + distribution + uni +mse
python flower_pCE_2D_v4_auxpCE_GatedCRFLoss.py --root_path ../data/FAZ_h5 --num_classes 2 --in_chns 1 --img_class faz --exp faz/WeaklySeg_pCE_2D_v5_auxpCE_detach_GatedCRFLoss_TwoLayersMLPBN_GAP_DisUni_2_mse --model unet_univ5 --max_iterations 30000 --iters 5 --eval_iters 5 --tsne_iters 200 --batch_size 12 --base_lr 0.01 --amp 0 --server_address 127.0.0.1:8199 --strategy FedUniV2.1 --min_num_clients 5 --img_size 256 --alpha 0.5 --beta 0.5 --prompt universal --attention dual --dual_init aggregated --label_prompt 1 --role server --client client_all --sup_type mask --gpu 0

python flower_pCE_2D_v4_auxpCE_GatedCRFLoss.py --root_path ../data/FAZ_h5 --num_classes 2 --in_chns 1 --img_class faz --exp faz/WeaklySeg_pCE_2D_v5_auxpCE_detach_GatedCRFLoss_TwoLayersMLPBN_GAP_DisUni_2_mse --model unet_univ5 --max_iterations 30000 --iters 5 --eval_iters 5 --tsne_iters 200 --batch_size 12 --base_lr 0.01 --amp 0 --server_address 127.0.0.1:8199 --strategy FedUniV2.1 --min_num_clients 5 --img_size 256 --alpha 0.5 --beta 0.5 --prompt universal --attention dual --dual_init aggregated --label_prompt 1 --role client --cid 0 --client client1 --sup_type scribble_noisy --gpu 0

python flower_pCE_2D_v4_auxpCE_GatedCRFLoss.py --root_path ../data/FAZ_h5 --num_classes 2 --in_chns 1 --img_class faz --exp faz/WeaklySeg_pCE_2D_v5_auxpCE_detach_GatedCRFLoss_TwoLayersMLPBN_GAP_DisUni_2_mse --model unet_univ5 --max_iterations 30000 --iters 5 --eval_iters 5 --tsne_iters 200 --batch_size 12 --base_lr 0.01 --amp 0 --server_address 127.0.0.1:8199 --strategy FedUniV2.1 --min_num_clients 5 --img_size 256 --alpha 0.5 --beta 0.5 --prompt universal --attention dual --dual_init aggregated --label_prompt 1 --role client --cid 1 --client client2 --sup_type keypoint --gpu 1

python flower_pCE_2D_v4_auxpCE_GatedCRFLoss.py --root_path ../data/FAZ_h5 --num_classes 2 --in_chns 1 --img_class faz --exp faz/WeaklySeg_pCE_2D_v5_auxpCE_detach_GatedCRFLoss_TwoLayersMLPBN_GAP_DisUni_2_mse --model unet_univ5 --max_iterations 30000 --iters 5 --eval_iters 5 --tsne_iters 200 --batch_size 12 --base_lr 0.01 --amp 0 --server_address 127.0.0.1:8199 --strategy FedUniV2.1 --min_num_clients 5 --img_size 256 --alpha 0.5 --beta 0.5 --prompt universal --attention dual --dual_init aggregated --label_prompt 1 --role client --cid 2 --client client3 --sup_type block --gpu 1

python flower_pCE_2D_v4_auxpCE_GatedCRFLoss.py --root_path ../data/FAZ_h5 --num_classes 2 --in_chns 1 --img_class faz --exp faz/WeaklySeg_pCE_2D_v5_auxpCE_detach_GatedCRFLoss_TwoLayersMLPBN_GAP_DisUni_2_mse --model unet_univ5 --max_iterations 30000 --iters 5 --eval_iters 5 --tsne_iters 200 --batch_size 12 --base_lr 0.01 --amp 0 --server_address 127.0.0.1:8199 --strategy FedUniV2.1 --min_num_clients 5 --img_size 256 --alpha 0.5 --beta 0.5 --prompt universal --attention dual --dual_init aggregated --label_prompt 1 --role client --cid 3 --client client4 --sup_type box --gpu 2

python flower_pCE_2D_v4_auxpCE_GatedCRFLoss.py --root_path ../data/FAZ_h5 --num_classes 2 --in_chns 1 --img_class faz --exp faz/WeaklySeg_pCE_2D_v5_auxpCE_detach_GatedCRFLoss_TwoLayersMLPBN_GAP_DisUni_2_mse --model unet_univ5 --max_iterations 30000 --iters 5 --eval_iters 5 --tsne_iters 200 --batch_size 12 --base_lr 0.01 --amp 0 --server_address 127.0.0.1:8199 --strategy FedUniV2.1 --min_num_clients 5 --img_size 256 --alpha 0.5 --beta 0.5 --prompt universal --attention dual --dual_init aggregated --label_prompt 1 --role client --cid 4 --client client5 --sup_type scribble --gpu 2




# 181 faz ablation

# MainAblation1_Prompt_DA_NoDecoder2
python flower_pCE_2D_v4_auxpCE_GatedCRFLoss_MainAbalation1_noDual.py --root_path ../data/FAZ_h5 --num_classes 2 --in_chns 1 --img_class faz --exp faz/MainAblation1_Prompt_DA_NoDecoder2 --model unet_univ5 --max_iterations 30000 --iters 5 --eval_iters 5 --tsne_iters 200 --batch_size 12 --base_lr 0.01 --amp 0 --server_address 127.0.0.1:8199 --strategy FedUniV2.1 --min_num_clients 5 --img_size 256 --alpha 0.0 --beta 0.0 --prompt universal --attention dual --dual_init aggregated --label_prompt 1 --role server --client client_all --sup_type mask --gpu 6

python flower_pCE_2D_v4_auxpCE_GatedCRFLoss_MainAbalation1_noDual.py --root_path ../data/FAZ_h5 --num_classes 2 --in_chns 1 --img_class faz --exp faz/MainAblation1_Prompt_DA_NoDecoder2 --model unet_univ5 --max_iterations 30000 --iters 5 --eval_iters 5 --tsne_iters 200 --batch_size 12 --base_lr 0.01 --amp 0 --server_address 127.0.0.1:8199 --strategy FedUniV2.1 --min_num_clients 5 --img_size 256 --alpha 0.0 --beta 0.0 --prompt universal --attention dual --dual_init aggregated --label_prompt 1 --role client --cid 0 --client client1 --sup_type scribble_noisy --gpu 6

python flower_pCE_2D_v4_auxpCE_GatedCRFLoss_MainAbalation1_noDual.py --root_path ../data/FAZ_h5 --num_classes 2 --in_chns 1 --img_class faz --exp faz/MainAblation1_Prompt_DA_NoDecoder2 --model unet_univ5 --max_iterations 30000 --iters 5 --eval_iters 5 --tsne_iters 200 --batch_size 12 --base_lr 0.01 --amp 0 --server_address 127.0.0.1:8199 --strategy FedUniV2.1 --min_num_clients 5 --img_size 256 --alpha 0.0 --beta 0.0 --prompt universal --attention dual --dual_init aggregated --label_prompt 1 --role client --cid 1 --client client2 --sup_type keypoint --gpu 6

python flower_pCE_2D_v4_auxpCE_GatedCRFLoss_MainAbalation1_noDual.py --root_path ../data/FAZ_h5 --num_classes 2 --in_chns 1 --img_class faz --exp faz/MainAblation1_Prompt_DA_NoDecoder2 --model unet_univ5 --max_iterations 30000 --iters 5 --eval_iters 5 --tsne_iters 200 --batch_size 12 --base_lr 0.01 --amp 0 --server_address 127.0.0.1:8199 --strategy FedUniV2.1 --min_num_clients 5 --img_size 256 --alpha 0.0 --beta 0.0 --prompt universal --attention dual --dual_init aggregated --label_prompt 1 --role client --cid 2 --client client3 --sup_type block --gpu 5

python flower_pCE_2D_v4_auxpCE_GatedCRFLoss_MainAbalation1_noDual.py --root_path ../data/FAZ_h5 --num_classes 2 --in_chns 1 --img_class faz --exp faz/MainAblation1_Prompt_DA_NoDecoder2 --model unet_univ5 --max_iterations 30000 --iters 5 --eval_iters 5 --tsne_iters 200 --batch_size 12 --base_lr 0.01 --amp 0 --server_address 127.0.0.1:8199 --strategy FedUniV2.1 --min_num_clients 5 --img_size 256 --alpha 0.0 --beta 0.0 --prompt universal --attention dual --dual_init aggregated --label_prompt 1 --role client --cid 3 --client client4 --sup_type box --gpu 5

python flower_pCE_2D_v4_auxpCE_GatedCRFLoss_MainAbalation1_noDual.py --root_path ../data/FAZ_h5 --num_classes 2 --in_chns 1 --img_class faz --exp faz/MainAblation1_Prompt_DA_NoDecoder2 --model unet_univ5 --max_iterations 30000 --iters 5 --eval_iters 5 --tsne_iters 200 --batch_size 12 --base_lr 0.01 --amp 0 --server_address 127.0.0.1:8199 --strategy FedUniV2.1 --min_num_clients 5 --img_size 256 --alpha 0.0 --beta 0.0 --prompt universal --attention dual --dual_init aggregated --label_prompt 1 --role client --cid 4 --client client5 --sup_type scribble --gpu 4


# MainAblation2_Prompt_DA_Dual_NoALA
python flower_pCE_2D_v4_auxpCE_GatedCRFLoss_MainAbalation2_noALA.py --root_path ../data/FAZ_h5 --num_classes 2 --in_chns 1 --img_class faz --exp faz/MainAblation2_Prompt_DA_Dual_NoALA --model unet_univ5 --max_iterations 30000 --iters 5 --eval_iters 5 --tsne_iters 200 --batch_size 12 --base_lr 0.01 --amp 0 --server_address 127.0.0.1:8198 --strategy FedUniV2.1 --min_num_clients 5 --img_size 256 --alpha 0.0 --beta 0.5 --prompt universal --attention dual --dual_init aggregated --label_prompt 1 --role server --client client_all --sup_type mask --gpu 4

python flower_pCE_2D_v4_auxpCE_GatedCRFLoss_MainAbalation2_noALA.py --root_path ../data/FAZ_h5 --num_classes 2 --in_chns 1 --img_class faz --exp faz/MainAblation2_Prompt_DA_Dual_NoALA --model unet_univ5 --max_iterations 30000 --iters 5 --eval_iters 5 --tsne_iters 200 --batch_size 12 --base_lr 0.01 --amp 0 --server_address 127.0.0.1:8198 --strategy FedUniV2.1 --min_num_clients 5 --img_size 256 --alpha 0.0 --beta 0.5 --prompt universal --attention dual --dual_init aggregated --label_prompt 1 --role client --cid 0 --client client1 --sup_type scribble_noisy --gpu 4

python flower_pCE_2D_v4_auxpCE_GatedCRFLoss_MainAbalation2_noALA.py --root_path ../data/FAZ_h5 --num_classes 2 --in_chns 1 --img_class faz --exp faz/MainAblation2_Prompt_DA_Dual_NoALA --model unet_univ5 --max_iterations 30000 --iters 5 --eval_iters 5 --tsne_iters 200 --batch_size 12 --base_lr 0.01 --amp 0 --server_address 127.0.0.1:8198 --strategy FedUniV2.1 --min_num_clients 5 --img_size 256 --alpha 0.0 --beta 0.5 --prompt universal --attention dual --dual_init aggregated --label_prompt 1 --role client --cid 1 --client client2 --sup_type keypoint --gpu 3

python flower_pCE_2D_v4_auxpCE_GatedCRFLoss_MainAbalation2_noALA.py --root_path ../data/FAZ_h5 --num_classes 2 --in_chns 1 --img_class faz --exp faz/MainAblation2_Prompt_DA_Dual_NoALA --model unet_univ5 --max_iterations 30000 --iters 5 --eval_iters 5 --tsne_iters 200 --batch_size 12 --base_lr 0.01 --amp 0 --server_address 127.0.0.1:8198 --strategy FedUniV2.1 --min_num_clients 5 --img_size 256 --alpha 0.0 --beta 0.5 --prompt universal --attention dual --dual_init aggregated --label_prompt 1 --role client --cid 2 --client client3 --sup_type block --gpu 3

python flower_pCE_2D_v4_auxpCE_GatedCRFLoss_MainAbalation2_noALA.py --root_path ../data/FAZ_h5 --num_classes 2 --in_chns 1 --img_class faz --exp faz/MainAblation2_Prompt_DA_Dual_NoALA --model unet_univ5 --max_iterations 30000 --iters 5 --eval_iters 5 --tsne_iters 200 --batch_size 12 --base_lr 0.01 --amp 0 --server_address 127.0.0.1:8198 --strategy FedUniV2.1 --min_num_clients 5 --img_size 256 --alpha 0.0 --beta 0.5 --prompt universal --attention dual --dual_init aggregated --label_prompt 1 --role client --cid 3 --client client4 --sup_type box --gpu 2

python flower_pCE_2D_v4_auxpCE_GatedCRFLoss_MainAbalation2_noALA.py --root_path ../data/FAZ_h5 --num_classes 2 --in_chns 1 --img_class faz --exp faz/MainAblation2_Prompt_DA_Dual_NoALA --model unet_univ5 --max_iterations 30000 --iters 5 --eval_iters 5 --tsne_iters 200 --batch_size 12 --base_lr 0.01 --amp 0 --server_address 127.0.0.1:8198 --strategy FedUniV2.1 --min_num_clients 5 --img_size 256 --alpha 0.0 --beta 0.5 --prompt universal --attention dual --dual_init aggregated --label_prompt 1 --role client --cid 4 --client client5 --sup_type scribble --gpu 2




# odoc MainAblation1
python flower_pCE_2D_v4_auxpCE_GatedCRFLoss_MainAbalation1_noDual.py --root_path ../data/ODOC_h5 --num_classes 3 --in_chns 3 --img_class odoc --exp odoc/MainAblation1_Prompt_DA_NoDecoder2 --model unet_univ5 --max_iterations 30000 --iters 5 --eval_iters 5 --tsne_iters 200 --batch_size 12 --base_lr 0.05 --amp 0 --server_address 127.0.0.1:8091 --strategy FedUniV2.1 --min_num_clients 5 --img_size 384 --alpha 0.0 --beta 0.0 --prompt universal --attention dual --dual_init aggregated --label_prompt 1 --role server --client client_all --sup_type mask --gpu 0

python flower_pCE_2D_v4_auxpCE_GatedCRFLoss_MainAbalation1_noDual.py --root_path ../data/ODOC_h5 --num_classes 3 --in_chns 3 --img_class odoc --exp odoc/MainAblation1_Prompt_DA_NoDecoder2 --model unet_univ5 --max_iterations 30000 --iters 5 --eval_iters 5 --tsne_iters 200 --batch_size 12 --base_lr 0.05 --amp 0 --server_address 127.0.0.1:8091 --strategy FedUniV2.1 --min_num_clients 5 --img_size 384 --alpha 0.0 --beta 0.0 --prompt universal --attention dual --dual_init aggregated --label_prompt 1 --role client --cid 0 --client client1 --sup_type scribble --gpu 3

python flower_pCE_2D_v4_auxpCE_GatedCRFLoss_MainAbalation1_noDual.py --root_path ../data/ODOC_h5 --num_classes 3 --in_chns 3 --img_class odoc --exp odoc/MainAblation1_Prompt_DA_NoDecoder2 --model unet_univ5 --max_iterations 30000 --iters 5 --eval_iters 5 --tsne_iters 200 --batch_size 12 --base_lr 0.05 --amp 0 --server_address 127.0.0.1:8091 --strategy FedUniV2.1 --min_num_clients 5 --img_size 384 --alpha 0.0 --beta 0.0 --prompt universal --attention dual --dual_init aggregated --label_prompt 1 --role client --cid 1 --client client2 --sup_type scribble_noisy --gpu 4

python flower_pCE_2D_v4_auxpCE_GatedCRFLoss_MainAbalation1_noDual.py --root_path ../data/ODOC_h5 --num_classes 3 --in_chns 3 --img_class odoc --exp odoc/MainAblation1_Prompt_DA_NoDecoder2 --model unet_univ5 --max_iterations 30000 --iters 5 --eval_iters 5 --tsne_iters 200 --batch_size 12 --base_lr 0.05 --amp 0 --server_address 127.0.0.1:8091 --strategy FedUniV2.1 --min_num_clients 5 --img_size 384 --alpha 0.0 --beta 0.0 --prompt universal --attention dual --dual_init aggregated --label_prompt 1 --role client --cid 2 --client client3 --sup_type scribble_noisy --gpu 5

python flower_pCE_2D_v4_auxpCE_GatedCRFLoss_MainAbalation1_noDual.py --root_path ../data/ODOC_h5 --num_classes 3 --in_chns 3 --img_class odoc --exp odoc/MainAblation1_Prompt_DA_NoDecoder2 --model unet_univ5 --max_iterations 30000 --iters 5 --eval_iters 5 --tsne_iters 200 --batch_size 12 --base_lr 0.05 --amp 0 --server_address 127.0.0.1:8091 --strategy FedUniV2.1 --min_num_clients 5 --img_size 384 --alpha 0.0 --beta 0.0 --prompt universal --attention dual --dual_init aggregated --label_prompt 1 --role client --cid 3 --client client4 --sup_type keypoint --gpu 6

python flower_pCE_2D_v4_auxpCE_GatedCRFLoss_MainAbalation1_noDual.py --root_path ../data/ODOC_h5 --num_classes 3 --in_chns 3 --img_class odoc --exp odoc/MainAblation1_Prompt_DA_NoDecoder2 --model unet_univ5 --max_iterations 30000 --iters 5 --eval_iters 5 --tsne_iters 200 --batch_size 12 --base_lr 0.05 --amp 0 --server_address 127.0.0.1:8091 --strategy FedUniV2.1 --min_num_clients 5 --img_size 384 --alpha 0.0 --beta 0.0 --prompt universal --attention dual --dual_init aggregated --label_prompt 1 --role client --cid 4 --client client5 --sup_type block --gpu 7



# odoc MainAblation2
python flower_pCE_2D_v4_auxpCE_GatedCRFLoss_MainAbalation2_noALA.py --root_path ../data/ODOC_h5 --num_classes 3 --in_chns 3 --img_class odoc --exp odoc/MainAblation2_Prompt_DA_Dual_NoALA --model unet_univ5 --max_iterations 30000 --iters 5 --eval_iters 5 --tsne_iters 200 --batch_size 12 --base_lr 0.05 --amp 0 --server_address 127.0.0.1:8091 --strategy FedUniV2.1 --min_num_clients 5 --img_size 384 --alpha 0.0 --beta 0.5 --prompt universal --attention dual --dual_init aggregated --label_prompt 1 --role server --client client_all --sup_type mask --gpu 0

python flower_pCE_2D_v4_auxpCE_GatedCRFLoss_MainAbalation2_noALA.py --root_path ../data/ODOC_h5 --num_classes 3 --in_chns 3 --img_class odoc --exp odoc/MainAblation2_Prompt_DA_Dual_NoALA --model unet_univ5 --max_iterations 30000 --iters 5 --eval_iters 5 --tsne_iters 200 --batch_size 12 --base_lr 0.05 --amp 0 --server_address 127.0.0.1:8091 --strategy FedUniV2.1 --min_num_clients 5 --img_size 384 --alpha 0.0 --beta 0.5 --prompt universal --attention dual --dual_init aggregated --label_prompt 1 --role client --cid 0 --client client1 --sup_type scribble --gpu 3

python flower_pCE_2D_v4_auxpCE_GatedCRFLoss_MainAbalation2_noALA.py --root_path ../data/ODOC_h5 --num_classes 3 --in_chns 3 --img_class odoc --exp odoc/MainAblation2_Prompt_DA_Dual_NoALA --model unet_univ5 --max_iterations 30000 --iters 5 --eval_iters 5 --tsne_iters 200 --batch_size 12 --base_lr 0.05 --amp 0 --server_address 127.0.0.1:8091 --strategy FedUniV2.1 --min_num_clients 5 --img_size 384 --alpha 0.0 --beta 0.5 --prompt universal --attention dual --dual_init aggregated --label_prompt 1 --role client --cid 1 --client client2 --sup_type scribble_noisy --gpu 4

python flower_pCE_2D_v4_auxpCE_GatedCRFLoss_MainAbalation2_noALA.py --root_path ../data/ODOC_h5 --num_classes 3 --in_chns 3 --img_class odoc --exp odoc/MainAblation2_Prompt_DA_Dual_NoALA --model unet_univ5 --max_iterations 30000 --iters 5 --eval_iters 5 --tsne_iters 200 --batch_size 12 --base_lr 0.05 --amp 0 --server_address 127.0.0.1:8091 --strategy FedUniV2.1 --min_num_clients 5 --img_size 384 --alpha 0.0 --beta 0.5 --prompt universal --attention dual --dual_init aggregated --label_prompt 1 --role client --cid 2 --client client3 --sup_type scribble_noisy --gpu 5

python flower_pCE_2D_v4_auxpCE_GatedCRFLoss_MainAbalation2_noALA.py --root_path ../data/ODOC_h5 --num_classes 3 --in_chns 3 --img_class odoc --exp odoc/MainAblation2_Prompt_DA_Dual_NoALA --model unet_univ5 --max_iterations 30000 --iters 5 --eval_iters 5 --tsne_iters 200 --batch_size 12 --base_lr 0.05 --amp 0 --server_address 127.0.0.1:8091 --strategy FedUniV2.1 --min_num_clients 5 --img_size 384 --alpha 0.0 --beta 0.5 --prompt universal --attention dual --dual_init aggregated --label_prompt 1 --role client --cid 3 --client client4 --sup_type keypoint --gpu 6

python flower_pCE_2D_v4_auxpCE_GatedCRFLoss_MainAbalation2_noALA.py --root_path ../data/ODOC_h5 --num_classes 3 --in_chns 3 --img_class odoc --exp odoc/MainAblation2_Prompt_DA_Dual_NoALA --model unet_univ5 --max_iterations 30000 --iters 5 --eval_iters 5 --tsne_iters 200 --batch_size 12 --base_lr 0.05 --amp 0 --server_address 127.0.0.1:8091 --strategy FedUniV2.1 --min_num_clients 5 --img_size 384 --alpha 0.0 --beta 0.5 --prompt universal --attention dual --dual_init aggregated --label_prompt 1 --role client --cid 4 --client client5 --sup_type block --gpu 7


# odoc MainAblation3
python flower_pCE_2D_v4_auxpCE_GatedCRFLoss.py --root_path ../data/ODOC_h5 --num_classes 3 --in_chns 3 --img_class odoc --exp odoc/MainAblation2_Prompt_DA_ALA_NoDual --model unet_univ5 --max_iterations 30000 --iters 5 --eval_iters 5 --tsne_iters 200 --batch_size 12 --base_lr 0.05 --amp 0 --server_address 127.0.0.1:8091 --strategy FedUniV2.1 --min_num_clients 5 --img_size 384 --alpha 0.0 --beta 0.0 --prompt universal --attention dual --dual_init aggregated --label_prompt 1 --role server --client client_all --sup_type mask --gpu 0

python flower_pCE_2D_v4_auxpCE_GatedCRFLoss.py --root_path ../data/ODOC_h5 --num_classes 3 --in_chns 3 --img_class odoc --exp odoc/MainAblation2_Prompt_DA_ALA_NoDual --model unet_univ5 --max_iterations 30000 --iters 5 --eval_iters 5 --tsne_iters 200 --batch_size 12 --base_lr 0.05 --amp 0 --server_address 127.0.0.1:8091 --strategy FedUniV2.1 --min_num_clients 5 --img_size 384 --alpha 0.0 --beta 0.0 --prompt universal --attention dual --dual_init aggregated --label_prompt 1 --role client --cid 0 --client client1 --sup_type scribble --gpu 3

python flower_pCE_2D_v4_auxpCE_GatedCRFLoss.py --root_path ../data/ODOC_h5 --num_classes 3 --in_chns 3 --img_class odoc --exp odoc/MainAblation2_Prompt_DA_ALA_NoDual --model unet_univ5 --max_iterations 30000 --iters 5 --eval_iters 5 --tsne_iters 200 --batch_size 12 --base_lr 0.05 --amp 0 --server_address 127.0.0.1:8091 --strategy FedUniV2.1 --min_num_clients 5 --img_size 384 --alpha 0.0 --beta 0.0 --prompt universal --attention dual --dual_init aggregated --label_prompt 1 --role client --cid 1 --client client2 --sup_type scribble_noisy --gpu 4

python flower_pCE_2D_v4_auxpCE_GatedCRFLoss.py --root_path ../data/ODOC_h5 --num_classes 3 --in_chns 3 --img_class odoc --exp odoc/MainAblation2_Prompt_DA_ALA_NoDual --model unet_univ5 --max_iterations 30000 --iters 5 --eval_iters 5 --tsne_iters 200 --batch_size 12 --base_lr 0.05 --amp 0 --server_address 127.0.0.1:8091 --strategy FedUniV2.1 --min_num_clients 5 --img_size 384 --alpha 0.0 --beta 0.0 --prompt universal --attention dual --dual_init aggregated --label_prompt 1 --role client --cid 2 --client client3 --sup_type scribble_noisy --gpu 5

python flower_pCE_2D_v4_auxpCE_GatedCRFLoss.py --root_path ../data/ODOC_h5 --num_classes 3 --in_chns 3 --img_class odoc --exp odoc/MainAblation2_Prompt_DA_ALA_NoDual --model unet_univ5 --max_iterations 30000 --iters 5 --eval_iters 5 --tsne_iters 200 --batch_size 12 --base_lr 0.05 --amp 0 --server_address 127.0.0.1:8091 --strategy FedUniV2.1 --min_num_clients 5 --img_size 384 --alpha 0.0 --beta 0.0 --prompt universal --attention dual --dual_init aggregated --label_prompt 1 --role client --cid 3 --client client4 --sup_type keypoint --gpu 6

python flower_pCE_2D_v4_auxpCE_GatedCRFLoss.py --root_path ../data/ODOC_h5 --num_classes 3 --in_chns 3 --img_class odoc --exp odoc/MainAblation2_Prompt_DA_ALA_NoDual --model unet_univ5 --max_iterations 30000 --iters 5 --eval_iters 5 --tsne_iters 200 --batch_size 12 --base_lr 0.05 --amp 0 --server_address 127.0.0.1:8091 --strategy FedUniV2.1 --min_num_clients 5 --img_size 384 --alpha 0.0 --beta 0.0 --prompt universal --attention dual --dual_init aggregated --label_prompt 1 --role client --cid 4 --client client5 --sup_type block --gpu 7

# odoc MainAblation4 NoDA
python flower_pCE_2D_v4_auxpCE_GatedCRFLoss_MainAblation4_noDA.py --root_path ../data/ODOC_h5 --num_classes 3 --in_chns 3 --img_class odoc --exp odoc/MainAblation4_Prompt_ALA_Dual_NoDA --model unet_univ5_ablation --max_iterations 30000 --iters 5 --eval_iters 5 --tsne_iters 200 --batch_size 12 --base_lr 0.05 --amp 0 --server_address 127.0.0.1:8091 --strategy FedALA --min_num_clients 5 --img_size 384 --alpha 0.0 --beta 0.5 --role server --client client_all --sup_type mask --gpu 0

python flower_pCE_2D_v4_auxpCE_GatedCRFLoss_MainAblation4_noDA.py --root_path ../data/ODOC_h5 --num_classes 3 --in_chns 3 --img_class odoc --exp odoc/MainAblation4_Prompt_ALA_Dual_NoDA --model unet_univ5_ablation --max_iterations 30000 --iters 5 --eval_iters 5 --tsne_iters 200 --batch_size 12 --base_lr 0.05 --amp 0 --server_address 127.0.0.1:8091 --strategy FedALA --min_num_clients 5 --img_size 384 --alpha 0.0 --beta 0.5 --role client --cid 0 --client client1 --sup_type scribble --gpu 3

python flower_pCE_2D_v4_auxpCE_GatedCRFLoss_MainAblation4_noDA.py --root_path ../data/ODOC_h5 --num_classes 3 --in_chns 3 --img_class odoc --exp odoc/MainAblation4_Prompt_ALA_Dual_NoDA --model unet_univ5_ablation --max_iterations 30000 --iters 5 --eval_iters 5 --tsne_iters 200 --batch_size 12 --base_lr 0.05 --amp 0 --server_address 127.0.0.1:8091 --strategy FedALA --min_num_clients 5 --img_size 384 --alpha 0.0 --beta 0.5 --role client --cid 1 --client client2 --sup_type scribble_noisy --gpu 4

python flower_pCE_2D_v4_auxpCE_GatedCRFLoss_MainAblation4_noDA.py --root_path ../data/ODOC_h5 --num_classes 3 --in_chns 3 --img_class odoc --exp odoc/MainAblation4_Prompt_ALA_Dual_NoDA --model unet_univ5_ablation --max_iterations 30000 --iters 5 --eval_iters 5 --tsne_iters 200 --batch_size 12 --base_lr 0.05 --amp 0 --server_address 127.0.0.1:8091 --strategy FedALA --min_num_clients 5 --img_size 384 --alpha 0.0 --beta 0.5 --role client --cid 2 --client client3 --sup_type scribble_noisy --gpu 5

python flower_pCE_2D_v4_auxpCE_GatedCRFLoss_MainAblation4_noDA.py --root_path ../data/ODOC_h5 --num_classes 3 --in_chns 3 --img_class odoc --exp odoc/MainAblation4_Prompt_ALA_Dual_NoDA --model unet_univ5_ablation --max_iterations 30000 --iters 5 --eval_iters 5 --tsne_iters 200 --batch_size 12 --base_lr 0.05 --amp 0 --server_address 127.0.0.1:8091 --strategy FedALA --min_num_clients 5 --img_size 384 --alpha 0.0 --beta 0.5 --role client --cid 3 --client client4 --sup_type keypoint --gpu 6

python flower_pCE_2D_v4_auxpCE_GatedCRFLoss_MainAblation4_noDA.py --root_path ../data/ODOC_h5 --num_classes 3 --in_chns 3 --img_class odoc --exp odoc/MainAblation4_Prompt_ALA_Dual_NoDA --model unet_univ5_ablation --max_iterations 30000 --iters 5 --eval_iters 5 --tsne_iters 200 --batch_size 12 --base_lr 0.05 --amp 0 --server_address 127.0.0.1:8091 --strategy FedALA --min_num_clients 5 --img_size 384 --alpha 0.0 --beta 0.5 --role client --cid 4 --client client5 --sup_type block --gpu 7


# ablation da
# sab
# odoc MLP dis+uni lr 0.05 mlp(ALA)
python flower_pCE_2D_v4_auxpCE_GatedCRFLoss.py --root_path ../data/ODOC_h5 --num_classes 3 --in_chns 3 --img_class odoc --exp odoc/DA_Ablation_Sab --model unet_univ5 --max_iterations 30000 --iters 5 --eval_iters 5 --tsne_iters 200 --batch_size 12 --base_lr 0.05 --amp 0 --server_address 127.0.0.1:8091 --strategy FedUniV2.1 --min_num_clients 5 --img_size 384 --alpha 0.5 --beta 0.5 --prompt universal --attention sab --dual_init aggregated --label_prompt 1 --role server --client client_all --sup_type mask --gpu 1

python flower_pCE_2D_v4_auxpCE_GatedCRFLoss.py --root_path ../data/ODOC_h5 --num_classes 3 --in_chns 3 --img_class odoc --exp odoc/DA_Ablation_Sab --model unet_univ5 --max_iterations 30000 --iters 5 --eval_iters 5 --tsne_iters 200 --batch_size 12 --base_lr 0.05 --amp 0 --server_address 127.0.0.1:8091 --strategy FedUniV2.1 --min_num_clients 5 --img_size 384 --alpha 0.5 --beta 0.5 --prompt universal --attention sab --dual_init aggregated --label_prompt 1 --role client --cid 0 --client client1 --sup_type scribble --gpu 3

python flower_pCE_2D_v4_auxpCE_GatedCRFLoss.py --root_path ../data/ODOC_h5 --num_classes 3 --in_chns 3 --img_class odoc --exp odoc/DA_Ablation_Sab --model unet_univ5 --max_iterations 30000 --iters 5 --eval_iters 5 --tsne_iters 200 --batch_size 12 --base_lr 0.05 --amp 0 --server_address 127.0.0.1:8091 --strategy FedUniV2.1 --min_num_clients 5 --img_size 384 --alpha 0.5 --beta 0.5 --prompt universal --attention sab --dual_init aggregated --label_prompt 1 --role client --cid 1 --client client2 --sup_type scribble_noisy --gpu 4

python flower_pCE_2D_v4_auxpCE_GatedCRFLoss.py --root_path ../data/ODOC_h5 --num_classes 3 --in_chns 3 --img_class odoc --exp odoc/DA_Ablation_Sab --model unet_univ5 --max_iterations 30000 --iters 5 --eval_iters 5 --tsne_iters 200 --batch_size 12 --base_lr 0.05 --amp 0 --server_address 127.0.0.1:8091 --strategy FedUniV2.1 --min_num_clients 5 --img_size 384 --alpha 0.5 --beta 0.5 --prompt universal --attention sab --dual_init aggregated --label_prompt 1 --role client --cid 2 --client client3 --sup_type scribble_noisy --gpu 5

python flower_pCE_2D_v4_auxpCE_GatedCRFLoss.py --root_path ../data/ODOC_h5 --num_classes 3 --in_chns 3 --img_class odoc --exp odoc/DA_Ablation_Sab --model unet_univ5 --max_iterations 30000 --iters 5 --eval_iters 5 --tsne_iters 200 --batch_size 12 --base_lr 0.05 --amp 0 --server_address 127.0.0.1:8091 --strategy FedUniV2.1 --min_num_clients 5 --img_size 384 --alpha 0.5 --beta 0.5 --prompt universal --attention sab --dual_init aggregated --label_prompt 1 --role client --cid 3 --client client4 --sup_type keypoint --gpu 6

python flower_pCE_2D_v4_auxpCE_GatedCRFLoss.py --root_path ../data/ODOC_h5 --num_classes 3 --in_chns 3 --img_class odoc --exp odoc/DA_Ablation_Sab --model unet_univ5 --max_iterations 30000 --iters 5 --eval_iters 5 --tsne_iters 200 --batch_size 12 --base_lr 0.05 --amp 0 --server_address 127.0.0.1:8091 --strategy FedUniV2.1 --min_num_clients 5 --img_size 384 --alpha 0.5 --beta 0.5 --prompt universal --attention sab --dual_init aggregated --label_prompt 1 --role client --cid 4 --client client5 --sup_type block --gpu 7


# ablation da
# cab
# odoc MLP dis+uni lr 0.05 mlp(ALA)
python flower_pCE_2D_v4_auxpCE_GatedCRFLoss.py --root_path ../data/ODOC_h5 --num_classes 3 --in_chns 3 --img_class odoc --exp odoc/DA_Ablation_Cab --model unet_univ5 --max_iterations 30000 --iters 5 --eval_iters 5 --tsne_iters 200 --batch_size 12 --base_lr 0.05 --amp 0 --server_address 127.0.0.1:8099 --strategy FedUniV2.1 --min_num_clients 5 --img_size 384 --alpha 0.5 --beta 0.5 --prompt universal --attention cab --dual_init aggregated --label_prompt 1 --role server --client client_all --sup_type mask --gpu 2

python flower_pCE_2D_v4_auxpCE_GatedCRFLoss.py --root_path ../data/ODOC_h5 --num_classes 3 --in_chns 3 --img_class odoc --exp odoc/DA_Ablation_Cab --model unet_univ5 --max_iterations 30000 --iters 5 --eval_iters 5 --tsne_iters 200 --batch_size 12 --base_lr 0.05 --amp 0 --server_address 127.0.0.1:8099 --strategy FedUniV2.1 --min_num_clients 5 --img_size 384 --alpha 0.5 --beta 0.5 --prompt universal --attention cab --dual_init aggregated --label_prompt 1 --role client --cid 0 --client client1 --sup_type scribble --gpu 3

python flower_pCE_2D_v4_auxpCE_GatedCRFLoss.py --root_path ../data/ODOC_h5 --num_classes 3 --in_chns 3 --img_class odoc --exp odoc/DA_Ablation_Cab --model unet_univ5 --max_iterations 30000 --iters 5 --eval_iters 5 --tsne_iters 200 --batch_size 12 --base_lr 0.05 --amp 0 --server_address 127.0.0.1:8099 --strategy FedUniV2.1 --min_num_clients 5 --img_size 384 --alpha 0.5 --beta 0.5 --prompt universal --attention cab --dual_init aggregated --label_prompt 1 --role client --cid 1 --client client2 --sup_type scribble_noisy --gpu 4

python flower_pCE_2D_v4_auxpCE_GatedCRFLoss.py --root_path ../data/ODOC_h5 --num_classes 3 --in_chns 3 --img_class odoc --exp odoc/DA_Ablation_Cab --model unet_univ5 --max_iterations 30000 --iters 5 --eval_iters 5 --tsne_iters 200 --batch_size 12 --base_lr 0.05 --amp 0 --server_address 127.0.0.1:8099 --strategy FedUniV2.1 --min_num_clients 5 --img_size 384 --alpha 0.5 --beta 0.5 --prompt universal --attention cab --dual_init aggregated --label_prompt 1 --role client --cid 2 --client client3 --sup_type scribble_noisy --gpu 5

python flower_pCE_2D_v4_auxpCE_GatedCRFLoss.py --root_path ../data/ODOC_h5 --num_classes 3 --in_chns 3 --img_class odoc --exp odoc/DA_Ablation_Cab --model unet_univ5 --max_iterations 30000 --iters 5 --eval_iters 5 --tsne_iters 200 --batch_size 12 --base_lr 0.05 --amp 0 --server_address 127.0.0.1:8099 --strategy FedUniV2.1 --min_num_clients 5 --img_size 384 --alpha 0.5 --beta 0.5 --prompt universal --attention cab --dual_init aggregated --label_prompt 1 --role client --cid 3 --client client4 --sup_type keypoint --gpu 6

python flower_pCE_2D_v4_auxpCE_GatedCRFLoss.py --root_path ../data/ODOC_h5 --num_classes 3 --in_chns 3 --img_class odoc --exp odoc/DA_Ablation_Cab --model unet_univ5 --max_iterations 30000 --iters 5 --eval_iters 5 --tsne_iters 200 --batch_size 12 --base_lr 0.05 --amp 0 --server_address 127.0.0.1:8099 --strategy FedUniV2.1 --min_num_clients 5 --img_size 384 --alpha 0.5 --beta 0.5 --prompt universal --attention cab --dual_init aggregated --label_prompt 1 --role client --cid 4 --client client5 --sup_type block --gpu 7



# ablation dual

python flower_pCE_2D_v4_auxpCE_GatedCRFLoss.py --root_path ../data/ODOC_h5 --num_classes 3 --in_chns 3 --img_class odoc --exp odoc/Prompt_Ablation_NoLabelPrompt --model unet_univ5 --max_iterations 30000 --iters 5 --eval_iters 5 --tsne_iters 200 --batch_size 12 --base_lr 0.05 --amp 0 --server_address 127.0.0.1:8099 --strategy FedUniV2.1 --min_num_clients 5 --img_size 384 --alpha 0.5 --beta 0.5 --prompt universal --attention dual --dual_init aggregated --label_prompt 1 --role server --client client_all --sup_type mask --gpu 0

python flower_pCE_2D_v4_auxpCE_GatedCRFLoss.py --root_path ../data/ODOC_h5 --num_classes 3 --in_chns 3 --img_class odoc --exp odoc/Prompt_Ablation_NoLabelPrompt --model unet_univ5 --max_iterations 30000 --iters 5 --eval_iters 5 --tsne_iters 200 --batch_size 12 --base_lr 0.05 --amp 0 --server_address 127.0.0.1:8099 --strategy FedUniV2.1 --min_num_clients 5 --img_size 384 --alpha 0.5 --beta 0.5 --prompt universal --attention dual --dual_init aggregated --label_prompt 1 --role client --cid 0 --client client1 --sup_type scribble --gpu 1

python flower_pCE_2D_v4_auxpCE_GatedCRFLoss.py --root_path ../data/ODOC_h5 --num_classes 3 --in_chns 3 --img_class odoc --exp odoc/Prompt_Ablation_NoLabelPrompt --model unet_univ5 --max_iterations 30000 --iters 5 --eval_iters 5 --tsne_iters 200 --batch_size 12 --base_lr 0.05 --amp 0 --server_address 127.0.0.1:8099 --strategy FedUniV2.1 --min_num_clients 5 --img_size 384 --alpha 0.5 --beta 0.5 --prompt universal --attention dual --dual_init aggregated --label_prompt 1 --role client --cid 1 --client client2 --sup_type scribble_noisy --gpu 4

python flower_pCE_2D_v4_auxpCE_GatedCRFLoss.py --root_path ../data/ODOC_h5 --num_classes 3 --in_chns 3 --img_class odoc --exp odoc/Prompt_Ablation_NoLabelPrompt --model unet_univ5 --max_iterations 30000 --iters 5 --eval_iters 5 --tsne_iters 200 --batch_size 12 --base_lr 0.05 --amp 0 --server_address 127.0.0.1:8099 --strategy FedUniV2.1 --min_num_clients 5 --img_size 384 --alpha 0.5 --beta 0.5 --prompt universal --attention dual --dual_init aggregated --label_prompt 1 --role client --cid 2 --client client3 --sup_type scribble_noisy --gpu 5

python flower_pCE_2D_v4_auxpCE_GatedCRFLoss.py --root_path ../data/ODOC_h5 --num_classes 3 --in_chns 3 --img_class odoc --exp odoc/Prompt_Ablation_NoLabelPrompt --model unet_univ5 --max_iterations 30000 --iters 5 --eval_iters 5 --tsne_iters 200 --batch_size 12 --base_lr 0.05 --amp 0 --server_address 127.0.0.1:8099 --strategy FedUniV2.1 --min_num_clients 5 --img_size 384 --alpha 0.5 --beta 0.5 --prompt universal --attention dual --dual_init aggregated --label_prompt 1 --role client --cid 3 --client client4 --sup_type keypoint --gpu 6

python flower_pCE_2D_v4_auxpCE_GatedCRFLoss.py --root_path ../data/ODOC_h5 --num_classes 3 --in_chns 3 --img_class odoc --exp odoc/Prompt_Ablation_NoLabelPrompt --model unet_univ5 --max_iterations 30000 --iters 5 --eval_iters 5 --tsne_iters 200 --batch_size 12 --base_lr 0.05 --amp 0 --server_address 127.0.0.1:8099 --strategy FedUniV2.1 --min_num_clients 5 --img_size 384 --alpha 0.5 --beta 0.5 --prompt universal --attention dual --dual_init aggregated --label_prompt 1 --role client --cid 4 --client client5 --sup_type block --gpu 7





# ajacent
python flower_pCE_2D_v4_auxpCE_GatedCRFLoss.py --root_path ../data/ODOC_h5 --num_classes 3 --in_chns 3 --img_class odoc --exp odoc/Prompt_Ablation_NoLabelPrompt --model unet_univ5 --max_iterations 30000 --iters 5 --eval_iters 5 --tsne_iters 200 --batch_size 12 --base_lr 0.05 --amp 0 --server_address 127.0.0.1:8092 --strategy FedUniV2.1 --min_num_clients 5 --img_size 384 --alpha 0.5 --beta 0.5 --prompt universal --attention dual --dual_init aggregated --label_prompt 0 --role server --client client_all --sup_type mask --gpu 3

python flower_pCE_2D_v4_auxpCE_GatedCRFLoss.py --root_path ../data/ODOC_h5 --num_classes 3 --in_chns 3 --img_class odoc --exp odoc/Prompt_Ablation_NoLabelPrompt --model unet_univ5 --max_iterations 30000 --iters 5 --eval_iters 5 --tsne_iters 200 --batch_size 12 --base_lr 0.05 --amp 0 --server_address 127.0.0.1:8092 --strategy FedUniV2.1 --min_num_clients 5 --img_size 384 --alpha 0.5 --beta 0.5 --prompt universal --attention dual --dual_init aggregated --label_prompt 0 --role client --cid 0 --client client1 --sup_type scribble --gpu 5

python flower_pCE_2D_v4_auxpCE_GatedCRFLoss.py --root_path ../data/ODOC_h5 --num_classes 3 --in_chns 3 --img_class odoc --exp odoc/Prompt_Ablation_NoLabelPrompt --model unet_univ5 --max_iterations 30000 --iters 5 --eval_iters 5 --tsne_iters 200 --batch_size 12 --base_lr 0.05 --amp 0 --server_address 127.0.0.1:8092 --strategy FedUniV2.1 --min_num_clients 5 --img_size 384 --alpha 0.5 --beta 0.5 --prompt universal --attention dual --dual_init aggregated --label_prompt 0 --role client --cid 1 --client client2 --sup_type scribble_noisy --gpu 4

python flower_pCE_2D_v4_auxpCE_GatedCRFLoss.py --root_path ../data/ODOC_h5 --num_classes 3 --in_chns 3 --img_class odoc --exp odoc/Prompt_Ablation_NoLabelPrompt --model unet_univ5 --max_iterations 30000 --iters 5 --eval_iters 5 --tsne_iters 200 --batch_size 12 --base_lr 0.05 --amp 0 --server_address 127.0.0.1:8092 --strategy FedUniV2.1 --min_num_clients 5 --img_size 384 --alpha 0.5 --beta 0.5 --prompt universal --attention dual --dual_init aggregated --label_prompt 0 --role client --cid 2 --client client3 --sup_type scribble_noisy --gpu 0

python flower_pCE_2D_v4_auxpCE_GatedCRFLoss.py --root_path ../data/ODOC_h5 --num_classes 3 --in_chns 3 --img_class odoc --exp odoc/Prompt_Ablation_NoLabelPrompt --model unet_univ5 --max_iterations 30000 --iters 5 --eval_iters 5 --tsne_iters 200 --batch_size 12 --base_lr 0.05 --amp 0 --server_address 127.0.0.1:8092 --strategy FedUniV2.1 --min_num_clients 5 --img_size 384 --alpha 0.5 --beta 0.5 --prompt universal --attention dual --dual_init aggregated --label_prompt 0 --role client --cid 3 --client client4 --sup_type keypoint --gpu 6

python flower_pCE_2D_v4_auxpCE_GatedCRFLoss.py --root_path ../data/ODOC_h5 --num_classes 3 --in_chns 3 --img_class odoc --exp odoc/Prompt_Ablation_NoLabelPrompt --model unet_univ5 --max_iterations 30000 --iters 5 --eval_iters 5 --tsne_iters 200 --batch_size 12 --base_lr 0.05 --amp 0 --server_address 127.0.0.1:8092 --strategy FedUniV2.1 --min_num_clients 5 --img_size 384 --alpha 0.5 --beta 0.5 --prompt universal --attention dual --dual_init aggregated --label_prompt 0 --role client --cid 4 --client client5 --sup_type block --gpu 7


# Prompt Just DIS
python flower_pCE_2D_v4_auxpCE_GatedCRFLoss_WO_UniPrompt.py --root_path ../data/ODOC_h5 --num_classes 3 --in_chns 3 --img_class odoc --exp odoc/Prompt_Ablation_JustDist --model unet_univ5 --max_iterations 30000 --iters 5 --eval_iters 5 --tsne_iters 200 --batch_size 12 --base_lr 0.05 --amp 0 --server_address 127.0.0.1:8092 --strategy FedUniV2.1 --min_num_clients 5 --img_size 384 --alpha 0.5 --beta 0.5 --prompt universal --attention dual --dual_init aggregated --label_prompt 0 --role server --client client_all --sup_type mask --gpu 0

python flower_pCE_2D_v4_auxpCE_GatedCRFLoss_WO_UniPrompt.py --root_path ../data/ODOC_h5 --num_classes 3 --in_chns 3 --img_class odoc --exp odoc/Prompt_Ablation_JustDist --model unet_univ5 --max_iterations 30000 --iters 5 --eval_iters 5 --tsne_iters 200 --batch_size 12 --base_lr 0.05 --amp 0 --server_address 127.0.0.1:8092 --strategy FedUniV2.1 --min_num_clients 5 --img_size 384 --alpha 0.5 --beta 0.5 --prompt universal --attention dual --dual_init aggregated --label_prompt 0 --role client --cid 0 --client client1 --sup_type scribble --gpu 3

python flower_pCE_2D_v4_auxpCE_GatedCRFLoss_WO_UniPrompt.py --root_path ../data/ODOC_h5 --num_classes 3 --in_chns 3 --img_class odoc --exp odoc/Prompt_Ablation_JustDist --model unet_univ5 --max_iterations 30000 --iters 5 --eval_iters 5 --tsne_iters 200 --batch_size 12 --base_lr 0.05 --amp 0 --server_address 127.0.0.1:8092 --strategy FedUniV2.1 --min_num_clients 5 --img_size 384 --alpha 0.5 --beta 0.5 --prompt universal --attention dual --dual_init aggregated --label_prompt 0 --role client --cid 1 --client client2 --sup_type scribble_noisy --gpu 4

python flower_pCE_2D_v4_auxpCE_GatedCRFLoss_WO_UniPrompt.py --root_path ../data/ODOC_h5 --num_classes 3 --in_chns 3 --img_class odoc --exp odoc/Prompt_Ablation_JustDist --model unet_univ5 --max_iterations 30000 --iters 5 --eval_iters 5 --tsne_iters 200 --batch_size 12 --base_lr 0.05 --amp 0 --server_address 127.0.0.1:8092 --strategy FedUniV2.1 --min_num_clients 5 --img_size 384 --alpha 0.5 --beta 0.5 --prompt universal --attention dual --dual_init aggregated --label_prompt 0 --role client --cid 2 --client client3 --sup_type scribble_noisy --gpu 5

python flower_pCE_2D_v4_auxpCE_GatedCRFLoss_WO_UniPrompt.py --root_path ../data/ODOC_h5 --num_classes 3 --in_chns 3 --img_class odoc --exp odoc/Prompt_Ablation_JustDist --model unet_univ5 --max_iterations 30000 --iters 5 --eval_iters 5 --tsne_iters 200 --batch_size 12 --base_lr 0.05 --amp 0 --server_address 127.0.0.1:8092 --strategy FedUniV2.1 --min_num_clients 5 --img_size 384 --alpha 0.5 --beta 0.5 --prompt universal --attention dual --dual_init aggregated --label_prompt 0 --role client --cid 3 --client client4 --sup_type keypoint --gpu 6

python flower_pCE_2D_v4_auxpCE_GatedCRFLoss_WO_UniPrompt.py --root_path ../data/ODOC_h5 --num_classes 3 --in_chns 3 --img_class odoc --exp odoc/Prompt_Ablation_JustDist --model unet_univ5 --max_iterations 30000 --iters 5 --eval_iters 5 --tsne_iters 200 --batch_size 12 --base_lr 0.05 --amp 0 --server_address 127.0.0.1:8092 --strategy FedUniV2.1 --min_num_clients 5 --img_size 384 --alpha 0.5 --beta 0.5 --prompt universal --attention dual --dual_init aggregated --label_prompt 0 --role client --cid 4 --client client5 --sup_type block --gpu 7


# Prompt test
python flower_pCE_2D_v4_auxpCE_GatedCRFLoss_WO_UniPrompt.py --root_path ../data/ODOC_h5 --num_classes 3 --in_chns 3 --img_class odoc --exp odoc/Prompt_Ablation_JustDist_test --model unet_univ5 --max_iterations 30000 --iters 5 --eval_iters 5 --tsne_iters 200 --batch_size 12 --base_lr 0.05 --amp 0 --server_address 127.0.0.1:8092 --strategy FedUniV2.1 --min_num_clients 5 --img_size 384 --alpha 0.5 --beta 0.5 --prompt universal --attention dual --dual_init aggregated --label_prompt 0 --role server --client client_all --sup_type mask --gpu 0

python flower_pCE_2D_v4_auxpCE_GatedCRFLoss_WO_UniPrompt.py --root_path ../data/ODOC_h5 --num_classes 3 --in_chns 3 --img_class odoc --exp odoc/Prompt_Ablation_JustDist_test --model unet_univ5 --max_iterations 30000 --iters 5 --eval_iters 5 --tsne_iters 200 --batch_size 12 --base_lr 0.05 --amp 0 --server_address 127.0.0.1:8092 --strategy FedUniV2.1 --min_num_clients 5 --img_size 384 --alpha 0.5 --beta 0.5 --prompt universal --attention dual --dual_init aggregated --label_prompt 0 --role client --cid 0 --client client1 --sup_type scribble --gpu 3

python flower_pCE_2D_v4_auxpCE_GatedCRFLoss_WO_UniPrompt.py --root_path ../data/ODOC_h5 --num_classes 3 --in_chns 3 --img_class odoc --exp odoc/Prompt_Ablation_JustDist_test --model unet_univ5 --max_iterations 30000 --iters 5 --eval_iters 5 --tsne_iters 200 --batch_size 12 --base_lr 0.05 --amp 0 --server_address 127.0.0.1:8092 --strategy FedUniV2.1 --min_num_clients 5 --img_size 384 --alpha 0.5 --beta 0.5 --prompt universal --attention dual --dual_init aggregated --label_prompt 0 --role client --cid 1 --client client2 --sup_type scribble_noisy --gpu 4

python flower_pCE_2D_v4_auxpCE_GatedCRFLoss_WO_UniPrompt.py --root_path ../data/ODOC_h5 --num_classes 3 --in_chns 3 --img_class odoc --exp odoc/Prompt_Ablation_JustDist_test --model unet_univ5 --max_iterations 30000 --iters 5 --eval_iters 5 --tsne_iters 200 --batch_size 12 --base_lr 0.05 --amp 0 --server_address 127.0.0.1:8092 --strategy FedUniV2.1 --min_num_clients 5 --img_size 384 --alpha 0.5 --beta 0.5 --prompt universal --attention dual --dual_init aggregated --label_prompt 0 --role client --cid 2 --client client3 --sup_type scribble_noisy --gpu 5

python flower_pCE_2D_v4_auxpCE_GatedCRFLoss_WO_UniPrompt.py --root_path ../data/ODOC_h5 --num_classes 3 --in_chns 3 --img_class odoc --exp odoc/Prompt_Ablation_JustDist_test --model unet_univ5 --max_iterations 30000 --iters 5 --eval_iters 5 --tsne_iters 200 --batch_size 12 --base_lr 0.05 --amp 0 --server_address 127.0.0.1:8092 --strategy FedUniV2.1 --min_num_clients 5 --img_size 384 --alpha 0.5 --beta 0.5 --prompt universal --attention dual --dual_init aggregated --label_prompt 0 --role client --cid 3 --client client4 --sup_type keypoint --gpu 6

python flower_pCE_2D_v4_auxpCE_GatedCRFLoss_WO_UniPrompt.py --root_path ../data/ODOC_h5 --num_classes 3 --in_chns 3 --img_class odoc --exp odoc/Prompt_Ablation_JustDist_test --model unet_univ5 --max_iterations 30000 --iters 5 --eval_iters 5 --tsne_iters 200 --batch_size 12 --base_lr 0.05 --amp 0 --server_address 127.0.0.1:8092 --strategy FedUniV2.1 --min_num_clients 5 --img_size 384 --alpha 0.5 --beta 0.5 --prompt universal --attention dual --dual_init aggregated --label_prompt 0 --role client --cid 4 --client client5 --sup_type block --gpu 7




# 1.只有prompt+da 没有decoder2(beta=0.0)
python flower_pCE_2D_v4_auxpCE_GatedCRFLoss_MainAbalation1_noDual.py --root_path ../data/ODOC_h5 --num_classes 3 --in_chns 3 --img_class odoc --exp odoc/MainAblation1_Prompt_DA_NoDecoder2 --model unet_univ5 --max_iterations 30000 --iters 5 --eval_iters 5 --tsne_iters 200 --batch_size 12 --base_lr 0.05 --amp 0 --server_address 127.0.0.1:8091 --strategy FedUniV2.1 --min_num_clients 5 --img_size 384 --alpha 0.0 --beta 0.0 --prompt universal --attention dual --dual_init aggregated --label_prompt 1 --role server --client client_all --sup_type mask --gpu 0

python flower_pCE_2D_v4_auxpCE_GatedCRFLoss_MainAbalation1_noDual.py --root_path ../data/ODOC_h5 --num_classes 3 --in_chns 3 --img_class odoc --exp odoc/MainAblation1_Prompt_DA_NoDecoder2 --model unet_univ5 --max_iterations 30000 --iters 5 --eval_iters 5 --tsne_iters 200 --batch_size 12 --base_lr 0.05 --amp 0 --server_address 127.0.0.1:8091 --strategy FedUniV2.1 --min_num_clients 5 --img_size 384 --alpha 0.0 --beta 0.0 --prompt universal --attention dual --dual_init aggregated --label_prompt 1 --role client --cid 0 --client client1 --sup_type scribble --gpu 3

python flower_pCE_2D_v4_auxpCE_GatedCRFLoss_MainAbalation1_noDual.py --root_path ../data/ODOC_h5 --num_classes 3 --in_chns 3 --img_class odoc --exp odoc/MainAblation1_Prompt_DA_NoDecoder2 --model unet_univ5 --max_iterations 30000 --iters 5 --eval_iters 5 --tsne_iters 200 --batch_size 12 --base_lr 0.05 --amp 0 --server_address 127.0.0.1:8091 --strategy FedUniV2.1 --min_num_clients 5 --img_size 384 --alpha 0.0 --beta 0.0 --prompt universal --attention dual --dual_init aggregated --label_prompt 1 --role client --cid 1 --client client2 --sup_type scribble_noisy --gpu 4

python flower_pCE_2D_v4_auxpCE_GatedCRFLoss_MainAbalation1_noDual.py --root_path ../data/ODOC_h5 --num_classes 3 --in_chns 3 --img_class odoc --exp odoc/MainAblation1_Prompt_DA_NoDecoder2 --model unet_univ5 --max_iterations 30000 --iters 5 --eval_iters 5 --tsne_iters 200 --batch_size 12 --base_lr 0.05 --amp 0 --server_address 127.0.0.1:8091 --strategy FedUniV2.1 --min_num_clients 5 --img_size 384 --alpha 0.0 --beta 0.0 --prompt universal --attention dual --dual_init aggregated --label_prompt 1 --role client --cid 2 --client client3 --sup_type scribble_noisy --gpu 5

python flower_pCE_2D_v4_auxpCE_GatedCRFLoss_MainAbalation1_noDual.py --root_path ../data/ODOC_h5 --num_classes 3 --in_chns 3 --img_class odoc --exp odoc/MainAblation1_Prompt_DA_NoDecoder2 --model unet_univ5 --max_iterations 30000 --iters 5 --eval_iters 5 --tsne_iters 200 --batch_size 12 --base_lr 0.05 --amp 0 --server_address 127.0.0.1:8091 --strategy FedUniV2.1 --min_num_clients 5 --img_size 384 --alpha 0.0 --beta 0.0 --prompt universal --attention dual --dual_init aggregated --label_prompt 1 --role client --cid 3 --client client4 --sup_type keypoint --gpu 6

python flower_pCE_2D_v4_auxpCE_GatedCRFLoss_MainAbalation1_noDual.py --root_path ../data/ODOC_h5 --num_classes 3 --in_chns 3 --img_class odoc --exp odoc/MainAblation1_Prompt_DA_NoDecoder2 --model unet_univ5 --max_iterations 30000 --iters 5 --eval_iters 5 --tsne_iters 200 --batch_size 12 --base_lr 0.05 --amp 0 --server_address 127.0.0.1:8091 --strategy FedUniV2.1 --min_num_clients 5 --img_size 384 --alpha 0.0 --beta 0.0 --prompt universal --attention dual --dual_init aggregated --label_prompt 1 --role client --cid 4 --client client5 --sup_type block --gpu 7





python flower_pCE_2D.py --root_path ../data/Polypdata_h5 --num_classes 2 --in_chns 3 --img_class polyp --exp polyp/FL_WeaklySeg_pCE_flower_FedLC_alpha0.1 --model unet_lc --max_iterations 30000 --iters 5 --eval_iters 5 --tsne_iters 200 --batch_size 12 --base_lr 0.001 --amp 0 --server_address [::]:8094 --strategy FedLC --min_num_clients 4 --alpha 0.1 --rep_iters 2 --role server --client client_all --sup_type mask --gpu 0

python flower_pCE_2D.py --root_path ../data/Polypdata_h5 --num_classes 2 --in_chns 3 --img_class polyp --exp polyp/FL_WeaklySeg_pCE_flower_FedLC_alpha0.1 --model unet_lc --max_iterations 30000 --iters 5 --eval_iters 5 --tsne_iters 200 --batch_size 12 --base_lr 0.001 --amp 0 --server_address [::]:8094 --strategy FedLC --min_num_clients 4 --alpha 0.1 --rep_iters 2 --role client --cid 0 --client client1 --sup_type keypoint --gpu 6

python flower_pCE_2D.py --root_path ../data/Polypdata_h5 --num_classes 2 --in_chns 3 --img_class polyp --exp polyp/FL_WeaklySeg_pCE_flower_FedLC_alpha0.1 --model unet_lc --max_iterations 30000 --iters 5 --eval_iters 5 --tsne_iters 200 --batch_size 12 --base_lr 0.001 --amp 0 --server_address [::]:8094 --strategy FedLC --min_num_clients 4 --alpha 0.1 --rep_iters 2 --role client --cid 1 --client client2 --sup_type scribble --gpu 6

python flower_pCE_2D.py --root_path ../data/Polypdata_h5 --num_classes 2 --in_chns 3 --img_class polyp --exp polyp/FL_WeaklySeg_pCE_flower_FedLC_alpha0.1 --model unet_lc --max_iterations 30000 --iters 5 --eval_iters 5 --tsne_iters 200 --batch_size 12 --base_lr 0.001 --amp 0 --server_address [::]:8094 --strategy FedLC --min_num_clients 4 --alpha 0.1 --rep_iters 2 --role client --cid 2 --client client3 --sup_type box --gpu 7

python flower_pCE_2D.py --root_path ../data/Polypdata_h5 --num_classes 2 --in_chns 3 --img_class polyp --exp polyp/FL_WeaklySeg_pCE_flower_FedLC_alpha0.1 --model unet_lc --max_iterations 30000 --iters 5 --eval_iters 5 --tsne_iters 200 --batch_size 12 --base_lr 0.001 --amp 0 --server_address [::]:8094 --strategy FedLC --min_num_clients 4 --alpha 0.1 --rep_iters 2 --role client --cid 3 --client client4 --sup_type block --gpu 7



# polyp meta
python flower_runner.py --port 8091 --procedure flower_pCE_2D --exp polyp/FL_WeaklySeg_pCE_flower_pCE_MetaFed --base_lr 0.01 --img_class polyp --model unet --gpus 0 0 0 1 1  --strategy MetaFed --init_iters 100 --common_iters 400 --amp 0 --debug 1 --iters 10 --eval_iters 10

python flower_pCE_2D.py --root_path ../data/Polypdata_h5 --num_classes 2 --in_chns 3 --img_class polyp --exp polyp/FL_WeaklySeg_pCE_flower_pCE_MetaFed --model unet --max_iterations 30000 --iters 10 --eval_iters 10 --tsne_iters 200 --batch_size 12 --base_lr 0.01 --amp 0 --server_address 127.0.0.1:8091 --strategy MetaFed --min_num_clients 4 --lam 0.5 --beta 0.5 --sort_lam 0.5 --sort_beta 0.5 --init_iters 100 --common_iters 400 --role server --client client_all --sup_type mask --gpu 5

python flower_pCE_2D.py --root_path ../data/Polypdata_h5 --num_classes 2 --in_chns 3 --img_class polyp --exp polyp/FL_WeaklySeg_pCE_flower_pCE_MetaFed --model unet --max_iterations 30000 --iters 10 --eval_iters 10 --tsne_iters 200 --batch_size 12 --base_lr 0.01 --amp 0 --server_address 127.0.0.1:8091 --strategy MetaFed --min_num_clients 4 --lam 0.5 --beta 0.5 --sort_lam 0.5 --sort_beta 0.5 --init_iters 100 --common_iters 400 --role client --cid 0 --client client1 --sup_type keypoint --gpu 5

python flower_pCE_2D.py --root_path ../data/Polypdata_h5 --num_classes 2 --in_chns 3 --img_class polyp --exp polyp/FL_WeaklySeg_pCE_flower_pCE_MetaFed --model unet --max_iterations 30000 --iters 10 --eval_iters 10 --tsne_iters 200 --batch_size 12 --base_lr 0.01 --amp 0 --server_address 127.0.0.1:8091 --strategy MetaFed --min_num_clients 4 --lam 0.5 --beta 0.5 --sort_lam 0.5 --sort_beta 0.5 --init_iters 100 --common_iters 400 --role client --cid 1 --client client2 --sup_type scribble --gpu 6

python flower_pCE_2D.py --root_path ../data/Polypdata_h5 --num_classes 2 --in_chns 3 --img_class polyp --exp polyp/FL_WeaklySeg_pCE_flower_pCE_MetaFed --model unet --max_iterations 30000 --iters 10 --eval_iters 10 --tsne_iters 200 --batch_size 12 --base_lr 0.01 --amp 0 --server_address 127.0.0.1:8091 --strategy MetaFed --min_num_clients 4 --lam 0.5 --beta 0.5 --sort_lam 0.5 --sort_beta 0.5 --init_iters 100 --common_iters 400 --role client --cid 2 --client client3 --sup_type box --gpu 6

python flower_pCE_2D.py --root_path ../data/Polypdata_h5 --num_classes 2 --in_chns 3 --img_class polyp --exp polyp/FL_WeaklySeg_pCE_flower_pCE_MetaFed --model unet --max_iterations 30000 --iters 10 --eval_iters 10 --tsne_iters 200 --batch_size 12 --base_lr 0.01 --amp 0 --server_address 127.0.0.1:8091 --strategy MetaFed --min_num_clients 4 --lam 0.5 --beta 0.5 --sort_lam 0.5 --sort_beta 0.5 --init_iters 100 --common_iters 400 --role client --cid 3 --client client4 --sup_type block --gpu 7






# Faz iteration ablation
# #iteration 20
python flower_pCE_2D_v4_auxpCE_GatedCRFLoss.py --root_path ../data/FAZ_h5 --num_classes 2 --in_chns 1 --img_class faz --exp faz/rebuttal_ablation_iteation20 --model unet_univ5 --max_iterations 30000 --iters 20 --eval_iters 20 --tsne_iters 200 --batch_size 12 --base_lr 0.01 --amp 0 --server_address 172.18.5.181:8093 --strategy FedUniV2.1 --min_num_clients 5 --img_size 256 --alpha 0.0 --beta 0.5 --prompt universal --attention dual --dual_init aggregated --label_prompt 1 --role server --client client_all --sup_type mask --gpu 0

python flower_pCE_2D_v4_auxpCE_GatedCRFLoss.py --root_path ../data/FAZ_h5 --num_classes 2 --in_chns 1 --img_class faz --exp faz/rebuttal_ablation_iteation20 --model unet_univ5 --max_iterations 30000 --iters 20 --eval_iters 20 --tsne_iters 200 --batch_size 12 --base_lr 0.01 --amp 0 --server_address 172.18.5.181:8093 --strategy FedUniV2.1 --min_num_clients 5 --img_size 256 --alpha 0.0 --beta 0.5 --prompt universal --attention dual --dual_init aggregated --label_prompt 1 --role client --cid 0 --client client1 --sup_type scribble_noisy --gpu 0

python flower_pCE_2D_v4_auxpCE_GatedCRFLoss.py --root_path ../data/FAZ_h5 --num_classes 2 --in_chns 1 --img_class faz --exp faz/rebuttal_ablation_iteation20 --model unet_univ5 --max_iterations 30000 --iters 20 --eval_iters 20 --tsne_iters 200 --batch_size 12 --base_lr 0.01 --amp 0 --server_address 172.18.5.181:8093 --strategy FedUniV2.1 --min_num_clients 5 --img_size 256 --alpha 0.0 --beta 0.5 --prompt universal --attention dual --dual_init aggregated --label_prompt 1 --role client --cid 1 --client client2 --sup_type keypoint --gpu 1

python flower_pCE_2D_v4_auxpCE_GatedCRFLoss.py --root_path ../data/FAZ_h5 --num_classes 2 --in_chns 1 --img_class faz --exp faz/rebuttal_ablation_iteation20 --model unet_univ5 --max_iterations 30000 --iters 20 --eval_iters 20 --tsne_iters 200 --batch_size 12 --base_lr 0.01 --amp 0 --server_address 172.18.5.181:8093 --strategy FedUniV2.1 --min_num_clients 5 --img_size 256 --alpha 0.0 --beta 0.5 --prompt universal --attention dual --dual_init aggregated --label_prompt 1 --role client --cid 2 --client client3 --sup_type block --gpu 1

python flower_pCE_2D_v4_auxpCE_GatedCRFLoss.py --root_path ../data/FAZ_h5 --num_classes 2 --in_chns 1 --img_class faz --exp faz/rebuttal_ablation_iteation20 --model unet_univ5 --max_iterations 30000 --iters 20 --eval_iters 20 --tsne_iters 200 --batch_size 12 --base_lr 0.01 --amp 0 --server_address 172.18.5.181:8093 --strategy FedUniV2.1 --min_num_clients 5 --img_size 256 --alpha 0.0 --beta 0.5 --prompt universal --attention dual --dual_init aggregated --label_prompt 1 --role client --cid 3 --client client4 --sup_type box --gpu 2

python flower_pCE_2D_v4_auxpCE_GatedCRFLoss.py --root_path ../data/FAZ_h5 --num_classes 2 --in_chns 1 --img_class faz --exp faz/rebuttal_ablation_iteation20 --model unet_univ5 --max_iterations 30000 --iters 20 --eval_iters 20 --tsne_iters 200 --batch_size 12 --base_lr 0.01 --amp 0 --server_address 172.18.5.181:8093 --strategy FedUniV2.1 --min_num_clients 5 --img_size 256 --alpha 0.0 --beta 0.5 --prompt universal --attention dual --dual_init aggregated --label_prompt 1 --role client --cid 4 --client client5 --sup_type scribble --gpu 2




# Faz iteration ablation
# #iteration 50
python flower_pCE_2D_v4_auxpCE_GatedCRFLoss.py --root_path ../data/FAZ_h5 --num_classes 2 --in_chns 1 --img_class faz --exp faz/rebuttal_ablation_iteation50 --model unet_univ5 --max_iterations 30000 --iters 50 --eval_iters 50 --tsne_iters 200 --batch_size 12 --base_lr 0.01 --amp 0 --server_address 172.18.5.181:8094 --strategy FedUniV2.1 --min_num_clients 5 --img_size 256 --alpha 0.0 --beta 0.5 --prompt universal --attention dual --dual_init aggregated --label_prompt 1 --role server --client client_all --sup_type mask --gpu 0

python flower_pCE_2D_v4_auxpCE_GatedCRFLoss.py --root_path ../data/FAZ_h5 --num_classes 2 --in_chns 1 --img_class faz --exp faz/rebuttal_ablation_iteation50 --model unet_univ5 --max_iterations 30000 --iters 50 --eval_iters 50 --tsne_iters 200 --batch_size 12 --base_lr 0.01 --amp 0 --server_address 172.18.5.181:8094 --strategy FedUniV2.1 --min_num_clients 5 --img_size 256 --alpha 0.0 --beta 0.5 --prompt universal --attention dual --dual_init aggregated --label_prompt 1 --role client --cid 0 --client client1 --sup_type scribble_noisy --gpu 7

python flower_pCE_2D_v4_auxpCE_GatedCRFLoss.py --root_path ../data/FAZ_h5 --num_classes 2 --in_chns 1 --img_class faz --exp faz/rebuttal_ablation_iteation50 --model unet_univ5 --max_iterations 30000 --iters 50 --eval_iters 50 --tsne_iters 200 --batch_size 12 --base_lr 0.01 --amp 0 --server_address 172.18.5.181:8094 --strategy FedUniV2.1 --min_num_clients 5 --img_size 256 --alpha 0.0 --beta 0.5 --prompt universal --attention dual --dual_init aggregated --label_prompt 1 --role client --cid 1 --client client2 --sup_type keypoint --gpu 7

python flower_pCE_2D_v4_auxpCE_GatedCRFLoss.py --root_path ../data/FAZ_h5 --num_classes 2 --in_chns 1 --img_class faz --exp faz/rebuttal_ablation_iteation50 --model unet_univ5 --max_iterations 30000 --iters 50 --eval_iters 50 --tsne_iters 200 --batch_size 12 --base_lr 0.01 --amp 0 --server_address 172.18.5.181:8094 --strategy FedUniV2.1 --min_num_clients 5 --img_size 256 --alpha 0.0 --beta 0.5 --prompt universal --attention dual --dual_init aggregated --label_prompt 1 --role client --cid 2 --client client3 --sup_type block --gpu 6

python flower_pCE_2D_v4_auxpCE_GatedCRFLoss.py --root_path ../data/FAZ_h5 --num_classes 2 --in_chns 1 --img_class faz --exp faz/rebuttal_ablation_iteation50 --model unet_univ5 --max_iterations 30000 --iters 50 --eval_iters 50 --tsne_iters 200 --batch_size 12 --base_lr 0.01 --amp 0 --server_address 172.18.5.181:8094 --strategy FedUniV2.1 --min_num_clients 5 --img_size 256 --alpha 0.0 --beta 0.5 --prompt universal --attention dual --dual_init aggregated --label_prompt 1 --role client --cid 3 --client client4 --sup_type box --gpu 6

python flower_pCE_2D_v4_auxpCE_GatedCRFLoss.py --root_path ../data/FAZ_h5 --num_classes 2 --in_chns 1 --img_class faz --exp faz/rebuttal_ablation_iteation50 --model unet_univ5 --max_iterations 30000 --iters 50 --eval_iters 50 --tsne_iters 200 --batch_size 12 --base_lr 0.01 --amp 0 --server_address 172.18.5.181:8094 --strategy FedUniV2.1 --min_num_clients 5 --img_size 256 --alpha 0.0 --beta 0.5 --prompt universal --attention dual --dual_init aggregated --label_prompt 1 --role client --cid 4 --client client5 --sup_type scribble --gpu 5




# Faz iteration ablation
# #iteration 100
python flower_pCE_2D_v4_auxpCE_GatedCRFLoss.py --root_path ../data/FAZ_h5 --num_classes 2 --in_chns 1 --img_class faz --exp faz/rebuttal_ablation_iteation100 --model unet_univ5 --max_iterations 30000 --iters 100 --eval_iters 100 --tsne_iters 200 --batch_size 12 --base_lr 0.01 --amp 0 --server_address 172.18.5.181:8095 --strategy FedUniV2.1 --min_num_clients 5 --img_size 256 --alpha 0.0 --beta 0.5 --prompt universal --attention dual --dual_init aggregated --label_prompt 1 --role server --client client_all --sup_type mask --gpu 0

python flower_pCE_2D_v4_auxpCE_GatedCRFLoss.py --root_path ../data/FAZ_h5 --num_classes 2 --in_chns 1 --img_class faz --exp faz/rebuttal_ablation_iteation100 --model unet_univ5 --max_iterations 30000 --iters 100 --eval_iters 100 --tsne_iters 200 --batch_size 12 --base_lr 0.01 --amp 0 --server_address 172.18.5.181:8095 --strategy FedUniV2.1 --min_num_clients 5 --img_size 256 --alpha 0.0 --beta 0.5 --prompt universal --attention dual --dual_init aggregated --label_prompt 1 --role client --cid 0 --client client1 --sup_type scribble_noisy --gpu 5

python flower_pCE_2D_v4_auxpCE_GatedCRFLoss.py --root_path ../data/FAZ_h5 --num_classes 2 --in_chns 1 --img_class faz --exp faz/rebuttal_ablation_iteation100 --model unet_univ5 --max_iterations 30000 --iters 100 --eval_iters 100 --tsne_iters 200 --batch_size 12 --base_lr 0.01 --amp 0 --server_address 172.18.5.181:8095 --strategy FedUniV2.1 --min_num_clients 5 --img_size 256 --alpha 0.0 --beta 0.5 --prompt universal --attention dual --dual_init aggregated --label_prompt 1 --role client --cid 1 --client client2 --sup_type keypoint --gpu 6

python flower_pCE_2D_v4_auxpCE_GatedCRFLoss.py --root_path ../data/FAZ_h5 --num_classes 2 --in_chns 1 --img_class faz --exp faz/rebuttal_ablation_iteation100 --model unet_univ5 --max_iterations 30000 --iters 100 --eval_iters 100 --tsne_iters 200 --batch_size 12 --base_lr 0.01 --amp 0 --server_address 172.18.5.181:8095 --strategy FedUniV2.1 --min_num_clients 5 --img_size 256 --alpha 0.0 --beta 0.5 --prompt universal --attention dual --dual_init aggregated --label_prompt 1 --role client --cid 2 --client client3 --sup_type block --gpu 6

python flower_pCE_2D_v4_auxpCE_GatedCRFLoss.py --root_path ../data/FAZ_h5 --num_classes 2 --in_chns 1 --img_class faz --exp faz/rebuttal_ablation_iteation100 --model unet_univ5 --max_iterations 30000 --iters 100 --eval_iters 100 --tsne_iters 200 --batch_size 12 --base_lr 0.01 --amp 0 --server_address 172.18.5.181:8095 --strategy FedUniV2.1 --min_num_clients 5 --img_size 256 --alpha 0.0 --beta 0.5 --prompt universal --attention dual --dual_init aggregated --label_prompt 1 --role client --cid 3 --client client4 --sup_type box --gpu 7

python flower_pCE_2D_v4_auxpCE_GatedCRFLoss.py --root_path ../data/FAZ_h5 --num_classes 2 --in_chns 1 --img_class faz --exp faz/rebuttal_ablation_iteation100 --model unet_univ5 --max_iterations 30000 --iters 100 --eval_iters 100 --tsne_iters 200 --batch_size 12 --base_lr 0.01 --amp 0 --server_address 172.18.5.181:8095 --strategy FedUniV2.1 --min_num_clients 5 --img_size 256 --alpha 0.0 --beta 0.5 --prompt universal --attention dual --dual_init aggregated --label_prompt 1 --role client --cid 4 --client client5 --sup_type scribble --gpu 7





# #iteration 200
python flower_pCE_2D_v4_auxpCE_GatedCRFLoss.py --root_path ../data/FAZ_h5 --num_classes 2 --in_chns 1 --img_class faz --exp faz/rebuttal_ablation_iteation200 --model unet_univ5 --max_iterations 30000 --iters 200 --eval_iters 200 --tsne_iters 200 --batch_size 12 --base_lr 0.01 --amp 0 --server_address 172.18.5.181:8095 --strategy FedUniV2.1 --min_num_clients 5 --img_size 256 --alpha 0.0 --beta 0.5 --prompt universal --attention dual --dual_init aggregated --label_prompt 1 --role server --client client_all --sup_type mask --gpu 0

python flower_pCE_2D_v4_auxpCE_GatedCRFLoss.py --root_path ../data/FAZ_h5 --num_classes 2 --in_chns 1 --img_class faz --exp faz/rebuttal_ablation_iteation200 --model unet_univ5 --max_iterations 30000 --iters 200 --eval_iters 200 --tsne_iters 200 --batch_size 12 --base_lr 0.01 --amp 0 --server_address 172.18.5.181:8095 --strategy FedUniV2.1 --min_num_clients 5 --img_size 256 --alpha 0.0 --beta 0.5 --prompt universal --attention dual --dual_init aggregated --label_prompt 1 --role client --cid 0 --client client1 --sup_type scribble_noisy --gpu 5

python flower_pCE_2D_v4_auxpCE_GatedCRFLoss.py --root_path ../data/FAZ_h5 --num_classes 2 --in_chns 1 --img_class faz --exp faz/rebuttal_ablation_iteation200 --model unet_univ5 --max_iterations 30000 --iters 200 --eval_iters 200 --tsne_iters 200 --batch_size 12 --base_lr 0.01 --amp 0 --server_address 172.18.5.181:8095 --strategy FedUniV2.1 --min_num_clients 5 --img_size 256 --alpha 0.0 --beta 0.5 --prompt universal --attention dual --dual_init aggregated --label_prompt 1 --role client --cid 1 --client client2 --sup_type keypoint --gpu 1

python flower_pCE_2D_v4_auxpCE_GatedCRFLoss.py --root_path ../data/FAZ_h5 --num_classes 2 --in_chns 1 --img_class faz --exp faz/rebuttal_ablation_iteation200 --model unet_univ5 --max_iterations 30000 --iters 200 --eval_iters 200 --tsne_iters 200 --batch_size 12 --base_lr 0.01 --amp 0 --server_address 172.18.5.181:8095 --strategy FedUniV2.1 --min_num_clients 5 --img_size 256 --alpha 0.0 --beta 0.5 --prompt universal --attention dual --dual_init aggregated --label_prompt 1 --role client --cid 2 --client client3 --sup_type block --gpu 1

python flower_pCE_2D_v4_auxpCE_GatedCRFLoss.py --root_path ../data/FAZ_h5 --num_classes 2 --in_chns 1 --img_class faz --exp faz/rebuttal_ablation_iteation200 --model unet_univ5 --max_iterations 30000 --iters 200 --eval_iters 200 --tsne_iters 200 --batch_size 12 --base_lr 0.01 --amp 0 --server_address 172.18.5.181:8095 --strategy FedUniV2.1 --min_num_clients 5 --img_size 256 --alpha 0.0 --beta 0.5 --prompt universal --attention dual --dual_init aggregated --label_prompt 1 --role client --cid 3 --client client4 --sup_type box --gpu 2

python flower_pCE_2D_v4_auxpCE_GatedCRFLoss.py --root_path ../data/FAZ_h5 --num_classes 2 --in_chns 1 --img_class faz --exp faz/rebuttal_ablation_iteation200 --model unet_univ5 --max_iterations 30000 --iters 200 --eval_iters 200 --tsne_iters 200 --batch_size 12 --base_lr 0.01 --amp 0 --server_address 172.18.5.181:8095 --strategy FedUniV2.1 --min_num_clients 5 --img_size 256 --alpha 0.0 --beta 0.5 --prompt universal --attention dual --dual_init aggregated --label_prompt 1 --role client --cid 4 --client client5 --sup_type scribble --gpu 2





# polyp MLP uni+dis Ours attention_concat
python flower_pCE_2D_v4_auxpCE_GatedCRFLoss_Rebuttal_AblationConcat.py --root_path ../data/Polypdata_h5 --num_classes 2 --in_chns 3 --img_class polyp --exp polyp/rebuttal_ablation_dual_attention_sab_concat_cab_ALAhasMLP --model unet_univ5_attention_concat --max_iterations 30000 --iters 5 --eval_iters 5 --tsne_iters 200 --batch_size 12 --base_lr 0.001 --amp 0 --server_address 127.0.0.1:8093 --strategy FedUniV2.1 --min_num_clients 4 --img_size 384 --alpha 0.5 --beta 0.1 --prompt universal --attention dual_concat --dual_init aggregated --label_prompt 1 --role server --client client_all --sup_type mask --gpu 0

python flower_pCE_2D_v4_auxpCE_GatedCRFLoss_Rebuttal_AblationConcat.py --root_path ../data/Polypdata_h5 --num_classes 2 --in_chns 3 --img_class polyp --exp polyp/rebuttal_ablation_dual_attention_sab_concat_cab_ALAhasMLP --model unet_univ5_attention_concat --max_iterations 30000 --iters 5 --eval_iters 5 --tsne_iters 200 --batch_size 12 --base_lr 0.001 --amp 0 --server_address 127.0.0.1:8093 --strategy FedUniV2.1 --min_num_clients 4 --img_size 384 --alpha 0.5 --beta 0.1 --prompt universal --attention dual_concat --dual_init aggregated --label_prompt 1 --role client --cid 0 --client client1 --sup_type keypoint --gpu 3

python flower_pCE_2D_v4_auxpCE_GatedCRFLoss_Rebuttal_AblationConcat.py --root_path ../data/Polypdata_h5 --num_classes 2 --in_chns 3 --img_class polyp --exp polyp/rebuttal_ablation_dual_attention_sab_concat_cab_ALAhasMLP --model unet_univ5_attention_concat --max_iterations 30000 --iters 5 --eval_iters 5 --tsne_iters 200 --batch_size 12 --base_lr 0.001 --amp 0 --server_address 127.0.0.1:8093 --strategy FedUniV2.1 --min_num_clients 4 --img_size 384 --alpha 0.5 --beta 0.1 --prompt universal --attention dual_concat --dual_init aggregated --label_prompt 1 --role client --cid 1 --client client2 --sup_type scribble --gpu 4

python flower_pCE_2D_v4_auxpCE_GatedCRFLoss_Rebuttal_AblationConcat.py --root_path ../data/Polypdata_h5 --num_classes 2 --in_chns 3 --img_class polyp --exp polyp/rebuttal_ablation_dual_attention_sab_concat_cab_ALAhasMLP --model unet_univ5_attention_concat --max_iterations 30000 --iters 5 --eval_iters 5 --tsne_iters 200 --batch_size 12 --base_lr 0.001 --amp 0 --server_address 127.0.0.1:8093 --strategy FedUniV2.1 --min_num_clients 4 --img_size 384 --alpha 0.5 --beta 0.1 --prompt universal --attention dual_concat --dual_init aggregated --label_prompt 1 --role client --cid 2 --client client3 --sup_type box --gpu 6

python flower_pCE_2D_v4_auxpCE_GatedCRFLoss_Rebuttal_AblationConcat.py --root_path ../data/Polypdata_h5 --num_classes 2 --in_chns 3 --img_class polyp --exp polyp/rebuttal_ablation_dual_attention_sab_concat_cab_ALAhasMLP --model unet_univ5_attention_concat --max_iterations 30000 --iters 5 --eval_iters 5 --tsne_iters 200 --batch_size 12 --base_lr 0.001 --amp 0 --server_address 127.0.0.1:8093 --strategy FedUniV2.1 --min_num_clients 4 --img_size 384 --alpha 0.5 --beta 0.1 --prompt universal --attention dual_concat --dual_init aggregated --label_prompt 1 --role client --cid 3 --client client4 --sup_type block --gpu 7




# iteration
# polyp MLP uni+dis Ours
python flower_pCE_2D_v4_auxpCE_GatedCRFLoss.py --root_path ../data/Polypdata_h5 --num_classes 2 --in_chns 3 --img_class polyp --exp polyp/WeaklySeg_pCE_2D_v4_auxpCE_detach_GatedCRFLoss_MLP_DisUni --model unet_univ5 --max_iterations 30000 --iters 5 --eval_iters 5 --tsne_iters 200 --batch_size 12 --base_lr 0.001 --amp 0 --server_address 127.0.0.1:8093 --strategy FedUniV2.1 --min_num_clients 4 --img_size 384 --alpha 0.5 --beta 0.1 --prompt universal --attention dual --dual_init aggregated --label_prompt 1 --role server --client client_all --sup_type mask --gpu 1

python flower_pCE_2D_v4_auxpCE_GatedCRFLoss.py --root_path ../data/Polypdata_h5 --num_classes 2 --in_chns 3 --img_class polyp --exp polyp/WeaklySeg_pCE_2D_v4_auxpCE_detach_GatedCRFLoss_MLP_DisUni --model unet_univ5 --max_iterations 30000 --iters 5 --eval_iters 5 --tsne_iters 200 --batch_size 12 --base_lr 0.001 --amp 0 --server_address 127.0.0.1:8093 --strategy FedUniV2.1 --min_num_clients 4 --img_size 384 --alpha 0.5 --beta 0.1 --prompt universal --attention dual --dual_init aggregated --label_prompt 1 --role client --cid 0 --client client1 --sup_type keypoint --gpu 2

python flower_pCE_2D_v4_auxpCE_GatedCRFLoss.py --root_path ../data/Polypdata_h5 --num_classes 2 --in_chns 3 --img_class polyp --exp polyp/WeaklySeg_pCE_2D_v4_auxpCE_detach_GatedCRFLoss_MLP_DisUni --model unet_univ5 --max_iterations 30000 --iters 5 --eval_iters 5 --tsne_iters 200 --batch_size 12 --base_lr 0.001 --amp 0 --server_address 127.0.0.1:8093 --strategy FedUniV2.1 --min_num_clients 4 --img_size 384 --alpha 0.5 --beta 0.1 --prompt universal --attention dual --dual_init aggregated --label_prompt 1 --role client --cid 1 --client client2 --sup_type scribble --gpu 3

python flower_pCE_2D_v4_auxpCE_GatedCRFLoss.py --root_path ../data/Polypdata_h5 --num_classes 2 --in_chns 3 --img_class polyp --exp polyp/WeaklySeg_pCE_2D_v4_auxpCE_detach_GatedCRFLoss_MLP_DisUni --model unet_univ5 --max_iterations 30000 --iters 5 --eval_iters 5 --tsne_iters 200 --batch_size 12 --base_lr 0.001 --amp 0 --server_address 127.0.0.1:8093 --strategy FedUniV2.1 --min_num_clients 4 --img_size 384 --alpha 0.5 --beta 0.1 --prompt universal --attention dual --dual_init aggregated --label_prompt 1 --role client --cid 2 --client client3 --sup_type box --gpu 4

python flower_pCE_2D_v4_auxpCE_GatedCRFLoss.py --root_path ../data/Polypdata_h5 --num_classes 2 --in_chns 3 --img_class polyp --exp polyp/WeaklySeg_pCE_2D_v4_auxpCE_detach_GatedCRFLoss_MLP_DisUni --model unet_univ5 --max_iterations 30000 --iters 5 --eval_iters 5 --tsne_iters 200 --batch_size 12 --base_lr 0.001 --amp 0 --server_address 127.0.0.1:8093 --strategy FedUniV2.1 --min_num_clients 4 --img_size 384 --alpha 0.5 --beta 0.1 --prompt universal --attention dual --dual_init aggregated --label_prompt 1 --role client --cid 3 --client client4 --sup_type block --gpu 5



# Polyp
# iteration
# polyp MLP uni+dis iteration20 
python flower_pCE_2D_v4_auxpCE_GatedCRFLoss.py --root_path ../data/Polypdata_h5 --num_classes 2 --in_chns 3 --img_class polyp --exp polyp/Rebuttal_Ablation_iter20 --model unet_univ5 --max_iterations 30000 --iters 20 --eval_iters 20 --tsne_iters 200 --batch_size 12 --base_lr 0.001 --amp 0 --server_address 127.0.0.1:8093 --strategy FedUniV2.1 --min_num_clients 4 --img_size 384 --alpha 0.5 --beta 0.1 --prompt universal --attention dual --dual_init aggregated --label_prompt 1 --role server --client client_all --sup_type mask --gpu 0

python flower_pCE_2D_v4_auxpCE_GatedCRFLoss.py --root_path ../data/Polypdata_h5 --num_classes 2 --in_chns 3 --img_class polyp --exp polyp/Rebuttal_Ablation_iter20 --model unet_univ5 --max_iterations 30000 --iters 20 --eval_iters 20 --tsne_iters 200 --batch_size 12 --base_lr 0.001 --amp 0 --server_address 127.0.0.1:8093 --strategy FedUniV2.1 --min_num_clients 4 --img_size 384 --alpha 0.5 --beta 0.1 --prompt universal --attention dual --dual_init aggregated --label_prompt 1 --role client --cid 0 --client client1 --sup_type keypoint --gpu 3

python flower_pCE_2D_v4_auxpCE_GatedCRFLoss.py --root_path ../data/Polypdata_h5 --num_classes 2 --in_chns 3 --img_class polyp --exp polyp/Rebuttal_Ablation_iter20 --model unet_univ5 --max_iterations 30000 --iters 20 --eval_iters 20 --tsne_iters 200 --batch_size 12 --base_lr 0.001 --amp 0 --server_address 127.0.0.1:8093 --strategy FedUniV2.1 --min_num_clients 4 --img_size 384 --alpha 0.5 --beta 0.1 --prompt universal --attention dual --dual_init aggregated --label_prompt 1 --role client --cid 1 --client client2 --sup_type scribble --gpu 4

python flower_pCE_2D_v4_auxpCE_GatedCRFLoss.py --root_path ../data/Polypdata_h5 --num_classes 2 --in_chns 3 --img_class polyp --exp polyp/Rebuttal_Ablation_iter20 --model unet_univ5 --max_iterations 30000 --iters 20 --eval_iters 20 --tsne_iters 200 --batch_size 12 --base_lr 0.001 --amp 0 --server_address 127.0.0.1:8093 --strategy FedUniV2.1 --min_num_clients 4 --img_size 384 --alpha 0.5 --beta 0.1 --prompt universal --attention dual --dual_init aggregated --label_prompt 1 --role client --cid 2 --client client3 --sup_type box --gpu 6

python flower_pCE_2D_v4_auxpCE_GatedCRFLoss.py --root_path ../data/Polypdata_h5 --num_classes 2 --in_chns 3 --img_class polyp --exp polyp/Rebuttal_Ablation_iter20 --model unet_univ5 --max_iterations 30000 --iters 20 --eval_iters 20 --tsne_iters 200 --batch_size 12 --base_lr 0.001 --amp 0 --server_address 127.0.0.1:8093 --strategy FedUniV2.1 --min_num_clients 4 --img_size 384 --alpha 0.5 --beta 0.1 --prompt universal --attention dual --dual_init aggregated --label_prompt 1 --role client --cid 3 --client client4 --sup_type block --gpu 7


# AuxiDecoder faz
## polyp MLP uni+dis Ours
python flower_pCE_2D_v4_auxpCE_GatedCRFLoss_Rebuttal_AuxiDecoder.py --root_path ../data/Polypdata_h5 --num_classes 2 --in_chns 3 --img_class polyp --exp polyp/Rebuttal_AuxiDecoder --model unet_univ5 --max_iterations 30000 --iters 5 --eval_iters 5 --tsne_iters 200 --batch_size 12 --base_lr 0.001 --amp 0 --server_address 127.0.0.1:8093 --strategy FedUniV2.1 --min_num_clients 4 --img_size 384 --alpha 0.5 --beta 0.1 --prompt universal --attention dual --dual_init aggregated --label_prompt 1 --role server --client client_all --sup_type mask --gpu 0

python flower_pCE_2D_v4_auxpCE_GatedCRFLoss_Rebuttal_AuxiDecoder.py --root_path ../data/Polypdata_h5 --num_classes 2 --in_chns 3 --img_class polyp --exp polyp/Rebuttal_AuxiDecoder --model unet_univ5 --max_iterations 30000 --iters 5 --eval_iters 5 --tsne_iters 200 --batch_size 12 --base_lr 0.001 --amp 0 --server_address 127.0.0.1:8093 --strategy FedUniV2.1 --min_num_clients 4 --img_size 384 --alpha 0.5 --beta 0.1 --prompt universal --attention dual --dual_init aggregated --label_prompt 1 --role client --cid 0 --client client1 --sup_type keypoint --gpu 3

python flower_pCE_2D_v4_auxpCE_GatedCRFLoss_Rebuttal_AuxiDecoder.py --root_path ../data/Polypdata_h5 --num_classes 2 --in_chns 3 --img_class polyp --exp polyp/Rebuttal_AuxiDecoder --model unet_univ5 --max_iterations 30000 --iters 5 --eval_iters 5 --tsne_iters 200 --batch_size 12 --base_lr 0.001 --amp 0 --server_address 127.0.0.1:8093 --strategy FedUniV2.1 --min_num_clients 4 --img_size 384 --alpha 0.5 --beta 0.1 --prompt universal --attention dual --dual_init aggregated --label_prompt 1 --role client --cid 1 --client client2 --sup_type scribble --gpu 4

python flower_pCE_2D_v4_auxpCE_GatedCRFLoss_Rebuttal_AuxiDecoder.py --root_path ../data/Polypdata_h5 --num_classes 2 --in_chns 3 --img_class polyp --exp polyp/Rebuttal_AuxiDecoder --model unet_univ5 --max_iterations 30000 --iters 5 --eval_iters 5 --tsne_iters 200 --batch_size 12 --base_lr 0.001 --amp 0 --server_address 127.0.0.1:8093 --strategy FedUniV2.1 --min_num_clients 4 --img_size 384 --alpha 0.5 --beta 0.1 --prompt universal --attention dual --dual_init aggregated --label_prompt 1 --role client --cid 2 --client client3 --sup_type box --gpu 6

python flower_pCE_2D_v4_auxpCE_GatedCRFLoss_Rebuttal_AuxiDecoder.py --root_path ../data/Polypdata_h5 --num_classes 2 --in_chns 3 --img_class polyp --exp polyp/Rebuttal_AuxiDecoder --model unet_univ5 --max_iterations 30000 --iters 5 --eval_iters 5 --tsne_iters 200 --batch_size 12 --base_lr 0.001 --amp 0 --server_address 127.0.0.1:8093 --strategy FedUniV2.1 --min_num_clients 4 --img_size 384 --alpha 0.5 --beta 0.1 --prompt universal --attention dual --dual_init aggregated --label_prompt 1 --role client --cid 3 --client client4 --sup_type block --gpu 7




# #Rebuttal AuxiDecoder faz
python  flower_pCE_2D_v4_auxpCE_GatedCRFLoss_Rebuttal_AuxiDecoder.py --root_path ../data/FAZ_h5 --num_classes 2 --in_chns 1 --img_class faz --exp faz/Rebuttal_AuxiDecoder --model unet_univ5 --max_iterations 30000 --iters 5 --eval_iters 5 --tsne_iters 200 --batch_size 12 --base_lr 0.01 --amp 0 --server_address 172.18.5.181:8095 --strategy FedUniV2.1 --min_num_clients 5 --img_size 256 --alpha 0.0 --beta 0.5 --prompt universal --attention dual --dual_init aggregated --label_prompt 1 --role server --client client_all --sup_type mask --gpu 0

python  flower_pCE_2D_v4_auxpCE_GatedCRFLoss_Rebuttal_AuxiDecoder.py --root_path ../data/FAZ_h5 --num_classes 2 --in_chns 1 --img_class faz --exp faz/Rebuttal_AuxiDecoder --model unet_univ5 --max_iterations 30000 --iters 5 --eval_iters 5 --tsne_iters 200 --batch_size 12 --base_lr 0.01 --amp 0 --server_address 172.18.5.181:8095 --strategy FedUniV2.1 --min_num_clients 5 --img_size 256 --alpha 0.0 --beta 0.5 --prompt universal --attention dual --dual_init aggregated --label_prompt 1 --role client --cid 0 --client client1 --sup_type scribble_noisy --gpu 1

python  flower_pCE_2D_v4_auxpCE_GatedCRFLoss_Rebuttal_AuxiDecoder.py --root_path ../data/FAZ_h5 --num_classes 2 --in_chns 1 --img_class faz --exp faz/Rebuttal_AuxiDecoder --model unet_univ5 --max_iterations 30000 --iters 5 --eval_iters 5 --tsne_iters 200 --batch_size 12 --base_lr 0.01 --amp 0 --server_address 172.18.5.181:8095 --strategy FedUniV2.1 --min_num_clients 5 --img_size 256 --alpha 0.0 --beta 0.5 --prompt universal --attention dual --dual_init aggregated --label_prompt 1 --role client --cid 1 --client client2 --sup_type keypoint --gpu 1

python  flower_pCE_2D_v4_auxpCE_GatedCRFLoss_Rebuttal_AuxiDecoder.py --root_path ../data/FAZ_h5 --num_classes 2 --in_chns 1 --img_class faz --exp faz/Rebuttal_AuxiDecoder --model unet_univ5 --max_iterations 30000 --iters 5 --eval_iters 5 --tsne_iters 200 --batch_size 12 --base_lr 0.01 --amp 0 --server_address 172.18.5.181:8095 --strategy FedUniV2.1 --min_num_clients 5 --img_size 256 --alpha 0.0 --beta 0.5 --prompt universal --attention dual --dual_init aggregated --label_prompt 1 --role client --cid 2 --client client3 --sup_type block --gpu 2

python  flower_pCE_2D_v4_auxpCE_GatedCRFLoss_Rebuttal_AuxiDecoder.py --root_path ../data/FAZ_h5 --num_classes 2 --in_chns 1 --img_class faz --exp faz/Rebuttal_AuxiDecoder --model unet_univ5 --max_iterations 30000 --iters 5 --eval_iters 5 --tsne_iters 200 --batch_size 12 --base_lr 0.01 --amp 0 --server_address 172.18.5.181:8095 --strategy FedUniV2.1 --min_num_clients 5 --img_size 256 --alpha 0.0 --beta 0.5 --prompt universal --attention dual --dual_init aggregated --label_prompt 1 --role client --cid 3 --client client4 --sup_type box --gpu 2

python  flower_pCE_2D_v4_auxpCE_GatedCRFLoss_Rebuttal_AuxiDecoder.py --root_path ../data/FAZ_h5 --num_classes 2 --in_chns 1 --img_class faz --exp faz/Rebuttal_AuxiDecoder --model unet_univ5 --max_iterations 30000 --iters 5 --eval_iters 5 --tsne_iters 200 --batch_size 12 --base_lr 0.01 --amp 0 --server_address 172.18.5.181:8095 --strategy FedUniV2.1 --min_num_clients 5 --img_size 256 --alpha 0.0 --beta 0.5 --prompt universal --attention dual --dual_init aggregated --label_prompt 1 --role client --cid 4 --client client5 --sup_type scribble --gpu 0


# AuxiDecoder polyp no diceloss
python flower_pCE_2D_v4_auxpCE_GatedCRFLoss_Rebuttal_AuxiDecoder_NoDiceLoss.py --root_path ../data/Polypdata_h5 --num_classes 2 --in_chns 3 --img_class polyp --exp polyp/Rebuttal_AuxiDecoder_NoDiceLoss --model unet_univ5 --max_iterations 30000 --iters 5 --eval_iters 5 --tsne_iters 200 --batch_size 12 --base_lr 0.001 --amp 0 --server_address 127.0.0.1:8093 --strategy FedUniV2.1 --min_num_clients 4 --img_size 384 --alpha 0.5 --beta 0.1 --prompt universal --attention dual --dual_init aggregated --label_prompt 1 --role server --client client_all --sup_type mask --gpu 0

python flower_pCE_2D_v4_auxpCE_GatedCRFLoss_Rebuttal_AuxiDecoder_NoDiceLoss.py --root_path ../data/Polypdata_h5 --num_classes 2 --in_chns 3 --img_class polyp --exp polyp/Rebuttal_AuxiDecoder_NoDiceLoss --model unet_univ5 --max_iterations 30000 --iters 5 --eval_iters 5 --tsne_iters 200 --batch_size 12 --base_lr 0.001 --amp 0 --server_address 127.0.0.1:8093 --strategy FedUniV2.1 --min_num_clients 4 --img_size 384 --alpha 0.5 --beta 0.1 --prompt universal --attention dual --dual_init aggregated --label_prompt 1 --role client --cid 0 --client client1 --sup_type keypoint --gpu 3

python flower_pCE_2D_v4_auxpCE_GatedCRFLoss_Rebuttal_AuxiDecoder_NoDiceLoss.py --root_path ../data/Polypdata_h5 --num_classes 2 --in_chns 3 --img_class polyp --exp polyp/Rebuttal_AuxiDecoder_NoDiceLoss --model unet_univ5 --max_iterations 30000 --iters 5 --eval_iters 5 --tsne_iters 200 --batch_size 12 --base_lr 0.001 --amp 0 --server_address 127.0.0.1:8093 --strategy FedUniV2.1 --min_num_clients 4 --img_size 384 --alpha 0.5 --beta 0.1 --prompt universal --attention dual --dual_init aggregated --label_prompt 1 --role client --cid 1 --client client2 --sup_type scribble --gpu 4

python flower_pCE_2D_v4_auxpCE_GatedCRFLoss_Rebuttal_AuxiDecoder_NoDiceLoss.py --root_path ../data/Polypdata_h5 --num_classes 2 --in_chns 3 --img_class polyp --exp polyp/Rebuttal_AuxiDecoder_NoDiceLoss --model unet_univ5 --max_iterations 30000 --iters 5 --eval_iters 5 --tsne_iters 200 --batch_size 12 --base_lr 0.001 --amp 0 --server_address 127.0.0.1:8093 --strategy FedUniV2.1 --min_num_clients 4 --img_size 384 --alpha 0.5 --beta 0.1 --prompt universal --attention dual --dual_init aggregated --label_prompt 1 --role client --cid 2 --client client3 --sup_type box --gpu 6

python flower_pCE_2D_v4_auxpCE_GatedCRFLoss_Rebuttal_AuxiDecoder_NoDiceLoss.py --root_path ../data/Polypdata_h5 --num_classes 2 --in_chns 3 --img_class polyp --exp polyp/Rebuttal_AuxiDecoder_NoDiceLoss --model unet_univ5 --max_iterations 30000 --iters 5 --eval_iters 5 --tsne_iters 200 --batch_size 12 --base_lr 0.001 --amp 0 --server_address 127.0.0.1:8093 --strategy FedUniV2.1 --min_num_clients 4 --img_size 384 --alpha 0.5 --beta 0.1 --prompt universal --attention dual --dual_init aggregated --label_prompt 1 --role client --cid 3 --client client4 --sup_type block --gpu 7




# #AuxiDecoder faz no diceloss
python  flower_pCE_2D_v4_auxpCE_GatedCRFLoss_Rebuttal_AuxiDecoder_NoDiceLoss.py --root_path ../data/FAZ_h5 --num_classes 2 --in_chns 1 --img_class faz --exp faz/Rebuttal_AuxiDecoder_NoDiceLoss --model unet_univ5 --max_iterations 30000 --iters 5 --eval_iters 5 --tsne_iters 200 --batch_size 12 --base_lr 0.01 --amp 0 --server_address 172.18.5.181:8095 --strategy FedUniV2.1 --min_num_clients 5 --img_size 256 --alpha 0.0 --beta 0.5 --prompt universal --attention dual --dual_init aggregated --label_prompt 1 --role server --client client_all --sup_type mask --gpu 0

python  flower_pCE_2D_v4_auxpCE_GatedCRFLoss_Rebuttal_AuxiDecoder_NoDiceLoss.py --root_path ../data/FAZ_h5 --num_classes 2 --in_chns 1 --img_class faz --exp faz/Rebuttal_AuxiDecoder_NoDiceLoss --model unet_univ5 --max_iterations 30000 --iters 5 --eval_iters 5 --tsne_iters 200 --batch_size 12 --base_lr 0.01 --amp 0 --server_address 172.18.5.181:8095 --strategy FedUniV2.1 --min_num_clients 5 --img_size 256 --alpha 0.0 --beta 0.5 --prompt universal --attention dual --dual_init aggregated --label_prompt 1 --role client --cid 0 --client client1 --sup_type scribble_noisy --gpu 1

python  flower_pCE_2D_v4_auxpCE_GatedCRFLoss_Rebuttal_AuxiDecoder_NoDiceLoss.py --root_path ../data/FAZ_h5 --num_classes 2 --in_chns 1 --img_class faz --exp faz/Rebuttal_AuxiDecoder_NoDiceLoss --model unet_univ5 --max_iterations 30000 --iters 5 --eval_iters 5 --tsne_iters 200 --batch_size 12 --base_lr 0.01 --amp 0 --server_address 172.18.5.181:8095 --strategy FedUniV2.1 --min_num_clients 5 --img_size 256 --alpha 0.0 --beta 0.5 --prompt universal --attention dual --dual_init aggregated --label_prompt 1 --role client --cid 1 --client client2 --sup_type keypoint --gpu 1

python  flower_pCE_2D_v4_auxpCE_GatedCRFLoss_Rebuttal_AuxiDecoder_NoDiceLoss.py --root_path ../data/FAZ_h5 --num_classes 2 --in_chns 1 --img_class faz --exp faz/Rebuttal_AuxiDecoder_NoDiceLoss --model unet_univ5 --max_iterations 30000 --iters 5 --eval_iters 5 --tsne_iters 200 --batch_size 12 --base_lr 0.01 --amp 0 --server_address 172.18.5.181:8095 --strategy FedUniV2.1 --min_num_clients 5 --img_size 256 --alpha 0.0 --beta 0.5 --prompt universal --attention dual --dual_init aggregated --label_prompt 1 --role client --cid 2 --client client3 --sup_type block --gpu 2

python  flower_pCE_2D_v4_auxpCE_GatedCRFLoss_Rebuttal_AuxiDecoder_NoDiceLoss.py --root_path ../data/FAZ_h5 --num_classes 2 --in_chns 1 --img_class faz --exp faz/Rebuttal_AuxiDecoder_NoDiceLoss --model unet_univ5 --max_iterations 30000 --iters 5 --eval_iters 5 --tsne_iters 200 --batch_size 12 --base_lr 0.01 --amp 0 --server_address 172.18.5.181:8095 --strategy FedUniV2.1 --min_num_clients 5 --img_size 256 --alpha 0.0 --beta 0.5 --prompt universal --attention dual --dual_init aggregated --label_prompt 1 --role client --cid 3 --client client4 --sup_type box --gpu 2

python  flower_pCE_2D_v4_auxpCE_GatedCRFLoss_Rebuttal_AuxiDecoder_NoDiceLoss.py --root_path ../data/FAZ_h5 --num_classes 2 --in_chns 1 --img_class faz --exp faz/Rebuttal_AuxiDecoder_NoDiceLoss --model unet_univ5 --max_iterations 30000 --iters 5 --eval_iters 5 --tsne_iters 200 --batch_size 12 --base_lr 0.01 --amp 0 --server_address 172.18.5.181:8095 --strategy FedUniV2.1 --min_num_clients 5 --img_size 256 --alpha 0.0 --beta 0.5 --prompt universal --attention dual --dual_init aggregated --label_prompt 1 --role client --cid 4 --client client5 --sup_type scribble --gpu 0






python flower_pCE_2D_v4_auxpCE_GatedCRFLoss.py --root_path ../data/Polypdata_h5 --num_classes 2 --in_chns 3 --img_class polyp --exp polyp/Rebuttal_Ablation_iter20 --model unet_univ5 --max_iterations 30000 --iters 20 --eval_iters 20 --tsne_iters 200 --batch_size 12 --base_lr 0.001 --amp 0 --server_address 127.0.0.1:8093 --strategy FedUniV2.1 --min_num_clients 4 --img_size 384 --alpha 0.5 --beta 0.1 --prompt universal --attention dual --dual_init aggregated --label_prompt 1 --role server --client client_all --sup_type mask --gpu 0

python flower_pCE_2D_v4_auxpCE_GatedCRFLoss.py --root_path ../data/Polypdata_h5 --num_classes 2 --in_chns 3 --img_class polyp --exp polyp/Rebuttal_Ablation_iter20 --model unet_univ5 --max_iterations 30000 --iters 20 --eval_iters 20 --tsne_iters 200 --batch_size 12 --base_lr 0.001 --amp 0 --server_address 127.0.0.1:8093 --strategy FedUniV2.1 --min_num_clients 4 --img_size 384 --alpha 0.5 --beta 0.1 --prompt universal --attention dual --dual_init aggregated --label_prompt 1 --role client --cid 0 --client client1 --sup_type keypoint --gpu 0

python flower_pCE_2D_v4_auxpCE_GatedCRFLoss.py --root_path ../data/Polypdata_h5 --num_classes 2 --in_chns 3 --img_class polyp --exp polyp/Rebuttal_Ablation_iter20 --model unet_univ5 --max_iterations 30000 --iters 20 --eval_iters 20 --tsne_iters 200 --batch_size 12 --base_lr 0.001 --amp 0 --server_address 127.0.0.1:8093 --strategy FedUniV2.1 --min_num_clients 4 --img_size 384 --alpha 0.5 --beta 0.1 --prompt universal --attention dual --dual_init aggregated --label_prompt 1 --role client --cid 1 --client client2 --sup_type scribble --gpu 1

python flower_pCE_2D_v4_auxpCE_GatedCRFLoss.py --root_path ../data/Polypdata_h5 --num_classes 2 --in_chns 3 --img_class polyp --exp polyp/Rebuttal_Ablation_iter20 --model unet_univ5 --max_iterations 30000 --iters 20 --eval_iters 20 --tsne_iters 200 --batch_size 12 --base_lr 0.001 --amp 0 --server_address 127.0.0.1:8093 --strategy FedUniV2.1 --min_num_clients 4 --img_size 384 --alpha 0.5 --beta 0.1 --prompt universal --attention dual --dual_init aggregated --label_prompt 1 --role client --cid 2 --client client3 --sup_type box --gpu 2

python flower_pCE_2D_v4_auxpCE_GatedCRFLoss.py --root_path ../data/Polypdata_h5 --num_classes 2 --in_chns 3 --img_class polyp --exp polyp/Rebuttal_Ablation_iter20 --model unet_univ5 --max_iterations 30000 --iters 20 --eval_iters 20 --tsne_iters 200 --batch_size 12 --base_lr 0.001 --amp 0 --server_address 127.0.0.1:8093 --strategy FedUniV2.1 --min_num_clients 4 --img_size 384 --alpha 0.5 --beta 0.1 --prompt universal --attention dual --dual_init aggregated --label_prompt 1 --role client --cid 3 --client client4 --sup_type block --gpu 3






# Polyp WSLSetting
python flower_pCE_2D_v4_auxpCE_Rebuttal_WSLSetting.py --root_path ../data/Polypdata_h5 --num_classes 2 --in_chns 3 --img_class polyp --exp polyp/Rebuttal_WSL_Setting_FedAvg_lr0.01 --model unet_univ5 --max_iterations 30000 --iters 5 --eval_iters 5 --tsne_iters 200 --batch_size 12 --base_lr 0.01 --amp 0 --server_address 172.18.5.181:8095 --strategy FedAvg --min_num_clients 4 --role server --client client_all --sup_type mask --gpu 5 --img_size 384

python flower_pCE_2D_v4_auxpCE_Rebuttal_WSLSetting.py --root_path ../data/Polypdata_h5 --num_classes 2 --in_chns 3 --img_class polyp --exp polyp/Rebuttal_WSL_Setting_FedAvg_lr0.01 --model unet_univ5 --max_iterations 30000 --iters 5 --eval_iters 5 --tsne_iters 200 --batch_size 12 --base_lr 0.01 --amp 0 --server_address 172.18.5.181:8095 --strategy FedAvg --min_num_clients 4 --role client --cid 0 --client client1 --sup_type keypoint --gpu 5 --img_size 384

python flower_pCE_2D_v4_auxpCE_Rebuttal_WSLSetting.py --root_path ../data/Polypdata_h5 --num_classes 2 --in_chns 3 --img_class polyp --exp polyp/Rebuttal_WSL_Setting_FedAvg_lr0.01 --model unet_univ5 --max_iterations 30000 --iters 5 --eval_iters 5 --tsne_iters 200 --batch_size 12 --base_lr 0.01 --amp 0 --server_address 172.18.5.181:8095 --strategy FedAvg --min_num_clients 4 --role client --cid 1 --client client2 --sup_type scribble --gpu 6 --img_size 384

python flower_pCE_2D_v4_auxpCE_Rebuttal_WSLSetting.py --root_path ../data/Polypdata_h5 --num_classes 2 --in_chns 3 --img_class polyp --exp polyp/Rebuttal_WSL_Setting_FedAvg_lr0.01 --model unet_univ5 --max_iterations 30000 --iters 5 --eval_iters 5 --tsne_iters 200 --batch_size 12 --base_lr 0.01 --amp 0 --server_address 172.18.5.181:8095 --strategy FedAvg --min_num_clients 4 --role client --cid 2 --client client3 --sup_type box --gpu 7 --img_size 384

python flower_pCE_2D_v4_auxpCE_Rebuttal_WSLSetting.py --root_path ../data/Polypdata_h5 --num_classes 2 --in_chns 3 --img_class polyp --exp polyp/Rebuttal_WSL_Setting_FedAvg_lr0.01 --model unet_univ5 --max_iterations 30000 --iters 5 --eval_iters 5 --tsne_iters 200 --batch_size 12 --base_lr 0.01 --amp 0 --server_address 172.18.5.181:8095 --strategy FedAvg --min_num_clients 4 --role client --cid 3 --client client4 --sup_type block --gpu 7 --img_size 384


# FAZ
python flower_pCE_2D_v4_auxpCE_Rebuttal_WSLSetting.py --root_path ../data/FAZ_h5 --num_classes 2 --in_chns 1 --img_class faz --exp faz/Rebuttal_WSL_Setting_FedAvg --model unet_univ5 --max_iterations 30000 --iters 5 --eval_iters 5 --tsne_iters 200 --batch_size 12 --base_lr 0.01 --amp 0 --server_address 172.18.5.181:8091 --strategy FedAvg --min_num_clients 5 --role server --client client_all --sup_type mask --gpu 0 --img_size 256

python flower_pCE_2D_v4_auxpCE_Rebuttal_WSLSetting.py --root_path ../data/FAZ_h5 --num_classes 2 --in_chns 1 --img_class faz --exp faz/Rebuttal_WSL_Setting_FedAvg --model unet_univ5 --max_iterations 30000 --iters 5 --eval_iters 5 --tsne_iters 200 --batch_size 12 --base_lr 0.01 --amp 0 --server_address 172.18.5.181:8091 --strategy FedAvg --min_num_clients 5 --role client --cid 0 --client client1 --sup_type scribble_noisy --gpu 0 --img_size 256

python flower_pCE_2D_v4_auxpCE_Rebuttal_WSLSetting.py --root_path ../data/FAZ_h5 --num_classes 2 --in_chns 1 --img_class faz --exp faz/Rebuttal_WSL_Setting_FedAvg --model unet_univ5 --max_iterations 30000 --iters 5 --eval_iters 5 --tsne_iters 200 --batch_size 12 --base_lr 0.01 --amp 0 --server_address 172.18.5.181:8091 --strategy FedAvg --min_num_clients 5 --role client --cid 1 --client client2 --sup_type keypoint --gpu 1 --img_size 256

python flower_pCE_2D_v4_auxpCE_Rebuttal_WSLSetting.py --root_path ../data/FAZ_h5 --num_classes 2 --in_chns 1 --img_class faz --exp faz/Rebuttal_WSL_Setting_FedAvg --model unet_univ5 --max_iterations 30000 --iters 5 --eval_iters 5 --tsne_iters 200 --batch_size 12 --base_lr 0.01 --amp 0 --server_address 172.18.5.181:8091 --strategy FedAvg --min_num_clients 5 --role client --cid 2 --client client3 --sup_type block --gpu 2 --img_size 256

python flower_pCE_2D_v4_auxpCE_Rebuttal_WSLSetting.py --root_path ../data/FAZ_h5 --num_classes 2 --in_chns 1 --img_class faz --exp faz/Rebuttal_WSL_Setting_FedAvg --model unet_univ5 --max_iterations 30000 --iters 5 --eval_iters 5 --tsne_iters 200 --batch_size 12 --base_lr 0.01 --amp 0 --server_address 172.18.5.181:8091 --strategy FedAvg --min_num_clients 5 --role client --cid 3 --client client4 --sup_type box --gpu 3 --img_size 256

python flower_pCE_2D_v4_auxpCE_Rebuttal_WSLSetting.py --root_path ../data/FAZ_h5 --num_classes 2 --in_chns 1 --img_class faz --exp faz/Rebuttal_WSL_Setting_FedAvg --model unet_univ5 --max_iterations 30000 --iters 5 --eval_iters 5 --tsne_iters 200 --batch_size 12 --base_lr 0.01 --amp 0 --server_address 172.18.5.181:8091 --strategy FedAvg --min_num_clients 5 --role client --cid 4 --client client5 --sup_type scribble --gpu 4 --img_size 256



# FedALA
python flower_runner_v2.1.py --port 8091 --procedure flower_pCE_2D_GatedCRFLoss_v2.1_auxpCE --exp faz/Rebuttal_WSL_Setting_FedALA --base_lr 0.01 --img_class faz --model unet_univ5 --gpus 0 0 0 1 1 1 --iters 5 --strategy FedALA --img_size 256 --debug 1



python flower_pCE_2D_v4_auxpCE_Rebuttal_WSLSetting.py --root_path ../data/FAZ_h5 --num_classes 2 --in_chns 1 --img_class faz --exp faz/Rebuttal_WSL_Setting_FedALA_evl_iter10 --model unet_univ5 --max_iterations 30000 --iters 5 --eval_iters 10 --tsne_iters 200 --batch_size 12 --base_lr 0.01 --amp 0 --server_address [::]:8091 --strategy FedALA --min_num_clients 5 --img_size 256 --role server --client client_all --sup_type mask --gpu 0

python flower_pCE_2D_v4_auxpCE_Rebuttal_WSLSetting.py --root_path ../data/FAZ_h5 --num_classes 2 --in_chns 1 --img_class faz --exp faz/Rebuttal_WSL_Setting_FedALA_evl_iter10 --model unet_univ5 --max_iterations 30000 --iters 5 --eval_iters 10 --tsne_iters 200 --batch_size 12 --base_lr 0.01 --amp 0 --server_address [::]:8091 --strategy FedALA --min_num_clients 5 --img_size 256 --role client --cid 0 --client client1 --sup_type scribble_noisy --gpu 0

python flower_pCE_2D_v4_auxpCE_Rebuttal_WSLSetting.py --root_path ../data/FAZ_h5 --num_classes 2 --in_chns 1 --img_class faz --exp faz/Rebuttal_WSL_Setting_FedALA_evl_iter10 --model unet_univ5 --max_iterations 30000 --iters 5 --eval_iters 10 --tsne_iters 200 --batch_size 12 --base_lr 0.01 --amp 0 --server_address [::]:8091 --strategy FedALA --min_num_clients 5 --img_size 256 --role client --cid 1 --client client2 --sup_type keypoint --gpu 6

python flower_pCE_2D_v4_auxpCE_Rebuttal_WSLSetting.py --root_path ../data/FAZ_h5 --num_classes 2 --in_chns 1 --img_class faz --exp faz/Rebuttal_WSL_Setting_FedALA_evl_iter10 --model unet_univ5 --max_iterations 30000 --iters 5 --eval_iters 10 --tsne_iters 200 --batch_size 12 --base_lr 0.01 --amp 0 --server_address [::]:8091 --strategy FedALA --min_num_clients 5 --img_size 256 --role client --cid 2 --client client3 --sup_type block --gpu 6

python flower_pCE_2D_v4_auxpCE_Rebuttal_WSLSetting.py --root_path ../data/FAZ_h5 --num_classes 2 --in_chns 1 --img_class faz --exp faz/Rebuttal_WSL_Setting_FedALA_evl_iter10 --model unet_univ5 --max_iterations 30000 --iters 5 --eval_iters 10 --tsne_iters 200 --batch_size 12 --base_lr 0.01 --amp 0 --server_address [::]:8091 --strategy FedALA --min_num_clients 5 --img_size 256 --role client --cid 3 --client client4 --sup_type box --gpu 7

python flower_pCE_2D_v4_auxpCE_Rebuttal_WSLSetting.py --root_path ../data/FAZ_h5 --num_classes 2 --in_chns 1 --img_class faz --exp faz/Rebuttal_WSL_Setting_FedALA_evl_iter10 --model unet_univ5 --max_iterations 30000 --iters 5 --eval_iters 10 --tsne_iters 200 --batch_size 12 --base_lr 0.01 --amp 0 --server_address [::]:8091 --strategy FedALA --min_num_clients 5 --img_size 256 --role client --cid 4 --client client5 --sup_type scribble --gpu 7



# FedALALC
python flower_runner.py --port 8094 --procedure flower_pCE_2D_GatedCRFLoss_v2.1_auxpCE --exp faz/Rebuttal_Ablation_FedICRA --base_lr 0.01 --img_class faz --model unet_lc_auxi --gpus 0 3 3 4 4 5 --iters 5 --eval_iters 5 --rep_iters 2 --alpha 5 --img_size 256 --strategy FedALALC --debug 1


python flower_pCE_2D_v4_auxpCE_Rebuttal_WSLSetting_GatedCRFLoss.py --root_path ../data/FAZ_h5 --num_classes 2 --in_chns 1 --img_class faz --exp faz/Rebuttal_Ablation_FedICRA --model unet_lc_auxi --max_iterations 30000 --iters 5 --eval_iters 5 --tsne_iters 200 --batch_size 12 --base_lr 0.01 --amp 0 --server_address [::]:8094 --strategy FedALALC --min_num_clients 5 --alpha 5 --rep_iters 2 --role server --client client_all --sup_type mask --gpu 0 --prompt onehot --img_size 256

python flower_pCE_2D_v4_auxpCE_Rebuttal_WSLSetting_GatedCRFLoss.py --root_path ../data/FAZ_h5 --num_classes 2 --in_chns 1 --img_class faz --exp faz/Rebuttal_Ablation_FedICRA --model unet_lc_auxi --max_iterations 30000 --iters 5 --eval_iters 5 --tsne_iters 200 --batch_size 12 --base_lr 0.01 --amp 0 --server_address [::]:8094 --strategy FedALALC --min_num_clients 5 --alpha 5 --rep_iters 2 --role client --cid 0 --client client1 --sup_type scribble_noisy --gpu 3 --prompt onehot --img_size 256

python flower_pCE_2D_v4_auxpCE_Rebuttal_WSLSetting_GatedCRFLoss.py --root_path ../data/FAZ_h5 --num_classes 2 --in_chns 1 --img_class faz --exp faz/Rebuttal_Ablation_FedICRA --model unet_lc_auxi --max_iterations 30000 --iters 5 --eval_iters 5 --tsne_iters 200 --batch_size 12 --base_lr 0.01 --amp 0 --server_address [::]:8094 --strategy FedALALC --min_num_clients 5 --alpha 5 --rep_iters 2 --role client --cid 1 --client client2 --sup_type keypoint --gpu 4 --prompt onehot --img_size 256

python flower_pCE_2D_v4_auxpCE_Rebuttal_WSLSetting_GatedCRFLoss.py --root_path ../data/FAZ_h5 --num_classes 2 --in_chns 1 --img_class faz --exp faz/Rebuttal_Ablation_FedICRA --model unet_lc_auxi --max_iterations 30000 --iters 5 --eval_iters 5 --tsne_iters 200 --batch_size 12 --base_lr 0.01 --amp 0 --server_address [::]:8094 --strategy FedALALC --min_num_clients 5 --alpha 5 --rep_iters 2 --role client --cid 2 --client client3 --sup_type block --gpu 5 --prompt onehot --img_size 256

python flower_pCE_2D_v4_auxpCE_Rebuttal_WSLSetting_GatedCRFLoss.py --root_path ../data/FAZ_h5 --num_classes 2 --in_chns 1 --img_class faz --exp faz/Rebuttal_Ablation_FedICRA --model unet_lc_auxi --max_iterations 30000 --iters 5 --eval_iters 5 --tsne_iters 200 --batch_size 12 --base_lr 0.01 --amp 0 --server_address [::]:8094 --strategy FedALALC --min_num_clients 5 --alpha 5 --rep_iters 2 --role client --cid 3 --client client4 --sup_type box --gpu 1 --prompt onehot --img_size 256

python flower_pCE_2D_v4_auxpCE_Rebuttal_WSLSetting_GatedCRFLoss.py --root_path ../data/FAZ_h5 --num_classes 2 --in_chns 1 --img_class faz --exp faz/Rebuttal_Ablation_FedICRA --model unet_lc_auxi --max_iterations 30000 --iters 5 --eval_iters 5 --tsne_iters 200 --batch_size 12 --base_lr 0.01 --amp 0 --server_address [::]:8094 --strategy FedALALC --min_num_clients 5 --alpha 5 --rep_iters 2 --role client --cid 4 --client client5 --sup_type scribble --gpu 2 --prompt onehot --img_size 256





# FedALA Polyp Rebuttal
python flower_pCE_2D_v4_auxpCE_Rebuttal_WSLSetting.py --root_path ../data/Polypdata_h5 --num_classes 2 --in_chns 3 --img_class polyp --exp polyp/Rebuttal_WSL_Setting_FedALA --model unet_univ5 --max_iterations 30000 --iters 5 --eval_iters 200 --tsne_iters 200 --batch_size 12 --base_lr 0.01 --amp 0 --server_address [::]:8096 --strategy FedALA --min_num_clients 4 --img_size 384 --role server --client client_all --sup_type mask --gpu 0

python flower_pCE_2D_v4_auxpCE_Rebuttal_WSLSetting.py --root_path ../data/Polypdata_h5 --num_classes 2 --in_chns 3 --img_class polyp --exp polyp/Rebuttal_WSL_Setting_FedALA --model unet_univ5 --max_iterations 30000 --iters 5 --eval_iters 200 --tsne_iters 200 --batch_size 12 --base_lr 0.01 --amp 0 --server_address [::]:8096 --strategy FedALA --min_num_clients 4 --img_size 384 --role client --cid 0 --client client1 --sup_type keypoint--gpu 1

python flower_pCE_2D_v4_auxpCE_Rebuttal_WSLSetting.py --root_path ../data/Polypdata_h5 --num_classes 2 --in_chns 3 --img_class polyp --exp polyp/Rebuttal_WSL_Setting_FedALA --model unet_univ5 --max_iterations 30000 --iters 5 --eval_iters 200 --tsne_iters 200 --batch_size 12 --base_lr 0.01 --amp 0 --server_address [::]:8096 --strategy FedALA --min_num_clients 4 --img_size 384 --role client --cid 1 --client client2 --sup_type scribble --gpu 2

python flower_pCE_2D_v4_auxpCE_Rebuttal_WSLSetting.py --root_path ../data/Polypdata_h5 --num_classes 2 --in_chns 3 --img_class polyp --exp polyp/Rebuttal_WSL_Setting_FedALA --model unet_univ5 --max_iterations 30000 --iters 5 --eval_iters 200 --tsne_iters 200 --batch_size 12 --base_lr 0.01 --amp 0 --server_address [::]:8096 --strategy FedALA --min_num_clients 4 --img_size 384 --role client --cid 2 --client client3 --sup_type box --gpu 3

python flower_pCE_2D_v4_auxpCE_Rebuttal_WSLSetting.py --root_path ../data/Polypdata_h5 --num_classes 2 --in_chns 3 --img_class polyp --exp polyp/Rebuttal_WSL_Setting_FedALA --model unet_univ5 --max_iterations 30000 --iters 5 --eval_iters 200 --tsne_iters 200 --batch_size 12 --base_lr 0.01 --amp 0 --server_address [::]:8096 --strategy FedALA --min_num_clients 4 --img_size 384 --role client --cid 3 --client client4 --sup_type block --gpu 4





# Prostate FedAvg Explore
# ##AdamW 0.001
python flower_pCE_2D_v4_AddProstate.py --root_path ../data/Prostate_h5 --num_classes 2 --in_chns 1 --img_class prostate --exp prostate/Rebuttal_WSL_Setting_FedAvg_AdamW_lr0.001 --model unet --max_iterations 30000 --iters 5 --eval_iters 5 --tsne_iters 200 --batch_size 12 --base_lr 0.001 --amp 0 --server_address 172.18.5.181:8091 --strategy FedAvg --min_num_clients 6 --role server --client client_all --sup_type mask --gpu 0 --img_size 384

python flower_pCE_2D_v4_AddProstate.py --root_path ../data/Prostate_h5 --num_classes 2 --in_chns 1 --img_class prostate --exp prostate/Rebuttal_WSL_Setting_FedAvg_AdamW_lr0.001 --model unet --max_iterations 30000 --iters 5 --eval_iters 5 --tsne_iters 200 --batch_size 12 --base_lr 0.001 --amp 0 --server_address 172.18.5.181:8091 --strategy FedAvg --min_num_clients 6 --role client --cid 0 --client client1 --sup_type block --gpu 0 --img_size 384

python flower_pCE_2D_v4_AddProstate.py --root_path ../data/Prostate_h5 --num_classes 2 --in_chns 1 --img_class prostate --exp prostate/Rebuttal_WSL_Setting_FedAvg_AdamW_lr0.001 --model unet --max_iterations 30000 --iters 5 --eval_iters 5 --tsne_iters 200 --batch_size 12 --base_lr 0.001 --amp 0 --server_address 172.18.5.181:8091 --strategy FedAvg --min_num_clients 6 --role client --cid 1 --client client2 --sup_type keypoint --gpu 1 --img_size 384

python flower_pCE_2D_v4_AddProstate.py --root_path ../data/Prostate_h5 --num_classes 2 --in_chns 1 --img_class prostate --exp prostate/Rebuttal_WSL_Setting_FedAvg_AdamW_lr0.001 --model unet --max_iterations 30000 --iters 5 --eval_iters 5 --tsne_iters 200 --batch_size 12 --base_lr 0.001 --amp 0 --server_address 172.18.5.181:8091 --strategy FedAvg --min_num_clients 6 --role client --cid 2 --client client3 --sup_type scribble --gpu 2 --img_size 384

python flower_pCE_2D_v4_AddProstate.py --root_path ../data/Prostate_h5 --num_classes 2 --in_chns 1 --img_class prostate --exp prostate/Rebuttal_WSL_Setting_FedAvg_AdamW_lr0.001 --model unet --max_iterations 30000 --iters 5 --eval_iters 5 --tsne_iters 200 --batch_size 12 --base_lr 0.001 --amp 0 --server_address 172.18.5.181:8091 --strategy FedAvg --min_num_clients 6 --role client --cid 3 --client client4 --sup_type keypoint --gpu 3 --img_size 384

python flower_pCE_2D_v4_AddProstate.py --root_path ../data/Prostate_h5 --num_classes 2 --in_chns 1 --img_class prostate --exp prostate/Rebuttal_WSL_Setting_FedAvg_AdamW_lr0.001 --model unet --max_iterations 30000 --iters 5 --eval_iters 5 --tsne_iters 200 --batch_size 12 --base_lr 0.001 --amp 0 --server_address 172.18.5.181:8091 --strategy FedAvg --min_num_clients 6 --role client --cid 4 --client client5 --sup_type scribble --gpu 4 --img_size 384

python flower_pCE_2D_v4_AddProstate.py --root_path ../data/Prostate_h5 --num_classes 2 --in_chns 1 --img_class prostate --exp prostate/Rebuttal_WSL_Setting_FedAvg_AdamW_lr0.001 --model unet --max_iterations 30000 --iters 5 --eval_iters 5 --tsne_iters 200 --batch_size 12 --base_lr 0.001 --amp 0 --server_address 172.18.5.181:8091 --strategy FedAvg --min_num_clients 6 --role client --cid 5 --client client6 --sup_type box --gpu 5 --img_size 384




# Prostate FedProx
python flower_pCE_2D_v4_AddProstate.py --root_path ../data/Prostate_h5 --num_classes 2 --in_chns 1 --img_class prostate --exp prostate/Rebuttal_WSL_Setting_FedProx_AdamW_lr0.001 --model unet --max_iterations 30000 --iters 5 --eval_iters 5 --tsne_iters 200 --batch_size 12 --base_lr 0.001 --amp 0 --server_address 172.18.5.181:8092 --strategy FedProx --min_num_clients 6 --role server --client client_all --sup_type mask --gpu 0 --img_size 384

python flower_pCE_2D_v4_AddProstate.py --root_path ../data/Prostate_h5 --num_classes 2 --in_chns 1 --img_class prostate --exp prostate/Rebuttal_WSL_Setting_FedProx_AdamW_lr0.001 --model unet --max_iterations 30000 --iters 5 --eval_iters 5 --tsne_iters 200 --batch_size 12 --base_lr 0.001 --amp 0 --server_address 172.18.5.181:8092 --strategy FedProx --min_num_clients 6 --role client --cid 0 --client client1 --sup_type block --gpu 4 --img_size 384

python flower_pCE_2D_v4_AddProstate.py --root_path ../data/Prostate_h5 --num_classes 2 --in_chns 1 --img_class prostate --exp prostate/Rebuttal_WSL_Setting_FedProx_AdamW_lr0.001 --model unet --max_iterations 30000 --iters 5 --eval_iters 5 --tsne_iters 200 --batch_size 12 --base_lr 0.001 --amp 0 --server_address 172.18.5.181:8092 --strategy FedProx --min_num_clients 6 --role client --cid 1 --client client2 --sup_type keypoint --gpu 5 --img_size 384

python flower_pCE_2D_v4_AddProstate.py --root_path ../data/Prostate_h5 --num_classes 2 --in_chns 1 --img_class prostate --exp prostate/Rebuttal_WSL_Setting_FedProx_AdamW_lr0.001 --model unet --max_iterations 30000 --iters 5 --eval_iters 5 --tsne_iters 200 --batch_size 12 --base_lr 0.001 --amp 0 --server_address 172.18.5.181:8092 --strategy FedProx --min_num_clients 6 --role client --cid 2 --client client3 --sup_type scribble --gpu 6 --img_size 384

python flower_pCE_2D_v4_AddProstate.py --root_path ../data/Prostate_h5 --num_classes 2 --in_chns 1 --img_class prostate --exp prostate/Rebuttal_WSL_Setting_FedProx_AdamW_lr0.001 --model unet --max_iterations 30000 --iters 5 --eval_iters 5 --tsne_iters 200 --batch_size 12 --base_lr 0.001 --amp 0 --server_address 172.18.5.181:8092 --strategy FedProx --min_num_clients 6 --role client --cid 3 --client client4 --sup_type keypoint --gpu 6 --img_size 384

python flower_pCE_2D_v4_AddProstate.py --root_path ../data/Prostate_h5 --num_classes 2 --in_chns 1 --img_class prostate --exp prostate/Rebuttal_WSL_Setting_FedProx_AdamW_lr0.001 --model unet --max_iterations 30000 --iters 5 --eval_iters 5 --tsne_iters 200 --batch_size 12 --base_lr 0.001 --amp 0 --server_address 172.18.5.181:8092 --strategy FedProx --min_num_clients 6 --role client --cid 4 --client client5 --sup_type scribble --gpu 7 --img_size 384

python flower_pCE_2D_v4_AddProstate.py --root_path ../data/Prostate_h5 --num_classes 2 --in_chns 1 --img_class prostate --exp prostate/Rebuttal_WSL_Setting_FedProx_AdamW_lr0.001 --model unet --max_iterations 30000 --iters 5 --eval_iters 5 --tsne_iters 200 --batch_size 12 --base_lr 0.001 --amp 0 --server_address 172.18.5.181:8092 --strategy FedProx --min_num_clients 6 --role client --cid 5 --client client6 --sup_type box --gpu 7 --img_size 384





# Prostate FedRep
python flower_pCE_2D_v4_AddProstate.py --root_path ../data/Prostate_h5 --num_classes 2 --in_chns 1 --img_class prostate --exp prostate/Rebuttal_WSL_Setting_FedRep_AdamW_lr0.01 --model unet --max_iterations 30000 --iters 5 --eval_iters 5 --tsne_iters 200 --batch_size 12 --base_lr 0.01 --amp 0 --server_address 127.0.0.1:8092 --strategy FedRep --min_num_clients 6 --role server --client client_all --sup_type mask --gpu 0 --img_size 384 --rep_iters 2

python flower_pCE_2D_v4_AddProstate.py --root_path ../data/Prostate_h5 --num_classes 2 --in_chns 1 --img_class prostate --exp prostate/Rebuttal_WSL_Setting_FedRep_AdamW_lr0.01 --model unet --max_iterations 30000 --iters 5 --eval_iters 5 --tsne_iters 200 --batch_size 12 --base_lr 0.01 --amp 0 --server_address 127.0.0.1:8092 --strategy FedRep --min_num_clients 6 --role client --cid 0 --client client1 --sup_type block --gpu 0 --img_size 384 --rep_iters 2

python flower_pCE_2D_v4_AddProstate.py --root_path ../data/Prostate_h5 --num_classes 2 --in_chns 1 --img_class prostate --exp prostate/Rebuttal_WSL_Setting_FedRep_AdamW_lr0.01 --model unet --max_iterations 30000 --iters 5 --eval_iters 5 --tsne_iters 200 --batch_size 12 --base_lr 0.01 --amp 0 --server_address 127.0.0.1:8092 --strategy FedRep --min_num_clients 6 --role client --cid 1 --client client2 --sup_type keypoint --gpu 1 --img_size 384 --rep_iters 2

python flower_pCE_2D_v4_AddProstate.py --root_path ../data/Prostate_h5 --num_classes 2 --in_chns 1 --img_class prostate --exp prostate/Rebuttal_WSL_Setting_FedRep_AdamW_lr0.01 --model unet --max_iterations 30000 --iters 5 --eval_iters 5 --tsne_iters 200 --batch_size 12 --base_lr 0.01 --amp 0 --server_address 127.0.0.1:8092 --strategy FedRep --min_num_clients 6 --role client --cid 2 --client client3 --sup_type scribble --gpu 1 --img_size 384 --rep_iters 2

python flower_pCE_2D_v4_AddProstate.py --root_path ../data/Prostate_h5 --num_classes 2 --in_chns 1 --img_class prostate --exp prostate/Rebuttal_WSL_Setting_FedRep_AdamW_lr0.01 --model unet --max_iterations 30000 --iters 5 --eval_iters 5 --tsne_iters 200 --batch_size 12 --base_lr 0.01 --amp 0 --server_address 127.0.0.1:8092 --strategy FedRep --min_num_clients 6 --role client --cid 3 --client client4 --sup_type keypoint --gpu 3 --img_size 384 --rep_iters 2

python flower_pCE_2D_v4_AddProstate.py --root_path ../data/Prostate_h5 --num_classes 2 --in_chns 1 --img_class prostate --exp prostate/Rebuttal_WSL_Setting_FedRep_AdamW_lr0.01 --model unet --max_iterations 30000 --iters 5 --eval_iters 5 --tsne_iters 200 --batch_size 12 --base_lr 0.01 --amp 0 --server_address 127.0.0.1:8092 --strategy FedRep --min_num_clients 6 --role client --cid 4 --client client5 --sup_type scribble --gpu 3 --img_size 384 --rep_iters 2

python flower_pCE_2D_v4_AddProstate.py --root_path ../data/Prostate_h5 --num_classes 2 --in_chns 1 --img_class prostate --exp prostate/Rebuttal_WSL_Setting_FedRep_AdamW_lr0.01 --model unet --max_iterations 30000 --iters 5 --eval_iters 5 --tsne_iters 200 --batch_size 12 --base_lr 0.01 --amp 0 --server_address 127.0.0.1:8092 --strategy FedRep --min_num_clients 6 --role client --cid 5 --client client6 --sup_type box --gpu 2 --img_size 384 --rep_iters 4




# Warm-up
python flower_pCE_2D_v4_auxpCE_GatedCRFLoss_Lamda_Warmup.py --root_path ../data/Polypdata_h5 --num_classes 2 --in_chns 3 --img_class polyp --exp polyp/Rebuttal_Warmup_Exp_iter500 --model unet_univ5 --max_iterations 30000 --iters 5 --eval_iters 5 --tsne_iters 200 --batch_size 12 --base_lr 0.001 --amp 0 --server_address 127.0.0.1:8091 --strategy FedUniV2.1 --min_num_clients 4 --img_size 384 --alpha 0.5 --beta 0.1 --prompt universal --attention dual --dual_init aggregated --label_prompt 1 --role server --client client_all --sup_type mask --gpu 0


# FAZ Warm-up
python flower_pCE_2D_v4_auxpCE_GatedCRFLoss_Lamda_Warmup.py --root_path ../data/FAZ_h5 --num_classes 2 --in_chns 1 --img_class faz --exp faz/Rebuttal_Warmup_Exp_iter500 --model unet_lc_auxi --max_iterations 30000 --iters 5 --eval_iters 5 --tsne_iters 200 --batch_size 12 --base_lr 0.01 --amp 0 --server_address [::]:8094 --strategy FedALALC --min_num_clients 5 --alpha 5 --rep_iters 2 --role server --client client_all --sup_type mask --gpu 4 --prompt universal --img_size 256

python flower_pCE_2D_v4_auxpCE_GatedCRFLoss_Lamda_Warmup.py --root_path ../data/FAZ_h5 --num_classes 2 --in_chns 1 --img_class faz --exp faz/Rebuttal_Warmup_Exp_iter500 --model unet_lc_auxi --max_iterations 30000 --iters 5 --eval_iters 5 --tsne_iters 200 --batch_size 12 --base_lr 0.01 --amp 0 --server_address [::]:8094 --strategy FedALALC --min_num_clients 5 --alpha 5 --rep_iters 2 --role client --cid 0 --client client1 --sup_type scribble_noisy --gpu 4 --prompt universal --img_size 256

python flower_pCE_2D_v4_auxpCE_GatedCRFLoss_Lamda_Warmup.py --root_path ../data/FAZ_h5 --num_classes 2 --in_chns 1 --img_class faz --exp faz/Rebuttal_Warmup_Exp_iter500 --model unet_lc_auxi --max_iterations 30000 --iters 5 --eval_iters 5 --tsne_iters 200 --batch_size 12 --base_lr 0.01 --amp 0 --server_address [::]:8094 --strategy FedALALC --min_num_clients 5 --alpha 5 --rep_iters 2 --role client --cid 1 --client client2 --sup_type keypoint --gpu 4 --prompt universal --img_size 256

python flower_pCE_2D_v4_auxpCE_GatedCRFLoss_Lamda_Warmup.py --root_path ../data/FAZ_h5 --num_classes 2 --in_chns 1 --img_class faz --exp faz/Rebuttal_Warmup_Exp_iter500 --model unet_lc_auxi --max_iterations 30000 --iters 5 --eval_iters 5 --tsne_iters 200 --batch_size 12 --base_lr 0.01 --amp 0 --server_address [::]:8094 --strategy FedALALC --min_num_clients 5 --alpha 5 --rep_iters 2 --role client --cid 2 --client client3 --sup_type block --gpu 5 --prompt universal --img_size 256

python flower_pCE_2D_v4_auxpCE_GatedCRFLoss_Lamda_Warmup.py --root_path ../data/FAZ_h5 --num_classes 2 --in_chns 1 --img_class faz --exp faz/Rebuttal_Warmup_Exp_iter500 --model unet_lc_auxi --max_iterations 30000 --iters 5 --eval_iters 5 --tsne_iters 200 --batch_size 12 --base_lr 0.01 --amp 0 --server_address [::]:8094 --strategy FedALALC --min_num_clients 5 --alpha 5 --rep_iters 2 --role client --cid 3 --client client4 --sup_type box --gpu 1 --prompt universal --img_size 256

python flower_pCE_2D_v4_auxpCE_GatedCRFLoss_Lamda_Warmup.py --root_path ../data/FAZ_h5 --num_classes 2 --in_chns 1 --img_class faz --exp faz/Rebuttal_Warmup_Exp_iter500 --model unet_lc_auxi --max_iterations 30000 --iters 5 --eval_iters 5 --tsne_iters 200 --batch_size 12 --base_lr 0.01 --amp 0 --server_address [::]:8094 --strategy FedALALC --min_num_clients 5 --alpha 5 --rep_iters 2 --role client --cid 4 --client client5 --sup_type scribble --gpu 2 --prompt universal --img_size 256






# Warm_up Exp * 2/3
python flower_pCE_2D_v4_auxpCE_GatedCRFLoss_Lamda_Warmup.py --root_path ../data/FAZ_h5 --num_classes 2 --in_chns 1 --img_class faz --exp faz/Rebuttal_Warmup_Exp_iter30000 --model unet_univ5 --max_iterations 30000 --iters 5 --eval_iters 5 --tsne_iters 5 --batch_size 12 --base_lr 0.01 --amp 0 --server_address 127.0.0.1:8097 --strategy FedUniV2.1 --min_num_clients 5 --img_size 256 --alpha 0.0 --beta 0.5 --prompt universal --attention dual --dual_init aggregated --label_prompt 1 --role server --client client_all --sup_type mask --gpu 0

python flower_pCE_2D_v4_auxpCE_GatedCRFLoss_Lamda_Warmup.py --root_path ../data/FAZ_h5 --num_classes 2 --in_chns 1 --img_class faz --exp faz/Rebuttal_Warmup_Exp_iter30000 --model unet_univ5 --max_iterations 30000 --iters 5 --eval_iters 5 --tsne_iters 5 --batch_size 12 --base_lr 0.01 --amp 0 --server_address 127.0.0.1:8097 --strategy FedUniV2.1 --min_num_clients 5 --img_size 256 --alpha 0.0 --beta 0.5 --prompt universal --attention dual --dual_init aggregated --label_prompt 1 --role client --cid 0 --client client1 --sup_type scribble_noisy --gpu 4

python flower_pCE_2D_v4_auxpCE_GatedCRFLoss_Lamda_Warmup.py --root_path ../data/FAZ_h5 --num_classes 2 --in_chns 1 --img_class faz --exp faz/Rebuttal_Warmup_Exp_iter30000 --model unet_univ5 --max_iterations 30000 --iters 5 --eval_iters 5 --tsne_iters 5 --batch_size 12 --base_lr 0.01 --amp 0 --server_address 127.0.0.1:8097 --strategy FedUniV2.1 --min_num_clients 5 --img_size 256 --alpha 0.0 --beta 0.5 --prompt universal --attention dual --dual_init aggregated --label_prompt 1 --role client --cid 1 --client client2 --sup_type keypoint --gpu 4

python flower_pCE_2D_v4_auxpCE_GatedCRFLoss_Lamda_Warmup.py --root_path ../data/FAZ_h5 --num_classes 2 --in_chns 1 --img_class faz --exp faz/Rebuttal_Warmup_Exp_iter30000 --model unet_univ5 --max_iterations 30000 --iters 5 --eval_iters 5 --tsne_iters 5 --batch_size 12 --base_lr 0.01 --amp 0 --server_address 127.0.0.1:8097 --strategy FedUniV2.1 --min_num_clients 5 --img_size 256 --alpha 0.0 --beta 0.5 --prompt universal --attention dual --dual_init aggregated --label_prompt 1 --role client --cid 2 --client client3 --sup_type block --gpu 5

python flower_pCE_2D_v4_auxpCE_GatedCRFLoss_Lamda_Warmup.py --root_path ../data/FAZ_h5 --num_classes 2 --in_chns 1 --img_class faz --exp faz/Rebuttal_Warmup_Exp_iter30000 --model unet_univ5 --max_iterations 30000 --iters 5 --eval_iters 5 --tsne_iters 5 --batch_size 12 --base_lr 0.01 --amp 0 --server_address 127.0.0.1:8097 --strategy FedUniV2.1 --min_num_clients 5 --img_size 256 --alpha 0.0 --beta 0.5 --prompt universal --attention dual --dual_init aggregated --label_prompt 1 --role client --cid 3 --client client4 --sup_type box --gpu 5

python flower_pCE_2D_v4_auxpCE_GatedCRFLoss_Lamda_Warmup.py --root_path ../data/FAZ_h5 --num_classes 2 --in_chns 1 --img_class faz --exp faz/Rebuttal_Warmup_Exp_iter30000 --model unet_univ5 --max_iterations 30000 --iters 5 --eval_iters 5 --tsne_iters 5 --batch_size 12 --base_lr 0.01 --amp 0 --server_address 127.0.0.1:8097 --strategy FedUniV2.1 --min_num_clients 5 --img_size 256 --alpha 0.0 --beta 0.5 --prompt universal --attention dual --dual_init aggregated --label_prompt 1 --role client --cid 4 --client client5 --sup_type scribble --gpu 6




# Warm_up Linear
python flower_pCE_2D_v4_auxpCE_GatedCRFLoss_Lamda_Warmup.py --root_path ../data/FAZ_h5 --num_classes 2 --in_chns 1 --img_class faz --exp faz/Rebuttal_Warmup_Linear_iter30000 --model unet_univ5 --max_iterations 30000 --iters 5 --eval_iters 5 --tsne_iters 5 --batch_size 12 --base_lr 0.01 --amp 0 --server_address 127.0.0.1:8096 --strategy FedUniV2.1 --min_num_clients 5 --img_size 256 --alpha 0.0 --beta 0.5 --prompt universal --attention dual --dual_init aggregated --label_prompt 1 --role server --client client_all --sup_type mask --gpu 0

python flower_pCE_2D_v4_auxpCE_GatedCRFLoss_Lamda_Warmup.py --root_path ../data/FAZ_h5 --num_classes 2 --in_chns 1 --img_class faz --exp faz/Rebuttal_Warmup_Linear_iter30000 --model unet_univ5 --max_iterations 30000 --iters 5 --eval_iters 5 --tsne_iters 5 --batch_size 12 --base_lr 0.01 --amp 0 --server_address 127.0.0.1:8096 --strategy FedUniV2.1 --min_num_clients 5 --img_size 256 --alpha 0.0 --beta 0.5 --prompt universal --attention dual --dual_init aggregated --label_prompt 1 --role client --cid 0 --client client1 --sup_type scribble_noisy --gpu 0

python flower_pCE_2D_v4_auxpCE_GatedCRFLoss_Lamda_Warmup.py --root_path ../data/FAZ_h5 --num_classes 2 --in_chns 1 --img_class faz --exp faz/Rebuttal_Warmup_Linear_iter30000 --model unet_univ5 --max_iterations 30000 --iters 5 --eval_iters 5 --tsne_iters 5 --batch_size 12 --base_lr 0.01 --amp 0 --server_address 127.0.0.1:8096 --strategy FedUniV2.1 --min_num_clients 5 --img_size 256 --alpha 0.0 --beta 0.5 --prompt universal --attention dual --dual_init aggregated --label_prompt 1 --role client --cid 1 --client client2 --sup_type keypoint --gpu 2

python flower_pCE_2D_v4_auxpCE_GatedCRFLoss_Lamda_Warmup.py --root_path ../data/FAZ_h5 --num_classes 2 --in_chns 1 --img_class faz --exp faz/Rebuttal_Warmup_Linear_iter30000 --model unet_univ5 --max_iterations 30000 --iters 5 --eval_iters 5 --tsne_iters 5 --batch_size 12 --base_lr 0.01 --amp 0 --server_address 127.0.0.1:8096 --strategy FedUniV2.1 --min_num_clients 5 --img_size 256 --alpha 0.0 --beta 0.5 --prompt universal --attention dual --dual_init aggregated --label_prompt 1 --role client --cid 2 --client client3 --sup_type block --gpu 6

python flower_pCE_2D_v4_auxpCE_GatedCRFLoss_Lamda_Warmup.py --root_path ../data/FAZ_h5 --num_classes 2 --in_chns 1 --img_class faz --exp faz/Rebuttal_Warmup_Linear_iter30000 --model unet_univ5 --max_iterations 30000 --iters 5 --eval_iters 5 --tsne_iters 5 --batch_size 12 --base_lr 0.01 --amp 0 --server_address 127.0.0.1:8096 --strategy FedUniV2.1 --min_num_clients 5 --img_size 256 --alpha 0.0 --beta 0.5 --prompt universal --attention dual --dual_init aggregated --label_prompt 1 --role client --cid 3 --client client4 --sup_type box --gpu 7

python flower_pCE_2D_v4_auxpCE_GatedCRFLoss_Lamda_Warmup.py --root_path ../data/FAZ_h5 --num_classes 2 --in_chns 1 --img_class faz --exp faz/Rebuttal_Warmup_Linear_iter30000 --model unet_univ5 --max_iterations 30000 --iters 5 --eval_iters 5 --tsne_iters 5 --batch_size 12 --base_lr 0.01 --amp 0 --server_address 127.0.0.1:8096 --strategy FedUniV2.1 --min_num_clients 5 --img_size 256 --alpha 0.0 --beta 0.5 --prompt universal --attention dual --dual_init aggregated --label_prompt 1 --role client --cid 4 --client client5 --sup_type scribble --gpu 7






python flower_pCE_2D.py --root_path ../data/Polypdata_h5 --num_classes 2 --in_chns 3 --img_class polyp --exp polyp/FL_WeaklySeg_pCE_flower_FedLC_alpha0.1 --model unet_lc --max_iterations 30000 --iters 5 --eval_iters 5 --tsne_iters 200 --batch_size 12 --base_lr 0.001 --amp 0 --server_address [::]:8094 --strategy FedLC --min_num_clients 4 --alpha 0.1 --rep_iters 2 --role server --client client_all --sup_type mask --gpu 0


# Prostate FedLC
python flower_pCE_2D_v4_AddProstate.py --root_path ../data/Prostate_h5 --num_classes 2 --in_chns 1 --img_class prostate --exp prostate/Rebuttal_WSL_Setting_FedLC_AdamW_lr0.01 --model unet_lc --max_iterations 30000 --iters 5 --eval_iters 5 --tsne_iters 200 --batch_size 12 --base_lr 0.01 --amp 0 --server_address 127.0.0.1:8092 --strategy FedLC --min_num_clients 6 --role server --client client_all --sup_type mask --gpu 0 --img_size 384 --rep_iters 2 --alpha 0.1

python flower_pCE_2D_v4_AddProstate.py --root_path ../data/Prostate_h5 --num_classes 2 --in_chns 1 --img_class prostate --exp prostate/Rebuttal_WSL_Setting_FedLC_AdamW_lr0.01 --model unet_lc --max_iterations 30000 --iters 5 --eval_iters 5 --tsne_iters 200 --batch_size 12 --base_lr 0.01 --amp 0 --server_address 127.0.0.1:8092 --strategy FedLC --min_num_clients 6 --role client --cid 0 --client client1 --sup_type block --gpu 0 --img_size 384 --rep_iters 2 --alpha 0.1

python flower_pCE_2D_v4_AddProstate.py --root_path ../data/Prostate_h5 --num_classes 2 --in_chns 1 --img_class prostate --exp prostate/Rebuttal_WSL_Setting_FedLC_AdamW_lr0.01 --model unet_lc --max_iterations 30000 --iters 5 --eval_iters 5 --tsne_iters 200 --batch_size 12 --base_lr 0.01 --amp 0 --server_address 127.0.0.1:8092 --strategy FedLC --min_num_clients 6 --role client --cid 1 --client client2 --sup_type keypoint --gpu 1 --img_size 384 --rep_iters 2 --alpha 0.1

python flower_pCE_2D_v4_AddProstate.py --root_path ../data/Prostate_h5 --num_classes 2 --in_chns 1 --img_class prostate --exp prostate/Rebuttal_WSL_Setting_FedLC_AdamW_lr0.01 --model unet_lc --max_iterations 30000 --iters 5 --eval_iters 5 --tsne_iters 200 --batch_size 12 --base_lr 0.01 --amp 0 --server_address 127.0.0.1:8092 --strategy FedLC --min_num_clients 6 --role client --cid 2 --client client3 --sup_type scribble --gpu 1 --img_size 384 --rep_iters 2 --alpha 0.1

python flower_pCE_2D_v4_AddProstate.py --root_path ../data/Prostate_h5 --num_classes 2 --in_chns 1 --img_class prostate --exp prostate/Rebuttal_WSL_Setting_FedLC_AdamW_lr0.01 --model unet_lc --max_iterations 30000 --iters 5 --eval_iters 5 --tsne_iters 200 --batch_size 12 --base_lr 0.01 --amp 0 --server_address 127.0.0.1:8092 --strategy FedLC --min_num_clients 6 --role client --cid 3 --client client4 --sup_type keypoint --gpu 3 --img_size 384 --rep_iters 2 --alpha 0.1

python flower_pCE_2D_v4_AddProstate.py --root_path ../data/Prostate_h5 --num_classes 2 --in_chns 1 --img_class prostate --exp prostate/Rebuttal_WSL_Setting_FedLC_AdamW_lr0.01 --model unet_lc --max_iterations 30000 --iters 5 --eval_iters 5 --tsne_iters 200 --batch_size 12 --base_lr 0.01 --amp 0 --server_address 127.0.0.1:8092 --strategy FedLC --min_num_clients 6 --role client --cid 4 --client client5 --sup_type scribble --gpu 3 --img_size 384 --rep_iters 2 --alpha 0.1

python flower_pCE_2D_v4_AddProstate.py --root_path ../data/Prostate_h5 --num_classes 2 --in_chns 1 --img_class prostate --exp prostate/Rebuttal_WSL_Setting_FedLC_AdamW_lr0.01 --model unet_lc --max_iterations 30000 --iters 5 --eval_iters 5 --tsne_iters 200 --batch_size 12 --base_lr 0.01 --amp 0 --server_address 127.0.0.1:8092 --strategy FedLC --min_num_clients 6 --role client --cid 5 --client client6 --sup_type box --gpu 2 --img_size 384 --rep_iters 4 --alpha 0.1





# FedLC Polyp Rebuttal
python flower_pCE_2D_v4_auxpCE_Rebuttal_WSLSetting.py --root_path ../data/Polypdata_h5 --num_classes 2 --in_chns 3 --img_class polyp --exp polyp/Rebuttal_WSL_Setting_FedLC_alpha0.01 --model unet_lc_auxi --max_iterations 30000 --iters 5 --eval_iters 5 --tsne_iters 200 --batch_size 12 --base_lr 0.01 --amp 0 --server_address [::]:8096 --strategy FedLC --min_num_clients 4 --img_size 384 --role server --client client_all --sup_type mask --gpu 0 --rep_iters 2 --alpha 0.1

python flower_pCE_2D_v4_auxpCE_Rebuttal_WSLSetting.py --root_path ../data/Polypdata_h5 --num_classes 2 --in_chns 3 --img_class polyp --exp polyp/Rebuttal_WSL_Setting_FedLC_alpha0.01 --model unet_lc_auxi --max_iterations 30000 --iters 5 --eval_iters 5 --tsne_iters 200 --batch_size 12 --base_lr 0.01 --amp 0 --server_address [::]:8096 --strategy FedLC --min_num_clients 4 --img_size 384 --role client --cid 0 --client client1 --sup_type keypoint --gpu 1 --rep_iters 2 --alpha 0.1

python flower_pCE_2D_v4_auxpCE_Rebuttal_WSLSetting.py --root_path ../data/Polypdata_h5 --num_classes 2 --in_chns 3 --img_class polyp --exp polyp/Rebuttal_WSL_Setting_FedLC_alpha0.01 --model unet_lc_auxi --max_iterations 30000 --iters 5 --eval_iters 5 --tsne_iters 200 --batch_size 12 --base_lr 0.01 --amp 0 --server_address [::]:8096 --strategy FedLC --min_num_clients 4 --img_size 384 --role client --cid 1 --client client2 --sup_type scribble --gpu 2 --rep_iters 2 --alpha 0.1

python flower_pCE_2D_v4_auxpCE_Rebuttal_WSLSetting.py --root_path ../data/Polypdata_h5 --num_classes 2 --in_chns 3 --img_class polyp --exp polyp/Rebuttal_WSL_Setting_FedLC_alpha0.01 --model unet_lc_auxi --max_iterations 30000 --iters 5 --eval_iters 5 --tsne_iters 200 --batch_size 12 --base_lr 0.01 --amp 0 --server_address [::]:8096 --strategy FedLC --min_num_clients 4 --img_size 384 --role client --cid 2 --client client3 --sup_type box --gpu 3 --rep_iters 2 --alpha 0.1

python flower_pCE_2D_v4_auxpCE_Rebuttal_WSLSetting.py --root_path ../data/Polypdata_h5 --num_classes 2 --in_chns 3 --img_class polyp --exp polyp/Rebuttal_WSL_Setting_FedLC_alpha0.01 --model unet_lc_auxi --max_iterations 30000 --iters 5 --eval_iters 5 --tsne_iters 200 --batch_size 12 --base_lr 0.01 --amp 0 --server_address [::]:8096 --strategy FedLC --min_num_clients 4 --img_size 384 --role client --cid 3 --client client4 --sup_type block --gpu 4 --rep_iters 2 --alpha 0.1






# #FedLC Faz
python flower_pCE_2D_v4_auxpCE_Rebuttal_WSLSetting.py --root_path ../data/FAZ_h5 --num_classes 2 --in_chns 1 --img_class faz --exp faz/Rebuttal_WSL_Setting_FedLC --model unet_lc_auxi --max_iterations 30000 --iters 5 --eval_iters 5 --tsne_iters 200 --batch_size 12 --base_lr 0.01 --amp 0 --server_address 127.0.0.1:8095 --strategy FedLC --min_num_clients 5 --img_size 256 --rep_iters 2 --role server --client client_all --sup_type mask --gpu 5

python flower_pCE_2D_v4_auxpCE_Rebuttal_WSLSetting.py --root_path ../data/FAZ_h5 --num_classes 2 --in_chns 1 --img_class faz --exp faz/Rebuttal_WSL_Setting_FedLC --model unet_lc_auxi --max_iterations 30000 --iters 5 --eval_iters 5 --tsne_iters 200 --batch_size 12 --base_lr 0.01 --amp 0 --server_address 127.0.0.1:8095 --strategy FedLC --min_num_clients 5 --img_size 256 --rep_iters 2 --role client --cid 0 --client client1 --sup_type scribble_noisy --gpu 5

python flower_pCE_2D_v4_auxpCE_Rebuttal_WSLSetting.py --root_path ../data/FAZ_h5 --num_classes 2 --in_chns 1 --img_class faz --exp faz/Rebuttal_WSL_Setting_FedLC --model unet_lc_auxi --max_iterations 30000 --iters 5 --eval_iters 5 --tsne_iters 200 --batch_size 12 --base_lr 0.01 --amp 0 --server_address 127.0.0.1:8095 --strategy FedLC --min_num_clients 5 --img_size 256 --rep_iters 2 --role client --cid 1 --client client2 --sup_type keypoint --gpu 6

python flower_pCE_2D_v4_auxpCE_Rebuttal_WSLSetting.py --root_path ../data/FAZ_h5 --num_classes 2 --in_chns 1 --img_class faz --exp faz/Rebuttal_WSL_Setting_FedLC --model unet_lc_auxi --max_iterations 30000 --iters 5 --eval_iters 5 --tsne_iters 200 --batch_size 12 --base_lr 0.01 --amp 0 --server_address 127.0.0.1:8095 --strategy FedLC --min_num_clients 5 --img_size 256 --rep_iters 2 --role client --cid 2 --client client3 --sup_type block --gpu 6

python flower_pCE_2D_v4_auxpCE_Rebuttal_WSLSetting.py --root_path ../data/FAZ_h5 --num_classes 2 --in_chns 1 --img_class faz --exp faz/Rebuttal_WSL_Setting_FedLC --model unet_lc_auxi --max_iterations 30000 --iters 5 --eval_iters 5 --tsne_iters 200 --batch_size 12 --base_lr 0.01 --amp 0 --server_address 127.0.0.1:8095 --strategy FedLC --min_num_clients 5 --img_size 256 --rep_iters 2 --role client --cid 3 --client client4 --sup_type box --gpu 7

python flower_pCE_2D_v4_auxpCE_Rebuttal_WSLSetting.py --root_path ../data/FAZ_h5 --num_classes 2 --in_chns 1 --img_class faz --exp faz/Rebuttal_WSL_Setting_FedLC --model unet_lc_auxi --max_iterations 30000 --iters 5 --eval_iters 5 --tsne_iters 200 --batch_size 12 --base_lr 0.01 --amp 0 --server_address 127.0.0.1:8095 --strategy FedLC --min_num_clients 5 --img_size 256 --rep_iters 2 --role client --cid 4 --client client5 --sup_type scribble --gpu 7




python flower_pCE_2D_v4_FT.py --root_path ../data/Polypdata_h5 --exp polyp/Rebuttal_WSL_Setting_FedLC_alpha0.01_1  --client client0 --sup_type keypoint --model unet_lc_auxi --in_chns 3 --num_classes 2 --base_lr 0.001 --gpus 7 --img_class polyp --amp 0 --img_size 384 --batch_size 12 --cid 0 --min_num_clients 4

python flower_pCE_2D_v4_FT.py --root_path ../data/Polypdata_h5 --exp polyp/Rebuttal_WSL_Setting_FedLC_alpha0.01_1  --client client1 --sup_type scribble --model unet_lc_auxi --in_chns 3 --num_classes 2 --base_lr 0.001 --gpus 7 --img_class polyp --amp 0 --img_size 384 --batch_size 12 --cid 1 --min_num_clients 4

python flower_pCE_2D_v4_FT.py --root_path ../data/Polypdata_h5 --exp polyp/Rebuttal_WSL_Setting_FedLC_alpha0.01_1  --client client2 --sup_type box --model unet_lc_auxi --in_chns 3 --num_classes 2 --base_lr 0.001 --gpus 7 --img_class polyp --amp 0 --img_size 384 --batch_size 12 --cid 2 --min_num_clients 4

python flower_pCE_2D_v4_FT.py --root_path ../data/Polypdata_h5 --exp polyp/Rebuttal_WSL_Setting_FedLC_alpha0.01_1  --client client3 --sup_type block --model unet_lc_auxi --in_chns 3 --num_classes 2 --base_lr 0.001 --gpus 7 --img_class polyp --amp 0 --img_size 384 --batch_size 12 --cid 3 --min_num_clients 4





# iteration
# #FedLC Faz
python flower_pCE_2D_v4.py --root_path ../data/FAZ_h5 --num_classes 2 --in_chns 1 --img_class faz --exp faz/Rebuttal_Frequency_FedLC_iteration100 --model unet_lc --max_iterations 30000 --iters 100 --eval_iters 100 --tsne_iters 200 --batch_size 12 --base_lr 0.01 --amp 0 --server_address 127.0.0.1:8097 --strategy FedLC --min_num_clients 5 --img_size 256 --rep_iters 2 --role server --client client_all --sup_type mask --gpu 0

python flower_pCE_2D_v4.py --root_path ../data/FAZ_h5 --num_classes 2 --in_chns 1 --img_class faz --exp faz/Rebuttal_Frequency_FedLC_iteration100 --model unet_lc --max_iterations 30000 --iters 100 --eval_iters 100 --tsne_iters 200 --batch_size 12 --base_lr 0.01 --amp 0 --server_address 127.0.0.1:8097 --strategy FedLC --min_num_clients 5 --img_size 256 --rep_iters 2 --role client --cid 0 --client client1 --sup_type scribble_noisy --gpu 3

python flower_pCE_2D_v4.py --root_path ../data/FAZ_h5 --num_classes 2 --in_chns 1 --img_class faz --exp faz/Rebuttal_Frequency_FedLC_iteration100 --model unet_lc --max_iterations 30000 --iters 100 --eval_iters 100 --tsne_iters 200 --batch_size 12 --base_lr 0.01 --amp 0 --server_address 127.0.0.1:8097 --strategy FedLC --min_num_clients 5 --img_size 256 --rep_iters 2 --role client --cid 1 --client client2 --sup_type keypoint --gpu 3

python flower_pCE_2D_v4.py --root_path ../data/FAZ_h5 --num_classes 2 --in_chns 1 --img_class faz --exp faz/Rebuttal_Frequency_FedLC_iteration100 --model unet_lc --max_iterations 30000 --iters 100 --eval_iters 100 --tsne_iters 200 --batch_size 12 --base_lr 0.01 --amp 0 --server_address 127.0.0.1:8097 --strategy FedLC --min_num_clients 5 --img_size 256 --rep_iters 2 --role client --cid 2 --client client3 --sup_type block --gpu 3

python flower_pCE_2D_v4.py --root_path ../data/FAZ_h5 --num_classes 2 --in_chns 1 --img_class faz --exp faz/Rebuttal_Frequency_FedLC_iteration100 --model unet_lc --max_iterations 30000 --iters 100 --eval_iters 100 --tsne_iters 200 --batch_size 12 --base_lr 0.01 --amp 0 --server_address 127.0.0.1:8097 --strategy FedLC --min_num_clients 5 --img_size 256 --rep_iters 2 --role client --cid 3 --client client4 --sup_type box --gpu 4

python flower_pCE_2D_v4.py --root_path ../data/FAZ_h5 --num_classes 2 --in_chns 1 --img_class faz --exp faz/Rebuttal_Frequency_FedLC_iteration100 --model unet_lc --max_iterations 30000 --iters 100 --eval_iters 100 --tsne_iters 200 --batch_size 12 --base_lr 0.01 --amp 0 --server_address 127.0.0.1:8097 --strategy FedLC --min_num_clients 5 --img_size 256 --rep_iters 2 --role client --cid 4 --client client5 --sup_type scribble --gpu 4




# FedALA
python flower_pCE_2D_v4.py --root_path ../data/FAZ_h5 --num_classes 2 --in_chns 1 --img_class faz --exp faz/Rebuttal_Frequency_FedALA_iteration100 --model unet --max_iterations 30000 --iters 100 --eval_iters 100 --tsne_iters 200 --batch_size 12 --base_lr 0.01 --amp 0 --server_address [::]:8093 --strategy FedALA --min_num_clients 5 --img_size 256 --role server --client client_all --sup_type mask --gpu 3

python flower_pCE_2D_v4.py --root_path ../data/FAZ_h5 --num_classes 2 --in_chns 1 --img_class faz --exp faz/Rebuttal_Frequency_FedALA_iteration100 --model unet --max_iterations 30000 --iters 100 --eval_iters 100 --tsne_iters 200 --batch_size 12 --base_lr 0.01 --amp 0 --server_address [::]:8093 --strategy FedALA --min_num_clients 5 --img_size 256 --role client --cid 0 --client client1 --sup_type scribble_noisy --gpu 3

python flower_pCE_2D_v4.py --root_path ../data/FAZ_h5 --num_classes 2 --in_chns 1 --img_class faz --exp faz/Rebuttal_Frequency_FedALA_iteration100 --model unet --max_iterations 30000 --iters 100 --eval_iters 100 --tsne_iters 200 --batch_size 12 --base_lr 0.01 --amp 0 --server_address [::]:8093 --strategy FedALA --min_num_clients 5 --img_size 256 --role client --cid 1 --client client2 --sup_type keypoint --gpu 4

python flower_pCE_2D_v4.py --root_path ../data/FAZ_h5 --num_classes 2 --in_chns 1 --img_class faz --exp faz/Rebuttal_Frequency_FedALA_iteration100 --model unet --max_iterations 30000 --iters 100 --eval_iters 100 --tsne_iters 200 --batch_size 12 --base_lr 0.01 --amp 0 --server_address [::]:8093 --strategy FedALA --min_num_clients 5 --img_size 256 --role client --cid 2 --client client3 --sup_type block --gpu 4

python flower_pCE_2D_v4.py --root_path ../data/FAZ_h5 --num_classes 2 --in_chns 1 --img_class faz --exp faz/Rebuttal_Frequency_FedALA_iteration100 --model unet --max_iterations 30000 --iters 100 --eval_iters 100 --tsne_iters 200 --batch_size 12 --base_lr 0.01 --amp 0 --server_address [::]:8093 --strategy FedALA --min_num_clients 5 --img_size 256 --role client --cid 3 --client client4 --sup_type box --gpu 5

python flower_pCE_2D_v4.py --root_path ../data/FAZ_h5 --num_classes 2 --in_chns 1 --img_class faz --exp faz/Rebuttal_Frequency_FedALA_iteration100 --model unet --max_iterations 30000 --iters 100 --eval_iters 100 --tsne_iters 200 --batch_size 12 --base_lr 0.01 --amp 0 --server_address [::]:8093 --strategy FedALA --min_num_clients 5 --img_size 256 --role client --cid 4 --client client5 --sup_type scribble --gpu 5