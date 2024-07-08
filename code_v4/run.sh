# FAZ Segmentation Task
##Server
python flower_pCE_2D_v4_auxpCE_GatedCRFLoss.py --root_path ../data/FAZ_h5 --num_classes 2 --in_chns 1 --img_class faz --exp faz/FedLPPA --model unet_univ5 --max_iterations 30000 --iters 5 --eval_iters 5 --tsne_iters 200 --batch_size 12 --base_lr 0.01 --amp 0 --server_address 127.0.0.1:8091 --strategy FedUniV2.1 --min_num_clients 5 --img_size 256 --alpha 0.1 --beta 0.5 --prompt universal --attention dual --dual_init aggregated --label_prompt 1 --role server --client client_all --sup_type mask --gpu 0

##Site A
python flower_pCE_2D_v4_auxpCE_GatedCRFLoss.py --root_path ../data/FAZ_h5 --num_classes 2 --in_chns 1 --img_class faz --exp faz/FedLPPA --model unet_univ5 --max_iterations 30000 --iters 5 --eval_iters 5 --tsne_iters 200 --batch_size 12 --base_lr 0.01 --amp 0 --server_address 127.0.0.1:8091 --strategy FedUniV2.1 --min_num_clients 5 --img_size 256 --alpha 0.1 --beta 0.5 --prompt universal --attention dual --dual_init aggregated --label_prompt 1 --role client --cid 0 --client client1 --sup_type scribble_noisy --gpu 1

##Site B
python flower_pCE_2D_v4_auxpCE_GatedCRFLoss.py --root_path ../data/FAZ_h5 --num_classes 2 --in_chns 1 --img_class faz --exp faz/FedLPPA --model unet_univ5 --max_iterations 30000 --iters 5 --eval_iters 5 --tsne_iters 200 --batch_size 12 --base_lr 0.01 --amp 0 --server_address 127.0.0.1:8091 --strategy FedUniV2.1 --min_num_clients 5 --img_size 256 --alpha 0.1 --beta 0.5 --prompt universal --attention dual --dual_init aggregated --label_prompt 1 --role client --cid 1 --client client2 --sup_type keypoint --gpu 2

##Site C
python flower_pCE_2D_v4_auxpCE_GatedCRFLoss.py --root_path ../data/FAZ_h5 --num_classes 2 --in_chns 1 --img_class faz --exp faz/FedLPPA --model unet_univ5 --max_iterations 30000 --iters 5 --eval_iters 5 --tsne_iters 200 --batch_size 12 --base_lr 0.01 --amp 0 --server_address 127.0.0.1:8091 --strategy FedUniV2.1 --min_num_clients 5 --img_size 256 --alpha 0.1 --beta 0.5 --prompt universal --attention dual --dual_init aggregated --label_prompt 1 --role client --cid 2 --client client3 --sup_type block --gpu 3

##Site D
python flower_pCE_2D_v4_auxpCE_GatedCRFLoss.py --root_path ../data/FAZ_h5 --num_classes 2 --in_chns 1 --img_class faz --exp faz/FedLPPA --model unet_univ5 --max_iterations 30000 --iters 5 --eval_iters 5 --tsne_iters 200 --batch_size 12 --base_lr 0.01 --amp 0 --server_address 127.0.0.1:8091 --strategy FedUniV2.1 --min_num_clients 5 --img_size 256 --alpha 0.1 --beta 0.5 --prompt universal --attention dual --dual_init aggregated --label_prompt 1 --role client --cid 3 --client client4 --sup_type box --gpu 4

##Site E
python flower_pCE_2D_v4_auxpCE_GatedCRFLoss.py --root_path ../data/FAZ_h5 --num_classes 2 --in_chns 1 --img_class faz --exp faz/FedLPPA --model unet_univ5 --max_iterations 30000 --iters 5 --eval_iters 5 --tsne_iters 200 --batch_size 12 --base_lr 0.01 --amp 0 --server_address 127.0.0.1:8091 --strategy FedUniV2.1 --min_num_clients 5 --img_size 256 --alpha 0.1 --beta 0.5 --prompt universal --attention dual --dual_init aggregated --label_prompt 1 --role client --cid 4 --client client5 --sup_type scribble --gpu 5
