## faz + pCE
python flower_runner.py --port 8091 --procedure flower_pCE_2D --exp faz/WeaklySeg_pCE --base_lr 0.01 --img_class faz --model unet --gpus 0 0 0 0 0 0

## faz + pCE + single scale tree
python flower_runner.py --port 8092 --procedure flower_pCE_TreeEnergyLoss_2D --exp faz/WeaklySeg_pCE_TreeEnergyLoss --base_lr 0.01 --img_class faz --model unet_head --gpus 0 0 0 1 1 1
## faz + pCE + multi scale tree (add)
python flower_runner.py --port 8093 --procedure flower_pCE_MScaleTreeEnergyLoss_ADD --exp faz/WeaklySeg_pCE_MScaleTreeEnergyLoss_ADD --base_lr 0.01 --img_class faz --model unet_head --gpus 0 0 1 1 2 2

## faz + pCE + multi scale tree (recurve)
python flower_runner.py --port 8094 --procedure flower_pCE_MScaleTreeEnergyLoss_Recurve --exp faz/WeaklySeg_pCE_MScaleTreeEnergyLoss_Recurve --base_lr 0.01 --img_class faz --model unet_head --gpus 0 0 1 1 2 2

## odoc + pCE
python flower_runner.py --port 8095 --procedure flower_pCE_2D --exp odoc/WeaklySeg_pCE --base_lr 0.01 --img_class odoc --model unet --gpus 0 0 1 1 2 2

## odoc + pCE + single scale tree
python flower_runner.py --port 8096 --procedure flower_pCE_TreeEnergyLoss_2D --exp odoc/WeaklySeg_pCE_TreeEnergyLoss --base_lr 0.01 --img_class odoc --model unet_head --gpus 0 0 1 1 2 2

## odoc + pCE + multi scale tree (add)
python flower_runner.py --port 8097 --procedure flower_pCE_MScaleTreeEnergyLoss_ADD --exp odoc/WeaklySeg_pCE_MScaleTreeEnergyLoss_ADD --base_lr 0.01 --img_class odoc --model unet_head --gpus 0 1 2 3 4 5

## odoc + pCE + multi scale tree (recurve)
python flower_runner.py --port 8097 --procedure flower_pCE_MScaleTreeEnergyLoss_Recurve --exp odoc/WeaklySeg_pCE_MScaleTreeEnergyLoss_Recurve --base_lr 0.01 --img_class odoc --model unet_head --gpus 0 1 2 3 4 5


python flower_runner.py --port 8094 --procedure flower_pCE_2D --exp polyp/FL_WeaklySeg_pCE_flower_FedLC_alpha0.1 --base_lr 0.001 --img_class polyp --model unet_lc --gpus 5 6 6 7 7 --iters 5 --eval_iters 5 --rep_iters 2 --alpha 0.1 --strategy FedLC  --img_size 384