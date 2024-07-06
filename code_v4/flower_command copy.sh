# Sever
python flower_train_weakly_supervised_pCE_2D.py --role server --min_num_clients 2 --fold fold1 --num_classes 4 --root_path ../data/ACDC --exp ACDC/WeaklySeg_pCE_debug --max_iterations 60000 --batch_size 12 --gpu 0
# Client 1
python flower_train_weakly_supervised_pCE_2D.py --role client --cid 0 --fold fold1 --num_classes 4 --root_path ../data/ACDC --exp ACDC/WeaklySeg_pCE_debug --max_iterations 60000 --batch_size 12 --gpu 0
# Client 2
python flower_train_weakly_supervised_pCE_2D.py --role client --cid 1 --fold fold1 --num_classes 4 --root_path ../data/ACDC --exp ACDC/WeaklySeg_pCE_debug --max_iterations 60000 --batch_size 12 --gpu 0


# Sever
python flower_train_weakly_supervised_pCE_TreeEnergyLoss_2D.py --role server --server_address [::]:8080 --min_num_clients 2 --fold fold1 --num_classes 4 --root_path ../data/ACDC --exp ACDC/WeaklySeg_pCE_TreeEnergyLoss_2D --max_iterations 60000 --batch_size 12 --gpu 2
# Client 1
python flower_train_weakly_supervised_pCE_TreeEnergyLoss_2D.py --role client --server_address [::]:8080 --cid 0 --fold fold1 --num_classes 4 --root_path ../data/ACDC --exp ACDC/WeaklySeg_pCE_TreeEnergyLoss_2D --max_iterations 60000 --batch_size 12 --gpu 2
# Client 2
python flower_train_weakly_supervised_pCE_TreeEnergyLoss_2D.py --role client --server_address [::]:8080 --cid 1 --fold fold1 --num_classes 4 --root_path ../data/ACDC --exp ACDC/WeaklySeg_pCE_TreeEnergyLoss_2D --max_iterations 60000 --batch_size 12 --gpu 2


# Server
python flower_train_weakly_supervised_pCE_MScaleTreeEnergyLoss_ADD.py --role server --server_address [::]:8081 --min_num_clients 2 --fold fold1 --num_classes 4 --root_path ../data/ACDC --exp ACDC/WeaklySeg_pCE_MScaleTreeEnergyLoss_ADD --max_iterations 60000 --batch_size 6 --gpu 0
# Client 1
python flower_train_weakly_supervised_pCE_MScaleTreeEnergyLoss_ADD.py --role client --server_address [::]:8081 --cid 0 --fold fold1 --num_classes 4 --root_path ../data/ACDC --exp ACDC/WeaklySeg_pCE_MScaleTreeEnergyLoss_ADD --max_iterations 60000 --batch_size 6 --gpu 0
# Client 2
python flower_train_weakly_supervised_pCE_MScaleTreeEnergyLoss_ADD.py --role client --server_address [::]:8081 --cid 1 --fold fold1 --num_classes 4 --root_path ../data/ACDC --exp ACDC/WeaklySeg_pCE_MScaleTreeEnergyLoss_ADD --max_iterations 60000 --batch_size 6 --gpu 0


# Server
python flower_train_weakly_supervised_pCE_MScaleTreeEnergyLoss_Recurve.py --role server --server_address [::]:8082 --min_num_clients 2 --fold fold1 --num_classes 4 --root_path ../data/ACDC --exp ACDC/WeaklySeg_pCE_MScaleTreeEnergyLoss_Recurve --max_iterations 60000 --batch_size 6 --gpu 3
# Client 1
python flower_train_weakly_supervised_pCE_MScaleTreeEnergyLoss_Recurve.py --role client --server_address [::]:8082 --cid 0 --fold fold1 --num_classes 4 --root_path ../data/ACDC --exp ACDC/WeaklySeg_pCE_MScaleTreeEnergyLoss_Recurve --max_iterations 60000 --batch_size 6 --gpu 3
# Client 2
python flower_train_weakly_supervised_pCE_MScaleTreeEnergyLoss_Recurve.py --role client --server_address [::]:8082 --cid 1 --fold fold1 --num_classes 4 --root_path ../data/ACDC --exp ACDC/WeaklySeg_pCE_MScaleTreeEnergyLoss_Recurve --max_iterations 60000 --batch_size 6 --gpu 3

