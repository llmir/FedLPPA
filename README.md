# FedLPPA: Learning Personalized Prompt and Aggregation for Federated Weakly-supervised Medical Image Segmentation
The official implementation of the paper: [**FedLPPA: Learning Personalized Prompt and Aggregation for Federated Weakly-supervised Medical Image Segmentation**](https://arxiv.org/abs/2402.17502)
![TEL](image/framework.png)


# Datasets
## Download the Datasets


<table>
  <tbody>
    <tr>
      <td align="center">Fundus</td>
      <td align="center">OCTA</td>
      <td align="center">Endoscopy</td>
      <td align="center">Prostate MRI</td>
    </tr>
    <tr>
      <td align="center"><a href="https://github.com/llmir/FedICRA/tree/master/data">Download</a></td> 
      <td align="center"><a href="https://github.com/llmir/FedICRA/tree/master/data">Download</a></td>  
      <td align="center"><a href="https://drive.google.com/drive/folders/1GsaJXeE_5yNPmd4DLDx-POO2Qr24szLK?usp=drive_link">Download</a></td>  
      <td align="center"><a href="https://drive.google.com/drive/folders/1rCQMrzSKPcsHjAGrgJUqszXityyho4JH?usp=drive_link">Download</a></td>  
    </tr>

  </tbody>
</table>


## Details of Training set
The detailed information of datasets can be found in the [paper](https://arxiv.org/abs/2402.17502)

# Visualization Results

![TEL](image/output.png)

# Requirements
Some important required packages are lised below:
* Pytorch 1.10.2
* cudatoolkit 11.3.1
* efficientnet-pytorch 0.7.1
* tensorboardx 2.5.1
* medpy 0.4.0
* scikit-image 0.19.3
* simpleitk  2.1.1.2
* flwr 1.0.0
* Python >= 3.9
# Usage
## 1. Clone this project
``` bash
git clone https://github.com/llmir/FedLPPA.git
cd FedLPPA/code_v4
```

## 2. Create a conda environment
``` bash
conda env create -n fed39v2 -f fedlppa.yaml
conda activate fed39v2
```
## 3. Data Preparation
Download the datasets to the dir 'FedLPPA/data' in the form of '.h5'.


## 4. Train the model
We first disclose the FedAvg and FedLPPA in 'train.sh'. The other comparative methods' scripts will be included in subsequent updates.
``` bash
##Server
python flower_pCE_2D_v4_FedLPPA.py --root_path ../data/FAZ_h5 --num_classes 2 --in_chns 1 --img_class faz --exp faz/FedLPPA --model unet_univ5 --max_iterations 30000 --iters 5 --eval_iters 5 --tsne_iters 200 --batch_size 12 --base_lr 0.01 --amp 0 --server_address 127.0.0.1:8091 --strategy FedUniV2.1 --min_num_clients 5 --img_size 256 --alpha 0.1 --beta 0.5 --prompt universal --attention dual --dual_init aggregated --label_prompt 1 --role server --client client_all --sup_type mask --gpu 0

##Site A
python flower_pCE_2D_v4_FedLPPA.py --root_path ../data/FAZ_h5 --num_classes 2 --in_chns 1 --img_class faz --exp faz/FedLPPA --model unet_univ5 --max_iterations 30000 --iters 5 --eval_iters 5 --tsne_iters 200 --batch_size 12 --base_lr 0.01 --amp 0 --server_address 127.0.0.1:8091 --strategy FedUniV2.1 --min_num_clients 5 --img_size 256 --alpha 0.1 --beta 0.5 --prompt universal --attention dual --dual_init aggregated --label_prompt 1 --role client --cid 0 --client client1 --sup_type scribble_noisy --gpu 1

##Site B
python flower_pCE_2D_v4_FedLPPA.py --root_path ../data/FAZ_h5 --num_classes 2 --in_chns 1 --img_class faz --exp faz/FedLPPA --model unet_univ5 --max_iterations 30000 --iters 5 --eval_iters 5 --tsne_iters 200 --batch_size 12 --base_lr 0.01 --amp 0 --server_address 127.0.0.1:8091 --strategy FedUniV2.1 --min_num_clients 5 --img_size 256 --alpha 0.1 --beta 0.5 --prompt universal --attention dual --dual_init aggregated --label_prompt 1 --role client --cid 1 --client client2 --sup_type keypoint --gpu 2

##Site C
python flower_pCE_2D_v4_FedLPPA.py --root_path ../data/FAZ_h5 --num_classes 2 --in_chns 1 --img_class faz --exp faz/FedLPPA --model unet_univ5 --max_iterations 30000 --iters 5 --eval_iters 5 --tsne_iters 200 --batch_size 12 --base_lr 0.01 --amp 0 --server_address 127.0.0.1:8091 --strategy FedUniV2.1 --min_num_clients 5 --img_size 256 --alpha 0.1 --beta 0.5 --prompt universal --attention dual --dual_init aggregated --label_prompt 1 --role client --cid 2 --client client3 --sup_type block --gpu 3

##Site D
python flower_pCE_2D_v4_FedLPPA.py --root_path ../data/FAZ_h5 --num_classes 2 --in_chns 1 --img_class faz --exp faz/FedLPPA --model unet_univ5 --max_iterations 30000 --iters 5 --eval_iters 5 --tsne_iters 200 --batch_size 12 --base_lr 0.01 --amp 0 --server_address 127.0.0.1:8091 --strategy FedUniV2.1 --min_num_clients 5 --img_size 256 --alpha 0.1 --beta 0.5 --prompt universal --attention dual --dual_init aggregated --label_prompt 1 --role client --cid 3 --client client4 --sup_type box --gpu 4

##Site E
python flower_pCE_2D_v4_FedLPPA.py --root_path ../data/FAZ_h5 --num_classes 2 --in_chns 1 --img_class faz --exp faz/FedLPPA --model unet_univ5 --max_iterations 30000 --iters 5 --eval_iters 5 --tsne_iters 200 --batch_size 12 --base_lr 0.01 --amp 0 --server_address 127.0.0.1:8091 --strategy FedUniV2.1 --min_num_clients 5 --img_size 256 --alpha 0.1 --beta 0.5 --prompt universal --attention dual --dual_init aggregated --label_prompt 1 --role client --cid 4 --client client5 --sup_type scribble --gpu 5
```
- root_path: The dataset root path.
- num_classes: The segmentation classes.
- batch_size: 12.
- image_size: Default value is 256.
- exp: save_path of models and 'log' file.
- server_address: Server communication port. If you train the server model and client models in one server, please set it to the similar format '127.0.0.1:8091' (
note that different experiments cannot use the same port).
- strategy: Choose a federated learning strategy, i.e., FedAvg, FedBN, FedRep and FedLPPA (FedUniv2.1).
- prompt: employ learnable prompts or one-hot. Two formats 'universal' or 'onehot'.
- attention: attention module selection.
- dual_init: Select a aggregated strategy.
- min_num_clients: The total number of clients.
- label_prompt: 0 or 1 , use or not use the sparse label prompts.
- role: 'Server' or 'Client'.
- cid: client_id.
- sup_type: Choose the format of sparse annotation.

## 5. Test the model

``` bash
python -u test_client4onemod_FL_Personalize.py --client client1 --num_classes 2 --in_chns 1 --root_path ../data/FAZ_h5/test/ --img_class faz --exp faz/ --min_num_clients 5 --cid 1 --model unet_univ5
```
Other samples can be found in [here](https://github.com/llmir/FedLPPA/blob/master/code_v4/test.sh).

# Acknowledgement
* [flower](https://github.com/mher/flower)
* [WSL4MIS](https://github.com/HiLab-git/WSL4MIS)
* [FedALA](https://github.com/TsingZ0/FedALA)
* [FedLC](https://github.com/jcwang123/FedLC)

# Citation
If you find FedLPPA useful in your research, please consider citing:
```
@article{lin2024fedlppa,
  title={FedLPPA: Learning Personalized Prompt and Aggregation for Federated Weakly-supervised Medical Image Segmentation},
  author={Lin, Li and Liu, Yixiang and Wu, Jiewei and Cheng, Pujin and Cai, Zhiyuan and Wong, Kenneth KY and Tang, Xiaoying},
  journal={arXiv preprint arXiv:2402.17502},
  year={2024}
}
```
If you have any questions, please feel free to contact us.

  

