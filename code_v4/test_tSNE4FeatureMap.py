import argparse
from email.mime import image
import os
import re
import shutil
from tkinter import image_types
import pandas as pd
import h5py
import nibabel as nib
import numpy as np
import SimpleITK as sitk
import torch, cv2
from medpy import metric
import random
# from scipy.ndimage import zoom
# from scipy.ndimage.interpolation import zoom
# from dataloaders.dataset import BaseDataSets, BaseDataSets_octa, RandomGenerator, RandomGeneratorv2, BaseDataSets_octasyn, RandomGeneratorv3, BaseDataSets_octasynv2, RandomGeneratorv4, train_aug, val_aug, BaseDataSets_octawithback, BaseDataSets_cornwithback,BaseDataSets_drivewithback,BaseDataSets_chasesyn, BaseDataSets_chasewithbackori, BaseDataSets_chasewithback
from tqdm import tqdm
from scipy import signal
import torch.nn.functional as F
import torch
from sklearn.manifold import TSNE
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import rcParams
import matplotlib.pyplot as plt
import seaborn as sns


# from networks.efficientunet import UNet
from networks.net_factory import net_factory

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='../data/ODOC_h5', help='Name of Experiment')
parser.add_argument('--exp', type=str,
                    default='odoc_bs_12/pCE_Unet', help='experiment_name')
parser.add_argument('--model', type=str,
                    default='unet', help='model_name')
parser.add_argument('--client', type=str,
                    default='client5', help='client')

parser.add_argument('--data_type', type=str,
                    default='octa', help='data type')
parser.add_argument('--num_classes', type=int,  default=3,
                    help='output channel of network')
parser.add_argument('--in_chns', type=int, default=3,
                    help='image channel')
parser.add_argument('--sup_type', type=str, default="scribble",
                    help='label')
parser.add_argument('--snapshot_path', type=str, default="../model_odoc_icctw/odoc_bs_12/odoc_LL_block_client5",
                    help='snapshot_path')
# parser.add_argument('--save_mode_path', type=str, default="iter_176_dice_0.8538.pth", help='save_mode_path')
parser.add_argument('--img_class', type=str,
                    default='faz', help='the img class(odoc or faz)')
parser.add_argument('--min_num_clients', type=int, default=5,help='min_num_client')
parser.add_argument('--cid', type=int, default=0,help='cid')
def get_client_ids(client,base_dir):
    client1_test_set = 'Domain1/test/'+pd.Series(os.listdir(base_dir+"/Domain1/test"))
    client1_training_set = 'Domain1/train/'+pd.Series(os.listdir(base_dir+"/Domain1/train"))
    client2_test_set = 'Domain2/test/'+pd.Series(os.listdir(base_dir+"/Domain2/test"))
    client2_training_set = 'Domain2/train/'+pd.Series(os.listdir(base_dir+"/Domain2/train"))
    client3_test_set = 'Domain3/test/'+pd.Series(os.listdir(base_dir+"/Domain3/test"))
    client3_training_set = 'Domain3/train/'+pd.Series(os.listdir(base_dir+"/Domain3/train"))
    client4_test_set = 'Domain4/test/'+pd.Series(os.listdir(base_dir+"/Domain4/test"))
    client4_training_set = 'Domain4/train/'+pd.Series(os.listdir(base_dir+"/Domain4/train"))
    client5_test_set = 'Domain5/test/'+pd.Series(os.listdir(base_dir+"/Domain5/test"))
    client5_training_set = 'Domain5/train/'+pd.Series(os.listdir(base_dir+"/Domain5/train"))
    client1_test_set = client1_test_set.tolist()
    client1_training_set = client1_training_set.tolist()
    client2_test_set = client2_test_set.tolist()
    client2_training_set = client2_training_set.tolist()
    client3_test_set = client3_test_set.tolist()
    client3_training_set = client3_training_set.tolist()
    client4_test_set = client4_test_set.tolist()
    client4_training_set = client4_training_set.tolist()
    client5_test_set = client5_test_set.tolist()
    client5_training_set = client5_training_set.tolist()

    if client == "client0":
        return [client1_training_set, client1_test_set]
    elif client == "client1":
        return [client2_training_set, client2_test_set]
    elif client == "client2":
        return [client3_training_set, client3_test_set]
    elif client == "client3":
        return [client4_training_set, client4_test_set]
    elif client == "client4":
        return [client5_training_set, client5_test_set]
    elif client == 'client_all':
        client_train_total=[]
        client_test_total=[]
        for i in client1_training_set:
            client_train_total.append(i)
        for i in client2_training_set:
            client_train_total.append(i)
        for i in client3_training_set:
            client_train_total.append(i)
        for i in client4_training_set:
            client_train_total.append(i)
        for i in client5_training_set:
            client_train_total.append(i)
        for n in client1_test_set:
            client_test_total.append(n)
        for n in client2_test_set:
            client_test_total.append(n)
        for n in client3_test_set:
            client_test_total.append(n)
        for n in client4_test_set:
            client_test_total.append(n)
        for n in client5_test_set:
            client_test_total.append(n)
        return[client_train_total, client_test_total]
        

    else:
        return "ERROR KEY"


def t_sne(n_components,data,label,list,FLAGS):
    params={'font.family':'',
        'font.serif':'',
        'font.style':'normal',
        'font.weight':'normal', 
        'font.size':11,
        }
    rcParams.update(params)
    if n_components == 2:
        tsne = TSNE(n_components=n_components, perplexity=10,random_state=90, n_iter=1000)
        z = tsne.fit_transform(data)
        # print(z)
        # print(isinstance(z,np.array()))
        fig = plt.figure(figsize=(6,6))
        plt.subplot(111)

        df = pd.DataFrame(z)
        df['label'] = label
        df['list'] = list
        # palet = sns.color_palette("hls",2)
        # flatui = ['#f3a598', '#faaf42','#480080','#0fa14a','#7a7c7f']
        flatui = ['#f3a598', '#66c7df','#faaf42','#9659ef','#11b855']
        palet = sns.color_palette(flatui)
        markers = {"Site A": "o", "Site B": "v","Site C": "H","Site D": "s","Site E":"^"}
        alpha = 0.8
        # size = {"OCTA500": 1, "Synthetic OCTA": 0.5}
    
        
        sns.scatterplot(x=z[:,0], y=z[:,1], hue=list, style=list, markers=markers, linewidth = 0.1, 
                palette=palet, alpha=alpha,
                data=df)
        # plt.savefig("t-SNE.png")
        # odoc
        plt.xlim(-150, 150)
        plt.ylim(-150, 150)
        # plt.xlim(-12, 17)
        # plt.ylim(-8,8)
        # faz
        # plt.xlim(-15, 18)
        # plt.ylim(-11,11)
        plt.legend(loc='lower right')
        fig.savefig('./{}.pdf'.format((FLAGS.exp).split("/")[-1]),dpi=700,format='pdf',bbox_inches='tight')
        # print(df)
    elif n_components == 3:
        tsne = TSNE(n_components=n_components, random_state=122, n_iter=2000)
        z = tsne.fit_transform(data)
        fig = plt.figure()
        ax = Axes3D(fig)
        ax.scatter(z[:, 0], z[:, 1], z[:, 2], c=label)
        plt.title("t-SNE Digits")
        plt.savefig("t-SNE_3d.png")
    else:
        print("The value of n_components can only be 2 or 3")
        
    plt.show()

def get_client_ids_polyp(client,base_dir):
    client1_test_set = 'Domain1/test/'+pd.Series(os.listdir(base_dir+"/Domain1/test"))
    client1_training_set = 'Domain1/train/'+pd.Series(os.listdir(base_dir+"/Domain1/train"))
    client2_test_set = 'Domain2/test/'+pd.Series(os.listdir(base_dir+"/Domain2/test"))
    client2_training_set = 'Domain2/train/'+pd.Series(os.listdir(base_dir+"/Domain2/train"))
    client3_test_set = 'Domain3/test/'+pd.Series(os.listdir(base_dir+"/Domain3/test"))
    client3_training_set = 'Domain3/train/'+pd.Series(os.listdir(base_dir+"/Domain3/train"))
    client4_test_set = 'Domain4/test/'+pd.Series(os.listdir(base_dir+"/Domain4/test"))
    client4_training_set = 'Domain4/train/'+pd.Series(os.listdir(base_dir+"/Domain4/train"))
    client1_test_set = client1_test_set.tolist()
    client1_training_set = client1_training_set.tolist()
    client2_test_set = client2_test_set.tolist()
    client2_training_set = client2_training_set.tolist()
    client3_test_set = client3_test_set.tolist()
    client3_training_set = client3_training_set.tolist()
    client4_test_set = client4_test_set.tolist()
    client4_training_set = client4_training_set.tolist()

    if client == "client0":
        return [client1_training_set, client1_test_set]
    elif client == "client1":
        return [client2_training_set, client2_test_set]
    elif client == "client2":
        return [client3_training_set, client3_test_set]
    elif client == "client3":
        return [client4_training_set, client4_test_set]
    elif client == 'client_all':
        client_train_total=[]
        client_test_total=[]
        for i in client1_training_set:
            client_train_total.append(i)
        for i in client2_training_set:
            client_train_total.append(i)
        for i in client3_training_set:
            client_train_total.append(i)
        for i in client4_training_set:
            client_train_total.append(i)
        for n in client1_test_set:
            client_test_total.append(n)
        for n in client2_test_set:
            client_test_total.append(n)
        for n in client3_test_set:
            client_test_total.append(n)
        for n in client4_test_set:
            client_test_total.append(n)
        return[client_train_total, client_test_total]
    else:
        return "ERROR KEY"
def calculate_metric_percase(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    if pred.sum() > 0:
        dice = metric.binary.dc(pred, gt)
        jaccard = metric.binary.jc(pred, gt)
        hd95 = metric.binary.hd95(pred, gt)
        assd = metric.binary.assd(pred, gt)
        se = metric.binary.sensitivity(pred, gt)
        sp = metric.binary.specificity(pred, gt)
        recall = metric.binary.recall(pred, gt)
        precision = metric.binary.precision(pred, gt)
        return dice, jaccard, hd95, assd, se, sp, recall, precision
    else:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

def test_single_image(case, net, test_save_path, FLAGS):

    h5f = h5py.File(FLAGS.root_path +
                            "/{}".format(case), 'r')
    image = h5f['image'][:]
    label = h5f['mask'][:]
    prediction = np.zeros_like(label)

    if len(image.shape) == 3:
        prediction = np.zeros_like(label)
        
        slice = image
        input = torch.from_numpy(slice).unsqueeze(
                0).float().cuda()
        net.eval()
        with torch.no_grad():
            out = torch.argmax(torch.softmax(
                net(input)[0], dim=1), dim=1).squeeze(1).squeeze(0)
            out = out.cpu().detach().numpy()
            # featuremap层数
            feature_lc = net(input)[1][-1]
            feature_lc = torch.mean(feature_lc,dim=1,keepdim=False)
            print(feature_lc.shape)
            feature_lc = feature_lc.squeeze(0).cpu().detach().numpy()
    # ###faz val
    elif len(image.shape) == 2:
        prediction = np.zeros_like(label)
        
        slice = image
        input = torch.from_numpy(slice).unsqueeze(
            0).unsqueeze(0).float().cuda()
        net.eval()
        with torch.no_grad():
            out = torch.argmax(torch.softmax(
                net(input)[0], dim=1), dim=1).squeeze(1).squeeze(0)
            out = out.cpu().detach().numpy()
            feature_lc = net(input)[1][-1]
            feature_lc = torch.mean(feature_lc,dim=1,keepdim=False)
            print(feature_lc.shape)
            feature_lc = feature_lc.squeeze(0).cpu().detach().numpy()
            print(feature_lc.shape)


    return feature_lc
  

def Inference(FLAGS):
    if FLAGS.img_class == 'odoc' or FLAGS.img_class == 'faz':
        train_ids, test_ids = get_client_ids(FLAGS.client,FLAGS.root_path)
    elif FLAGS.img_class == 'polyp':
        train_ids, test_ids = get_client_ids_polyp(FLAGS.client,FLAGS.root_path)
    elif FLAGS.img_class == 'prostate':
        train_ids, test_ids = get_client_ids_prostate(FLAGS.client,FLAGS.root_path)
    image_list = []
    feature_lc = []
    image_list = test_ids
    snapshot_path = "../model/{}/".format(
        FLAGS.exp)
    test_save_path = "../model/{}_test/{}/".format(
        FLAGS.exp,FLAGS.client)
    if os.path.exists(test_save_path):
        shutil.rmtree(test_save_path)
    os.makedirs(test_save_path+'/pre/')
    os.makedirs(test_save_path+'/feature_lc/')
    import torch.backends.cudnn as cudnn
    cudnn.benchmark = False
    cudnn.deterministic = True
    net = net_factory(FLAGS,net_type=FLAGS.model, in_chns=FLAGS.in_chns,
                      class_num=FLAGS.num_classes)
    save_mode_path = os.path.join(
        snapshot_path, '{}_async_{}_best_model.pth'.format(FLAGS.client,FLAGS.model).replace("client","client_"))
    # save_mode_path = os.path.join(
    #     snapshot_path, '{}/{}_best_model.pth'.format(FLAGS.client,FLAGS.model))
    # save_mode_path = os.path.join(
    #     snapshot_path, '{}_best_model.pth'.format(FLAGS.model))
    net.load_state_dict(torch.load(save_mode_path))
    print("init weight from {}".format(save_mode_path))
    net.eval()

    
    if FLAGS.img_class=='faz' or FLAGS.img_class=='polyp' or FLAGS.img_class=='prostate':
        for case in tqdm(image_list):
            print(case)
            img_feature = test_single_image(case, net, test_save_path, FLAGS)
            feature_lc.append(img_feature)
            
        
            
    if FLAGS.img_class=='odoc':
        for case in tqdm(image_list):
            print(case)
            img_feature = test_single_image(case, net, test_save_path, FLAGS)
            feature_lc.append(img_feature)

    return feature_lc



if __name__ == '__main__':
    
    FLAGS = parser.parse_args()
    total = 0.0
    os.environ['CUDA_VISIBLE_DEVICES'] = '3'
    seed = 2022 
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  
    np.random.seed(seed)  
    random.seed(seed)  
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    data = []
    # for i in range(5):
    #     data = Inference(FLAGS)
    
    # print(total)
    
    for i in range(5):
        # for i in [5]:
        FLAGS.client = "client{}".format(i)
        FLAGS.cid = i
        print("Inference client{}".format(i))
        feature_lc = Inference(FLAGS)
        data.append(feature_lc)
    # print(total/5)
    # data = np.array([data[i] for i in range(5)])
    x = np.concatenate((data[0],data[1],data[2],data[3],data[4]),axis=0)
    print("data.shape",x.shape)
    label = np.array([1]*np.array(data[0]).shape[0]+[2]*np.array(data[1]).shape[0]+[3]*np.array(data[2]).shape[0]+[4]*np.array(data[3]).shape[0]+[5]*np.array(data[4]).shape[0])
    list = ['Site A']*np.array(data[0]).shape[0]+['Site B']*(np.array(data[1]).shape[0])+['Site C']*(np.array(data[2]).shape[0])+['Site D']*(np.array(data[3]).shape[0])+['Site E']*(np.array(data[4]).shape[0])
    data = x.reshape(x.shape[0],x.shape[1]*x.shape[2])
    t_sne(2,data,label,list,FLAGS)
