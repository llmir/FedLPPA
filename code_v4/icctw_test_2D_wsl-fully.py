import argparse
import os
import re
import shutil
import numpy as np
import pandas as pd
import h5py
import nibabel as nib
import numpy as np
import SimpleITK as sitk
import torch
from medpy import metric
from scipy.ndimage import zoom
from scipy.ndimage.interpolation import zoom
from tqdm import tqdm
import logging
from tensorboardX import SummaryWriter
# from networks.efficientunet import UNet
from networks.net_factory import net_factory

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='../data/odoc', help='Name of Experiment')
parser.add_argument('--exp', type=str,
                    default='odoc_thin/WeaklySeg_pCE_Proposed', help='experiment_name')
parser.add_argument('--model', type=str,
                    default='unet_cct', help='model_name')
parser.add_argument('--fold', type=str,
                    default='fold4', help='fold')
parser.add_argument('--num_classes', type=int,  default=3,
                    help='output channel of network')
parser.add_argument('--sup_type', type=str, default="scribble",
                    help='label')
parser.add_argument('--base_dir', type=str, default="../data/odoc",
                    help='dataset address')
tensorboardx_path="../model_odoc_batchsize_12_test/{}_{}/{}".format(parser.parse_args().exp,parser.parse_args().fold,parser.parse_args().sup_type).replace('fold','Domain')
if os.path.exists(tensorboardx_path):
        shutil.rmtree(tensorboardx_path)
os.makedirs(tensorboardx_path)
writer = SummaryWriter(tensorboardx_path + '/log')
def get_fold_ids(fold, base_dir):
        
    domain1_testing_set = 'Domain1/test_h5/'+pd.Series(os.listdir(base_dir+"/Domain1/test_h5"))
    domain1_training_set = 'Domain1/train/'+pd.Series(os.listdir(base_dir+"/Domain1/train"))
    domain2_testing_set = 'Domain2/test_h5/'+pd.Series(os.listdir(base_dir+"/Domain2/test_h5"))
    domain2_training_set = 'Domain2/train/'+pd.Series(os.listdir(base_dir+"/Domain2/train"))
    domain3_testing_set = 'Domain3/test_h5/'+pd.Series(os.listdir(base_dir+"/Domain3/test_h5"))
    domain3_training_set = 'Domain3/train/'+pd.Series(os.listdir(base_dir+"/Domain3/train"))
    domain4_testing_set = 'Domain4/test_h5/'+pd.Series(os.listdir(base_dir+"/Domain4/test_h5"))
    domain4_training_set = 'Domain4/train/'+pd.Series(os.listdir(base_dir+"/Domain4/train"))
    fold1_testing_set = domain1_testing_set.tolist()
    fold1_training_set = domain1_training_set.tolist()
    fold2_testing_set = domain2_testing_set.tolist()
    fold2_training_set = domain2_training_set.tolist()
    fold3_testing_set = domain3_testing_set.tolist()
    fold3_training_set = domain3_training_set.tolist()
    fold4_testing_set = domain4_testing_set.tolist()
    fold4_training_set = domain4_training_set.tolist()
        
    if fold == "fold1":
        return [fold1_training_set, fold1_testing_set]
    elif fold == "fold2":
        return [fold2_training_set, fold2_testing_set]
    if fold == "fold3":
        return [fold3_training_set, fold3_testing_set]
    elif fold == "fold4":
        return [fold4_training_set, fold4_testing_set]
       
    else:
        return "ERROR KEY"
       


def calculate_metric_percase(pred, gt, spacing):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    dice = metric.binary.dc(pred, gt)
    asd = metric.binary.asd(pred, gt, voxelspacing=spacing)
    hd95 = metric.binary.hd95(pred, gt, voxelspacing=spacing)
    return dice, hd95, asd


def test_single_volume(case, net, test_save_path, FLAGS, iter_num):
    h5f = h5py.File(FLAGS.root_path +
                            "/{}".format(case), 'r')
    image = h5f['image'][:]
    label = h5f['label'][:]
    prediction = np.zeros_like(label)

    slice = image[:, :, :]
            # x, y = slice.shape[0], slice.shape[1]
            # slice = zoom(
            #     slice, (patch_size[0] / x, patch_size[1] / y), order=0)
    input = torch.from_numpy(slice).unsqueeze(
                0).float().cuda()
    net.eval()
    with torch.no_grad():
        out = torch.argmax(torch.softmax(
                    net(input)[0], dim=1), dim=1).squeeze(0)
        out = out.cpu().detach().numpy()
                # pred = zoom(
                #     out, (x / patch_size[0], y / patch_size[1]), order=0)
        prediction = out
        writer.add_image('test/Image_{}'.format(FLAGS.fold).replace('fold','Domain'),
                                 image, iter_num, dataformats='CHW')
        writer.add_image('test/Label_{}'.format(FLAGS.fold).replace('fold','Domain'),
                                 label * 50, iter_num, dataformats='HW')
        writer.add_image('test/Prediction_{}'.format(FLAGS.fold).replace('fold','Domain'),
                                 out * 50, iter_num, dataformats='HW')
        
    case = case.replace(".h5", "")
    case = case.replace("/test_h5", "/test/image")
    case2 = case.replace("/image", "/mask")

    org_img_path = "../data/odoc/{}.png".format(case)
    org_mask_itk = "../data/odoc/{}.png".format(case2)
    org_mask_itk = sitk.ReadImage(org_mask_itk)
    spacing = org_mask_itk.GetSpacing()
    if(np.any(prediction==1)):
        first_metric = calculate_metric_percase(
            prediction == 1, label == 1, (spacing[0], spacing[1]))
        second_metric = calculate_metric_percase(
            prediction >= 1, label >= 1, (spacing[0], spacing[1]))  
        return first_metric, second_metric
    else:
        print("bad_img",case)
        first_metric = calculate_metric_percase(
            prediction == 2, label == 2, (spacing[0], spacing[1]))
        second_metric = calculate_metric_percase(
            prediction >= 2, label >= 2, (spacing[0], spacing[1]))  
        return first_metric, second_metric
    
        

    # img_itk = sitk.GetImageFromArray(image.astype(np.float32))
    # img_itk.CopyInformation(org_img_itk)
    # prd_itk = sitk.GetImageFromArray(prediction.astype(np.uint8))
    # prd_itk.CopyInformation(org_mask_itk)
    # lab_itk = sitk.GetImageFromArray(label.astype(np.float32))
    # lab_itk.CopyInformation(org_mask_itk)
    # sitk.WriteImage(prd_itk, 'model_odoc2/odoc/WeaklySeg_pCE_Proposed_fold4/scribble/scribble_predictions/'+case2.replace('/mask','/pre')+'.png')
    # sitk.WriteImage(img_itk, test_save_path + case + "_img.png")
    # sitk.WriteImage(lab_itk, test_save_path + case + "_gt.png")
    


def Inference(FLAGS):
    train_ids, test_ids = get_fold_ids(FLAGS.fold,FLAGS.base_dir)
    image_list = []
    image_list = test_ids
    snapshot_path = "../model_odoc_icctw/{}_{}/{}".format(
        FLAGS.exp, FLAGS.fold, FLAGS.sup_type).replace('fold','domain')
    
    test_save_path = "../model_odoc_batsize_12_test/{}_{}/{}/{}_predictions/".format(
        FLAGS.exp, FLAGS.fold, FLAGS.sup_type, FLAGS.model)
    if os.path.exists(test_save_path):
        shutil.rmtree(test_save_path)
    os.makedirs(test_save_path)
    
    net = net_factory(net_type=FLAGS.model, in_chns=3,
                      class_num=FLAGS.num_classes)
    save_mode_path = os.path.join(
        snapshot_path, 'unet_cct_best_model.pth')
    net.load_state_dict(torch.load(save_mode_path))
    print("init weight from {}".format(save_mode_path))
    net.eval()
    
    
    first_total = 0.0
    second_total = 0.0
    third_total = 0.0
    num=1
    for case in tqdm(image_list):
        print(case)
        first_metric, second_metric = test_single_volume(
            case, net, test_save_path, FLAGS, iter_num=num)
        first_total += np.asarray(first_metric)
        second_total += np.asarray(second_metric)
        writer.add_scalar('info/{}_dice_cup'.format(FLAGS.fold).replace('fold','Domain'), first_metric[0], num)
        writer.add_scalar('info/{}_dice_disc'.format(FLAGS.fold).replace('fold','Domain'), second_metric[0],num)
        writer.add_scalar('info/{}_dice_mean'.format(FLAGS.fold).replace('fold','Domain'), (first_metric[0]+second_metric[0])/2,num)
        num+=1
    avg_metric = [first_total / len(image_list), second_total /
                  len(image_list)]
    
    print(avg_metric)
    print((avg_metric[0] + avg_metric[1])/ 2)
    return ((avg_metric[0] + avg_metric[1]) / 2)[0]


if __name__ == '__main__':
    FLAGS = parser.parse_args()
    total = 0.0
    for i in [1,3,4]:
        # for i in [5]:
        FLAGS.fold = "fold{}".format(i)
        print("Inference fold{}".format(i))
        mean_dice = Inference(FLAGS)

        total += mean_dice
    writer.close()
    print(total/3)
