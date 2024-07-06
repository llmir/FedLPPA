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
import sys
# from networks.efficientunet import UNet
from networks.net_factory import net_factory

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='../data/odoc', help='Name of Experiment')
parser.add_argument('--exp', type=str,
                    default='odoc/WeaklySeg_pCE_MumfordShah_Loss', help='experiment_name')
parser.add_argument('--model', type=str,
                    default='unet', help='model_name')
parser.add_argument('--client', type=str,
                    default='client1', help='client')
parser.add_argument('--num_classes', type=int,  default=3,
                    help='output channel of network')
parser.add_argument('--sup_type', type=str, default="scribble",
                    help='label')
parser.add_argument('--base_dir', type=str, default="../data/odoc",
                    help='dataset address')

def get_client_ids(self, client):
    client1_val_set = 'Domain1/val/'+pd.Series(os.listdir( self._base_dir+"/Domain1/val"))
    client1_training_set = 'Domain1/train/'+pd.Series(os.listdir( self._base_dir+"/Domain1/train"))
    client2_val_set = 'Domain2/val/'+pd.Series(os.listdir( self._base_dir+"/Domain2/val"))
    client2_training_set = 'Domain2/train/'+pd.Series(os.listdir( self._base_dir+"/Domain2/train"))
    client3_val_set = 'Domain3/val/'+pd.Series(os.listdir( self._base_dir+"/Domain3/val"))
    client3_training_set = 'Domain3/train/'+pd.Series(os.listdir( self._base_dir+"/Domain3/train"))
    client4_val_set = 'Domain4/val/'+pd.Series(os.listdir( self._base_dir+"/Domain4/val"))
    client4_training_set = 'Domain4/train/'+pd.Series(os.listdir( self._base_dir+"/Domain4/train"))
    client5_val_set = 'Domain5/val/'+pd.Series(os.listdir( self._base_dir+"/Domain5/val"))
    client5_training_set = 'Domain5/train/'+pd.Series(os.listdir( self._base_dir+"/Domain5/train"))
    client1_val_set = client1_val_set.tolist()
    client1_training_set = client1_training_set.tolist()
    client2_val_set = client2_val_set.tolist()
    client2_training_set = client2_training_set.tolist()
    client3_val_set = client3_val_set.tolist()
    client3_training_set = client3_training_set.tolist()
    client4_val_set = client4_val_set.tolist()
    client4_training_set = client4_training_set.tolist()
    client5_val_set = client5_val_set.tolist()
    client5_training_set = client5_training_set.tolist()
    
    if client == "client1":
        return [client1_training_set, client1_val_set]
    elif client == "client2":
        return [client2_training_set, client2_val_set]
    elif client == "client3":
        return [client3_training_set, client3_val_set]
    elif client == "client4":
        return [client4_training_set, client4_val_set]
    elif client == "client5":
        return [client5_training_set, client5_val_set]

    else:
        return "ERROR KEY"



def calculate_metric_percase(pred, gt, spacing):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    if pred.sum() > 0:
        dice = metric.binary.dc(pred, gt)
        asd = metric.binary.asd(pred, gt, voxelspacing=spacing)
        hd95 = metric.binary.hd95(pred, gt, voxelspacing=spacing)
        recall = metric.binary.recall(pred, gt)
        precision = metric.binary.precision(pred, gt)
        jc = metric.binary.jc(pred, gt)
        specificity = metric.binary.specificity(pred, gt)
        ravd = metric.binary.ravd(pred, gt)
        return dice, asd, hd95, recall, precision, jc, specificity, ravd
    else:
        return 0, 0, 0, 0, 0, 0,0,0


def test_single_volume(case, net, test_save_path, FLAGS):
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
                    net(input), dim=1), dim=1).squeeze(0)
        out = out.cpu().detach().numpy()
                # pred = zoom(
                #     out, (x / patch_size[0], y / patch_size[1]), order=0)
        prediction = out
            
    case = case.replace(".h5", "")
    case = case.replace("/test_h5", "/test/image")
    case2 = case.replace("/image", "/mask")

    org_img_path = "../data/odoc/{}.png".format(case)
    org_mask_itk = "../data/odoc/{}.png".format(case2)
    org_mask_itk = sitk.ReadImage(org_mask_itk)
    spacing = org_mask_itk.GetSpacing()

    first_metric = calculate_metric_percase(
        prediction == 1, label == 1, (spacing[0], spacing[1]))
    second_metric = calculate_metric_percase(
        prediction >= 1, label >= 1, (spacing[0], spacing[1]))
    

    # img_itk = sitk.GetImageFromArray(image.astype(np.float32))
    # img_itk.CopyInformation(org_img_itk)
    # prd_itk = sitk.GetImageFromArray(prediction.astype(np.uint8))
    # prd_itk.CopyInformation(org_mask_itk)
    # lab_itk = sitk.GetImageFromArray(label.astype(np.float32))
    # lab_itk.CopyInformation(org_mask_itk)
    # sitk.WriteImage(prd_itk, 'model_odoc2/odoc/WeaklySeg_pCE_Proposed_client4/scribble/scribble_predictions/'+case2.replace('/mask','/pre')+'.png')
    # sitk.WriteImage(img_itk, test_save_path + case + "_img.png")
    # sitk.WriteImage(lab_itk, test_save_path + case + "_gt.png")
    return first_metric, second_metric


def Inference(FLAGS):
    train_ids, test_ids = get_client_ids(FLAGS.client,FLAGS.base_dir)
    image_list = []
    image_list = test_ids
    snapshot_path = "../model_odoc_icctw/{}_{}/{}".format(
        FLAGS.exp, FLAGS.client, FLAGS.sup_type)
    test_save_path = "../model_odoc/{}_{}/{}/{}_predictions/".format(
        FLAGS.exp, FLAGS.client, FLAGS.sup_type, FLAGS.model)
    if os.path.exists(test_save_path):
        shutil.rmtree(test_save_path)
    os.makedirs(test_save_path)
    net = net_factory(net_type=FLAGS.model, in_chns=3,
                      class_num=FLAGS.num_classes)
    save_mode_path = os.path.join(
        snapshot_path, 'unet_best_model.pth')
    net.load_state_dict(torch.load(save_mode_path))
    print("init weight from {}".format(save_mode_path))
    net.eval()

    dice_total = 0.0
    asd_total = 0.0
    hd95_total = 0.0
    recall_total = 0.0
    precision_total = 0.0
    jc_total = 0.0
    specificity_total = 0.0
    ravd_total = 0.0
    for case in tqdm(image_list):
        print(case)
        dice, asd, hd95, recall, precision, jc, specificity, ravd = test_single_volume(
            case, net, test_save_path, FLAGS)
        dice_total += np.asarray(dice)
        asd_total += np.asarray(asd)
        hd95_total += np.asarray(hd95)
        recall_total += np.asarray(recall)
        precision_total += np.asarray(precision)
        jc_total += np.asarray(jc)
        specificity_total += np.asarray(specificity)
        ravd_total += np.asarray(ravd)
    target_metric = [dice_total, asd_total, hd95_total, recall_total, precision_total, jc_total, specificity_total, ravd_total]
    return target_metric


if __name__ == '__main__':
    FLAGS = parser.parse_args()
    total = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    snapshot_path="../model_odoc_icctw/{}_{}/{}".format(FLAGS.exp, FLAGS.client, FLAGS.sup_type)
    logging.basicConfig(filename=snapshot_path+"/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(FLAGS))
    for i in [1, 2, 3, 4, 5]:
        # for i in [5]:
        FLAGS.client = "client{}".format(i)
        print("Inference client{}".format(i))
        target = Inference(FLAGS)
        total += target
    print(total/5)
