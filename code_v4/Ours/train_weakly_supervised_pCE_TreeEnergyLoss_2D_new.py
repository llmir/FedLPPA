import argparse
import logging
import os
import random
import shutil
import sys
import time

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.nn import BCEWithLogitsLoss
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import make_grid
from tqdm import tqdm

from dataloaders import utils
from dataloaders.dataset import BaseDataSets, BaseDataSets_octa, RandomGenerator, RandomGeneratorv2, BaseDataSets_octasyn, RandomGeneratorv3, BaseDataSets_octasynv2, RandomGeneratorv4, train_aug, val_aug, RandomGeneratorv4, BaseDataSets_octawithback, BaseDataSets_cornwithback,BaseDataSets_drivewithback,BaseDataSets_chasesyn, BaseDataSets_chasewithbackori, BaseDataSets_chasewithback
from networks.net_factory import net_factory
from utils import losses, metrics, ramps
from utils.gate_crf_loss import ModelLossSemsegGatedCRF
from val_2D import test_single_volume, test_single_volume_cct_corn, test_single_volume_new, test_single_volume_ds, test_single_image, test_single_image_corn
from utils.TreeEnergyLoss.kernels.lib_tree_filter.modules.tree_filter import MinimumSpanningTree
from utils.TreeEnergyLoss.kernels.lib_tree_filter.modules.tree_filter import TreeFilter2D

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='../data/OCTA500', help='Name of Experiment')
parser.add_argument('--exp', type=str,
                    default='OCTA_pCE_TreeELoss', help='experiment_name')
parser.add_argument('--fold', type=int,
                    default=10000, help='cross validation')
parser.add_argument('--fold1', type=str,
                    default='all', help='sample name')
parser.add_argument('--sup_type', type=str,
                    default='scribble', help='supervision type')
parser.add_argument('--data_type', type=str,
                    default='octa', help='data type')
parser.add_argument('--model', type=str,
                    default='unet', help='model_name')
parser.add_argument('--num_classes', type=int,  default=4,
                    help='output channel of network')
parser.add_argument('--max_iterations', type=int,
                    default=30000, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=6,
                    help='batch_size per gpu')
parser.add_argument('--deterministic', type=int,  default=1,
                    help='whether use deterministic training')
parser.add_argument('--base_lr', type=float,  default=0.01,
                    help='segmentation network learning rate')
parser.add_argument('--patch_size', type=list,  default=[960, 960],
                    help='patch size of network input')
parser.add_argument('--seed', type=int,  default=2022, help='random seed')
parser.add_argument('--treeloss_weight', type=float,  default=0.6, help='treeloss_weight')
args = parser.parse_args()


def tv_loss(predication):
    min_pool_x = nn.functional.max_pool2d(
        predication * -1, (3, 3), 1, 1) * -1
    contour = torch.relu(nn.functional.max_pool2d(
        min_pool_x, (3, 3), 1, 1) - min_pool_x)
    # length
    length = torch.mean(torch.abs(contour))
    return length

class TreeEnergyLoss(nn.Module):
    def __init__(self):
        super(TreeEnergyLoss, self).__init__()
        # self.configer = configer
        # if self.configer is None:
        #     print("self.configer is None")

        # self.weight = self.configer.get('tree_loss', 'params')['weight']
        # self.weight = weight
        self.mst_layers = MinimumSpanningTree(TreeFilter2D.norm2_distance)
        self.tree_filter_layers = TreeFilter2D(groups=1, sigma=0.02) ##pls see the paper for the sigma!!!!!

    def forward(self, preds, low_feats, high_feats, unlabeled_ROIs, weight):
        # scale low_feats via high_feats
        with torch.no_grad():
            batch, _, h, w = preds.size()
            # print("preds.size()", preds.size())
            low_feats = F.interpolate(low_feats, size=(h, w), mode='bilinear', align_corners=False)
            # print("low_feats.size()", low_feats.size())
            unlabeled_ROIs = F.interpolate(unlabeled_ROIs.unsqueeze(1).float(), size=(h, w), mode='nearest')
            # print("unlabeled_ROIs.size()", unlabeled_ROIs.size())
            N = unlabeled_ROIs.sum()
            # print('high_feats.size()', high_feats.size())
            # print("N", N)

        prob = torch.softmax(preds, dim=1)
        # print("prob.size()", prob.size())
        # low-level MST
        tree = self.mst_layers(low_feats)
        AS = self.tree_filter_layers(feature_in=prob, embed_in=low_feats, tree=tree)  # [b, n, h, w]

        # high-level MST
        if high_feats is not None:
            high_feats = F.interpolate(high_feats, size=(h, w), mode='bilinear', align_corners=False)
            # print('new high_feats.size()', high_feats.size())
            tree = self.mst_layers(high_feats)
            # print('tree.size()', tree.size())
            AS = self.tree_filter_layers(feature_in=AS, embed_in=high_feats, tree=tree, low_tree=False)  # [b, n, h, w]

        tree_loss = (unlabeled_ROIs * torch.abs(prob - AS)).sum()
        if N > 0:
            tree_loss /= N

        return weight * tree_loss


def train(args, snapshot_path):
    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size
    max_iterations = args.max_iterations
    tree_loss_weight = args.treeloss_weight
    data_type = args.data_type

    model = net_factory(net_type=args.model, in_chns=1, class_num=num_classes)
    if data_type == 'octa':
        db_train = BaseDataSets_octa(base_dir=args.root_path, split="train", transform=transforms.Compose([
        RandomGeneratorv2(args.patch_size)]), fold=args.fold, sup_type=args.sup_type)
    elif data_type == 'octa_withback':
        db_train = BaseDataSets_octawithback(base_dir=args.root_path, split="train", transform=transforms.Compose([
        # RandomGeneratorv2(args.patch_size)]), fold=args.fold, sup_type=args.sup_type)
        RandomGeneratorv3(args.patch_size)]), fold=args.fold, sup_type=args.sup_type)
    elif data_type == 'corn':
        db_train = BaseDataSets_cornwithback(base_dir=args.root_path, split="train", transform=transforms.Compose([
            RandomGeneratorv3(args.patch_size)]), fold=args.fold1, sup_type=args.sup_type)
            # RandomGeneratorv2(args.patch_size)]), fold=args.fold1, sup_type=args.sup_type)
    elif data_type == 'corn_wobz':
        db_train = BaseDataSets_cornwithback(base_dir=args.root_path, split="train", transform=transforms.Compose([
            # RandomGeneratorv3(args.patch_size)]), fold=args.fold1, sup_type=args.sup_type)
            RandomGeneratorv2(args.patch_size)]), fold=args.fold1, sup_type=args.sup_type)
    elif data_type == 'drive':
        db_train = BaseDataSets_drivewithback(base_dir=args.root_path, split="train", transform=transforms.Compose([
            RandomGeneratorv3(args.patch_size)]), fold=args.fold1, sup_type=args.sup_type)
            # RandomGeneratorv2(args.patch_size)]), fold=args.fold1, sup_type=args.sup_type)
    elif data_type == 'drive_wobz':
        db_train = BaseDataSets_drivewithback(base_dir=args.root_path, split="train", transform=transforms.Compose([
            # RandomGeneratorv3(args.patch_size)]), fold=args.fold1, sup_type=args.sup_type)
            RandomGeneratorv2(args.patch_size)]), fold=args.fold1, sup_type=args.sup_type)
    elif data_type == 'chase':
        db_train = BaseDataSets_chasewithback(base_dir=args.root_path, split="train", transform=transforms.Compose([
            RandomGeneratorv3(args.patch_size)]), fold=args.fold1, sup_type=args.sup_type)
            # RandomGeneratorv2(args.patch_size)]), fold=args.fold1, sup_type=args.sup_type)
    elif data_type == 'chase_wobz':
        db_train = BaseDataSets_chasewithback(base_dir=args.root_path, split="train", transform=transforms.Compose([
            # RandomGeneratorv3(args.patch_size)]), fold=args.fold1, sup_type=args.sup_type)
            RandomGeneratorv2(args.patch_size)]), fold=args.fold1, sup_type=args.sup_type)
    else:
        db_train = BaseDataSets(base_dir=args.root_path, split="train", transform=transforms.Compose([
        RandomGenerator(args.patch_size)
    ]), fold=args.fold, sup_type=args.sup_type)
    
    if data_type == 'octa':
        db_val = BaseDataSets_octa(base_dir=args.root_path,
                            fold=args.fold, split="val")
    elif data_type == 'octa_withback':
        db_val = BaseDataSets_octawithback(base_dir=args.root_path,
                            fold=args.fold, split="val")
    elif data_type == 'corn' or data_type == 'corn_wobz' or data_type == 'corn_syn':
        db_val = BaseDataSets_cornwithback(base_dir=args.root_path,
                            fold=args.fold1, split="val")
    elif data_type == 'drive' or data_type == 'drive_wobz' or data_type == 'drive_syn':
        db_val = BaseDataSets_drivewithback(base_dir=args.root_path,
                            fold=args.fold1, split="val")
    elif data_type == 'chase' or data_type == 'chase_wobz' or data_type == 'chase_syn':
        db_val = BaseDataSets_chasewithback(base_dir=args.root_path,
                            fold=args.fold1, split="val")
    else:
        db_val = BaseDataSets(base_dir=args.root_path, fold=args.fold, split="val")

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True,
                             num_workers=8, pin_memory=True, worker_init_fn=worker_init_fn)
    valloader = DataLoader(db_val, batch_size=1, shuffle=False,
                           num_workers=1)

    model.train()

    optimizer = optim.SGD(model.parameters(), lr=base_lr,
                          momentum=0.9, weight_decay=0.0001)
    ce_loss = CrossEntropyLoss(ignore_index=2)
    dice_loss = losses.DiceLoss(num_classes)
    gatecrf_loss = ModelLossSemsegGatedCRF()

    tree_loss = TreeEnergyLoss()

    writer = SummaryWriter(snapshot_path + '/log')
    logging.info("{} iterations per epoch".format(len(trainloader)))

    iter_num = 0
    max_epoch = max_iterations // len(trainloader) + 1
    best_performance = 0.0
    iterator = tqdm(range(max_epoch), ncols=70)
    loss_gatedcrf_kernels_desc = [{"weight": 1, "xy": 6, "rgb": 0.1}]
    loss_gatedcrf_radius = 5
    for epoch_num in iterator:
        for i_batch, sampled_batch in enumerate(trainloader):

            volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()

            out = model(volume_batch)
            outputs, feature, de1, de2, de3, de4 = out[0], out[1], out[2], out[3], out[4], out[5]
            outputs_soft = torch.softmax(outputs, dim=1)

            loss_ce = ce_loss(outputs, label_batch[:].long())
            # out_gatedcrf = gatecrf_loss(
            #     outputs_soft,
            #     loss_gatedcrf_kernels_desc,
            #     loss_gatedcrf_radius,
            #     volume_batch,
            #     256,
            #     256,
            # )["loss"]
            unlabeled_RoIs = (sampled_batch['label'] == 2)
            unlabeled_RoIs = unlabeled_RoIs.cuda()
            # print("unlabeled_RoIs.unique", unlabeled_RoIs.unique())
            three_channel = sampled_batch['image'].repeat(3,1)
            three_channel = three_channel.cuda()
            # print("three_channel", three_channel.min(), three_channel.max())
            out_tree_loss = tree_loss(outputs, three_channel, de3, unlabeled_RoIs, tree_loss_weight)  #############
            loss = loss_ce + out_tree_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_

            iter_num = iter_num + 1
            writer.add_scalar('info/lr', lr_, iter_num)
            writer.add_scalar('info/total_loss', loss, iter_num)
            writer.add_scalar('info/loss_ce', loss_ce, iter_num)
            writer.add_scalar('info/out_treeloss', out_tree_loss, iter_num)

            logging.info(
                'iteration %d : loss : %f, loss_ce: %f, loss_tree: %f' %
                (iter_num, loss.item(), loss_ce.item(), out_tree_loss.item()))

            if iter_num % 20 == 0:
                image = volume_batch[0, ...]
                image = (image - image.min()) / (image.max() - image.min())
                writer.add_image('train/Image', image, iter_num)
                outputs = torch.argmax(torch.softmax(
                    outputs, dim=1), dim=1, keepdim=True)
                writer.add_image('train/Prediction',
                                 outputs[0, ...] * 127, iter_num)
                labs = label_batch[0, ...].unsqueeze(0) * 127
                writer.add_image('train/GroundTruth', labs, iter_num)

            if iter_num > 60 and iter_num % 4 == 0:
                model.eval()
                metric_list = 0.0
                for i_batch, sampled_batch in enumerate(valloader):
                    metric_i = test_single_image_corn(
                        sampled_batch["image"], sampled_batch["label"], model, classes=num_classes)
                    metric_list += np.array(metric_i)
                metric_list = metric_list / len(db_val)
                for class_i in range(num_classes-1):
                    writer.add_scalar('info/val_{}_dice'.format(class_i+1),
                                      metric_list[class_i, 0], iter_num)
                    writer.add_scalar('info/val_{}_hd95'.format(class_i+1),
                                      metric_list[class_i, 1], iter_num)
                    writer.add_scalar('info/val_{}_assd'.format(class_i+1),
                                      metric_list[class_i, 2], iter_num)
                    writer.add_scalar('info/val_{}_dice1'.format(class_i+1),
                                      metric_list[class_i, 3], iter_num)
                    writer.add_scalar('info/val_{}_dice2'.format(class_i+1),
                                      metric_list[class_i, 4], iter_num)
                    writer.add_scalar('info/val_{}_dice3'.format(class_i+1),
                                      metric_list[class_i, 5], iter_num)
                    writer.add_scalar('info/val_{}_se'.format(class_i+1),
                                      metric_list[class_i, 6], iter_num)
                    writer.add_scalar('info/val_{}_sp'.format(class_i+1),
                                      metric_list[class_i, 7], iter_num)

                performance = np.mean(metric_list, axis=0)[0]

                mean_hd95 = np.mean(metric_list, axis=0)[1]
                mean_assd = np.mean(metric_list, axis=0)[2]
                mean_dice1 = np.mean(metric_list, axis=0)[3]
                mean_dice2 = np.mean(metric_list, axis=0)[4]
                mean_dice3 = np.mean(metric_list, axis=0)[5]
                mean_se = np.mean(metric_list, axis=0)[6]
                mean_sp = np.mean(metric_list, axis=0)[7]
                writer.add_scalar('info/val_mean_dice', performance, iter_num)
                writer.add_scalar('info/val_mean_hd95', mean_hd95, iter_num)
                writer.add_scalar('info/val_mean_assd', mean_assd, iter_num)
                writer.add_scalar('info/val_mean_dice1', mean_dice1, iter_num)
                writer.add_scalar('info/val_mean_dice2', mean_dice2, iter_num)
                writer.add_scalar('info/val_mean_dice3', mean_dice3, iter_num)
                writer.add_scalar('info/val_mean_se', mean_se, iter_num)
                writer.add_scalar('info/val_mean_sp', mean_sp, iter_num)

                if performance > best_performance:
                    best_performance = performance
                    save_mode_path = os.path.join(snapshot_path,
                                                  'iter_{}_dice_{}.pth'.format(
                                                      iter_num, round(best_performance, 4)))
                    save_best = os.path.join(snapshot_path,
                                             '{}_best_model.pth'.format(args.model))
                    torch.save(model.state_dict(), save_mode_path)
                    torch.save(model.state_dict(), save_best)

                logging.info(
                    'iteration %d : mean_dice : %f mean_hd95 : %f mean_assd: %f mean_dice1: %f' % (iter_num, performance, mean_hd95, mean_assd, mean_dice1))
                model.train()

            if iter_num % 3000 == 0:
                save_mode_path = os.path.join(
                    snapshot_path, 'iter_' + str(iter_num) + '.pth')
                torch.save(model.state_dict(), save_mode_path)
                logging.info("save model to {}".format(save_mode_path))

            if iter_num >= max_iterations:
                break
        if iter_num >= max_iterations:
            iterator.close()
            break
    writer.close()
    return "Training Finished!"


if __name__ == "__main__":
    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    snapshot_path = "../model/{}_{}/{}".format(
        args.exp, args.fold1, args.sup_type)
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    # if os.path.exists(snapshot_path + '/code'):
    #     shutil.rmtree(snapshot_path + '/code')
    # shutil.copytree('.', snapshot_path + '/code',
    #                 shutil.ignore_patterns(['.git', '__pycache__']))

    logging.basicConfig(filename=snapshot_path+"/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    train(args, snapshot_path)
