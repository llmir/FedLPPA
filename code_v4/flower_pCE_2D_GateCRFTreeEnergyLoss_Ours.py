# -*- coding:utf-8 -*-
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
from torch.nn.modules.loss import CrossEntropyLoss, KLDivLoss, MSELoss
from torch.utils.data import DataLoader
from torchvision import transforms

import flwr as fl
from flwr.common.logger import log
from flwr.server import ServerConfig
from flwr.server.client_manager import SimpleClientManager
from collections import OrderedDict
from logging import DEBUG, INFO
import timeit
import copy
from torch.cuda.amp import autocast, GradScaler

from dataloaders import utils
from dataloaders.dataset import BaseDataSets, RandomGenerator
from networks.net_factory import net_factory
from utils import losses, metrics, ramps
from utils.gate_crf_loss import ModelLossSemsegGatedCRF
from val_2D import test_single_volume, test_single_volume_ds
from flower_common import (BaseClient, MyModel, fit_metrics_aggregation_fn, TreeEnergyLoss,MScaleRecurveTreeEnergyLoss, evaluate, get_evaluate_fn,
                        get_strategy, MyServer, VAL_METRICS, get_fedrep_local_keys, get_evaluate_metrics_aggregation_fn,
                        get_bn_stats)



class MyClient(BaseClient):

    def __init__(self, args, model, trainloader, valloader, amp=False):
        super(MyClient, self).__init__(args, model, trainloader, valloader)
        self.amp = amp
        if self.amp:
            self.scaler = GradScaler()
        self.best_performance = 0.0

    def _train(self, config):
        self.model.train()

        # optimizer
        if self.args.strategy == 'FedRep':
            local_keys = get_fedrep_local_keys(self.args.model, self.args.in_chns, self.args.num_classes)
            decay_params, nondecay_params = [], []
            for name, param in self.model.named_parameters():
                if 'bias' in name or (name.replace('model.', '') not in local_keys):
                    nondecay_params += [param]
                else:
                    decay_params += [param]
            optimize_params = [{'params': decay_params, 'weight_decay': 0.0001},
                            {'params': nondecay_params, 'weight_decay': 0}]
            optimizer = optim.SGD(optimize_params, lr=self.current_lr,
                                momentum=0.9)
        else:
            optimizer = optim.AdamW(self.model.parameters(),lr=self.current_lr,betas=(0.9, 0.999),eps=1e-8,weight_decay=1e-2, amsgrad=False)

        ce_loss = CrossEntropyLoss(ignore_index=self.args.num_classes)
        dice_loss = losses.pDLoss(self.args.num_classes, ignore_index=self.args.num_classes)
        mse_loss = MSELoss()
        gatecrf_loss = ModelLossSemsegGatedCRF()

        tree_loss = TreeEnergyLoss()
        tree_loss_muti = MScaleRecurveTreeEnergyLoss()
        # writer = SummaryWriter(snapshot_path + '/log')
        log(INFO, '{} iterations per epoch'.format(len(self.trainloader)))

        if self.args.strategy == 'FedProx':
            server_model = copy.deepcopy(self.model)

        uncertainty_list = []
        loss_gatedcrf_kernels_desc = [{"weight": 1, "xy": 6, "rgb": 0.1}]
        loss_gatedcrf_radius = 5
        for i_iter in range(config['iters']):
            # genearate sampled batches
            if self.current_iter % len(self.trainloader) == 0:
                print('generating sampled batches......')
                self.sampled_batches.clear()
                for i_batch, sampled_batch in enumerate(self.trainloader):
                    self.sampled_batches.append(sampled_batch)

            idx = self.current_iter % len(self.trainloader)
            sampled_batch = self.sampled_batches[idx]
            # print(self.current_iter, i_iter, idx)

            if self.args.img_class == 'faz':
                volume_batch, label_batch = sampled_batch['image'].unsqueeze(1), sampled_batch['label']
                volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()
            elif self.args.img_class == 'odoc' or self.args.img_class == 'polyp':
                volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
                volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()

            # gradient settings
            if self.args.strategy in ['FedLC', 'FedALALC', 'FedAPLC']:
                local_keys = ['decoder.out_conv.weight', 'decoder.out_conv.bias']

            if self.args.strategy in ['FedRep', 'FedLC', 'FedALALC', 'FedAPLC']:
                if i_iter < self.args.iters - self.args.rep_iters:
                    for name, param in self.model.named_parameters():
                        if name.replace('model.', '') in local_keys:
                            # print(self.args.cid, name, True)
                            param.requires_grad = True
                        else:
                            param.requires_grad = False
                else:
                    for name, param in self.model.named_parameters():
                        if name.replace('model.', '') in local_keys:
                            # print(self.args.cid, name, False)
                            param.requires_grad = False
                        else:
                            param.requires_grad = True

            ## training procedures
            with autocast(enabled=self.amp):
                # forward
                out = self.model(volume_batch)
                if self.args.model == 'fcnet':
                    high_feats, outputs = out
                elif self.args.model in ['deeplabv3plus', 'treefcn']:
                    outputs, _, high_feats = out
                elif self.args.model == 'unet_head':
                    outputs, feature, de1, de2, de3, de4, aux_output = out
                    high_feats = aux_output
                elif self.args.model == 'unet_multihead':
                    outputs, feature, de1, de2, de3, de4, aux_output1, aux_output2, aux_output3 = out
                    high_feats = aux_output1
                elif self.args.model == 'unet_lc':
                    outputs, feature, de1, de2, de3, de4, heatmaps, aux_output = out
                    high_feats = aux_output
                elif self.args.model == 'unet_lc_mutihead':
                    outputs, feature, de1, de2, de3, de4, heatmaps, aux_output1, aux_output2, aux_output3 = out
                    high_feats = aux_output
                else:
                    outputs, feature, de1, de2, de3, de4 = out

                outputs_soft = torch.softmax(outputs, dim=1)

                loss_ce = ce_loss(outputs, label_batch[:].long())
                unlabeled_RoIs = (sampled_batch['label'] == self.args.num_classes)
                unlabeled_RoIs = unlabeled_RoIs.cuda()
                if self.args.img_class == 'faz':
                    three_channel = volume_batch.repeat(1, 3, 1, 1)
                elif self.args.img_class == 'odoc' or self.args.img_class =='polyp':
                    three_channel = volume_batch
                three_channel = three_channel.cuda()
                out_tree_loss, heatmaps_tree = tree_loss(outputs, three_channel, high_feats, unlabeled_RoIs, self.args.tree_loss_weight)
                out_gatedcrf = gatecrf_loss(
                    outputs_soft,
                    loss_gatedcrf_kernels_desc,
                    loss_gatedcrf_radius,
                    volume_batch,
                    self.args.img_size,
                    self.args.img_size
                )["loss"]
                loss = loss_ce + out_tree_loss + 0.1 * out_gatedcrf

                if self.args.strategy == 'FedProx' and i_iter > 0:
                    w_diff = torch.tensor(0.).cuda()
                    for w, w_t in zip(server_model.parameters(), self.model.parameters()):
                        w_diff += torch.pow(torch.norm(w - w_t), 2)
                    w_diff = torch.sqrt(w_diff)
                    loss_prox = self.args.mu / 2. * w_diff
                    loss += loss_prox

                if self.args.strategy in ['MetaFed', 'FedAN']:
                    rot_times = random.randrange(0, 4)
                    rotated_volume_batch = torch.rot90(volume_batch, rot_times, [2, 3])
                    T = 8
                    _, _, w, h = volume_batch.shape
                    volume_batch_r = rotated_volume_batch.repeat(2, 1, 1, 1)
                    stride = volume_batch_r.shape[0] // 2
                    preds = torch.zeros([stride * T, self.args.num_classes, w, h]).cuda()
                    for i in range(T // 2):
                        ema_inputs = volume_batch_r + \
                            torch.clamp(torch.randn_like(
                                volume_batch_r) * 0.1, -0.2, 0.2)
                        with torch.no_grad():
                            preds[2 * stride * i:2 * stride *
                                (i + 1)] = self.model(ema_inputs)[0]
                    preds = F.softmax(preds, dim=1)
                    preds = preds.reshape(T, stride, self.args.num_classes, w, h)
                    preds = torch.mean(preds, dim=0)
                    uncertainty = -1.0 * \
                        torch.sum(preds * torch.log(preds + 1e-6), dim=1, keepdim=True)
                    uncertainty_list.append(torch.mean(uncertainty).item())

                if self.args.strategy == 'MetaFed':
                    if self.model.meta_flag and (self.current_iter + 1) > self.args.init_iters:
                        self.model.teacher_model.eval()
                        teacher_out = self.model.teacher_model(volume_batch)
                        teacher_feature = teacher_out[1]
                        loss_meta_feature = torch.tensor(0.0).cuda()
                        for i_feature in range(len(feature)):
                            loss_meta_feature += F.mse_loss(feature[i_feature], teacher_feature[i_feature]).detach()
                        teacher_outputs = teacher_out[0]
                        teacher_outputs_soft = torch.softmax(teacher_outputs, dim=1)
                        # loss_meta = F.cross_entropy(outputs_soft.log(), teacher_outputs_soft).detach()
                        loss_meta = F.kl_div(outputs_soft.log(), teacher_outputs_soft).detach()
                        loss = torch.add(loss, loss_meta, alpha=self.model.lam)
                        loss = torch.add(loss, loss_meta_feature, alpha=self.args.beta)
                        lam = self.model.lam
                        # print(self.current_iter + 1, loss, loss_ce, loss_meta)
                    else:
                        loss_meta = torch.tensor(0.0)
                        lam = 0.0

                if (self.args.strategy in ['FedAP', 'FedAPLC'] and (self.current_iter + 1) == self.args.iters) \
                    or (self.args.strategy == 'MetaFed' and (self.current_iter + 1) == (self.args.common_iters + self.args.iters)):
                    print(self.args.cid, 'get_bn_stats', self.current_iter + 1)
                    bnm, bnv = get_bn_stats(self.args, self.model.model, self.trainloader)

                if self.args.strategy in ['FedLC', 'FedALALC', 'FedAPLC']:
                    loss_lc = 0
                    for other_client in range(self.args.min_num_clients):
                        if other_client == self.args.cid:
                           continue 
                        with torch.no_grad():
                            _heatmaps = self.model(volume_batch, other_client)[-2]
                            # print(heatmaps[-1].shape, _heatmaps[-1].shape)
                            loss_lc += mse_loss(heatmaps[-1], _heatmaps[-1].detach())
                            # loss_lc += mse_loss(heatmaps[-1], _heatmaps[-1])
                    loss_lc = -loss_lc / (self.args.min_num_clients - 1)
                    loss = torch.add(loss, loss_lc, alpha=self.args.alpha)

            # update model parameters
            optimizer.zero_grad()
            if self.amp:
                self.scaler.scale(loss).backward()
                self.scaler.step(optimizer)
                self.scaler.update()
            else:
                loss.backward()
                optimizer.step()

            self.current_iter = self.current_iter + 1
            log(INFO, 'client %d : iteration %d : lr: %f, loss : %f, loss_ce: %f, loss_tree: %f' % (self.cid, self.current_iter, self.current_lr, loss.item(), loss_ce.item(), out_tree_loss.item()))

            lr_ = self.args.base_lr * (1.0 - self.current_iter / self.args.max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_
            self.current_lr = lr_

        # pack general metrics
        image = volume_batch[1, :, :, :]
        image = (image - image.min()) / (image.max() - image.min())
        outputs = torch.argmax(torch.softmax(outputs, dim=1), dim=1, keepdim=True)
        outputs = outputs[1, ...] * 50
        labs = label_batch[1, ...].unsqueeze(0) * 50
        if self.args.img_class == 'odoc' or self.args.img_class == 'polyp':
            outputs, labs = outputs.repeat(3, 1, 1), labs.repeat(3, 1, 1)

        metrics_ = {
            'client_{}_lr'.format(self.cid): self.current_lr,
            'client_{}_total_loss'.format(self.cid): loss.item(),
            'client_{}_loss_ce'.format(self.cid): loss_ce.item(),
            'client_{}_Image'.format(self.cid): fl.common.ndarray_to_bytes(image.cpu().numpy()),
            'client_{}_Prediction'.format(self.cid): fl.common.ndarray_to_bytes(outputs.cpu().numpy()),
            'client_{}_GroundTruth'.format(self.cid):fl.common.ndarray_to_bytes(labs.cpu().numpy()),
        }

        # pack strategy-specific metrics
        if self.args.strategy == 'FedProx':
            metrics_['client_{}_loss_prox'.format(self.cid)] = loss_prox.item()
        if self.args.strategy == 'MetaFed':
            metrics_['client_{}_loss_meta'.format(self.cid)] = loss_meta.item()
            metrics_['client_{}_loss_meta_feature'.format(self.cid)] = loss_meta_feature.item()
            metrics_['client_{}_lam'.format(self.cid)] = lam

        if self.args.strategy in ['MetaFed', 'FedAN']:
            overall_uncertainty = np.mean(uncertainty_list)
            metrics_['client_{}_uncertainty'.format(self.cid)] = overall_uncertainty

        if (self.args.strategy in ['FedAP', 'FedAPLC'] and self.current_iter == self.args.iters) \
            or (self.args.strategy == 'MetaFed' and self.current_iter == (self.args.common_iters + self.args.iters)):
            metrics_['client_{}_bnl'.format(self.cid)] = len(bnm)
            print(self.cid, bnm[0].shape, bnv[0].shape, len(bnm))
            for i_bn in range(len(bnm)):
                metrics_['client_{}_bnm_{}'.format(self.cid, i_bn)] = fl.common.ndarray_to_bytes(bnm[i_bn])
                metrics_['client_{}_bnv_{}'.format(self.cid, i_bn)] =  fl.common.ndarray_to_bytes(bnv[i_bn])

        if self.args.strategy in ['FedLC', 'FedALALC', 'FedAPLC']:
            metrics_['client_{}_loss_lc'.format(self.cid)] = loss_lc.item()

        return loss.item(), metrics_


from flower_common import PretrainDataset
from torchvision.utils import make_grid
from tqdm import tqdm
def pretrain_model(args, writer, worker_init_fn):
    db_train = PretrainDataset(args.root_path, args.sup_type_list, transform=transforms.Compose([
        RandomGenerator(args.patch_size, img_class=args.img_class)
    ]))
    trainloader = DataLoader(db_train, batch_size=args.batch_size, shuffle=True,
                             num_workers=4, pin_memory=True, worker_init_fn=worker_init_fn)

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    model = net_factory(args, net_type=args.model, in_chns=args.in_chns, class_num=args.num_classes)
    model.cuda()

    if args.amp:
        scaler = GradScaler()
    optimizer = optim.SGD(model.parameters(), lr=args.base_lr,
                          momentum=0.9, weight_decay=0.0001)
    ce_loss = CrossEntropyLoss(ignore_index=args.num_classes)
    dice_loss = losses.pDLoss(args.num_classes,ignore_index=args.num_classes)
    iter_num = 0
    max_epoch = args.pretrain_iters // len(trainloader) + 1
    iterator = tqdm(range(max_epoch), ncols=70)
    model.train()
    for epoch_num in iterator:
        for i_batch, sampled_batch in enumerate(trainloader):
            if args.img_class == 'faz':
                volume_batch, label_batch = sampled_batch['image'].unsqueeze(1), sampled_batch['label']
                volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()
            elif args.img_class == 'odoc' or args.img_class == 'polyp':
                volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
                volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()
    
            with autocast(enabled=(args.amp == 1)):
                outputs = model(volume_batch)[0]
                outputs_soft = torch.softmax(outputs, dim=1)
                loss_ce = ce_loss(outputs, label_batch[:].long())
                loss = loss_ce

            optimizer.zero_grad()
            if args.amp:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            iter_num = iter_num + 1
            log(INFO, 'pretrain : iteration %d : lr: %f, loss : %f, loss_ce: %f' % (iter_num, args.base_lr, loss.item(), loss_ce.item()))


            if iter_num % args.iters == 0:
                image = volume_batch[1, :, :, :]
                image = (image - image.min()) / (image.max() - image.min())
                outputs = torch.argmax(torch.softmax(outputs, dim=1), dim=1, keepdim=True)
                outputs = outputs[1, ...] * 50
                labs = label_batch[1, ...].unsqueeze(0) * 50
                if args.img_class == 'odoc' or args.img_class == 'polyp':
                    outputs, labs = outputs.repeat(3, 1, 1), labs.repeat(3, 1, 1)

                image_list = np.array([image.cpu().numpy(), outputs.cpu().numpy(), labs.cpu().numpy()])
                writer.add_image('pretrain/grid_image', make_grid(torch.tensor(image_list)), iter_num)

            if iter_num >= args.pretrain_iters:
                save_mode_path = os.path.join(args.snapshot_path, 'fedap_pretrain.pth')
                torch.save(model.state_dict(), save_mode_path)
                log(INFO, 'save model to {}'.format(save_mode_path))
                iterator.close()
                break

    return save_mode_path, model.state_dict()


def main():
    parser = argparse.ArgumentParser()
    ## flower related arguments
    parser.add_argument('--server_address', type=str,
                        default='[::]:8080', help='gRPC server address (default: [::]:8080)')
    parser.add_argument('--gpu', type=int,
                        required=True, help='GPU index')
    parser.add_argument('--role', type=str,
                        required=True, help='Role')
    # server
    parser.add_argument('--iters', type=int,
                        default=20, help='Number of iters (default: 20)')
    parser.add_argument('--eval_iters', type=int,
                        default=200, help='Number of iters (default: 200)')
    parser.add_argument('--rep_iters', type=int,
                        default=12, help='Number of iters (default: 12)')
    parser.add_argument('--sample_fraction', type=float,
                        default=1.0, help='Fraction of available clients used for fit/evaluate (default: 1.0)')
    parser.add_argument('--min_num_clients', type=int,
                        default=2, help='Minimum number of available clients required (default: 2)')
    parser.add_argument('--strategy', type=str,
                        default='FedAvg', help='Federated learning algorithm (default: FedAvg)')
    parser.add_argument('--mu', type=float, default=1e-3,
                        help='The hyper parameter for FedProx')
    parser.add_argument('--lam', type=float, default=0.5,
                        help='The hyper parameter for MetaFed/FedAN')
    parser.add_argument('--sort_lam', type=float, default=0.5,
                        help='The hyper parameter for MetaFed')
    parser.add_argument('--sort_beta', type=float, default=0.5,
                        help='The hyper parameter for MetaFed')
    parser.add_argument('--beta', type=float, default=0.5,
                        help='The hyper parameter for MetaFed/FedAN')
    parser.add_argument('--alpha', type=float, default=0.5,
                        help='The hyper parameter for FedLC/FedALALC/FedAPLC')
    parser.add_argument('--init_iters', type=int,
                        default=100, help='Number of iters (default: 100)')
    parser.add_argument('--common_iters', type=int,
                        default=400, help='Number of iters (default: 400)')
    parser.add_argument('--pretrain_iters', type=int,
                        default=150, help='Number of iters (default: 150)')
    parser.add_argument('--pretrain', type=int,  default=0,
                        help='pretrain')
    parser.add_argument('--model_momentum', type=float, default=0.5,
                        help='The hyper parameter for FedAP/FedAPLC')
    # client
    parser.add_argument('--cid', type=int, default=0, help='Client CID (no default)')

    ## WSL4MIS related arguments
    parser.add_argument('--root_path', type=str,
                    default='../data/FAZ_h5', help='Name of Experiment')
    parser.add_argument('--exp', type=str,
                        default='faz_pCE', help='experiment_name')
    parser.add_argument('--client', type=str,
                        default='client1', help='cross validation')
    parser.add_argument('--sup_type', type=str,
                        default='mask', help='supervision label type(scr ; label ; scr_n ; keypoint ; block)')
    parser.add_argument('--sup_type_list', nargs='+', type=str,
                        help='supervision label type')
    parser.add_argument('--model', type=str,
                        default='unet', help='model_name')
    parser.add_argument('--num_classes', type=int,  default=2,
                        help='output channel of network')
    parser.add_argument('--max_iterations', type=int,
                        default=30000, help='maximum epoch number to train')
    parser.add_argument('--batch_size', type=int, default=12,
                        help='batch_size per gpu')
    parser.add_argument('--in_chns', type=int, default=1,
                        help='image channel')
    parser.add_argument('--deterministic', type=int,  default=1,
                        help='whether use deterministic training')
    parser.add_argument('--amp', type=int,  default=0,
                        help='whether use amp training')
    parser.add_argument('--base_lr', type=float,  default=0.01,
                        help='segmentation network learning rate')
    parser.add_argument('--patch_size', type=list,  default=[256, 256],
                        help='patch size of network input')
    parser.add_argument('--img_class', type=str,
                        default='faz', help='the img class(odoc or faz)')
    parser.add_argument('--img_size', type=int,  default=256,
                    help='h*w')
    parser.add_argument('--seed', type=int,  default=2022, help='random seed')
    parser.add_argument('--tree_loss_weight', type=float,  default=0.1, help='treeloss_weight')
    args = parser.parse_args()

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

    snapshot_path = '../model/{}'.format(
        args.exp)
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    setattr(args, 'snapshot_path', snapshot_path)

    # Check arguments
    assert args.iters > 0
    assert args.eval_iters > 0 and (args.eval_iters % args.iters == 0)
    assert args.max_iterations > 0 and (args.max_iterations % args.eval_iters == 0)

    if args.strategy == 'FedRep':
        assert args.iters > args.rep_iters
    elif args.strategy == 'MetaFed':
        assert args.eval_iters == args.iters
        assert args.init_iters > args.eval_iters and (args.init_iters % args.eval_iters == 0)
        assert args.common_iters > args.init_iters and (args.common_iters % args.eval_iters == 0)
        assert args.model in ['unet', 'unet_head', 'unet_multihead']
    elif args.strategy == 'FedAP':
        assert args.pretrain_iters > args.eval_iters and (args.pretrain_iters % args.eval_iters == 0)
        assert args.model in ['unet', 'unet_head', 'unet_multihead']
    elif args.strategy == 'FedALA':
        assert args.model in ['unet', 'unet_head', 'unet_multihead']
    elif args.strategy == 'FedLC':
        assert args.iters > args.rep_iters
        assert args.model in ['unet_lc']
    elif args.strategy == 'FedALALC':
        assert args.iters > args.rep_iters
        assert args.model in ['unet_lc']
    elif args.strategy == 'FedAPLC':
        assert args.pretrain_iters > args.eval_iters and (args.pretrain_iters % args.eval_iters == 0)
        assert args.iters > args.rep_iters
        assert args.model in ['unet_lc']

    assert args.role in ['server', 'client']
    assert args.img_class in ['odoc', 'faz', 'polyp']
    if args.img_class == 'faz':
        assert args.sup_type in ['mask', 'scribble', 'scribble_noisy', 'block', 'box', 'keypoint']
    else:
        assert args.sup_type in ['mask', 'scribble', 'scribble_noisy', 'block', 'box', 'keypoint']

    # Configure logger
    if args.role == 'server':
        if os.path.exists(snapshot_path + '/code'):
            shutil.rmtree(snapshot_path + '/code')
        shutil.copytree('.', snapshot_path + '/code',
                        shutil.ignore_patterns(['.git', '__pycache__']))
        fl.common.logger.configure('server', filename=os.path.join(snapshot_path, 'server.log'))
        writer = SummaryWriter(snapshot_path + '/log')
    else:
        fl.common.logger.configure('client_{}'.format(args.cid), filename=os.path.join(snapshot_path, 'client_{}.log'.format(args.cid)))

    log(INFO, 'Arguments: {}'.format(args))

    # Load model and data
    db_train = BaseDataSets(base_dir=args.root_path, split="train", transform=transforms.Compose([
        RandomGenerator(args.patch_size,img_class=args.img_class)
    ]), client=args.client, sup_type=args.sup_type, img_class=args.img_class)
    db_val = BaseDataSets(base_dir=args.root_path,
                          client=args.client, split="val", img_class=args.img_class)

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    trainloader = DataLoader(db_train, batch_size=args.batch_size, shuffle=True,
                             num_workers=4, pin_memory=True, worker_init_fn=worker_init_fn)
    valloader = DataLoader(db_val, batch_size=1, shuffle=False,
                           num_workers=0)

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

    if args.strategy in ['FedLC', 'FedALALC', 'FedAPLC']:
        class MyModelV2(MyModel):
            def forward(self, x, emb_idx=None):
                return self.model(x, emb_idx)
        model = MyModelV2(args, net_factory(args, net_type=args.model, in_chns=args.in_chns, class_num=args.num_classes), trainloader, valloader)
    else:
        model = MyModel(args, net_factory(args, net_type=args.model, in_chns=args.in_chns, class_num=args.num_classes), trainloader, valloader)
    model.cuda()

    if args.role == 'server':
        def fit_config(server_round):
            config = {
                'iter_global': server_round,
                'iters': args.iters,
                'eval_iters': args.eval_iters,
                'batch_size': args.batch_size,
                'stage': 'fit'
            }
            return config

        def evaluate_config_fn(server_round):
            config = {
                'iter_global': server_round,
                'iters': args.iters,
                'eval_iters': args.eval_iters,
                'batch_size': args.batch_size,
                'stage': 'evaluate'
            }
            return config

        # Create strategy
        kwargs = {
            "fraction_fit": args.sample_fraction,
            "min_fit_clients": args.min_num_clients,
            "min_available_clients": args.min_num_clients,
            "evaluate_fn": get_evaluate_fn(args, valloader, amp=(args.amp == 1)),
            "on_fit_config_fn": fit_config,
            "on_evaluate_config_fn": evaluate_config_fn,
            "fit_metrics_aggregation_fn": fit_metrics_aggregation_fn,
            "evaluate_metrics_aggregation_fn": get_evaluate_metrics_aggregation_fn(args, val_metrics=VAL_METRICS),
            "accept_failures": False
        
        }
        '''weights = [val.cpu().numpy() for _, val in model.model.state_dict().items()]
        initial_parameters = fl.common.ndarrays_to_parameters(weights)
        if args.strategy == 'FedAdagrad':
            kwargs.update({'eta': 5e-3, 'eta_l': 5e-3, 'tau': 1e-9,
                        'initial_parameters': initial_parameters})
        elif args.strategy == 'FedAdam':
            kwargs.update({'eta': 5e-3, 'eta_l': 5e-3, 'beta_1': 0.9,
                        'beta_2': 0.99, 'tau': 1e-9, 'initial_parameters': initial_parameters})
        elif args.strategy == 'FedYogi':
            kwargs.update({'eta': 5e-3, 'eta_l': 5e-3, 'beta_1': 0.9,
                        'beta_2': 0.99, 'tau': 1e-9, 'initial_parameters': initial_parameters})'''
        if args.strategy in ['FedAP', 'FedAPLC']:
            if args.pretrain:
                start_time = time.time()
                writer = SummaryWriter(snapshot_path + '/pretrain_log')
                pretrain_path, pretrain_state_dict = pretrain_model(args, writer, worker_init_fn)
                print(time.time() - start_time)
                return
            else:
                pretrain_state_dict = torch.load('{}/fedap_pretrain.pth'.format(args.snapshot_path), map_location='cpu')
                weights = [val.cpu().numpy() for _, val in pretrain_state_dict.items()]
                initial_parameters = fl.common.ndarrays_to_parameters(weights)
                kwargs.update({'iters': args.iters, 'model_momentum': args.model_momentum, 'initial_parameters': initial_parameters})

        if args.strategy == 'FedAN':
            kwargs.update({'iters': args.iters, 'lam': args.lam, 'beta': args.beta})

        strategy = get_strategy(args.strategy, **kwargs)
        # Start server
        state_dict_keys = model.model.state_dict().keys()
        train_scalar_metrics = ['lr', 'total_loss', 'loss_ce']
        train_image_metrics = ['Image', 'Prediction', 'GroundTruth']
        val_metrics = VAL_METRICS
        server = MyServer(
            args=args, writer=writer, state_dict_keys=state_dict_keys, train_scalar_metrics=train_scalar_metrics,
            train_image_metrics=train_image_metrics, val_metrics=val_metrics, client_manager=SimpleClientManager(), strategy=strategy
        )
        fl.server.start_server(
            server_address=args.server_address,
            server=server,
            config=ServerConfig(num_rounds=args.max_iterations, round_timeout=None)
        )
    else:
        client = MyClient(args, model, trainloader, valloader, amp=(args.amp == 1))
        fl.client.start_client(server_address=args.server_address, client=client)



if __name__ == '__main__':
    main()
