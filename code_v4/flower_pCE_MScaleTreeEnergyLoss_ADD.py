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
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import make_grid
from tqdm import tqdm

import flwr as fl
from flwr.common.logger import log
from flwr.server import ServerConfig
from flwr.server.server import Server
from flwr.server.client_manager import ClientManager, SimpleClientManager
from flwr.server.history import History
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
from flower_common import BaseClient, MyModel, fit_metrics_aggregation_fn, MScaleAddTreeEnergyLoss, get_evaluate_fn



class MyClient(BaseClient):

    def __init__(self, args, model, trainloader, valloader, amp=False):
        super(MyClient, self).__init__(args, model, trainloader, valloader)
        self.amp = amp
        if self.amp:
            self.scaler = GradScaler()

    def _train(self, config):
        self.model.train()

        optimizer = optim.SGD(self.model.parameters(), lr=self.current_lr,
                            momentum=0.9, weight_decay=0.0001)
        ce_loss = CrossEntropyLoss(ignore_index=self.args.num_classes)
        dice_loss = losses.DiceLoss(self.args.num_classes)
        gatecrf_loss = ModelLossSemsegGatedCRF()

        tree_loss = MScaleAddTreeEnergyLoss()

        # writer = SummaryWriter(snapshot_path + '/log')
        log(INFO, '{} iterations per epoch'.format(len(self.trainloader)))

        loss_gatedcrf_kernels_desc = [{"weight": 1, "xy": 6, "rgb": 0.1}]
        loss_gatedcrf_radius = 5

        for i_iter in range(int(config['iters'])):
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
            elif self.args.img_class == 'odoc':
                volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
                volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()

            with autocast(enabled=self.amp):
                out = self.model(volume_batch)
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
                unlabeled_RoIs = (sampled_batch['label'] == self.args.num_classes)
                unlabeled_RoIs = unlabeled_RoIs.cuda()
                # print("unlabeled_RoIs.unique", unlabeled_RoIs.unique())
                if self.args.img_class == 'faz':
                    three_channel = volume_batch.repeat(1, 3, 1, 1)
                elif self.args.img_class == 'odoc':
                    three_channel = volume_batch
                three_channel = three_channel.cuda()
                # print("three_channel", three_channel.min(), three_channel.max())
                out_tree_loss = tree_loss(outputs, three_channel, feature[-1], de2, de3, unlabeled_RoIs, self.args.tree_loss_weight)
                loss = loss_ce + out_tree_loss

            optimizer.zero_grad()
            if self.amp:
                self.scaler.scale(loss).backward()
                self.scaler.step(optimizer)
                self.scaler.update()
            else:
                loss.backward()
                optimizer.step()

            self.current_iter = self.current_iter + 1
            log(INFO, 'iteration %d : lr: %f, loss : %f, loss_ce: %f, loss_tree: %f' % (self.current_iter, self.current_lr, loss.item(), loss_ce.item(), out_tree_loss.item()))

            lr_ = self.args.base_lr * (1.0 - self.current_iter / self.args.max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_
            self.current_lr = lr_

        image = volume_batch[1, :, :, :]
        image = (image - image.min()) / (image.max() - image.min())
        outputs = torch.argmax(torch.softmax(
                    outputs, dim=1), dim=1, keepdim=True)
        outputs = outputs[1, ...] * 50
        labs = label_batch[1, ...].unsqueeze(0) * 50
        if self.args.img_class == 'odoc':
            outputs, labs = outputs.repeat(3, 1, 1), labs.repeat(3, 1, 1)

        metrics_ = {
            'client_{}_lr'.format(self.cid): self.current_lr,
            'client_{}_total_loss'.format(self.cid): loss.item(),
            'client_{}_loss_ce'.format(self.cid): loss_ce.item(),
            'client_{}_out_treeloss'.format(self.cid): out_tree_loss.item(),
            'client_{}_Image'.format(self.cid): fl.common.ndarray_to_bytes(image.cpu().numpy()),
            'client_{}_Prediction'.format(self.cid): fl.common.ndarray_to_bytes(outputs.cpu().numpy()),
            'client_{}_GroundTruth'.format(self.cid):fl.common.ndarray_to_bytes(labs.cpu().numpy()),
        }

        return {'loss': loss.item()}, metrics_


class MyServer(Server):

    def __init__(self, args, writer, state_dict_keys, client_manager, strategy):
        super(MyServer, self).__init__(client_manager=client_manager, strategy=strategy)
        self.args = args
        self.writer = writer
        self.state_dict_keys = state_dict_keys

    # pylint: disable=too-many-locals
    def fit(self, num_rounds, timeout):
        '''Run federated averaging for a number of rounds.'''
        history = History()

        # Initialize parameters
        log(INFO, 'Initializing global parameters')
        self.parameters = self._get_initial_parameters(timeout=timeout)
        log(INFO, 'Evaluating initial parameters')
        res = self.strategy.evaluate(0, parameters=self.parameters)
        print(self.strategy.evaluate.__name__)
        if res is not None:
            log(
                INFO,
                'initial parameters (loss, other metrics): %s, %s',
                res[0],
                res[1],
            )
            history.add_loss_centralized(server_round=0, loss=res[0])
            history.add_metrics_centralized(server_round=0, metrics=res[1])

        # Run federated learning for num_rounds
        log(INFO, 'FL starting')
        start_time = timeit.default_timer()

        num_classes = self.args.num_classes
        max_iterations = self.args.max_iterations
        snapshot_path = self.args.snapshot_path
        iters = self.args.iters
        min_num_clients = self.args.min_num_clients
        tree_loss_weight = self.args.tree_loss_weight

        best_performance = 0.0
        iterator = tqdm(range(1, num_rounds+1+iters, iters), ncols=70)
        for current_round in iterator:
            iter_num = current_round - 1
            # Train model and replace previous global model
            res_fit = self.fit_round(server_round=current_round, timeout=timeout)
            parameters_prime, metrics_prime, _ = res_fit
            self.parameters = parameters_prime

            weights = fl.common.parameters_to_ndarrays(self.parameters)
            state_dict = OrderedDict(
                {k: torch.tensor(v) for k, v in zip(self.state_dict_keys, weights)}
            )

            images = []
            for client_id in range(min_num_clients):
                self.writer.add_scalar('info/client_{}_lr'.format(client_id), metrics_prime['client_{}_lr'.format(client_id)], iter_num)
                self.writer.add_scalar('info/client_{}_total_loss'.format(client_id), metrics_prime['client_{}_total_loss'.format(client_id)], iter_num)
                self.writer.add_scalar('info/client_{}_loss_ce'.format(client_id), metrics_prime['client_{}_loss_ce'.format(client_id)], iter_num)
                self.writer.add_scalar('info/client_{}_out_treeloss'.format(client_id), metrics_prime['client_{}_out_treeloss'.format(client_id)], iter_num)
                images.append(fl.common.bytes_to_ndarray(metrics_prime['client_{}_Image'.format(client_id)]))
                images.append(fl.common.bytes_to_ndarray(metrics_prime['client_{}_Prediction'.format(client_id)]))
                images.append(fl.common.bytes_to_ndarray(metrics_prime['client_{}_GroundTruth'.format(client_id)]))

            self.writer.add_image(
                'train/grid_image',
                make_grid(torch.tensor(np.array(images)), nrow=6),
                iter_num
            )

            # Evaluate model using strategy implementation
            if iter_num > 0 and iter_num % 200 == 0:
                res_cen = self.strategy.evaluate(current_round, parameters=self.parameters)
                loss_cen, metrics_cen = res_cen
                log(
                    INFO,
                    'fit progress: (%s, %s, %s, %s)',
                    current_round,
                    loss_cen,
                    metrics_cen,
                    timeit.default_timer() - start_time,
                )
                for class_i in range(num_classes-1):
                    self.writer.add_scalar('info/val_{}_dice'.format(class_i+1),
                                      metrics_cen['val_{}_dice'.format(class_i+1)], iter_num)
                    self.writer.add_scalar('info/val_{}_hd95'.format(class_i+1),
                                      metrics_cen['val_{}_hd95'.format(class_i+1)], iter_num)
                    self.writer.add_scalar('info/val_{}_recall'.format(class_i+1),
                                      metrics_cen['val_{}_recall'.format(class_i+1)], iter_num)
                    self.writer.add_scalar('info/val_{}_precision'.format(class_i+1),
                                      metrics_cen['val_{}_precision'.format(class_i+1)], iter_num)
                    self.writer.add_scalar('info/val_{}_jc'.format(class_i+1),
                                      metrics_cen['val_{}_jc'.format(class_i+1)], iter_num)
                    self.writer.add_scalar('info/val_{}_specificity'.format(class_i+1),
                                      metrics_cen['val_{}_specificity'.format(class_i+1)], iter_num)
                    self.writer.add_scalar('info/val_{}_ravd'.format(class_i+1),
                                      metrics_cen['val_{}_ravd'.format(class_i+1)], iter_num)

                self.writer.add_scalar('info/val_mean_dice', metrics_cen['val_mean_dice'], iter_num)
                self.writer.add_scalar('info/val_mean_hd95', metrics_cen['val_mean_hd95'], iter_num)
                self.writer.add_scalar('info/val_mean_recall', metrics_cen['val_mean_recall'], iter_num)
                self.writer.add_scalar('info/val_mean_precision', metrics_cen['val_mean_precision'], iter_num)
                self.writer.add_scalar('info/val_mean_jc', metrics_cen['val_mean_jc'], iter_num)
                self.writer.add_scalar('info/val_mean_specificity', metrics_cen['val_mean_specificity'], iter_num)
                self.writer.add_scalar('info/val_mean_ravd', metrics_cen['val_mean_ravd'], iter_num)

                history.add_loss_centralized(server_round=current_round, loss=loss_cen)
                history.add_metrics_centralized(
                    server_round=current_round, metrics=metrics_cen
                )

                if metrics_cen['val_mean_dice'] > best_performance:
                    best_performance = metrics_cen['val_mean_dice']
                    save_mode_path = os.path.join(snapshot_path,
                                                    'iter_{}_dice_{}.pth'.format(
                                                        iter_num, round(best_performance, 4)))
                    save_best = os.path.join(snapshot_path,
                                                '{}_best_model.pth'.format(self.args.model))
                    torch.save(state_dict, save_mode_path)
                    torch.save(state_dict, save_best)

                logging.info('iteration %d : mean_dice : %f mean_hd95 : %f : mean_recall : %f mean_precision : %f : mean_jc : %f mean_specificity : %f : mean_ravd : %f' % (iter_num, metrics_cen['val_mean_dice'], metrics_cen['val_mean_hd95'], metrics_cen['val_mean_recall'], metrics_cen['val_mean_precision'], metrics_cen['val_mean_jc'], metrics_cen['val_mean_specificity'], metrics_cen['val_mean_ravd'], ))

            if iter_num % 3000 == 0:
                save_mode_path = os.path.join(
                    snapshot_path, 'iter_' + str(iter_num) + '.pth')
                torch.save(state_dict, save_mode_path)
                logging.info('save model to {}'.format(save_mode_path))

            if iter_num >= max_iterations:
                break

            # Evaluate model on a sample of available clients
            '''res_fed = self.evaluate_round(server_round=current_round, timeout=timeout)
            if res_fed:
                loss_fed, evaluate_metrics_fed, _ = res_fed
                if loss_fed:
                    history.add_loss_distributed(
                        server_round=current_round, loss=loss_fed
                    )
                    history.add_metrics_distributed(
                        server_round=current_round, metrics=evaluate_metrics_fed
                    )'''

        # Bookkeeping
        end_time = timeit.default_timer()
        elapsed = end_time - start_time
        log(INFO, 'FL finished in %s', elapsed)
        return history


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
    parser.add_argument('--sample_fraction', type=float,
                        default=1.0, help='Fraction of available clients used for fit/evaluate (default: 1.0)')
    parser.add_argument('--min_num_clients', type=int,
                        default=2, help='Minimum number of available clients required (default: 2)')
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
    parser.add_argument('--seed', type=int,  default=2022, help='random seed')
    parser.add_argument('--tree_loss_weight', type=float,  default=0.6, help='treeloss_weight')
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
    assert args.role in ['server', 'client']
    assert args.img_class in ['odoc', 'faz']
    if args.img_class == 'faz':
        assert args.sup_type in ['mask', 'scribble', 'block', 'box', 'keypoint']
    else:
        assert args.sup_type in ['mask', 'scribble', 'block', 'scribble_noisy', 'keypoint']

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
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    model = MyModel(net_factory(net_type=args.model, in_chns=args.in_chns, class_num=args.num_classes))
    model.cuda()
    db_train = BaseDataSets(base_dir=args.root_path, split='train', transform=transforms.Compose([
        RandomGenerator(args.patch_size, img_class=args.img_class)
    ]), client=args.client, sup_type=args.sup_type)
    db_val = BaseDataSets(base_dir=args.root_path,
                          client=args.client, split='val')

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    trainloader = DataLoader(db_train, batch_size=args.batch_size, shuffle=True,
                             num_workers=8, pin_memory=True, worker_init_fn=worker_init_fn)
    valloader = DataLoader(db_val, batch_size=1, shuffle=False,
                           num_workers=1)

    if args.role == 'server':
        def fit_config(server_round):
            config = {
                'iter_global': str(server_round),
                'iters': str(args.iters),
                'batch_size': str(args.batch_size),
            }
            return config

        # Create strategy
        strategy = fl.server.strategy.FedAvg(
            fraction_fit=args.sample_fraction,
            min_fit_clients=args.min_num_clients,
            min_available_clients=args.min_num_clients,
            evaluate_fn=get_evaluate_fn(args, valloader, amp=(args.amp == 1)),
            on_fit_config_fn=fit_config,
            fit_metrics_aggregation_fn=fit_metrics_aggregation_fn,
            evaluate_metrics_aggregation_fn=None
        )
        # Start server
        state_dict_keys = model.model.state_dict().keys()
        fl.server.start_server(
            server_address=args.server_address,
            server=MyServer(args=args, writer=writer, state_dict_keys=state_dict_keys, client_manager=SimpleClientManager(), strategy=strategy),
            config=ServerConfig(num_rounds=args.max_iterations, round_timeout=None)
        )
    else:
        client = MyClient(args, model, trainloader, valloader, amp=(args.amp == 1))
        fl.client.start_client(server_address=args.server_address, client=client)



if __name__ == '__main__':
    main()
