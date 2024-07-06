# -*- coding:utf-8 -*-
from multiprocessing import Pool
import os
import time
import copy
import shutil
import random
import argparse
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from torchvision import transforms

import flwr as fl
from flwr.common.logger import log
from flwr.server import ServerConfig
from flwr.server.client_manager import SimpleClientManager
from logging import DEBUG, INFO
import copy

from dataloaders.dataset import BaseDataSets, RandomGenerator
from networks.net_factory import net_factory
from flower_common import (MyModel, fit_metrics_aggregation_fn, get_evaluate_fn, get_strategy,
                        MyServer, VAL_METRICS, get_evaluate_metrics_aggregation_fn)



class Dict2Object(object):
    def __init__(self, d):
        for k, v in d.items():
            if isinstance(k, (list, tuple)):
                setattr(self, k, [Dict2Object(x) if isinstance(x, dict) else x for x in v])
            else:
                setattr(self, k, Dict2Object(v) if isinstance(v, dict) else v)

    def __repr__(self):
        return 'Namespace: {}'.format(self.__dict__)


def main(args, client_class, train_scalar_metrics, train_image_metrics, val_metrics):
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
    assert args.eval_iters > 0 and (args.eval_iters % args.iters == 0)
    assert args.max_iterations > 0 and (args.max_iterations % args.eval_iters == 0)
    if args.strategy == 'FedRep':
        assert args.iters > 0 and args.iters > args.rep_iters
    assert args.role in ['server', 'client']
    assert args.img_class in ['odoc', 'faz']
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
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    model = MyModel(args, net_factory(net_type=args.model, in_chns=args.in_chns, class_num=args.num_classes))
    model.to('cuda: {}'.format(args.gpu))

    db_train = BaseDataSets(base_dir=args.root_path, split='train', transform=transforms.Compose([
        RandomGenerator(args.patch_size, img_class=args.img_class)
    ]), client=args.client, sup_type=args.sup_type)
    db_val = BaseDataSets(base_dir=args.root_path,
                          client=args.client, split='val')

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    trainloader = DataLoader(db_train, batch_size=args.batch_size, shuffle=True,
                             num_workers=4, pin_memory=True, worker_init_fn=worker_init_fn)
    valloader = DataLoader(db_val, batch_size=1, shuffle=False,
                           num_workers=1)

    if args.role == 'server':
        def fit_config(server_round):
            config = {
                'iter_global': str(server_round),
                'iters': str(args.iters),
                'eval_iters': str(args.eval_iters),
                'batch_size': str(args.batch_size),
            }
            return config

        # Create strategy
        kwargs = {
            "fraction_fit": args.sample_fraction,
            "min_fit_clients": args.min_num_clients,
            "min_available_clients": args.min_num_clients,
            "evaluate_fn": get_evaluate_fn(args, valloader, amp=(args.amp == 1)),
            "on_fit_config_fn": fit_config,
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

        strategy = get_strategy(args.strategy, **kwargs)
        # Start server
        state_dict_keys = model.model.state_dict().keys()
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
        client = client_class(args, model, trainloader, valloader, amp=(args.amp == 1))
        fl.client.start_client(server_address=args.server_address, client=client)


def run_client(args, client_class, train_scalar_metrics, train_image_metrics, val_metrics, debug):
    print('Running "{}"\n'.format(args.__dict__))
    if debug == 0:
        main(args, client_class, train_scalar_metrics, train_image_metrics, val_metrics)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int,
                        required=True, help='the communication port')
    parser.add_argument('--debug', type=int,  default=0,
                        help='whether use debug mode')
    # Procedure related arguments
    parser.add_argument('--procedure', type=str,
                        required=True, help='the training procedure')
    parser.add_argument('--exp', type=str,
                        required=True, help='experiment_name')
    parser.add_argument('--gpus', nargs='+', type=int,
                        required=True, help='gpu indexes')
    parser.add_argument('--base_lr', type=float,  default=0.01,
                        help='segmentation network learning rate')
    parser.add_argument('--model', type=str,
                        default='unet', help='model_name')
    parser.add_argument('--img_class', type=str,
                        default='faz', help='the img class(odoc or faz)')
    parser.add_argument('--max_iterations', type=int,
                        default=30000, help='maximum epoch number to train')
    parser.add_argument('--iters', type=int,
                        default=10, help='Number of iters (default: 20)')
    parser.add_argument('--eval_iters', type=int,
                        default=200, help='Number of iters (default: 200)')
    parser.add_argument('--rep_iters', type=int,
                        default=12, help='Number of iters (default: 12)')
    parser.add_argument('--mu', type=float, default=1e-3,
                        help='The hyper parameter for FedProx')
    parser.add_argument('--batch_size', type=int, default=12,
                        help='batch_size per gpu')
    parser.add_argument('--tree_loss_weight', type=float,  default=0.6, help='treeloss_weight')
    parser.add_argument('--strategy', type=str,
                        default='FedAvg', help='Federated learning algorithm (default: FedAvg)')
    parser.add_argument('--amp', type=int,  default=0,
                        help='whether use amp training')
    args = parser.parse_args()

    assert args.img_class in ['odoc', 'faz']
    assert args.procedure in [
        'flower_pCE_2D', 'flower_pCE_MScaleTreeEnergyLoss_ADD', 'flower_pCE_MScaleTreeEnergyLoss_Recurve', 'flower_pCE_TreeEnergyLoss_2D', 'flower_pCE_2D_GatedCRFLoss_v2'
    ]
    assert len(args.gpus) == 6

    if args.img_class == 'faz':
        root_path = '../data/FAZ_h5'
        num_classes = 2
        in_chns = 1
        mask_dict = {
            'client1': 'scribble_noisy',
            'client2': 'keypoint',
            'client3': 'block',
            'client4': 'box',
            'client5': 'scribble'
        }
    if args.img_class =='odoc':
        root_path = '../data/ODOC_h5'
        num_classes = 3
        in_chns = 3
        mask_dict = {
            'client1': 'scribble',
            'client2': 'scribble_noisy',
            'client3': 'scribble_noisy',
            'client4': 'keypoint',
            'client5': 'block'
        }

    common_arg_dict = {
        'root_path': root_path, 'num_classes': num_classes,
        'in_chns': in_chns, 'img_class': args.img_class, 'exp': args.exp, 'model': args.model,
        'max_iterations': args.max_iterations, 'iters': args.iters, 'eval_iters': args.eval_iters,
        'batch_size': args.batch_size, 'base_lr': args.base_lr, 'amp': args.amp, 'server_address': '[::]:{} '.format(args.port),
        'strategy': args.strategy, 'deterministic': True, 'sample_fraction': 1.0, 'patch_size': [256, 256], 'seed': 2022
    }

    client_args = [{'role': 'server', 'min_num_clients': len(mask_dict), 'client': 'client_all', 'sup_type': 'mask', 'gpu': args.gpus[0]}] + \
                    [{'role': 'client', 'cid': i, 'client': client, 'sup_type': sup_type, 'gpu': args.gpus[i + 1]}
                    for i, (client, sup_type) in enumerate(mask_dict.items())]

    if args.procedure in ['flower_pCE_MScaleTreeEnergyLoss_ADD', 'flower_pCE_MScaleTreeEnergyLoss_Recurve', 'flower_pCE_TreeEnergyLoss_2D']:
        common_arg_dict['tree_loss_weight'] = args.tree_loss_weight

    if args.procedure == 'flower_pCE_2D':
        from flower_pCE_2D import MyClient
        client_class = MyClient
        train_scalar_metrics = ['lr', 'total_loss', 'loss_ce']
        train_image_metrics = ['Image', 'Prediction', 'GroundTruth']
        val_metrics = VAL_METRICS
    elif args.procedure == 'flower_pCE_TreeEnergyLoss_2D':
        from flower_pCE_TreeEnergyLoss_2D import MyClient
        client_class = MyClient
        train_scalar_metrics = ['lr', 'total_loss', 'loss_ce', 'out_treeloss']
        train_image_metrics = ['Image', 'Prediction', 'GroundTruth', 'Pseudo'] + \
                                ['Heatmap_{}'.format(i) for i in range(num_classes)]
        val_metrics = VAL_METRICS

    if args.strategy == 'FedProx':
        common_arg_dict['mu'] = args.mu
    elif args.strategy == 'FedRep':
        common_arg_dict['rep_iters'] = args.rep_iters

    pool = Pool(len(client_args))
    for i in range(len(client_args)):
        args_dict = copy.deepcopy(common_arg_dict)
        args_dict.update(client_args[i])
        args_ = Dict2Object(args_dict)
        pool.apply_async(run_client, [args_, client_class, train_scalar_metrics, train_image_metrics, val_metrics, args.debug])
        if i == 0:
            time.sleep(6)
        else:
            time.sleep(1)

    pool.close()
    pool.join()