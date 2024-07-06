# -*- coding:utf-8 -*-
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import flwr as fl
from flwr.common import (GetParametersRes, Status, FitRes, EvaluateRes, ndarrays_to_parameters,
                        parameters_to_ndarrays, GetPropertiesIns, GetPropertiesRes)
from flwr.common.logger import log
from flwr.server.server import Server, fit_clients, evaluate_clients
from flwr.server.history import History
from flwr.server.strategy.strategy import Strategy
from flwr.server.strategy.fedavg import FedAvg
from flwr.server.strategy.fedadagrad import FedAdagrad
from flwr.server.strategy.fedadam import FedAdam
from flwr.server.strategy.fedyogi import FedYogi
from collections import OrderedDict
from logging import DEBUG, INFO, WARNING
from torchvision.utils import make_grid
from tqdm import tqdm

import timeit
import copy
from functools import reduce
import math
from torch.cuda.amp import autocast, GradScaler

from networks.net_factory import net_factory
from val_2D import test_single_volume, test_single_volume_ds
from utils.TreeEnergyLoss.kernels.lib_tree_filter.modules.tree_filter import MinimumSpanningTree
from utils.TreeEnergyLoss.kernels.lib_tree_filter.modules.tree_filter import TreeFilter2D


from sklearn.manifold import TSNE
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import rcParams
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


class BaseClient(fl.client.Client):

    def __init__(self, args, model, trainloader, valloader):
        self.args = args
        self.cid = args.cid
        self.model = model
        self.trainloader = trainloader
        self.valloader = valloader
        self.current_iter = 0
        self.current_lr = self.args.base_lr
        self.sampled_batches = []
        self.properties = {'cid': self.cid}

    def get_parameters(self, ins):
        print('Client {}: get_parameters'.format(self.cid))
        weights = self.model.get_weights(ins.config)
        parameters = fl.common.ndarrays_to_parameters(weights)
        return GetParametersRes(parameters=parameters, status=Status('OK', 'Success'))

    def get_properties(self, ins):
        print('Client {}: get_properties'.format(self.cid))
        return GetPropertiesRes(properties=self.properties, status=Status('OK', 'Success'))

    def fit(self, ins):
        print('Client {}: fit'.format(self.cid))

        weights = fl.common.parameters_to_ndarrays(ins.parameters)
        config = ins.config
        fit_begin = timeit.default_timer()

        self.model.set_weights(weights, config)
        loss, metrics_ = self._train(config)

        weights_prime = self.model.get_weights(config)
        params_prime = fl.common.ndarrays_to_parameters(weights_prime)
        num_examples_train = len(self.trainloader)
        fit_duration = timeit.default_timer() - fit_begin
        metrics_['fit_duration'] = fit_duration

        return FitRes(
            status=Status('OK', 'Success'),
            parameters=params_prime,
            num_examples=num_examples_train,
            metrics=metrics_
        )

    def evaluate(self, ins):
        print('Client {}: evaluate'.format(self.cid))

        weights = fl.common.parameters_to_ndarrays(ins.parameters)
        config = ins.config

        self.model.set_weights(weights, config)
        loss, metrics_ = self._validate(config)

        return EvaluateRes(
            status=Status('OK', 'Success'),
            loss=loss,
            num_examples=len(self.valloader),
            metrics=metrics_
        )

    def _train(self, config):
        raise NotImplementedError

    def _validate(self, config):
        self.model.eval()
        val_metrics = evaluate(self.args, self.model, self.valloader, self.amp)
        if self.args.strategy == 'MetaFed':
            val_metrics['val_uncertainty'] = evaluate_uncertainty(self.args, self.model, self.valloader, self.amp)
            if self.current_iter > self.args.init_iters:
                teacher_val_metrics = evaluate(self.args, self.model.teacher_model, self.valloader, self.amp)
                if self.current_iter <= self.args.common_iters:
                    if val_metrics['val_mean_dice'] > teacher_val_metrics['val_mean_dice']:
                        self.model.meta_flag = True
                    else:
                        self.model.meta_flag = False
                else:
                    if teacher_val_metrics['val_mean_dice'] <= val_metrics['val_mean_dice']:
                        self.model.lam = 0.0
                    else:
                        self.model.lam = (10**(min(1, (teacher_val_metrics['val_mean_dice']-val_metrics['val_mean_dice'])*5)))/10*self.args.lam
                    self.model.meta_flag = True

            print(self.args.cid, self.model.meta_flag, self.model.lam)

        if val_metrics['val_mean_dice'] > self.best_performance:
            self.best_performance = val_metrics['val_mean_dice']
            state_dict = self.model.model.state_dict()
            save_mode_path = os.path.join(self.args.snapshot_path, 'client_{}_async_iter_{}_dice_{}.pth'.format(
                                        self.cid, self.current_iter, round(self.best_performance, 4)))
            save_best = os.path.join(self.args.snapshot_path, 'client_{}_async_{}_best_model.pth'.format(self.cid, self.args.model))
            torch.save(state_dict, save_mode_path)
            torch.save(state_dict, save_best)
            log(INFO, 'save model to {}'.format(save_mode_path))

        if (self.args.strategy in ['FedLC', 'FedALALC', 'FedAPLC', 'FedUni', 'FedUniV2', 'FedUniV2.1']) \
            and (self.current_iter % self.args.tsne_iters == 0):
            tsne_feature_ = tsne_feature(self.args, self.model, self.valloader, self.amp)
            val_metrics['tsne_feature'] = fl.common.ndarray_to_bytes(tsne_feature_.cpu().numpy())

        val_metrics = { 'client_{}_{}'.format(self.cid, k): v for k, v in val_metrics.items() }

        return 0.0, val_metrics


def tsne_feature(args, model, dataloader, amp=False):
    feature_lc_list = []
    for i_batch, sampled_batch in enumerate(dataloader):
        sampled_batch['image'], sampled_batch['label']
        image = sampled_batch['image'].squeeze(1).squeeze(0).cpu().detach().numpy()
        label = sampled_batch['label'].squeeze(1).squeeze(0).cpu().detach().numpy()
        model.eval()

        # ###odoc val
        if len(image.shape) == 3:
            input = torch.from_numpy(image).unsqueeze(0).float().cuda()
        # ###faz val
        elif len(image.shape) == 2:
            input = torch.from_numpy(image).unsqueeze(0).unsqueeze(0).float().cuda()
        with torch.no_grad():
            feature_lc = model(input)[1][-1]
            # print(feature_lc.shape)
            feature_lc = torch.mean(feature_lc, dim=1, keepdim=False)
            # print(feature_lc.shape)
            feature_lc_list.append(feature_lc)

    all_feature_lc = torch.cat(feature_lc_list, dim=0)
    all_feature_lc = torch.flatten(all_feature_lc, start_dim=1)
    # print(all_feature_lc.shape)
    return all_feature_lc


def tsne(n_components, data, label, list):
    params = {
        'font.family':'',
        'font.serif':'',
        'font.style':'normal',
        'font.weight':'normal', 
        'font.size':11,
    }
    rcParams.update(params)
    if n_components == 2:
        tsne = TSNE(n_components=n_components, perplexity=10, random_state=90, n_iter=1000)
        z = tsne.fit_transform(data)
        fig = plt.figure(figsize=(6,6))
        plt.subplot(111)

        df = pd.DataFrame(z)
        df['label'] = label
        df['list'] = list
        # palet = sns.color_palette("hls",2)
        # flatui = ['#f3a598', '#faaf42','#480080','#0fa14a','#7a7c7f']
        flatui = ['#f3a598', '#66c7df','#faaf42','#9659ef','#11b855','#7a7c7f']
        palet = sns.color_palette(flatui)
        markers = {'Site A': "o", 'Site B': 'v', 'Site C': 'H', 'Site D': 's', 'Site E': '^', 'Site F': 'o'}
        alpha = 0.8

        sns.scatterplot(x=z[:,0], y=z[:,1], hue=list, style=list, markers=markers, linewidth = 0.1, 
                        palette=palet, alpha=alpha, data=df)

        plt.xlim(-150, 150)
        plt.ylim(-150, 150)
        plt.legend(loc='lower right')

    elif n_components == 3:
        tsne = TSNE(n_components=n_components, random_state=122, n_iter=2000)
        z = tsne.fit_transform(data)
        fig = plt.figure()
        ax = Axes3D(fig)
        ax.scatter(z[:, 0], z[:, 1], z[:, 2], c=label)
        plt.title("t-SNE Digits")
    else:
        print("The value of n_components can only be 2 or 3")

    return fig


VAL_METRICS = ['dice', 'hd95', 'recall', 'precision', 'jc', 'specificity', 'ravd']
def evaluate(args, model, dataloader, amp=False):
    metric_list = 0.0
    metrics_ = {}
    num = 0
    for i_batch, sampled_batch in enumerate(dataloader):
        metric_i = test_single_volume(
            sampled_batch['image'], sampled_batch['label'], model, classes=args.num_classes, amp=amp)
        if not np.all(metric_i == 0):
            num +=1
        else:
            continue
        metric_list += np.array(metric_i)
    metric_list = metric_list / num
    # print("metric_list.shape =", metric_list.shape)

    for class_i in range(args.num_classes-1):
        for metric_i, metric_name in enumerate(VAL_METRICS):
            metrics_['val_{}_{}'.format(class_i+1, metric_name)] = metric_list[class_i, metric_i]

    for metric_i, metric_name in enumerate(VAL_METRICS):
        metrics_['val_mean_{}'.format(metric_name)] = np.mean(metric_list, axis=0)[metric_i]
    return metrics_


def get_evaluate_fn(args, valloader, amp=False):
    def evaluate_fn(server_round, weights, place):
        model = net_factory(args, net_type=args.model, in_chns=args.in_chns, class_num=args.num_classes)
        state_dict = OrderedDict({
            k: torch.tensor(v) for k, v in zip(model.state_dict().keys(), weights)
        })
        model.load_state_dict(state_dict, strict=True)
        model.cuda()
        model.eval()
        metrics_ = evaluate(args, model, valloader, amp)
        return 0.0, metrics_

    return evaluate_fn


import random
def evaluate_uncertainty(args, model, dataloader, amp=False):
    uncertainty_list = []
    for i_batch, sampled_batch in enumerate(dataloader):
        if args.img_class == 'faz' or args.img_class == 'prostate':
            volume_batch, label_batch = sampled_batch['image'].unsqueeze(1), sampled_batch['label']
            volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()
        elif args.img_class == 'odoc' or args.img_class == 'polyp':
            volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()

        with autocast(enabled=amp):
            rot_times = random.randrange(0, 4)
            rotated_volume_batch = torch.rot90(volume_batch, rot_times, [2, 3])
            T = 8
            _, _, w, h = volume_batch.shape
            volume_batch_r = rotated_volume_batch.repeat(2, 1, 1, 1)
            stride = volume_batch_r.shape[0] // 2
            preds = torch.zeros([stride * T, args.num_classes, w, h]).cuda()
            for i in range(T // 2):
                ema_inputs = volume_batch_r + \
                    torch.clamp(torch.randn_like(
                        volume_batch_r) * 0.1, -0.2, 0.2)
                with torch.no_grad():
                    preds[2 * stride * i:2 * stride *
                          (i + 1)] = model(ema_inputs)[0]
            preds = F.softmax(preds, dim=1)
            preds = preds.reshape(T, stride, args.num_classes, w, h)
            preds = torch.mean(preds, dim=0)
            uncertainty = -1.0 * \
                torch.sum(preds * torch.log(preds + 1e-6), dim=1, keepdim=True)
            uncertainty_list.append(torch.mean(uncertainty).item())

    overall_uncertainty = np.mean(uncertainty_list)
    return overall_uncertainty


class MyServer(Server):

    def __init__(self, args, writer, state_dict_keys, train_scalar_metrics, train_image_metrics, val_metrics, client_manager, strategy):
        super(MyServer, self).__init__(client_manager=client_manager, strategy=strategy)
        self.args = args
        self.writer = writer
        self.state_dict_keys = state_dict_keys
        self.train_scalar_metrics = train_scalar_metrics
        self.train_image_metrics = train_image_metrics
        self.val_metrics = val_metrics

    # pylint: disable=too-many-locals
    def fit(self, num_rounds, timeout):
        '''Run federated averaging for a number of rounds.'''
        history = History()

        # Initialize parameters
        log(INFO, 'Initializing global parameters')
        self.parameters = self._get_initial_parameters(timeout=timeout)
        log(INFO, 'Evaluating initial parameters')
        res = self.strategy.evaluate(0, parameters=self.parameters)
        print(res)
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
        client_id_list = range(min_num_clients)
 
        if len(self.train_image_metrics * 2) > 6:
            nrow = len(self.train_image_metrics)
        else:
            nrow = len(self.train_image_metrics) * 2

        def parameters_to_state_dict(parameters):
            weights = fl.common.parameters_to_ndarrays(parameters)
            state_dict = OrderedDict(
                {k: torch.tensor(v) for k, v in zip(self.state_dict_keys, weights)}
            )
            return state_dict

        def get_fedap_client_state_dict(parameters, client_id, client_parameters):
            num_weights = len(self.state_dict_keys)
            start_idx = 0 + client_id * num_weights
            end_idx = num_weights + client_id * num_weights
            client_weights = parameters_to_ndarrays(parameters)[start_idx:end_idx]
            central_parameters = ndarrays_to_parameters(client_weights)
            client_state_dict = get_client_state_dict(central_parameters, client_parameters)
            return client_state_dict

        def get_client_state_dict(central_parameters, client_parameters):
            central_state_dict = parameters_to_state_dict(central_parameters)
            client_state_dict = parameters_to_state_dict(client_parameters)

            if self.args.strategy == 'MetaFed':
                return client_state_dict

            if self.args.strategy in ['FedBN', 'FedAP', 'FedAPLC']:
                local_keys = []
                for key in self.state_dict_keys:
                    if 'num_batches_tracked' in key:
                        local_keys += [
                            key.replace('num_batches_tracked', 'weight'),
                            key.replace('num_batches_tracked', 'bias'),
                            key.replace('num_batches_tracked', 'running_mean'),
                            key.replace('num_batches_tracked', 'running_var'),
                            key
                        ]
            elif self.args.strategy == 'FedRep':
                local_keys = get_fedrep_local_keys(self.args.model, self.args.in_chns, self.args.num_classes)
            elif self.args.strategy == 'FedUni':
                local_keys = get_feduni_local_keys(self.args.model, self.args.in_chns, self.args.num_classes)
            elif self.args.strategy in ['FedUniV2', 'FedUniV2.1']:
                local_keys = get_feduniv2_local_keys(self.args.model, self.state_dict_keys, self.args.in_chns, self.args.num_classes)
            else:
                local_keys = []

            for key in self.state_dict_keys:
                # print(key)
                if key not in local_keys:
                    client_state_dict[key] = central_state_dict[key]
            return client_state_dict

        best_performance = 0.0
        iterator = tqdm(range(iters, num_rounds+iters, iters), ncols=70)
        for current_round in iterator:
            iter_num = current_round
            # Train model and replace previous global model
            res_fit = self.fit_round(server_round=current_round, timeout=timeout)
            if res_fit[0] is None:
                log(INFO, 'round {}: fit failed'.format(current_round))
                continue

            parameters_prime, metrics_prime, (results_prime, failtures_prime) = res_fit
            self.parameters = parameters_prime
            images = []
            if self.args.strategy in ['FedUniV2', 'FedUniV2.1']:
                prompts_list = []

            for client_id in client_id_list:
                for metric_name in self.train_scalar_metrics:
                    self.writer.add_scalar('info/client_{}_{}'.format(client_id, metric_name), metrics_prime['client_{}_{}'.format(client_id, metric_name)], iter_num)
                for metric_name in self.train_image_metrics:
                    images.append(fl.common.bytes_to_ndarray(metrics_prime['client_{}_{}'.format(client_id, metric_name)]))

                if self.args.strategy == 'FedProx':
                    self.writer.add_scalar('info/client_{}_loss_prox'.format(client_id), metrics_prime['client_{}_loss_prox'.format(client_id)], iter_num)
                elif self.args.strategy == 'MetaFed':
                    self.writer.add_scalar('info/client_{}_loss_meta'.format(client_id), metrics_prime['client_{}_loss_meta'.format(client_id)], iter_num)
                    self.writer.add_scalar('info/client_{}_loss_meta_feature'.format(client_id), metrics_prime['client_{}_loss_meta_feature'.format(client_id)], iter_num)
                    self.writer.add_scalar('info/client_{}_lam'.format(client_id), metrics_prime['client_{}_lam'.format(client_id)], iter_num)
                elif self.args.strategy in ['FedLC', 'FedALALC', 'FedAPLC']:
                    self.writer.add_scalar('info/client_{}_loss_lc'.format(client_id), metrics_prime['client_{}_loss_lc'.format(client_id)], iter_num)
                elif self.args.strategy == 'FedUni':
                    self.writer.add_scalar('info/client_{}_loss_uni'.format(client_id), metrics_prime['client_{}_loss_uni'.format(client_id)], iter_num)
                    self.writer.add_scalar('info/client_{}_distance_prompt'.format(client_id), metrics_prime['client_{}_distance_prompt'.format(client_id)], iter_num)
                elif self.args.strategy in ['FedUniV2', 'FedUniV2.1']:
                    self.writer.add_scalar('info/client_{}_loss_uni'.format(client_id), metrics_prime['client_{}_loss_uni'.format(client_id)], iter_num)
                    self.writer.add_scalar('info/client_{}_loss_pls'.format(client_id), metrics_prime['client_{}_loss_pls'.format(client_id)], iter_num)
                    prompts_list.append(fl.common.bytes_to_ndarray(metrics_prime['client_{}_prompts'.format(client_id)]))
                    self.writer.add_scalar('info/client_{}_distance_prompt'.format(client_id), metrics_prime['client_{}_distance_prompt'.format(client_id)], iter_num)
                    self.writer.add_scalar('info/client_{}_distance_prompt_distribution'.format(client_id), metrics_prime['client_{}_distance_prompt_distribution'.format(client_id)], iter_num)
                    self.writer.add_scalar('info/client_{}_distance_prompt_uni'.format(client_id), metrics_prime['client_{}_distance_prompt_uni'.format(client_id)], iter_num)

                if self.args.strategy in ['MetaFed', 'FedAN']:
                    self.writer.add_scalar('info/client_{}_uncertainty'.format(client_id), metrics_prime['client_{}_uncertainty'.format(client_id)], iter_num)

            self.writer.add_image(
                'train/grid_image',
                make_grid(torch.tensor(np.array(images)), nrow=nrow),
                iter_num
            )

            # Evaluate model using strategy implementation
            if iter_num > 0 and iter_num % self.args.eval_iters == 0:

                if self.args.strategy not in PERSONALIZED_FL:
                    res_cen = self.strategy.evaluate(current_round, parameters=self.parameters)
                    if res_cen is None:
                        log(INFO, 'round {}: evaluate failed'.format(current_round))
                        continue
                    loss_cen, metrics_cen = res_cen
                    log(
                        INFO,
                        'fit progress: (%s, %s, %s, %s)',
                        current_round,
                        loss_cen,
                        metrics_cen,
                        timeit.default_timer() - start_time,
                    )

                res_fed = self.evaluate_round(server_round=current_round, timeout=timeout)
                if res_fed[0] is None:
                    log(INFO, 'round {}: evaluate failed'.format(current_round))
                    continue
                loss_fed, evaluate_metrics_fed, (results_fed, failtures_fed) = res_fed
                # print(loss_fed, evaluate_metrics_fed.keys())
                for client_id in client_id_list:
                    for class_i in range(num_classes-1):
                        for metric_name in self.val_metrics:
                            self.writer.add_scalar('info_client_{}/val_{}_{}'.format(client_id, class_i+1, metric_name),
                                                evaluate_metrics_fed['client_{}_val_{}_{}'.format(client_id, class_i+1, metric_name)], iter_num)
                    for metric_name in self.val_metrics:
                        self.writer.add_scalar('info_client_{}/val_mean_{}'.format(client_id, metric_name),
                                                evaluate_metrics_fed['client_{}_val_mean_{}'.format(client_id, metric_name)], iter_num)
                    self.writer.add_scalar('info/client_{}_val_mean_dice'.format(client_id),
                                        evaluate_metrics_fed['client_{}_val_mean_dice'.format(client_id)], iter_num)

                if (self.args.strategy in ['FedLC', 'FedALALC', 'FedAPLC', 'FedUni', 'FedUniV2', 'FedUniV2.1']) \
                    and (iter_num % self.args.tsne_iters == 0):
                    site_list = ['Site A', 'Site B', 'Site C', 'Site D', 'Site E', 'Site F']
                    tsne_feature_list = []
                    labels = []
                    sites = []
                    for client_id in client_id_list:
                        tsne_feature = fl.common.bytes_to_ndarray(evaluate_metrics_fed['client_{}_tsne_feature'.format(client_id)])
                        tsne_feature_list.append(tsne_feature)
                        labels += [client_id + 1] * tsne_feature.shape[0]
                        sites += [site_list[client_id]] * tsne_feature.shape[0]

                    all_tsne_feature = np.concatenate(tsne_feature_list, axis=0)
                    # print(labels, sites)
                    # print(iter_num, all_tsne_feature.shape, len(labels), len(sites))
                    tsne_figure = tsne(2, all_tsne_feature, labels, sites)
                    self.writer.add_figure('evaluate/tsne_image', tsne_figure, iter_num)

                # print(evaluate_metrics_fed.keys())

                if self.args.strategy not in PERSONALIZED_FL:
                    mean_metrics = metrics_cen
                else:
                    mean_metrics = evaluate_metrics_fed
                # print(mean_metrics.keys())

                for class_i in range(num_classes-1):
                    for metric_name in self.val_metrics:
                        self.writer.add_scalar('info/val_{}_{}'.format(class_i+1, metric_name), mean_metrics['val_{}_{}'.format(class_i+1, metric_name)], iter_num)

                metric_log = 'iteration {} : '.format(iter_num)
                for metric_name in self.val_metrics:
                    metric_log += 'mean_{} : {}; '.format(metric_name, mean_metrics['val_mean_{}'.format(metric_name)])
                    self.writer.add_scalar('info/val_mean_{}'.format(metric_name), mean_metrics['val_mean_{}'.format(metric_name)], iter_num)
                    self.writer.add_scalar('info/val_avg_mean_{}'.format(metric_name), evaluate_metrics_fed['val_avg_mean_{}'.format(metric_name)], iter_num)

                val_mean_dice = mean_metrics['val_mean_dice']
                log(INFO, metric_log)

                if val_mean_dice > best_performance:
                    best_performance = val_mean_dice
                    log(INFO, 'best_performance: {}'.format(best_performance))
                    if self.args.strategy not in PERSONALIZED_FL:
                        state_dict = parameters_to_state_dict(self.parameters)
                        save_mode_path = os.path.join(snapshot_path, 'iter_{}_dice_{}.pth'.format(
                                                    iter_num, round(best_performance, 4)))
                        save_best = os.path.join(snapshot_path, '{}_best_model.pth'.format(self.args.model))
                        torch.save(state_dict, save_mode_path)
                        torch.save(state_dict, save_best)
                        log(INFO, 'save model to {}'.format(save_mode_path))

                    for client_id in client_id_list:
                        first_metric_name = 'client_{}_{}'.format(client_id, self.train_scalar_metrics[0])
                        for _, fit_res in results_prime:
                            # print(client_id, first_metric_name in fit_res.metrics.keys())
                            if first_metric_name in fit_res.metrics.keys():
                                if self.args.strategy in ['FedAP', 'FedAPLC']:
                                    client_state_dict = get_fedap_client_state_dict(self.parameters, client_id, fit_res.parameters)
                                else:
                                    client_state_dict = get_client_state_dict(self.parameters, fit_res.parameters)
                                client_save_mode_path = os.path.join(snapshot_path, 'client_{}_iter_{}_dice_{}.pth'.format(
                                    client_id, iter_num, round(evaluate_metrics_fed['client_{}_val_mean_dice'.format(client_id)], 4)
                                ))
                                client_save_best = os.path.join(snapshot_path, 'client_{}_{}_best_model.pth'.format(
                                                                client_id, self.args.model))
                                torch.save(client_state_dict, client_save_mode_path)
                                torch.save(client_state_dict, client_save_best)
                                log(INFO, 'save model to {}'.format(client_save_mode_path))

            if iter_num > 0 and iter_num % 3000 == 0:
                if self.args.strategy not in PERSONALIZED_FL:
                    state_dict = parameters_to_state_dict(self.parameters)
                    save_mode_path = os.path.join(snapshot_path, 'iter_{}.pth'.format(iter_num))
                    torch.save(state_dict, save_mode_path)
                    log(INFO, 'save model to {}'.format(save_mode_path))

                for client_id in client_id_list:
                    first_metric_name = 'client_{}_{}'.format(client_id, self.train_scalar_metrics[0])
                    for _, fit_res in results_prime:
                        if first_metric_name in fit_res.metrics.keys():
                            if self.args.strategy in ['FedAP', 'FedAPLC']:
                                client_state_dict = get_fedap_client_state_dict(self.parameters, client_id, fit_res.parameters)
                            else:
                                client_state_dict = get_client_state_dict(self.parameters, fit_res.parameters)
                            client_save_mode_path = os.path.join(snapshot_path, 'client_{}_iter_{}.pth'.format(client_id, iter_num))
                            torch.save(client_state_dict, client_save_mode_path)
                            log(INFO, 'save model to {}'.format(client_save_mode_path))


            # MetaFed parameters
            if self.args.strategy == 'MetaFed':
                weight_list = []
                num_examples_list = []
                performance_list = []
                metafed_weight_list = []
                val_dice_list, val_uncertainty_list = [], []
                for client_id in client_id_list:
                    first_metric_name = 'client_{}_{}'.format(client_id, self.train_scalar_metrics[0])
                    for _, fit_res in results_prime:
                        if first_metric_name in fit_res.metrics.keys():
                            weight_list += parameters_to_ndarrays(fit_res.parameters)
                            num_examples_list.append(fit_res.num_examples)
                            val_dice_list.append(evaluate_metrics_fed['client_{}_val_mean_dice'.format(client_id)])
                            val_uncertainty_list.append(evaluate_metrics_fed['client_{}_val_uncertainty'.format(client_id)])
                            self.writer.add_scalar('info_client_{}/val_uncertainty'.format(client_id), evaluate_metrics_fed['client_{}_val_uncertainty'.format(client_id)], iter_num)

                if iter_num <= self.args.common_iters:
                    for client_id in client_id_list:
                        performance = val_dice_list[client_id] + self.args.sort_lam * (1 / val_uncertainty_list[client_id])
                        performance_list.append(performance)
                    performance_array = np.array(performance_list)
                    metafed_weight_array = np.zeros((len(client_id_list), len(client_id_list)))
                else:
                    performance_array = np.zeros((5, ))
                    # FedAvg
                    '''metafed_weight_list = num_examples_list
                    metafed_weight_array = np.array([ metafed_weight_list for _ in range(len(client_id_list))])'''
                    # FedAN (FedMix)
                    '''num_examples_total = sum(num_examples_list)
                    inverse_uncertainty_total = sum([
                        (1 / val_uncertainty_list[client_id]) ** self.args.sort_beta
                        for client_id in client_id_list
                    ])
                    for client_id in client_id_list:
                        normalized_inverse_uncertainty = ((1 / val_uncertainty_list[client_id]) ** self.args.sort_beta) / inverse_uncertainty_total
                        normalized_num_examples = num_examples_list[client_id] / num_examples_total
                        metafed_weight = (normalized_num_examples + self.args.sort_lam * normalized_inverse_uncertainty) / (1 + self.args.sort_lam * normalized_inverse_uncertainty)
                        metafed_weight_list.append(metafed_weight)
                    metafed_weight_array = np.array([ metafed_weight_list for _ in range(len(client_id_list))])'''
                    # FedAP
                    model_momentum=0.5
                    if iter_num <= self.args.common_iters + self.args.iters:
                        bnmlist, bnvlist = [], []
                        for client_id in client_id_list:
                            bnm, bnv = [], []
                            bnl = metrics_prime['client_{}_bnl'.format(client_id)]
                            for i_bn in range(bnl):
                                bnm_key = 'client_{}_bnm_{}'.format(client_id, i_bn)
                                bnv_key = 'client_{}_bnv_{}'.format(client_id, i_bn)
                                bnm.append( fl.common.bytes_to_ndarray(metrics_prime[bnm_key]) )
                                bnv.append( fl.common.bytes_to_ndarray(metrics_prime[bnv_key]) )
                            bnmlist.append(bnm)
                            bnvlist.append(bnv)
                        self.weight_matrix1 = get_weight_matrix1(model_momentum, bnmlist, bnvlist)
                    metafed_weight_array = self.weight_matrix1

                weight_list.append(performance_array)
                weight_list.append(metafed_weight_array)
                self.parameters = ndarrays_to_parameters(weight_list)
                '''print(iter_num, 'val_dice_list', val_dice_list)
                print(iter_num, 'val_uncertainty_list', val_uncertainty_list)
                print(iter_num, 'performance_array', performance_array)
                print(iter_num, 'metafed_weight_array', metafed_weight_array)'''

            # FedUniV2/FedUniV2.1 parameters
            if self.args.strategy in ['FedUniV2', 'FedUniV2.1']:
                weight_list = []
                num_examples_list = []
                performance_list = []
                for client_id in client_id_list:
                    first_metric_name = 'client_{}_{}'.format(client_id, self.train_scalar_metrics[0])
                    for _, fit_res in results_prime:
                        if first_metric_name in fit_res.metrics.keys():
                            print(first_metric_name)
                            weight_list += parameters_to_ndarrays(fit_res.parameters)
                            num_examples_list.append(fit_res.num_examples)
                            performance_list.append(evaluate_metrics_fed['client_{}_val_mean_dice'.format(client_id)])

                weight_list += parameters_to_ndarrays(self.parameters)
                performance_array = np.array(performance_list)
                weight_list.append(performance_array)
                prompts_array = np.mean(prompts_list, axis=0)
                weight_list.append(prompts_array)
                self.parameters = ndarrays_to_parameters(weight_list)
                '''print(iter_num, 'val_dice_list', val_dice_list)
                print(iter_num, 'performance_array', performance_array)'''


            if iter_num >= max_iterations:
                break

        # Bookkeeping
        end_time = timeit.default_timer()
        elapsed = end_time - start_time
        log(INFO, 'FL finished in %s', elapsed)
        return history


def fit_metrics_aggregation_fn(fit_metrics):
    metrics = { k: v for _, client_metrics in fit_metrics for k, v in client_metrics.items() }
    return metrics


def get_evaluate_metrics_aggregation_fn(args, val_metrics):
    def evaluate_metrics_aggregation_fn(evaluate_metrics):
        metrics = { k: v for _, client_metrics in evaluate_metrics for k, v in client_metrics.items() }
        weights = {}
        for client_id in range(args.min_num_clients):
            first_metric_name = 'client_{}_val_mean_{}'.format(client_id, val_metrics[0])
            for client_num_examples, client_metrics in evaluate_metrics:
                if first_metric_name in client_metrics.keys():
                    weights['client_{}'.format(client_id)] = client_num_examples
        # print(weights)

        def weighted_metric(metric_name):
            num_total_examples = sum([client_num_examples for client_num_examples in weights.values()])
            weighted_metric = [weights['client_{}'.format(client_id)] * metrics['client_{}_{}'.format(client_id, metric_name)]
                                for client_id in range(args.min_num_clients)]
            return sum(weighted_metric) / num_total_examples

        def mean_metric(metric_name):
            return np.mean([metrics['client_{}_{}'.format(client_id, metric_name)]
                            for client_id in range(args.min_num_clients)])

        metrics.update({'val_{}_{}'.format(class_i+1, metric_name): weighted_metric('val_{}_{}'.format(class_i+1, metric_name))
                        for class_i in range(args.num_classes-1) for metric_name in val_metrics})
        metrics.update({'val_mean_{}'.format(metric_name): weighted_metric('val_mean_{}'.format(metric_name))
                        for metric_name in val_metrics})
        metrics.update({'val_avg_mean_{}'.format(metric_name): mean_metric('val_mean_{}'.format(metric_name))
                        for metric_name in val_metrics})

        return metrics

    return evaluate_metrics_aggregation_fn


PERSONALIZED_FL = ['FedBN', 'FedAP', 'FedRep', 'FedLC', 'MetaFed', 'FedALA', 'FedALALC', 'FedAPLC', 'FedUni', 'FedUniV2', 'FedUniV2.1']
CENTRALIZED_FL = ['FedAvg', 'FedAN', 'FedAdagrad', 'FedAdam', 'FedYogi', 'FedProx']
def get_strategy(name, **kwargs):
    assert name in (CENTRALIZED_FL + PERSONALIZED_FL)
    if name == 'FedAvg':
        strategy = FedAvg(**kwargs)
    elif name == 'FedAdagrad':
        strategy = FedAdagrad(**kwargs)
    elif name == 'FedAdam':
        strategy = FedAdam(**kwargs)
    elif name == 'FedYogi':
        strategy = FedYogi(**kwargs)
    elif name == 'FedProx':
        strategy = FedProx(**kwargs)
    elif name == 'FedBN':
        strategy = FedBN(**kwargs)
    elif name == 'FedAP':
        strategy = FedAP(**kwargs)
    elif name == 'FedRep':
        strategy = FedRep(**kwargs)
    elif name == 'FedLC':
        strategy = FedLC(**kwargs)
    elif name == 'MetaFed':
        strategy = MetaFed(**kwargs)
    elif name == 'FedAN':
        strategy = FedAN(**kwargs)
    elif name == 'FedALA':
        strategy = FedALA(**kwargs)
    elif name == 'FedALALC':
        strategy = FedALALC(**kwargs)
    elif name == 'FedAPLC':
        strategy = FedAPLC(**kwargs)
    elif name == 'FedUni':
        strategy = FedUni(**kwargs)
    elif name in ['FedUniV2', 'FedUniV2.1']:
        strategy = FedUniV2(**kwargs)
    else:
        raise NotImplementedError

    return strategy


class FedProx(FedAvg):

    def __repr__(self) -> str:
        rep = f"FedProx(accept_failures={self.accept_failures})"
        return rep


class FedBN(FedAvg):

    def __repr__(self) -> str:
        rep = f"FedBN(accept_failures={self.accept_failures})"
        return rep


def get_form(model):
    tmpm = []
    tmpv = []
    for name in model.state_dict().keys():
        if 'encoder' not in name:
            continue
        if 'running_mean' in name:
            tmpm.append(model.state_dict()[name].detach().to('cpu').numpy())
            # print(name)
        if 'running_var' in name:
            tmpv.append(model.state_dict()[name].detach().to('cpu').numpy())
    return tmpm, tmpv


def get_wasserstein(m1, v1, m2, v2, mode='nosquare'):
    w = 0
    bl = len(m1)
    for i in range(bl):
        tw = 0
        tw += (np.sum(np.square(m1[i]-m2[i])))
        tw += (np.sum(np.square(np.sqrt(v1[i]) - np.sqrt(v2[i]))))
        if mode == 'square':
            w += tw
        else:
            w += math.sqrt(tw)
    return w


class metacount(object):
    def __init__(self, numpyform):
        super(metacount, self).__init__()
        self.count = 0
        self.mean = []
        self.var = []
        self.bl = len(numpyform)
        for i in range(self.bl):
            self.mean.append(np.zeros(len(numpyform[i])))
            self.var.append(np.zeros(len(numpyform[i])))

    def update(self, m, tm, tv):
        tmpcount = self.count+m
        for i in range(self.bl):
            tmpm = (self.mean[i]*self.count + tm[i]*m)/tmpcount
            self.var[i] = (self.count*(self.var[i]+np.square(tmpm -
                           self.mean[i])) + m*(tv[i]+np.square(tmpm-tm[i])))/tmpcount
            self.mean[i] = tmpm
        self.count = tmpcount

    def getmean(self):
        return self.mean

    def getvar(self):
        return self.var


def get_bn_stats(args, model, trainloader):
    # print(model.state_dict().keys())
    def get_feature_list(x):
        feature_list = []
        for block in model.encoder.in_conv.conv_conv:
            # print(x.shape, type(block), type(model.encoder.in_conv.conv_conv))
            if isinstance(block, nn.BatchNorm2d):
                feature_list.append(x.clone().detach())
            x = block(x)

        x = model.encoder.down1.maxpool_conv[0](x)
        for block in model.encoder.down1.maxpool_conv[1].conv_conv:
            if isinstance(block, nn.BatchNorm2d):
                feature_list.append(x.clone().detach())
            x = block(x)

        x = model.encoder.down2.maxpool_conv[0](x)
        for block in model.encoder.down2.maxpool_conv[1].conv_conv:
            if isinstance(block, nn.BatchNorm2d):
                feature_list.append(x.clone().detach())
            x = block(x)

        x = model.encoder.down3.maxpool_conv[0](x)
        for block in model.encoder.down3.maxpool_conv[1].conv_conv:
            if isinstance(block, nn.BatchNorm2d):
                feature_list.append(x.clone().detach())
            x = block(x)

        x = model.encoder.down4.maxpool_conv[0](x)
        for block in model.encoder.down4.maxpool_conv[1].conv_conv:
            if isinstance(block, nn.BatchNorm2d):
                feature_list.append(x.clone().detach())
            x = block(x)

        return feature_list

    model.eval()
    for i_batch, sampled_batch in enumerate(trainloader):
        if args.img_class == 'faz' or args.img_class == 'prostate':
            volume_batch, label_batch = sampled_batch['image'].unsqueeze(1), sampled_batch['label']
            volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()
        elif args.img_class == 'odoc' or args.img_class == 'polyp':
            volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()

    avgmeta = metacount(get_form(model)[0])
    with torch.no_grad():
        for i_batch, sampled_batch in enumerate(trainloader):
            if args.img_class == 'faz' or args.img_class == 'prostate':
                volume_batch, label_batch = sampled_batch['image'].unsqueeze(1), sampled_batch['label']
                volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()
            elif args.args.img_class == 'odoc' or args.img_class == 'polyp':
                volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
                volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()

            fea = get_feature_list(volume_batch)
            nl = len(sampled_batch)
            tm, tv = [], []
            i = 0
            for item in fea:
                if len(item.shape) == 4:
                    tm.append(torch.mean(
                        item, dim=[0, 2, 3]).detach().to('cpu').numpy())
                    tv.append(
                        torch.var(item, dim=[0, 2, 3]).detach().to('cpu').numpy())
                else:
                    tm.append(torch.mean(
                        item, dim=0).detach().to('cpu').numpy())
                    tv.append(
                        torch.var(item, dim=0).detach().to('cpu').numpy())

            avgmeta.update(nl, tm, tv)

    bnm = avgmeta.getmean()
    bnv = avgmeta.getvar()
    model.train()
    return bnm, bnv


def get_weight_matrix1(model_momentum, bnmlist, bnvlist):
    client_num = len(bnmlist)
    weight_m = np.zeros((client_num, client_num))
    for i in range(client_num):
        for j in range(client_num):
            if i == j:
                weight_m[i, j] = 0
            else:
                tmp = get_wasserstein(
                    bnmlist[i], bnvlist[i], bnmlist[j], bnvlist[j])
                if tmp == 0:
                    weight_m[i, j] = 100000000000000
                else:
                    weight_m[i, j] = 1/tmp
    weight_s = np.sum(weight_m, axis=1)
    weight_s = np.repeat(weight_s, client_num).reshape(
        (client_num, client_num))
    weight_m = (weight_m/weight_s)*(1-model_momentum)
    for i in range(client_num):
        weight_m[i, i] = model_momentum
    return weight_m


from dataloaders.dataset import BaseDataSets, RandomGenerator
from torch.utils.data import Dataset
class PretrainDataset(Dataset):

    def __init__(self, root_path, sup_type_list, transform) :
        super().__init__()
        self.transform = transform
        self.sample_list = []
        for i_client, sup_type in enumerate(sup_type_list):
            db_train = BaseDataSets(base_dir=root_path, split='train', transform=None, client='client{}'.format(i_client + 1), sup_type=sup_type)
            self.sample_list += [sample for sample in db_train]

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        sample = self.sample_list[idx]
        if self.transform:
            sample = self.transform(sample)
        return sample


class FedAP(FedAvg):

    def __init__(self, fraction_fit=1.0, fraction_evaluate=1.0, min_fit_clients=2, min_evaluate_clients=2,
                min_available_clients=2, evaluate_fn=None, on_fit_config_fn=None, on_evaluate_config_fn=None, 
                accept_failures=True, initial_parameters=None, fit_metrics_aggregation_fn=None, evaluate_metrics_aggregation_fn=None, iters=10, model_momentum=0.5):
        super(FedAP, self).__init__(
            fraction_fit=fraction_fit,
            fraction_evaluate=fraction_evaluate,
            min_fit_clients=min_fit_clients,
            min_evaluate_clients=min_evaluate_clients,
            min_available_clients=min_available_clients,
            evaluate_fn=evaluate_fn,
            on_fit_config_fn=on_fit_config_fn,
            on_evaluate_config_fn=on_evaluate_config_fn,
            accept_failures=accept_failures,
            initial_parameters=initial_parameters,
            fit_metrics_aggregation_fn=fit_metrics_aggregation_fn,
            evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn,
        )
        self.iters = iters
        self.model_momentum = model_momentum

    def _aggregate(self, server_round, results):
        """Compute weighted average."""
        sorted_weights = []
        sorted_num_examples = []
        sorted_metrics = []
        for client_id in range(len(results)):
            for weights, num_examples, metrics in results:
                if 'client_{}_lr'.format(client_id) in metrics.keys():
                    # print('client_{} _aggregate'.format(client_id))
                    sorted_weights.append(weights)
                    sorted_num_examples.append(num_examples)
                    sorted_metrics.append(metrics)

        # Create a list of weights, each multiplied by the related number of examples
        if server_round <= self.iters:
            bnmlist, bnvlist = [], []
            for client_id, metrics in enumerate(sorted_metrics):
                print('client_{}_bnl'.format(client_id), metrics['client_{}_bnl'.format(client_id)])
                bnm, bnv = [], []
                bnl = metrics['client_{}_bnl'.format(client_id)]
                for i_bn in range(bnl):
                    bnm_key = 'client_{}_bnm_{}'.format(client_id, i_bn)
                    bnv_key = 'client_{}_bnv_{}'.format(client_id, i_bn)
                    bnm.append( fl.common.bytes_to_ndarray(metrics[bnm_key]) )
                    bnv.append( fl.common.bytes_to_ndarray(metrics[bnv_key]) )
                bnmlist.append(bnm)
                bnvlist.append(bnv)
            self.fedap_weights = get_weight_matrix1(self.model_momentum, bnmlist, bnvlist)

        print(server_round, [sum(self.fedap_weights[client_id]) for client_id in range(len(results))], 'fedap_weights', self.fedap_weights)
        # print(server_round, type(self.fedap_weights), sum(self.fedap_weights[client_id]))

        all_weights_prime = []
        for client_id in range(len(results)):
            weighted_weights = [
                [layer * fedap_weight for layer in weights]
                for weights, fedap_weight in zip(sorted_weights, self.fedap_weights[client_id])
            ]

            # Compute average weights of each layer
            weights_prime = [
                reduce(np.add, layer_updates) / sum(self.fedap_weights[client_id])
                for layer_updates in zip(*weighted_weights)
            ]
            all_weights_prime += weights_prime

        return all_weights_prime

    def aggregate_fit(self, server_round, results, failures):
        """Aggregate fit results using weighted average."""
        if not results:
            return None, {}
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}

        # Convert results
        weights_results = [
            (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples, fit_res.metrics)
            for _, fit_res in results
        ]
        parameters_aggregated = ndarrays_to_parameters(self._aggregate(server_round, weights_results))

        # Aggregate custom metrics if aggregation fn was provided
        metrics_aggregated = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)
        elif server_round == 1:  # Only log this warning once
            log(WARNING, "No fit_metrics_aggregation_fn provided")

        return parameters_aggregated, metrics_aggregated

    def __repr__(self) -> str:
        rep = f"FedAP(accept_failures={self.accept_failures})"
        return rep


class FedRep(FedAvg):

    def __repr__(self) -> str:
        rep = f"FedRep(accept_failures={self.accept_failures})"
        return rep


class FedAPLC(FedAP):

    def __repr__(self) -> str:
        rep = f"FedAPLC(accept_failures={self.accept_failures})"
        return rep


class FedAN(FedAvg):

    def __init__(self, fraction_fit=1.0, fraction_evaluate=1.0, min_fit_clients=2, min_evaluate_clients=2,
                min_available_clients=2, evaluate_fn=None, on_fit_config_fn=None, on_evaluate_config_fn=None, 
                accept_failures=True, initial_parameters=None, fit_metrics_aggregation_fn=None, evaluate_metrics_aggregation_fn=None, iters=10, lam=1, beta=1):
        super(FedAN, self).__init__(
            fraction_fit=fraction_fit,
            fraction_evaluate=fraction_evaluate,
            min_fit_clients=min_fit_clients,
            min_evaluate_clients=min_evaluate_clients,
            min_available_clients=min_available_clients,
            evaluate_fn=evaluate_fn,
            on_fit_config_fn=on_fit_config_fn,
            on_evaluate_config_fn=on_evaluate_config_fn,
            accept_failures=accept_failures,
            initial_parameters=initial_parameters,
            fit_metrics_aggregation_fn=fit_metrics_aggregation_fn,
            evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn,
        )
        self.iters = iters
        self.lam = lam
        self.beta = beta

    def _aggregate(self, server_round, results):
        """Compute weighted average."""
        sorted_weights = []
        sorted_num_examples = []
        sorted_metrics = []
        for client_id in range(len(results)):
            for weights, num_examples, metrics in results:
                if 'client_{}_lr'.format(client_id) in metrics.keys():
                    # print('client_{} _aggregate'.format(client_id))
                    sorted_weights.append(weights)
                    sorted_num_examples.append(num_examples)
                    sorted_metrics.append(metrics)

        # Create a list of weights, each multiplied by the related number of examples
        fedan_weights = []
        num_examples_total = sum(sorted_num_examples)
        inverse_uncertainty_total = sum([
            (1 / sorted_metrics[client_id]['client_{}_uncertainty'.format(client_id)]) ** self.beta
            for client_id in range(len(results))
        ])
        for client_id, metrics in enumerate(sorted_metrics):
            print('client_{}_uncertainty'.format(client_id), metrics['client_{}_uncertainty'.format(client_id)])
            normalized_inverse_uncertainty = ((1 / metrics['client_{}_uncertainty'.format(client_id)]) ** self.beta) / inverse_uncertainty_total
            normalized_num_examples = sorted_num_examples[client_id] / num_examples_total
            fedan_weight = (normalized_num_examples + self.lam * normalized_inverse_uncertainty) / (1 + self.lam * normalized_inverse_uncertainty)
            '''uncertainty = metrics['client_{}_uncertainty'.format(client_id)] 
            fedan_weight = 1 / (uncertainty + 1e-6)'''
            fedan_weights.append(fedan_weight)

        print(server_round, sum(fedan_weights), 'fedan_weights', fedan_weights)

        weighted_weights = [
            [layer * fedan_weight for layer in weights]
            for weights, fedan_weight in zip(sorted_weights, fedan_weights)
        ]

        # Compute average weights of each layer
        weights_prime = [
            reduce(np.add, layer_updates) / sum(fedan_weights)
            for layer_updates in zip(*weighted_weights)
        ]

        return weights_prime

    def aggregate_fit(self, server_round, results, failures):
        """Aggregate fit results using weighted average."""
        if not results:
            return None, {}
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}

        # Convert results
        weights_results = [
            (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples, fit_res.metrics)
            for _, fit_res in results
        ]
        parameters_aggregated = ndarrays_to_parameters(self._aggregate(server_round, weights_results))

        # Aggregate custom metrics if aggregation fn was provided
        metrics_aggregated = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)
        elif server_round == 1:  # Only log this warning once
            log(WARNING, "No fit_metrics_aggregation_fn provided")

        return parameters_aggregated, metrics_aggregated

    def __repr__(self) -> str:
        rep = f"FedAN(accept_failures={self.accept_failures})"
        return rep


class FedLC(FedAvg):

    def __repr__(self) -> str:
        rep = f"FedLC(accept_failures={self.accept_failures})"
        return rep


class MetaFed(FedAvg):

    def __repr__(self) -> str:
        rep = f"MetaFed(accept_failures={self.accept_failures})"
        return rep


class FedALA(FedAvg):

    def __repr__(self) -> str:
        rep = f"FedALA(accept_failures={self.accept_failures})"
        return rep


class FedALALC(FedAvg):

    def __repr__(self) -> str:
        rep = f"FedALALC(accept_failures={self.accept_failures})"
        return rep


class FedUni(FedAvg):

    def __repr__(self) -> str:
        rep = f"FedUni(accept_failures={self.accept_failures})"
        return rep


class FedUniV2(FedAvg):

    def __repr__(self) -> str:
        rep = f"FedUniV2(accept_failures={self.accept_failures})"
        return rep


def get_fedrep_local_keys(model_name, in_chns, num_classes):
    assert model_name in ['unet', 'unet_head']
    if model_name == 'unet':
        local_keys = ['decoder.out_conv.weight', 'decoder.out_conv.bias']
    elif model_name == 'unet_head':
        local_keys = ['decoder.out_conv.weight', 'decoder.out_conv.bias', 'decoder.dsn_head.4.weight']
        '''local_keys = []
        model = net_factory(net_type=model_name, in_chns=in_chns, class_num=num_classes)
        for name, param in model.named_parameters():
            if 'out_conv' in name or 'dsn_head' in name:
                local_keys += [name]'''
    else:
        local_keys = []

    return local_keys


def get_fedlc_local_keys(model_name, in_chns, num_classes):
    assert model_name in ['lcnet']
    if model_name == 'lcnet':
        local_keys = []
    else:
        local_keys = []

    return local_keys


def get_feduni_local_keys(model_name, in_chns, num_classes):
    assert model_name in ['unet_uni']
    if model_name == 'unet_uni':
        local_keys = ['decoder.out_conv.weight', 'decoder.out_conv.bias']
    else:
        local_keys = []


def get_feduniv2_local_keys(model_name, model_state_dict_keys, in_chns, num_classes):
    assert model_name in ['unet_univ2', 'unet_univ3', 'unet_univ4', 'unet_univ5','unet_univ5_attention_concat'] 
    p_keywords = []
    if model_name == 'unet_univ2' or model_name == 'unet_univ3' or model_name == 'unet_univ4' or 'unet_univ5' or 'unet_univ5_attention_concat':
        local_keys = ['decoder.out_conv.weight', 'decoder.out_conv.bias']
        # for name in model_state_dict_keys:
        #         for p_keyword in p_keywords:
        #             if p_keyword in name:
        #                 local_keys.append(name)
    else:
        local_keys = []

    return local_keys


class MyModel(nn.Module):

    def __init__(self, args, model, trainloader, valloader):
        super(MyModel, self).__init__()
        self.args = args
        self.model = model
        self.trainloader = trainloader
        self.valloader = valloader
        self.amp=(args.amp == 1)
        if self.amp:
            self.scaler = GradScaler()

        num_params = 0
        for key in self.model.state_dict().keys():
            num_params += self.model.state_dict()[key].numel()
        print('{} parameters: {:.2f}M'.format(self.args.model, num_params / 1e6))
        # print(*self.model.state_dict().keys())
        '''if self.args.cid == 0:
            for name, param in self.model.named_parameters():
                print(name, param.shape)'''

        if self.args.strategy in ['FedBN', 'FedAP', 'FedAPLC']:
            self.local_keys = []
            for key in self.model.state_dict().keys():
                if 'num_batches_tracked' in key:
                    self.local_keys += [
                        key.replace('num_batches_tracked', 'weight'),
                        key.replace('num_batches_tracked', 'bias'),
                        key.replace('num_batches_tracked', 'running_mean'),
                        key.replace('num_batches_tracked', 'running_var'),
                        key
                    ]
        elif self.args.strategy == 'FedRep':
            self.local_keys = get_fedrep_local_keys(self.args.model, self.args.in_chns, self.args.num_classes)
        elif self.args.strategy == 'FedUni':
            self.local_keys = get_feduni_local_keys(self.args.model, self.args.in_chns, self.args.num_classes)
        elif self.args.strategy in ['FedUniV2', 'FedUniV2.1']:
            self.local_keys = get_feduniv2_local_keys(self.args.model, self.model.state_dict().keys(), self.args.in_chns, self.args.num_classes)

        if hasattr(self, 'local_keys'):
            print('client {} local_keys:'.format(self.args.cid), self.local_keys)

        if self.args.strategy == 'MetaFed':
            self.meta_flag = False
            self.lam = self.args.lam
            self.teacher_model = copy.deepcopy(self.model).cuda()
            self.client_id_list = [client_id for client_id in range(self.args.min_num_clients)]
            self.teacher_list = [2, 0, 4, 1, 3]
        elif self.args.strategy in ['FedALA', 'FedALALC']:
            self.start_phase = True
        elif self.args.strategy in ['FedUniV2', 'FedUniV2.1']:
            self.start_phase = True
            self.client_id_list = [client_id for client_id in range(self.args.min_num_clients)]
            self.decoder_auxiliary_keys = [
                k for k in self.model.state_dict().keys() if 'decoder_auxiliary' in k
            ]
            

    def forward(self, x):
        return self.model(x)

    def get_weights(self, config):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_weights(self, weights, config):
        # print('Setting weights')
        # MetaFed
        if self.args.strategy == 'MetaFed':
            def get_teacher_id_v1(weight_dict):
                metric_list = []
                for client_id in self.client_id_list:
                    state_dict = OrderedDict({
                        k: torch.tensor(v) for k, v in zip(self.model.state_dict().keys(), weight_dict['client_{}'.format(client_id)])
                    })
                    self.teacher_model.load_state_dict(state_dict, strict=True)
                    self.teacher_model.cuda()
                    self.teacher_model.eval()
                    '''metrics_ = evaluate(self.args, self.teacher_model, self.valloader, amp=(self.args.amp==1))
                    metric_list.append(metrics_['val_mean_dice'])'''
                    uncertainty = evaluate_uncertainty(self.args, self.teacher_model, self.valloader, amp=(self.args.amp==1))
                    metric_list.append(uncertainty)

                # teacher_id = torch.argmax(torch.tensor(metric_list))
                teacher_id = torch.argmin(torch.tensor(metric_list))
                # print(self.args.cid, metric_list, teacher_id)
                return teacher_id, metric_list

            def get_teacher_id_v2(performance_list):
                sorted_performance_idxs = sorted(range(len(performance_list)), key=lambda k: performance_list[k], reverse=True)
                performance_map = {
                    str(sorted_performance_idxs[i_idx]): sorted_performance_idxs[i_idx - 1]
                    for i_idx in range(len(sorted_performance_idxs))
                }
                # print(self.args.cid, performance_list, sorted_performance_idxs, performance_map)
                teacher_id = performance_map[str(self.args.cid)]
                return teacher_id

            if config['stage'] == 'fit':
                if config['iter_global'] <= self.args.iters:
                    state_dict = OrderedDict({
                        k: torch.tensor(v) for k, v in zip(self.model.state_dict().keys(), weights)
                    })
                    self.model.load_state_dict(state_dict, strict=False)
                else:
                    weight_dict = {}
                    all_state_dict_temp = {}
                    num_weights = len(self.model.state_dict().items())
                    for client_id in self.client_id_list:
                        start_idx = 0 + client_id * num_weights
                        end_idx = num_weights + client_id * num_weights
                        weight_dict['client_{}'.format(client_id)] = weights[start_idx:end_idx]
                        all_state_dict_temp['client_{}'.format(client_id)] = {
                            k: torch.tensor(v) for k, v in zip(self.model.state_dict().keys(), weights[start_idx:end_idx])
                        }
                    performance_idx = num_weights + client_id * num_weights
                    performance_list = weights[performance_idx]
                    metafed_weight_idx = num_weights + client_id * num_weights + 1
                    metafed_weight_list = weights[metafed_weight_idx][self.args.cid]
                    print(self.args.cid, performance_list, metafed_weight_list)

                     # teacher model
                    if config['iter_global'] <= self.args.common_iters + self.args.iters:
                        # teacher_id = self.client_id_list[int(self.args.cid) - 1]
                        # teacher_id = self.teacher_list[int(self.args.cid)]
                        # teacher_id, _ = get_teacher_id_v1(weight_dict)
                        teacher_id = get_teacher_id_v2(performance_list)
                        print('client_{}'.format(teacher_id), self.args.cid, config, not self.meta_flag)
                        teacher_state_dict_temp = {
                            k: torch.tensor(v)
                            for k, v in zip(self.model.state_dict().keys(), weight_dict['client_{}'.format(teacher_id)])
                        }
                    else:
                        teacher_state_dict_temp = {}
                        for k, v in zip(self.model.state_dict().keys(), weight_dict['client_{}'.format(self.args.cid)]):
                            if 'num_batches_tracked' in k:
                                continue
                            teacher_state_dict_temp[k] = torch.zeros_like(torch.tensor(v))
                            for client_id in self.client_id_list:
                                teacher_state_dict_temp[k] += (all_state_dict_temp['client_{}'.format(client_id)][k] * metafed_weight_list[client_id])
                            teacher_state_dict_temp[k] = teacher_state_dict_temp[k] / sum(metafed_weight_list)

                    self.teacher_model.load_state_dict(OrderedDict(teacher_state_dict_temp), strict=True)

                    # student model
                    if not self.meta_flag:
                        print('test')
                        local_keys = []
                        for key in self.model.state_dict().keys():
                            if 'num_batches_tracked' in key:
                                local_keys.append(key.replace('num_batches_tracked', 'weight'))
                                local_keys.append(key.replace('num_batches_tracked', 'bias'))
                                local_keys.append(key.replace('num_batches_tracked', 'running_mean'))
                                local_keys.append(key.replace('num_batches_tracked', 'running_var'))
                                local_keys.append(key)
                                # print(self.args.cid, key)
                            '''if 'encoder' not in key:
                                local_keys.append(key)
                                # print(self.args.cid, key)'''
                        local_keys = set(local_keys)
                        # print(self.args.cid, local_keys)
                        '''for key in local_keys:
                            teacher_state_dict_temp.pop(key)'''
                        self.model.load_state_dict(OrderedDict(teacher_state_dict_temp), strict=False)
                    else:
                        state_dict = OrderedDict({
                            k: torch.tensor(v)
                            for k, v in zip(self.model.state_dict().keys(), weight_dict['client_{}'.format(self.args.cid)])
                        })
                        self.model.load_state_dict(state_dict, strict=False)
            '''else:
                state_dict = OrderedDict({
                    k: torch.tensor(v)
                    for k, v in zip(self.model.state_dict().keys(), weights)
                })
                self.model.load_state_dict(state_dict, strict=False)'''

        # FedAP, FedAPLC
        elif self.args.strategy in ['FedAP', 'FedAPLC']:
            if config['stage'] == 'fit' and config['iter_global'] <= self.args.iters:
                state_dict = OrderedDict({
                    k: torch.tensor(v) for k, v in zip(self.model.state_dict().keys(), weights)
                })
                self.model.load_state_dict(state_dict, strict=False)
            else:
                num_weights = len(self.model.state_dict().items())
                all_state_dict_temp = {}
                for client_id in range(self.args.min_num_clients):
                    start_idx = 0 + client_id * num_weights
                    end_idx = num_weights + client_id * num_weights
                    all_state_dict_temp['client_{}'.format(client_id)] = {
                        k: torch.tensor(v) for k, v in zip(self.model.state_dict().keys(), weights[start_idx:end_idx])
                    }
                state_dict_temp = all_state_dict_temp['client_{}'.format(self.args.cid)]
                for key in self.local_keys:
                    state_dict_temp.pop(key)
                state_dict =  OrderedDict(state_dict_temp)
                self.model.load_state_dict(state_dict, strict=False)

        # FedALA, FedALALC, FedUniV2, FedUniV2.1
        elif self.args.strategy in ['FedALA', 'FedALALC', 'FedUniV2', 'FedUniV2.1']:
            eta = 1.0
            num_pre_loss = 10
            threshold = 0.1
            server_model = copy.deepcopy(self.model)
            # FedALA, FedALALC
            if self.args.strategy not in ['FedUniV2', 'FedUniV2.1']:
                server_state_dict = OrderedDict({
                    k: torch.tensor(v) for k, v in zip(self.model.state_dict().keys(), weights)
                })
            # FedUniV2, FedUniV2.1
            else:
                if config['stage'] == 'fit':
                    if config['iter_global'] <= self.args.iters:
                        server_state_dict = OrderedDict({
                            k: torch.tensor(v) for k, v in zip(self.model.state_dict().keys(), weights)
                        })
                    else:
                        weight_dict = {}
                        all_state_dict_temp = {}
                        num_weights = len(self.model.state_dict().items())
                        print(num_weights)
                        for client_id in self.client_id_list:
                            start_idx = 0 + client_id * num_weights
                            end_idx = num_weights + client_id * num_weights
                            weight_dict['client_{}'.format(client_id)] = weights[start_idx:end_idx]
                            all_state_dict_temp['client_{}'.format(client_id)] = {
                                k: torch.tensor(v) for k, v in zip(self.model.state_dict().keys(), weights[start_idx:end_idx])
                            }
                        start_idx = 0 + self.args.min_num_clients * num_weights
                        end_idx = num_weights + self.args.min_num_clients * num_weights
                        weight_dict['server'] = weights[start_idx:end_idx]
                        all_state_dict_temp['server'] = {
                            k: torch.tensor(v) for k, v in zip(self.model.state_dict().keys(), weights[start_idx:end_idx])
                        }
                        #performance_idx = num_weights + self.args.min_num_clients * num_weights
                        #performance_list = weights[performance_idx]
                        #print(self.args.cid, performance_list)
                        # prompt_idx = num_weights + self.args.min_num_clients * num_weights
                        # prompts_array = weights[prompt_idx]
                        # print("prompts_array.shape=")
                        # print(self.args.cid, prompts_array.shape)


                        
# #####distribution + uni

                        prompt_idx_distribution = self.args.min_num_clients * num_weights
                        prompt_idx_uni = prompt_idx_distribution + 1
                        prompts_array_distribution = np.expand_dims(weights[prompt_idx_distribution], axis=1)
                        prompts_array_uni = np.expand_dims(weights[prompt_idx_uni], axis=0)
                        prompts_array_uni = np.tile(prompts_array_uni, (self.args.min_num_clients, 1, 1, 1))
                        # print("prompts_distribution_array.shape=")
                        # print(self.args.cid, prompts_array_distribution.shape)
                        # print("prompts_uni_array.shape=")
                        # print(self.args.cid, prompts_array_uni.shape)
                        prompts_array = np.concatenate((prompts_array_distribution, prompts_array_uni), axis=1)

                        
                        
                        # print("prompts_array.shape=")
                        # print(self.args.cid, prompts_array.shape)

# #####

# ##### just dist
                        # prompt_idx_distribution = self.args.min_num_clients * num_weights
                        # # prompt_idx_uni = prompt_idx_distribution + 1
                        # prompts_array_distribution = np.expand_dims(weights[prompt_idx_distribution], axis=1)
                        # prompts_array = prompts_array_distribution
# #####


                        server_state_dict_temp = copy.deepcopy(all_state_dict_temp['server'])
                        if self.args.strategy == 'FedUniV2':
                            for key in self.local_keys:
                                server_state_dict_temp.pop(key)

                        # print('fit', len(self.decoder_auxiliary_keys), len(weights))
                        assert self.args.dual_init in ['random', 'adjacent', 'nearest', 'aggregated']
                        if self.args.dual_init in ['random', 'adjacent', 'nearest']:
                            if self.args.dual_init == 'random':
                                selected_id = np.random.randint(0, self.args.min_num_clients)
                            elif self.args.dual_init == 'adjacent':
                                selected_id = self.client_id_list[self.args.cid - 1]
                            else:
                                a = prompts_array.reshape(prompts_array.shape[0], -1)
                                b = prompts_array[self.args.cid].reshape(1, -1)
                                dot = np.dot(a, b.T)
                                norm_array1 = np.linalg.norm(a, axis=1)
                                norm_array2 = np.linalg.norm(b)
                                c = dot / (np.outer(norm_array1, norm_array2))
                                c = c.flatten()
                                c[c<0]=0
                                log(INFO, 'Iteration {}: Client{} Affinity Matrixes {}'.format(config['iter_global'], self.args.cid, c))
                                print(c, sum(c))
                                # print(c, np.argsort(c))
                                selected_id = np.argsort(c)[-2]
        
                            print(self.args.cid, 'selected_id: ', selected_id)
                            for key in self.decoder_auxiliary_keys:
                                server_state_dict_temp[key] = all_state_dict_temp['client_{}'.format(selected_id)][key]
                        else:
                            a = prompts_array.reshape(prompts_array.shape[0], -1)
                            b = prompts_array[self.args.cid].reshape(1, -1)
                            dot = np.dot(a, b.T)
                            norm_array1 = np.linalg.norm(a, axis=1)
                            norm_array2 = np.linalg.norm(b)
                            c = dot / (np.outer(norm_array1, norm_array2))
                            c = c.flatten()
                            c[c<0]=0
                            log(INFO, 'Iteration {}: Client{} Affinity Matrixes {}'.format(config['iter_global'], self.args.cid, c))
                            print(c, sum(c))
                            for key in self.decoder_auxiliary_keys:
                                if 'num_batches_tracked' in key:
                                    continue
                                server_state_dict_temp[key] = torch.zeros_like(server_state_dict_temp[key])
                                for client_id in self.client_id_list:
                                    server_state_dict_temp[key] += (all_state_dict_temp['client_{}'.format(client_id)][key] * c[client_id])
                                server_state_dict_temp[key] = server_state_dict_temp[key] / sum(c)

                        server_state_dict = OrderedDict(server_state_dict_temp)

                else:
                    server_state_dict_temp = {
                        k: torch.tensor(v) for k, v in zip(self.model.state_dict().keys(), weights)
                    }
                    for key in self.local_keys:
                        server_state_dict_temp.pop(key)
                    for key in self.decoder_auxiliary_keys:
                        server_state_dict_temp.pop(key)
                    server_state_dict = OrderedDict(server_state_dict_temp)

            self.model.load_state_dict(server_state_dict, strict=False)
            temp_model = copy.deepcopy(self.model)

            # p_keywords = ['out_conv', 'up4', 'up3', 'up2','up1','down4','down3']
            p_keywords = ['out_conv', 'up4', 'up3', 'up2','up1','mlp']
            # p_keywords = ['out_conv', 'up4', 'up3']
            # p_keywords = ['out_conv']
            p_keys = []
            for name, _ in self.model.named_parameters():
                for p_keyword in p_keywords:
                    if p_keyword in name:
                        p_keys.append(name)
            # print(p_keys)

            params = list(self.model.parameters())
            server_params = list(server_model.parameters())
            temp_params = list(temp_model.parameters())

            if torch.sum(server_params[0] - params[0]) == 0:
                print('skip', config)
                return

            if self.args.strategy in ['FedALA', 'FedALALC', 'FedUniV2', 'FedUniV2.1'] and config['iter_global'] <= 50:
                print('skip', config)
                return

            # only consider higher layers
            def get_params_p(model):
                params_p = []
                for name, param in model.named_parameters():
                    if name in p_keys:
                        params_p.append(param)
                        # print(name)
                return params_p

            params_p = get_params_p(self.model)
            server_params_p = get_params_p(server_model)
            temp_params_p = get_params_p(temp_model)

            # frozen the lower layers to reduce computational cost in Pytorch
            for name, param in temp_model.named_parameters():
                if name in p_keys:
                    param.requires_grad = True
                else:
                    param.requires_grad = False

            # initialize the weight to all ones in the beginning
            if not hasattr(self, 'weights'):
                self.fedala_weights = [torch.ones_like(param.data).cuda() for param in params_p]

            # initialize the higher layers in the temp local model
            for temp_param, param, server_param, fedala_weight in zip(temp_params_p, params_p, server_params_p,
                                                self.fedala_weights):
                temp_param.data = param + (server_param - param) * fedala_weight

            # used to obtain the gradient of higher layers
            # no need to use optimizer.step(), so lr=0
            # optimizer = torch.optim.SGD(temp_params_p, lr=0)
            optimizer = torch.optim.AdamW(temp_params_p,lr=0,betas=(0.9, 0.999),eps=1e-8,weight_decay=1e-2, amsgrad=False)
            ce_loss = nn.CrossEntropyLoss(ignore_index=self.args.num_classes)

            # weight learning
            losses = []
            count = 0
            while True:
                for i_batch, sampled_batch in enumerate(self.trainloader):

                    if self.args.img_class == 'faz' or self.args.img_class == 'prostate':
                        volume_batch, label_batch = sampled_batch['image'].unsqueeze(1), sampled_batch['label']
                        volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()
                    elif self.args.img_class == 'odoc' or self.args.img_class == 'polyp':
                        volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
                        volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()

                    with autocast(enabled=self.amp):
                        outputs = temp_model(volume_batch)[0]
                        loss = ce_loss(outputs, label_batch[:].long())

                    optimizer.zero_grad()
                    if self.amp:
                        self.scaler.scale(loss).backward()
                        self.scaler.step(optimizer)
                        self.scaler.update()
                    else:
                        loss.backward()
                        optimizer.step()

                    # update weight in this batch
                    for temp_param, param, server_param, fedala_weight in zip(temp_params_p, params_p, server_params_p,
                                                self.fedala_weights):
                        # print(type(temp_param), type(server_param), type(param))
                        # print(param)
                        if temp_param.grad == None: # ignore calculation when no gradient given
                            continue
                        fedala_weight.data = torch.clamp(
                            fedala_weight - eta * (temp_param.grad * (server_param - param)), 0, 1)

                    # update temp local model in this batch
                    for temp_param, param, server_param, fedala_weight in zip(temp_params_p, params_p, server_params_p,
                                                self.fedala_weights):
                        temp_param.data = param + (server_param - param) * fedala_weight

                losses.append(loss.item())
                count += 1

                print('Client:', self.args.cid, '\tStd:', np.std(losses[-num_pre_loss:]),
                    '\tALA epochs:', count, self.start_phase)

                # only train one epoch in the subsequent iterations
                if not self.start_phase:
                    break

                # train the weight until convergence
                if len(losses) > num_pre_loss and np.std(losses[-num_pre_loss:]) < threshold:
                    print('Client:', self.args.cid, '\tStd:', np.std(losses[-num_pre_loss:]),
                        '\tALA epochs:', count)
                    break

            self.start_phase = False

            # obtain initialized local model
            for param, temp_param in zip(params_p, temp_params_p):
                param.data = temp_param.data.clone()

        # Other federagted algorithms
        else:
            state_dict_temp =  {
                k: torch.tensor(v) for k, v in zip(self.model.state_dict().keys(), weights)
            }

            if self.args.strategy in ['FedBN', 'FedRep', 'FedUni']:
                for key in self.local_keys:
                    state_dict_temp.pop(key)
                # print(self.local_keys)

            state_dict = OrderedDict(state_dict_temp)
            self.model.load_state_dict(state_dict, strict=False)


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

        return weight * tree_loss, AS


class MScaleAddTreeEnergyLoss(nn.Module):
    def __init__(self):
        super(MScaleAddTreeEnergyLoss, self).__init__()
        # self.configer = configer
        # if self.configer is None:
        #     print("self.configer is None")

        # self.weight = self.configer.get('tree_loss', 'params')['weight']
        # self.weight = weight
        self.mst_layers = MinimumSpanningTree(TreeFilter2D.norm2_distance)
        self.tree_filter_layers = TreeFilter2D(groups=1, sigma=0.02) ##pls see the paper for the sigma!!!!!

    def forward(self, preds, low_feats, high_feats_1, high_feats_2, high_feats_3, unlabeled_ROIs, weight):
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
        if high_feats_1 is not None:
            high_feats_1 = F.interpolate(high_feats_1, size=(h, w), mode='bilinear', align_corners=False)
            # print('new high_feats.size()', high_feats.size())
            tree = self.mst_layers(high_feats_1)
            # print('tree.size()', tree.size())
            AS_1 = self.tree_filter_layers(feature_in=AS, embed_in=high_feats_1, tree=tree, low_tree=False)  # [b, n, h, w]
            
        if high_feats_2 is not None:
            high_feats_2 = F.interpolate(high_feats_2, size=(h, w), mode='bilinear', align_corners=False)
            # print('new high_feats.size()', high_feats.size())
            tree = self.mst_layers(high_feats_2)
            # print('tree.size()', tree.size())
            AS_2 = self.tree_filter_layers(feature_in=AS, embed_in=high_feats_2, tree=tree, low_tree=False)  # [b, n, h, w]
            
        if high_feats_3 is not None:
            high_feats_3 = F.interpolate(high_feats_3, size=(h, w), mode='bilinear', align_corners=False)
            # print('new high_feats.size()', high_feats.size())
            tree = self.mst_layers(high_feats_3)
            # print('tree.size()', tree.size())
            AS_3 = self.tree_filter_layers(feature_in=AS, embed_in=high_feats_3, tree=tree, low_tree=False)  # [b, n, h, w]

        tree_loss_1 = (unlabeled_ROIs * torch.abs(prob - AS_1)).sum()
        tree_loss_2 = (unlabeled_ROIs * torch.abs(prob - AS_2)).sum()
        tree_loss_3 = (unlabeled_ROIs * torch.abs(prob - AS_3)).sum()
        tree_loss = tree_loss_1 + tree_loss_2 + tree_loss_3
        
        if N > 0:
            tree_loss /= N

        return weight * tree_loss, AS_1, AS_2, AS_3


class MScaleRecurveTreeEnergyLoss(nn.Module):
    def __init__(self):
        super(MScaleRecurveTreeEnergyLoss, self).__init__()
        # self.configer = configer
        # if self.configer is None:
        #     print("self.configer is None")

        # self.weight = self.configer.get('tree_loss', 'params')['weight']
        # self.weight = weight
        self.mst_layers = MinimumSpanningTree(TreeFilter2D.norm2_distance)
        self.tree_filter_layers = TreeFilter2D(groups=1, sigma=0.02) ##pls see the paper for the sigma!!!!!

    def forward(self, preds, low_feats, high_feats_1, high_feats_2, high_feats_3, unlabeled_ROIs, weight):
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
        if high_feats_1 is not None:
            high_feats_1 = F.interpolate(high_feats_1, size=(h, w), mode='bilinear', align_corners=False)
            # print('new high_feats.size()', high_feats.size())
            tree = self.mst_layers(high_feats_1)
            # print('tree.size()', tree.size())
            AS_1 = self.tree_filter_layers(feature_in=AS, embed_in=high_feats_1, tree=tree, low_tree=False)  # [b, n, h, w]
            
        if high_feats_2 is not None:
            high_feats_2 = F.interpolate(high_feats_2, size=(h, w), mode='bilinear', align_corners=False)
            # print('new high_feats.size()', high_feats.size())
            tree = self.mst_layers(high_feats_2)
            # print('tree.size()', tree.size())
            AS_2 = self.tree_filter_layers(feature_in=AS_1, embed_in=high_feats_2, tree=tree, low_tree=False)  # [b, n, h, w]
            
        if high_feats_3 is not None:
            high_feats_3 = F.interpolate(high_feats_3, size=(h, w), mode='bilinear', align_corners=False)
            # print('new high_feats.size()', high_feats.size())
            tree = self.mst_layers(high_feats_3)
            # print('tree.size()', tree.size())
            AS_3 = self.tree_filter_layers(feature_in=AS_2, embed_in=high_feats_3, tree=tree, low_tree=False)  # [b, n, h, w]

        # tree_loss_1 = (unlabeled_ROIs * torch.abs(prob - AS_1)).sum()
        # tree_loss_2 = (unlabeled_ROIs * torch.abs(prob - AS_2)).sum()
        # tree_loss_3 = (unlabeled_ROIs * torch.abs(prob - AS_3)).sum()
        # tree_loss = tree_loss_1 + tree_loss_2 + tree_loss_3
        tree_loss = (unlabeled_ROIs * torch.abs(prob - AS_3)).sum()
        
        if N > 0:
            tree_loss /= N

        return weight * tree_loss, AS_1, AS_2, AS_3