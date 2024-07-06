import argparse
import logging
import os
import random
import shutil
import sys

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
from PIL import Image
from scipy.ndimage import zoom
from tensorboardX import SummaryWriter
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from dataloaders.dataset import BaseDataSets, TwoStreamBatchSampler
from dataloaders.dataset_s2l import BaseDataSets_s2l, RandomGenerator_s2l
from networks.net_factory import net_factory
from utils import losses, metrics, ramps
from val_2D import test_single_volume

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='../data/odoc', help='Name of Experiment')
parser.add_argument('--exp', type=str,
                    default='odoc_bs_4/pCE_SPS', help='experiment_name')
parser.add_argument('--client', type=str,
                    default='client1', help='cross validation')
parser.add_argument('--sup_type', type=str,
                    default='fsl', help='supervision type')
parser.add_argument('--model', type=str,
                    default='unet', help='model_name')
parser.add_argument('--num_classes', type=int,  default=3,
                    help='output channel of network')
parser.add_argument('--max_iterations', type=int,
                    default=60000, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=4,
                    help='batch_size per gpu')
parser.add_argument('--deterministic', type=int,  default=1,
                    help='whether use deterministic training')
parser.add_argument('--base_lr', type=float,  default=0.01,
                    help='segmentation network learning rate')
parser.add_argument('--patch_size', type=list,  default=[3, 800, 800],
                    help='patch size of network input')
parser.add_argument('--seed', type=int,  default=1337, help='random seed')
parser.add_argument('--gpus',nargs='+', type=int, default=[0,1,2],
                    help='gpu index,must set CUDA_VISIBLE_DEVICES at terminal')
parser.add_argument('--labeled_bs', type=int, default=6,
                    help='labeled_batch_size per gpu')
parser.add_argument('--labeled_num', type=int, default=3, help='labeled data')

# costs
parser.add_argument('--ema_decay', type=float,  default=0.99, help='ema_decay')
parser.add_argument('--consistency', type=float,
                    default=0.1, help='consistency')
parser.add_argument('--consistency_rampup', type=float,
                    default=200.0, help='consistency_rampup')


parser.add_argument('--period_iter', type=int, default=100)
parser.add_argument('--thr_iter', type=int, default=6000)
parser.add_argument('--thr_conf', type=float, default=0.8)
parser.add_argument('--alpha', type=float, default=0.2)

args = parser.parse_args()


def patients_to_slices(dataset, patients_num):
    ref_dict = None
    if "ACDC" in dataset:
        ref_dict = {"4": 68, "8": 146, "16": 310,
                    "24": 450, "32": 588, "40": 724, "80": 1512}
    else:
        print("Error")
    return ref_dict[str(patients_num)]


def train(args, snapshot_path):
    base_lr = args.base_lr
    num_classes = args.num_classes
    max_iterations = args.max_iterations

    model = net_factory(net_type=args.model, in_chns=3, class_num=num_classes)
    db_train = BaseDataSets_s2l(base_dir=args.root_path, client=args.client,
                                transform=transforms.Compose([RandomGenerator_s2l(args.patch_size)]))
    db_val = BaseDataSets(base_dir=args.root_path,
                          client=args.client, split="val")

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    total_slices = len(db_train)
    print("Total silices is: {}, labeled slices is: {}".format(
        total_slices, total_slices))

    trainloader = DataLoader(db_train, batch_size=args.batch_size, shuffle=True,
                             num_workers=8, pin_memory=True, worker_init_fn=worker_init_fn)
    valloader = DataLoader(db_val, batch_size=1, shuffle=False, num_workers=1)

    model.train()

    optimizer = optim.SGD(model.parameters(), lr=base_lr,
                          momentum=0.9, weight_decay=0.0001)
    ce_loss = CrossEntropyLoss(ignore_index=3)
    u_ce_loss = CrossEntropyLoss(ignore_index=3)

    writer = SummaryWriter(snapshot_path + '/log')
    logging.info("{} iterations per epoch".format(len(trainloader)))

    iter_num = 0
    max_epoch = max_iterations // len(trainloader) + 1
    best_performance = 0.0
    iterator = tqdm(range(max_epoch), ncols=70)
    for epoch_num in iterator:
        for i_batch, sampled_batch in enumerate(trainloader):

            volume_batch, label_batch, weight_batch = sampled_batch[
                'image'], sampled_batch['scribble'], sampled_batch['weight']
            volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()

            outputs = model(volume_batch)
            loss_ce = ce_loss(outputs, label_batch.long())
            if iter_num < args.thr_iter:
                loss = loss_ce
            else:
                scribbles = label_batch.long().cpu()
                mean_0, mean_1, mean_2 = weight_batch[..., 0], weight_batch[..., 1], weight_batch[..., 2]
                    # weight_batch[..., 3]
                # print(torch.zeros_like(mean_0).long().dtype, (4*torch.ones_like(scribbles.long())).dtype)

                u_labels_0 = torch.where((mean_0 > args.thr_conf) & (scribbles == 3),
                                         torch.zeros_like(mean_0), 3. * torch.ones_like(scribbles)).cuda()
                u_labels_1 = torch.where((mean_1 > args.thr_conf) & (scribbles == 3),
                                         torch.zeros_like(mean_1) + 1, 3. * torch.ones_like(scribbles)).cuda()
                u_labels_2 = torch.where((mean_2 > args.thr_conf) & (scribbles == 3),
                                         torch.zeros_like(mean_2) + 2, 3. * torch.ones_like(scribbles)).cuda()
                # u_labels_3 = torch.where((mean_3 > args.thr_conf) & (scribbles == 4),
                #                          torch.zeros_like(mean_3) + 3, 4. * torch.ones_like(scribbles)).cuda()
                u_labels = torch.ones_like(u_labels_0).long() * 3
                u_labels[u_labels_0 == 0] = 0
                u_labels[u_labels_1 == 1] = 1
                u_labels[u_labels_2 == 2] = 2
                # u_labels[u_labels_3 == 3] = 3
                loss_u = u_ce_loss(outputs, u_labels)
                loss = loss_ce + 0.5 * loss_u

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
            if iter_num > args.thr_iter:
                writer.add_scalar('info/loss_weight', loss_u, iter_num)

            # if iter_num % 20 == 0:
            #     image = volume_batch[1, 0:1, :, :]
            #     image = (image - image.min()) / (image.max() - image.min())
            #     writer.add_image('train/Image', image, iter_num)
            #     outputs = torch.argmax(torch.softmax(
            #         outputs, dim=1), dim=1, keepdim=True)
            #     writer.add_image('train/Prediction',
            #                      outputs[1, ...] * 50, iter_num)
            #     labs = label_batch[1, ...].unsqueeze(0) * 50
            #     writer.add_image('train/GroundTruth', labs, iter_num)
            if iter_num % 20 == 0:
                image = volume_batch[1, 0:3, :, :]
                
                image = (image - image.min()) / (image.max() - image.min())
                writer.add_image('train/Image', image, iter_num)
                outputs2 = torch.argmax(torch.softmax(
                    outputs, dim=1), dim=1, keepdim=True)
                writer.add_image('train/Prediction',
                                 outputs2[1,0,...] * 50, iter_num,dataformats='HW')

                labs = label_batch[1, ...].unsqueeze(0) * 50

                writer.add_image('train/GroundTruth', labs, iter_num,dataformats='CHW')

            if iter_num > 0 and iter_num % 200 == 0:
                model.eval()
                metric_list = 0.0
                for i_batch, sampled_batch in enumerate(valloader):
                    if args.img_class == 'faz':
                        volume_batch, label_batch = sampled_batch['image'].unsqueeze(1), sampled_batch['label']
                        volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()
                    elif args.img_class == 'odoc':
                        volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
                        volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()
                    
                    outputs_val = model(volume_batch)[0]
                    outputs_soft_val = torch.softmax(outputs_val, dim=1)
                    loss_ce_val = ce_loss(outputs_val, label_batch[:].long())
                    if iter_num < args.thr_iter:
                        loss_val = loss_ce
                    else:
                        scribbles = label_batch.long().cpu()
                        mean_0, mean_1, mean_2 = weight_batch[..., 0], weight_batch[..., 1], weight_batch[..., 2]
                            # weight_batch[..., 3]
                        # print(torch.zeros_like(mean_0).long().dtype, (4*torch.ones_like(scribbles.long())).dtype)

                        u_labels_0 = torch.where((mean_0 > args.thr_conf) & (scribbles == 3),
                                                torch.zeros_like(mean_0), 3. * torch.ones_like(scribbles)).cuda()
                        u_labels_1 = torch.where((mean_1 > args.thr_conf) & (scribbles == 3),
                                                torch.zeros_like(mean_1) + 1, 3. * torch.ones_like(scribbles)).cuda()
                        u_labels_2 = torch.where((mean_2 > args.thr_conf) & (scribbles == 3),
                                                torch.zeros_like(mean_2) + 2, 3. * torch.ones_like(scribbles)).cuda()
                        # u_labels_3 = torch.where((mean_3 > args.thr_conf) & (scribbles == 4),
                        #                          torch.zeros_like(mean_3) + 3, 4. * torch.ones_like(scribbles)).cuda()
                        u_labels = torch.ones_like(u_labels_0).long() * 3
                        u_labels[u_labels_0 == 0] = 0
                        u_labels[u_labels_1 == 1] = 1
                        u_labels[u_labels_2 == 2] = 2
                        # u_labels[u_labels_3 == 3] = 3
                        loss_u = u_ce_loss(outputs, u_labels)
                        loss_val = loss_ce + 0.5 * loss_u
                    metric_i = test_single_volume(
                        sampled_batch["image"], sampled_batch["label"], model, classes=num_classes)
                    metric_list = metric_list+np.array(metric_i)
                    
                metric_list = metric_list / len(db_val)
                for class_i in range(num_classes-1):
                    writer.add_scalar('info/val_{}_dice'.format(class_i+1),
                                      metric_list[class_i, 0], iter_num)
                    writer.add_scalar('info/val_{}_hd95'.format(class_i+1),
                                      metric_list[class_i, 1], iter_num)
                    writer.add_scalar('info/val_{}_recall'.format(class_i+1),
                                      metric_list[class_i, 2], iter_num)
                    writer.add_scalar('info/val_{}_precision'.format(class_i+1),
                                      metric_list[class_i, 3], iter_num)
                    writer.add_scalar('info/val_{}_jc'.format(class_i+1),
                                      metric_list[class_i, 4], iter_num)
                    writer.add_scalar('info/val_{}_specificity'.format(class_i+1),
                                      metric_list[class_i, 5], iter_num)
                    writer.add_scalar('info/val_{}_ravd'.format(class_i+1),
                                      metric_list[class_i, 6], iter_num)
                    writer.add_scalar('info/total_loss_val', loss_val, iter_num)
                    writer.add_scalar('info/loss_ce_val', loss_ce_val, iter_num)
                    writer.add_scalar('info/val_total_loss', loss_val, iter_num)
                    writer.add_scalar('info/val_ce_val', loss_ce_val, iter_num)

                performance = np.mean(metric_list, axis=0)[0]
                mean_hd95 = np.mean(metric_list, axis=0)[1]
                mean_recall = np.mean(metric_list, axis=0)[2]
                mean_precision = np.mean(metric_list, axis=0)[3]
                mean_jc = np.mean(metric_list, axis=0)[4]
                mean_specificity = np.mean(metric_list, axis=0)[5]
                mean_ravd = np.mean(metric_list, axis=0)[6]
                writer.add_scalar('info/val_mean_dice', performance, iter_num)
                writer.add_scalar('info/val_mean_hd95', mean_hd95, iter_num)
                writer.add_scalar('info/val_mean_recall', mean_recall, iter_num)
                writer.add_scalar('info/val_mean_precision', mean_precision, iter_num)
                writer.add_scalar('info/val_mean_jc', mean_jc, iter_num)
                writer.add_scalar('info/val_mean_specificity', mean_specificity, iter_num)
                writer.add_scalar('info/val_mean_ravd', mean_ravd, iter_num)

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
                    'iteration %d : mean_dice : %f mean_hd95 : %f' % (iter_num, performance, mean_hd95))
                model.train()
            # if iter_num > 0 and iter_num % 200 == 0:
            #     model.eval()
            #     metric_list = 0.0
            #     for i_batch, sampled_batch in enumerate(valloader):
            #         metric_i = test_single_volume(
            #             sampled_batch["image"], sampled_batch["label"], model, classes=num_classes)
            #         metric_list += np.array(metric_i)
            #     metric_list = metric_list / len(db_val)
            #     for class_i in range(num_classes-1):
            #         writer.add_scalar(
            #             'info/val_{}_dice'.format(class_i+1), metric_list[class_i, 0], iter_num)
            #         writer.add_scalar(
            #             'info/val_{}_hd95'.format(class_i+1), metric_list[class_i, 1], iter_num)

            #     performance = np.mean(metric_list, axis=0)[0]
            #     mean_hd95 = np.mean(metric_list, axis=0)[1]
            #     writer.add_scalar('info/val_mean_dice', performance, iter_num)
            #     writer.add_scalar('info/val_mean_hd95', mean_hd95, iter_num)

            #     if performance > best_performance:
            #         best_performance = performance
            #         save_mode_path = os.path.join(snapshot_path,
            #                                       'iter_{}_dice_{}.pth'.format(
            #                                           iter_num, round(best_performance, 4)))
            #         save_best = os.path.join(
            #             snapshot_path, '{}_best_model.pth'.format(args.model))
            #         torch.save(model.state_dict(), save_mode_path)
            #         torch.save(model.state_dict(), save_best)

            #     logging.info(
            #         'iteration %d : mean_dice : %f mean_hd95 : %f' % (iter_num, performance, mean_hd95))
            #     model.train()

            if iter_num % 3000 == 0:
                save_mode_path = os.path.join(
                    snapshot_path, 'iter_' + str(iter_num) + '.pth')
                torch.save(model.state_dict(), save_mode_path)
                logging.info("save model to {}".format(save_mode_path))

            if iter_num > 0 and iter_num % args.period_iter == 0:
                logging.info("update weight start")
                ds = trainloader.dataset
                if not os.path.exists(os.path.join(snapshot_path, 'ensemble', str(iter_num))):
                    os.makedirs(os.path.join(snapshot_path,
                                'ensemble', str(iter_num)))
                # for idx, images in tqdm(ds.images.items(), total=len(ds)):
                for idx, images in ds.images.items():
                    img = images['image']
                    # img = zoom(
                    #     img, (256 / img.shape[0], 256 / img.shape[1]), order=0)
                    img = torch.from_numpy(img).unsqueeze(
                        0).cuda()
                    with torch.no_grad():
                        pred = torch.nn.functional.softmax(model(img), dim=1)
                    pred = pred.squeeze(0).cpu().numpy()
                    # pred = zoom(
                    #     pred, (1, images['image'].shape[0] / 256, images['image'].shape[1] / 256), order=0)
                    pred = torch.from_numpy(pred)
                    weight = torch.from_numpy(images['weight'])
                    x0, x1, x2 = pred[0], pred[1], pred[2]
                    weight[..., 0] = args.alpha * x0 + \
                        (1 - args.alpha) * weight[..., 0]
                    weight[..., 1] = args.alpha * x1 + \
                        (1 - args.alpha) * weight[..., 1]
                    weight[..., 2] = args.alpha * x2 + \
                        (1 - args.alpha) * weight[..., 2]
                    trainloader.dataset.images[idx]['weight'] = weight.numpy()

                    # img = Image.fromarray(np.array((weight[..., 1] + weight[..., 2] + weight[..., 3]).cpu().numpy() * 255, dtype=np.uint8))
                    # img = img.convert('RGB')
                    # if not os.path.exists(os.path.join(snapshot_path, 'ensemble', str(iter_num), images['id'].split('_')[0])):
                    #     os.mkdir(os.path.join(snapshot_path, 'ensemble', str(iter_num), images['id'].split('_')[0]))
                    # img.save(os.path.join(snapshot_path, 'ensemble', str(iter_num), images['id'].split('_')[0], images['id'].replace('.h5', '.png')))
                logging.info("update weight end")

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

    snapshot_path = "../model_odoc_icctw/{}_{}/{}".format(
        args.exp, args.client, args.sup_type).replace('client','domain')
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    if os.path.exists(snapshot_path + '/code'):
        shutil.rmtree(snapshot_path + '/code')
    shutil.copytree('.', snapshot_path + '/code',
                    shutil.ignore_patterns(['.git', '__pycache__']))

    logging.basicConfig(filename=snapshot_path+"/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    train(args, snapshot_path)
