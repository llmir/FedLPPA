

'''for i in range(20, 30020, 20):
    print(i)


a = [(i, j) for i in [1, 2] for j in [3, 4]]
print(a)'''


from flower_common import get_bn_stats, MyModel
from dataloaders.dataset import BaseDataSets, RandomGenerator
from networks.net_factory import net_factory
from torch.utils.data import DataLoader
from torchvision import transforms
import random, os
import torch
import torch.nn as nn
import torch.optim as optim



class Args(object):

    def __init__(self):
        self.img_class = 'faz'
        self.model = 'unet_uni'
        self.in_chns = 1
        self.num_classes = 2
        self.root_path = '../data/FAZ_h5'
        self.root_path = '../data/FAZ_h5'
        self.patch_size = [256, 256]
        self.client = 'client1'
        self.sup_type = 'scribble_noisy'
        self.gpu = 2
        self.strategy = 'FedAP'
        self.batch_size = 12
        self.seed = 2022
        self.min_num_clients = 5
        self.cid = 2

args = Args()
os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
model = net_factory(args, net_type=args.model, in_chns=args.in_chns, class_num=args.num_classes)
model.cuda()
print(model)

db_train = BaseDataSets(base_dir=args.root_path, split='train', transform=transforms.Compose([
        RandomGenerator(args.patch_size, img_class=args.img_class)
]), client=args.client, sup_type=args.sup_type)

def worker_init_fn(worker_id):
    random.seed(args.seed + worker_id)

trainloader = DataLoader(db_train, batch_size=args.batch_size, shuffle=True,
                            num_workers=4, pin_memory=True, worker_init_fn=worker_init_fn)

'''bnm, bnv = get_bn_stats(args, model, trainloader)
print(len(bnm), [bnm[i].shape for i, _ in enumerate(bnm)])
print(len(bnv), [bnv[i].shape for i, _ in enumerate(bnv)])'''

ce_loss = nn.CrossEntropyLoss(ignore_index=args.num_classes)
optimizer = optim.AdamW(model.parameters(),lr=0.01,betas=(0.9, 0.999),eps=1e-8,weight_decay=1e-2, amsgrad=False)

model.train()
for i_batch, sampled_batch in enumerate(trainloader):
    if args.img_class == 'faz':
        volume_batch, label_batch = sampled_batch['image'].unsqueeze(1), sampled_batch['label']
        volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()
    elif args.img_class == 'odoc' or args.img_class == 'polyp':
        volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
        volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()

    print('batch indx: {}'.format(i_batch))
    out = model(volume_batch)
    outputs, feature, de1, de2, de3, de4 = out
    loss_ce = ce_loss(outputs, label_batch[:].long())
    loss = loss_ce
    print(loss.item())

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print()
