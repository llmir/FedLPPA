import itertools
import os
import random
import re
from collections import defaultdict
from glob import glob
import pandas as pd
import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy import ndimage
from scipy.ndimage.interpolation import zoom
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import Sampler
from torchvision import transforms


class BaseDataSets_s2l(Dataset):
    def __init__(self, base_dir=None, split='train', transform=None, client="client1", num=None,sup_type="label",img_class='odoc'):
        self._base_dir = base_dir
        self.sample_list = []
        self.split = split
        self.img_class=img_class
        self.sup_type = sup_type
        self.transform = transform
        train_ids, test_ids = self._get_client_ids(client)
        # self.all_slices = os.listdir(self._base_dir + "/ACDC_training_slices")
        if self.split == 'train':
            self.sample_list = train_ids
            print("total {} samples".format(len(self.sample_list)))

            self.images = defaultdict(dict)
            for idx, case in enumerate(self.sample_list):
                h5f = h5py.File(self._base_dir +"/{}".format(case), 'r')
                img = h5f['image']
                mask = h5f['mask']
                sup_label = h5f[self.sup_type]
                self.images[idx]['id'] = case
                self.images[idx]['image'] = np.array(img)
                self.images[idx]['mask'] = np.array(mask)
                self.images[idx][self.sup_type] = np.array(sup_label)
                h, w = mask.shape
                # self.images[idx]['weight'] = np.zeros((h, w, 4), dtype=np.float32)
                if img_class=='odoc':
                    self.images[idx]['weight'] = np.zeros((h, w, 3), dtype=np.float32)
                if img_class=='faz':
                    self.images[idx]['weight'] = np.zeros((h, w, 2), dtype=np.float32)
        elif self.split == 'val':
            self.sample_list=test_ids      
        print("total {} samples".format(len(self.sample_list)))
            

    def _get_client_ids(self, client):
        client1_test_set = 'Domain1/test/'+pd.Series(os.listdir( self._base_dir+"/Domain1/test"))
        client1_training_set = 'Domain1/train/'+pd.Series(os.listdir( self._base_dir+"/Domain1/train"))
        client2_test_set = 'Domain2/test/'+pd.Series(os.listdir( self._base_dir+"/Domain2/test"))
        client2_training_set = 'Domain2/train/'+pd.Series(os.listdir( self._base_dir+"/Domain2/train"))
        client3_test_set = 'Domain3/test/'+pd.Series(os.listdir( self._base_dir+"/Domain3/test"))
        client3_training_set = 'Domain3/train/'+pd.Series(os.listdir( self._base_dir+"/Domain3/train"))
        client4_test_set = 'Domain4/test/'+pd.Series(os.listdir( self._base_dir+"/Domain4/test"))
        client4_training_set = 'Domain4/train/'+pd.Series(os.listdir( self._base_dir+"/Domain4/train"))
        client5_test_set = 'Domain5/test/'+pd.Series(os.listdir( self._base_dir+"/Domain5/test"))
        client5_training_set = 'Domain5/train/'+pd.Series(os.listdir( self._base_dir+"/Domain5/train"))
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
        
        if client == "client1":
            return [client1_training_set, client1_test_set]
        elif client == "client2":
            return [client2_training_set, client2_test_set]
        elif client == "client3":
            return [client3_training_set, client3_test_set]
        elif client == "client4":
            return [client4_training_set, client4_test_set]
        elif client == "client5":
            return [client5_training_set, client5_test_set]
       
        else:
            return "ERROR KEY"
    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        if self.split == 'train':
            case = self.images[idx]['id'][:]
            image = self.images[idx]['image'][:]
            # print('\nimg=',np.array(image).shape)
            mask = self.images[idx]['mask'][:]
            # print('\nmask=',np.array(mask.shape))
            sup_label = self.images[idx][self.sup_type][:]
            # print('scribble=',np.array(scribble).shape)
            weight = self.images[idx]['weight'][:]
            # print('weight=',np.array(weight).shape)
            sample = {'image': image, 'mask': mask,
                    'sup_label': sup_label, 'weight': weight}
            sample = self.transform(sample)
            sample['id'] = case
        if self.split == 'val':
            case = self.sample_list[idx]
            h5f = h5py.File(self._base_dir +
                            "/{}".format(case), 'r')
            image = h5f['image'][:]
            label = h5f['mask'][:]
            sample = {'image': image, 'label': label}
            sample["id"] = idx

        return sample


# def random_rot_flip(image, label, scribble, weight):
#     # k = np.random.randint(0, 4)
#     # image = np.rot90(image, k)
#     # label = np.rot90(label, k)
#     # scribble = np.rot90(scribble, k)
#     # weight = np.rot90(weight, k)
#     axis = np.random.randint(0, 2)
#     image = np.flip(image, axis=axis).copy()
#     label = np.flip(label, axis=axis).copy()
#     scribble = np.flip(scribble, axis=axis).copy()
#     weight = np.flip(weight, axis=axis).copy()
#     return image, label, scribble, weight


def random_rot_flip(image, label, sup_label, weight, img_class='odoc'):
    if img_class == 'odoc':
        k = np.random.randint(0, 4)
        image = np.rot90(image, k, (1,2))
        label = np.rot90(label, k, (0,1))
        weight = np.rot90(weight, k,(0,1))
        sup_label = np.rot90(sup_label, k,(0,1))
        axis = np.random.randint(1, 3)
        image = np.flip(image, axis=axis).copy()
        label = np.flip(label, axis=axis-1).copy()
        weight = np.flip(weight, axis=axis-1).copy()
        sup_label = np.flip(sup_label, axis=axis-1).copy()
        
        return image, label, sup_label, weight
    if img_class == 'faz':
        k = np.random.randint(0, 4)
        image = np.rot90(image, k)
        label = np.rot90(label, k)
        sup_label = np.rot90(sup_label, k)
        weight = np.rot90(weight, k)
        axis = np.random.randint(0, 2)
        image = np.flip(image, axis=axis).copy()
        label = np.flip(label, axis=axis).copy()
        sup_label = np.flip(sup_label, axis=axis).copy()
        weight = np.flip(weight, axis=axis).copy()
        return image, label, sup_label, weight


def random_rotate(image, label, sup_label, weight, img_class='odoc'):
    if img_class=='faz':
        angle = np.random.randint(-45, 45)
        image = ndimage.rotate(image, angle, order=0, reshape=False, cval=0.8)
        label = ndimage.rotate(label, angle, order=0,
                            reshape=False, mode="constant", cval=2)
        sup_label = ndimage.rotate(sup_label, angle, order=0, reshape=False,cval=2)
        weight = ndimage.rotate(weight, angle, order=0, reshape=False)
        return image, label, sup_label, weight
    if img_class=='odoc':
        angle = np.random.randint(-45, 45)
        image = ndimage.rotate(image, angle, axes=(1,2), order=0, reshape=False)
        label = ndimage.rotate(label, angle, axes=(0,1), order=0,reshape=False, mode="constant", cval=3)
        sup_label = ndimage.rotate(sup_label, angle,axes=(0,1), order=0, reshape=False,cval=3)
        weight = ndimage.rotate(weight, angle, axes=(0,1),order=0, reshape=False)
        return image, label, sup_label, weight

class RandomGenerator_s2l(object):
    def __init__(self, output_size,img_class):
        self.output_size = output_size
        self.img_class = img_class

    def __call__(self, sample):
        image, label, sup_label, weight = sample['image'], sample['mask'], sample['sup_label'], sample['weight']
        if random.random() > 0.5:
            image, label, sup_label, weight = random_rot_flip(
                image, label, sup_label, weight,img_class=self.img_class)
        elif random.random() > 0.5:
            image, label, sup_label, weight = random_rotate(
                image, label, sup_label, weight,img_class=self.img_class)
        # x, y ,z= image.shape
        # image = zoom(
        #     image, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        # label = zoom(
        #     label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        # sup_label = zoom(
        #     sup_label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        # weight = zoom(
        #     weight, (self.output_size[0] / x, self.output_size[1] / y, 1), order=0)
        image = torch.from_numpy(image.astype(np.float32))
        label = torch.from_numpy(label.astype(np.uint8))
        sup_label = torch.from_numpy(sup_label.astype(np.uint8))
        weight = torch.from_numpy(weight.astype(np.float32))
        sample = {'image': image, 'mask': label,
                  'sup_label': sup_label, 'weight': weight}
        return sample


class TwoStreamBatchSampler(Sampler):
    """Iterate two sets of indices

    An 'epoch' is one iteration through the primary indices.
    During the epoch, the secondary indices are iterated through
    as many times as needed.
    """

    def __init__(self, primary_indices, secondary_indices, batch_size, secondary_batch_size):
        self.primary_indices = primary_indices
        self.secondary_indices = secondary_indices
        self.secondary_batch_size = secondary_batch_size
        self.primary_batch_size = batch_size - secondary_batch_size

        assert len(self.primary_indices) >= self.primary_batch_size > 0
        assert len(self.secondary_indices) >= self.secondary_batch_size > 0

    def __iter__(self):
        primary_iter = iterate_once(self.primary_indices)
        secondary_iter = iterate_eternally(self.secondary_indices)
        return (
            primary_batch + secondary_batch
            for (primary_batch, secondary_batch)
            in zip(grouper(primary_iter, self.primary_batch_size),
                   grouper(secondary_iter, self.secondary_batch_size))
        )

    def __len__(self):
        return len(self.primary_indices) // self.primary_batch_size


def iterate_once(iterable):
    return np.random.permutation(iterable)


def iterate_eternally(indices):
    def infinite_shuffles():
        while True:
            yield np.random.permutation(indices)
    return itertools.chain.from_iterable(infinite_shuffles())


def grouper(iterable, n):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3) --> ABC DEF"
    args = [iter(iterable)] * n
    return zip(*args)


if __name__ == '__main__':
    data_root = '../data/ACDC/'
    labeled_slice = 146

    db_train = BaseDataSets_s2l(base_dir=data_root, split="train", num=None,
                            transform=transforms.Compose([RandomGenerator_s2l([256, 256])]))
    db_val = BaseDataSets_s2l(base_dir=data_root, split="val")
    total_slices = len(db_train)
    labeled_slice = 146
    print("Total silices is: {}, labeled slices is: {}".format(
        total_slices, labeled_slice))
    labeled_idxs = list(range(0, labeled_slice))
    unlabeled_idxs = list(range(labeled_slice, total_slices))
    batch_sampler = TwoStreamBatchSampler(
        labeled_idxs, unlabeled_idxs, 24, 24 - 12)

    trainloader = DataLoader(
        db_train, batch_sampler=batch_sampler, num_workers=8, pin_memory=True)
    valloader = DataLoader(db_val, batch_size=1, shuffle=False, num_workers=1)

    total_slices = len(db_train)
    print("Total silices is: {}, labeled slices is: {}".format(
        total_slices, labeled_slice))
    print("{} iterations per epoch".format(len(trainloader)))

    for i_batch, sampled_batch in enumerate(trainloader):
        volume_batch, mask_batch, label_batch, pseudo_batch = sampled_batch[
            'image'], sampled_batch['mask'], sampled_batch['scribble'], sampled_batch['pseudo']
        case = sampled_batch['id'][:12]
        print(volume_batch.shape, mask_batch.shape,
              label_batch.shape, pseudo_batch.shape)
        print(case)
        print(torch.unique(mask_batch))