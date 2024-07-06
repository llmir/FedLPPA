# save images in h5
import glob
import os
from this import s
import h5py
import numpy as np
import SimpleITK as sitk
import cv2

# saving images in volume level


class MedicalImageDeal(object):
    def __init__(self, img, percent=1):
        self.img = img
        self.percent = percent

    @property
    def valid_img(self):
        from skimage import exposure
        cdf = exposure.cumulative_distribution(self.img)
        watershed = cdf[1][cdf[0] >= self.percent][0]
        return np.clip(self.img, self.img.min(), watershed)

    @property
    def norm_img(self):
        return (self.img - self.img.min()) / (self.img.max() - self.img.min())


slice_num = 0
# odoc data processing
for m in [1,2,3,4,5]:
    for type in ['train']:
        for case in sorted(glob.glob("../data/ODOC/Domain{}/{}/mask/*.png".format(m,type))):
            
            if type!= 'train':
                # ###label_processing
                label_itk = sitk.ReadImage(case)
                label = sitk.GetArrayFromImage(label_itk)
                label=label.transpose(2,0,1)
                # disc
                label[label==170]=1
                # cup
                label[label==85]=2
                # ###img_processing
                image_path = case.replace("mask", "imgs")
                image_itk = sitk.ReadImage(image_path)
                image = sitk.GetArrayFromImage(image_itk)
                image=image.transpose(2,0,1)
                image = MedicalImageDeal(image, percent=1).valid_img
                image = (image - image.min()) / (image.max() - image.min())
                print(image.shape)
                image = image.astype(np.float32)
                f = h5py.File(
                    case.replace('/ODOC','/ODOC_h5').replace('mask/','').replace('.png','.h5'), 'w')
                f.create_dataset(
                    'image', data=image, compression="gzip")
                f.create_dataset('mask', data=label[0], compression="gzip")
                f.close()
                
            elif type == 'train':
                # ###label_processing
                label_itk = sitk.ReadImage(case)
                label = sitk.GetArrayFromImage(label_itk)
                label=label.transpose(2,0,1)
                # cup
                label[label==170]=1
                # disc
                label[label==85]=2
                # ###scr_porcessing
                scribble_path = case.replace("mask", "scr")
                scribble_itk = sitk.ReadImage(scribble_path)
                scribbl = sitk.GetArrayFromImage(scribble_itk)
                print(scribbl.shape)
                scribbl=scribbl.transpose(2,0,1)
                scribble=scribbl
                print("scribble=",scribble.shape)
                scribble[scribble==0]=0
                scribble[scribble==85]=2
                scribble[scribble==170]=1
                scribble[scribble==255]=3
                print(np.unique(scribble))
                print("scribble=",scribble.shape)
                # ###scr_noisy_porcessing
                scribble_n_path = case.replace("mask", "scr_n")
                scribble_n_itk = sitk.ReadImage(scribble_n_path)
                scribbl_n = sitk.GetArrayFromImage(scribble_n_itk)
                scribbl_n=scribbl_n.transpose(2,0,1)
                scribble_n=scribbl_n
                scribble_n[scribble_n==0]=0
                scribble_n[scribble_n==85]=2
                scribble_n[scribble_n==170]=1
                scribble_n[scribble_n==255]=3
                print(np.unique(scribble))
                print("scribble_n=",scribble_n.shape)
                # ###block_porcessing
                block_path = case.replace("mask", "block")
                block_itk = sitk.ReadImage(block_path)
                bloc = sitk.GetArrayFromImage(block_itk)
                print("block=",bloc.shape)
                bloc=bloc.transpose(2,0,1)
                block=bloc
                
                block[block==0]=0
                block[block==85]=2
                block[block==170]=1
                block[block==255]=3
                print(np.unique(block))
                
                # ###keypoint
                keypoint_path = case.replace("mask", "keypoint")
                keypoint_itk = sitk.ReadImage(keypoint_path)
                keypoin = sitk.GetArrayFromImage(keypoint_itk)
                keypoin=keypoin.transpose(2,0,1)
                keypoint=keypoin
                print("keypoint=",keypoint.shape)
                keypoint[keypoint==0]=0
                keypoint[keypoint==85]=2
                keypoint[keypoint==170]=1
                keypoint[keypoint==255]=3
                print(np.unique(keypoint))
                print("keypoint=",keypoint.shape)
                # ###img_processing
                image_path = case.replace("mask", "imgs")
                image_itk = sitk.ReadImage(image_path)
                image = sitk.GetArrayFromImage(image_itk)
                image=image.transpose(2,0,1)
                image = MedicalImageDeal(image, percent=1).valid_img
                image = (image - image.min()) / (image.max() - image.min())
                print(image.shape)
                image = image.astype(np.float32)
                item = case.split("/")[-1].split(".")[0]
                if image.shape != label.shape:
                    print("Error")
                print(item)
                f = h5py.File(
                    case.replace('/ODOC','/ODOC_h5').replace('mask/','').replace('.png','.h5'), 'w')
                f.create_dataset(
                    'image', data=image, compression="gzip")
                f.create_dataset('mask', data=label[0], compression="gzip")
                f.create_dataset('scribble', data=scribble[0], compression="gzip")
                f.create_dataset('scribble_noisy', data=scribble_n[0], compression="gzip")
                f.create_dataset('block', data=block[0], compression="gzip")
                f.create_dataset('keypoint', data=keypoint[0], compression="gzip")
                f.close()
            slice_num += 1
print("Converted all ACDC volumes to 2D slices")
print("Total {} slices".format(slice_num))

slice_num = 0
# faz data processing
# for m in [1,2,3,4,5]:
#     for type in ['train','test']:
#         for case in sorted(glob.glob("../data/FAZ/Domain{}/{}/mask/*.png".format(m,type))):
            
#             if type!='train':
#                 # ###label_processing
#                 label_itk = sitk.ReadImage(case)
#                 label = sitk.GetArrayFromImage(label_itk)
#                 label=label.transpose(2,0,1)
#                 # disc
#                 label[label==127]=1
#                 # cup
#                 # ###img_processing
#                 image_path = case.replace("mask", "imgs")
#                 image_itk = sitk.ReadImage(image_path)
#                 image = sitk.GetArrayFromImage(image_itk)
#                 image=image.transpose(2,0,1)
#                 image = MedicalImageDeal(image, percent=1).valid_img
#                 image = (image - image.min()) / (image.max() - image.min())
#                 print(image.shape)
#                 image = image.astype(np.float32)
#                 f = h5py.File(
#                     case.replace('/FAZ','/FAZ_h5').replace('mask/','').replace('.png','.h5'), 'w')
#                 f.create_dataset(
#                     'image', data=image[0], compression="gzip")
#                 f.create_dataset('mask', data=label[0], compression="gzip")
#                 f.close()
                
#             elif type == 'train':
#                 # ###label_processing
#                 label_itk = sitk.ReadImage(case)
#                 label = sitk.GetArrayFromImage(label_itk)
#                 label=label.transpose(2,0,1)
#                 # FAZ
#                 label[label==127]=1
#                 # ###scr_porcessing
#                 scribble_path = case.replace("mask", "scr")
#                 scribble_itk = sitk.ReadImage(scribble_path)
#                 scribbl = sitk.GetArrayFromImage(scribble_itk)
#                 scribble=scribbl.transpose(2,0,1)
#                 scribble[scribble==0]=0
#                 scribble[scribble==127]=1
#                 scribble[scribble==254]=2
#                 print(np.unique(scribble))
#                 print("scribble=",scribble.shape)
#                 # ###scr_noisy_porcessing
#                 scribble_n_path = case.replace("mask", "scr_n")
#                 scribble_n_itk = sitk.ReadImage(scribble_n_path)
#                 scribbl_n = sitk.GetArrayFromImage(scribble_n_itk)
#                 scribble_n=scribbl_n.transpose(2,0,1)
#                 scribble_n[scribble_n==0]=0
#                 scribble_n[scribble_n==127]=1
#                 scribble_n[scribble_n==254]=2
#                 print(np.unique(scribble))
#                 print("scribble_n=",scribble_n.shape)
#                 # ###block_porcessing
#                 block_path = case.replace("mask", "block")
#                 block_itk = sitk.ReadImage(block_path)
#                 bloc = sitk.GetArrayFromImage(block_itk)
#                 block=bloc.transpose(2,0,1)
#                 block[block==0]=0
#                 block[block==127]=1
#                 block[block==254]=2
#                 print(np.unique(block))
#                 print("block=",block.shape)

#                 # ###box
#                 box_path = case.replace("mask", "box")
#                 box_itk = sitk.ReadImage(box_path)
#                 bo = sitk.GetArrayFromImage(box_itk)
#                 box=bo.transpose(2,0,1)
#                 box[box==0]=0
#                 box[box==127]=1
#                 box[box==254]=2
#                 print(np.unique(box))
#                 print("box=",box.shape)
#                 # ###keypoint
#                 keypoint_path = case.replace("mask", "keypoint")
#                 keypoint_itk = sitk.ReadImage(keypoint_path)
#                 keypoin = sitk.GetArrayFromImage(keypoint_itk)
#                 keypoint=keypoin.transpose(2,0,1)
#                 keypoint[keypoint==0]=0
#                 keypoint[keypoint==127]=1
#                 keypoint[keypoint==254]=2
#                 print(np.unique(keypoint))
#                 print("keypoint=",keypoint.shape)
#                 # ###img_processing
#                 image_path = case.replace("mask", "imgs")
#                 image_itk = sitk.ReadImage(image_path)
#                 image = sitk.GetArrayFromImage(image_itk)
#                 image=image.transpose(2,0,1)
#                 image = MedicalImageDeal(image, percent=1).valid_img
#                 image = (image - image.min()) / (image.max() - image.min())
#                 print(image.shape)
#                 image = image.astype(np.float32)
#                 item = case.split("/")[-1].split(".")[0]
#                 if image.shape != label.shape:
#                     print("Error")
#                 print('faz?',np.all(image[0]==image[1])==True)
#                 print(item)
#                 f = h5py.File(
#                     case.replace('/FAZ','/FAZ_h5').replace('mask/','').replace('.png','.h5'), 'w')
#                 f.create_dataset(
#                     'image', data=image[0], compression="gzip")
#                 f.create_dataset('mask', data=label[0], compression="gzip")
#                 f.create_dataset('scribble', data=scribble[0], compression="gzip")
#                 f.create_dataset('scribble_noisy', data=scribble_n[0], compression="gzip")
#                 f.create_dataset('block', data=block[0], compression="gzip")
#                 f.create_dataset('keypoint', data=keypoint[0], compression="gzip")
#                 f.create_dataset('box', data=box[0], compression="gzip")
#                 f.close()
#             slice_num += 1

print("Converted all FAZ volumes to 2D slices")
print("Total {} slices".format(slice_num))