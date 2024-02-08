from torch.utils.data import Dataset
import nibabel as nib
from config import *
from monai.transforms import (
    AddChanneld,
    AsChannelFirstd,
    Compose,
    CropForegroundd,
    LoadImaged,
    NormalizeIntensityd,
    Orientationd,
    RandCropByPosNegLabeld,
    RandSpatialCropSamplesd,
    ScaleIntensityRanged,
    Spacingd,
    SpatialPadd,
    ToTensord,
    RandAffined,
    RandGaussianNoised,
    RandGibbsNoised,
    Resized
)
import config
import numpy as np
from utility import window_center_adjustment

resize_transforms = Compose(
        [   Resized(keys=["image", "label"],spatial_size=(256,256,256), mode = 'nearest'),  
            
        ]
    )

train_transforms = Compose(
        [
            Resized(keys=["image", "label"],spatial_size=(config.RESIZE_SIZE_D,config.RESIZE_SIZE_H,config.RESIZE_SIZE_L), mode = 'nearest'),
            RandCropByPosNegLabeld(
            keys=["image", "label"],
            label_key = "label",
            spatial_size=(config.CROP_SIZE_D,config.CROP_SIZE_H,config.CROP_SIZE_L)
            ,image_key = "image"
            ),
            # RandGaussianNoised(keys = ["image"], prob=0.2, mean=0.0, std=0.05),
            # RandGibbsNoised(keys=["image"], prob = 0.2, alpha = (0.1,0.6)),
            NormalizeIntensityd(keys=['image'],
                                subtrahend=0.5,
                                divisor=0.5),
            ToTensord(keys=["image", "label"]),
        ]
    )

valid_transforms = Compose(
        [
            Resized(keys=["image", "label"],spatial_size=(config.RESIZE_SIZE_D,config.RESIZE_SIZE_H,config.RESIZE_SIZE_L), mode = 'nearest'),
            NormalizeIntensityd(keys=['image'],
                                subtrahend=0.5,
                                divisor=0.5),
            ToTensord(keys=["image", "label"]),
        ]
    )

to_tensor_transform = ToTensord(keys='tensor');

class KneeDataset(Dataset):
    def __init__(self, img_list, mask_list, train = True) -> None:
        super().__init__();
        self.__imgs_list = img_list;
        self.__mask_list = mask_list;
        self.__train = train;
        
    def __len__(self):
        return len(self.__imgs_list);
    def __getitem__(self, index):
        img_path, mask_path = self.__imgs_list[index], self.__mask_list[index];

        #if self.__train is True:
        img = nib.load(img_path);
        mask = nib.load(mask_path);
        canonical_mask = nib.as_closest_canonical(mask)
        canonical_img = nib.as_closest_canonical(img)
        img = canonical_img.get_fdata();
        mask = canonical_mask.get_fdata();
        # else:
        #     img = pickle.load(open(img_path,'rb'));
        #     mask = pickle.load(open(mask_path,'rb'));

        img = np.expand_dims(img[:,:,:], 0);
        mask = np.expand_dims(mask[:,:,:], 0);
        # t = resize_transforms({"image":img, 'label':mask});

        # img = t['image'];
        
        # mask = t['label']
        # mask_val = t['label'].squeeze(dim=0);
       # m = torch.sum(mask);
       # print(m.item());
        
        if self.__train is True:
            #we do range scale here ourselves because each image range might differ.
            img = window_center_adjustment(img, np.max(img));

            #DEBUG TRAIN DATASET
            # img_tmp = img.squeeze();
            # mask_tmp = mask.squeeze();
            # for i in range(img.shape[0]):
            #     msk = mask_tmp[i].astype("uint8")*255;
            #     im = (img_tmp[i]).astype("uint8");
            #     b = cv2.addWeighted(msk, 0.5, im, 0.5, 0.0);
            #     cv2.imshow('mask', msk);
            #     cv2.imshow('im', im);
            #     cv2.imshow('b', b);
            #     cv2.waitKey();
            #===================================

            t = train_transforms({"image":img, 'label':mask})[0];
            img = t['image'].squeeze(dim=0);
            mask = t['label'].squeeze(dim=0);
        else:
            #we do range scale here ourselves because each image range might differ.
            img = window_center_adjustment(img, np.max(img));

            #DEBUG VALID DATASET
#             img_tmp = img.squeeze();
#             mask_tmp = mask.squeeze();
#             for i in range(img_tmp.shape[0]):
#                 msk = mask_tmp[i].astype("uint8")*255;
#                 im = (img_tmp[i]).astype("uint8");
#                 b = cv2.addWeighted(msk, 0.5, im, 0.5, 0.0);
#                 cv2.imshow('mask', msk);
#                 cv2.imshow('im', im);
#                 cv2.imshow('b', b);
#                 cv2.waitKey();
            #===================================

            t = valid_transforms({"image":img, "label":mask});
            img = t['image'].squeeze(dim=0);
            mask = t['label'].squeeze(dim=0);
        
       # print(np.shape(img))
       # print(np.shape(mask))

        return img, mask;