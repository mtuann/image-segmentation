from os.path import splitext
from os import listdir
import numpy as np
from glob import glob
import torch
from torch.utils.data import Dataset
import logging
from PIL import Image


class BasicDataset(Dataset):
    def __init__(self, imgs_dir, masks_dir, scale=1, mask_suffix=''):
        self.imgs_dir = imgs_dir
        self.masks_dir = masks_dir
        self.scale = scale
        self.mask_suffix = mask_suffix
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'

        self.ids = [splitext(file)[0] for file in listdir(imgs_dir)
                    if not file.startswith('.')]
        logging.info(f'Creating dataset with {len(self.ids)} examples')

    def __len__(self):
        return len(self.ids)

    @classmethod
    def preprocess(cls, pil_img, scale):
        w, h = pil_img.size
#         newW, newH = int(scale * w), int(scale * h)
        newW, newH = 112, 112
        assert newW > 0 and newH > 0, 'Scale is too small'
        pil_img = pil_img.resize((newW, newH))

        img_nd = np.array(pil_img) # (w, h) -> (h, w, c)

        if len(img_nd.shape) == 2:
            img_nd = np.expand_dims(img_nd, axis=2) # HWC (C = 1), extend from (H, W) -> ([ [[], [], .... ] ]) 

        # HWC to CHW
        img_trans = img_nd.transpose((2, 0, 1))
        if img_trans.max() > 1:
            img_trans = img_trans / 255

        return img_trans

    def __getitem__(self, i):
        idx = self.ids[i]
        
        mask_file = glob(self.masks_dir + idx + self.mask_suffix + '.*')
        img_file = glob(self.imgs_dir + idx + '.*')

        assert len(mask_file) == 1, \
            f'Either no mask or multiple masks found for the ID {idx}: {mask_file}'
        assert len(img_file) == 1, \
            f'Either no image or multiple images found for the ID {idx}: {img_file}'
        
        mask = Image.open(mask_file[0]) # (w, h)
        img = Image.open(img_file[0]) # # (w, h)

        assert img.size == mask.size, \
            f'Image and mask {idx} should be the same size, but are {img.size} and {mask.size}'

        img = self.preprocess(img, self.scale)
        mask = self.preprocess(mask, self.scale) # (batch, c, h, w)
        mask = np.expand_dims(mask[0], axis=0) # (batch, 1, h, w)

        return {
            'image': torch.from_numpy(img).type(torch.FloatTensor),
            'mask': torch.from_numpy(mask).type(torch.FloatTensor)
        }


class EchoDataset(BasicDataset):
    def __init__(self, imgs_dir, masks_dir, scale=1):
#         "/data.local/tuannm/Git-Code/image-segmentation/data/imgs"
#         "/data.local/tuannm/Git-Code/image-segmentation/data/masks"
#         imgs_dir = "/data.local/tuannm/Git-Code/image-segmentation/data/imgs/" # them dau "/" de glob
#         masks_dir = "/data.local/tuannm/Git-Code/image-segmentation/data/masks/"
        
        super().__init__(imgs_dir, masks_dir, scale, mask_suffix='') # mask_suffix='_mask'