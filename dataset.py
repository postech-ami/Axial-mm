import os
import os.path as osp
from os.path import splitext
from glob import glob
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

def read_gen(file_name, pil=False):
    ext = splitext(file_name)[-1]
    if ext == '.png' or ext == '.jpeg' or ext == '.ppm' or ext == '.jpg':
        return Image.open(file_name)

class VideoDataset(Dataset):
    def __init__(self,
                 root: str):
        super(VideoDataset, self).__init__()
        self.root = root
        self.image_list = []
        self.diff = []
      
        images = sorted(glob(osp.join(self.root, '*.jpg')))
        if len(images) == 0:
            images = sorted(glob(osp.join(self.root, '*.png')))

        self.image_list = images

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        img = Image.open(self.image_list[idx])
        img = np.array(img).astype(np.uint8)[..., :3]
        H, W, _ = img.shape
        img = img[:H//4*4, :W//4*4, :]
        img = torch.from_numpy(img).permute(2, 0, 1).float()
        
        return img
