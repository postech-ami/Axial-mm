import os
import os.path as osp
from glob import glob
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
import frame_utils

class MM_dataset(Dataset):
    def __init__(self,  data_dir, split='training', transform=None):
        super(MM_dataset, self).__init__()

        split_num = 500

        if split == 'training':
            self.frameA_list = sorted(glob(osp.join(data_dir, "*A.png")))[:-split_num]
            self.frameB_list = sorted(glob(osp.join(data_dir, "*B.png")))[:-split_num]
            self.frameC_list = sorted(glob(osp.join(data_dir, "*C.png")))[:-split_num]
            self.amp_list = sorted(glob(osp.join(data_dir, "*amp.png")))[:-split_num]
            self.alpha_list = sorted(glob(osp.join(data_dir, "*alpha.flo")))[:-split_num]
            self.theta_list = sorted(glob(osp.join(data_dir, "*theta.txt")))[:-split_num]
            self.transform = transform

        else:
            self.frameA_list = sorted(glob(osp.join(data_dir, "*A.png")))[-split_num:]
            self.frameB_list = sorted(glob(osp.join(data_dir, "*B.png")))[-split_num:]
            self.frameC_list = sorted(glob(osp.join(data_dir, "*C.png")))[-split_num:]
            self.amp_list = sorted(glob(osp.join(data_dir, "*amp.png")))[-split_num:]
            self.alpha_list = sorted(glob(osp.join(data_dir, "*alpha.flo")))[-split_num:]
            self.theta_list = sorted(glob(osp.join(data_dir, "*theta.txt")))[-split_num:]
            self.transform = transform

        print("number of data is %d" % len(self.frameA_list))

    def __getitem__(self, index):

        amp = np.array(Image.open(self.amp_list[index]), dtype=np.float32) / 127.5 - 1.0
        A = np.array(Image.open(self.frameA_list[index]), dtype=np.float32) / 127.5 - 1.0
        B = np.array(Image.open(self.frameB_list[index]), dtype=np.float32) / 127.5 - 1.0
        C = np.array(Image.open(self.frameC_list[index]), dtype=np.float32) / 127.5 - 1.0
        alpha_map = frame_utils.read_gen(self.alpha_list[index])
        theta_file = open(self.theta_list[index], 'r')
        theta = theta_file.readline()
        theta = torch.ones(1) * float(theta)
        theta = theta.unsqueeze(0).unsqueeze(0)

        H, W, _ = A.shape

        # random crop
        x0 = np.random.randint(0, W - 384 - 1)
        y0 = np.random.randint(0, H - 384 - 1)

        A = A[y0:y0+384, x0:x0+384, ...]
        B = B[y0:y0+384, x0:x0+384, ...]
        C = C[y0:y0+384, x0:x0+384, ...]
        amp = amp[y0:y0+384, x0:x0+384, ...]
        alpha_map = alpha_map[y0:y0+384, x0:x0+384, ...]

        amp = torch.from_numpy(amp.copy()).permute(2, 0, 1)
        A = torch.from_numpy(A.copy()).permute(2, 0, 1)
        B = torch.from_numpy(B.copy()).permute(2, 0, 1)
        C = torch.from_numpy(C.copy()).permute(2, 0, 1)
        alpha_map = torch.from_numpy(alpha_map.copy()).permute(2, 0, 1)

        mag_map = F.interpolate(alpha_map.unsqueeze(0), scale_factor=0.5, mode='bilinear')[0]

        sample = {'amplified': amp, 'frameA': A, 'frameB': B, 'frameC': C, 'mag_map': mag_map, 'theta': theta}
        if self.transform is not None:
            sample = self.transform(sample)

        return sample

    def __len__(self):
        return len(self.frameA_list)

