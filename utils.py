# Some codes from https://github.com/Newmu/dcgan_code
# References
# 1. https://wikidocs.net/57165
# 2. https://pytorch.org/docs/master/_modules/torch/utils/data/sampler.html#Sampler

import torch
import random
import numpy as np
import matplotlib.pyplot as plt
from os.path import *

from PIL import Image
from torch.utils.data.sampler import Sampler
import scipy.signal as sig


import cv2
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

TAG_CHAR = np.array([202021.25], np.float32)

def readFlow(fn):
    """ Read .flo file in Middlebury format"""
    # Code adapted from:
    # http://stackoverflow.com/questions/28013200/reading-middlebury-flow-files-with-python-bytes-array-numpy

    # WARNING: this will work on little-endian architectures (eg Intel x86) only!
    # print 'fn = %s'%(fn)
    with open(fn, 'rb') as f:
        magic = np.fromfile(f, np.float32, count=1)
        if 202021.25 != magic:
            print('Magic number incorrect. Invalid .flo file')
            return None
        else:
            w = np.fromfile(f, np.int32, count=1)
            h = np.fromfile(f, np.int32, count=1)
            # print 'Reading %d x %d flo file\n' % (w, h)
            data = np.fromfile(f, np.float32, count=2*int(w)*int(h))
            # Reshape data into 3D array (columns, rows, bands)
            # The reshape here is for visualization, the original code is (w,h,2)
            return np.resize(data, (int(h), int(w), 2))


def read_gen(file_name, pil=False):
    ext = splitext(file_name)[-1]
    if ext == '.png' or ext == '.jpeg' or ext == '.ppm' or ext == '.jpg':
        return Image.open(file_name)
    elif ext == '.bin' or ext == '.raw':
        return np.load(file_name)
    elif ext == '.flo':
        return readFlow(file_name).astype(np.float32)
    return []

class ToTensor(object):
  def __call__(self, sample, istrain=True, number=1):
    if istrain:
      amplified, frameA, frameB, frameC, mag_factor = sample['amplified'], sample['frameA'], sample['frameB'], sample['frameC'], sample['mag_factor']
      
      # swap color axis because
      # numpy image: H x W x C
      # torch image: C X H X W
      amplified = amplified.transpose((2, 0, 1))
      frameA = frameA.transpose((2, 0, 1))
      frameB = frameB.transpose((2, 0, 1))
      frameC = frameC.transpose((2, 0, 1))

      # convert tensor
      amplified = torch.from_numpy(amplified)
      frameA = torch.from_numpy(frameA)
      frameB = torch.from_numpy(frameB)
      frameC = torch.from_numpy(frameC)
      mag_factor = torch.from_numpy(mag_factor)
      mag_factor = mag_factor.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)

      ToTensor_sample = {'amplified': amplified, 'frameA': frameA, 'frameB': frameB, 'frameC': frameC, 'mag_factor': mag_factor}
    else:
      
      if number == 1:
          prev_frame, frame, mag_factor = sample['prev_frame'], sample['frame'], sample['mag_factor']
      
          # swap color axis because
          # numpy image: H x W x C
          # torch image: C X H X W
          prev_frame = prev_frame.transpose((2, 0, 1))
          frame = frame.transpose((2, 0, 1))

          # convert tensor
          prev_frame = torch.from_numpy(prev_frame)
          frame = torch.from_numpy(frame)
          mag_factor = torch.from_numpy(mag_factor)
          mag_factor = mag_factor.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)

          ToTensor_sample = {'prev_frame': prev_frame, 'frame': frame, 'mag_factor': mag_factor}

      else:
          prev_frame_x, prev_frame_y, frame_x, frame_y, mag_factor = sample['prev_frame_x'], sample['prev_frame_y'], sample['frame_x'], sample['frame_y'], sample['mag_factor']
          
          # swap color axis because
          # numpy image: H x W x C
          # torch image: C X H X W
          prev_frame_x = prev_frame_x.transpose((2, 0, 1))
          prev_frame_y = prev_frame_y.transpose((2, 0, 1))
          frame_x = frame_x.transpose((2, 0, 1))
          frame_y = frame_y.transpose((2, 0, 1))

          # convert tensor
          prev_frame_x = torch.from_numpy(prev_frame_x)
          prev_frame_y = torch.from_numpy(prev_frame_y)
          frame_x = torch.from_numpy(frame_x)
          frame_y = torch.from_numpy(frame_y)

          mag_factor = torch.from_numpy(mag_factor)
          mag_factor = mag_factor.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)

          ToTensor_sample = {'prev_frame_x': prev_frame_x, 'prev_frame_y': prev_frame_y, 'frame_x': frame_x, 'frame_y': frame_y, 'mag_factor': mag_factor}

    return ToTensor_sample

class ToNumpy(object):
  def __call__(self, amplified):
    amplified = amplified.permute((1, 2, 0))
    amplified = amplified.cpu().detach().numpy()
    return amplified

class shot_noise(object):
  # This function approximate poisson noise upto 2nd order.
  def __init__(self, n):
    self.n = n

  def _get_shot_noise(self, image):
    n = torch.zeros_like(image).normal_(mean=0.0, std=1.0)
    # strength ~ sqrt image value in 255, divided by 127.5 to convert
    # back to -1, 1 range.

    n_str = torch.sqrt(torch.as_tensor(image + 1.0)) / torch.sqrt(torch.as_tensor(127.5))
    return torch.mul(n, n_str)

  def _preproc_shot_noise(self, image, n):
    nn = np.random.uniform(0, n)
    return (image + nn * self._get_shot_noise(image)).clamp(-1, 1)

  def __call__(self, sample):
    amplified, frameA, frameB, frameC, mag_map, theta = sample['amplified'], sample['frameA'], sample['frameB'], sample['frameC'], sample['mag_map'], sample['theta']
    # add shot noise
    frameA = self._preproc_shot_noise(frameA, self.n)
    frameB = self._preproc_shot_noise(frameB, self.n)
    frameC = self._preproc_shot_noise(frameC, self.n)

    preproc_sample = {'amplified': amplified, 'frameA': frameA, 'frameB': frameB, 'frameC': frameC, 'mag_map': mag_map, 'theta': theta}
    return preproc_sample

class num_sampler(Sampler):
# Sampling a specific number of multiple-th indices from data.
  def __init__(self, data, is_val=True, shuffle=False, num=10):
    self.num_samples = len(data)
    self.is_val = is_val
    self.shuffle = shuffle
    self.num = num

  def __iter__(self):
    k = []
    for i in range(self.num_samples):
      if self.is_val: # case of validation dataset
        if i%self.num == self.num-1:
          k.append(i)
      else: # case of train dataset
        if i%self.num != self.num-1:
          k.append(i)

    if self.shuffle:
      random.shuffle(k)
    return iter(k)

  def __len__(self):
    return self.num_samples

def inverse_transform(image):
    return (image + 1.) / 2.

def imsave(im, path):
    if issubclass(im.dtype.type, np.floating):
        im = im * 255
        im = im.astype('uint8')
    im = Image.fromarray(im)
    return im.save(path,"PNG")
  
def get_npimg(im):
    if issubclass(im.dtype.type, np.floating):
          im = im * 255
          im = im.astype('uint8')
    return im

def save_images(image, im_path):
      result = image.squeeze()
      result = ToNumpy()(result)
      result = np.clip(result, -1, 1)
      result = inverse_transform(result)
      result = get_npimg(result)
      plt.imsave(im_path, result)
      print('save image at {}'.format(im_path))

def return_save_images(image):
      result = image.squeeze()
      result = ToNumpy()(result)
      result = np.clip(result, -1, 1)
      result = inverse_transform(result)
      result = get_npimg(result)

      return result


def gen_img(image):
    result = image.squeeze()
    result = ToNumpy()(result)
    result = np.clip(result, -1, 1)
    result = inverse_transform(result)
    result = get_npimg(result)
    result = np.transpose(result, (2,0,1))

    return result

def summary_imgs(amp, A):
    print("amp shape :",amp.shape)
    amp=gen_img(amp[0])
    amp_gt = gen_img(A[0])

    return amp, amp_gt


def define_filter(FS, freq, filter_type, n_filter_tap):
    FS = FS 
    nyq = FS / 2.0
    n=1

    assert len(freq) == 2, "freq must be [low, high]"
    
    fc=np.array([freq[0], freq[1]])
    w_c=2*fc/FS

    if filter_type == "iir":
        [filter_b,a] = sig.cheby1(n, 20, w_c, btype='bandpass')
        filter_a = [a[1],a[2]]
        
    elif filter_type == "fir":
        filter_b = sig.firwin(n_filter_tap, [fc[0], fc[1]], nyq=nyq, pass_zero=False) # n_filter_tap = 2
        filter_a = []
        
    elif filter_type == "butter":
        filter_b, filter_a = sig.butter(1, [fc[0]/nyq, fc[1]/nyq], btype='bandpass') # n_filter_tap = 2
        filter_a = filter_a[1:]
        
    elif filter_type == "differenceOfIIR":
        filter_b = [fc[1] - fc[0], fc[0] - fc[1]]
        filter_a = [-1.0*(2.0 - fc[1] - fc[0]), (1.0 - fc[0]) * (1.0 - fc[1])]
    elif filter_type == "dynamic":
        filter_b, filter_a = [], []
    elif filter_type == "static":
        filter_b, filter_a = [], []
    else:
        raise ValueError('Filter type must be either '
                        '["fir", "butter", "differenceOfIIR"] got ' + \
                        filter_type)
    return filter_a, filter_b