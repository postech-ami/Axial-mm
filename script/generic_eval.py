import argparse
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from pytorch_msssim import SSIM
from glob import glob
import os.path as osp
from os import path
import sys

sys.path.append(path.dirname( path.dirname( path.abspath(__file__) ) ))

from module import magnet

def img_load_and_normalize(dir, device):
    img = Image.open(dir)
    img = np.array(img).astype(np.uint8)[..., :3]
    img = torch.from_numpy(img).permute(2, 0, 1).float().unsqueeze(0).to(device)
    img = img / 127.5 - 1.0

    return img

def main(args):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Create model.
    model = magnet(args, device)

    # Load a trained model.
    print("=> loading checkpoint '{}'".format(args.checkpoint))
    checkpoint = torch.load(args.checkpoint ,map_location=device)
    state_dict = checkpoint['model_state_dict']
   
    if args.is_single_gpu_trained:
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:]
            new_state_dict[name]=v                    
        model.load_state_dict(new_state_dict)
    else:
        model.load_state_dict(state_dict)
    model.load_state_dict(new_state_dict)
    model.to(device)
    model.eval()

    # Evalution data setting
    generation_mode = args.mode # ["subpixel, noise005"]

    # The subpixel experiment measures the network's performance for various magnitudes of small motions.
    if args.mode == "subpixel":
        subpixel_motion_list = [0.040, 0.050, 0.063, 0.079, 0.10, 0.13, 0.16, 0.20, 0.25, 0.32, 0.40, 0.50, 0.63, 0.79, 1.00] # 
        alpha_list = []

        # Set the motion magnification factor (alpha) so that the amplified motion reaches a magnitude of 10 pixels, 
        # following the method described in the referenced paper:
        # Referenced paper: Oh, Tae-Hyun, et al. "Learning-based video motion magnification." Proceedings of the European Conference on Computer Vision (ECCV). 2018.
        for idx in range(len(subpixel_motion_list)):
            alpha = 10 / subpixel_motion_list[idx] 
            alpha_list.append(alpha)

    # The noise005 experiment measures the network's performance under various levels of additive noise when the small motion is 0.05 pixels.
    # This is an important experiment due to the challenge of distinguishing between small motion and photometric noise, as presented in the following reference paper.
    # Referenced paper: Oh, Tae-Hyun, et al. "Learning-based video motion magnification." Proceedings of the European Conference on Computer Vision (ECCV). 2018.
    elif generation_mode == "noise005":
        subpixel_motion_list = []
        alpha_list = []
        
        # Set the motion magnification factor (alpha) so that the amplified motion reaches a magnitude of 10 pixels, 
        # following the method described in the referenced paper:
        # Referenced paper: Oh, Tae-Hyun, et al. "Learning-based video motion magnification." Proceedings of the European Conference on Computer Vision (ECCV). 2018.
        for idx in range(21):
            subpixel_motion_list.append(0.05)
            alpha_list.append(10 / 0.05)
            
    # Evaluation metric is SSIM
    ssim_module = SSIM(data_range=255, size_average=True, channel=3) # channel=1 for grayscale images

    print("Data mode is %s" % args.mode)

    for idx in range(len(alpha_list)):
        data_path = args.data_path
        modefolder = "mode%02d" % idx
        image1_list = sorted(glob(osp.join(data_path, generation_mode, modefolder, '*A.png')))
        image2_list = sorted(glob(osp.join(data_path, generation_mode, modefolder, '*B.png')))
        ground_truth_list = sorted(glob(osp.join(data_path, generation_mode, modefolder, '*Y.png')))
        
        model_ssim_list = [] 
        input_ssim_list = []

        # Each mode is evaluated with 1,000 data points. 
        # For each mode, the subpixel test uses different magnitudes of small motions, while the noise005 test employs different levels of additive noise.
        with torch.no_grad():
            for i in range(1000):
            
                img1 = F.pad(img_load_and_normalize(image1_list[i], device), (5, 5, 5, 5), "constant", -1)
                img2 = F.pad(img_load_and_normalize(image2_list[i], device), (5, 5, 5, 5), "constant", -1)
                gt = F.pad(img_load_and_normalize(ground_truth_list[i], device), (5, 5, 5, 5), "constant", -1)
                alpha = torch.ones([1, 2, 1, 1], device=device).float() * alpha_list[idx]
                theta = torch.zeros(1, device=device).float() 

                # Our model takes theta as an input, but in other traditional (generic) motion magnification methods, theta can be omitted.
                output = model.inference(img1, img2, alpha, theta) 

                img1 = img1[:, :, 8:-8, 8:-8]
                img2 = img2[:, :, 8:-8, 8:-8]
                gt = gt[:, :, 8:-8, 8:-8]
                output = output[:, :, 8:-8, 8:-8]

                input = (torch.clamp(img2, min=-1, max=+1) + 1.) / 2. * 255
                gt = (torch.clamp(gt, min=-1, max=+1) + 1.) / 2. * 255
                output = (torch.clamp(output, min=-1, max=+1) + 1.) / 2. * 255
                
                ours_ssim_val = ssim_module(output, gt)
                # The SSIM between the input image and the GT image is provided as a reference value.
                input_ssim_val = ssim_module(input, gt)

                model_ssim_list.append(ours_ssim_val.detach().cpu().item())
                input_ssim_list.append(input_ssim_val.detach().cpu().item())


        messsage = "Mode%02d; SSIM of model; %.4f; SSIM of input; %.4f;" % (idx, 
        np.mean(model_ssim_list),
        np.mean(input_ssim_list)
        )
        print(messsage) 

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Swin Transformer Based Video Motion Magnification')

    # evluation parameters
    parser.add_argument('--mode', type=str, default="subpixel")
    parser.add_argument('--data_path', required=True)
    parser.add_argument('--checkpoint_path', dest='checkpoint', default="./model/axial_mm.tar",
                    help='Path of checkpoint file for load model')
    parser.add_argument('--is_single_gpu_trained', dest='is_single_gpu_trained', action='store_true',
                        help='Whether the pretrained model was trained on a single gpu.')

    args = parser.parse_args()

    main(args)
