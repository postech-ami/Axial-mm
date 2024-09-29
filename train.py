# most codes from https://people.csail.mit.edu/tiam/deepmag/
# Refereces
# 1. https://discuss.pytorch.org/t/solved-keyerror-unexpected-key-module-encoder-embedding-weight-in-state-dict/1686

import os
import torch
import time

import numpy as np

from tqdm import tqdm, trange
from glob import glob
from module import magnet
from torchvision import transforms
from dataloader import MM_dataset
from utils import *
import cv2 
from utils import define_filter
from dataset import VideoDataset
from torch.utils.data import DataLoader

class mag(object):
    def __init__(self, args):
        self.args = args
        
        # preprocessing
        self.poisson_noise_n = 0.3

        # iter size
        self.train_batch_size = args.batch_size
        self.val_batch_size = args.batch_size

        # for exponential decay
        self.decay_steps = 3000
        self.lr_decay = 1.0
        self.lr = 0.0002
        self.gamma=0.97
        
        # for optimizer
        self.betal = 0.9

        # for training
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.device_id = [0,1]
        self.batch_size = args.batch_size
        self.is_load = True
        self.num_epoch = args.epochs
        self.remain_epoch = self.num_epoch
        self.tex_loss_w = 1.0
        self.sha_loss_w = 1.0
        self.seed=0
        
        self.n_frames = 0

        # loss
        self.train_losses = []
        self.val_losses = []

        self.cnt=0
        self.steps = 0 
    
    def _load(self):
        torch.manual_seed(self.seed)
        self.model = magnet(self.args, self.device)

        # Training from scratch
        if torch.cuda.device_count()>1 and not self.is_load:
            print("Lets use ", torch.cuda.device_count(), "GPUS!")
            self.model=torch.nn.DataParallel(self.model,device_ids=self.device_id)

        self.model.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, betas=(self.betal, 0.9), weight_decay=0,
                                          amsgrad=False)
        self.criterion = torch.nn.L1Loss()
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=self.decay_steps, gamma=self.gamma)

        if self.is_load:
            checkpoint = torch.load(self.args.checkpoint,map_location=self.device)
            state_dict = checkpoint['model_state_dict']

            if self.args.is_single_gpu_trained:
                from collections import OrderedDict
                new_state_dict = OrderedDict()
                for k, v in state_dict.items():
                    name = k[7:]
                    new_state_dict[name]=v                    
                self.model.load_state_dict(new_state_dict)
            else:
                self.model.load_state_dict(state_dict)

            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.remain_epoch = self.num_epoch
            print("Load pretrained model!")


    def get_data(self):

        train_dataset = MM_dataset(self.args.data_path, split='training',
                                        transform = transforms.Compose([shot_noise(self.poisson_noise_n)]))
        val_dataset = MM_dataset(self.args.data_path, split='validation')

        self.total_batch_size = self.batch_size*len(self.device_id)
        print(f"We are using {self.total_batch_size} batch size")
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                   batch_size=self.total_batch_size,
                                                   num_workers = 4,
                                                   sampler=num_sampler(train_dataset,is_val=False,shuffle=True))
        val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                                 batch_size=self.total_batch_size,
                                                 num_workers = 4,
                                                 sampler=num_sampler(val_dataset))

        return train_dataset, val_dataset, train_loader, val_loader

    def _forward_backward_propagation(self, train_loader, epoch):
        # training
        self.model.train()
        epoch_start = time.perf_counter()
        for i, sample in enumerate(train_loader):
            amplified, frameA, frameB, frameC, mag_map, theta = sample['amplified'].to(self.device), \
                                                            sample['frameA'].to(self.device), \
                                                            sample['frameB'].to(self.device), \
                                                            sample['frameC'].to(self.device), \
                                                            sample['mag_map'].to(self.device), \
                                                            sample['theta'].to(self.device)
            
            
            self.optimizer.zero_grad()
           
            Y, Va, Vb, _, _, Mb_x, Mb_y, Mbb_x, Mbb_y  = self.model(amplified, frameA, frameB, frameC, mag_map, theta)
            recon_loss = self.criterion(Y, amplified)
            texture_loss = self.criterion(Va, Vb)
            shape_loss =  (self.criterion(Mb_x, Mbb_x) + self.criterion(Mb_y, Mbb_y)) * 0.5
            loss = recon_loss + self.tex_loss_w * texture_loss + self.sha_loss_w * shape_loss

            loss.backward()
            self.optimizer.step()
            self.scheduler.step()

            end = time.perf_counter() - epoch_start
         
            self.cnt += 1
            print('[Iter: %d] train_loss: %.3f, dur_total : %.2f' % (i + 1, loss.detach().item(), end), end='\r')
          
            self.total_cnt+=1


        save_folder = os.path.join("checkpoints")
        os.makedirs(save_folder, exist_ok=True)
        torch.save({'epoch': epoch + 1,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'train_loss': loss.detach().item(),
                        'val_loss': 1},
                        save_folder + '/epoch_{}.tar'.format(epoch + 1))


    def play(self, vid_dir, out_dir, amp_x, amp_y, theta, velocity_mag=False):
        """Magnify a video in the two-frames mode.
        Args:
            vid_dir: directory containing video frames. videos are processed
                in sorted order.
            out_dir: directory to place output frames and resulting video.
            amplification_factor: the amplification factor,
                with 0 being no change.
            velocity_mag: if True, process video in Dynamic mode.
        """
        self._load()

        vid_frames = sorted(glob(os.path.join(vid_dir, '*.jpg')))
        if len(vid_frames) == 0:
            vid_frames = sorted(glob(os.path.join(vid_dir, '*.png')))
        height, width = Image.open(vid_frames[0]).size

        if height % 4 != 0 or width % 4 != 0:
            height = height // 4 * 4
            width = width // 4 * 4

        if velocity_mag:
            print("Running in Dynamic mode")
        else:
            print("Running in Static mode")

        # The FPS of the output video is set to 30
        video = cv2.VideoWriter(out_dir, cv2.VideoWriter_fourcc(*'mp4v'), 30, (height, width))

        prev_frame = vid_frames[0]
        desc = out_dir 

        for frame in tqdm(vid_frames, desc=desc):
            out_amp = self.inference(prev_frame, frame, amp_x, amp_y, theta)
            video.write(cv2.cvtColor(out_amp, cv2.COLOR_BGR2RGB))
            if velocity_mag:
                prev_frame = frame
        
        video.release()

    def inference(self, prev_frame, frame, amp_x, amp_y, theta):
        """Run Magnification on two frames.
        Args:
            prev_frame: path to first frame
            frame: path to second frame
            amplification_factor: float for amplification factor
        """
        # convert image to tensor until 200 line
        # Load image and resize, normalize to -1~1
        prev_frame = Image.open(prev_frame)
        frame = Image.open(frame)
        prev_frame = np.asarray(prev_frame, dtype = 'float32') / 127.5 - 1.0
        frame = np.asarray(frame, dtype='float32') / 127.5 - 1.0

        if len(prev_frame.shape) == 3:
            H, W, C = prev_frame.shape
        else:
            H, W = prev_frame.shape

        prev_frame = prev_frame[:H//4*4, :W//4*4, :]
        frame = frame[:H//4*4, :W//4*4, :]

        # change pre_frame, frame, mag_factor to Tensor
        amp_factor = np.array(amp_x, dtype = 'float32')
        sample = {'prev_frame': prev_frame, 'frame': frame, 'mag_factor': amp_factor}
        sample = ToTensor()(sample, istrain=False)

        prev_frame = sample['prev_frame'].unsqueeze(0).to(self.device)
        frame = sample['frame'].unsqueeze(0).to(self.device)

        prev_frame = prev_frame[:, :3, ...]
        frame = frame[:, :3, ...]

        # inference a magnified image   

        # Encoder
        texture_a, shape_ax, shape_ay = self.model.encoder(prev_frame)
        texture_b, shape_bx, shape_by = self.model.encoder(frame)

        # Manipulator
        theta = torch.ones(1).to(self.device) * theta * np.pi / 180 
        mag_factor = torch.ones_like(shape_ax[:, :1, ...]).to(self.device)
        mag_factor = torch.cat([mag_factor * amp_x, mag_factor * amp_y], dim=1)
        mag_factor = mag_factor
        out_shape_enc_x, out_shape_enc_y = self.model.res_manipulator(shape_ax, shape_ay, shape_bx, shape_by, mag_factor, theta)

        # Decoder
        out = self.model.decoder(texture_b, out_shape_enc_x, out_shape_enc_y)

        out = return_save_images(out)

        return out


    def train(self):
        # define for multi-gpu
        self.is_load=False
        self._load() #self.model
        
        # get train, validation dataset
        _, _, train_loader, val_loader = self.get_data()
        torch.cuda.empty_cache()
        self.total_cnt=0
        self.val_cnt=0

        for epoch in range(self.remain_epoch):
            print(f'\nStarting epoch {epoch}\n')
            self._forward_backward_propagation(train_loader, epoch)
            torch.cuda.empty_cache()
        

    def play_temporal(self, vid_dir, out_dir, alpha_x, alpha_y, theta, freq, fs, filter_type, n_filter_tap):
        """Magnify video with a temporal filter.

         Args:
         vid_dir: directory containing video frames videos are processed
              in sorted order.
          out_dir: directory to place output frames and resulting video.
          amplification_factor: the amplification factor,
              with 0 being no change.
          fl: low cutoff frequency.
          fh: high cutoff frequency.
          fs: sampling rate of the video.
          n_filter_tap: number of filter tap to use.
          filter_type: Type of filter to use. Can be one of "butter", or "differenceOfIIR".
          For "differenceOfIIR", fl and fh specifies rl and rh coefficients as in Wadhwa et al.
          # Note
          Not yet construct the FIR filter
        """

        cnt = 0

        videoset = VideoDataset(root=vid_dir)
        dataloader = DataLoader(videoset, batch_size=1, pin_memory=False, shuffle=False, num_workers=0)

        self.n_filter_tap = videoset.__len__()
        filter_a, filter_b = define_filter(fs, freq, filter_type, self.n_filter_tap)

        theta = torch.ones(1).to(self.device) * theta * np.pi / 180 

        self.n_frames = len(videoset)
        self._load()
    
        vid_name = out_dir
        
        print("video name is %s:" % vid_name)
        
        _, height, width = videoset[0].shape
        
        # The FPS of the output video is set to 30
        video = cv2.VideoWriter(vid_name, cv2.VideoWriter_fourcc(*'mp4v'), 30, (width, height))

        save_frames = []
        total_index = 0 

        with torch.no_grad():
            if filter_type=="differenceOfIIR" or filter_type=="butter":
                x_state = []
                y_state = []

                for blob in tqdm(dataloader):
                    
                    sample = blob.to(self.device)
                    sample = sample / 127.5 - 1.0
                    _, _, h, w = sample.shape
                    
                    texture_enc, x_xaxis, x_yaxis = self.model.encoder(sample)
                    x = torch.cat([x_xaxis, x_yaxis], dim=1)

                    x_state.insert(0, x)
                    # set up initial condition.
                    while len(x_state) < len(filter_b):
                        x_state.insert(0, x)
                    if len(x_state) > len(filter_b):
                        x_state = x_state[:len(filter_b)]
                    y = torch.zeros_like(x)
                    for i in range(len(x_state)):
                        y += x_state[i] * filter_b[i]
                    for i in range(len(y_state)):
                        y -= y_state[i] * filter_a[i]

                    # update y state
                    y_state.insert(0, y)    
                    if len(y_state) > len(filter_a):
                        y_state = y_state[:len(filter_a)]
                    
                    mag_factor_x = torch.ones_like(y[:, :1, ...]) * alpha_x
                    mag_factor_y = torch.ones_like(y[:, :1, ...]) * alpha_y
                    mag_factor = torch.cat([mag_factor_x, mag_factor_y], dim=1) 

                    total_index += 1

                    out_enc_xaxis, out_enc_yaxis = self.model.res_manipulator(torch.zeros_like(y[:, :32, ...]),
                                                            torch.zeros_like(y[:, :32, ...]),
                                                            y[:, :32, ...], y[:, 32:, ...], mag_factor, theta)
                    
                    out_enc = torch.cat([out_enc_xaxis, out_enc_yaxis], dim=1)
                    out_enc += x - y

                    out = self.model.decoder(texture_enc, out_enc[:, :32, ...], out_enc[:, 32:, ...]) 


                    save_im = np.array(out[0].permute(1,2,0).clamp(-1, 1).cpu().detach() * 127.5 + 127.5, dtype=np.uint8)
                    save_frames.append(save_im)

            elif filter_type == "fir" :
                x_state = None  
                x_first = None
                idx =0 
                for blob in tqdm(dataloader, desc="Getting encoding"):
                    frame = blob
                    
                    frame = frame.to(self.device)
                    frame = torch.floor(frame) / 127.5 - 1.0
                    _, x_axis, x_yaxis = self.model.encoder(frame)
                    x = torch.cat([x_axis, x_yaxis], dim=1) 

                    if idx == 0:
                        x_first = x.clone().cpu().detach()

                    x = x.cpu().detach()

                    if x_state is None:
                        x_state = torch.zeros(
                            size=x.shape + (len(dataloader),), dtype=torch.float32
                        )
                    x_state[:, :, :, :, idx] = x

                    idx += 1

                x_state = x_state
                
                ## torch.fft code
                filter_b = torch.from_numpy(filter_b).to(self.device)
                filter_fft = torch.fft.fft(
                    torch.fft.ifftshift(filter_b), n=x_state.shape[-1]
                ).to(self.device)
                
                for idx in trange(x_state.shape[1], desc="Applying FIR filter"):
                    x_state_idx = x_state[:, idx, :, :].to(self.device)
                
                    ## torch.fft code
                    x_fft = torch.fft.fft(x_state_idx, dim=-1)
                    x_fft *= filter_fft[np.newaxis, np.newaxis, np.newaxis, :]
                    x_state[:, idx, :, :] = torch.fft.ifft(x_fft).cpu().detach()
                    
                del x_fft, filter_fft, filter_b
                
                x_first = x_first.to(self.device)
                idx = 0 
                for blob in tqdm(dataloader, desc="Getting encoding"):
                    frame = blob[0:1,:]
                    frame = frame.to(self.device)
                    frame = torch.floor(frame) / 127.5 - 1.0
                    texture_enc, _, _ = self.model.encoder(frame)

                    # manipulation and decoding
                    filtered_enc = x_state[:, :, :, :, idx].to(self.device)
                    
                    mag_factor_x = torch.ones_like(filtered_enc[:, :1, ...]) * alpha_x
                    mag_factor_y = torch.ones_like(filtered_enc[:, :1, ...]) * alpha_y
                    mag_factor = torch.cat([mag_factor_x, mag_factor_y], dim=1)

                    output_shape_enc_xaxis, output_shape_enc_y_axis = self.model.res_manipulator(torch.zeros_like(filtered_enc[:, :32, ...]),
                                                                  torch.zeros_like(filtered_enc[:, :32, ...]),
                                                                  filtered_enc[:, :32, ...],
                                                                  filtered_enc[:, 32:, ...], mag_factor, theta)
                    output_shape_enc = torch.cat([output_shape_enc_xaxis, output_shape_enc_y_axis], dim=1)
                    output_shape_enc += x_first - filtered_enc
                    out_amp = self.model.decoder(texture_enc, output_shape_enc[:, :32, ...], output_shape_enc[:, 32:, ...])

                    save_im = np.array(out_amp[0].permute(1,2,0).clamp(-1, 1).cpu().detach() * 127.5 + 127.5, dtype=np.uint8)
                    save_frames.append(save_im)

                    idx += 1

            for  i in range(len(save_frames)):
                video.write(cv2.cvtColor(save_frames[i], cv2.COLOR_BGR2RGB))
            video.release()


