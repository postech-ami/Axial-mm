import os
import argparse
from train import mag
from utils import *

parser = argparse.ArgumentParser(description='')
parser.add_argument('--phase', dest='phase', default='train',
                    help='train, play, play_temporal')
parser.add_argument('--checkpoint_path', dest='checkpoint', default="./model/axial_mm.tar" ,
                    help='Path of checkpoint file for load model')
parser.add_argument('--data_path', dest='data_path', default=None,
                    help='Path of dataset directory for train model')

# for inference
parser.add_argument('--vid_dir', dest='vid_dir', default=None,
                    help='Video folder to run the network on.')
parser.add_argument('--out_dir', dest='out_dir', default=None,
                    help='Output folder of the video run.')
parser.add_argument('--alpha_x',
                    type=float, default=20,
                    help='Magnification factor for x_axis.')
parser.add_argument('--alpha_y',
                    type=float, default=20,
                    help='Magnification factor for y axis.')
parser.add_argument('--theta',
                    type=float, default=45,
                    help='radian.')
parser.add_argument('--velocity_mag', dest='velocity_mag', action='store_true',
                    help='Whether to do velocity magnification.')
parser.add_argument('--is_single_gpu_trained', dest='is_single_gpu_trained', action='store_true',
                    help='Whether the pretrained model was trained on a single gpu.')

# For temporal operation.
parser.add_argument('--fl', dest='fl', type=float, default=0.04,
                    help='Low cutoff Frequency.')      
parser.add_argument('--fh', dest='fh', type=float, default=0.4,
                    help='High cutoff Frequency.')
parser.add_argument('--fs', dest='fs', type=float, default=30,
                    help='Sampling rate.')
parser.add_argument("--freq", type=float, nargs='+', default=[1, 6], help='filter low, high')

parser.add_argument('--n_filter_tap', dest='n_filter_tap', type=int, default=2,
                    help='Number of filter tap required.')
parser.add_argument('--filter_type', dest='filter_type', type=str, default='differenceOfIIR',
                    help='Type of filter to use, must be Butter or differenceOfIIR.')

# For multi-gpu setting
parser.add_argument('-n', '--nodes', default=1, type=int)
parser.add_argument('-g', '--gpus', default=8, type=int, help='number of gpus per node')
parser.add_argument('-nr', '--nr', default=0, type=int, help='ranking within the nodes')
parser.add_argument('--epochs', default=50, type=int)
parser.add_argument('--batch_size', default=4, type=int)

arguments = parser.parse_args()

def main(args):

    model = mag(args)

    if args.phase == 'train':
        model.train()
  
    elif args.phase == "play":
        os.makedirs("./outputs", exist_ok=True)
        save_vid_name = os.path.basename(args.vid_dir) + "_magxy" + str(int(args.alpha_x)) + "_" + str(int(args.alpha_y)) + "_theta" + str(int(args.theta))
        if args.velocity_mag:
            save_vid_name = save_vid_name + "_dynamic.mp4"
        else:
            save_vid_name = save_vid_name + "_static.mp4"

        save_vid_name = os.path.join("./outputs", save_vid_name)

        model.play(args.vid_dir,
                   save_vid_name,
                   args.alpha_x,
                   args.alpha_y,
                   args.theta,
                   args.velocity_mag)
        
    elif args.phase == "play_temporal":
        os.makedirs("./outputs", exist_ok=True)
        save_vid_name = os.path.basename(args.vid_dir) + 'magxy{}_{}_theta{}_fl{}_fh{}_fs{}_n{}_{}.mp4'.format(int(args.alpha_x), int(args.alpha_y), int(args.theta), args.freq[0], args.freq[1], args.fs,
                                                      args.n_filter_tap,
                                                      args.filter_type)
        save_vid_name = os.path.join("./outputs", save_vid_name)

        model.play_temporal(args.vid_dir,
                            save_vid_name,
                            args.alpha_x,
                            args.alpha_y,
                            args.theta,
                            args.freq,
                            args.fs,
                            args.filter_type,                
                            args.n_filter_tap)

    else:
        raise ValueError('Invalid phase argument. '
                         'Expected ["train", "play", "play_temporal"], '
                         'got ' + args.phase)


if __name__ == '__main__':
    main(arguments)
