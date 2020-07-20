
import argparse

class train_opts:
    def __init__(self):
        


        self.parser = argparse.ArgumentParser(description='Unsupervised Scale-consistent Depth and Ego-motion Learning from Monocular Video (KITTI and CityScapes)',
                                         formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        self.parser.add_argument('--data', metavar='DIR', help='path to dataset',
                            default="/home/roit/datasets/kitti_416128")
        self.parser.add_argument('--sequence-length', type=int, metavar='N',
                            help='sequence length for training', default=3)
        self.parser.add_argument('--dataset-format', default='sequential', metavar='STR',
                            help='dataset format, stacked: stacked frames (from original TensorFlow code) \
                            sequential: sequential folders (easier to convert to with a non KITTI/Cityscape dataset')
        self.parser.add_argument('--padding-mode', type=str, choices=['zeros', 'border'], default='zeros',
                            help='padding mode for image warping : this is important for photometric differenciation when going outside target image.'
                            ' zeros will null gradients outside target image.'
                            ' border will only null gradients of the coordinate outside (x or y)')
        self.parser.add_argument('-j', '--workers', default=4, type=int,
                            metavar='N', help='number of data loading workers')
        self.parser.add_argument('--epochs', default=20, type=int,
                            metavar='N', help='number of total epochs to run')
        self.parser.add_argument('--epoch-size', default=1000, type=int, metavar='N',
                            help='manual epoch size (will match dataset size if not set)')
        self.parser.add_argument('-b', '--batch-size', default=8,
                            type=int, metavar='N', help='mini-batch size')
        self.parser.add_argument('--lr', '--learning-rate', default=1e-4,
                            type=float, metavar='LR', help='initial learning rate')
        self.parser.add_argument('--momentum', default=0.9, type=float,
                            metavar='M', help='momentum for sgd, alpha parameter for adam')
        self.parser.add_argument('--beta', default=0.999, type=float,
                            metavar='M', help='beta parameters for adam')
        self.parser.add_argument('--weight-decay', '--wd', default=0,
                            type=float, metavar='W', help='weight decay')
        self.parser.add_argument('--print-freq', default=10, type=int,
                            metavar='N', help='print frequency')
        self.parser.add_argument('--pretrained-disp', dest='pretrained_disp',
                            metavar='PATH', help='path to pre-trained dispnet model',
                            #default='/home/roit/models/SCBian/dispnet_model_best.pth'
                                 default=None
                                 )
        self.parser.add_argument('--pretrained-pose', dest='pretrained_pose',
                            metavar='PATH', help='path to pre-trained Pose net model',
                            #default='/home/roit/models/SCBian/k_pose.tar'
                            default=None
                            )
        self.parser.add_argument('--seed', default=0, type=int,
                            help='seed for random functions, and network initialization')
        self.parser.add_argument('--log-summary', default='progress_log_summary.csv',
                            metavar='PATH', help='csv where to save per-epoch train and valid stats')
        self.parser.add_argument('--log-full', default='progress_log_full.csv',
                            metavar='PATH', help='csv where to save per-gradient descent train stats')
        self.parser.add_argument('--with-gt', action='store_true', help='use ground truth for validation. \
                            You need to store it in npy 2D arrays see data/kitti_raw_loader.py for an example')
        self.parser.add_argument('--dispnet', dest='dispnet', type=str, default='DispNet',
                            choices=['DispNet', 'DispResNet'], help='depth network architecture.')
        self.parser.add_argument('--num-scales', '--number-of-scales',
                            type=int, help='the number of scales', metavar='W', default=1)
        self.parser.add_argument('-p', '--photo-loss-weight', type=float,
                            help='weight for photometric loss', metavar='W', default=1)
        self.parser.add_argument('-s', '--smooth-loss-weight', type=float,
                            help='weight for disparity smoothness loss', metavar='W', default=0.1)
        self.parser.add_argument('-c', '--geometry-consistency-weight', type=float,
                            help='weight for depth consistency loss', metavar='W', default=0.5)
        self.parser.add_argument('--with-ssim', action='store_true', help='use ssim loss',)
        self.parser.add_argument('--with-mask', action='store_true',
                            help='use the the mask for handling moving objects and occlusions')
        #self.parser.add_argument('--name', dest='name', type=str, required=True,
        #                    help='name of the experiment, checkpoints are stored in checpoints/name')

    def parse(self):
        self.options = self.parser.parse_args()
        return self.options