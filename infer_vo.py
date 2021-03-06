import torch
from imageio import imread, imsave
from skimage.transform import resize as imresize
import numpy as np
from path import Path
import argparse
from tqdm import tqdm

from inverse_warp import *
from scipy.ndimage.interpolation import zoom
import matplotlib.pyplot as plt
import models
import cv2

parser = argparse.ArgumentParser(description='Script for visualizing depth map and masks',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--pretrained-posenet",
                    type=str, help="pretrained PoseNet path",
                    #default="/home/roit/models/SCBian/exp_pose_model_best.pth",
                    #default='/home/roit/models/SCBian/dispnet_model_best.pth'
                    default='/home/roit/models/SCBian/k_pose.tar'
                    )
parser.add_argument('--scale_factor',default=32.4)

parser.add_argument("--img-height", default=256, type=int, help="Image height")
parser.add_argument("--img-width", default=832, type=int, help="Image width")
parser.add_argument("--dataset-dir", type=str, help="Dataset directory",
                    #default='/mnt/datasets/kitti_odo_gray/dataset/sequences/'
                    default='/media/roit/greenp2/datasets/kitti_odo_color/sequences/'
                    )
parser.add_argument("--output-dir", type=str,
                    help="Output directory for saving predictions in a big 3D numpy file",
                    default='./')
parser.add_argument("--img-exts", default=['png', 'jpg', 'bmp'],
                    nargs='*', type=str, help="images extensions to glob")
parser.add_argument('--type',default='color',choices=['gray','color'])
parser.add_argument("--sequence", default='05',
                    type=str, help="sequence to test")

device = torch.device(
    "cuda") if torch.cuda.is_available() else torch.device("cpu")


def load_tensor_image(filename, args):
    img = imread(filename).astype(np.float32)
    h, w, _ = img.shape
    if (h != args.img_height or w != args.img_width):
        img = imresize(img, (args.img_height, args.img_width)
                       ).astype(np.float32)
    img = np.transpose(img, (2, 0, 1))
    tensor_img = (
        (torch.from_numpy(img).unsqueeze(0)/255 - 0.5)/0.5).to(device)
    return tensor_img


@torch.no_grad()
def main():
    args = parser.parse_args()
    print('->model load: {}'.format(args.pretrained_posenet))
    weights_pose = torch.load(args.pretrained_posenet)
    pose_net = models.PoseNet().to(device)
    pose_net.load_state_dict(weights_pose['state_dict'], strict=False)
    pose_net.eval()
    if args.type =='gray':
        image_dir = Path(args.dataset_dir + args.sequence + "/image_1/")#gray 01, color 23
    else:
        image_dir = Path(args.dataset_dir + args.sequence + "/image_2/")#gray 01, color 23

    output_dir = Path(args.output_dir)
    print('-> out dif {}'.format(output_dir))
    output_dir.makedirs_p()

    test_files = sum([image_dir.files('*.{}'.format(ext))
                      for ext in args.img_exts], [])
    test_files.sort()
    print('{} files to test'.format(len(test_files)))

    global_pose = np.identity(4)
    poses = [global_pose[0:3, :].reshape(1, 12)]

    n = len(test_files)
    tensor_img1 = load_tensor_image(test_files[0], args)

    for iter in tqdm(range(n - 1)):
        tensor_img2 = load_tensor_image(test_files[iter+1], args)
        pose = pose_net(tensor_img1, tensor_img2)
        pose_mat = pose_vec2mat(pose).squeeze(0).cpu().numpy()#1,6-->3x4


        pose_mat = np.vstack([pose_mat, np.array([0, 0, 0, 1])])#4X4
        global_pose = global_pose @ np.linalg.inv(pose_mat)

        pose = global_pose[0:3, :].reshape(1, 12)


        poses.append(pose)

        # update
        tensor_img1 = tensor_img2

    poses = np.concatenate(poses, axis=0)
    if args.scale_factor:
            poses[:,3]*=args.scale_factor#x-axis
            poses[:,11]*=args.scale_factor#z-axis
    filename = Path(args.output_dir + args.sequence + ".txt")
    np.savetxt(filename, poses, delimiter=' ', fmt='%1.8e')


if __name__ == '__main__':
    main()
