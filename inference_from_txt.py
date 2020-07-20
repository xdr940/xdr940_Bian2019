import torch
from skimage.transform import resize as imresize
from imageio import imread
import numpy as np
from path import Path
import argparse
from tqdm import tqdm
from models import DispNet,DispResNet
import models
import time
from opts.inference_from_txt_opts import inferenece_from_txt_opts
import  matplotlib.pyplot as plt
from utils import readlines

device = torch.device(
    "cuda") if torch.cuda.is_available() else torch.device("cpu")



def load_tensor_image(filename, args):
    img = imread(filename).astype(np.float32)
    h, w, _ = img.shape
    if (h != args.img_height or w != args.img_width):
        img = imresize(img, (args.img_height, args.img_width)).astype(np.float32)
    img = np.transpose(img, (2, 0, 1))
    tensor_img = ((torch.from_numpy(img).unsqueeze(0) / 255 - 0.45) / 0.225).to(device)
    return tensor_img


@torch.no_grad()
def main(args):

    disp_net = models.DispResNet(args.resnet_layers, False).to(device)


    weights_p = Path(args.models_path)/args.model_name
    dataset_path = Path(args.dataset_path)
    root = Path(args.wk_root)

    #model
    weights = torch.load(weights_p)
    disp_net.load_state_dict(weights['state_dict'])
    disp_net.eval()

    #inputs
    txt = root / 'splits' / args.split / args.txt_file
    print('\n')
    print('-> inference file: ', txt)
    print('-> model_path: ', weights_p)


    rel_paths = readlines(txt)


    test_files = []
    if args.split in ['custom','custom_lite','eigen','eigen_zhou']:#kitti
        for item in  rel_paths:
            item = item.split(' ')
            if item[2]=='l':camera ='image_02'
            elif item[2]=='r': camera= 'image_01'
            test_files.append(dataset_path/item[0]/camera/'data'/"{:010d}.png".format(int(item[1])))


    print('-> {} files to test'.format(len(test_files)))

    output_dir = Path(args.output_dir)
    output_dir.makedirs_p()

    avg_time = 0
    for img_p in  tqdm(test_files):
        tgt_img = load_tensor_image(img_p, args)
        # tgt_img = load_tensor_image( dataset_dir + test_files[j], args)

        # compute speed
        torch.cuda.synchronize()
        t_start = time.time()

        output = disp_net(tgt_img)

        torch.cuda.synchronize()
        elapsed_time = time.time() - t_start

        avg_time += elapsed_time

        pred_disp = output.cpu().numpy()[0, 0]

        pred_depth = 1/pred_disp

        rel_path = img_p.relpath(dataset_path)
        if args.split in ['eigen']:
            output_name = str(rel_path).split('/')[-4]+'_{}'.format(rel_path.stem)

        plt.imsave(output_dir/output_name+'.png',pred_disp,cmap='magma')



    avg_time /= len(test_files)
    print('Avg Time: ', avg_time, ' seconds.')
    print('Avg Speed: ', 1.0 / avg_time, ' fps')


if __name__ == '__main__':

    args = inferenece_from_txt_opts().parse()
    main(args)