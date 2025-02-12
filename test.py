from __future__ import print_function, absolute_import
import os
import sys
import cv2
import time
import datetime
import warnings
import argparse
import os.path as osp
import numpy as np
import tifffile

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch.optim import lr_scheduler

from data_manager import Climate_dataset

from data_loader import Climate_dataloader
import transforms as T
import models
from models.Unet import *
from utils import AverageMeter, Logger, save_checkpoint


parser = argparse.ArgumentParser(description='Train video model with cross entropy loss')
# Datasets
parser.add_argument('-j', '--workers', default=6, type=int,
                    help="number of data loading workers (default: 4)")
parser.add_argument('--pin-memory', default=True, type=bool)

parser.add_argument('--max', default=100, type=float)

# Optimization options
parser.add_argument('--max-epoch', default=50, type=int,
                    help="maximum epochs to run")
parser.add_argument('--start-epoch', default=0, type=int,
                    help="manual epoch number (useful on restarts)")
parser.add_argument('--train-batch', default=30, type=int,
                    help="train batch size")
parser.add_argument('--test-batch', default=30, type=int, )
parser.add_argument('--lr', '--learning-rate', default=0.0001, type=float,
                    help="initial learning rate, use 0.0001 for rnn, use 0.0003 for pooling and attention")
parser.add_argument('--stepsize', default=25, type=int,
                    help="stepsize to decay learning rate (>0 means this is enabled)")
parser.add_argument('--gamma', default=0.1, type=float,
                    help="learning rate decay")
parser.add_argument('--weight-decay', default=5e-04, type=float,
                    help="weight decay (default: 5e-04)")
parser.add_argument('--margin', type=float, default=0.3, help="margin for triplet loss")

parser.add_argument('--gap', type=int, default=7)
parser.add_argument('--train_portion', type=float, default=0.7)


# Miscs
parser.add_argument('--seed', type=int, default=1, help="manual seed")
parser.add_argument('--gpu-devices', default='0', type=str, help='gpu device ids for CUDA_VISIBLE_DEVICES')

parser.add_argument('--save_dir', default='/data/alven/wenhuasdrive/MNHS-SPHPM-CARE/GLOBALENVIRONHealth/CMIP6/Generated_by_zk/Testing_results_final/fire_o3_sigmoid', type=str)
args = parser.parse_args()


def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_devices

    # setting the seed
    torch.manual_seed(args.seed)
    cudnn.benchmark = True
    torch.cuda.manual_seed_all(args.seed)

    print("==========\nArgs:{}\n==========".format(args))
   
    print("Initializing dataset")
    dataset = Climate_dataset(num_train=args.train_portion, gap=args.gap)

    # loading the land mark
    landmark = load_landmark()
    elevation = load_elevation()

    # mean and std for indicators
    mean_indicators = torch.Tensor([65.30, 6.60, 1.20, 26.96, 18.06, 29.06, 25.73, 26.23, 25.23])[:, None, None].cuda()
    std_indicators = torch.Tensor([28.47, 6.11, 3.01, 10.52, 9.47, 16.38, 7.39, 7.54, 7.23])[:, None, None].cuda()


    valloader = DataLoader(
        Climate_dataloader(dataset.val_data, mean=mean_indicators, std=std_indicators),
        batch_size=args.test_batch, shuffle=False, num_workers=args.workers,
        pin_memory=args.pin_memory, drop_last=False,
    )

    print("Initializing model:")
    model = UNet()
    model = nn.DataParallel(model).cuda()
    model.load_state_dict(torch.load('best.pth'))


    rMSE = test(args, model, valloader, landmark, elevation, mean=mean_indicators, std=std_indicators, save_dir=args.save_dir)



def test(args, model, valloader, landmark, elevation, mean, std, save_dir):
    model.eval()
    samples_scaled = []
    samples_normalized = []
    ground_truth = []
    landmark_tensor = torch.Tensor(landmark * 1).cuda()

    with torch.no_grad():

        for batch_idx, (indicators_today, indicators_target, total_O3_today, total_O3_target, today_embedding, target_embedding, target_date) in enumerate(valloader):
            
            indicators_today, indicators_target = indicators_today.cuda(), indicators_target.cuda()
            today_embedding, target_embedding = today_embedding.cuda(), target_embedding.cuda()

            # for indicators, we apply the z-score normalization
            indicators_today = normalizaton_z_score(indicators_today, mean=mean, std=std)
            indicators_target = normalizaton_z_score(indicators_target, mean=mean, std=std)

            b = indicators_today.size(0)

            indicators_today = torch.cat((indicators_today, elevation.repeat(b, 1, 1, 1)), dim=1)
            indicators_target = torch.cat((indicators_target, elevation.repeat(b, 1, 1, 1)), dim=1)

            # for O3, PM25, we apply the max-min normalization
            total_O3_today = total_O3_today.cuda()/args.max
            total_O3_target = total_O3_target.cuda()[:, :, landmark]

            
            factors = total_O3_today
            
            outputs = model(indicators_today, indicators_target, factors, today_embedding, target_embedding)
            outputs = torch.clip(outputs, min=0, max=1)

            ###############
            # for rMSE calculation
            outputs = outputs * args.max * landmark_tensor[None, None, :, :]
            
            for i in range(b):
                date = target_date[i]

                save_tif = outputs[i] 
                outputs_temp = outputs[i, :, landmark]
                rmse = torch.sqrt( ((outputs_temp - total_O3_target[i])**2 ).sum() / 272862)

                save_tif = save_tif.cpu().numpy()
                name = date.split('.')[0] + '_' + str(rmse.item())[0:5] + '.tiff'
                tifffile.imsave(os.path.join(save_dir, name), save_tif)
                # breakpoint()


    return None


def load_landmark(path='/data/alven/wenhuasdrive/MNHS-SPHPM-CARE/GLOBALENVIRONHealth/CMIP6/Generated_by_zk/land_zk.png'):
    landmark = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    landmark = landmark == 255
    return landmark

def load_elevation(path='/data/alven/wenhuasdrive/MNHS-SPHPM-CARE/GLOBALENVIRONHealth/CMIP6/Generated_by_zk/elevation_zk.png'):
    elevation = cv2.imread(path, cv2.IMREAD_GRAYSCALE)/255

    elevation = torch.Tensor(elevation).cuda()[None, None, :, :] #.repeat(bs, 1, 1, 1)
    return elevation

def normalizaton_z_score(tensor, mean, std):
    return (tensor - mean) / std

def normalization_min_max(tensor, max, min):
    return (tensor - min) / (max - min)


if __name__ == '__main__':
    main()
