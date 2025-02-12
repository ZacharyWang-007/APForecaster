from __future__ import print_function, absolute_import
import os
import sys
import cv2
import argparse
import os.path as osp

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler

from data_manager import Climate_dataset

from data_loader import Climate_dataloader
import transforms as T
from models.Unet import *
from utils import AverageMeter, Logger


parser = argparse.ArgumentParser(description='Train video model with cross entropy loss')
# Datasets and loaders
parser.add_argument('--gap', type=int, 
                    default=7)
parser.add_argument('--train_portion', type=float, 
                    default=0.7)

parser.add_argument('--max', type=float, 
                    default=100)
parser.add_argument('--high', type=float, 
                    default=60)
parser.add_argument('--print-step', type=float, 
                    default=40)

#################
parser.add_argument('--targets', type=str, 
                    default='o3', help='pm25 or o3')
parser.add_argument('--path-indicators', type=str, 
                    default='/data/alven/wenhuasdrive/MNHS-SPHPM-CARE/GLOBALENVIRONHealth/CMIP6/Generated_by_zk/Upload_GitHub/indicators/historical')
parser.add_argument('--path-pm25-o3', type=str, 
                    default='/data/alven/wenhuasdrive/MNHS-SPHPM-CARE/GLOBALENVIRONHealth/CMIP6/Generated_by_zk/Upload_GitHub/PM25_O3')
parser.add_argument('--land', type=str, 
                    default='/data/alven/wenhuasdrive/MNHS-SPHPM-CARE/GLOBALENVIRONHealth/CMIP6/Generated_by_zk/Upload_GitHub/land_zk.png')
parser.add_argument('--mask', type=str, 
                    default='/data/alven/wenhuasdrive/MNHS-SPHPM-CARE/GLOBALENVIRONHealth/CMIP6/Generated_by_zk/Upload_GitHub/elevation_zk.png')
#################

parser.add_argument('-j', '--workers', type=int, 
                    default=9)
parser.add_argument('--pin-memory', type=bool, 
                    default=True)

# Optimization options
parser.add_argument('--max-epoch', type=int, 
                    default=50)
parser.add_argument('--train-batch', type=int, 
                    default=30)
parser.add_argument('--test-batch', type=int, 
                    default=30)

parser.add_argument('--lr', '--learning-rate', type=float, 
                    default=0.0001)
parser.add_argument('--stepsize', type=int, 
                    default=25, help="stepsize to decay learning rate")
parser.add_argument('--gamma', type=float, 
                    default=0.1, help="learning rate decay")
parser.add_argument('--weight-decay', type=float, 
                    default=5e-04, help="weight decay (default: 5e-04)")

parser.add_argument('--seed', type=int, 
                    default=1)
parser.add_argument('--gpu-devices', type=str, 
                    default='0,1,2')
args = parser.parse_args()


def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_devices

    sys.stdout = Logger(osp.join('./log', 'log_train' + str(len(os.listdir('./log'))) + '.txt'))

    # setting the seed
    torch.manual_seed(args.seed)
    cudnn.benchmark = True
    torch.cuda.manual_seed_all(args.seed)

    print("==========\nArgs:{}\n==========".format(args))
   
    print("Initializing dataset")
    dataset = Climate_dataset(path_indicators=args.path_indicators, 
                              path_PM25_O3=args.path_pm25_o3, 
                              num_train=args.train_portion, 
                              gap=args.gap,
                              target=args.targets)

    # loading the land mark
    landmark = load_landmark(path=args.land)
    elevation = load_elevation(path=args.mask)

    # mean and std for indicators
    mean_indicators = torch.Tensor([65.30, 6.60, 1.20, 26.96, 18.06, 29.06, 25.73, 26.23, 25.23])[:, None, None].cuda()
    std_indicators = torch.Tensor([28.47, 6.11, 3.01, 10.52, 9.47, 16.38, 7.39, 7.54, 7.23])[:, None, None].cuda()

    trainloader = DataLoader(
        Climate_dataloader(dataset.train_data, mean=mean_indicators, std=std_indicators),
        shuffle=True, batch_size=args.train_batch, num_workers=args.workers,
        pin_memory=args.pin_memory, drop_last=True,
    )

    valloader = DataLoader(
        Climate_dataloader(dataset.val_data, mean=mean_indicators, std=std_indicators),
        batch_size=args.test_batch, shuffle=False, num_workers=args.workers,
        pin_memory=args.pin_memory, drop_last=False,
    )

    print("Initializing model:")
    model = UNet()
    model = nn.DataParallel(model).cuda()

    criterion = nn.MSELoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=args.stepsize, gamma=args.gamma)

    best_rMSE = 1e2

    for epoch in range(args.max_epoch):
        print("==> Epoch {}/{}".format(epoch+1, args.max_epoch))
        
        # for model training
        train(args, model, criterion, optimizer, trainloader, landmark, elevation, mean=mean_indicators, std=std_indicators)
        scheduler.step()
        
        print("==> Test")
        rMSE = test(args, model, criterion, valloader, landmark, elevation, mean=mean_indicators, std=std_indicators)

        if rMSE < best_rMSE:
            print('last best model')
            best_rMSE = rMSE
            torch.save(model.state_dict(), './best.pth')
    print('Best model with {} rMSE'.format(best_rMSE))


def train(args, model, criterion, optimizer, trainloader, landmark, elevation, mean, std):
    model.train()
    losses = AverageMeter()

    for batch_idx, (indicators_today, indicators_target, pollutant_today, pollutant_target, embedding_today, embedding_target,  _) in enumerate(trainloader):
        
        ##########################
        # data preparation
        indicators_today, indicators_target = indicators_today.cuda(), indicators_target.cuda()
        embedding_today, embedding_target = embedding_today.cuda(), embedding_target.cuda()

        # for indicators, we apply the z-score normalization
        indicators_today = normalizaton_z_score(indicators_today, mean=mean, std=std)
        indicators_target = normalizaton_z_score(indicators_target, mean=mean, std=std)

        b = indicators_today.size(0)

        indicators_today = torch.cat((indicators_today, elevation.repeat(b, 1, 1, 1)), dim=1)
        indicators_target = torch.cat((indicators_target, elevation.repeat(b, 1, 1, 1)), dim=1)

        # for O3, PM25, we apply the max-min normalization
        pollutant_today = torch.clip(pollutant_today.cuda()/args.max, min=0, max=1)
        pollutant_target = pollutant_target.cuda()[:, :, landmark]

        factors = pollutant_today
        ############################

        outputs = model(indicators_today, indicators_target, factors, embedding_today, embedding_target)
        outputs = outputs[:, :, landmark] * args.max

        map_overall = ((pollutant_target <=args.max) * 1).detach()
        map_max = (((pollutant_target <=args.max) * 1) * ((pollutant_target > args.high) * 1)).detach()

        loss = criterion(outputs * map_overall , pollutant_target * map_overall) + 0.5 * criterion(outputs * map_max, pollutant_target * map_max)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.update(loss.item(),b)

        if (batch_idx+1) % args.print_step == 0:
            print("Batch {}/{}\t Loss {:.6f} ({:.6f})".format(batch_idx+1, len(trainloader), losses.val, losses.avg))


def test(args, model, criterion, valloader, landmark, elevation, mean, std):
    model.eval()
    samples_scaled, ground_truth = [], []
    top_counts, drop_nodes = 0, 0

    with torch.no_grad():
        for _, (indicators_today, indicators_target, pollutant_today, pollutant_target, embedding_today, embedding_target, _) in enumerate(valloader):
            
            indicators_today, indicators_target = indicators_today.cuda(), indicators_target.cuda()
            embedding_today, embedding_target = embedding_today.cuda(), embedding_target.cuda()

            # for indicators, we apply the z-score normalization
            indicators_today = normalizaton_z_score(indicators_today, mean=mean, std=std)
            indicators_target = normalizaton_z_score(indicators_target, mean=mean, std=std)
            b = indicators_today.size(0)

            indicators_today = torch.cat((indicators_today, elevation.repeat(b, 1, 1, 1)), dim=1)
            indicators_target = torch.cat((indicators_target, elevation.repeat(b, 1, 1, 1)), dim=1)

            # for O3, PM25, we apply the max-min normalization
            pollutant_today = pollutant_today.cuda()/args.max
            pollutant_target = pollutant_target.cuda()[:, :, landmark]

            factors = pollutant_today
            
            outputs = model(indicators_today, indicators_target, factors, embedding_today, embedding_target)
            outputs = outputs[:, :, landmark]
            outputs = torch.clip(outputs, max=1, min=0) * args.max

            map_overall = ((pollutant_target <=args.max) * 1).detach()
            map_max = (((pollutant_target <=args.max) * 1) * ((pollutant_target > args.high) * 1)).detach()

            drop_nodes += (1 - map_overall).sum()

            ground_truth.append(pollutant_target * map_overall)
            samples_scaled.append(outputs * map_overall)

            top_counts += map_max.sum()

    ground_truth = torch.cat(ground_truth, dim=0).squeeze()
    samples_scaled = torch.cat(samples_scaled, dim=0).squeeze()

    num_nodes = ground_truth.size(0) * ground_truth.size(1) - drop_nodes
    rMSE = torch.sqrt( ((samples_scaled - ground_truth)**2 ).sum() / num_nodes )
    print('rMSE of the scaled data {}'.format(rMSE))
    
    return rMSE


def load_landmark(path=None):
    landmark = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    landmark = landmark > 0
    return landmark

def load_elevation(path=None):
    elevation = cv2.imread(path, cv2.IMREAD_GRAYSCALE)/255

    elevation = torch.Tensor(elevation).cuda()[None, None, :, :] #.repeat(bs, 1, 1, 1)
    return elevation

def normalizaton_z_score(tensor, mean, std):
    return (tensor - mean) / std

def normalization_min_max(tensor, max, min):
    return (tensor - min) / (max - min)


if __name__ == '__main__':
    main()
