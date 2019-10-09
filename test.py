from __future__ import print_function, absolute_import
import os
import gc
import sys
import time
import math
import h5py
import scipy
import datetime
import argparse
import os.path as osp
import numpy as np

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

import utils.data_manager as data_manager
from utils.video_loader import VideoDataset, ImageDataset
import transforms.spatial_transforms as ST
import transforms.temporal_transforms as TT
import torchvision.transforms as T
import models

from utils.utils import AverageMeter, Logger, save_checkpoint
from utils.eval_metrics import evaluate

parser = argparse.ArgumentParser(description='Testing using all frames')
# Datasets
parser.add_argument('--root', type=str, default='/data/datasets/')
parser.add_argument('-d', '--dataset', type=str, default='mars',
                    choices=data_manager.get_names())
parser.add_argument('-j', '--workers', default=4, type=int)
parser.add_argument('--height', type=int, default=256)
parser.add_argument('--width', type=int, default=128)
# Augment
parser.add_argument('--test_frames', default=32, type=int, help='frames/clip for test')
# Optimization options
parser.add_argument('--test_batch', default=1, type=int, help="has to be 1")
parser.add_argument('--img_test_batch', default=128, type=int)
# Architecture
parser.add_argument('--vid_arch', type=str, default='vid_nonlocalresnet50')
parser.add_argument('--img_arch', type=str, default='img_resnet50')
# Miscs
parser.add_argument('--resume', type=str, default='log/best_model.pth.tar', metavar='PATH')
parser.add_argument('--save_dir', type=str, default='log')
parser.add_argument('--gpu_devices', default='0', type=str)

args = parser.parse_args()

def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_devices
    print(torch.cuda.device_count())
    use_gpu = torch.cuda.is_available()

    sys.stdout = Logger(osp.join(args.save_dir, 'log_test.txt'))
    print("==========\nArgs:{}\n==========".format(args))

    print("Initializing dataset {}".format(args.dataset))
    dataset = data_manager.init_dataset(name=args.dataset, root=args.root)

    # Data augmentation
    spatial_transform_test = ST.Compose([
                ST.Scale((args.height, args.width), interpolation=3),
                ST.ToTensor(),
                ST.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
    temporal_transform_test = None

    transform_test_img = T.Compose([
        T.Resize((args.height, args.width), interpolation=3),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    pin_memory = True if use_gpu else False

    queryloader = DataLoader(
        VideoDataset(dataset.query, spatial_transform=spatial_transform_test, temporal_transform=temporal_transform_test),
        batch_size=args.test_batch, shuffle=False, num_workers=0,
        pin_memory=pin_memory, drop_last=False
    )

    galleryloader = DataLoader(
        VideoDataset(dataset.gallery, spatial_transform=spatial_transform_test, temporal_transform=temporal_transform_test),
        batch_size=args.test_batch, shuffle=False, num_workers=0,
        pin_memory=pin_memory, drop_last=False
    )

    queryimgloader = DataLoader(
        ImageDataset(dataset.query_img, transform=transform_test_img),
        batch_size=args.img_test_batch, shuffle=False, num_workers=args.workers,
        pin_memory=pin_memory, drop_last=False
    )

    galleryimgloader = DataLoader(
        ImageDataset(dataset.gallery_img, transform=transform_test_img),
        batch_size=args.img_test_batch, shuffle=False, num_workers=args.workers,
        pin_memory=pin_memory, drop_last=False
    )

    print("Initializing model: {} and {}".format(args.vid_arch, args.img_arch))
    vid_model = models.init_model(name=args.vid_arch)
    img_model = models.init_model(name=args.img_arch)
    print("Video model size: {:.5f}M".format(sum(p.numel() for p in vid_model.parameters())/1000000.0))
    print("Image model size: {:.5f}M".format(sum(p.numel() for p in img_model.parameters())/1000000.0))


    print("Loading checkpoint from '{}'".format(args.resume))
    checkpoint = torch.load(args.resume)
    vid_model.load_state_dict(checkpoint['vid_model_state_dict'])
    img_model.load_state_dict(checkpoint['img_model_state_dict'])

    if use_gpu:
        vid_model = vid_model.cuda()
        img_model = img_model.cuda()

    print("Evaluate")
    with torch.no_grad():
        test(vid_model, img_model, queryloader, galleryloader, queryimgloader, galleryimgloader, use_gpu)

def extract_vid_feature(model, vids, use_gpu):
    n, c, f, h, w = vids.size()
    assert(n == 1)

    feat = torch.FloatTensor()
    for i in range(math.ceil(f/args.test_frames)):
        clip = vids[:, :, i*args.test_frames:(i+1)*args.test_frames, :, :]
        if use_gpu:
            clip = clip.cuda()
        output = model(clip)
        output = output.data.cpu()
        feat = torch.cat((feat, output), 1)

    feat = feat.mean(1)

    return feat


def test(vid_model, img_model, queryloader, galleryloader, queryimgloader, galleryimgloader, use_gpu):
    since = time.time()
    vid_model.eval()
    img_model.eval()

    print("Extract video features")
    vid_qf, vid_q_pids, vid_q_camids = [], [], []
    for batch_idx, (vids, pids, camids) in enumerate(queryloader):
        if (batch_idx+1)%1000==0 or (batch_idx+1)%len(queryloader)==0:
            print("{}/{}".format(batch_idx+1, len(queryloader)))

        vid_qf.append(extract_vid_feature(vid_model, vids, use_gpu).squeeze())
        vid_q_pids.extend(pids)
        vid_q_camids.extend(camids)
    vid_qf = torch.stack(vid_qf)
    vid_q_pids = np.asarray(vid_q_pids)
    vid_q_camids = np.asarray(vid_q_camids)
    print("Extracted features for query set, obtained {} matrix".format(vid_qf.shape))

    vid_gf, vid_g_pids, vid_g_camids = [], [], []
    for batch_idx, (vids, pids, camids) in enumerate(galleryloader):
        if (batch_idx + 1) % 1000==0 or (batch_idx+1)%len(galleryloader)==0:
            print("{}/{}".format(batch_idx+1, len(galleryloader)))

        vid_gf.append(extract_vid_feature(vid_model, vids, use_gpu).squeeze())
        vid_g_pids.extend(pids)
        vid_g_camids.extend(camids)
    vid_gf = torch.stack(vid_gf)
    vid_g_pids = np.asarray(vid_g_pids)
    vid_g_camids = np.asarray(vid_g_camids)

    if args.dataset == 'mars':
        # gallery set must contain query set, otherwise 140 query imgs will not have ground truth.
        vid_gf = torch.cat((vid_qf, vid_gf), 0)
        vid_g_pids = np.append(vid_q_pids, vid_g_pids)
        vid_g_camids = np.append(vid_q_camids, vid_g_camids)
    print("Extracted features for gallery set, obtained {} matrix".format(vid_gf.shape))

    print("Extract image features")
    img_qf, img_q_pids, img_q_camids = [], [], []
    for batch_idx, (imgs, pids, camids) in enumerate(queryimgloader):
        if use_gpu:
            imgs = imgs.cuda()
        feat = img_model(imgs).data.cpu()
        img_qf.append(feat)
        img_q_pids.extend(pids)
        img_q_camids.extend(camids)
    img_qf = torch.cat(img_qf, 0)
    img_q_pids = np.asarray(img_q_pids)
    img_q_camids = np.asarray(img_q_camids)
    print("Extracted features for query set, obtained {} matrix".format(img_qf.shape))

    img_gf, img_g_pids, img_g_camids = [], [], []
    for batch_idx, (imgs, pids, camids) in enumerate(galleryimgloader):
        if use_gpu:
            imgs = imgs.cuda()
        feat = img_model(imgs).data.cpu()
        img_gf.append(feat)
        img_g_pids.extend(pids)
        img_g_camids.extend(camids)
    img_gf = torch.cat(img_gf, 0)
    img_g_pids = np.asarray(img_g_pids)
    img_g_camids = np.asarray(img_g_camids)

    if args.dataset == 'mars':
        # gallery set must contain query set, otherwise 140 query imgs will not have ground truth.
        img_gf = torch.cat((img_qf, img_gf), 0)
        img_g_pids = np.append(img_q_pids, img_g_pids)
        img_g_camids = np.append(img_q_camids, img_g_camids)
    print("Extracted features for gallery set, obtained {} matrix".format(img_gf.shape))

    time_elapsed = time.time() - since
    print('Extracting features complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print("Computing distance matrix")        
    m, n = vid_qf.size(0), vid_gf.size(0)
    img_distmat = torch.zeros((m,n))
    vid_distmat = torch.zeros((m,n))
    i2v_distmat = torch.zeros((m,n))
    v2i_distmat = torch.zeros((m,n))

    img_q_norm = torch.norm(img_qf, p=2, dim=1, keepdim=True)
    img_g_norm = torch.norm(img_gf, p=2, dim=1, keepdim=True)
    vid_q_norm = torch.norm(vid_qf, p=2, dim=1, keepdim=True)
    vid_g_norm = torch.norm(vid_gf, p=2, dim=1, keepdim=True)
    img_qf = img_qf.div(img_q_norm.expand_as(img_qf))
    img_gf = img_gf.div(img_g_norm.expand_as(img_gf))
    vid_qf = vid_qf.div(vid_q_norm.expand_as(vid_qf))
    vid_gf = vid_gf.div(vid_g_norm.expand_as(vid_gf))
    
    for i in range(m):
        img_distmat[i] = - torch.mm(img_qf[i:i+1], img_gf.t())
        vid_distmat[i] = - torch.mm(vid_qf[i:i+1], vid_gf.t())
        i2v_distmat[i] = - torch.mm(img_qf[i:i+1], vid_gf.t())
        v2i_distmat[i] = - torch.mm(vid_qf[i:i+1], img_gf.t())

    img_distmat = img_distmat.numpy()
    vid_distmat = vid_distmat.numpy()
    i2v_distmat = i2v_distmat.numpy()
    v2i_distmat = v2i_distmat.numpy()

    print('image to image')
    cmc, mAP = evaluate(img_distmat, img_q_pids, img_g_pids, img_q_camids, img_g_camids)
    print('top1:{:.2%} top5:{:.2%} top10:{:.2%} mAP:{:.2%}'.format(cmc[0],cmc[4],cmc[9],mAP))

    print('video to video')
    cmc, mAP = evaluate(vid_distmat, vid_q_pids, vid_g_pids, vid_q_camids, vid_g_camids)
    print('top1:{:.2%} top5:{:.2%} top10:{:.2%} mAP:{:.2%}'.format(cmc[0],cmc[4],cmc[9],mAP))

    print('video to image')
    cmc, mAP = evaluate(v2i_distmat, vid_q_pids, img_g_pids, vid_q_camids, img_g_camids)
    print('top1:{:.2%} top5:{:.2%} top10:{:.2%} mAP:{:.2%}'.format(cmc[0],cmc[4],cmc[9],mAP))

    print('image to video')
    cmc, mAP = evaluate(i2v_distmat, img_q_pids, vid_g_pids, img_q_camids, vid_g_camids)
    print('top1:{:.2%} top5:{:.2%} top10:{:.2%} mAP:{:.2%}'.format(cmc[0],cmc[4],cmc[9],mAP))

    return cmc[0]

if __name__ == '__main__':
    main()
