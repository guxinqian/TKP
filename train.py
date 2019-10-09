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
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch.optim import lr_scheduler

import utils.data_manager as data_manager
from utils.video_loader import VideoDataset, ImageDataset
import transforms.spatial_transforms as ST
import transforms.temporal_transforms as TT
import torchvision.transforms as T
import models
from utils.losses import FeatureBasedTKP, SimilarityBasedTKP, HeterogeneousTripletLoss
from utils.utils import AverageMeter, Logger, save_checkpoint
from utils.eval_metrics import evaluate
from utils.samplers import RandomIdentitySampler

parser = argparse.ArgumentParser(description='Train img to video model')
# Datasets
parser.add_argument('--root', type=str, default='/data/datasets/')
parser.add_argument('-d', '--dataset', type=str, default='mars',
                    choices=data_manager.get_names())
parser.add_argument('-j', '--workers', default=4, type=int)
parser.add_argument('--height', type=int, default=256)
parser.add_argument('--width', type=int, default=128)
# Augment
parser.add_argument('--seq_len', type=int, default=4, help="the length of video clips")
parser.add_argument('--sample_stride', type=int, default=8, help="sampling stride of video clips")
# Optimization options
parser.add_argument('--max_epoch', default=150, type=int)
parser.add_argument('--start_epoch', default=0, type=int)
parser.add_argument('--train_batch', default=16, type=int)
parser.add_argument('--test_batch', default=128, type=int)
parser.add_argument('--img_test_batch', default=512, type=int)
parser.add_argument('--lr', '--learning-rate', default=0.0003, type=float)
parser.add_argument('--stepsize', default=[60, 120], nargs='+', type=int)
parser.add_argument('--gamma', default=0.1, type=float, help="learning rate decay")
parser.add_argument('--weight_decay', default=5e-04, type=float)
parser.add_argument('--num_instances', type=int, default=4, help="number of instances per identity")
# Architecture
parser.add_argument('--vid_arch', type=str, default='vid_nonlocalresnet50')
parser.add_argument('--img_arch', type=str, default='img_resnet50')
# Loss
parser.add_argument('--bp_to_vid', action='store_true', help="weather the TKP loss BP to vid model")
# Miscs
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--resume', type=str, default='', metavar='PATH')
parser.add_argument('--evaluate', action='store_true', help="evaluation only")
parser.add_argument('--eval_step', type=int, default=10)
parser.add_argument('--start_eval', type=int, default=0, help="start to evaluate after specific epoch")
parser.add_argument('--save_dir', type=str, default='log')
parser.add_argument('--gpu_devices', default='0', type=str)

args = parser.parse_args()

def main():
    torch.manual_seed(args.seed)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_devices
    use_gpu = torch.cuda.is_available()

    if not args.evaluate:
        sys.stdout = Logger(osp.join(args.save_dir, 'log_train.txt'))
    else:
        sys.stdout = Logger(osp.join(args.save_dir, 'log_test.txt'))
    print("==========\nArgs:{}\n==========".format(args))

    if use_gpu:
        print("Currently using GPU {}".format(args.gpu_devices))
        torch.cuda.manual_seed_all(args.seed)
    else:
        print("Currently using CPU (GPU is highly recommended)")

    print("Initializing dataset {}".format(args.dataset))
    dataset = data_manager.init_dataset(name=args.dataset, root=args.root)

    # Data augmentation
    spatial_transform_train = ST.Compose([
                ST.Scale((args.height, args.width), interpolation=3),
                ST.RandomHorizontalFlip(),
                ST.ToTensor(),
                ST.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
    temporal_transform_train = TT.TemporalRandomCrop(size=args.seq_len, stride=args.sample_stride)

    spatial_transform_test = ST.Compose([
                ST.Scale((args.height, args.width), interpolation=3),
                ST.ToTensor(),
                ST.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
    temporal_transform_test = TT.TemporalBeginCrop()

    transform_test_img = T.Compose([
        T.Resize((args.height, args.width), interpolation=3),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    pin_memory = True if use_gpu else False

    if args.dataset == 'dukevid':
        trainloader = DataLoader(
        VideoDataset(dataset.train_dense, spatial_transform=spatial_transform_train, temporal_transform=temporal_transform_train),
        sampler=RandomIdentitySampler(dataset.train_dense, num_instances=args.num_instances),
        batch_size=args.train_batch, num_workers=args.workers,
        pin_memory=pin_memory, drop_last=True,
        )
    else:
        trainloader = DataLoader(
            VideoDataset(dataset.train, spatial_transform=spatial_transform_train, temporal_transform=temporal_transform_train),
            sampler=RandomIdentitySampler(dataset.train, num_instances=args.num_instances),
            batch_size=args.train_batch, num_workers=args.workers,
            pin_memory=pin_memory, drop_last=True,
        )

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
    classifier = models.init_model(name='classifier', num_classes=dataset.num_train_pids)
    print("Video model size: {:.5f}M".format(sum(p.numel() for p in vid_model.parameters())/1000000.0))
    print("Image model size: {:.5f}M".format(sum(p.numel() for p in img_model.parameters())/1000000.0))

    criterion = nn.CrossEntropyLoss()
    criterion_tkp_f = FeatureBasedTKP(bp_to_vid=args.bp_to_vid)
    criterion_tkp_d = SimilarityBasedTKP(distance='euclidean', bp_to_vid=args.bp_to_vid)
    criterion_i2v = HeterogeneousTripletLoss(margin=0.3, distance='euclidean')
    optimizer = torch.optim.Adam([
             {'params': vid_model.parameters(), 'lr': args.lr},
             {'params': img_model.parameters(), 'lr': args.lr},
             {'params': classifier.parameters(), 'lr': args.lr}
             ], weight_decay=args.weight_decay)
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=args.stepsize, gamma=args.gamma)
    start_epoch = args.start_epoch

    if args.resume:
        print("Loading checkpoint from '{}'".format(args.resume))
        checkpoint = torch.load(args.resume)
        vid_model.load_state_dict(checkpoint['vid_model_state_dict'])
        img_model.load_state_dict(checkpoint['img_model_state_dict'])
        classifier.load_state_dict(checkpoint['classifier_state_dict'])
        start_epoch = checkpoint['epoch']

    if use_gpu:
        vid_model = vid_model.cuda()
        img_model = img_model.cuda()
        classifier = classifier.cuda()

    if args.evaluate:
        print("Evaluate only")
        with torch.no_grad():
            test(vid_model, img_model, queryloader, galleryloader, queryimgloader, galleryimgloader, use_gpu)
        return

    start_time = time.time()
    train_time = 0
    best_rank1 = -np.inf
    best_epoch = 0
    print("==> Start training")

    for epoch in range(start_epoch, args.max_epoch):
        scheduler.step()

        start_train_time = time.time()
        train(epoch, vid_model, img_model, classifier, criterion, criterion_tkp_f, criterion_tkp_d, criterion_i2v, optimizer, trainloader, use_gpu)
        torch.cuda.empty_cache()
        train_time += round(time.time() - start_train_time)

        
        if (epoch+1) >= args.start_eval and args.eval_step > 0 and (epoch+1) % args.eval_step == 0 or (epoch+1) == args.max_epoch:
            print("==> Test")
            with torch.no_grad():
                rank1 = test(vid_model, img_model, queryloader, galleryloader, queryimgloader, galleryimgloader, use_gpu)
                torch.cuda.empty_cache()
            is_best = rank1 > best_rank1
            if is_best: 
                best_rank1 = rank1
                best_epoch = epoch + 1

            vid_model_state_dict = vid_model.state_dict()
            img_model_state_dict = img_model.state_dict()
            classifier_state_dict = classifier.state_dict()

            save_checkpoint({
                'vid_model_state_dict': vid_model_state_dict,
                'img_model_state_dict': img_model_state_dict,
                'classifier_state_dict': classifier_state_dict,
                'rank1': rank1,
                'epoch': epoch,
            }, is_best, osp.join(args.save_dir, 'checkpoint_ep' + str(epoch+1) + '.pth.tar'))

    print("==> Best Rank-1 {:.1%}, achieved at epoch {}".format(best_rank1, best_epoch))

    elapsed = round(time.time() - start_time)
    elapsed = str(datetime.timedelta(seconds=elapsed))
    train_time = str(datetime.timedelta(seconds=train_time))
    print("Finished. Total elapsed time (h:m:s): {}. Training time (h:m:s): {}.".format(elapsed, train_time))

def train(epoch, vid_model, img_model, classifier, criterion, criterion_tkp_f, criterion_tkp_d, criterion_i2v, optimizer, trainloader, use_gpu):
    batch_vid_loss = AverageMeter()
    batch_img_loss = AverageMeter()
    batch_TKP_F_loss = AverageMeter()
    batch_TKP_D_loss = AverageMeter()
    batch_i2v_loss = AverageMeter()
    batch_v2i_loss = AverageMeter()
    batch_i2i_loss = AverageMeter()
    batch_v2v_loss = AverageMeter()
    batch_vid_corrects = AverageMeter()
    batch_img_corrects = AverageMeter()
    batch_time = AverageMeter()
    data_time = AverageMeter()

    vid_model.train()
    img_model.train()
    classifier.train()

    end = time.time()
    for batch_idx, (vids, pids, _) in enumerate(trainloader):
        if (pids-pids[0]).sum() == 0:
            continue
            
        b, c, t, h, w = vids.size()
        img_pids = pids.unsqueeze(1).repeat(1,t).view(-1)

        if use_gpu:
            vids, pids, img_pids = vids.cuda(), pids.cuda(), img_pids.cuda()

        # measure data loading time
        data_time.update(time.time() - end)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward
        vid_features, frame_features = vid_model(vids)
        vid_outputs = classifier(vid_features)

        imgs = vids.permute(0, 2, 1, 3, 4).contiguous().view(b*t, c, h, w)
        img_features = img_model(imgs) 
        img_outputs = classifier(img_features)

        # compute loss
        vid_loss = criterion(vid_outputs, pids)
        img_loss = criterion(img_outputs, img_pids)
        TKP_F_loss = criterion_tkp_f(img_features, frame_features)
        TKP_D_loss = criterion_tkp_d(img_features, frame_features)
        i2v_loss = criterion_i2v(img_features, vid_features, img_pids, pids)
        v2i_loss = criterion_i2v(vid_features, img_features, pids, img_pids)
        i2i_loss = criterion_i2v(img_features, img_features, img_pids, img_pids)
        v2v_loss = criterion_i2v(vid_features, vid_features, pids, pids)
        loss = vid_loss + img_loss + i2v_loss + i2i_loss + v2v_loss + v2i_loss + TKP_F_loss + TKP_D_loss

        # backward + optimize
        loss.backward()
        optimizer.step()

        # statistics
        _, vid_preds = torch.max(vid_outputs.data, 1)
        batch_vid_corrects.update(torch.sum(vid_preds == pids.data).float()/b, b)

        _, img_preds = torch.max(img_outputs.data, 1)
        batch_img_corrects.update(torch.sum(img_preds == img_pids.data).float()/(b*t), b*t)

        batch_vid_loss.update(vid_loss.item(), b)
        batch_img_loss.update(img_loss.item(), b*t)
        batch_TKP_F_loss.update(TKP_F_loss.item(), b*t)
        batch_TKP_D_loss.update(TKP_D_loss.item(), b*t)
        batch_i2v_loss.update(i2v_loss.item(), b*t)
        batch_i2i_loss.update(i2i_loss.item(), b*t)
        batch_v2v_loss.update(v2v_loss.item(), b)
        batch_v2i_loss.update(v2i_loss.item(), b)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

    print('Epoch{0} '
              'Time:{batch_time.sum:.1f}s '
              'Data:{data_time.sum:.1f}s '
              'vXent:{vid_xent.avg:.4f} '
              'iXent:{img_xent.avg:.4f} '
              'TKP_F:{TKP_F.avg:.4f} '
              'TKP_D:{TKP_D.avg:.4f} '
              'i2v:{i2v.avg:.4f} '
              'v2i:{v2i.avg:.4f} '
              'i2i:{i2i.avg:.4f} '
              'v2v:{v2v.avg:.4f} '
              'vAcc:{vid_acc.avg:.2%} '
              'iAcc:{img_acc.avg:.2%} '.format(
               epoch+1, batch_time=batch_time, data_time=data_time,
               vid_xent=batch_vid_loss, 
               img_xent=batch_img_loss, 
               TKP_F = batch_TKP_F_loss, TKP_D = batch_TKP_D_loss,
               i2v = batch_i2v_loss, v2i = batch_v2i_loss, i2i = batch_i2i_loss, v2v = batch_v2v_loss,
               vid_acc=batch_vid_corrects, img_acc=batch_img_corrects))


def test(vid_model, img_model, queryloader, galleryloader, queryimgloader, galleryimgloader, use_gpu):
    since = time.time()
    vid_model.eval()
    img_model.eval()

    print("Extract video features")
    vid_qf, vid_q_pids, vid_q_camids = [], [], []
    for batch_idx, (vids, pids, camids) in enumerate(queryloader):
        # if (batch_idx+1)%1000==0 or (batch_idx+1)%len(queryloader)==0:
        #     print("{}/{}".format(batch_idx+1, len(queryloader)))
        if use_gpu:
            vids = vids.cuda()
        feat = vid_model(vids)
        feat = feat.mean(1)
        feat = feat.data.cpu()

        vid_qf.append(feat)
        vid_q_pids.extend(pids)
        vid_q_camids.extend(camids)
    vid_qf = torch.cat(vid_qf, 0)
    vid_q_pids = np.asarray(vid_q_pids)
    vid_q_camids = np.asarray(vid_q_camids)
    print("Extracted features for query set, obtained {} matrix".format(vid_qf.shape))

    vid_gf, vid_g_pids, vid_g_camids = [], [], []
    for batch_idx, (vids, pids, camids) in enumerate(galleryloader):
        # if (batch_idx + 1) % 1000==0 or (batch_idx+1)%len(galleryloader)==0:
        #     print("{}/{}".format(batch_idx+1, len(galleryloader)))
        if use_gpu:
            vids = vids.cuda()
        feat = vid_model(vids)
        feat = feat.mean(1)
        feat = feat.data.cpu()

        vid_gf.append(feat)
        vid_g_pids.extend(pids)
        vid_g_camids.extend(camids)
    vid_gf = torch.cat(vid_gf, 0)
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
