from __future__ import absolute_import

import torch
from torch import nn
from torch.autograd import Variable


__all__ = ['FeatureBasedTKP', 'SimilarityBasedTKP', 'HeterogeneousTripletLoss']


class HeterogeneousTripletLoss(nn.Module):
    """Triplet loss with hard positive/negative mining.

    Reference:
    Hermans et al. In Defense of the Triplet Loss for Person Re-Identification. arXiv:1703.07737.

    Code imported from https://github.com/Cysu/open-reid/blob/master/reid/loss/triplet.py.

    Args:
        margin (float): margin for triplet.
    """
    def __init__(self, margin=0.3, distance='euclidean', use_gpu=True):
        super(HeterogeneousTripletLoss, self).__init__()
        if distance not in ['euclidean', 'consine']:
            raise KeyError("Unsupported distance: {}".format(distance))
        self.distance = distance
        self.margin = margin
        self.use_gpu = use_gpu
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)

    def forward(self, inputs1, inputs2, targets1, targets2):
        """
        Args:
            inputs: feature matrix with shape (batch_size, feat_dim)
            targets: ground truth labels with shape (num_classes)
        """
        m = inputs1.size(0)
        n = inputs2.size(0)

        # Compute pairwise distance, replace by the official when merged
        if self.distance == 'euclidean':
            dist1 = torch.pow(inputs1, 2).sum(dim=1, keepdim=True).expand(m, n)
            dist2 = torch.pow(inputs2, 2).sum(dim=1, keepdim=True).expand(n, m)
            dist = dist1 + dist2.t()
            dist.addmm_(1, -2, inputs1, inputs2.t())
            dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
        elif self.distance == 'consine':
            fnorm1 = torch.norm(inputs1, p=2, dim=1, keepdim=True)
            fnorm2 = torch.norm(inputs2, p=2, dim=1, keepdim=True)
            l2norm1 = inputs1.div(fnorm1.expand_as(inputs1))
            l2norm2 = inputs2.div(fnorm2.expand_as(inputs2))
            dist = - torch.mm(l2norm1, l2norm2.t())

        if self.use_gpu: 
            targets1 = targets1.cuda()
            targets2 = targets2.cuda()
        # For each anchor, find the hardest positive and negative
        mask = targets1.expand(n, m).t().eq(targets2.expand(m, n))
        dist_ap, dist_an = [], []
        for i in range(m):
            dist_ap.append(dist[i][mask[i]].max().unsqueeze(0))
            dist_an.append(dist[i][mask[i] == 0].min().unsqueeze(0))
        dist_ap = torch.cat(dist_ap)
        dist_an = torch.cat(dist_an)

        # Compute ranking hinge loss
        y = torch.ones_like(dist_an)
        loss = self.ranking_loss(dist_an, dist_ap, y)
        return loss


class FeatureBasedTKP(nn.Module):
    def __init__(self, bp_to_vid=False):
        super(FeatureBasedTKP, self).__init__()
        self.bp_to_vid = bp_to_vid

    def forward(self, images, videos):
        n, c = images.size()
        assert(images.size() == videos.size())

        # Do not BP to videos features
        if not self.bp_to_vid:
            videos = videos.detach()

        dist = torch.pow(images-videos, 2).sum(dim=1, keepdim=False)
        loss = dist.mean()

        return loss


class SimilarityBasedTKP(nn.Module):
    def __init__(self, distance='euclidean', bp_to_vid=False):
        super(SimilarityBasedTKP, self).__init__()
        self.distance = distance
        self.bp_to_vid = bp_to_vid

    def forward(self, images, videos):
        n, c = images.size()
        assert(images.size() == videos.size())

        # Do not BP to videos features
        if not self.bp_to_vid:
            videos = videos.detach()

        if self.distance == 'euclidean':
            img_distmat = torch.pow(images, 2).sum(dim=1, keepdim=True).expand(n, n)
            img_distmat = img_distmat + img_distmat.t()
            img_distmat.addmm_(1, -2, images, images.t())
            img_distmat = img_distmat.clamp(min=1e-12).sqrt()  # for numerical stability
            vid_distmat = torch.pow(videos, 2).sum(dim=1, keepdim=True).expand(n, n)
            vid_distmat = vid_distmat + vid_distmat.t()
            vid_distmat.addmm_(1, -2, videos, videos.t())
            vid_distmat = vid_distmat.clamp(min=1e-12).sqrt()  # for numerical stability
        elif self.distance == 'consine':
            img_norm = torch.norm(images, p=2, dim=1, keepdim=True)
            vid_norm = torch.norm(videos, p=2, dim=1, keepdim=True)
            images = images.div(img_norm.expand_as(images))
            videos = videos.div(vid_norm.expand_as(videos))
            img_distmat = torch.mm(images, images.t())
            vid_distmat = torch.mm(videos, videos.t())

        loss = torch.pow(img_distmat-vid_distmat, 2).mean()

        return loss