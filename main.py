import random
from random import sample
import argparse
from typing import List
import itertools
import numpy as np
import os
import cv2
import pickle
import torch
from skimage import morphology
from collections import OrderedDict
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import confusion_matrix
from sklearn.covariance import LedoitWolf
from skimage.segmentation import mark_boundaries
import matplotlib.pyplot as plt
import matplotlib
from torch.utils.data import DataLoader
from module.model import GaussianCnnPredictor
from module.tools import get_bbxes, denormalization
import datasets.mvtec as mvtec

def parse_args():
    parser = argparse.ArgumentParser('PaDiM')
    parser.add_argument('--data_path', type=str, default='datasets/mvtec_anomaly_detection')
    parser.add_argument('--save_path', type=str, default='./mvtec_result')
    parser.add_argument('--arch', type=str, choices=['resnet18', 'wide_resnet50_2'], default='resnet18')
    return parser.parse_args()
    
def main():

    print(f'cuda is available: {torch.cuda.is_available()}')

    args = parse_args()

    os.makedirs(os.path.join(args.save_path, 'temp_%s' % args.arch), exist_ok=True)

    class_name = 'bottle'
    data_path = args.data_path
    save_path = args.save_path
    arch = args.arch

    train_dataset = mvtec.MVTecDataset(data_path, class_name=class_name, is_train=True, ext = '.png')
    train_dataloader = DataLoader(train_dataset, batch_size=32, pin_memory=True)
    test_dataset = mvtec.MVTecDataset(data_path, class_name=class_name, is_train=False, ext = '.png')
    test_dataloader = DataLoader(test_dataset, batch_size=32, pin_memory=True)

    model = GaussianCnnPredictor(arch = args.arch)
    model.fit(train_dataloader)
    heatmaps = model.predict(test_dataloader)

    binaries, bbxes, judges = get_bbxes(heatmaps, 50, 0)

    test_imgs, labels = [], []
    for (x, y) in test_dataloader:
        test_imgs.extend(x.cpu().detach().numpy())
        labels.append(y.cpu().detach().numpy().tolist())
    labels = list(itertools.chain.from_iterable(labels))

    print(confusion_matrix(labels, judges))

    plot_fig(test_imgs, heatmaps, num = 50)


#     # calculate image-level ROC AUC score
#     img_scores = scores.reshape(scores.shape[0], -1).max(axis=1)
#     gt_list = np.asarray(gt_list)
#     fpr, tpr, _ = roc_curve(gt_list, img_scores)
#     img_roc_auc = roc_auc_score(gt_list, img_scores)
#     print('image ROCAUC: %.3f' % (img_roc_auc))
#     fig_img_rocauc.plot(fpr, tpr, label='%s img_ROCAUC: %.3f' % (class_name, img_roc_auc))
    
#     # get optimal threshold
#     gt_mask = np.asarray(gt_mask_list)
#     precision, recall, thresholds = precision_recall_curve(gt_mask.flatten(), scores.flatten())
#     a = 2 * precision * recall
#     b = precision + recall
#     f1 = np.divide(a, b, out=np.zeros_like(a), where=b != 0)
#     threshold = thresholds[np.argmax(f1)]

#     # calculate per-pixel level ROCAUC
#     fpr, tpr, _ = roc_curve(gt_mask.flatten(), scores.flatten())
#     per_pixel_rocauc = roc_auc_score(gt_mask.flatten(), scores.flatten())
#     total_pixel_roc_auc.append(per_pixel_rocauc)
#     print('pixel ROCAUC: %.3f' % (per_pixel_rocauc))

#     fig_pixel_rocauc.plot(fpr, tpr, label='%s ROCAUC: %.3f' % (class_name, per_pixel_rocauc))
#     save_dir = save_path + '/' + f'pictures_{arch}'
#     os.makedirs(save_dir, exist_ok=True)
#     plot_fig(test_imgs, scores, gt_mask_list, threshold, save_dir, class_name)

def plot_fig(test_imgs: List, scores: np.array, num: int):
    fig, ax = plt.subplots(1, 1, figsize=(12, 3))
    fig.subplots_adjust(right=0.9)
    img = test_imgs[num]
    img = denormalization(img)
    ax.imshow(img, cmap='gray', interpolation='none')
    fig.savefig('original.jpg', dpi=100)
    ax.imshow(scores[num], cmap='jet', alpha=0.5, interpolation='none')
    fig.savefig('with_heatmap.jpg', dpi=100)

# def plot_fig(test_img, scores, gts, threshold, save_dir, class_name):
#     num = len(scores)
#     vmax = scores.max() * 255.
#     vmin = scores.min() * 255.
#     for i in range(num):
#         img = test_img[i]
#         img = denormalization(img)
#         gt = gts[i].transpose(1, 2, 0).squeeze()
#         heat_map = scores[i] * 255
#         mask = scores[i]
#         mask[mask > threshold] = 1
#         mask[mask <= threshold] = 0
#         kernel = morphology.disk(4)
#         mask = morphology.opening(mask, kernel)
#         mask *= 255
#         vis_img = mark_boundaries(img, mask, color=(1, 0, 0), mode='thick')
#         fig_img, ax_img = plt.subplots(1, 5, figsize=(12, 3))
#         fig_img.subplots_adjust(right=0.9)
#         norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
#         for ax_i in ax_img:
#             ax_i.axes.xaxis.set_visible(False)
#             ax_i.axes.yaxis.set_visible(False)
#         ax_img[0].imshow(img)
#         ax_img[0].title.set_text('Image')
#         ax_img[1].imshow(gt, cmap='gray')
#         ax_img[1].title.set_text('GroundTruth')
#         ax = ax_img[2].imshow(heat_map, cmap='jet', norm=norm)
#         ax_img[2].imshow(img, cmap='gray', interpolation='none')
#         ax_img[2].imshow(heat_map, cmap='jet', alpha=0.5, interpolation='none')
#         ax_img[2].title.set_text('Predicted heat map')
#         ax_img[3].imshow(mask, cmap='gray')
#         ax_img[3].title.set_text('Predicted mask')
#         ax_img[4].imshow(vis_img)
#         ax_img[4].title.set_text('Segmentation result')
#         left = 0.92
#         bottom = 0.15
#         width = 0.015
#         height = 1 - 2 * bottom
#         rect = [left, bottom, width, height]
#         cbar_ax = fig_img.add_axes(rect)
#         cb = plt.colorbar(ax, shrink=0.6, cax=cbar_ax, fraction=0.046)
#         cb.ax.tick_params(labelsize=8)
#         font = {
#             'family': 'serif',
#             'color': 'black',
#             'weight': 'normal',
#             'size': 8,
#         }
#         cb.set_label('Anomaly Score', fontdict=font)

#         fig_img.savefig(os.path.join(save_dir, class_name + '_{}'.format(i)), dpi=100)
#         plt.close()


if __name__ == '__main__':
    main()