import random
from random import sample
import argparse
import numpy as np
import os
import cv2
import pickle
import torch
from skimage import morphology
from tqdm import tqdm
from collections import OrderedDict
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import precision_recall_curve
from sklearn.covariance import LedoitWolf
from scipy.spatial.distance import mahalanobis
from scipy.ndimage import gaussian_filter
from skimage.segmentation import mark_boundaries
import matplotlib.pyplot as plt
import matplotlib

import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.models import wide_resnet50_2, resnet18
import datasets.mvtec as mvtec





def parse_args():
    parser = argparse.ArgumentParser('PaDiM')
    parser.add_argument('--data_path', type=str, default='datasets/mvtec_anomaly_detection')
    parser.add_argument('--save_path', type=str, default='./mvtec_result')
    parser.add_argument('--arch', type=str, choices=['resnet18', 'wide_resnet50_2'], default='resnet18')
    return parser.parse_args()

class GaussianCnnPredictor():
    def __init__(self, arch: str):
        
        # device setup
        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device('cuda' if self.use_cuda else 'cpu')
        
        # load model
        if arch == 'resnet18':
            self.model = resnet18(pretrained=True, progress=True)
            t_d = 448
            d = 100
        elif arch == 'wide_resnet50_2':
            self.model = wide_resnet50_2(pretrained=True, progress=True)
            t_d = 1792
            d = 550 
        self.model.to(self.device)
        self.model.eval()   
        random.seed(1024)    
        torch.manual_seed(1024)

        if self.use_cuda:
            torch.cuda.manual_seed_all(1024)

        self.idx = torch.tensor(sample(range(0, t_d), d))

        self.outputs = []

        def hook(module, input, output):
            self.outputs.append(output)

        self.model.layer1[-1].register_forward_hook(hook)
        self.model.layer2[-1].register_forward_hook(hook)
        self.model.layer3[-1].register_forward_hook(hook)

    def get_embedding(self, dataloader: DataLoader):
        train_outputs = OrderedDict([('layer1', []), ('layer2', []), ('layer3', [])])

        # extract train set features
        for (x, _, _) in tqdm(dataloader, '| feature extraction |'):
            # model prediction
            with torch.no_grad():
                _ = self.model(x.to(self.device))
            # get intermediate layer outputs
            for k, v in zip(train_outputs.keys(), self.outputs):
                train_outputs[k].append(v.cpu().detach())
            # initialize hook outputs
            self.outputs = []
        for k, v in train_outputs.items():
            train_outputs[k] = torch.cat(v, 0)

        self.size = x.size(2)

        # Embedding concat
        embedding_vectors = train_outputs['layer1']
        for layer_name in ['layer2', 'layer3']:
            embedding_vectors = embedding_concat(embedding_vectors, train_outputs[layer_name])

         # randomly select d dimension
        embedding_vectors = torch.index_select(embedding_vectors, 1, self.idx)
        # calculate multivariate Gaussian distribution
        B, C, H, W = embedding_vectors.size()
        embedding_vectors = embedding_vectors.view(B, C, H * W)

        return embedding_vectors, B, C, H, W

    def fit(self, dataloader: DataLoader):
        print("fit start")
        embedding_vectors, B, C, H, W = self.get_embedding(dataloader)
        print("got embedding")
        mean = torch.mean(embedding_vectors, dim=0).numpy()
        cov = torch.zeros(C, C, H * W).numpy()
        I = np.identity(C)

        for i in tqdm(range(H * W)):
            # cov[:, :, i] = LedoitWolf().fit(embedding_vectors[:, :, i].numpy()).covariance_
            cov[:, :, i] = np.cov(embedding_vectors[:, :, i].numpy(), rowvar=False) + 0.01 * I
        # save learned distribution
        self.train_outputs = [mean, cov]

    def predict(self, dataloader: DataLoader):
        print("predict start")
        embedding_vectors, B, C, H, W = self.get_embedding(dataloader)
        print("got embedding")
        embedding_vectors = embedding_vectors.numpy()

        dist_list = []
        for i in tqdm(range(H * W)):
            mean = self.train_outputs[0][:, i]
            conv_inv = np.linalg.inv(self.train_outputs[1][:, :, i])
            dist = [mahalanobis(sample[:, i], mean, conv_inv) for sample in embedding_vectors]
            dist_list.append(dist)

        dist_list = np.array(dist_list).transpose(1, 0).reshape(B, H, W)

        # upsample
        dist_list = torch.tensor(dist_list)
        score_map = F.interpolate(dist_list.unsqueeze(1), size=self.size, mode='bilinear',
                                    align_corners=False).squeeze().numpy()
        
        # apply gaussian smoothing on the score map
        for i in range(score_map.shape[0]):
            score_map[i] = gaussian_filter(score_map[i], sigma=4)
        
        # Normalization
        max_score = score_map.max()
        min_score = score_map.min()
        scores = (score_map - min_score) / (max_score - min_score)

        return scores


    
def main():

    print(torch.cuda.is_available())

    args = parse_args()

    model = GaussianCnnPredictor(arch = args.arch)

    os.makedirs(os.path.join(args.save_path, 'temp_%s' % args.arch), exist_ok=True)

    class_name = 'bottle'
    data_path = args.data_path
    save_path = args.save_path
    arch = args.arch

    train_dataset = mvtec.MVTecDataset(data_path, class_name=class_name, is_train=True)
    train_dataloader = DataLoader(train_dataset, batch_size=32, pin_memory=True)
    test_dataset = mvtec.MVTecDataset(data_path, class_name=class_name, is_train=False)
    test_dataloader = DataLoader(test_dataset, batch_size=32, pin_memory=True)

    # model.fit(train_dataloader)
    # scores = model.predict(test_dataloader)

    # scores = np.array(scores*255, np.uint8)

    test_imgs = []
    # extract test set features
    for (x, _, _) in tqdm(test_dataloader, '| feature extraction |'):
        test_imgs.extend(x.cpu().detach().numpy())

    fig_img, ax_img = plt.subplots(1, 1, figsize=(12, 3))
    fig_img.subplots_adjust(right=0.9)
    img = test_imgs[0]
    img = denormalization(img)
    ax_img.imshow(img, cmap='gray', interpolation='none')
    # ax_img.imshow(scores[0], cmap='jet', alpha=0.5, interpolation='none')

    fig_img.savefig('bbb.jpg', dpi=100)
    
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


def denormalization(x):
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    x = (((x.transpose(1, 2, 0) * std) + mean) * 255.).astype(np.uint8)
    
    return x


def embedding_concat(x, y):
    B, C1, H1, W1 = x.size()
    _, C2, H2, W2 = y.size()
    s = int(H1 / H2)
    x = F.unfold(x, kernel_size=s, dilation=1, stride=s)
    x = x.view(B, C1, -1, H2, W2)
    z = torch.zeros(B, C1 + C2, x.size(2), H2, W2)
    for i in range(x.size(2)):
        z[:, :, i, :, :] = torch.cat((x[:, :, i, :, :], y), 1)
    z = z.view(B, -1, H2 * W2)
    z = F.fold(z, kernel_size=s, output_size=(H1, W1), stride=s)

    return z


if __name__ == '__main__':
    main()