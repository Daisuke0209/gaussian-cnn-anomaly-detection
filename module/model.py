import random
from random import sample
from collections import OrderedDict
import gc
import cv2
from tqdm import tqdm
import numpy as np
from scipy.spatial.distance import mahalanobis
from scipy.ndimage import gaussian_filter
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
from torchvision.models import wide_resnet50_2, resnet18

class GaussianCnnPredictor():
    """
    This class detect anomaly in input image using Cnn and Gaussian
    """
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
        """get_embedding
        get embeddings as image feature using pretrained CNN

        Parameters
        -------
        dataloader : DataLoader
            Dataset to get embeddings

        Returns
        -------
        embedding_vectors : torch.Tensor
            embedding vectors using pretrained CNN
        """
        train_outputs = OrderedDict([('layer1', []), ('layer2', []), ('layer3', [])])

        # extract train set features
        for (x, _) in tqdm(dataloader, '| feature extraction |'):
            # model prediction
            with torch.no_grad():
                _ = self.model(x.to(self.device))
            # get intermediate layer outputs
            for k, v in zip(train_outputs.keys(), self.outputs):
                train_outputs[k].append(v.cpu().detach())
            # initialize hook outputs
            self.outputs = []
        print("feature extraction done")
        for k, v in train_outputs.items():
            train_outputs[k] = torch.cat(v, 0)

        self.size = x.size(2)

        # Embedding concat
        embedding_vectors = train_outputs['layer1']
        del train_outputs['layer1']
        gc.collect()
        for layer_name in ['layer2', 'layer3']:
            embedding_vectors = embedding_concat(embedding_vectors, train_outputs[layer_name])
            del train_outputs[layer_name]
            gc.collect()
        print("combined embedding features")

        # randomly select d dimension
        embedding_vectors = torch.index_select(embedding_vectors, 1, self.idx)
        print("selected embedding features")
        # calculate multivariate Gaussian distribution
        B, C, H, W = embedding_vectors.size()
        embedding_vectors = embedding_vectors.view(B, C, H * W)

        return embedding_vectors, B, C, H, W

    def fit(self, dataloader: DataLoader):
        """fit
        fit function. get features of inuput dataset

        Parameters
        -------
        dataloader : DataLoader
            Dataset to get embeddings
        """
        print("fit start")
        embedding_vectors, B, C, H, W = self.get_embedding(dataloader)
        print("got embedding")
        # Average over the entire input images
        mean = torch.mean(embedding_vectors, dim=0).numpy()
        cov = torch.zeros(C, C, H * W).numpy()
        I = np.identity(C)

        for i in tqdm(range(H * W)):
            # cov[:, :, i] = LedoitWolf().fit(embedding_vectors[:, :, i].numpy()).covariance_
            cov[:, :, i] = np.cov(embedding_vectors[:, :, i].numpy(), rowvar=False) + 0.01 * I
        
        del embedding_vectors
        gc.collect()

        # save learned distribution
        self.train_outputs = [mean, cov]

    def predict(self, dataloader: DataLoader):
        """predict
        predict function. get heatmap of inuput dataset

        Parameters
        -------
        dataloader : DataLoader
            Dataset to get embeddings

        Returns
        -------
        heatmaps : np.array
            heatmap for anomaly level
        """
        print("predict start")
        embedding_vectors, B, C, H, W = self.get_embedding(dataloader)
        print("got embedding")
        embedding_vectors = embedding_vectors.numpy()

        # calcurate mahalanobis distance from OK feature distributions
        dist_list = []
        for i in tqdm(range(H * W)):
            mean = self.train_outputs[0][:, i]
            conv_inv = np.linalg.inv(self.train_outputs[1][:, :, i])
            dist = [mahalanobis(sample[:, i], mean, conv_inv) for sample in embedding_vectors]
            dist_list.append(dist)
        print(f'dist[0]: {dist[0]}')
        print(f'len(dist): {len(dist)}')
        print(f'len(dist_list): {len(dist_list)}')
        
        print("got distances")
        del embedding_vectors
        gc.collect()

        dist_list = np.array(dist_list).transpose(1, 0).reshape(B, H, W)
        print(f'dist_list.shape: {dist_list.shape}')

        # upsample
        dist_list = torch.tensor(dist_list)
        score_map = F.interpolate(dist_list.unsqueeze(1), size=self.size, mode='bilinear',
                                    align_corners=False).squeeze().numpy()
        print(f'score_map.shape: {score_map.shape}')
        del dist_list
        # apply gaussian smoothing on the score map
        for i in range(score_map.shape[0]):
            score_map[i] = gaussian_filter(score_map[i], sigma=4)
        
        # Normalization
        max_score = score_map.max()
        min_score = score_map.min()
        heatmaps = (score_map - min_score) / (max_score - min_score)
        heatmaps = np.array(heatmaps*255, np.uint8)
        return heatmaps


def embedding_concat(x, y):
    """embedding_concat
    Combine the two input tensors

    Parameters
    -------
    x : torch.Tensor
        input tensor
    y : torch.Tensor
        input tensor

    Returns
    -------
    z : torch.Tensor
        output tensor
    """
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