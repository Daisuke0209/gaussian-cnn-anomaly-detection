import unittest
import sys
sys.path.append('..')
import numpy as np
from torch.utils.data import DataLoader
from module.model import GaussianCnnPredictor
import module.mvtec as mvtec


class TestPredictor(unittest.TestCase):
    def setUp(self):
        self.data_path = 'test_dataset'
        self.class_name = 'bottle-small'
        self.arch = 'resnet18'

    def test_fit_predict(self):
        """test_fit_predict
        test fit and predict method of GaussianCnnPredictor
        """
        train_dataset = mvtec.MVTecDataset(self.data_path, class_name=self.class_name, is_train=True)
        train_dataloader = DataLoader(train_dataset, batch_size=2, pin_memory=True)
        test_dataset = mvtec.MVTecDataset(self.data_path, class_name=self.class_name, is_train=False)
        test_dataloader = DataLoader(test_dataset, batch_size=2, pin_memory=True)

        model = GaussianCnnPredictor(arch = self.arch)
        model.fit(train_dataloader)
        heatmaps = model.predict(test_dataloader)

        self.assertTrue(np.abs(np.average(heatmaps[0]) - 73.10817920918367) < 1e-6)
        self.assertTrue(np.abs(np.average(heatmaps[1]) - 82.39164142219387) < 1e-6)

        
