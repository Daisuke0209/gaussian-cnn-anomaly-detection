import unittest
import sys
sys.path.append('..')
import numpy as np
from module import tools


class TestTools(unittest.TestCase):
    def setUp(self):
        self.heatmap = np.array(np.random.rand(10, 10)*255, np.uint8)

    def test_get_bbx(self):
        """test_get_bbx
        test get_bbx function
        """
        binary, bbxes, judge = tools.get_bbx(self.heatmap, threshold=100, min_detected_area=10)
        self.assertTrue(type(binary) == np.ndarray)
        self.assertTrue(type(bbxes) == list)
        self.assertTrue(type(judge) == str)
