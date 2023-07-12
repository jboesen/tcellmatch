import unittest
import torch as tc
import tcellmatch.api as tm
import os
import shutil
import numpy as np

class TestEstimatorFfn(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.ffn = tm.models.EstimatorFfn()

        self.ffn.x_train = np.random.randn(50, 1, 40, 26)
        self.ffn.x_val = np.random.randn(10, 1, 40, 26)
        self.ffn.x_test = np.random.randn(10, 1, 40, 26)
        self.ffn.covariates_train = np.random.randn(50, 2)
        self.ffn.covariates_val = np.random.randn(10, 2)
        self.ffn.covariates_test = np.random.randn(10, 2)
        self.ffn.y_train = np.random.randn(50, 51)
        self.ffn.y_val = np.random.randn(10, 51)
        self.ffn.y_test = np.random.randn(10, 51)

        self.ffn.build_bilstm(
            topology = [10, 10],
            aa_embedding_dim=0,
            optimizer='adam',
            lr=0.001,
            loss='pois',
            label_smoothing=0,
            use_covariates=False,
            one_hot_y=False
        )
    
    def test_predict(self):
        self.ffn.predict()
        assert self.ffn.predictions is not None