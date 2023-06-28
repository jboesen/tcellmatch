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
        self.ffn.clone_train = np.random.randn(50)
        self.ffn.idx_train_val = np.array(range(0, 50))


    def test_train_fn_mse(self):
        self.ffn.build_bilstm(
            topology = [10, 10],
            residual_connection=True,
            aa_embedding_dim=0,
            optimizer='adam',
            lr=0.001,
            loss='mse',
            label_smoothing=0,
            use_covariates=False,
            one_hot_y=False
        )
        EPOCHS = 2
        train_curve, val_curve = self.ffn.train(
            epochs=EPOCHS,
            batch_size=8,
            log_dir='training_runs',
            save_antigen_loss=False,
            allow_early_stopping=True,
            use_existing_eval_partition=False,
        )

        # Checking that the train_curve and val_curve are not None
        assert train_curve is not None
        assert val_curve is not None

        # Checking that the train_curve and val_curve contain proper values (loss should be >=0)
        assert all(x >= 0 for x in train_curve)
        assert all(x >= 0 for x in val_curve)

    def test_train_fn_pois(self):
        self.ffn.build_bilstm(
            topology = [10, 10],
            residual_connection=True,
            aa_embedding_dim=0,
            optimizer='adam',
            lr=0.001,
            loss='pois',
            label_smoothing=0,
            use_covariates=False,
            one_hot_y=False
        )
        EPOCHS = 2
        train_curve, val_curve = self.ffn.train(
            epochs=EPOCHS,
            batch_size=8,
            log_dir='training_runs',
            save_antigen_loss=False,
            allow_early_stopping=True,
            use_existing_eval_partition=False,
        )

        # Checking that the train_curve and val_curve are not None
        assert train_curve is not None
        assert val_curve is not None
