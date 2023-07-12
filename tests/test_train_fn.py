import unittest
import torch as tc
import torch.nn.functional as F
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
            aa_embedding_dim=0,
            optimizer='adam',
            lr=0.001,
            loss='mse',
            label_smoothing=0,
            use_covariates=False,
            one_hot_y=False,
        )

        EPOCHS = 2
        train_curve, val_curve, antigen_loss, antigen_loss_val = self.ffn.train(
            epochs=EPOCHS,
            batch_size=8,
            log_dir='training_runs',
            allow_early_stopping=True,
            use_existing_eval_partition=False,
            use_wandb=False
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
            aa_embedding_dim=0,
            optimizer='adam',
            lr=0.001,
            loss='pois',
            label_smoothing=0,
            use_covariates=False,
            one_hot_y=False,
        )
        EPOCHS = 20
        BATCH_SIZE = 5000
        train_curve, val_curve, antigen_loss, antigen_loss_val = self.ffn.train(
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            log_dir='training_runs',
            allow_early_stopping=False,
            use_existing_eval_partition=False,
            use_wandb=False
        )

        loss_fn = tc.nn.PoissonNLLLoss(full=True)
        # Test final loss

        tc_x = tc.as_tensor(self.ffn.x_train[self.ffn.idx_val], dtype=tc.float32)
        tc_y = tc.as_tensor(self.ffn.y_train[self.ffn.idx_val], dtype=tc.float32)
        n_datapoints = len(self.ffn.x_train[self.ffn.idx_val])
        with tc.no_grad():
            y_pred = self.ffn.model(tc_x)
        # save y_pred
        loss = loss_fn(y_pred, tc_y).item()
        assert abs(loss - val_curve[-1]) < 1e-5, "Incorrect train curve"
        # Checking that the train_curve and val_curve are not None
        assert train_curve is not None
        assert val_curve is not None
    
    def test_mse_loss(self):
        def mean_squared_error(y_true, y_pred):
            error = y_true - y_pred
            loss = tc.mean(tc.square(error))
            return loss

        preds = tc.rand(100, 51)
        y = tc.rand(100, 51)

        mse = mean_squared_error(y, preds)
        ffn_mse = self.ffn._get_loss_function('mse')(y, preds).item()

        assert abs(mse - ffn_mse) < 1e-5, 'MSE loss is not correct'

    def test_poisson_loss(self):
        def stirling(n : tc.Tensor):
            stirling_approx = (n * tc.log(n) - n + 0.5 * tc.log(2 * tc.pi * n))
            stirling_approx[n <= 1] = 0
            return stirling_approx

        def poisson_loss(y_true, y_pred):
            # y_pred is assumed to be log(λ)
            y_pred_lambda = tc.exp(y_pred)
            loss = tc.mean(y_pred_lambda - y_true * y_pred + stirling(y_true))
            return loss

        preds = tc.rand(100,)
        # Generate some integers for testing the Poisson loss
        y = tc.randint(5, 100, (100,)).int()

        poisson = poisson_loss(y, preds)
        ffn_poisson = self.ffn._get_loss_function('pois')(preds, y)
        print(poisson, ffn_poisson)
        assert tc.isclose(poisson, ffn_poisson), 'Incorrect Poisson loss'

    # def test_bce_loss(self):
    #     def binary_cross_entropy(y_true, y_pred):
    #         epsilon = 1e-15
    #         y_pred = tc.clip(y_pred, epsilon, 1 - epsilon)
    #         loss = -tc.mean(y_true * tc.log(y_pred) + (1 - y_true) * tc.log(1 - y_pred))
    #         return loss

    #     num_samples = 200
    #     num_classes = 20

    #     indices_y = tc.randint(0, num_classes, (num_samples,))
    #     y = F.one_hot(indices_y, num_classes=num_classes).float()  # Need to ensure y is float type for binary_cross_entropy

    #     # preds should be probabilities, so we'll use softmax
    #     logits = tc.randn(num_samples, num_classes)
    #     preds = tc.softmax(logits, dim=1)

    #     bce = binary_cross_entropy(y, preds)
    #     ffn_bce = self.ffn._get_loss_function('bce')(y, preds).item()

    #     print('\n\n\n\n bce', bce, ffn_bce)
    #     assert abs(bce - ffn_bce) < 1e-5, 'BCE loss is not correct'

