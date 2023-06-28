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
            residual_connection=True,
            aa_embedding_dim=0,
            optimizer='adam',
            lr=0.001,
            loss='pois',
            label_smoothing=0,
            use_covariates=False,
            one_hot_y=False
        )
    
    
    def test_save_load(self):
        try:
            os.makedirs("save_test", exist_ok=True)
            # save_yhat means save predictions
            self.ffn.save_model_full(f'save_test', save_yhat=True, save_train_data=True)
            before_eval = self.ffn.evaluate(test_only=True)
            original_state_dict = {name: param.clone() for name, param in self.ffn.model.state_dict().items()}
            ffn2 = tm.models.EstimatorFfn()
            ffn2.load_model_full(fn=f'save_test')
            ffn2.evaluate(test_only=True)
            after_eval = ffn2.evaluate(test_only=True)
            # assert before_eval == after_eval
            for name, param in ffn2.model.state_dict().items():
                assert np.allclose(param.cpu().numpy(), original_state_dict[name].cpu().numpy(), atol=1e-6), f"Internal state mismatch in {name}"
            # assert True
        finally:
            shutil.rmtree('save_test')
