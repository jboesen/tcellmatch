import unittest
import torch as tc
import tcellmatch.api as tm

class TestEstimatorFfn(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.ffn = tm.models.EstimatorFfn()

        self.ffn.x_train = tc.randn((50, 1, 40, 26))
        self.ffn.x_val = tc.randn((10, 1, 40, 26))
        self.ffn.x_test = tc.randn((10, 1, 40, 26))
        self.ffn.covariates_train = tc.randn((50, 2))
        self.ffn.covariates_val = tc.randn((10, 2))
        self.ffn.covariates_test = tc.randn((10, 2))
        self.ffn.y_train = tc.randn((50, 51))
        self.ffn.y_val = tc.randn((10, 51))
        self.ffn.y_test = tc.randn((10, 51))

    # TODO: test saving with one-hot and counts
    def test_build_bilstm(self):

        self.ffn.build_bilstm(
            topology=[10, 10],
            residual_connection=True,
            aa_embedding_dim=0,
            optimizer='adam',
            lr=0.001,
            loss='pois',
            label_smoothing=0,
            use_covariates=False,
            one_hot_y=False
        )
        # Add assertions to verify the expected behavior of the method
        self.assertIsNotNone(self.ffn.model)  # Example assertion

    def test_build_sa(self):        
        self.ffn.build_self_attention(
            residual_connection=True,
            aa_embedding_dim=0,
            attention_size=[5, 5],
            attention_heads=[4, 4],
            optimizer='adam',
            lr=0.001,
            loss='mmd',
            label_smoothing=0
        )
        self.assertIsNotNone(self.ffn.model) 
    
    def test_build_bigru(self):
        self.ffn.build_bigru(
            topology=[10, 10],
            residual_connection=True,
            aa_embedding_dim=0,
            optimizer='adam',
            lr=0.001,
            loss='pois',
            label_smoothing=0,
            use_covariates=False,
            one_hot_y=False
        )
        self.assertIsNotNone(self.ffn.model)
    
    def test_build_cnn(self):
        self.ffn.build_cnn(
            topology=[10, 10],
            residual_connection=True,
            aa_embedding_dim=0,
            optimizer='adam',
            lr=0.001,
            loss='pois',
            label_smoothing=0,
            use_covariates=False,
            one_hot_y=False
        )
        self.assertIsNotNone(self.ffn.model)

if __name__ == '__main__':
    unittest.main()