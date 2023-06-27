import torch as tc
import tcellmatch.api as tm

# def test constructors
# 1. Train one iteration
def test_build_bilstm():

    ffn = tm.models.EstimatorFfn()

    ffn.build_bilstm(
        topology = [10, 10],
        residual_connection=True,
        aa_embedding_dim=0,
        optimizer='adam',
        lr=0.001,
        loss='pois' if USE_BIND_COUNTS else 'wcbe',
        label_smoothing=0,
        use_covariates=False,
        one_hot_y=not USE_BIND_COUNTS
    )

def test_build_sa():
    ffn = tm.models.EstimatorFfn()

    ffn.build_self_attention(
        residual_connection=True,
        aa_embedding_dim=0,
        # hidden size of each attention layer
        attention_size=[5, 5],
        # number of heads at each layer
        attention_heads=[4, 4],
        optimizer='adam',
        lr=0.001,
        loss='mmd' if USE_BIND_COUNTS else 'wbce',
        label_smoothing=0
    )