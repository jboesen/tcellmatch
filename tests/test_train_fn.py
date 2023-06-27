import torch as tc
import tcellmatch.api as tm

# def test constructors
# 1. Train one iteration
def test_build_bilstm():
    # generate dummy data
    ffn = tm.models.EstimatorFn()

    ffn.x_train = tc.randn((50, 1, 40, 26))
    ffn.x_val = tc.randn((10, 1, 40, 26))
    ffn.x_test = tc.randn((10, 1, 40, 26))
    ffn.covariates_train = tc.randn((50, 2))
    ffn.covariates_val = tc.randn((10, 2))
    ffn.covariates_test = tc.randn((10, 2))
    ffn.y_train = tc.randn((50, 51))
    ffn.y_val = tc.randn((10, 51))
    ffn.y_test = tc.randn((10, 51))

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

    EPOCHS = 2
    train_curve, val_curve = ffn.train(
        epochs=EPOCHS,
        batch_size=8,
        log_dir='training_runs',
        save_antigen_loss=False,
        allow_early_stopping=True,
        use_existing_eval_partition=True
    )

    # Checking that the train_curve and val_curve are not None
    assert train_curve is not None
    assert val_curve is not None


    # Checking that the train_curve and val_curve contain proper values (loss should be >=0)
    assert all(x >= 0 for x in train_curve)
    assert all(x >= 0 for x in val_curve)
