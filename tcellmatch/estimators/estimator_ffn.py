# pylint: disable=E1101
# pylint: disable=C0302
# pylint: disable=C0301
# pylint: disable=C0114
# pylint: disable=C0103
# above disables line too long, docstring, can't find function, module too long
import os
import pickle
from typing import Union, List
import numpy as np
import pandas as pd
import scipy.sparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import train_test_split

from tcellmatch.models.models_ffn import ModelBiRnn, ModelSa, ModelConv, ModelLinear, ModelNoseq
from tcellmatch.models.model_inception import ModelInception
from tcellmatch.estimators.additional_metrics import pr_global, pr_label, auc_global, auc_label, \
    deviation_global, deviation_label
from tcellmatch.estimators.estimator_base import EstimatorBase
from tcellmatch.estimators.losses import WeightedBinaryCrossEntropy, MMD
from tcellmatch.estimators.metrics import custom_r2, custom_logr2

from typing import List, Tuple


class EstimatorFfn(EstimatorBase):

    model: torch.nn.Module
    model_hyperparam: dict
    train_hyperparam: dict
    history: dict
    evaluations: dict
    evaluations_custom: dict

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        EstimatorBase.__init__(self=self)
        self.criterion = None
        self.optimizer = None
        # ? these are old params, may delete later... use most of them
        self.model_hyperparam = None
        self.train_hyperparam = None
        self.wbce_weight = None
        # Training and evaluation output containers.
        self.history = None
        self.results_test = None
        self.predictions = None
        self.evaluations = None
        self.evaluations_custom = None

    def _out_activation(self, loss) -> str:
        """ Decide whether network output activation

        This decision is based on the loss function.

        :param loss: loss name

            - "categorical_crossentropy", "cce" for single boolean binding events with binary crossentropy loss.
            - "binary_crossentropy", "bce" for multiple boolean binding events with binary crossentropy loss.
            - "weighted_binary_crossentropy", "wbce" for multiple boolean binding events with
                weighted binary crossentropy loss.
            - "mean_squared_error", "mse" for continuous value prediction with mean squared error loss.
            - "mean_squared_logarithmic_error", "msle" for continuous value prediction with mean squared
                logarithmic error loss.
            - "poisson", "pois" for count value prediction based on Poisson log-likelihood.
        :return: How network output transformed:

            - "categorical_crossentropy", "cce": softmax
            - "binary_crossentropy", "bce": sigmoid
            - "weighted_binary_crossentropy", "wbce": sigmoid
            - "mean_squared_error", "mse": linear
            - "mean_squared_logarithmic_error", "msle": exp
            - "poisson", "pois": exp
        """
        if loss.lower() in ["categorical_crossentropy", "cce"]:
            return "softmax"
        elif loss.lower() in ["binary_crossentropy", "bce"]:
            return "sigmoid"
        elif loss.lower() in ["weighted_binary_crossentropy", "wbce"]:
            return "linear"  # Cost function expect logits.
        elif loss.lower() in ["mean_squared_error", "mse"]:
            return "linear"
        elif loss.lower() in ["mean_squared_logarithmic_error", "msle"]:
            return "exponential"
        elif loss.lower() in ["poisson", "pois"]:
            return "exponential"
        elif loss.lower() == 'mmd':
            return "softmax"
        else:
            raise ValueError(f"Loss {loss} not recognized.")

    def set_wbce_weight(self, weight):
        """ Overwrites automatically computed weight that is chosen based on training data.

        :param weight: Weight to use.
        :return:
        """
        self.wbce_weight = weight


    def build_bilstm(
            self,
            topology: List[int],
            split: bool = False,
            aa_embedding_dim: Union[None, int] = None,
            depth_final_dense: int = 1,
            residual_connection: bool = False,
            dropout: float = 0.0,
            optimizer: str = "adam",
            lr: float = 0.005,
            loss: str = "bce",
            label_smoothing: float = 0,
            optimize_for_gpu: bool = True,
            dtype: str = "float32",
            use_covariates: bool = True,
            one_hot_y: bool = True
    ):
        """ Build a BiLSTM-based feed-forward model to use in the estimator.

        :param topology: The depth of each bilstm layer (length of feature vector)
        :param residual_connection: apply residual connection or not.
        :param aa_embedding_dim: Dimension of the linear amino acid embedding, ie number of 1x1 convolutional filters.
            This is set to the input dimension if aa_embedding_dim==0.
        :param depth_final_dense: Number of final densely connected layers. They all have labels_dim number of units
            and relu activation functions, apart from the last, which has either linear or sigmoid activation,
            depending on out_probabilities.
        :param dropout: Drop-out rate for training.
        :param optimizer: str optimizer name or instance of tf.keras.optimizers
        :param loss: loss name

            - "categorical_crossentropy", "cce" for single boolean binding events with binary crossentropy loss.
            - "binary_crossentropy", "bce" for multiple boolean binding events with binary crossentropy loss.
            - "weighted_binary_crossentropy", "wbce" for multiple boolean binding events with
                weighted binary crossentropy loss.
            - "mean_squared_error", "mse" for continuous value prediction with mean squared error loss.
            - "mean_squared_logarithmic_error", "msle" for continuous value prediction with mean squared
                logarithmic error loss.
            - "poisson", "pois" for count value prediction based on Poisson log-likelihood.
        :param label_smoothing: Fraction of the label interval to take out during smoothing. The labels are mapped
            from [0, 1] into [label_smoothing/2, 1-label_smoothing/2] throught the following transform:

                f(x) = x*(1-label_smoothing) + 0.5*label_smoothing
        :param optimize_for_gpu: Whether to choose implementation optimized for GPU.
        :param dtype:
        :return:
        """
        self._build_sequential(
            model="bilstm",
            topology=topology,
            split=split,
            aa_embedding_dim=aa_embedding_dim,
            depth_final_dense=depth_final_dense,
            residual_connection=residual_connection,
            dropout=dropout,
            optimizer=optimizer,
            lr=lr,
            loss=loss,
            label_smoothing=label_smoothing,
            optimize_for_gpu=optimize_for_gpu,
            dtype=dtype,
            use_covariates=use_covariates,
            one_hot_y=one_hot_y
        )

    # ! Still in tf
    def build_bigru(
            self,
            topology: List[int],
            split: bool = False,
            aa_embedding_dim: Union[None, int] = None,
            depth_final_dense: int = 1,
            residual_connection: bool = False,
            dropout: float = 0.0,
            optimizer: str = "adam",
            lr: float = 0.005,
            loss: str = "bce",
            label_smoothing: float = 0,
            optimize_for_gpu: bool = True,
            dtype: str = "float32"
    ):
        """ Build a BiGRU-based feed-forward model to use in the estimator.

        :param topology: The depth of each bilstm layer (length of feature vector)
        :param residual_connection: apply residual connection or not.
        :param aa_embedding_dim: Dimension of the linear amino acid embedding, ie number of 1x1 convolutional filters.
            This is set to the input dimension if aa_embedding_dim==0.
        :param depth_final_dense: Number of final densely connected layers. They all have labels_dim number of units
            and relu activation functions, apart from the last, which has either linear or sigmoid activation,
            depending on out_probabilities.
        :param dropout: Drop-out rate for training.
        :param optimizer: str optimizer name or instance of tf.keras.optimizers
        :param loss: loss name

            - "categorical_crossentropy", "cce" for single boolean binding events with binary crossentropy loss.
            - "binary_crossentropy", "bce" for multiple boolean binding events with binary crossentropy loss.
            - "weighted_binary_crossentropy", "wbce" for multiple boolean binding events with
                weighted binary crossentropy loss.
            - "mean_squared_error", "mse" for continuous value prediction with mean squared error loss.
            - "mean_squared_logarithmic_error", "msle" for continuous value prediction with mean squared
                logarithmic error loss.
            - "poisson", "pois" for count value prediction based on Poisson log-likelihood.s
        :param label_smoothing: Fraction of the label interval to take out during smoothing. The labels are mapped
            from [0, 1] into [label_smoothing/2, 1-label_smoothing/2] throught the following transform:

                f(x) = x*(1-label_smoothing) + 0.5*label_smoothing
        :param optimize_for_gpu: Whether to choose implementation optimized for GPU.
        :param dtype:
        :return:
        """
        self._build_sequential(
            model="bigru",
            topology=topology,
            split=split,
            aa_embedding_dim=aa_embedding_dim,
            depth_final_dense=depth_final_dense,
            residual_connection=residual_connection,
            dropout=dropout,
            optimizer=optimizer,
            lr=lr,
            loss=loss,
            label_smoothing=label_smoothing,
            optimize_for_gpu=optimize_for_gpu,
            dtype=dtype
        )

    def _build_sequential(
            self,
            model: str,
            topology: List[int],
            split: bool,
            aa_embedding_dim: Union[None, int],
            depth_final_dense: int,
            residual_connection: bool,
            dropout: float,
            optimizer: str,
            lr: float,
            loss: str,
            label_smoothing: float,
            optimize_for_gpu: bool,
            dtype: str = "float32",
            use_covariates: bool = True,
            one_hot_y: bool = True
    ):
        """ Build a BiLSTM-based feed-forward model to use in the estimator.

        :param topology: The depth of each bilstm layer (length of feature vector)
        :param residual_connection: apply residual connection or not.
        :param aa_embedding_dim: Dimension of the linear amino acid embedding, ie number of 1x1 convolutional filters.
            This is set to the input dimension if aa_embedding_dim==0.
        :param depth_final_dense: Number of final densely connected layers. They all have labels_dim number of units
            and relu activation functions, apart from the last, which has either linear or sigmoid activation,
            depending on out_probabilities.
        :param dropout: Drop-out rate for training.
        :param optimizer: str optimizer name or instance of tf.keras.optimizers
        :param loss: loss name

            - "categorical_crossentropy", "cce" for single boolean binding events with binary crossentropy loss.
            - "binary_crossentropy", "bce" for multiple boolean binding events with binary crossentropy loss.
            - "weighted_binary_crossentropy", "wbce" for multiple boolean binding events with
                weighted binary crossentropy loss.
            - "mean_squared_error", "mse" for continuous value prediction with mean squared error loss.
            - "mean_squared_logarithmic_error", "msle" for continuous value prediction with mean squared
                logarithmic error loss.
            - "poisson", "pois" for count value prediction based on Poisson log-likelihood.
        :param label_smoothing: Fraction of the label interval to take out during smoothing. The labels are mapped
            from [0, 1] into [label_smoothing/2, 1-label_smoothing/2] throught the following transform:

                f(x) = x*(1-label_smoothing) + 0.5*label_smoothing
        :param optimize_for_gpu: Whether to choose implementation optimized for GPU.
        :param dtype:
        :return:
        """
        # Save model settings:
        self.model_hyperparam = {
            "model": model,
            "topology": topology,
            "split": split,
            "aa_embedding_dim": aa_embedding_dim,
            "depth_final_dense": depth_final_dense,
            "residual_connection": residual_connection,
            "dropout": dropout,
            "optimizer": optimizer,
            "lr": lr,
            "loss": loss,
            "label_smoothing": label_smoothing,
            "optimize_for_gpu": optimize_for_gpu,
            "dtype": dtype,
            "use_covariates": use_covariates
        }

        self.model = ModelBiRnn(
            input_shapes=(
                self.x_train.shape[1],
                self.x_train.shape[2],
                self.x_train.shape[3],
                self.covariates_train.shape[1] if use_covariates else 0,
                self.tcr_len
            ),
            model=model.lower(),
            labels_dim=self.y_train.shape[1],
            topology=topology,
            split=split,
            residual_connection=residual_connection,
            aa_embedding_dim=aa_embedding_dim,
            depth_final_dense=depth_final_dense,
            out_activation=self._out_activation(loss=loss),
            dropout=dropout,
            one_hot_y=one_hot_y
        )

        # Define loss and optimizer
        self.criterion = self.get_loss_function(loss, label_smoothing)
        # TODO: (maybe) this is hard-coded, but it's Adam anyway
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

    def get_loss_function(self, loss: str, label_smoothing : float = 0.0):
        """
        :param loss: loss name

            - "categorical_crossentropy", "cce" for single boolean binding events with binary crossentropy loss.
            - "binary_crossentropy", "bce" for multiple boolean binding events with binary crossentropy loss.
            - "weighted_binary_crossentropy", "wbce" for multiple boolean binding events with
                weighted binary crossentropy loss.
            - "mean_squared_error", "mse" for continuous value prediction with mean squared error loss.
            - "mean_squared_logarithmic_error", "msle" for continuous value prediction with mean squared
                logarithmic error loss.
            - "poisson", "pois" for count value prediction based on Poisson log-likelihood.
        :param label_smoothing: Fraction of the label interval to take out during smoothing. The labels are mapped
            from [0, 1] into [label_smoothing/2, 1-label_smoothing/2] throught the following transform:

                f(x) = x*(1-label_smoothing) + 0.5*label_smoothing
        """

        # label smoothing case
        if label_smoothing and (loss == "binary_crossentropy" or loss == "bce"):
            return nn.BCEWithLogitsLoss(pos_weight=torch.full([1], label_smoothing))
        if label_smoothing and (loss == "categorical_crossentropy" or loss == "cce"):
            return nn.CrossEntropyLoss(weight=torch.full([1], label_smoothing))

        # no label smoothing
        if loss.lower() == "mmd":
            return MMD
        if loss == "categorical_crossentropy" or loss == "cce":
            return nn.CrossEntropyLoss()
        elif loss == "binary_crossentropy" or loss == "bce":
            return nn.BCELoss()
        elif loss == "weighted_binary_crossentropy" or loss == "wbce":
            return nn.BCEWithLogitsLoss()
        elif loss == "mean_squared_error" or loss == "mse":
            return nn.MSELoss()
        elif loss == "poisson" or loss == "pois":
            return nn.PoissonNLLLoss()
        else:
            raise ValueError("Invalid loss name: " + loss)
        return None


    def build_self_attention(
            self,
            attention_size: List[int],
            attention_heads: List[int],
            aa_embedding_dim: Union[None, int] = None,
            depth_final_dense: int = 1,
            residual_connection: bool = False,
            dropout: float = 0.0,
            split: bool = False,
            optimizer: str = "adam",
            lr: float = 0.005,
            loss: str = "bce",
            label_smoothing: float = 0,
            one_hot_y: bool = False,
            dtype: str = "float32",
    ):
        """ Build a self-attention-based feed-forward model to use in the estimator.

        :param attention_size: hidden size for attention, could be divided by attention_heads.
        :param attention_heads: number of heads in attention.
        :param residual_connection: apply residual connection or not.
        :param aa_embedding_dim: Dimension of the linear amino acid embedding, ie number of 1x1 convolutional filters.
            This is set to the input dimension if aa_embedding_dim==0.
        :param depth_final_dense: Number of final densely connected layers. They all have labels_dim number of units
            and relu activation functions, apart from the last, which has either linear or sigmoid activation,
            depending on out_probabilities.
        :param dropout: Drop-out rate for training.
        :param optimizer: str optimizer name
        :param loss: loss name

            - "categorical_crossentropy", "cce" for single boolean binding events with binary crossentropy loss.
            - "binary_crossentropy", "bce" for multiple boolean binding events with binary crossentropy loss.
            - "weighted_binary_crossentropy", "wbce" for multiple boolean binding events with
                weighted binary crossentropy loss.
            - "mean_squared_error", "mse" for continuous value prediction with mean squared error loss.
            - "mean_squared_logarithmic_error", "msle" for continuous value prediction with mean squared
                logarithmic error loss.
            - "poisson", "pois" for count value prediction based on Poisson log-likelihood.
        :param label_smoothing: Fraction of the label interval to take out during smoothing. The labels are mapped
            from [0, 1] into [label_smoothing/2, 1-label_smoothing/2] throught the following transform:

                f(x) = x*(1-label_smoothing) + 0.5*label_smoothing
        :param dtype:
        :return:
        """
        # i.e., indicator for whether we're not loading a previously trained model
        is_new_model = hasattr(self, 'x_train') and self.x_train is not None

        if is_new_model:
            self.model_hyperparam = {
                "model": "selfattention",
                "attention_size": attention_size,
                "attention_heads": attention_heads,
                "split": split,
                "aa_embedding_dim": aa_embedding_dim,
                "depth_final_dense": depth_final_dense,
                "residual_connection": residual_connection,
                "dropout": dropout,
                "optimizer": optimizer,
                "lr": lr,
                "loss": loss,
                "label_smoothing": label_smoothing,
                "dtype": dtype,
                "cov_train_shape": self.covariates_train.shape,
                "tcr_len": self.tcr_len,
                "x_train_shape": self.x_train.shape,
                "y_train_shape": self.y_train.shape
            }

        # Use these so we cover the cases that we're loading hyperparams from storage AND training new model
        x_shape = self.model_hyperparam["x_train_shape"]
        cov_shape = self.model_hyperparam["cov_train_shape"]
        y_shape = self.model_hyperparam["y_train_shape"]
        tcr_len = self.model_hyperparam["tcr_len"]

        self.model = ModelSa(
            input_shapes=(
                x_shape[1],
                x_shape[2],
                x_shape[3],
                cov_shape[1],
                tcr_len
            ),
            input_covar_shape = cov_shape,
            labels_dim=y_shape[1],
            attention_size=attention_size,
            attention_heads=attention_heads,
            residual_connection=residual_connection,
            split=split,
            aa_embedding_dim=aa_embedding_dim,
            out_activation=self._out_activation(loss=loss),
            depth_final_dense=depth_final_dense,
            dropout=dropout,
            one_hot_y=one_hot_y
        )

        # Define loss and optimizer
        self.criterion = self.get_loss_function(loss, label_smoothing)
        # TODO: (maybe) this is hard-coded, but it's Adam anyway
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

    # ! Still in tf
    def build_conv(
            self,
            activations: List[str],
            filter_widths: List[int],
            filters: List[int],
            strides: Union[List[Union[int, None]], None] = None,
            pool_sizes: Union[List[Union[int, None]], None] = None,
            pool_strides: Union[List[Union[int, None]], None] = None,
            batch_norm: bool = False,
            aa_embedding_dim: Union[None, int] = None,
            depth_final_dense: int = 1,
            dropout: float = 0.0,
            split: bool = False,
            optimizer: str = "adam",
            lr: float = 0.005,
            loss: str = "bce",
            label_smoothing: float = 0,
            dtype: str = "float32"
    ):
        """ Build a self-attention-based feed-forward model to use in the estimator.

        :param activations: Activation function. Refer to documentation of tf.keras.layers.Conv2D
        :param filter_widths: Number of neurons per filter. Refer to documentation of tf.keras.layers.Conv2D
        :param filters: NUmber of filters / output channels. Refer to documentation of tf.keras.layers.Conv2D
        :param strides: Stride size for convolution on sequence. Refer to documentation of tf.keras.layers.Conv2D
        :param pool_sizes: Size of max-pooling, ie. number of output nodes to pool over.
            Refer to documentation of tf.keras.layers.MaxPool2D:pool_size
        :param pool_strides: Stride of max-pooling.
            Refer to documentation of tf.keras.layers.MaxPool2D:strides
        :param batch_norm: Whether to perform batch normalization.
        :param aa_embedding_dim: Dimension of the linear amino acid embedding, ie number of 1x1 convolutional filters.
            This is set to the input dimension if aa_embedding_dim==0.
        :param depth_final_dense: Number of final densely connected layers. They all have labels_dim number of units
            and relu activation functions, apart from the last, which has either linear or sigmoid activation,
            depending on out_probabilities.
        :param dropout: Drop-out rate for training.
        :param optimizer: str optimizer name or instance of tf.keras.optimizers
        :param loss: loss name

            - "categorical_crossentropy", "cce" for single boolean binding events with binary crossentropy loss.
            - "binary_crossentropy", "bce" for multiple boolean binding events with binary crossentropy loss.
            - "weighted_binary_crossentropy", "wbce" for multiple boolean binding events with
                weighted binary crossentropy loss.
            - "mean_squared_error", "mse" for continuous value prediction with mean squared error loss.
            - "mean_squared_logarithmic_error", "msle" for continuous value prediction with mean squared
                logarithmic error loss.
            - "poisson", "pois" for count value prediction based on Poisson log-likelihood.
        :param label_smoothing: Fraction of the label interval to take out during smoothing. The labels are mapped
            from [0, 1] into [label_smoothing/2, 1-label_smoothing/2] throught the following transform:

                f(x) = x*(1-label_smoothing) + 0.5*label_smoothing
        :param dtype:
        :return:
        """

        # Save model settings.
        self.model_hyperparam = {
            "model": "conv",
            "activations": activations,
            "filter_widths": filter_widths,
            "filters": filters,
            "strides": strides,
            "pool_sizes": pool_sizes,
            "pool_strides": pool_strides,
            "batch_norm": batch_norm,
            "split": split,
            "aa_embedding_dim": aa_embedding_dim,
            "depth_final_dense": depth_final_dense,
            "dropout": dropout,
            "optimizer": optimizer,
            "lr": lr,
            "loss": loss,
            "label_smoothing": label_smoothing,
            "dtype": dtype
        }

        # Build model.
        self.model = ModelConv(
            input_shapes=(
                self.x_train.shape[1],
                self.x_train.shape[2],
                self.x_train.shape[3],
                self.covariates_train.shape[1],
                self.tcr_len
            ),
            labels_dim=self.y_train.shape[1],
            activations=activations,
            filter_widths=filter_widths,
            filters=filters,
            strides=strides,
            pool_sizes=pool_sizes,
            pool_strides=pool_strides,
            batch_norm=batch_norm,
            split=split,
            aa_embedding_dim=aa_embedding_dim,
            out_activation=self._out_activation(loss=loss),
            depth_final_dense=depth_final_dense,
            dropout=dropout
        )
        self._compile_model(
            optimizer=optimizer,
            lr=lr,
            loss=loss,
            label_smoothing=label_smoothing
        )

    # ! Still in tf
    def build_inception(
            self,
            n_filters_1x1: List[int],
            n_filters_out: List[int],
            n_hidden: int = 10,
            residual_connection: bool = True,
            aa_embedding_dim: Union[None, int] = None,
            depth_final_dense: int = 1,
            final_pool: str = "average",
            dropout: float = 0.0,
            split: bool = False,
            optimizer: str = "adam",
            lr: float = 0.005,
            loss: str = "bce",
            label_smoothing: float = 0,
            dtype: str = "float32"
    ):
        """ Build a self-attention-based feed-forward model to use in the estimator.

        :param n_filters_1x1:
        :param n_filters_out:
        :param n_filters_final:
        :param n_hidden:
         :param residual_connection: apply residual connection or not.
        :param aa_embedding_dim: Dimension of the linear amino acid embedding, ie number of 1x1 convolutional filters.
            This is set to the input dimension if aa_embedding_dim==0.
        :param depth_final_dense: Number of final densely connected layers. They all have labels_dim number of units
            and relu activation functions, apart from the last, which has either linear or sigmoid activation,
            depending on out_probabilities.
        :param final_pool:
        :param dropout: Drop-out rate for training.
        :param optimizer: str optimizer name or instance of tf.keras.optimizers
        :param loss: loss name

            - "categorical_crossentropy", "cce" for single boolean binding events with binary crossentropy loss.
            - "binary_crossentropy", "bce" for multiple boolean binding events with binary crossentropy loss.
            - "weighted_binary_crossentropy", "wbce" for multiple boolean binding events with
                weighted binary crossentropy loss.
            - "mean_squared_error", "mse" for continuous value prediction with mean squared error loss.
            - "mean_squared_logarithmic_error", "msle" for continuous value prediction with mean squared
                logarithmic error loss.
            - "poisson", "pois" for count value prediction based on Poisson log-likelihood.
        :param label_smoothing: Fraction of the label interval to take out during smoothing. The labels are mapped
            from [0, 1] into [label_smoothing/2, 1-label_smoothing/2] throught the following transform:

                f(x) = x*(1-label_smoothing) + 0.5*label_smoothing
        :param dtype:
        :return:
        """
        # Save model settings.
        self.model_hyperparam = {
            "model": "inception",
            "n_filters_1x1": n_filters_1x1,
            "n_filters_out": n_filters_out,
            "n_hidden": n_hidden,
            "split": split,
            "final_pool": final_pool,
            "residual_connection": residual_connection,
            "aa_embedding_dim": aa_embedding_dim,
            "depth_final_dense": depth_final_dense,
            "dropout": dropout,
            "optimizer": optimizer,
            "lr": lr,
            "loss": loss,
            "dtype": dtype
        }

        # Build model.
        self.model = ModelInception(
            input_shapes=(
                self.x_train.shape[1],
                self.x_train.shape[2],
                self.x_train.shape[3],
                self.covariates_train.shape[1],
                self.tcr_len
            ),
            labels_dim=self.y_train.shape[1],
            n_filters_1x1=n_filters_1x1,
            n_filters_out=n_filters_out,
            n_hidden=n_hidden,
            split=split,
            final_pool=final_pool,
            residual_connection=residual_connection,
            aa_embedding_dim=aa_embedding_dim,
            depth_final_dense=depth_final_dense,
            out_activation=self._out_activation(loss=loss),
            dropout=dropout
        )

        self._compile_model(
            optimizer=optimizer,
            lr=lr,
            loss=loss,
            label_smoothing=label_smoothing
        )

    def build_linear(
            self,
            aa_embedding_dim: Union[None, int] = None,
            optimizer: str = "adam",
            lr: float = 0.005,
            loss: str = "bce",
            label_smoothing: float = 0,
            dtype: str = "float32"
    ):
        """ Build a linear feed-forward model to use in the estimator.

        :param aa_embedding_dim: Dimension of the linear amino acid embedding, ie number of 1x1 convolutional filters.
            This is set to the input dimension if aa_embedding_dim==0.
        :param optimizer: str optimizer name or instance of tf.keras.optimizers
        :param loss: loss name

            - "categorical_crossentropy", "cce" for single boolean binding events with binary crossentropy loss.
            - "binary_crossentropy", "bce" for multiple boolean binding events with binary crossentropy loss.
            - "weighted_binary_crossentropy", "wbce" for multiple boolean binding events with
                weighted binary crossentropy loss.
            - "mean_squared_error", "mse" for continuous value prediction with mean squared error loss.
            - "mean_squared_logarithmic_error", "msle" for continuous value prediction with mean squared
                logarithmic error loss.
            - "poisson", "pois" for count value prediction based on Poisson log-likelihood.
        :param label_smoothing: Fraction of the label interval to take out during smoothing. The labels are mapped
            from [0, 1] into [label_smoothing/2, 1-label_smoothing/2] throught the following transform:

                f(x) = x*(1-label_smoothing) + 0.5*label_smoothing
        :param dtype:
        :return:
        """
        # Save model settings.
        self.model_hyperparam = {
            "model": "linear",
            "aa_embedding_dim": aa_embedding_dim,
            "optimizer": optimizer,
            "lr": lr,
            "loss": loss,
            "label_smoothing": label_smoothing,
            "dtype": dtype
        }

        # Build model.
        self.model = ModelLinear(
            input_shapes=(
                self.x_train.shape[1],
                self.x_train.shape[2],
                self.x_train.shape[3],
                self.covariates_train.shape[1],
                self.tcr_len
            ),
            labels_dim=self.y_train.shape[1],
            aa_embedding_dim=aa_embedding_dim,
            out_activation=self._out_activation(loss=loss)
        )
        self._compile_model(
            optimizer=optimizer,
            lr=lr,
            loss=loss,
            label_smoothing=label_smoothing
        )

    def build_noseq(
            self,
            optimizer: str = "adam",
            lr: float = 0.005,
            loss: str = "bce",
            label_smoothing: float = 0,
            dtype: str = "float32"
    ):
        """ Build a dense feed-forward model to use in the estimator that does not include the sequence data.

        :param optimizer: str optimizer name or instance of tf.keras.optimizers
        :param loss: loss name

            - "categorical_crossentropy", "cce" for single boolean binding events with binary crossentropy loss.
            - "binary_crossentropy", "bce" for multiple boolean binding events with binary crossentropy loss.
            - "weighted_binary_crossentropy", "wbce" for multiple boolean binding events with
                weighted binary crossentropy loss.
            - "mean_squared_error", "mse" for continuous value prediction with mean squared error loss.
            - "mean_squared_logarithmic_error", "msle" for continuous value prediction with mean squared
                logarithmic error loss.
            - "poisson", "pois" for count value prediction based on Poisson log-likelihood.
        :param label_smoothing: Fraction of the label interval to take out during smoothing. The labels are mapped
            from [0, 1] into [label_smoothing/2, 1-label_smoothing/2] throught the following transform:

                f(x) = x*(1-label_smoothing) + 0.5*label_smoothing
        :param dtype:
        :return:
        """
        # Save model settings.
        self.model_hyperparam = {
            "model": "noseq",
            "optimizer": optimizer,
            "lr": lr,
            "loss": loss,
            "label_smoothing": label_smoothing,
            "dtype": dtype
        }

        # Build model.
        self.model = ModelNoseq(
            input_shapes=(
                self.x_train.shape[1],
                self.x_train.shape[2],
                self.x_train.shape[3],
                self.covariates_train.shape[1],
                self.tcr_len
            ),
            labels_dim=self.y_train.shape[1],
            out_activation=self._out_activation(loss=loss)
        )
        self._compile_model(
            optimizer=optimizer,
            lr=lr,
            loss=loss,
            label_smoothing=label_smoothing
        )


    def train(
        self,
        epochs : int = 1000,
        batch_size : int = 128,
        validation_split : float = 0.1,
        patience : int =20,
        lr_schedule_min_lr: float = 1e-5,
        lr_schedule_factor : float= 0.2,
        lr_schedule_patience : int = 5,
        log_dir : str | None = None,
        use_existing_eval_partition : bool = False,
        validation_batch_size: int = 256,
        allow_early_stopping: bool = False,
        save_antigen_loss: bool = False
    ) -> Tuple[List[float], List[float]]:
        """
        Trains the model based on the given training parameters and data.

        :param epochs: The total number of training iterations. Default is 1000.
        :param batch_size: The size of the batch for gradient updates. Default is 128.
        :param validation_split: Fraction of training data to be used for validation. Default is 0.1.
        :param patience: Number of epochs with no improvement after which training will be stopped. Default is 20.
        :param lr_schedule_min_lr: Lower bound for the learning rate. Default is 1e-5.
        :param lr_schedule_factor: Factor by which learning rate will be reduced when there's no improvement. Default is 0.2.
        :param lr_schedule_patience: Number of epochs with no improvement after which learning rate will be reduced. Default is 5.
        :param log_dir: Directory where TensorBoard logs will be saved. If None, logs won't be saved. Default is None.
        :param use_existing_eval_partition: If set to True, use existing train/eval partition. If False, create new partition. Default is False.
        :param validation_batch_size: Size of batch for validation data. Default is 256.
        :param allow_early_stopping: If set to True, enables early stopping when validation loss doesn't improve. Default is True.
        :param save_antigen_loss: If set to True, exports train loss of ith epoch on jth antigen to self.antigen_loss. Default is False.

        :raises ValueError: If use_existing_eval_partition is True but no eval partition exists.
        
        :return: None
        """

        # Set up optimizer and learning rate scheduler
        optimizer = self.optimizer
        lr_scheduler = ReduceLROnPlateau(optimizer,
                                          mode='min',
                                          factor=lr_schedule_factor,
                                          patience=lr_schedule_patience,
                                          min_lr=lr_schedule_min_lr)

        # Early stopping initialization
        early_stopping_counter = 0
        # this is a placeholder value
        best_val_loss = np.inf

        use_covariates = self.model.has_covariates if hasattr(self.model, 'has_covariates') else True

        writer = None
        if log_dir is not None:
            writer = SummaryWriter(log_dir=log_dir)

        if use_existing_eval_partition:
            if not self.idx_train_val or not self.idx_train or not self.idx_val:
                raise ValueError("ERROR: use_existing_eval_partition is True, but no eval partition exists")
            idx_val = np.array([self.idx_train_val.tolist().index(x)
                                for x in self.idx_train_val if x in self.idx_val])
            idx_train = np.array([self.idx_train_val.tolist().index(x)
                                  for x in self.idx_train_val if x in self.idx_train])
        else:
            # Split training data into training and evaluation.
            # Perform this splitting based on clonotypes.
            clones = np.unique(self.clone_train)
            # pick random subset of clones for validation
            clones_eval = np.random.choice(clones, size=int(len(clones) * validation_split), replace=False)
            # use non-eval clones for train
            clones_train = np.setdiff1d(clones, clones_eval)
            # turn these clones into indices of the clones
            idx_val = np.argwhere(np.isin(self.clone_train, clones_eval)).flatten()
            idx_train = np.argwhere(np.isin(self.clone_train, clones_train)).flatten()
            # turn these indices into indices of the train-val set (we do this b/c
            # there can now be multiple of one type of clone)
            self.idx_train = self.idx_train_val[idx_train]
            self.idx_val = self.idx_train_val[idx_val]

            # Assert that split is exclusive and complete:
            assert len(set(clones_eval).intersection(set(clones_train))) == 0, \
                "ERROR: train-test assignment was not exclusive on level of clones"
            assert len(set(idx_val).intersection(set(idx_train))) == 0, \
                "ERROR: train-test assignment was not exclusive on level of cells"
            assert len(clones_eval) + len(clones_train) == len(clones), \
                "ERROR: train-test split was not complete on the level of clones"
            assert len(idx_val) + len(idx_train) == len(self.clone_train), \
                "ERROR: train-test split was not complete on the level of cells"
        
        print("Number of observations in evaluation data: %i" % len(idx_val))

        print("Number of observations in training data: %i" % len(idx_train)) 

        # np is in float64, but model is in float32
        if use_covariates:
            train_data = TensorDataset(
                torch.from_numpy(self.x_train[idx_train]).to(torch.float32),
                torch.from_numpy(self.covariates_train[idx_train]).to(torch.float32),
                torch.from_numpy(self.y_train[idx_train]).to(torch.float32)
                )
            val_data = TensorDataset(
                torch.from_numpy(self.x_train[idx_val]).to(torch.float32),
                torch.from_numpy(self.covariates_train[idx_val]).to(torch.float32),
                torch.from_numpy(self.y_train[idx_val]).to(torch.float32)
                )
        else:
            train_data = TensorDataset(
                torch.from_numpy(self.x_train[idx_train]).to(torch.float32),
                torch.from_numpy(self.covariates_train[idx_train]).to(torch.float32),
                torch.from_numpy(self.y_train[idx_train]).to(torch.float32)
                )
            val_data = TensorDataset(
                torch.from_numpy(self.x_train[idx_val]).to(torch.float32),
                torch.from_numpy(self.covariates_train[idx_val]).to(torch.float32),
                torch.from_numpy(self.y_train[idx_val]).to(torch.float32)
                )

        train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
        # self.train_loader = train_loader
        val_loader = DataLoader(dataset=val_data, batch_size=validation_batch_size, shuffle=False)
        val_loss_list = []
        train_loss_list = []
        num_classes = self.y_train.shape[-1]
        if save_antigen_loss:
            self.antigen_loss = np.zeros((epochs, num_classes))
        # Training loop
        for epoch in range(epochs):
            # Training phase
            self.model.train()
            running_loss = 0.0
            for x, covariates, y in train_loader:
                x, covariates, y = x.to(self.device), covariates.to(self.device), y.to(self.device)

                optimizer.zero_grad()

                outputs = self.model(x, covariates) if use_covariates else self.model(x)
                # loss = F.mse_loss(outputs, y)
                loss = self.criterion(outputs, y)
                if save_antigen_loss:
                    for i in range(num_classes):
                        antigen_loss[epoch, i]=self.criterion(outputs[:,i],  y[:,i])
                
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * x.size(0)

            # Validation phase
            self.model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for x, covariates, y in val_loader:
                    x, covariates, y = x.to(self.device), covariates.to(self.device), y.to(self.device)

                    outputs = self.model(x, covariates) if use_covariates else self.model(x)
                    loss = F.mse_loss(outputs, y)
                    val_loss += loss.item() * x.size(0)
            # Calculate average losses
            train_loss = running_loss / len(train_loader.dataset)
            val_loss = val_loss / len(val_loader.dataset)
            val_loss_list.append(val_loss)
            train_loss_list.append(train_loss)

            # Update learning rate
            lr_scheduler.step(val_loss)

            # Write to tensorboard
            if writer is not None:
                writer.add_scalar('Train/Loss', train_loss, epoch)
                writer.add_scalar('Val/Loss', val_loss, epoch)

            # Check for early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                early_stopping_counter = 0
            else:
                early_stopping_counter += 1
                if early_stopping_counter >= patience and allow_early_stopping:
                    print('stopped early')
                    break

        # TODO (maybe): add more capabilties to this writer...
        if writer is not None:
            writer.close()
    
        return train_loss_list, val_loss_list

    @property
    def idx_train_in_train_val(self):
        return np.intersect1d(self.idx_train_val, self.idx_train, return_indices=True)[1]

    @property
    def idx_val_in_train_val(self):
        return np.intersect1d(self.idx_train_val, self.idx_val, return_indices=True)[1]

    def evaluate(
            self,
            batch_size: int = 1024,
            test_only: bool = True,
    ):
        """ Evaluate loss on test data.

        :param batch_size: Batch size for evaluation.
        :return:
        """
        results_test = self.evaluate_any(
            x=self.x_test,
            covar=self.covariates_test,
            y=self.y_test,
            batch_size=batch_size
        )
        if not test_only:
            results_val = self.evaluate_any(
                x=self.x_train[self.idx_val_in_train_val],
                covar=self.covariates_train[self.idx_val_in_train_val],
                y=self.y_train[self.idx_val_in_train_val],
                batch_size=batch_size,
            )
            results_train = self.evaluate_any(
                x=self.x_train[self.idx_train_in_train_val],
                covar=self.covariates_train[self.idx_train_in_train_val],
                y=self.y_train[self.idx_train_in_train_val],
                batch_size=batch_size,
            )
            self.evaluations = {
                "test": results_test,
                "val": results_val,
                "train": results_train
            }
        else:
            self.evaluations = {
                    "test": results_test,
                }
        return self.evaluations

    def evaluate_any(
            self,
            x,
            covar,
            y,
            antigen_col : int | None = None,
            batch_size: int = 1024,
    ):
        """ Evaluate loss on supplied data.

        :param batch_size: Batch size for evaluation.
        :param antigen_col: Use to calculate loss for a single antigenm.
        :return: Dictionary of metrics
        """
        self.model.eval()  # Set the model to evaluation mode
        with torch.no_grad():  # Turn off gradients for evaluation
            # Create Tensor datasets for the features, covariates, and targets
            data = TensorDataset(torch.tensor(x, dtype=torch.float32),
                                torch.tensor(covar, dtype=torch.float32),
                                torch.tensor(y, dtype=torch.float32))

            # Create a DataLoader
            data_loader = DataLoader(data, batch_size=batch_size)

            total_loss = 0
            correct_predictions = 0
            total_predictions = 0
            for batch_x, batch_covar, batch_y in data_loader:
                # Move tensors to the right device
                batch_x = batch_x.to(self.device)
                batch_covar = batch_covar.to(self.device)
                batch_y = batch_y.to(self.device)

                # Forward pass
                if hasattr(self.model, 'has_covariates') and self.model.has_covariates:
                    outputs = self.model(batch_x, batch_covar)
                else:
                    outputs = self.model(batch_x)

                # Compute loss
                if antigen_col:
                    loss = self.criterion(outputs[:, antigen_col], batch_y[:, antigen_col])
                else:
                    loss = self.criterion(outputs, batch_y)

                total_loss += loss.item() * batch_x.size(0)

                # Compute binary accuracy
                predicted_labels = torch.round(torch.sigmoid(outputs))
                correct_predictions += (predicted_labels == batch_y).sum().item()
                total_predictions += batch_y.size(0)

            # Average the loss over all observations
            avg_loss = total_loss / len(data)
            accuracy = correct_predictions / total_predictions

            return {self.criterion: avg_loss, "Binary Accuracy": accuracy}

    def evaluate_custom(
            self,
            classification_metrics: bool = True,
            regression_metrics: bool = False,
            transform: str = None
    ):
        """ Obtain custom evaluation metrics for classification task on train, val and test data.
        """
        results_test = self.evaluate_custom_any(
            yhat=self.predict_any(x=self.x_test, covar=self.covariates_test, batch_size=1024),
            yobs=self.y_test,
            nc=self.nc_test,
            labels=np.asarray(self.peptide_seqs_test),
            labels_unique=self.peptide_seqs_unique,
            classification_metrics=classification_metrics,
            regression_metrics=regression_metrics,
            transform_flavour=transform
        )
        results_val = self.evaluate_custom_any(
            yhat=self.predict_any(
                x=self.x_train[self.idx_val_in_train_val],
                covar=self.covariates_train[self.idx_val_in_train_val],
                batch_size=1024
            ),
            yobs=self.y_train[self.idx_val_in_train_val],
            nc=self.nc_train[self.idx_val_in_train_val] if self.nc_train is not None else None,
            labels=np.asarray(self.peptide_seqs_train)[self.idx_val_in_train_val] \
                if self.peptide_seqs_train is not None else None,
            labels_unique=self.peptide_seqs_unique,
            classification_metrics=classification_metrics,
            regression_metrics=regression_metrics,
            transform_flavour=transform
        )
        results_train = self.evaluate_custom_any(
            yhat=self.predict_any(
                x=self.x_train[self.idx_train_in_train_val],
                covar=self.covariates_train[self.idx_train_in_train_val],
                batch_size=1024
            ),
            yobs=self.y_train[self.idx_train_in_train_val],
            nc=self.nc_train[self.idx_train_in_train_val] if self.nc_train is not None else None,
            labels=np.asarray(self.peptide_seqs_train)[self.idx_train_in_train_val] \
                if self.peptide_seqs_train is not None else None,
            labels_unique=self.peptide_seqs_unique,
            classification_metrics=classification_metrics,
            regression_metrics=regression_metrics,
            transform_flavour=transform
        )
        self.evaluations_custom = {
            "test": results_test,
            "val": results_val,
            "train": results_train
        }

    def _evaluate_custom_any(
            self,
            yhat,
            yobs,
            nc,
            classification_metrics: bool,
            regression_metrics: bool,
            labels=None,
            labels_unique=None,
            transform_flavour: str = None
    ):
        """ Obtain custom evaluation metrics for classification task on any data.
        """
        metrics_global = {}
        metrics_local = {}
        if regression_metrics:
            mse_global, msle_global, r2_global, r2log_global = deviation_global(
                y_hat=[yhat], y_obs=[yobs]
            )
            mse_label, msle_label, r2_label, r2log_label = deviation_label(
                y_hat=[yhat], y_obs=[yobs], labels=[labels], labels_unique=labels_unique
            )
            metrics_global.update({
                "mse": mse_global,
                "msle": msle_global,
                "r2": r2_global,
                "r2log": r2log_global
            })
            metrics_local.update({
                "mse": mse_label,
                "msle": msle_label,
                "r2": r2_label,
                "r2log": r2log_label
            })

        if classification_metrics:
            if transform_flavour is not None:
                yhat, yobs = self.transform_predictions_any(
                    yhat=yhat,
                    yobs=yobs,
                    nc=nc,
                    flavour=transform_flavour
                )
            score_auc_global = auc_global(y_hat=[yhat], y_obs=[yobs])
            prec_global, rec_global, tp_global, tn_global, fp_global, fn_global = pr_global(
                y_hat=[yhat], y_obs=[yobs]
            )
            score_auc_label = auc_label(
                y_hat=[yhat], y_obs=[yobs], labels=[labels], labels_unique=labels_unique
            )
            prec_label, rec_label, tp_label, tn_label, fp_label, fn_label = pr_label(
                y_hat=[yhat], y_obs=[yobs], labels=[labels], labels_unique=labels_unique
            )
            metrics_global.update({
                "auc": score_auc_global,
                "prec": prec_global,
                "rec": rec_global,
                "tp": tp_global,
                "tn": tn_global,
                "fp": fp_global,
                "fn": fn_global
            })
            metrics_local.update({
                "auc": score_auc_label,
                "prec": prec_label,
                "rec": rec_label,
                "tp": tp_label,
                "tn": tn_label,
                "fp": fp_label,
                "fn": fn_label
            })

        return {
            "global": metrics_global,
            "local": metrics_local
        }

    def evaluate_custom_any(
            self,
            yhat,
            yobs,
            nc,
            labels=None,
            labels_unique=None,
            classification_metrics: bool = True,
            regression_metrics: bool = False,
            transform_flavour: str = None
    ):
        """
        Obtain custom evaluation metrics for classification task.

        Ignores labels as samples are not structured by labels (ie one sample contains observations on all labels.

        :param yhat:
        :param yobs:
        :param nc:
        :param labels:
        :param transform_flavour:
        :return:
        """
        return self._evaluate_custom_any(
            yhat=yhat,
            yobs=yobs,
            nc=nc,
            classification_metrics=classification_metrics,
            regression_metrics=regression_metrics,
            transform_flavour=transform_flavour,
            labels=labels,
            labels_unique=labels_unique
        )

    def predict(self, batch_size: int = 128, save_embeddings: bool = False):
        """
        Predict labels on test data.

        :param batch_size: Batch size for evaluation.
        :return:
        """
        self.model.eval()  # Put the model in evaluation mode

        # Create a DataLoader for the test data
        test_data = TensorDataset(torch.from_numpy(self.x_test), torch.from_numpy(self.covariates_test))
        test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
        self.test_loader = test_loader
        all_outputs = []

        use_covariates = self.model.has_covariates if hasattr(self.model, 'has_covariates') else True
        with torch.no_grad():  # Disable gradient computation
            for batch in test_loader:
                x, covariates = batch
                datatype = next(self.model.parameters()).dtype
                x = x.to(self.device, dtype=datatype)
                covariates = covariates.to(self.device, dtype=datatype)
                x, covariates = x.to(self.device), covariates.to(self.device)

                # Perform the forward pass
                outputs = self.model(x, covariates) if use_covariates else self.model(x)

                all_outputs.append(outputs.cpu().numpy())  # Transfer outputs back to CPU and convert to numpy array

        self.predictions = np.concatenate(all_outputs)

    def predict_any(
            self,
            x,
            covar,
            batch_size: int = 128
    ):
        """ Predict labels on any data.

        :param batch_size: Batch size for evaluation.
        :return:
        """
        return self.model.predict(
            x=(x, covar),
            batch_size=batch_size,
            verbose=0
        )

    def transform_predictions_any(
            self,
            yhat,
            yobs,
            nc,
            flavour="10x_cd8_v1"
    ):
        """ Transform model predictions and ground truth labels on test data.

        Transform predictions and self.y_test

            - "10x_cd8" Use this setting to transform the real valued output of a network trained with MSE loss
                into probability space by using the bound/unbound classifier published with the 10x data set:
                An antigen is bound if it has (1) at least 10 counts and (2) at least 5 times more counts
                than the highest observed negative control and (3) is the highest count pMHC.
                Requires negative controls to be defined during reading.

        :param flavour: Type of transform to use, see function description.
        :return:
        """
        if flavour == "10x_cd8_v1":
            if self.model_hyperparam["loss"] not in ["mse", "msle", "poisson"]:
                raise ValueError("Do not use transform_predictions with flavour=='10x_cd8_v1' on a model fit "
                                 "with a loss that is not mse, msle or poisson.")

            if nc.shape[1] == 0:
                raise ValueError("Negative controls were not set, supply these during data reading.")

            predictions_new = np.zeros(yhat.shape)
            idx_bound_predictions = [np.where(np.logical_and(
                np.logical_and(x > 10., np.max(x) == x),  # At least 10 UMIs and is maximum element of cell.
                x > 5. * np.max(nc[i, :])
                # At least 5x as many UMIs as highest negative control UMI count in cell.
            ))[0] for i, x in enumerate(yhat)]
            for i, j in enumerate(idx_bound_predictions):
                if len(j) > 0:
                    predictions_new[i, j[-1]] = 1.  # Chose last label if two labels are called.
            yhat = predictions_new

            y_test_new = np.zeros(yobs.shape)
            idx_bound_y = [np.where(np.logical_and(
                np.logical_and(x > 10., np.max(x) == x),  # At least 10 UMIs and is maximum element of cell.
                x > 5. * np.max(nc[i, :])
                # At least 5x as many UMIs as highest negative control UMI count in cell.
            ))[0] for i, x in enumerate(yobs)]
            for i, j in enumerate(idx_bound_y):
                if len(j) > 0:
                    y_test_new[i, j[-1]] = 1.  # Chose last label if two labels are called.
            yobs = y_test_new
        else:
            raise ValueError(f"flavour {flavour} not recognized")

        return yhat, yobs

    def transform_predictions(
            self,
            flavour="10x_cd8_v1"
    ):
        """ Transform model predictions and ground truth labels on test data.

        Transform predictions and self.y_test

            - "10x_cd8" Use this setting to transform the real valued output of a network trained with MSE loss
                into probability space by using the bound/unbound classifier published with the 10x data set:
                An antigen is bound if it has (1) at least 10 counts and (2) at least 5 times more counts
                than the highest observed negative control and (3) is the highest count pMHC.
                Requires negative controls to be defined during reading.

        :param flavour: Type of transform to use, see function description.
        :return:
        """
        if flavour == "10x_cd8_v1":
            if self.model_hyperparam["loss"] not in ["mse", "msle", "poisson"]:
                raise ValueError("Do not use transform_predictions with flavour=='10x_cd8_v1' on a model fit "
                                 "with a loss that is not mse, msle or poisson.")

            if self.nc_test.shape[1] == 0:
                raise ValueError("Negative controls were not set, supply these during data reading.")

            predictions_new = np.zeros(self.predictions.shape)
            idx_bound_predictions = [np.where(np.logical_and(
                np.logical_and(x > 10., np.max(x) == x),  # At least 10 UMIs and is maximum element of cell.
                x > 5. * np.max(self.nc_test[i, :])  # At least 5x as many UMIs as highest negative control UMI count in cell.
            ))[0] for i, x in enumerate(self.predictions)]
            for i, j in enumerate(idx_bound_predictions):
                if len(j) > 0:
                    predictions_new[i, j[-1]] = 1.  # Chose last label if two labels are called.
            self.predictions = predictions_new

            y_test_new = np.zeros(self.y_test.shape)
            idx_bound_y = [np.where(np.logical_and(
                np.logical_and(x > 10., np.max(x) == x),  # At least 10 UMIs and is maximum element of cell.
                x > 5. * np.max(self.nc_test[i, :])  # At least 5x as many UMIs as highest negative control UMI count in cell.
            ))[0] for i, x in enumerate(self.y_test)]
            for i, j in enumerate(idx_bound_y):
                if len(j) > 0:
                    y_test_new[i, j[-1]] = 1.  # Chose last label if two labels are called.
            self.y_test = y_test_new
        else:
            raise ValueError(f"flavour {flavour} not recognized")

    def save_results(
            self,
            fn
    ):
        """ Save training history, test loss and test predictions.

        Will generate the following files:

            - fn+"history.pkl": training history dictionary
            - fn+"evaluations.npy": loss on test data
            - fn+"evaluations_custom.npy": loss on test data

        :param self:
        :param fn: Path and file name prefix to write to.
        :param save_labels: Whether to save ground truth labels. Use this for saving disk space.
        :return:
        """
        with open(fn + "_history.pkl", 'wb') as f:
            pickle.dump(self.history, f)
        with open(fn + "_evaluations.pkl", 'wb') as f:
            pickle.dump(self.evaluations, f)
        with open(fn + "_evaluations_custom.pkl", 'wb') as f:
            pickle.dump(self.evaluations_custom, f)
        if self.label_ids is not None:
            pd.DataFrame({"label": self.label_ids}).to_csv(fn + "_labels.csv")
        with open(fn + "_peptide_seqs_unique.pkl", 'wb') as f:
            pickle.dump(self.peptide_seqs_unique, f)

    def load_results(
            self,
            fn
    ):
        """ Load training history, test loss and test predictions.

        Will add the following entries to this instance from files:

            - fn+"history.pkl": training history dictionary
            - fn+"evaluations.npy": loss on test data
            - fn+"evaluations_custom.npy": loss on test data

        :param self:
        :param fn: Path and file name prefix to read from.
        :return:
        """
        with open(fn + "_history.pkl", 'rb') as f:
            self.history = pickle.load(f)
        with open(fn + "_evaluations.pkl", 'rb') as f:
            self.evaluations = pickle.load(f)
        with open(fn + "_evaluations_custom.pkl", 'rb') as f:
            self.evaluations_custom = pickle.load(f)

    def save_model_full(
            self,
            fn,
            reduce_size: bool = False,
            save_yhat: bool = True,
            save_train_data: bool = False
    ):
        """ Save model settings, data and weights.

        Saves all data necessary to perform full one-step model reloading with self.load_model().

        :param self:
        :param fn: Path and file name prefix to write to.
        :param reduce_size: Whether to save storage efficient, ie only elements that are absolutely necessary.
        :return:
        """
        self.save_model(fn=f'{fn}/model')
        self.save_estimator_args(fn=f'{fn}/args')
        self.save_data(
            fn=f'{fn}/data',
            train=save_train_data,
            test=True,
            reduce_size=reduce_size
        )
        if save_yhat:
            self.save_predictions(
                fn=fn,
                train=save_train_data,
                test=True
            )

    def save_model(
            self,
            fn
    ):
        """ Save model weights.

        :param self:
        :param fn: Path and file name prefix to write to. Will be suffixed with .tf to use tf weight saving.
        :return:
        """
        torch.save(self.model.state_dict(), fn)


    def load_model_full(
            self,
            fn: str = None,
            fn_settings: str = None,
            fn_data: str = None,
            fn_model: str = None
    ):
        """ Load entire model, this is possible if model weights, data and settings were stored.

        :param self:
        :param fn: Path and file name prefix to read model settings, data and model from.
        :param fn_settings: Path and file name prefix to read model settings from.
        :param fn_data: Path and file name prefix to read all fitting relevant data objects from.
        :param fn_model: Path and file name prefix to read model weights from.
        :param log_dir: Directory to save tensorboard callback to. Disabled if None. This is given to allow the user
             to choose between a new logging directory and the directory from the saved settings.

                - None if you want to enforce no logging.
                - "previous" if you want to use the directory saved in the settings.
                - any other string: This will be the new directory.
        :return:
        """

        if not (fn_settings and fn_data and fn_model) and not fn:
            raise ValueError("Please supply either fn or all of fn_settings, fn_data and fn_model.")
        if not fn_settings:
            self.load_data(fn=f'{fn}/data')
            self.load_model(
                fn=fn
            )
        else:
            self.load_data(fn=fn_data)
            self.load_model(
                fn_settings=fn_settings,
                fn_model=fn_model
            )


    def load_model(
            self,
            fn: str = None,
            fn_settings: str = None,
            fn_model: str = None
    ):
        """ Load model from .tf weights.

        :param self:
        :param fn: Path and file name prefix to read model settings from.
        :return:
        """

        if not fn_settings or not fn_model:
            self.load_model_settings(fn=f'{fn}/args')
            self.initialise_model_from_settings()
            self.model.load_state_dict(torch.load(f'{fn}/model'))  # Load the saved state dictionary
        else:
            self.load_model_settings(fn=fn_settings)
            self.initialise_model_from_settings()
            self.model.load_state_dict(torch.load(fn_model))  # Load the saved state dictionary
        

    def save_estimator_args(
            self,
            fn
    ):
        """ Save model settings.

        :param self:
        :param fn: Path and file name prefix to write to.
        :return:
        """
        # Save model args.
        with open(fn + "_model_args.pkl", 'wb') as f:
            pickle.dump(self.model.args, f)
        # Save model settings.
        with open(fn + "_model_settings.pkl", 'wb') as f:
            pickle.dump(self.model_hyperparam, f)
        # Save training settings.
        with open(fn + "_train_settings.pkl", 'wb') as f:
            pickle.dump(self.train_hyperparam, f)

    def load_model_settings(
            self,
            fn
    ):
        """ Load model settings.

        :param self:
        :param fn: Path and file name prefix to read weights from.
        :return:
        """
        # Load model settings.
        with open(fn + "_model_settings.pkl", 'rb') as f:
            self.model_hyperparam = pickle.load(f)
        # Load training settings.
        with open(fn + "_train_settings.pkl", 'rb') as f:
            self.train_hyperparam = pickle.load(f)

    def initialise_model_from_settings(self):
        """

        :return:
        """
        # Build model.
        if self.model_hyperparam["model"].lower() in ["bilstm", "bigru"]:
            self._build_sequential(
                split=self.model_hyperparam["split"],
                model=self.model_hyperparam["model"],
                topology=self.model_hyperparam["topology"],
                aa_embedding_dim=self.model_hyperparam["aa_embedding_dim"],
                depth_final_dense=self.model_hyperparam["depth_final_dense"],
                residual_connection=self.model_hyperparam["residual_connection"],
                dropout=self.model_hyperparam["dropout"],
                optimizer=self.model_hyperparam["optimizer"],
                lr=self.model_hyperparam["lr"],
                loss=self.model_hyperparam["loss"],
                label_smoothing=self.model_hyperparam["label_smoothing"],
                optimize_for_gpu=self.model_hyperparam["optimize_for_gpu"],
                dtype=self.model_hyperparam["dtype"],
                use_covariates=self.model_hyperparam["use_covariates"]
            )
        elif self.model_hyperparam["model"].lower() in ["sa", "selfattention"]:
            self.build_self_attention(
                residual_connection=self.model_hyperparam["residual_connection"],
                aa_embedding_dim=self.model_hyperparam["aa_embedding_dim"],
                attention_size=self.model_hyperparam["attention_size"],
                attention_heads=self.model_hyperparam["attention_heads"],
                optimizer=self.model_hyperparam["optimizer"],
                lr=self.model_hyperparam["lr"],
                loss=self.model_hyperparam["loss"],
                label_smoothing=self.model_hyperparam["label_smoothing"],
            )
        elif self.model_hyperparam["model"].lower() in ["conv", "convolutional"]:
            self.build_conv(
                activations=self.model_hyperparam["activations"],
                filter_widths=self.model_hyperparam["filter_widths"],
                filters=self.model_hyperparam["filters"],
                strides=self.model_hyperparam["strides"],
                pool_sizes=self.model_hyperparam["pool_sizes"],
                pool_strides=self.model_hyperparam["pool_strides"],
                batch_norm=self.model_hyperparam["batch_norm"],
                aa_embedding_dim=self.model_hyperparam["aa_embedding_dim"],
                depth_final_dense=self.model_hyperparam["depth_final_dense"],
                dropout=self.model_hyperparam["dropout"],
                optimizer=self.model_hyperparam["optimizer"],
                lr=self.model_hyperparam["lr"],
                loss=self.model_hyperparam["loss"],
                label_smoothing=self.model_hyperparam["label_smoothing"],
                dtype=self.model_hyperparam["dtype"]
            )
        elif self.model_hyperparam["model"].lower() in ["inception"]:
            self.build_inception(
                split=self.model_hyperparam["split"],
                n_filters_1x1=self.model_hyperparam["n_filters_1x1"],
                n_filters_out=self.model_hyperparam["n_filters_out"],
                n_hidden=self.model_hyperparam["n_hidden"],
                final_pool=self.model_hyperparam["final_pool"],
                residual_connection=self.model_hyperparam["residual_connection"],
                aa_embedding_dim=self.model_hyperparam["aa_embedding_dim"],
                depth_final_dense=self.model_hyperparam["depth_final_dense"],
                dropout=self.model_hyperparam["dropout"],
                optimizer=self.model_hyperparam["optimizer"],
                lr=self.model_hyperparam["lr"],
                loss=self.model_hyperparam["loss"],
                label_smoothing=self.model_hyperparam["label_smoothing"],
                dtype=self.model_hyperparam["dtype"]
            )
        elif self.model_hyperparam["model"].lower() in ["linear"]:
            self.build_linear(
                aa_embedding_dim=self.model_hyperparam["aa_embedding_dim"],
                optimizer=self.model_hyperparam["optimizer"],
                lr=self.model_hyperparam["lr"],
                loss=self.model_hyperparam["loss"],
                label_smoothing=self.model_hyperparam["label_smoothing"],
                dtype=self.model_hyperparam["dtype"]
            )
        elif self.model_hyperparam["model"].lower() in ["noseq"]:
            self.build_noseq(
                optimizer=self.model_hyperparam["optimizer"],
                lr=self.model_hyperparam["lr"],
                loss=self.model_hyperparam["loss"],
                label_smoothing=self.model_hyperparam["label_smoothing"],
                dtype=self.model_hyperparam["dtype"]
            )
        else:
            assert False

    def save_weights_tonumpy(
            self,
            fn
    ):
        """ Save model weights to pickled list of numpy arrays.

        :param fn: Path and file name prefix to write to.
        :return:
        """
        weights = self.model.get_weights()
        with open(fn + "_weights.pkl", 'wb') as f:
            pickle.dump(weights, f)

    def load_weights_asnumpy(
            self,
            fn
    ):
        """ Load model weights.

        :param fn: Path and file name prefix to write to.
        :return: List of model weights as numpy arrays.
        """
        with open(fn + "_weights.pkl", 'rb') as f:
            weights = pickle.load(f)
        return weights

    def save_data(
            self,
            fn,
            train: bool,
            test: bool,
            reduce_size: bool = False
    ):
        """ Save train and test data.

        :param fn: Path and file name prefix to write all fitting relevant data objects to.
        :param reduce_size: Whether to save storage efficient, ie only elements that are absolutely necessary.
        :return:
        """
        if train:
            if not reduce_size:
                scipy.sparse.save_npz(
                    matrix=scipy.sparse.csr_matrix(np.reshape(self.x_train, [self.x_train.shape[0], -1])),
                    file=fn + "_x_train.npz"
                )
            np.save(arr=self.x_train.shape, file=fn + "_x_train_shape.npy")
            if not reduce_size and self.covariates_train.shape[1] > 0:
                if not isinstance(self.covariates_train, scipy.sparse.csr_matrix):
                    covariates_train = scipy.sparse.csr_matrix(np.reshape(
                        self.covariates_train,
                        [self.covariates_train.shape[0], -1]
                    ))
                else:
                    covariates_train = self.covariates_train
                scipy.sparse.save_npz(matrix=covariates_train, file=fn + "_covariates_train.npz")
            np.save(arr=self.covariates_train.shape, file=fn + "_covariates_train_shape.npy")
            if not reduce_size:
                if not isinstance(self.y_train, scipy.sparse.csr_matrix):
                    y_train = scipy.sparse.csr_matrix(self.y_train)
                else:
                    y_train = self.y_train
                scipy.sparse.save_npz(matrix=y_train, file=fn + "_y_train.npz")
            np.save(arr=self.y_train.shape, file=fn + "_y_train_shape.npy")
            if not reduce_size and self.nc_train is not None and self.nc_train.shape[1] > 0:
                if not isinstance(self.nc_train, scipy.sparse.csr_matrix):
                    nc_train = scipy.sparse.csr_matrix(self.nc_train)
                else:
                    nc_train = self.nc_train
                scipy.sparse.save_npz(matrix=nc_train, file=fn + "_nc_train.npz")
            if self.nc_train is not None:
                np.save(arr=self.nc_train.shape, file=fn + "_nc_train_shape.npy")
            else:
                np.save(arr=np.array([None]), file=fn + "_nc_train_shape.npy")
            np.save(arr=self.clone_train, file=fn + "_clone_train.npy")
            if self.peptide_seqs_train is not None:
                pd.DataFrame({"antigen": self.peptide_seqs_train}).to_csv(fn + "_peptide_seqs_train.csv")

        if self.x_test is not None and test:
            if not reduce_size:
                scipy.sparse.save_npz(
                    matrix=scipy.sparse.csr_matrix(np.reshape(self.x_test, [self.x_test.shape[0], -1])),
                    file=fn + "_x_test.npz"
                )
            np.save(arr=self.x_test.shape, file=fn + "_x_test_shape.npy")
            if not reduce_size and self.covariates_test.shape[1] > 0:
                if not isinstance(self.covariates_test, scipy.sparse.csr_matrix):
                    covariates_test = scipy.sparse.csr_matrix(np.reshape(
                        self.covariates_test,
                        [self.covariates_test.shape[0], -1]
                    ))
                else:
                    covariates_test = self.covariates_test
                scipy.sparse.save_npz(matrix=covariates_test, file=fn + "_covariates_test.npz")
            np.save(arr=self.covariates_test.shape, file=fn + "_covariates_test_shape.npy")
            if not reduce_size:
                if not isinstance(self.y_test, scipy.sparse.csr_matrix):
                    y_test = scipy.sparse.csr_matrix(self.y_test)
                else:
                    y_test = self.y_test
                scipy.sparse.save_npz(matrix=y_test, file=fn + "_y_test.npz")
            np.save(arr=self.y_test.shape, file=fn + "_y_test_shape.npy")
            if not reduce_size and self.nc_test is not None and self.nc_test.shape[1] > 0:
                if not isinstance(self.nc_test, scipy.sparse.csr_matrix):
                    nc_test = scipy.sparse.csr_matrix(self.nc_test)
                else:
                    nc_test = self.nc_test
                scipy.sparse.save_npz(matrix=nc_test, file=fn + "_nc_test.npz")
            if self.nc_test is not None:
                np.save(arr=self.nc_test.shape, file=fn + "_nc_test_shape.npy")
            else:
                np.save(arr=np.array([None]), file=fn + "_nc_test_shape.npy")
            np.save(arr=self.clone_test, file=fn + "_clone_test.npy")
            if self.peptide_seqs_test is not None:
                pd.DataFrame({"antigen": self.peptide_seqs_test}).to_csv(fn + "_peptide_seqs_test.csv")

        pd.DataFrame({"antigen": self.peptide_seqs_unique}).to_csv(fn + "_peptide_seqs_unique.csv")
        self.save_idx(fn=fn)

    def load_data(
            self,
            fn
    ):
        """ Load train and test data.

        Note: Cryptic numpy pickle error is thrown if a csr_matrix containing only a single None is loaded.

        :param fn: Path and file name prefix to read all fitting relevant data objects from.
        :return:
        """
        x_train_shape = np.load(file=fn + "_x_train_shape.npy")
        if os.path.isfile(fn + "_x_train.npz"):
            self.x_train = np.reshape(np.asarray(
                scipy.sparse.load_npz(file=fn + "_x_train.npz").todense()
            ), x_train_shape)
        else:
            # Fill x with small all zero array to allow model loading.
            self.x_train = np.zeros(x_train_shape)
        covariates_train_shape = np.load(file=fn + "_covariates_train_shape.npy")
        if os.path.isfile(fn + "_covariates_train.npz") and covariates_train_shape[1] > 0:
            self.covariates_train = np.reshape(np.asarray(scipy.sparse.load_npz(
                file=fn + "_covariates_train.npz"
            ).todense()), covariates_train_shape)
        else:
            self.covariates_train = np.zeros(covariates_train_shape)
        self.x_len = x_train_shape[2]
        if os.path.isfile(fn + "_y_train_shape.npy"):
            y_train_shape = np.load(file=fn + "_y_train_shape.npy")
        else:
            y_train_shape = None
        if os.path.isfile(fn + "_y_train.npz"):
            self.y_train = np.asarray(scipy.sparse.load_npz(file=fn + "_y_train.npz").todense())
        else:
            if y_train_shape is not None:  # depreceated, remove
                self.y_train = np.zeros(y_train_shape)
        if os.path.isfile(fn + "_nc_train.npz"):
            self.nc_train = np.asarray(scipy.sparse.load_npz(file=fn + "_nc_train.npz").todense())
        else:
            self.nc_train = None
        self.clone_train = np.load(file=fn + "_clone_train.npy")

        if os.path.isfile(fn + "_x_test_shape.npy"):
            x_test_shape = np.load(file=fn + "_x_test_shape.npy")
            if os.path.isfile(fn + "_x_test.npz"):
                self.x_test = np.reshape(np.asarray(
                    scipy.sparse.load_npz(file=fn + "_x_test.npz").todense()
                ), x_test_shape)
            covariates_test_shape = np.load(file=fn + "_covariates_test_shape.npy")
            if os.path.isfile(fn + "_covariates_test.npz") and covariates_test_shape[1] > 0:
                self.covariates_test = np.reshape(np.asarray(scipy.sparse.load_npz(
                    file=fn + "_covariates_test.npz"
                ).todense()), covariates_test_shape)
            else:
                self.covariates_test = np.zeros(covariates_test_shape)
            if os.path.isfile(fn + "_y_test_shape.npy"):
                y_test_shape = np.load(file=fn + "_y_test_shape.npy")
            else:
                y_test_shape = None
            if os.path.isfile(fn + "_y_test.npz"):
                self.y_test = np.asarray(scipy.sparse.load_npz(file=fn + "_y_test.npz").todense())
            else:
                if y_test_shape is not None:  # depreceated, remove
                    self.y_test = np.zeros(y_test_shape)
            if os.path.isfile(fn + "_nc_test.npz"):
                self.nc_test = np.asarray(scipy.sparse.load_npz(file=fn + "_nc_test.npz").todense())
            else:
                self.nc_test = None
            self.clone_test = np.load(file=fn + "_clone_test.npy", allow_pickle=True)

        self.load_idx(fn=fn)

    def save_predictions(
            self,
            fn,
            train: bool,
            test: bool
    ):
        """ Save predictions.

        :param fn: Path and file name prefix to write all fitting relevant data objects to.
        :param reduce_size: Whether to save storage efficient, ie only elements that are absolutely necessary.
        :return:
        """
        if train:
            yhat_train = self.predict_any(x=self.x_train, covar=self.covariates_train)
            np.save(arr=yhat_train, file=fn + "_yhat_train.npy")

        if self.x_test is not None and test:
            yhat_test = self.predict_any(x=self.x_test, covar=self.covariates_test)
            np.save(arr=yhat_test, file=fn + "_yhat_test.npy")

    def serialize(self, filename):
        with open(filename, "wb") as file:
            pickle.dump(self, file)

    @staticmethod
    def deserialize(filename):
        with open(filename, "rb") as file:
            return pickle.load(file)
