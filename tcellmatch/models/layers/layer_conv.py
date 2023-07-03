# import tensorflow as tf
from typing import Union, Tuple

# !! TODO: not implemented in torch

import torch
import torch.nn as nn
from typing import Union, Tuple

class LayerConv(nn.Module):
    def __init__(
            self,
            in_channels: int,
            activation: str,
            filter_width: int,
            filters: int,
            stride: int,
            pool_size: int,
            pool_stride: int,
            batch_norm: bool = True,
            dropout: float = 0.0,
            dtype=torch.float32
    ):
        """

        Note: Addition of batch normalisation results in non-trainable weights in this layer.

        :param activation: Activation function. Refer to documentation of tf.keras.layers.Conv2D
        :param filter_width: Number of neurons per filter. Refer to documentation of tf.keras.layers.Conv2D
        :param filters: Number of filters / output channels. Refer to documentation of tf.keras.layers.Conv2D
        :param stride: Stride size for convolution on sequence. Refer to documentation of tf.keras.layers.Conv2D
        :param pool_size: Size of max-pooling, ie. number of output nodes to pool over.
        :param pool_stride: Stride of max-pooling.
        :param batch_norm: Whether to perform batch normalization.
        :param dropout: Dropout rate to use during training.
        :param input_shape:
        :param trainable:
        :param dtype:
        """
        super(LayerConv, self).__init__()
        
        self.activation = activation
        self.filter_width = filter_width
        self.filters = filters
        self.stride = stride
        self.pool_size = pool_size
        self.pool_stride = pool_stride
        self.batch_norm = batch_norm
        self.dropout = dropout
        self.in_channels = in_channels
        self.dtype = dtype
        
        self.sublayer_conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=self.filters,
            kernel_size=self.filter_width,
            stride=self.stride if self.stride is not None else None,
            padding=self.filter_width // 2 # 'same' padding
        )

        if self.batch_norm:
            self.sublayer_batchnorm = nn.BatchNorm1d(
                num_features=self.filters
            )
            
        if self.activation.lower() == 'relu':
            self.sublayer_act = nn.ReLU()
        elif self.activation.lower() == 'softmax':
            self.sublayer_act = nn.Softmax()
        elif self.activation.lower() == 'sigmoid':
            self.sublayer_act = nn.Sigmoid()
        # add more activation options if needed

        if self.dropout > 0:
            self.sublayer_dropout = nn.Dropout(p=self.dropout)
        print("pool params: size", self.pool_size, "stride", self.pool_stride)
        if self.pool_size is not None:
            self.sublayer_pool = nn.MaxPool1d(
                kernel_size=self.pool_size,
                stride=self.pool_stride if self.pool_stride is not None else self.pool_size,
                padding=self.pool_size // 2  # 'same' padding
            )

    def forward(self, x):
        """ 
        Forward pass through layer.

        :param x: input tensor
        :param training: Whether forward pass is in context of training or prediction: Use drop-out only during
            training.
        :return: output tensor
        """
        x = self.sublayer_conv(x)
        if self.batch_norm:
            x = self.sublayer_batchnorm(x, training=self.training)
        x = self.sublayer_act(x)
        # if self.dropout > 0 and training:
        #     x = self.sublayer_dropout(x)
        if self.pool_size is not None:
            x = self.sublayer_pool(x)
        return x

