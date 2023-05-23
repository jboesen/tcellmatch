# pylint: disable=E1101
# pylint: disable=C0302
# pylint: disable=C0301
# pylint: disable=C0114
# pylint: disable=C0103
import torch
import torch.nn as nn

class LayerAaEmbedding(nn.Module):
    """ A layer class that implements amino acid embedding.

    Instances of this class can be used as layers in the context of tensorflow Models.
    This layer implements 1x1 convolutions to map a given amino acid embedding (such as one-hot) into a learnable
    new space of choosable dimensionality.

    """
    sublayer_conv2d: torch.nn.Conv2d
    fwd_pass: list

    def __init__(
            self,
            shape_embedding: int,
            squeeze_2D_sequence: bool,
            trainable: bool = True,
            dropout: float = 0.1,
            input_shape=None,
            # dtype=tf.float32
            dtype=torch.float32
    ):
        super().__init__()
        if shape_embedding < 0:
            raise ValueError("aa_embedding_dim has to be >0")
        self.dropout = dropout
        # self.input_shapes = input_shape
        self.sublayer_conv2d = None
        self.shape_embedding = shape_embedding if shape_embedding != 0 else input_shape[-1]
        # self.shape_embedding = shape_embedding
        # !! This is sketch
        self.sublayer_conv2d = torch.nn.Conv1d(
            in_channels=input_shape[-2],
            out_channels=input_shape[-2],
            # in_channels=40,
            # out_channels=40,
            kernel_size=1,
            stride=1,
            padding=0,
            dilation=1,
            groups=1,
            bias=True,
            padding_mode='zeros'
        )
    def forward(self, x):
        if self.shape_embedding is not None:
            x = self.sublayer_conv2d(x)
        return x