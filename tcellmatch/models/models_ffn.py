# pylint: disable=E1101
# pylint: disable=C0302
# pylint: disable=C0301
# pylint: disable=C0114
# pylint: disable=C0103
from typing import List, Union

from tcellmatch.models.layers.layer_aa_embedding import LayerAaEmbedding
from tcellmatch.models.layers.layer_attention import LayerMultiheadSelfAttention
from tcellmatch.models.layers.layer_conv import LayerConv
from tcellmatch.models.layers.layer_bilstm import BiLSTM
import torch
import torch.nn as nn

class ModelBiRnn(nn.Module):

    forward_pass: list
    forward_pass_pep: list
    final_dense: list

    def __init__(
            self,
            labels_dim: int,
            input_shapes: tuple,
            model: str,
            topology: List[int],
            split: bool = False,
            aa_embedding_dim: int = 0,
            depth_final_dense: int = 1,
            out_activation: str = "linear",
            dropout: float = 0.0,
            one_hot_y: bool = False,
    ):
        """ BiLSTM-based feed-forward network with optional 1x1 convolutional embedding layer.

        Build the feed forward network as a tf.keras.Model object.
        The 1x1 convolutional embedding layer can be used to create a lower-dimensional embedding of
        one hot encoded amino acids before the sequence model is used.

        :param model: Layer type to use: {"bilstm", "bigru"}.
        :param topology: The depth of each bilstm layer (length of feature vector)
        :param dropout: drop out rate for bilstm.
        :param aa_embedding_dim: Dimension of the linear amino acid embedding, ie number of 1x1 convolutional filters.
            This is set to the input dimension if aa_embedding_dim==0.
        :param depth_final_dense: Number of final densely connected layers. They all have labels_dim number of units
            and relu activation functions, apart from the last, which has either linear or sigmoid activation,
            depending on out_probabilities.
        :param out_activation: Identifier of output activation function, this depends on
            assumption on labels and cost function:

            - "linear" for binding strength data measured as counts
            - "sigmoid" for binary binding events with multiple events per cell
            - "softmax" for binary binding events with one event per cell
        """
        super(ModelBiRnn, self).__init__()
        self.args = {
            "labels_dim": labels_dim,
            "input_shapes": input_shapes,
            "model": model,
            "aa_embedding_dim": aa_embedding_dim,
            "depth_final_dense": depth_final_dense,
            "out_activation": out_activation,
            "dropout": dropout,
            "split": split,
        }
        # return_sequences=False means we only return the state of last cell in LSTM
        self.labels_dim = labels_dim
        self.input_shapes = input_shapes
        self.model = model
        self.topology = topology
        self.aa_embedding_dim = aa_embedding_dim
        self.depth_final_dense = depth_final_dense
        self.out_activation = out_activation
        self.dropout = dropout
        self.split = split
        self.run_eagerly = False
        self.x_len = input_shapes[4]
        self.embed = None
        self.bi_layers = nn.ModuleList()
        self.bi_peptide_layers = nn.ModuleList()
        self.linear_layers = nn.ModuleList()

        input_dim = (input_shapes[0], input_shapes[1], input_shapes[2])
        input_covar_shape = (input_shapes[3],)
        self.has_covariates = input_shapes[3] != 0
        # Optional amino acid embedding:
        if aa_embedding_dim is not None:
            self.embed = LayerAaEmbedding(
                    shape_embedding=self.aa_embedding_dim,
                    input_shape=input_dim
                )
        # Split input into tcr and peptide
        if all(w == topology[0] for w in topology):
            if self.model.lower() == "bilstm":
                self.bi_layers.append(
                    nn.LSTM(
                        input_size=input_dim[-1],
                        hidden_size=topology[0],
                        num_layers = len(topology),
                        dropout=0,
                        bias=True,
                        batch_first=True,
                        bidirectional=True
                    )
                )
                if split:
                    self.bi_peptide_layers.append(
                        nn.LSTM(
                            input_size=input_dim[-1],
                            hidden_size=topology[0],
                            num_layers = len(topology),
                            dropout=0,
                            bias=True,
                            batch_first=True,
                            bidirectional=True
                        )
                    )
            elif self.model.lower() == "bigru":
                self.bi_layers.append(
                    nn.GRU(
                        input_size=input_dim[-1],
                        hidden_size=topology[0],
                        num_layers = len(topology),
                        dropout=0,
                        bias=True,
                        bidirectional=True
                    )
                )
                if split:
                    self.bi_peptide_layers.append(
                        nn.GRU(
                            input_size=input_dim[-1],
                            hidden_size=topology[0],
                            num_layers = len(topology),
                            dropout=0,
                            bias=True,
                            bidirectional=True
                        )
                    )

        else:
        # if True:
            raise ValueError('Hidden layers must be of the same dimension')

        for i in range(self.depth_final_dense):
            # calculate dimensions
            # last_layer = self.bi_layers[-1].lstm if self.model.lower() == "bilstm" else self.bi_layers[-1]
            if split:
                # 2 peptide x 2 for bidirectionality
                in_shape = self.bi_layers[-1].hidden_size * 4 + input_covar_shape[-1]
            else:
                in_shape = self.bi_layers[-1].hidden_size * 2 + input_covar_shape[-1]


            self.linear_layers.append(nn.Linear(
                in_features=in_shape if i == 0 else self.labels_dim,
                out_features=self.labels_dim,
                bias=True
            ))
            # we don't want softmax if we're predicting bindings for each
            if i == self.depth_final_dense - 1 and one_hot_y:
                self.linear_layers.append(nn.Softmax(dim=1))

    def forward(
            self,
            x : torch.Tensor,
            covar: torch.Tensor | None = None,
            save_embeddings: bool = False,
            fn: str = 'bilstm_embeddings',
        ):
        # do this in function to avoid always passing the same object
        if covar is None:
            covar = torch.Tensor([[]])
        x = torch.squeeze(x, dim=1)
        x = 2 * (x - 0.5)
        x = self.embed(x)
        if self.split:
            pep = x[:, self.x_len:, :]
            x = x[:, :self.x_len, :]  # TCR sequence from here on.
        for layer in self.bi_layers:
            # Check if the layer is LSTM or GRU
            if isinstance(layer, nn.LSTM):
                lstm_output = layer(x)
                output, (h_n, c_n) = lstm_output
            elif isinstance(layer, nn.GRU):
                gru_output = layer(x)
                output, h_n = gru_output

            # Use the output at the last time step for both directions
            x = output[:, -1, :]
        if len(x.size()) > 2:
            x = x[:, -1, :]
        for layer in self.bi_peptide_layers:
            pep = layer(pep)

        if self.split:
            x = torch.cat([x, pep], axis=1)
        
        # Optional concatenation of non-sequence covariates.
        if covar is not None and covar.shape[1] > 0:
            x = torch.cat([x, covar], axis=1)
        if save_embeddings:
            torch.save(x, f'{fn}.pt')
        for layer in self.linear_layers:
            x = layer(x)
        return x
    
    def get_embeddings(
        self,
        x : torch.Tensor,
        covar: torch.Tensor = None,
        save_embeddings: bool = False,
        fn: str = 'bilstm_embeddings',
        ):
        """
        Forward pass without linear layer.

        :param x: input tensor
        :param covar: input tensor
        :return: output tensor
        """
        # just the above method with the linear layers removed
        # do this in function to avoid always passing the same object
        # do this in function to avoid always passing the same object
        if covar is None:
            covar = torch.Tensor([[]])
        x = torch.squeeze(x, dim=1)
        x = 2 * (x - 0.5)
        x = self.embed(x)
        if self.split:
            pep = x[:, self.x_len:, :]
            x = x[:, :self.x_len, :]  # TCR sequence from here on.
        for layer in self.bi_layers:
            # Check if the layer is LSTM or GRU
            if isinstance(layer, nn.LSTM):
                lstm_output = layer(x)
                output, (h_n, c_n) = lstm_output
            elif isinstance(layer, nn.GRU):
                gru_output = layer(x)
                output, h_n = gru_output

            # Use the output at the last time step for both directions
            x = output[:, -1, :]
        if len(x.size()) > 2:
            x = x[:, -1, :]
        for layer in self.bi_peptide_layers:
            pep = layer(pep)

        if self.split:
            x = torch.cat([x, pep], axis=1)
        
        # Optional concatenation of non-sequence covariates.
        if covar is not None and covar.shape[1] > 0:
            x = torch.cat([x, covar], axis=1)

        return x

class ModelSa(nn.Module):

    forward_pass: list
    final_dense: list

    def __init__(
            self,
            labels_dim: int,
            input_shapes: tuple,
            attention_size: List[int],
            attention_heads: List[int],
            split: bool,
            residual_connection=False,
            aa_embedding_dim: Union[int, None] = 0,
            depth_final_dense: int = 1,
            out_activation: str = "linear",
            dropout: float = 0.0,
            one_hot_y: bool = True
    ):
        """ Self-attention-based feed-forward network.

        Build the feed forward network as a nn.Module object.

        :param dropout: drop out rate for lstm.
        :param attention_size: hidden size for attention, could be divided by attention_heads.
        :param attention_heads: number of heads in attention.
        :param residual_connection: apply residual connection or not.
        :param aa_embedding_dim: Dimension of the linear amino acid embedding, ie number of 1x1 convolutional filters.
            This is set to the input dimension if aa_embedding_dim==0.
        :param depth_final_dense: Number of final densely connected layers. They all have labels_dim number of units
            and relu activation functions, apart from the last, which has either linear or sigmoid activation,
            depending on out_probabilities.
        :param out_activation: Identifier of output activation function, this depends on
            assumption on labels and cost function:

            - "linear" for binding strength data measured as counts
            - "sigmoid" for binary binding events with multiple events per cell
            - "softmax" for binary binding events with one event per cell
        """
        super(ModelSa, self).__init__()
        self.args = {
            "labels_dim": labels_dim,
            "input_shapes": input_shapes,
            "attention_size": attention_size,
            "attention_heads": attention_heads,
            "aa_embedding_dim": aa_embedding_dim,
            "depth_final_dense": depth_final_dense,
            "out_activation": out_activation,
            "dropout": dropout,
            "split": split
        }
        self.input_shapes = input_shapes
        self.labels_dim = labels_dim
        self.attention_size = attention_size
        self.attention_heads = attention_heads
        self.residual_connection = residual_connection
        self.aa_embedding_dim = aa_embedding_dim
        self.depth_final_dense = depth_final_dense
        self.out_activation = out_activation
        self.dropout = dropout
        self.split = split
        assert not split, "not implemented"
        self.run_eagerly = False
        self.x_len = input_shapes[4]
        self.embed_attention_layers = nn.ModuleList()
        self.linear_layers = nn.ModuleList()
        self.has_covariates = input_shapes[3] != 0

        tcr_shape = (input_shapes[0], input_shapes[1], input_shapes[2])

        if aa_embedding_dim is not None:
            self.embed_attention_layers.append(
                LayerAaEmbedding(
                    shape_embedding=self.aa_embedding_dim,
                    input_shape=tcr_shape
                )
            )

        # Self-attention layers.
        for i, (w, n) in enumerate(zip(self.attention_size, self.attention_heads)):
            self.embed_attention_layers.append(
                LayerMultiheadSelfAttention(
                    width_embedding=w,
                    n_heads=n,
                    residual_connection=self.residual_connection,
                    attention_dropout=self.dropout,
                    name="sa_" + str(i),
                    input_shape=tcr_shape
                )
            )

        # Linear Layers
        for i in range(self.depth_final_dense):
            # input_shape = input_tcr.shape[-1] * input_tcr.shape[-2] + input_covar_shape[-1]
            input_shape = tcr_shape[-1] * tcr_shape[-2] + input_shapes[3]
            self.linear_layers.append(nn.Linear(
                in_features=input_shape if i == 0 else self.labels_dim,
                out_features=self.labels_dim,
                bias=True
            ))
            if i == self.depth_final_dense - 1 and one_hot_y:
                self.linear_layers.append(nn.Softmax(dim=1))

    def forward(self, x, input_covar=None):
        if input_covar is None:
            input_covar = torch.Tensor([[]])
        x = torch.squeeze(x, dim=1)
        x = 2 * (x - 0.5)
        for layer in self.embed_attention_layers:
            x = layer(x)
        x = x.view(-1, x.size(1) * x.size(2))

        # Optional concatenation of non-sequence covariates.
        if input_covar is not None and input_covar.shape[-1] > 0:
            x = torch.cat([x, input_covar], axis=1)

        for layer in self.linear_layers:
            x = layer(x)
        return x

    def get_embeddings(self, x, input_covar=None):
        """
        Forward pass without linear layer.

        :param x: input tensor
        :param covar: input tensor
        :return: output tensor
        """
        # forward method w/o linear layers
        x = torch.squeeze(x, dim=1)
        x = 2 * (x - 0.5)
        for layer in self.embed_attention_layers:
            x = layer(x)
        x = x.view(-1, x.size(1) * x.size(2))

        # Optional concatenation of non-sequence covariates.
        if input_covar is not None and input_covar.shape[-1] > 0:
            x = torch.cat([x, input_covar], axis=1)
        return x

class ModelConv(nn.Module):
    """ A layer class that implements sequence convolution.

        Instances of this class can be used as layers in the context of nn Models.
        This layer implements convolution and pooling. Uses the following sequence:

        convolution -> batch normalisation -> activation -> drop-out -> pooling
        """
    def __init__(
            self,
            labels_dim: int,
            n_conv_layers: int,
            input_shapes: tuple,
            split: bool = False,
            activations: List[str] | None = None,
            filters: List[int] | None = None,
            filter_widths: List[int] | None = None,
            strides: List[int] | None = None,
            pool_sizes: None | List[int] = None,
            pool_strides: None | List[int] = None,
            batch_norm: bool = True,
            aa_embedding_dim: int | None = 0,
            depth_final_dense: int = 1,
            out_activation: str = "linear",
            dropout: float = 0.0,
            one_hot_y: bool = False
    ):
        """ Convolution-based feed-forward network.

        Build the feed forward network as a torch.nn object.

        :param activations: Activation functions by hidden layers.
            Refer to documentation of tf.keras.layers.Activation
        :param filters: The width of the each hidden layer or number of filters if convolution.
            Refer to documentation of tf.keras.layers.Conv2D
        :param filter_widths: The width of filters if convolutional layer, otherwise ignored.
            Refer to documentation of tf.keras.layers.Conv2D
        :param strides: The strides if convolutional layer, otherwise ignored.
            Refer to documentation of tf.keras.layers.Conv2D
        :param pool_sizes: Size of max-pooling, ie. number of output nodes to pool over, by layer.
            Refer to documentation of tf.keras.layers.MaxPool2D:pool_size
        :param pool_strides: Stride of max-pooling by layer.
            Refer to documentation of tf.keras.layers.MaxPool2D:strides
        :param aa_embedding_dim: Dimension of the linear amino acid embedding, ie number of 1x1 convolutional filters.
            This is set to the input dimension if aa_embedding_dim==0.
        :param depth_final_dense: Number of final densely connected layers. They all have labels_dim number of units
            and relu activation functions, apart from the last, which has either linear or sigmoid activation,
            depending on out_probabilities.
        :param out_activation: Identifier of output activation function, this depends on
            assumption on labels and cost function:

            - "linear" for binding strength data measured as counts
            - "sigmoid" for binary binding events with multiple events per cell
            - "softmax" for binary binding events with one event per cell
        """
        super(ModelConv, self).__init__()
        if not activations:
            activations = ["relu"] * n_conv_layers
        if not filters:
            filters = [1] * n_conv_layers
        if not filter_widths:
            filter_widths = [1] * n_conv_layers
        if not strides:
            strides = [1] * n_conv_layers
        if not pool_sizes:
            pool_sizes = [None] * n_conv_layers
        if not pool_strides:
            pool_strides = [None] * n_conv_layers

        self.args = {
            "labels_dim": labels_dim,
            "input_shapes": input_shapes,
            "aa_embedding_dim": aa_embedding_dim,
            "out_activation": out_activation,
            "activations": activations,
            "filters": filters,
            "filter_widths": filter_widths,
            "strides": strides,
            "pool_strides": pool_strides,
            "pool_sizes": pool_sizes,
            "batch_norm": batch_norm,
            "depth_final_dense": depth_final_dense,
            "dropout": dropout,
            "split": split
        }
        self.labels_dim = labels_dim
        self.input_shapes = input_shapes
        self.aa_embedding_dim = aa_embedding_dim
        self.out_activation = out_activation
        self.activations = activations
        self.filters = filters
        self.filter_widths = filter_widths
        self.strides = strides
        self.pool_strides = pool_strides if pool_strides is not None else [None for x in self.activations]
        self.pool_sizes = pool_sizes if pool_sizes is not None else [None for x in self.activations]
        self.batch_norm = batch_norm
        self.depth_final_dense = depth_final_dense
        self.dropout = dropout
        self.split = split
        assert not split, "not implemented"
        self.run_eagerly = False
        self.x_len = input_shapes[4]
        self.has_covariates = input_shapes[3] != 0
        self.conv_layers = nn.ModuleList()
        self.linear_layers = nn.ModuleList()
        input_dim = (input_shapes[0], input_shapes[1], input_shapes[2])
        input_covar_shape = (input_shapes[3],)
        if self.aa_embedding_dim is not None:
            self.embed = LayerAaEmbedding(
                shape_embedding=self.aa_embedding_dim,
                input_shape=input_dim
            )
        assert n_conv_layers == len(self.activations), "n_conv_layers must be equal to the length of activations"
        assert n_conv_layers == len(self.filters), "n_conv_layers must be equal to the length of filters"
        assert n_conv_layers == len(self.filter_widths), "n_conv_layers must be equal to the length of filter_widths"
        assert n_conv_layers == len(self.strides), "n_conv_layers must be equal to the length of strides"
        assert n_conv_layers == len(self.pool_sizes), "n_conv_layers must be equal to the length of pool_sizes"
        assert n_conv_layers == len(self.pool_strides), "n_conv_layers must be equal to the length of pool_strides"

        for i, (a, f, fw, s, psize, pstride) in enumerate(zip(
                self.activations,
                self.filters,
                self.filter_widths,
                self.strides,
                self.pool_sizes,
                self.pool_strides,
        )):
            self.conv_layers.append(LayerConv(
                in_channels=input_dim[-2] if i == 0 else self.filters[i - 1],
                activation=a,
                filter_width=fw,
                filters=f,
                stride=s,
                pool_size=psize,
                pool_stride=pstride,
                batch_norm=self.batch_norm,
                dropout=self.dropout
            ))

        for i in range(self.depth_final_dense):
            # calculate dimensions
            if i == 0:
                last_conv = self.conv_layers[-1]
                in_shape = input_dim[-1]
                if last_conv.pool_size is not None:
                    for j in range(len(self.conv_layers)):
                        ps = self.conv_layers[j].pool_size
                        stride = self.conv_layers[j].pool_stride if self.conv_layers[j].pool_stride is not None \
                             else self.conv_layers[j].pool_size
                        in_shape = (in_shape + 2 * (ps // 2) - (ps - 1) - 1) // stride + 1
                        if j == len(self.conv_layers) - 1:
                            in_shape *= self.conv_layers[j].filters
                in_shape += input_covar_shape[-1]
            else:
                in_shape = self.labels_dim

            self.linear_layers.append(nn.Linear(
                in_features=in_shape if i == 0 else self.labels_dim,
                out_features=self.labels_dim,
                bias=True
            ))

            # add activation
            if i == self.depth_final_dense - 1 and one_hot_y:
                self.linear_layers.append(nn.Softmax())
            elif i < self.depth_final_dense - 1:
                self.linear_layers.append(nn.ReLU())
        
    def forward(self, x, covar):
        """
        Forward pass through layer.

        :param x: input tensor
        :param covar: input tensor
        :return: output tensor
        """
        if covar is None:
            covar = torch.Tensor([[]])
        x = torch.squeeze(x, dim=1)
        x = 2 * (x - 0.5)
        if hasattr(self, 'embed') and self.embed is not None:
            x = self.embed(x)
        for i, convolve in enumerate(self.conv_layers):
            x = convolve(x)
        x = torch.reshape(x, (-1, x.shape[1] * x.shape[2]))
        if covar.shape[1] > 0:
            x = torch.concat([x, covar], axis=1)
        for layer in self.linear_layers:
            x = layer(x)

        return x

    def get_embeddings(self, x, covar : torch.Tensor | None = None):
        """
        Forward pass without linear layer.

        :param x: input tensor
        :param covar: input tensor
        :return: output tensor
        """
        if covar is None:
            covar = torch.Tensor([[]])
        x = torch.squeeze(x, dim=1)
        x = 2 * (x - 0.5)
        if hasattr(self, 'embed') and self.embed is not None:
            x = self.embed(x)
        
        for i, layer in enumerate(self.conv_layers):
            x = layer(x)
        x = torch.reshape(x, (-1, x.shape[1] * x.shape[2]))
        if covar.shape[1] > 0:
            x = torch.concat([x, covar], axis=1)
        return x


class ModelLinear:
    def __init__(self):
        raise NotImplementedError("ModelLinear not implemented in torch")

class ModelNoseq:
    def __init__(self):
        raise NotImplementedError("ModelNoseq not implemented in torch")

# # !! Still uses tf
# class ModelLinear:

#     def __init__(
#             self,
#             labels_dim: int,
#             input_shapes: tuple,
#             aa_embedding_dim: Union[int, None] = 0,
#             out_activation: str = "linear"
#     ):
#         """ Linear densely conected feed-forward network.

#         Build the feed forward network as a tf.keras.Model object.

#         :param aa_embedding_dim: Dimension of the linear amino acid embedding, ie number of 1x1 convolutional filters.
#             This is set to the input dimension if aa_embedding_dim==0.
#         :param out_activation: Identifier of output activation function, this depends on
#             assumption on labels and cost function:

#             - "linear" for binding strength data measured as counts
#             - "sigmoid" for binary binding events with multiple events per cell
#             - "softmax" for binary binding events with one event per cell
#         """
#         self.args = {
#             "labels_dim": labels_dim,
#             "aa_embedding_dim": aa_embedding_dim,
#             "out_activation": out_activation,
#             "input_shapes": input_shapes
#         }
#         self.aa_embedding_dim = aa_embedding_dim
#         self.labels_dim = labels_dim
#         self.out_activation = out_activation
#         self.run_eagerly = False
#         self.x_len = input_shapes[4]

#         input_tcr = tf.keras.layers.Input(
#             shape=(input_shapes[0], input_shapes[1], input_shapes[2]),
#             name='input_tcr'
#         )
#         input_covar = tf.keras.layers.Input(
#             shape=(input_shapes[3]),
#             name='input_covar'
#         )

#         x = input_tcr
#         x = tf.squeeze(x, axis=[1])  # squeeze out chain
#         x = 2 * (x - 0.5)
#         # Amino acid embedding layer.
#         if aa_embedding_dim is not None:
#             x = LayerAaEmbedding(
#                 shape_embedding=self.aa_embedding_dim,
#                 squeeze_2D_sequence=False
#             )(x)
#         x = tf.reshape(x, [-1, x.shape[1] * x.shape[2]])
#         if input_covar.shape[1] > 0:
#             x = tf.concat([x, input_covar], axis=1)

#         # Linear layer.
#         output = tf.keras.layers.Dense(
#             units=self.labels_dim,
#             activation=out_activation,
#             use_bias=True,
#             kernel_initializer='glorot_uniform',
#             bias_initializer='zeros',
#             kernel_regularizer=None,
#             bias_regularizer=None,
#             activity_regularizer=None
#         )(x)

#         self.training_model = tf.keras.models.Model(
#             inputs=[input_tcr, input_covar],
#             outputs=output,
#             name='model_linear'
#         )

# # !! Still uses tf
# class ModelNoseq:

#     forward_pass: List[tf.keras.layers.Dense]

#     def __init__(
#             self,
#             labels_dim: int,
#             input_shapes: tuple,
#             out_activation: str = "linear",
#             n_layers: int = 3
#     ):
#         """ Densely conected feed-forward network that does not include sequence data.

#         Build the feed forward network as a tf.keras.Model object.

#         :param out_activation: Identifier of output activation function, this depends on
#             assumption on labels and cost function:

#             - "linear" for binding strength data measured as counts
#             - "sigmoid" for binary binding events with multiple events per cell
#             - "softmax" for binary binding events with one event per cell
#         """
#         self.args = {
#             "labels_dim": labels_dim,
#             "out_activation": out_activation,
#             "input_shapes": input_shapes,
#             "n_layers": n_layers
#         }
#         self.labels_dim = labels_dim
#         self.out_activation = out_activation
#         self.n_layers = n_layers
#         self.run_eagerly = False
#         self.x_len = input_shapes[4]

#         input_tcr = tf.keras.layers.Input(
#             shape=(input_shapes[0], input_shapes[1], input_shapes[2]),
#             name='input_tcr'
#         )
#         input_covar = tf.keras.layers.Input(
#             shape=(input_shapes[3]),
#             name='input_covar'
#         )
#         x = input_covar
#         # Linear layers.
#         for i in range(n_layers):
#             x = tf.keras.layers.Dense(
#                 units=self.labels_dim,
#                 activation="relu" if i < n_layers-1 else out_activation,
#                 use_bias=True,
#                 kernel_initializer='glorot_uniform',
#                 bias_initializer='zeros',
#                 kernel_regularizer=None,
#                 bias_regularizer=None,
#                 activity_regularizer=None
#             )(x)
#         output = x

#         self.training_model = tf.keras.models.Model(
#             inputs=[input_tcr, input_covar],
#             outputs=output,
#             name='model_noseq'
#         )
