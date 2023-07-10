# TcellMatch: Predicting T-cell to epitope specificity.

TcellMatch is a collection of models to predict antigen specificity of single T cells based on CDR3 sequences and other 
single cell modalities, such as RNA counts and surface protein counts. 
As labeled training data, either defined CDR3-antigen pairs from data bases such as IEDB or VDJdb are used, or 
pMHC multimer "stained" single cells in which pMHC binding is used as a label indicating specificity to the loaded 
peptide.

This package can be used to train such models and contains the model code, data loaders and 
grid search helper functions.
Accompanying paper: https://www.embopress.org/doi/full/10.15252/msb.20199416

# Installation
This package can be locally installed via pip by first cloning a copy of this repository into you local directory
and then running `pip install -e .` at the top level of this clone's directory.

# Archiectures
Tcellmatch contains several architectures for predicting T cell specificity. They are detailed below.

## BiLSTM

- Input Layer: n x (alpha/beta chain dimension) x CDR3 sequence length x one-hot encoding for each amino acid; here, the shape is `n x 1 x 40 x 26`

- Hidden Layers:
    1. **Embedding Layer**: An optional 1x1 convolutional layer which is used to create lower-dimensional embeddings of one-hot encoded amino acids before the sequence model is applied. The size of the embedding can be configured using the `aa_embedding_dim` parameter.

        Input Shape: (batch_size, sequence_length, aa_embedding_dim)

        Output Shape: (batch_size, sequence_length, embedding_dim)

    2. **Bi-directional Recurrent Layers**: A sequence of BiLSTM or BiGRU layers (configured via the `model` parameter), with a user-specified number of layers and dimensions. These layers process the sequence data and provide capability to capture complex temporal dependencies.

        Input Shape: (batch_size, sequence_length, embedding_dim)

        Output Shape: (batch_size, sequence_length, hidden_size)

    3. **Reshaping Layer**: The output from the BiLSTM/BiGRU layers is reshaped to be 2D in preparation for the fully connected layers, and non-sequence covariates, if provided, are concatenated with the sequence-derived representations.

        Input Shape: (batch_size, sequence_length, hidden_size)

        Output Shape: (batch_size, hidden_size * 2 \[+ hidden_size * 2 if split\] + num_covariates)

    4. **Dense Layers**: A user-specified number of final fully connected layers (`depth_final_dense`) are used for the final task-specific prediction.

        Input Shape: (batch_size, hidden_size * 2 \[+ hidden_size * 2 if split\] + num_covariates)

        Output Shape: (batch_size, labels_dim)

- Activation: ReLU or Softmax depending on task

## Self-Attention

- Input Layer: n x (alpha/beta chain dimension) x CDR3 sequence length x one-hot encoding for each amino acid; here, the shape is ``n x 1 x 40 x 26``
- Hidden Layers:
    1. **Embedding Layer**: An optional 1x1 convolutional layer which is used to create lower-dimensional embeddings of one-hot encoded amino acids before the sequence model is applied. The size of the embedding can be configured using the `aa_embedding_dim` parameter.

        Input Shape: (batch_size, sequence_length, aa_embedding_dim)

        Output Shape: (batch_size, sequence_length, attention_size)

    2. **Reshaping Layer**: The output from the self-attention layers is reshaped to be 2D in preparation for the fully connected layers, and non-sequence covariates, if provided, are concatenated with the sequence-derived representations

        Input Shape: (batch_size, sequence_length, attention_size)

        Output Shape: (batch_size, sequence_length * attention_size + num_covariates)

    3. **Dense Layers**: A user-specified number of final fully connected layers (`depth_final_dense`) are used for the final task-specific prediction. 

        Input Shape: (batch_size, sequence_length * attention_size + num_covariates)

        Output Shape: (batch_size, labels_dim)

- Activation: ReLU or Softmax depending on task

## BiGRU
- Input Layer: n x (alpha/beta chain dimension) x CDR3 sequence length x one-hot encoding for each amino acid; here, the shape is ``n x 1 x 40 x 26``
- Hidden Layers:
    1. **Embedding Layer**: This layer is optional and its purpose is to create lower-dimensional embeddings of one-hot encoded amino acids before the sequence model is applied. The dimension of the embedding can be modified using the aa_embedding_dim parameter.

        Input Shape: (batch_size, sequence_length, aa_embedding_dim)

        Output Shape: (batch_size, sequence_length, embedding_dim)

    2. **Bi-directional Recurrent Layers**: A sequence of BiGRU layers, as specified by the model parameter. 

        Input Shape: (batch_size, sequence_length, embedding_dim)

        Output Shape: (batch_size, sequence_length, hidden_size)

    3. **Reshaping Layer**: The output from the BiGRU layers is reshaped into a 2D format to prepare for the fully connected layers. If non-sequence covariates are provided, they are concatenated with the sequence-derived representations at this point.

        Input Shape: (batch_size, sequence_length, hidden_size)

        Output Shape: (batch_size, hidden_size * 2 + hidden_size ∗ 2 if split + num_covariates)

    4. **Dense Layers**: The final task-specific prediction is generated using a user-defined number of fully connected layers (``depth_final_dense``).

        Input Shape: (batch_size, hidden_size * 2 +hidden_size ∗ 2 if split + num_covariates)

        Output Shape: (batch_size, labels_dim)

## CNN
- Input Layer: n x (alpha/beta chain dimension) x CDR3 sequence length x one-hot encoding for each amino acid; here, the shape is ``n x 1 x 40 x 26``
- Hidden Layers:
    1. **Embedding Layer**: This layer is optional and its purpose is to create lower-dimensional embeddings of one-hot encoded amino acids before the sequence model is applied. The dimension of the embedding can be modified using the aa_embedding_dim parameter.

        Input Shape: (batch_size, sequence_length, aa_embedding_dim)

        Output Shape: (batch_size, sequence_length, embedding_dim)

    2. **Convolutional Layers**: A sequence of convolutional layers. Each convolutional layer comprises an operation that applies a set of filters on the input data. The number of filters and the filter width can vary in each layer.

        For a given layer, let's consider the following parameters:

        ``filter_width``: The width of the filters in this layer.
        ``filters``: The number of filters (or output channels) in this layer.
        ``stride``: The stride size for the convolution operation.
        ``pool_size``: The size of the max-pooling window.
        ``pool_stride``: The stride for moving the pooling window.
        Using these parameters, the output sequence length after a convolution operation can be calculated using the following formula:

        output_sequence = ⌊(sequence_length - filter_width) / pool_stride + 1⌋

        If a max pooling operation is applied after the convolution, it further reduces the sequence length. The output sequence length after max pooling can be calculated using:

        output_sequence = ⌊(output_sequence - pool_size) / pool_stride + 1⌋

        We apply this function ``n_conv_layers`` times to get an ouput size of:

        Output Shape: (batch_size, output_sequence, filters)

    3. **Reshaping Layer**: The output from the BiGRU layers is reshaped into a 2D format to prepare for the fully connected layers. If non-sequence covariates are provided, they are concatenated with the sequence-derived representations at this point.

        Input Shape: (batch_size, output_sequence, filters)

        Output Shape: (batch_size, output_sequence *  filters + num_covariates)

    4. **Dense Layers**: The final task-specific prediction is generated using a user-defined number of fully connected layers (``depth_final_dense``).

        Input Shape: (batch_size, output_sequence * filters + num_covariates)

        Output Shape: (batch_size, labels_dim)



