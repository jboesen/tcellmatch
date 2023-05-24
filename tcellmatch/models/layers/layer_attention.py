import torch
import torch.nn as nn
import torch.nn.functional as F

class LayerMultiheadSelfAttention(nn.Module):
    def __init__(
            self,
            width_embedding: int,
            n_heads: int,
            residual_connection: bool,
            attention_dropout: float = 0.,
            name="sa",
            dtype=torch.float32,
            input_shape=None,
    ):
        super(LayerMultiheadSelfAttention, self).__init__()
        self.total_width_embedding = int(width_embedding * n_heads)
        self.n_heads = n_heads
        self.width_embedding = width_embedding
        self.qk_dropout = nn.Dropout(attention_dropout)
        self.residual_connection = residual_connection
        self.dtype = dtype
        self.output_shape = None

        # The q, k, v embedding layers
        self.q_embedding = nn.Linear(input_shape[-1], self.total_width_embedding, bias=False)
        self.k_embedding = nn.Linear(input_shape[-1], self.total_width_embedding, bias=False)
        self.v_embedding = nn.Linear(input_shape[-1], self.total_width_embedding, bias=False)

        # Final dense layer
        self.final_linear = nn.Linear(self.total_width_embedding, input_shape[-1])

        if self.residual_connection:
            self.layer_norm = nn.LayerNorm(input_shape[-1], eps=1e-6)

    def forward(self, inputs):
        # Project input in linear embedding using Q, K, V

        q = self.q_embedding(inputs)
        k = self.k_embedding(inputs)
        v = self.v_embedding(inputs)


        # Reshape q, k, v for multihead attention
        q = q.view(q.size(0), -1, self.n_heads, self.width_embedding).permute(0, 2, 1, 3)  # [batch_size, n_heads, seq_len, width_embedding]
        k = k.view(k.size(0), -1, self.n_heads, self.width_embedding).permute(0, 2, 1, 3)
        v = v.view(v.size(0), -1, self.n_heads, self.width_embedding).permute(0, 2, 1, 3)


        # Scale q
        q = q / (self.n_heads ** 0.5)

        # Calculate attention
        qk = torch.matmul(q, k.permute(0, 1, 3, 2))  # [batch_size, n_heads, seq_len, seq_len]
        qk = F.softmax(qk, dim=-1)
        # TODO: make this only if training
        if self.training:
            qk = self.qk_dropout(qk)


        # Compute the weighted value (context)
        qkv = torch.matmul(qk, v)  # [batch_size, n_heads, seq_len, width_embedding]

        qkv = qkv.permute(0, 2, 1, 3).contiguous()  # [batch_size, seq_len, n_heads, width_embedding]
        output = qkv.view(qkv.size(0), -1, self.total_width_embedding)  # [batch_size, seq_len, total_width_embedding]

        output = self.final_linear(output)

        if self.residual_connection:
            output = self.layer_norm(inputs + output)

        return output
