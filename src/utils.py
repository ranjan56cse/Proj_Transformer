import math
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F


# ==================================================
# Replicate blocks into a stack (Encoder/Decoder)
# ==================================================
def make_layer_stack(block, N=6) -> nn.ModuleList:
    """
    Method to replicate the existing block to N set of blocks
    :param block: class inherited from nn.Module, mainly it is the encoder or decoder part of the architecture
    :param N: the number of stack, in the original paper they used 6
    :return: a set of N blocks
    """
    block_stack = nn.ModuleList([copy.deepcopy(block) for _ in range(N)])
    return block_stack


# ==================================================
# Rotary Positional Embeddings (RoPE)
# ==================================================
class RoPE(nn.Module):
    def __init__(self, embed_dim):
        super(RoPE, self).__init__()
        if embed_dim % 2 != 0:
            raise ValueError("RoPE requires even embedding dimension.")
        self.embed_dim = embed_dim

    def forward(self, q, k):
        seq_len = q.size(-2)
        device = q.device

        position = torch.arange(seq_len, device=device).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.embed_dim, 2, device=device) * -(math.log(10000.0) / self.embed_dim)
        )

        sinusoid_inp = torch.einsum("i , j -> i j", position.squeeze(), div_term)
        sin, cos = sinusoid_inp.sin(), sinusoid_inp.cos()

        # expand to match batch/heads
        sin, cos = map(
            lambda t: t[None, None, :, :].repeat(q.size(0), q.size(1), 1, 1),
            (sin, cos)
        )

        def apply_rotary(x):
            x1, x2 = x[..., ::2], x[..., 1::2]
            return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)

        return apply_rotary(q), apply_rotary(k)


# ==================================================
# Relative Position Bias
# ==================================================
class RelativePosBias(nn.Module):
    def __init__(self, num_heads, max_distance=128):
        super(RelativePosBias, self).__init__()
        self.num_heads = num_heads
        self.max_distance = max_distance
        self.relative_bias = nn.Embedding(2 * max_distance - 1, num_heads)

    def forward(self, seq_len, device=None):
        context_position = torch.arange(seq_len, dtype=torch.long, device=device)[:, None]
        memory_position = torch.arange(seq_len, dtype=torch.long, device=device)[None, :]
        relative_position = memory_position - context_position  # (seq_len, seq_len)
        relative_position = relative_position.clamp(-self.max_distance + 1, self.max_distance - 1)
        relative_position = relative_position + self.max_distance - 1
        values = self.relative_bias(relative_position)  # (seq_len, seq_len, num_heads)
        return values.permute(2, 0, 1).unsqueeze(0)  # (1, num_heads, seq_len, seq_len)


# ==================================================
# Wrapper to switch between RoPE & Relative Pos Bias
# ==================================================
class PositionalEncoding(nn.Module):
    def __init__(self, method="rope", embed_dim=512, num_heads=8, max_seq_len=512):
        super(PositionalEncoding, self).__init__()
        self.method = method.lower()

        if self.method == "rope":
            self.encoding = RoPE(embed_dim)
        elif self.method == "rel_pos_bias":
            self.encoding = RelativePosBias(num_heads=num_heads, max_distance=max_seq_len)
        else:
            raise ValueError(f"Unknown positional encoding method: {method}")

    def forward(self, *args, **kwargs):
        return self.encoding(*args, **kwargs)


# ==================================================
# Multi-Head Attention with configurable Positional Encoding
# ==================================================
class MultiHeadAttention(nn.Module):

    def __init__(self, embed_dim=512, heads=8, pos_encoding="rope", max_seq_len=512):
        """
        Multi-Head Attention with support for RoPE or Relative Position Bias.
        :param embed_dim: embedding dimension
        :param heads: number of attention heads
        :param pos_encoding: "rope" or "rel_pos_bias"
        :param max_seq_len: max sequence length (for relative pos bias)
        """
        super(MultiHeadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.heads = heads
        self.d_k = int(self.embed_dim / self.heads)
        assert embed_dim % heads == 0, f"embed_dim ({embed_dim}) must be divisible by heads ({heads})"

        # Linear projections
        self.w_q = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.w_v = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.w_k = nn.Linear(self.embed_dim, self.embed_dim, bias=False)

        self.fc_out = nn.Linear(self.embed_dim, self.embed_dim)

        # Positional encoding
        self.pos_encoding_type = pos_encoding
        self.pos_encoding = PositionalEncoding(method=pos_encoding, embed_dim=embed_dim, num_heads=heads, max_seq_len=max_seq_len)

    def split_heads(self, x):
        batch_size, seq_len, embed_dim = x.size()
        x = x.view(batch_size, seq_len, self.heads, self.d_k)
        return x.transpose(1, 2)  # (batch, heads, seq_len, d_k)

    def forward(self, key, query, value, mask=None):
        batch_size = key.size(0)
        k_len, q_len, v_len = key.size(1), query.size(1), value.size(1)

        key = self.w_k(key)
        query = self.w_q(query)
        value = self.w_v(value)

        # Split heads
        key = self.split_heads(key)
        query = self.split_heads(query)
        value = self.split_heads(value)

        # Apply positional encoding
        if self.pos_encoding_type == "rope":
            query, key = self.pos_encoding(query, key)
            rel_bias = None
        elif self.pos_encoding_type == "rel_pos_bias":
            rel_bias = self.pos_encoding(seq_len=k_len, device=key.device)
        else:
            rel_bias = None

        # Scaled dot-product attention
        product = torch.einsum("bhqd,bhkd->bhqk", [query, key])  # (batch, heads, q_len, k_len)
        product = product / math.sqrt(self.d_k)

        # Add relative position bias if applicable
        if rel_bias is not None:
            product = product + rel_bias

        if mask is not None:
            product = product.masked_fill(~mask, float("-1e20"))

        scores = F.softmax(product, dim=-1)

        # Attention output
        output = torch.einsum("bhqk,bhkd->bhqd", [scores, value])  # (batch, heads, q_len, d_k)

        # Concatenate heads
        output = output.transpose(1, 2).contiguous().view(batch_size, q_len, self.heads * self.d_k)

        return self.fc_out(output)
