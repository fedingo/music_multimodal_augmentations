from torch import nn
import torch
import math
from torch.distributions import Categorical
from models.base_model_output import AbstractModelOutput

class AttentionModelOutput(AbstractModelOutput):
    PROPERTIES = [
        "pooler_output",
        "modal_distribution",
        "modal_attention",
    ]


class SequenceAttention(nn.Module):

    def __init__(self, emb_size, n_inputs, num_heads=8) -> None:
        super().__init__()

        self.emb_size = emb_size
        self.cls_base = nn.Parameter(torch.empty(num_heads, emb_size))

        self.num_heads = num_heads
        self.head_size = emb_size // num_heads

        self.head_proj = nn.Parameter(torch.empty(num_heads, emb_size, self.head_size))
        self.head_bias = nn.Parameter(torch.empty(num_heads, 1, self.head_size))

        nn.init.uniform(self.cls_base)
        nn.init.kaiming_uniform_(self.head_proj, a=math.sqrt(5))
        bound = 1 / math.sqrt(self.head_size)
        nn.init.uniform_(self.head_bias, -bound, bound)


    def forward(self, *encoders_outputs):

        vectors = [o.last_hidden_state for o in encoders_outputs]
        
        # Each vector should be of the shape B x L_n x emb_size
        batch_size, _, emb_size = vectors[0].shape

        for idx, vector in enumerate(vectors):
            assert len(vector.shape) == 3, f"Unexpected shape {vector.shape} for input vector, instead of ({batch_size}, -1, {emb_size})"
            assert vector.size(0) == batch_size, f"Batch size for {idx}-th input vector is not matching {batch_size}"
            assert vector.size(2) == emb_size, f"Embedding dimension for {idx}-th vector is not matching {emb_size}"

        
        sequence = torch.cat(vectors, dim=1)
        attention = nn.functional.softmax(sequence@(self.cls_base.T), dim=1)
        # Batched matrix multiplication -> Output Shape: B x num_heads x emb_size
        cls_per_head = torch.bmm(attention.permute(0,2,1), sequence)

        # Heads Projection
        cls_out = torch.bmm(cls_per_head.permute(1,0,2), self.head_proj) + self.head_bias
        cls_out = cls_out.permute(1,0,2).contiguous().view(batch_size, emb_size)

        modal_len = [v.size(1) for v in vectors]
        breakpoints = [sum(modal_len[:i]) for i in range(len(modal_len))]

        assert sequence.size(1) == breakpoints[-1] + modal_len[-1], \
            "Partition does not reach the end of the distribution"

        modal_coeffs = []
        for start, length in zip(breakpoints, modal_len):
            modal_attention = attention[:, start:start+length, :]
            modal_coeff = torch.sum(modal_attention)/(batch_size*self.num_heads)
            modal_coeffs.append(modal_coeff)

        modal_attn = torch.stack(modal_coeffs)

         #? CLS Output Shape: B x emb_size
        return AttentionModelOutput(
            pooler_output = cls_out,
            modal_distribution = modal_attn,
        )
