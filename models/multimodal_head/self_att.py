from torch import nn, Tensor
from typing import Optional, List
import torch
import math
from torch.distributions import Categorical
from models.base_model_output import AbstractModelOutput
import numpy as np
from . import ResidualHead, ConcatHead

class ModalOutput(AbstractModelOutput):
    PROPERTIES = [
        "hidden_states",
        "pooler_output",
    ]

class AttentionModelOutput(AbstractModelOutput):
    PROPERTIES = [
        "hidden_states",
        "pooler_output",
        "modal_distribution",
    ]


class SelfAttention(nn.Module):

    def __init__(self, emb_size, n_inputs, num_heads=8, dropout: float = 0.1) -> None:
        super().__init__()

        self.norms = nn.ModuleList([nn.LayerNorm(emb_size) for _ in range(n_inputs)])

        #! Self Attention Block 
        self.self_attn = nn.MultiheadAttention(emb_size, 
                                               batch_first=True,
                                               num_heads=num_heads, 
                                               dropout=0)
        self.dropout1 = nn.Dropout(dropout)

        # self.combination_head = ResidualHead(emb_size, n_inputs, normalize=False, activation=False)
        self.combination_head = ConcatHead(emb_size, n_inputs)

        # # Feedforward Block
        # self.linear1 = nn.Linear(emb_size, dim_feedforward)
        # self.dropout = Dropout(dropout)
        # self.linear2 = Linear(dim_feedforward, d_model, **factory_kwargs)
        # self.norm2 = LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        # self.dropout2 = Dropout(dropout)


    def __self_attention_block(self, x: Tensor) -> Tensor:
        out = self.self_attn(x, x, x)
        
        return self.dropout1(out[0])
    

    def forward(self, *encoders_outputs):

        vectors = [o.last_hidden_state for o in encoders_outputs]
        
        # Each vector should be of the shape B x L_n x emb_size
        batch_size, _, emb_size = vectors[0].shape

        for idx, vector in enumerate(vectors):
            assert len(vector.shape) == 3, f"Unexpected shape {vector.shape} for input vector, instead of ({batch_size}, -1, {emb_size})"
            assert vector.size(0) == batch_size, f"Batch size for {idx}-th input vector is not matching {batch_size}"
            assert vector.size(2) == emb_size, f"Embedding dimension for {idx}-th vector is not matching {emb_size}"

        vectors = [norm(vector) for norm, vector in zip(self.norms, vectors)]
        source_lens = np.cumsum([0]+[v.size(1) for v in vectors])

        hidden_states = torch.cat(vectors, dim=1)
        attention_output = self.__self_attention_block(hidden_states)

        hidden_states = hidden_states + attention_output

        modal_outputs = []
        for b,e in zip(source_lens, source_lens[1:]):
            modal_emb = hidden_states[:, b:e]
            modal_outputs.append(ModalOutput(
                hidden_states=modal_emb,
                pooler_output=torch.mean(modal_emb, axis=1)
            ))

        return self.combination_head(*modal_outputs)
    