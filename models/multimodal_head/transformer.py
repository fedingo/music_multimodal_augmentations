from torch import nn
import torch
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
        "modal_outputs",
    ]


class Transformer(nn.Module):

    N_LAYERS = 1

    def __init__(self, emb_size, n_inputs, num_heads=8, dropout: float = 0.1) -> None:
        super().__init__()

        self.norms = nn.ModuleList([nn.LayerNorm(emb_size) for _ in range(n_inputs)])

        encoder_layer = nn.TransformerEncoderLayer(emb_size, num_heads, emb_size, dropout, batch_first=True, norm_first=True)
        encoder_norm = nn.LayerNorm(emb_size, eps=1e-5)
        self.encoder = nn.TransformerEncoder(encoder_layer, self.N_LAYERS, encoder_norm)

        # self.combination_head = ResidualHead(emb_size, n_inputs, normalize=False, activation=False)
        self.combination_head = ConcatHead(emb_size, n_inputs)

    def step(self, step):
        pass

    def forward(self, *encoders_outputs):

        vectors = [o.last_hidden_state for o in encoders_outputs]
        
        # Each vector should be of the shape B x L_n x emb_size
        batch_size, _, emb_size = vectors[0].shape

        for idx, vector in enumerate(vectors):
            assert len(vector.shape) == 3, f"Unexpected shape {vector.shape} for input vector, instead of ({batch_size}, -1, {emb_size})"
            assert vector.size(0) == batch_size, f"Batch size for {idx}-th input vector is not matching {batch_size}"
            assert vector.size(2) == emb_size, f"Embedding dimension for {idx}-th vector is not matching {emb_size}"

        # Main Encoder Job
        vectors = [norm(vector) for norm, vector in zip(self.norms, vectors)]
        hidden_states = torch.cat(vectors, dim=1)
        hidden_states = self.encoder(hidden_states)

        # Extract modal specific info
        modal_outputs = []
        source_lens = np.cumsum([0]+[v.size(1) for v in vectors])

        for b,e in zip(source_lens, source_lens[1:]):
            modal_emb = hidden_states[:, b:e]
            modal_outputs.append(ModalOutput(
                hidden_states=modal_emb,
                pooler_output=torch.mean(modal_emb, axis=1)
            ))

        return self.combination_head(*modal_outputs)


class Transformer1L(Transformer):
    N_LAYERS=1

class Transformer3L(Transformer):
    N_LAYERS=3
