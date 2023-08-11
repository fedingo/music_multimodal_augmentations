from torch import nn
from torch.nn import functional as F
import torch
from .info import InfoDist
from models.base_model_output import AbstractModelOutput

class ConcatModelOutput(AbstractModelOutput):
    PROPERTIES = [
        "pooler_output",
        "modal_distribution",
    ]

class ConcatHead(nn.Module):

    def __init__(self, emb_size, n_inputs):
        super().__init__()
        # Inputs: `n_inputs` vectors of dimension B x emb_size
        # Outputs: a vector of dimensions B x emb_size
        
        self.projections = nn.ModuleList([nn.Linear(emb_size, emb_size) for _ in range(n_inputs)])
        self.norms = nn.ModuleList([nn.LayerNorm(emb_size) for _ in range(n_inputs)])
        self.activation = F.relu

    def forward(self, *encoders_outputs):
        # We extract as input the pooler vectors (1 per sample)
        vectors = [o.pooler_output for o in encoders_outputs]
                             
        # Each vector should be of the shape B x emb_size
        batch_size, emb_size = vectors[0].shape

        for idx, vector in enumerate(vectors):
            assert len(vector.shape) == 2, f"Unexpected shape {vector.shape} for input vector, instead of ({batch_size}, {emb_size})"
            assert vector.size(0) == batch_size, f"Batch size for {idx}-th input vector is not matching {batch_size}"
            assert vector.size(1) == emb_size, f"Embedding dimension for {idx}-th vector is not matching {emb_size}"

        # Normalization
        vectors = [norm(vector) for norm, vector in zip(self.norms, vectors)]
        # Activation
        vectors = [self.activation(vector) for vector in vectors]
        # Projections
        vectors = [proj(vector) for proj, vector in zip(self.projections, vectors)]

        # Estimate Information Distribution
        info_dist = InfoDist(*vectors)
        # Concatenation of n_inputs
        output_vector = torch.sum(torch.stack(vectors, dim=-1), dim=-1)

        return ConcatModelOutput(
            pooler_output = output_vector,
            modal_distribution = info_dist
        )