from torch import nn
from torch.nn import functional as F
import torch
from models.base_model_output import AbstractModelOutput
from .info import InfoDist

class ConcatModelOutput(AbstractModelOutput):
    PROPERTIES = [
        "pooler_output",
        "modal_distribution",
    ]


class ResidualHead(nn.Module):

    def __init__(self, emb_size, n_inputs, normalize=True, activation=True):
        super().__init__()
        # Inputs: `n_inputs` vectors of dimension B x emb_size
        # Outputs: a vector of dimensions B x emb_size
        
        self.normalize = normalize
        self.activation = activation
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
        if self.normalize:
            vectors = [norm(vector) for norm, vector in zip(self.norms, vectors)]

        # Estimate Information Distribution
        info_dist = InfoDist(*vectors)

        # Activation
        if self.activation:
            vectors = [self.activation(vector) for vector in vectors]

        # Residual Connection
        output_vector = torch.sum(torch.stack(vectors, dim=-1), dim=-1)

        return ConcatModelOutput(
            pooler_output = output_vector,
            modal_distribution = info_dist
        )