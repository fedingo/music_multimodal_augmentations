import torch
from torch import nn
from typing import List


class SourceEmbedding(nn.Module):

    def __init__(self, emb_size, n_sources) -> None:
        super().__init__()
        
        self.n_sources = n_sources
        self.embedding = nn.Embedding(n_sources, emb_size)

    def forward(self, *vectors) -> List[torch.Tensor]:
        
        device = vectors[0].device
        assert len(vectors) == self.n_sources, "Unexpected amount of source vectors for Embedding Layer"

        result = []
        for idx, vector in enumerate(vectors):

            emb = self.embedding(torch.LongTensor([idx]).to(device)).squeeze()
            result.append(vector + emb)

        return result