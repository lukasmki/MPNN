from .base_classes import EmbedderBase
import torch


class MPNN_Embedder(EmbedderBase):
    def __init__(self):
        super(EmbedderBase).__init__()

    def forward(self, data: dict) -> dict:
        # create embeddings
        graph = {"node": torch.Tensor(), "edge": torch.Tensor()}

        # return graph
