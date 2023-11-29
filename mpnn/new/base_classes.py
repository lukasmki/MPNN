import torch
import torch.nn as nn


class NetworkBase(nn.Module):
    def __init__(self, n_layers: int = 3):
        super(NetworkBase, self).__init__()
        self.n_layers = n_layers

        # submodules
        self.embedder = EmbedderBase()
        self.layers = nn.ModuleList([MessagePasserBase() for _ in range(n_layers)])
        self.predictor = PredictorBase()

    def forward(self, data: dict) -> dict:
        # compute graph embedding
        graph = self.embedder(data)

        # message passing
        for i_layer in range(self.n_layers):
            graph = self.layers[i_layer](graph)

        # output
        return self.predictor(graph)


class EmbedderBase(nn.Module):
    def __init__(self, pbc: list[bool] | None = None):
        super(EmbedderBase, self).__init__()
        self.pbc = pbc

    def forward(self, data: dict) -> dict:
        """Create graph representation"""
        raise NotImplementedError
        # graph = {"node": torch.Tensor()}
        # return graph


class MessagePasserBase(nn.Module):
    def __init__(self):
        super(MessagePasserBase, self).__init__()

    def forward(self, graph: dict, n: int = 0) -> dict:
        """Compute message passing given graph representation"""
        raise NotImplementedError
        # graph["node"] += graph["edge"]
        # return graph


class PredictorBase(nn.Module):
    def __init__(self, pbc: list[float] | None = None):
        super(PredictorBase, self).__init__()

    def forward(self, graph: dict) -> dict:
        """Compute output given final graph"""
        raise NotImplementedError


if __name__ == "__main__":
    test = NetworkBase()
    print(test)
