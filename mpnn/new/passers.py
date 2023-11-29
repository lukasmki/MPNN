from .base_classes import MessagePasserBase
import torch


class MPNN_MessagePasser(MessagePasserBase):
    def __init__(self):
        super(MessagePasserBase).__init__()

    def forward(self, graph: dict) -> dict:
        # make messages

        # pass messages
        graph["node"] = graph["node"] + torch.sum(graph["edge"], axis=-2)

        return graph
