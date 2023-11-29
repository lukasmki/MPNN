import torch
from torch import nn
from mpnn.layers import Dense, PolynomialCutoff


class MessagePassing(nn.Module):
    def __init__(self, n_features, resolution, activation, cutoff):
        super(MessagePassing, self).__init__()

        self.phi_rbf = Dense(resolution, n_features, activation=None)
        self.cutoff_function = PolynomialCutoff(cutoff, p=9)

        self.phi_a = nn.Sequential(
            Dense(n_features, n_features, activation=activation),
            Dense(n_features, n_features, activation=None),
        )

    def update(self, a, p, msij):
        a = a + torch.sum(msij, 2)
        p = p + msij
        return a, p

    def forward(self, a, p, rbf, D, N, NM):
        # radial basis
        rbf_msij = self.phi_rbf(rbf)  # B, A, N, nf
        rbf_msij = rbf_msij * self.cutoff_function(D).unsqueeze(-1)

        # map embeddings to features
        a_msij = self.phi_a(a)

        # copying atomic features for multiplication (B, A, N, nf)
        ai_msij = a_msij.repeat(1, 1, rbf_msij.size(2))
        ai_msij = ai_msij.view(rbf_msij.size())

        # reshaping of neighbor array (B, A, N, nf)
        b, a, n, nf = *N.size(), a_msij.size()[-1]
        tmp_n = N.view(-1, a * n, 1).expand(-1, -1, nf)

        # neighbor features (B, A, N, nf)
        aj_msij = torch.gather(a_msij, dim=1, index=tmp_n)
        aj_msij = aj_msij.view(b, a, n, nf)

        # symmetric messages (B, A, N, nf)
        msij = ai_msij * aj_msij * rbf_msij

        # neighbor mask
        msij = msij * NM.unsqueeze(-1)

        # feature update
        # Here is where NewtonNet's physics-based update system is inserted
        # Just do a standard message update instead
        a, p = self.update(a, p, msij)

        return a, p
