import torch
from torch import nn
from torch.nn.init import xavier_uniform_, zeros_
import numpy as np


class Dense(nn.Linear):
    def __init__(
        self,
        in_features,
        out_features,
        bias=True,
        activation=None,
        dropout=None,
        weight_init=xavier_uniform_,
        weight_gain=1.0,
        bias_init=zeros_,
        precision=torch.float64,
    ):
        self.weight_init = weight_init
        self.weight_gain = weight_gain
        self.bias_init = bias_init

        super(Dense, self).__init__(in_features, out_features, bias, dtype=precision)

        self.activation = activation
        self.dropout = nn.Dropout(dropout) if dropout else dropout

    def reset_parameters(self) -> None:
        self.weight_init(self.weight, self.weight_gain)
        if self.bias is not None:
            self.bias_init(self.bias)

    def forward(self, x):
        x = super(Dense, self).forward(x)
        if self.activation:
            x = self.activation(x)
        if self.dropout:
            x = self.dropout(x)
        return x


class RadialBesselLayer(nn.Module):
    def __init__(self, n_radial=16, cutoff=5.0, device=None):
        super(RadialBesselLayer, self).__init__()

        self.inv_cutoff = 1.0 / cutoff
        self.frequencies = nn.Parameter(
            torch.tensor(
                np.arange(1, n_radial + 1) * np.pi, dtype=torch.float64, device=device
            ),
            requires_grad=False,
        )

    def forward(self, distances):
        d_scaled = distances * self.inv_cutoff
        d_scaled = d_scaled.unsqueeze(-1)
        return torch.sin(self.frequencies * d_scaled)


class PolynomialCutoff(nn.Module):
    def __init__(self, cutoff=5.0, p=9):
        super(PolynomialCutoff, self).__init__()
        self.register_buffer("cutoff", torch.FloatTensor([cutoff]))
        self.register_buffer("p", torch.FloatTensor([p]))

    def forward(self, distances):
        d = distances / self.cutoff
        cutoffs = (
            1
            - 0.5 * (self.p + 1) * (self.p + 2) * d.pow(self.p)
            + self.p * (self.p + 2) * d.pow(self.p + 1)
            - 0.5 * self.p * (self.p + 1) * d.pow(self.p + 2)
        )
        cutoffs *= (distances < self.cutoff).float()
        return cutoffs


class ShellProvider(nn.Module):
    def __init__(self, cutoff=5.0):
        super(ShellProvider, self).__init__()
        self.cutoff = cutoff

    def forward(self, R, N, NM=None):
        B, A, _ = R.size()
        idx_m = torch.arange(B, device=R.device, dtype=torch.long)[:, None, None]
        R_neigh = R[idx_m, N[:, :, :], :]

        # distance vector (B, A, N, 3)
        V = R_neigh - R[:, :, None, :]

        # distances (B, A, N)
        D = torch.norm(V, 2, 3, dtype=torch.float64)

        if NM is not None:
            tmp_dist = torch.zeros_like(D)
            tmp_dist[NM != 0] = D[NM != 0]
            D = tmp_dist

        if self.cutoff is not None:
            within_cutoff = D < self.cutoff
            if NM is not None:
                within_cutoff[NM == 0] = False

            # determine the required number of neighbors for cutoff
            NC = torch.zeros((B, A), dtype=int)
            D_temp = [[[] for _ in range(A)] for _ in range(B)]
            V_temp = [[[] for _ in range(A)] for _ in range(B)]
            N_temp = [[[] for _ in range(A)] for _ in range(B)]
            NM_temp = [[[] for _ in range(A)] for _ in range(B)]
            for i in range(B):
                for j in range(A):
                    num_neigh = within_cutoff[i, j].sum()
                    NC[i, j] = num_neigh
                    D_temp[i][j] = D[i, j, within_cutoff[i, j]]
                    V_temp[i][j] = V[i, j, within_cutoff[i, j]]
                    N_temp[i][j] = N[i, j, within_cutoff[i, j]]
                    NM_temp[i][j] = torch.tensor([1] * num_neigh)
            n = NC.max()

            # don't determine number of neighbors for cutoff, just pad to max_n_neighbors
            # n = N.size(2)

            # fill tensors with values
            D = torch.zeros((B, A, n), device=R.device)
            V = torch.zeros((B, A, n, 3), device=R.device)
            N = torch.zeros((B, A, n), device=R.device, dtype=torch.long)
            NM = torch.zeros((B, A, n), device=R.device)
            for i in range(B):
                for j in range(A):
                    D[i, j, : NC[i, j]] = D_temp[i][j]
                    V[i, j, : NC[i, j]] = V_temp[i][j]
                    N[i, j, : NC[i, j]] = N_temp[i][j]
                    NM[i, j, : NC[i, j]] = NM_temp[i][j]

        return D, V, N, NM


class ScaleShift(nn.Module):
    def __init__(self, mean, stdev):
        super(ScaleShift, self).__init__()
        self.register_buffer("mean", mean)
        self.register_buffer("stddev", stdev)

    def forward(self, input):
        y = input * self.stddev + self.mean
        return y
