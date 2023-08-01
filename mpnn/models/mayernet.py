import torch
from torch import nn
from torch.autograd import grad
from mpnn.models import MPNN
from mpnn.utils import swish


class MayerNet(nn.Module):
    def __init__(
        self,
        device,
        n_features=128,
        n_interax=3,
        resolution=20,
        activation=swish,
        cutoff=5.0,
        shell_cutoff=10.0,
    ):
        super(MayerNet, self).__init__()

        self.device = device

        self.chgbond = MPNN(
            device=self.device,
            n_features=n_features,
            n_interax=n_interax,
            resolution=resolution,
            activation=activation,
            cutoff=cutoff,
            shell_cutoff=shell_cutoff,
        )
        self.delta = MPNN(
            device=self.device,
            n_features=n_features,
            n_interax=n_interax,
            resolution=resolution,
            activation=activation,
            cutoff=cutoff,
            shell_cutoff=shell_cutoff,
        )

    def gather_neighbors(self, inputs, N):
        n_features = inputs.size()[-1]
        n_dim = inputs.dim()
        b, a, n = N.size()  # batch, atoms, neighbors size
        if n_dim == 3:
            N = N.view(-1, a * n, 1)  # B,A*N,1
            N = N.expand(-1, -1, n_features)
            out = torch.gather(inputs, dim=1, index=N)
            return out.view(b, a, n, n_features)  # B,A,N,n_features
        elif n_dim == 4:
            N = N.view(-1, a * n, 1, 1)  # B,A*N,1,1
            N = N.expand(-1, -1, 3, n_features)
            out = torch.gather(inputs, dim=1, index=N)
            return out.view(b, a, n, 3, n_features)  # B,A,N,3,n_features

    def sum_neighbors(self, x, mask, dim=2):
        dim_diff = x.dim() - mask.dim()
        for _ in range(dim_diff):
            mask = mask.unsqueeze(-1)
        x = x * mask
        out = torch.sum(x, dim=dim)
        return out

    def forward(self, data):
        CBO = self.chgbond(data)
        R, Z, N = CBO["R"], CBO["Z"], CBO["N"]
        Q, B, D = CBO["Ai"], CBO["Pij"], CBO["D"]

        qi = Q.repeat(1, 1, B.size(2))
        qi = qi.view(B.size())
        qj = self.gather_neighbors(Q, N).squeeze(-1)

        D_inv = torch.reciprocal(D)
        D_inv = torch.nan_to_num(D_inv, posinf=0.0)

        ATOMIC_TO_KCALMOL = 332.063711  # unit conv
        E_coul = (
            0.5
            * torch.sum(D_inv * qi * qj, dim=(1, 2)).unsqueeze(-1)
            * ATOMIC_TO_KCALMOL
        )
        E_bond = (
            -0.5
            * torch.sum(D_inv * B * B * 0.5, dim=(1, 2)).unsqueeze(-1)
            * ATOMIC_TO_KCALMOL
        )

        DELTA = self.delta(data)
        dE = torch.sum(DELTA["Ai"], dim=1)

        E = E_coul + E_bond + dE

        F = grad(
            E,
            R,
            grad_outputs=torch.ones_like(E),
            create_graph=False,
            retain_graph=True,
        )[0]
        F = -1.0 * F

        # output
        output = dict()
        output.update({"R": R, "Z": Z, "N": N})
        output.update({"E": E, "F": F})
        output.update({"Q": Q, "B": B, "D": D})
        return output
