import torch
from torch import nn
from torch.autograd import grad
from mpnn.models import MPNN
from mpnn.layers import Dense, ShellProvider


class ReboML(nn.Module):
    def __init__(
        self,
        device,
        chgbond=None,
        chgbond_freeze=True,
        n_interax=3,
        n_features=128,
        resolution=20,
        activation=nn.SiLU(),
        cutoff=5.0,
        shell_cutoff=10.0,
    ):
        super(ReboML, self).__init__()

        self.device = device

        if chgbond is not None:
            self.chgbond = torch.load(chgbond)
        else:
            self.chgbond = MPNN(
                device=self.device,
                n_features=n_features,
                n_interax=n_interax,
                resolution=resolution,
                activation=activation,
                cutoff=cutoff,
                shell_cutoff=shell_cutoff,
            )
        if chgbond_freeze:
            for param in self.chgbond.parameters():
                param.requires_grad = False

        # self.att = nn.ModuleDict({k: Exponential() for k in ["1", "8", "64"]})
        # self.rep = nn.ModuleDict({k: Exponential() for k in ["1", "8", "64"]})

        self.shell = ShellProvider(cutoff=shell_cutoff)

        self.att = nn.Sequential(
            Dense(n_features, 128, activation=activation),
            Dense(128, 64, activation=activation),
            Dense(64, 1, activation=None),
        )

        self.rep = nn.Sequential(
            Dense(n_features, 128, activation=activation),
            Dense(128, 64, activation=activation),
            Dense(64, 1, activation=None),
        )

    def forward(self, data):
        R, Z, N, NM = data["R"], data["Z"], data["N"], data["NM"]
        R.requires_grad_()
        D, V, N, NM = self.shell(R, N, NM)

        CBO = self.chgbond(data)
        Q, B = CBO["Ai"], CBO["Pij"]
        GQ, GB = CBO["GAi"], CBO["GPij"]

        # print(Q.shape, B.shape, D.shape)
        # > torch.Size([5, 6]) torch.Size([5, 6, 5]) torch.Size([5, 6, 5])
        # n_batch, n_atoms, n_neigh = B.shape

        E_a = self.att(CBO["p"]).squeeze(-1)
        E_r = self.rep(CBO["p"]).squeeze(-1)

        E = E_r - B * E_a
        E = 0.5 * torch.sum(E, dim=(-1, -2))

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
        output.update({"E(R)": E_r, "E(A)": E_a})
        output.update({"Q": Q, "B": B, "D": D})
        output.update({"GQ": GQ, "GB": GB})
        return output


class Exponential(nn.Module):
    def __init__(self):
        super(Exponential, self).__init__()
        self.A = nn.Parameter(
            torch.tensor(150, dtype=torch.float32),
            requires_grad=True,
        )
        self.a = nn.Parameter(
            torch.tensor(0.1, dtype=torch.float32),
            requires_grad=True,
        )
        self.B = nn.Parameter(
            torch.tensor(150, dtype=torch.float32),
            requires_grad=True,
        )
        self.b = nn.Parameter(
            torch.tensor(0.1, dtype=torch.float32),
            requires_grad=True,
        )

    def forward(self, r):
        return self.A * torch.exp(-1.0 * self.a * r) + self.B * torch.reciprocal(r)
