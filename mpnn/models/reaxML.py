import torch
from torch import nn
from torch.autograd import grad
from mpnn.models import MPNN


class ReaxML(nn.Module):
    def __init__(
        self,
        device,
        chgbond=None,
        n_interax=3,
        n_features=128,
        resolution=20,
        activation=nn.SiLU(),
        cutoff=5.0,
        shell_cutoff=10.0,
    ):
        super(ReaxML, self).__init__()

        self.device = device

        if chgbond is not None:
            self.chgbond = torch.load(chgbond)
            for param in self.chgbond.parameters():
                param.requires_grad = False

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

        self.ctx = nn.ParameterDict(
            {
                # H-H
                "001-001-d0s": torch.tensor(157.5488, requires_grad=True),
                "001-001-pbe1": torch.tensor(-0.1661, requires_grad=True),
                "001-001-pbe2": torch.tensor(6.2500, requires_grad=True),
                # H-O
                "001-008-d0s": torch.tensor(216.7852, requires_grad=True),
                "001-008-pbe1": torch.tensor(-1.0000, requires_grad=True),
                "001-008-pbe2": torch.tensor(1.6492, requires_grad=True),
                # O-O
                "008-008-d0s": torch.tensor(122.9794, requires_grad=True),
                "008-008-pbe1": torch.tensor(1.0000, requires_grad=True),
                "008-008-pbe2": torch.tensor(0.1958, requires_grad=True),
            }
        )

    def get_batch_params(self, zi, zj):
        d0ij = torch.zeros(zi.size(0))
        pbe1 = torch.zeros(zi.size(0))
        pbe2 = torch.zeros(zi.size(0))
        for i in range(zi.size(0)):
            if zi[i] * zj[i] == 0:
                continue
            if zi[i] < zj[i]:
                param_key = f"{zi[i]:03}-{zj[i]:03}"
            else:
                param_key = f"{zj[i]:03}-{zi[i]:03}"
            d0ij[i] = self.ctx[param_key + "-d0s"]
            pbe1[i] = self.ctx[param_key + "-pbe1"]
            pbe2[i] = self.ctx[param_key + "-pbe2"]
        return d0ij, pbe1, pbe2

    def forward(self, data):
        CBO = self.chgbond(data)
        R, Z, N = CBO["R"], CBO["Z"], CBO["N"]
        Q, B, D = CBO["Ai"], CBO["Pij"], CBO["D"]

        # print(Q.shape, B.shape, D.shape)
        # > torch.Size([5, 6]) torch.Size([5, 6, 5]) torch.Size([5, 6, 5])

        n_batch, n_atoms, n_neigh = B.shape
        n_atoms = torch.sum(Z[0] != 0)

        E = torch.zeros((n_batch, 1))

        for iatom in range(n_atoms - 1):
            zi = Z[::, iatom]
            qi = Q[::, iatom]
            for jatom in range(iatom + 1, n_atoms):
                zj = Z[::, jatom]
                qj = Q[::, jatom]
                bij = torch.abs(B[::, jatom, iatom])
                dij = D[::, jatom, iatom]

                d0ij, pbe1, pbe2 = self.get_batch_params(zi, zj)

                # reaxff bond energy
                E_bond_ij = -d0ij * bij * torch.exp(pbe1 * (1 - torch.pow(bij, pbe2)))
                # E_coul_ij = qi * qj / dij * 332.063711
                E_bond_ij = (E_bond_ij) * (zi != 0) * (zj != 0)
                E += E_bond_ij.unsqueeze(-1)

        # F = grad(
        #     E,
        #     R,
        #     grad_outputs=torch.ones_like(E),
        #     create_graph=False,
        #     retain_graph=True,
        # )[0]
        # F = -1.0 * F

        # output
        output = dict()
        output.update({"R": R, "Z": Z, "N": N})
        output.update({"E": E})
        output.update({"Q": Q, "B": B, "D": D})
        return output
