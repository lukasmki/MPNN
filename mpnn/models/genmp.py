import torch
from torch import nn
from torch.autograd import grad
from mpnn.layers import Dense, RadialBesselLayer, PolynomialCutoff, ShellProvider


class GENMP(nn.Module):
    def __init__(
        self,
        device,
        n_features=128,
        n_interax=1,
        resolution=20,
        activation=nn.SiLU(),
        cutoff=5.0,
        shell_cutoff=10.0,
    ):
        super(GENMP, self).__init__()
        self.device = device

        self.n_features = n_features
        self.n_interax = n_interax
        self.activation = activation
        self.cutoff = cutoff

        self.atom_embedding = nn.Embedding(10, 128, padding_idx=0, dtype=torch.float64)
        self.pair_embedding = nn.Embedding(10, 128, padding_idx=0, dtype=torch.float64)
        self.shell = ShellProvider(cutoff=shell_cutoff)
        self.distance_expansion = RadialBesselLayer(
            resolution, cutoff=cutoff, device=device
        )
        self.iterations = nn.ModuleList(
            [
                MessagePassing(n_features, resolution, activation, cutoff)
                for _ in range(n_interax)
            ]
        )
        self.atomic_property = AtomicProperty(n_features, activation, dropout=None)
        self.pair_property = PairProperty(n_features, activation, dropout=None)

    def forward(self, data):
        R, Z, N, NM = data["R"], data["Z"], data["N"], data["NM"]
        D, V, N, NM = self.shell(R, N, NM)
        R.requires_grad_()
        n_batch, n_atoms, n_neigh = N.size()

        rbf = self.distance_expansion(D)

        # atomic embeddings
        a = self.atom_embedding(Z)  # B, A, nf

        # pair embeddings by multiplying atomic embeddings
        ai = a.repeat(1, 1, n_neigh)
        ai = ai.view((n_batch, n_atoms, n_neigh, self.n_features))
        tmp_n = N.view(-1, n_atoms * n_neigh, 1).expand(-1, -1, self.n_features)
        aj = torch.gather(a, dim=1, index=tmp_n)
        aj = aj.view((n_batch, n_atoms, n_neigh, self.n_features))
        p = ai * aj  # B, A, N, nf

        # message passing layers
        for i_interax in range(self.n_interax):
            a, p = self.iterations[i_interax](a, p, rbf, D, N, NM)

        # prediciton
        output = {"R": R, "Z": Z, "N": N, "NM": NM, "D": D, "V": V}

        atom_pred = self.atomic_property(a)
        pair_pred = self.pair_property(p)

        output.update({"Ai": atom_pred.squeeze(-1), "Pij": pair_pred.squeeze(-1)})

        # compute graph energy
        # reshape (B, A, A)
        A = torch.zeros((n_batch, n_atoms, n_atoms, 1), dtype=torch.float64)
        A[::, ~torch.eye(n_atoms, dtype=torch.bool)] = pair_pred[
            ::, torch.ones((n_atoms, n_neigh), dtype=torch.bool)
        ]
        A[::, torch.eye(n_atoms, dtype=torch.bool)] = atom_pred
        A = A.squeeze(-1)

        eigval, eigvec = torch.linalg.eig(A)
        graph_energy = torch.sum(eigval.real, dim=-1).unsqueeze(-1)

        # and the gradients
        graph_gradients = grad(
            graph_energy,
            R,
            grad_outputs=torch.ones_like(graph_energy),
            create_graph=False,
            retain_graph=True,
        )
        output.update(
            {
                "E(G)": graph_energy,
                "F(G)": graph_gradients,
                "E(val)": eigval,
                "E(vec)": eigvec,
            }
        )
        return output


class MessagePassing(nn.Module):
    def __init__(self, n_features, resolution, activation, cutoff):
        super(MessagePassing, self).__init__()

        self.phi_rbf = Dense(resolution, n_features, activation=None)
        self.cutoff_function = PolynomialCutoff(cutoff, p=9)

        self.phi_a = nn.Sequential(
            Dense(n_features, n_features, activation=activation),
            Dense(n_features, n_features, activation=None),
        )

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
        a = a + torch.sum(msij, 2)
        p = p + msij

        return a, p


class AtomicProperty(nn.Module):
    def __init__(self, n_features, activation, dropout=None):
        super(AtomicProperty, self).__init__()
        self.phi_atom = nn.Sequential(
            Dense(n_features, 128, activation=activation, dropout=dropout),
            Dense(128, 64, activation=activation, dropout=dropout),
            Dense(64, 1, activation=None, dropout=0.0),
        )

    def forward(self, a):
        return self.phi_atom(a)  # B, A, 1


class PairProperty(nn.Module):
    def __init__(self, n_features, activation, dropout=None):
        super(PairProperty, self).__init__()
        self.phi_pair = nn.Sequential(
            Dense(n_features, 128, activation=activation, dropout=dropout),
            Dense(128, 64, activation=activation, dropout=dropout),
            Dense(64, 1, activation=None, dropout=0.0),
        )

    def forward(self, msij):
        return self.phi_pair(msij)  # B,A,N,1
