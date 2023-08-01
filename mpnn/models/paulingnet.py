import torch
from torch import nn
from torch.autograd import grad
from mpnn.layers import (
    Dense,
    RadialBesselLayer,
    PolynomialCutoff,
    ShellProvider,
    ScaleShift,
)
from mpnn.utils import swish


class PaulingNet(nn.Module):
    def __init__(
        self,
        device,
        n_features=128,
        n_interax=3,
        resolution=20,
        activation=swish,
        cutoff=5.0,
        shell_cutoff=10.0,
        normalizer=(0.0, 1.0),
    ):
        super(PaulingNet, self).__init__()
        self.device = device

        self.n_features = n_features
        self.n_interax = n_interax
        self.activation = activation
        self.cutoff = cutoff
        if normalizer:
            self.normalizer = normalizer
            self.inverse_normalize = ScaleShift(
                mean=torch.tensor(normalizer[0], device=device),
                stdev=torch.tensor(normalizer[1], device=device),
            )

        self.atom_embedding = nn.Embedding(10, 128, padding_idx=0)
        self.pair_embedding = nn.Embedding(10, 128, padding_idx=0)
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
        R, Z, N = data["R"], data["Z"], data["N"]
        AM, NM = data["AM"], data["NM"]
        R.requires_grad_()

        D, V, N, NM = self.shell(R, N, NM)
        n_batch, n_atoms, n_neigh = N.size()

        rbf = self.distance_expansion(D)

        # atomic embeddings
        a = self.atom_embedding(Z)  # B, A, nf
        q_dynamics = torch.zeros_like(a)  # B, A, nf
        q_latent = torch.zeros((a.size(0), a.size(1), 1), device=R.device)  # B, A, 1
        b_dynamics = torch.zeros(
            NM.size() + (self.n_features,), device=R.device
        )  # B, A, N, nf
        b_latent = torch.zeros_like(NM)  # B, A, N
        e_dynamics = torch.zeros_like(a)  # B, A, nf

        # message passing layers
        for i_interax in range(self.n_interax):
            a, q_dynamics, b_dynamics, e_dynamics, q_latent, b_latent = self.iterations[
                i_interax
            ](a, q_dynamics, b_dynamics, e_dynamics, q_latent, b_latent, rbf, D, N, NM)

        # prediciton
        output = {"R": R, "Z": Z, "N": N, "NM": NM, "D": D, "V": V}

        # latent
        b_latent = b_latent * (NM != 0)
        q_latent = q_latent * AM[..., None]
        q_latent = q_latent.squeeze(-1)
        output.update({"Q": q_latent, "B": b_latent})

        # energy/force
        Ei = self.atomic_property(a) * AM[..., None]
        E = torch.sum(Ei, 1)
        if self.normalizer:
            E = self.inverse_normalize(E)

        F = grad(
            E,
            R,
            grad_outputs=torch.ones_like(E),
            create_graph=False,
            retain_graph=True,
        )[0]
        F = -1.0 * F

        output.update({"E": E, "F": F})
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

        self.phi_q = nn.Sequential(
            Dense(n_features, n_features, activation=activation),
            Dense(n_features, 1, activation=None),
        )
        self.phi_qm = nn.Sequential(
            Dense(n_features, n_features, activation=activation),
            Dense(n_features, n_features, activation=None),
        )

        self.phi_b = nn.Sequential(
            Dense(n_features, n_features, activation=activation),
            Dense(n_features, 1, activation=None),
        )

        self.phi_bm = nn.Sequential(
            Dense(n_features, n_features, activation=activation),
            Dense(n_features, n_features, activation=None),
        )

        self.phi_e = nn.Sequential(
            Dense(n_features, n_features, activation=activation),
            Dense(n_features, n_features, activation=None),
        )

    def forward(
        self, a, q_dynamics, b_dynamics, e_dynamics, q_latent, b_latent, rbf, D, N, NM
    ):
        # radial basis
        rbf_msij = self.phi_rbf(rbf)  # B, A, N, nf
        rbf_msij = rbf_msij * self.cutoff_function(D).unsqueeze(-1)

        # map embeddings to features
        a_msij = self.phi_a(a)

        # copying atomic features for multiplication (B, A, N, nf)
        ai_msij = a_msij.repeat(1, 1, rbf_msij.size(2))
        ai_msij = ai_msij.view(rbf_msij.size())

        # reshaping of neighbor array (B, A, N, nf)
        b_size, a_size, n_size, nf_size = *N.size(), a_msij.size()[-1]
        tmp_n = N.view(-1, a_size * n_size, 1).expand(-1, -1, nf_size)

        # neighbor features (B, A, N, nf)
        aj_msij = torch.gather(a_msij, dim=1, index=tmp_n)
        aj_msij = aj_msij.view(b_size, a_size, n_size, nf_size)

        # symmetric messages (B, A, N, nf)
        msij = ai_msij * aj_msij * rbf_msij

        # feature update
        # Here is where NewtonNet's physics-based update system is inserted
        # Now using a charge - bond order update system

        # Dynamics: Charge Module
        q = self.phi_q(a)  # (B, A, 1)
        q_latent = q_latent + q  # (B, A, 1)
        q_dynamics = q_dynamics + q * self.phi_qm(a)  # (B, A, nf)

        qi = q_dynamics.repeat(1, 1, rbf_msij.size(2))
        qi = qi.view(rbf_msij.size())
        qj = torch.gather(q_dynamics, dim=1, index=tmp_n)
        qj = qj.view(b_size, a_size, n_size, nf_size)
        qiqj = qi * qj

        # Dynamics: Bond Module
        bij = self.phi_b(msij)
        b_latent = b_latent + bij.squeeze(-1)
        b_dynamics = b_dynamics + bij * self.phi_bm(msij)

        # Dynamics: Inverse Distance
        D_inv = torch.reciprocal(D).unsqueeze(-1)
        D_inv = torch.nan_to_num(D_inv, posinf=0.0)

        # Dynamics: Energy Module
        de_i = torch.sum(D_inv * (qiqj - b_dynamics), 2)
        de_i = self.phi_e(a) * de_i
        a = a + de_i
        e_dynamics = e_dynamics + de_i

        return a, q_dynamics, b_dynamics, e_dynamics, q_latent, b_latent


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
