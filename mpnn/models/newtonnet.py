import torch
from torch import nn
from torch.autograd import grad
from mpnn.layers import (
    Dense,
    RadialBesselLayer,
    PolynomialCutoff,
    ShellProvider,
)
from mpnn.utils import swish


class NewtonNet(nn.Module):
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
        super(NewtonNet, self).__init__()
        self.device = device

        self.n_features = n_features
        self.n_interax = n_interax
        self.activation = activation
        self.cutoff = cutoff

        self.atom_embedding = nn.Embedding(10, 128, padding_idx=0)

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
        self.atomic_energy = AtomicProperty(n_features, activation, dropout=None)

    def forward(self, data):
        R, Z, N = data["R"], data["Z"], data["N"]
        R.requires_grad_()

        AM, NM = data["AM"], data["NM"]
        D, V, N, NM = self.shell(R, N, NM)
        n_batch, n_atoms, n_neigh = N.size()

        rbf = self.distance_expansion(D)

        # atomic embeddings
        a = self.atom_embedding(Z)  # B, A, nf
        f_dir = torch.zeros_like(R)  # B,A,3
        f_dynamics = torch.zeros(
            R.size() + (self.n_features,), device=R.device
        )  # B,A,3,nf
        r_dynamics = torch.zeros(
            R.size() + (self.n_features,), device=R.device
        )  # B,A,3,nf
        e_dynamics = torch.zeros_like(a)  # B,A,nf

        # message passing layers
        for i_interax in range(self.n_interax):
            (
                a,
                f_dir,
                f_dynamics,
                r_dynamics,
                e_dynamics,
            ) = self.iterations[
                i_interax
            ](a, f_dir, f_dynamics, r_dynamics, e_dynamics, rbf, D, V, N, NM)

        # prediciton
        output = {"R": R, "Z": Z, "N": N, "NM": NM, "D": D, "V": V}

        # energy
        Ei = self.atomic_energy(a) * AM[..., None]
        E = torch.sum(Ei, 1)

        # gradients
        F = grad(
            E,
            R,
            grad_outputs=torch.ones_like(E),
            create_graph=False,
            retain_graph=True,
        )[0]
        F = -1.0 * F

        output.update({"E": E, "F": F, "F_latent": f_dir})

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

        self.phi_f = Dense(n_features, 1, activation=None, bias=False)
        self.phi_f_scale = nn.Sequential(
            Dense(n_features, n_features, activation=activation),
            Dense(n_features, n_features, activation=None),
        )
        self.phi_r = nn.Sequential(
            Dense(n_features, n_features, activation=activation, weight_gain=0.001),
            Dense(n_features, n_features, activation=None),
        )
        self.phi_r_ext = nn.Sequential(
            Dense(n_features, n_features, activation=activation, bias=False),
            Dense(n_features, n_features, activation=None, bias=False),
        )

        self.phi_e = nn.Sequential(
            Dense(n_features, n_features, activation=activation),
            Dense(n_features, n_features, activation=None),
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

    def forward(self, a, f_dir, f_dynamics, r_dynamics, e_dynamics, rbf, D, V, N, NM):
        # radial basis
        rbf_msij = self.phi_rbf(rbf)  # B, A, N, nf
        rbf_msij = rbf_msij * self.cutoff_function(D).unsqueeze(-1)

        # map embeddings to features
        a_msij = self.phi_a(a)

        # copying atomic features for multiplication (B, A, N, nf)
        ai_msij = a_msij.repeat(1, 1, rbf_msij.size(2))
        ai_msij = ai_msij.view(rbf_msij.size())

        # neighbor features (B, A, N, nf)
        aj_msij = self.gather_neighbors(a_msij, N)

        # symmetric messages (B, A, N, nf)
        msij = ai_msij * aj_msij * rbf_msij

        # Here is where NewtonNet's physics-based update system is inserted
        # Dynamics: Force Module
        F_ij = self.phi_f(msij) * V  # B,A,N,3
        F_i_dir = self.sum_neighbors(F_ij, NM, dim=2)  # B,A,3
        f_dir = f_dir + F_i_dir

        F_ij = self.phi_f_scale(msij).unsqueeze(-2) * F_ij.unsqueeze(-1)  # B,A,N,3,nf
        F_i = self.sum_neighbors(F_ij, NM, dim=2)  # B,A,3,nf

        f_dynamics = f_dynamics + F_i  # update

        # Dynamics: Displacement Module
        dr_i = self.phi_r(a).unsqueeze(-2) * F_i  # B,A,3,nf

        dr_j = self.gather_neighbors(r_dynamics, N)  # B,A,N,3,nf
        dr_j = self.phi_r_ext(msij).unsqueeze(-2) * dr_j  # B,A,N,3,nf
        dr_ext = self.sum_neighbors(dr_j, NM, dim=2)  # B,A,3,nf

        r_dynamics = r_dynamics + dr_i + dr_ext  # update

        # Dynamics: Energy Module
        de_i = -1.0 * torch.sum(f_dynamics * r_dynamics, dim=-2)  # B,A,nf
        de_i = self.phi_e(a) * de_i
        a = a + de_i
        e_dynamics = e_dynamics + de_i

        return a, f_dir, f_dynamics, r_dynamics, e_dynamics


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
