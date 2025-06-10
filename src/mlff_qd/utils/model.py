import torch
import torch.nn as nn
import schnetpack as spk
import schnetpack.transform as trn
import schnetpack.nn as snn
import schnetpack.nn.so3 as so3
from e3nn import o3
from e3nn.nn import FullyConnectedNet
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
import torchmetrics
from e3nn.nn import FullyConnectedNet

# PrecomputeSphericalHarmonics for SchNetPack models
class PrecomputeSphericalHarmonics(trn.Transform):
    def __init__(self, lmax=2):
        super().__init__()
        self.lmax = lmax

    def forward(self, inputs):
        rij = inputs['_Rij']
        radial_ij_raw = torch.norm(rij, dim=-1)
        dir_ij = rij / (radial_ij_raw.unsqueeze(-1) + 1e-8)
        inputs['_dir_ij_sh'] = compute_spherical_harmonics(dir_ij, lmax=self.lmax)
        return inputs

def compute_spherical_harmonics(dir_ij, lmax=2):
    with torch.no_grad():
        x, y, z = dir_ij[:, 0], dir_ij[:, 1], dir_ij[:, 2]
        r = torch.norm(dir_ij, dim=-1)
        theta = torch.acos(z / (r + 1e-8))
        phi = torch.atan2(y, x)
        
        pi_tensor = torch.tensor(torch.pi, device=dir_ij.device)
        sqrt_pi = torch.sqrt(pi_tensor)
        
        Y_0_0 = torch.ones_like(x) * (1 / (2 * sqrt_pi))
        sqrt_3_4pi = torch.sqrt(torch.tensor(3.0, device=dir_ij.device) / (4 * sqrt_pi))
        sin_theta = torch.sin(theta)
        Y_1_m1 = sqrt_3_4pi * sin_theta * torch.sin(phi)
        Y_1_0 = sqrt_3_4pi * torch.cos(theta)
        Y_1_1 = sqrt_3_4pi * sin_theta * torch.cos(phi)
        sqrt_15_16pi = torch.sqrt(torch.tensor(15.0, device=dir_ij.device) / (16 * sqrt_pi))
        sqrt_15_4pi = torch.sqrt(torch.tensor(15.0, device=dir_ij.device) / (4 * sqrt_pi))
        sqrt_5_16pi = torch.sqrt(torch.tensor(5.0, device=dir_ij.device) / (16 * sqrt_pi))
        sin_theta_sq = sin_theta ** 2
        cos_theta = torch.cos(theta)
        Y_2_m2 = sqrt_15_16pi * sin_theta_sq * torch.sin(2 * phi)
        Y_2_m1 = sqrt_15_4pi * sin_theta * cos_theta * torch.sin(phi)
        Y_2_0 = sqrt_5_16pi * (3 * cos_theta ** 2 - 1)
        Y_2_1 = sqrt_15_4pi * sin_theta * cos_theta * torch.cos(phi)
        Y_2_2 = sqrt_15_16pi * sin_theta_sq * torch.cos(2 * phi)
        
        return torch.stack([Y_0_0, Y_1_m1, Y_1_0, Y_1_1, Y_2_m2, Y_2_m1, Y_2_0, Y_2_1, Y_2_2], dim=-1)

# SchNetPack SO3 Convolution
class SchnetSO3ConvWrapper(nn.Module):
    def __init__(self, lmax, n_radial, n_atom_basis, cutoff, n_interactions=4):
        super(SchnetSO3ConvWrapper, self).__init__()
        self.n_atom_basis = n_atom_basis
        self.embedding = nn.Embedding(100, (lmax + 1) ** 2 * n_atom_basis)
        self.rbf = spk.nn.GaussianRBF(n_rbf=n_radial, cutoff=cutoff)
        self.register_buffer("cutoff", torch.tensor(cutoff, dtype=torch.float32))
        self.cutoff_fn = spk.nn.CosineCutoff(cutoff=self.cutoff)
        self.interactions = nn.ModuleList([
            so3.SO3Convolution(lmax=lmax, n_radial=n_radial, n_atom_basis=n_atom_basis)
            for _ in range(n_interactions)
        ])

    def forward(self, inputs):
        z = inputs['_atomic_numbers']
        x = self.embedding(z)
        x = x.view(-1, (self.interactions[0].lmax + 1) ** 2, self.n_atom_basis)
        
        rij = inputs['_Rij']
        radial_ij_raw = torch.norm(rij, dim=-1)
        radial_ij = self.rbf(radial_ij_raw)
        dir_ij_sh = inputs['_dir_ij_sh']  # Precomputed
        cutoff_ij = self.cutoff_fn(radial_ij_raw).unsqueeze(-1)
        idx_i = inputs['_idx_i']
        idx_j = inputs['_idx_j']
        
        for interaction in self.interactions:
            x = interaction(x, radial_ij, dir_ij_sh, cutoff_ij, idx_i, idx_j)
        
        inputs['scalar_representation'] = x
        return inputs

# SchNetPack SO3 Tensor Product
class SchnetSO3TensorWrapper(nn.Module):
    def __init__(self, n_atom_basis, n_radial, cutoff, lmax=2, n_interactions=4):
        super(SchnetSO3TensorWrapper, self).__init__()
        self.n_atom_basis = n_atom_basis
        self.lmax = lmax
        self.scalar_embedding = nn.Embedding(100, n_atom_basis)
        self.rbf = spk.nn.GaussianRBF(n_rbf=n_radial, cutoff=cutoff)
        self.interactions = nn.ModuleList([
            so3.SO3TensorProduct(lmax=lmax)
            for _ in range(n_interactions)
        ])
        self.feature_init_1 = nn.Linear(n_atom_basis, n_atom_basis * (lmax + 1) ** 2)
        self.feature_init_2 = nn.Linear(n_atom_basis, n_atom_basis * (lmax + 1) ** 2)
        self.update_layers = nn.ModuleList([
            nn.Linear(n_atom_basis * (lmax + 1) ** 2, n_atom_basis)
            for _ in range(n_interactions)
        ])
        self.radial_proj = nn.Linear(n_radial, (lmax + 1) ** 2)
        self.register_buffer("cutoff", torch.tensor(cutoff, dtype=torch.float32))
        self.cutoff_fn = spk.nn.CosineCutoff(cutoff=self.cutoff)

    def forward(self, inputs):
        z = inputs['_atomic_numbers']
        n_atoms = z.shape[0]
        scalar_feats = self.scalar_embedding(z)
        
        rij = inputs['_Rij']
        radial_ij_raw = torch.norm(rij, dim=-1)
        radial_ij = self.rbf(radial_ij_raw)
        dir_ij_sh = inputs['_dir_ij_sh']
        cutoff_ij = self.cutoff_fn(radial_ij_raw).unsqueeze(-1)
        idx_i = inputs['_idx_i']
        idx_j = inputs['_idx_j']
        
        x = scalar_feats
        for i in range(len(self.interactions)):
            x1 = self.feature_init_1(x).view(-1, (self.lmax + 1) ** 2, self.n_atom_basis)
            x2 = self.feature_init_2(x).view(-1, (self.lmax + 1) ** 2, self.n_atom_basis)
            updated_feats = self.interactions[i](x1, x2)
            radial_proj = self.radial_proj(radial_ij)
            cutoff_ij_exp = cutoff_ij.unsqueeze(1)
            neighbor_contrib = (
                x2[idx_j] * dir_ij_sh.unsqueeze(-1) * radial_proj.unsqueeze(-1) * cutoff_ij_exp
            )
            neighbor_agg = snn.scatter_add(neighbor_contrib, idx_i, dim_size=x1.shape[0], dim=0)
            updated_feats = updated_feats + neighbor_agg
            x = self.update_layers[i](updated_feats.view(-1, self.n_atom_basis * (self.lmax + 1) ** 2))
        
        inputs['scalar_representation'] = updated_feats
        return inputs

# e3nn Convolution updated new

class E3nnSO3ConvWrapper(nn.Module):
    def __init__(self, n_atom_basis, n_radial, cutoff, lmax=2, n_interactions=1):
        super(E3nnSO3ConvWrapper, self).__init__()
        self.n_atom_basis = n_atom_basis
        self.lmax = lmax
        
        self.embedding = nn.Embedding(100, n_atom_basis)
        self.rbf = spk.nn.GaussianRBF(n_rbf=n_radial, cutoff=cutoff)
        
        irreps_in = o3.Irreps(f"{n_atom_basis}x0e")
        irreps_sh = o3.Irreps(" + ".join([f"1x{l}e" for l in range(lmax + 1)]))
        irreps_out = o3.Irreps(" + ".join([f"{n_atom_basis}x{l}e" for l in range(lmax + 1)]))
        
        # Calculate number of paths for TensorProduct (one weight per irrep)
        n_paths = (lmax + 1) ** 2  # Number of irreps (e.g., 9 for lmax=2, 16 for lmax=3)
        self.radial_net = FullyConnectedNet([n_radial, 64, n_paths], act=torch.nn.ReLU())
        
        instructions = [(0, l, l, "uvu", True) for l in range(lmax + 1)]
        
        self.interactions = nn.ModuleList([
            o3.TensorProduct(
                irreps_in1=irreps_in if i == 0 else irreps_out,
                irreps_in2=irreps_sh,
                irreps_out=irreps_out,
                instructions=instructions,
                internal_weights=True,
                shared_weights=True
            ) for i in range(n_interactions)
        ])
        
        # Reduce layer only applied at the end
        self.reduce_layer = nn.Linear(n_atom_basis * (lmax + 1) ** 2, n_atom_basis)
        self.cutoff_fn = spk.nn.CosineCutoff(cutoff=cutoff)

    def forward(self, inputs):
        device = next(self.parameters()).device
        
        z = inputs['_atomic_numbers'].to(device)
        x = self.embedding(z.long())  # [n_atoms, n_atom_basis]
        
        rij = inputs['_Rij'].to(device)
        idx_i = inputs['_idx_i'].to(device)
        idx_j = inputs['_idx_j'].to(device)
        n_atoms = inputs['_n_atoms'].sum().item()
        
        radial_ij_raw = torch.norm(rij, dim=-1)
        radial_ij = self.rbf(radial_ij_raw).to(device)
        
        sh = o3.SphericalHarmonics(range(self.lmax + 1), normalize=True, normalization='component')
        dir_ij = rij / (radial_ij_raw.unsqueeze(-1) + 1e-8)
        dir_ij_sh = sh(dir_ij).to(device)
        
        radial_weights = self.radial_net(radial_ij)  # [n_edges, (lmax + 1)^2]
        radial_weights = radial_weights.view(-1, (self.lmax + 1) ** 2, 1)  # [n_edges, (lmax + 1)^2, 1]
        radial_weights = radial_weights * self.cutoff_fn(radial_ij_raw).unsqueeze(-1).unsqueeze(-1)
        
        for i, interaction in enumerate(self.interactions):
            neighbor_contrib = interaction(x[idx_j], dir_ij_sh)  # [n_edges, n_atom_basis * (lmax + 1)^2]
            neighbor_contrib = neighbor_contrib.view(-1, (self.lmax + 1) ** 2, self.n_atom_basis)  # [n_edges, (lmax + 1)^2, n_atom_basis]
            neighbor_contrib = neighbor_contrib * radial_weights  # Broadcasting: [n_edges, (lmax + 1)^2, n_atom_basis] * [n_edges, (lmax + 1)^2, 1]
            neighbor_contrib = neighbor_contrib.view(-1, self.n_atom_basis * (self.lmax + 1) ** 2)  # [n_edges, n_atom_basis * (lmax + 1)^2]
            x = snn.scatter_add(neighbor_contrib, idx_i, dim_size=n_atoms, dim=0)  # [n_atoms, n_atom_basis * (lmax + 1)^2]
            if i == len(self.interactions) - 1:  # Apply reduce_layer only at the end
                x = self.reduce_layer(x)  # [n_atoms, n_atom_basis]
        
        inputs['scalar_representation'] = x
        return inputs

# e3nn Tensor Product
class E3nnSO3TensorWrapper(nn.Module):
    def __init__(self, n_atom_basis, n_radial, cutoff, lmax=2, n_interactions=4):
        super(E3nnSO3TensorWrapper, self).__init__()
        self.n_atom_basis = n_atom_basis
        self.lmax = lmax
        self.scalar_embedding = nn.Embedding(100, n_atom_basis)
        self.rbf = spk.nn.GaussianRBF(n_rbf=n_radial, cutoff=cutoff)
        
        irreps_in = o3.Irreps(f"{n_atom_basis}x0e")
        irreps_out = o3.Irreps(f"{n_atom_basis}x0e")
        
        instructions = [(0, 0, 0, "uvu", True)]
        
        self.interactions = nn.ModuleList([
            o3.TensorProduct(
                irreps_in1=irreps_in,
                irreps_in2=irreps_in,
                irreps_out=irreps_out,
                instructions=instructions,
                internal_weights=True,
                shared_weights=True
            )
            for _ in range(n_interactions)
        ])
        
        self.radial_proj = nn.Linear(n_radial, n_atom_basis)
        self.register_buffer("cutoff", torch.tensor(cutoff, dtype=torch.float32))
        self.cutoff_fn = spk.nn.CosineCutoff(cutoff=self.cutoff)

    def forward(self, inputs):
        z = inputs['_atomic_numbers']
        n_atoms = z.shape[0]
        scalar_feats = self.scalar_embedding(z)
        
        rij = inputs['_Rij']
        radial_ij_raw = torch.norm(rij, dim=-1)
        radial_ij = self.rbf(radial_ij_raw)
        
        sh = o3.SphericalHarmonics(range(self.lmax + 1), normalize=True, normalization='component')
        dir_ij = rij / (radial_ij_raw.unsqueeze(-1) + 1e-8)
        dir_ij_sh = sh(dir_ij)
        
        cutoff_ij = self.cutoff_fn(radial_ij_raw).unsqueeze(-1)
        idx_i = inputs['_idx_i']
        idx_j = inputs['_idx_j']
        
        x = scalar_feats
        for i, interaction in enumerate(self.interactions):
            x1 = x
            x2 = x
            updated_feats = interaction(x1, x2)
            radial_proj = self.radial_proj(radial_ij)
            neighbor_contrib = x[idx_j] * radial_proj * cutoff_ij
            neighbor_agg = snn.scatter_add(neighbor_contrib, idx_i, dim_size=x.shape[0], dim=0)
            x = updated_feats + neighbor_agg
        
        inputs['scalar_representation'] = x
        return inputs


class HybridPaiNNE3nnSO3(nn.Module):
    def __init__(self, n_atom_basis, n_radial, cutoff, lmax=2, n_interactions=4):
        super(HybridPaiNNE3nnSO3, self).__init__()
        self.n_atom_basis = n_atom_basis
        self.lmax = lmax
        self.rbf = spk.nn.GaussianRBF(n_rbf=n_radial, cutoff=cutoff)
        
        
        self.n_painn = 1  # Fixed minimum PaiNN layers
        self.n_e3nn = max(0, n_interactions - self.n_painn)  # Remaining as e3nn
        
        # PaiNN with 1 interaction
        self.painn = spk.representation.PaiNN(
            n_atom_basis=n_atom_basis,
            n_interactions=self.n_painn,
            radial_basis=self.rbf,
            cutoff_fn=spk.nn.CosineCutoff(cutoff)
        )
        
        # e3nn SO3 Convolution layers
        irreps_in = o3.Irreps(f"{n_atom_basis}x0e")
        irreps_sh = o3.Irreps(" + ".join([f"1x{l}e" for l in range(lmax + 1)]))
        irreps_out = o3.Irreps(f"{n_atom_basis}x0e + {n_atom_basis}x1e + {n_atom_basis}x2e")
        instructions = [(0, l, l, "uvu", True) for l in range(lmax + 1)]
        
        self.e3nn_interactions = nn.ModuleList([
            o3.TensorProduct(
                irreps_in1=irreps_in if i == 0 and self.n_painn > 0 else irreps_out,
                irreps_in2=irreps_sh,
                irreps_out=irreps_out,
                instructions=instructions,
                internal_weights=True,
                shared_weights=True
            )
            for i in range(self.n_e3nn)
        ])
        
        self.radial_net = FullyConnectedNet([n_radial, 64, n_atom_basis * (lmax + 1) ** 2], act=torch.nn.ReLU())
        self.reduce_layer = nn.Linear(n_atom_basis * (lmax + 1) ** 2, n_atom_basis)
        self.register_buffer("cutoff", torch.tensor(cutoff, dtype=torch.float32))
        self.cutoff_fn = spk.nn.CosineCutoff(cutoff=self.cutoff)

    def forward(self, inputs):
        # PaiNN layer
        inputs = self.painn(inputs)
        x = inputs['scalar_representation']  # [n_atoms, n_atom_basis]
        
        # e3nn SO3 layers
        rij = inputs['_Rij']
        radial_ij_raw = torch.norm(rij, dim=-1)
        radial_ij = self.rbf(radial_ij_raw)
        sh = o3.SphericalHarmonics(range(self.lmax + 1), normalize=True, normalization='component')
        dir_ij = rij / (radial_ij_raw.unsqueeze(-1) + 1e-8)
        dir_ij_sh = sh(dir_ij)
        cutoff_ij = self.cutoff_fn(radial_ij_raw).unsqueeze(-1).unsqueeze(-1)
        idx_i = inputs['_idx_i']
        idx_j = inputs['_idx_j']
        n_atoms = inputs['_n_atoms'].sum().item()
        
        radial_weights = self.radial_net(radial_ij)
        radial_weights = radial_weights.view(-1, (self.lmax + 1) ** 2, self.n_atom_basis)
        
        for interaction in self.e3nn_interactions:
            neighbor_contrib = interaction(x[idx_j], dir_ij_sh)
            neighbor_contrib = neighbor_contrib.view(-1, (self.lmax + 1) ** 2, self.n_atom_basis)
            neighbor_contrib = neighbor_contrib * radial_weights * cutoff_ij
            neighbor_contrib = neighbor_contrib.view(-1, (self.lmax + 1) ** 2 * self.n_atom_basis)
            x = snn.scatter_add(neighbor_contrib, idx_i, dim_size=n_atoms, dim=0)
        
        x = self.reduce_layer(x)
        inputs['scalar_representation'] = x
        return inputs
        


class HybridPaiNNSchNetE3nnSO3(nn.Module):
    def __init__(self, n_atom_basis, n_radial, cutoff, lmax=2, n_interactions=5):
        super(HybridPaiNNSchNetE3nnSO3, self).__init__()
        self.n_atom_basis = n_atom_basis
        self.lmax = lmax
        self.rbf = spk.nn.GaussianRBF(n_rbf=n_radial, cutoff=cutoff)
        
        # PaiNN: 1 interaction
        self.painn = spk.representation.PaiNN(
            n_atom_basis=n_atom_basis,
            n_interactions=1,
            radial_basis=self.rbf,
            cutoff_fn=spk.nn.CosineCutoff(cutoff)
        )
        
        # SchNet: 1 interaction
        self.schnet = spk.representation.SchNet(
            n_atom_basis=n_atom_basis,
            n_interactions=1,
            radial_basis=self.rbf,
            cutoff_fn=spk.nn.CosineCutoff(cutoff)
        )
        
        # e3nn SO3: Remaining layers (n_interactions - 2)
        self.n_e3nn = max(0, n_interactions - 2)  # 1 PaiNN + 1 SchNet = 2, rest e3nn
        irreps_in = o3.Irreps(f"{n_atom_basis}x0e")
        irreps_sh = o3.Irreps(" + ".join([f"1x{l}e" for l in range(lmax + 1)]))
        irreps_out = o3.Irreps(f"{n_atom_basis}x0e + {n_atom_basis}x1e + {n_atom_basis}x2e")
        instructions = [(0, l, l, "uvu", True) for l in range(lmax + 1)]
        
        self.e3nn_interactions = nn.ModuleList([
            o3.TensorProduct(
                irreps_in1=irreps_in if i == 0 else irreps_out,
                irreps_in2=irreps_sh,
                irreps_out=irreps_out,
                instructions=instructions,
                internal_weights=True,
                shared_weights=True
            )
            for i in range(self.n_e3nn)
        ])
        
        self.radial_net = FullyConnectedNet([n_radial, 64, n_atom_basis * (lmax + 1) ** 2], act=torch.nn.ReLU())
        self.reduce_layer = nn.Linear(n_atom_basis * (lmax + 1) ** 2, n_atom_basis)
        self.register_buffer("cutoff", torch.tensor(cutoff, dtype=torch.float32))
        self.cutoff_fn = spk.nn.CosineCutoff(cutoff=self.cutoff)

    def forward(self, inputs):
        # PaiNN layer
        inputs = self.painn(inputs)
        x = inputs['scalar_representation']  # [n_atoms, n_atom_basis]
        
        # SchNet layer
        inputs['scalar_representation'] = x  # Pass PaiNN output to SchNet
        inputs = self.schnet(inputs)
        x = inputs['scalar_representation']  # [n_atoms, n_atom_basis]
        
        # e3nn SO3 layers
        rij = inputs['_Rij']
        radial_ij_raw = torch.norm(rij, dim=-1)
        radial_ij = self.rbf(radial_ij_raw)
        sh = o3.SphericalHarmonics(range(self.lmax + 1), normalize=True, normalization='component')
        dir_ij = rij / (radial_ij_raw.unsqueeze(-1) + 1e-8)
        dir_ij_sh = sh(dir_ij)
        cutoff_ij = self.cutoff_fn(radial_ij_raw).unsqueeze(-1).unsqueeze(-1)
        idx_i = inputs['_idx_i']
        idx_j = inputs['_idx_j']
        n_atoms = inputs['_n_atoms'].sum().item()
        
        radial_weights = self.radial_net(radial_ij)
        radial_weights = radial_weights.view(-1, (self.lmax + 1) ** 2, self.n_atom_basis)
        
        for interaction in self.e3nn_interactions:
            neighbor_contrib = interaction(x[idx_j], dir_ij_sh)
            neighbor_contrib = neighbor_contrib.view(-1, (self.lmax + 1) ** 2, self.n_atom_basis)
            neighbor_contrib = neighbor_contrib * radial_weights * cutoff_ij
            neighbor_contrib = neighbor_contrib.view(-1, (self.lmax + 1) ** 2 * self.n_atom_basis)
            x = snn.scatter_add(neighbor_contrib, idx_i, dim_size=n_atoms, dim=0)
        
        x = self.reduce_layer(x)
        inputs['scalar_representation'] = x
        return inputs
        


class OptimizedHybridPaiNNE3nn(nn.Module):
    def __init__(self, n_atom_basis, n_radial, cutoff, lmax=2, n_interactions=3):
        super(OptimizedHybridPaiNNE3nn, self).__init__()
        self.n_atom_basis = n_atom_basis
        self.lmax = lmax
        self.rbf = spk.nn.GaussianRBF(n_rbf=n_radial, cutoff=cutoff)
        
        # PaiNN: 1 interaction, streamlined
        self.painn = spk.representation.PaiNN(
            n_atom_basis=n_atom_basis,
            n_interactions=1,
            radial_basis=self.rbf,
            cutoff_fn=spk.nn.CosineCutoff(cutoff)
        )
        
        # e3nn SO3: 2 layers
        self.n_e3nn = max(0, n_interactions - 1)
        irreps_in = o3.Irreps(f"{n_atom_basis}x0e")
        irreps_sh = o3.Irreps(" + ".join([f"1x{l}e" for l in range(lmax + 1)]))
        irreps_out = o3.Irreps(f"{n_atom_basis}x0e + {n_atom_basis}x1e + {n_atom_basis}x2e")
        instructions = [(0, l, l, "uvu", True) for l in range(lmax + 1)]
        
        self.e3nn_interactions = nn.ModuleList([
            o3.TensorProduct(
                irreps_in1=irreps_in if i == 0 else irreps_out,
                irreps_in2=irreps_sh,
                irreps_out=irreps_out,
                instructions=instructions,
                internal_weights=True,
                shared_weights=True
            )
            for i in range(self.n_e3nn)
        ])
        
        self.radial_net = FullyConnectedNet([n_radial, n_atom_basis * (lmax + 1) ** 2], act=torch.nn.ReLU())
        self.reduce_layer = nn.Linear(n_atom_basis * (lmax + 1) ** 2, n_atom_basis)
        self.register_buffer("cutoff", torch.tensor(cutoff, dtype=torch.float32))
        self.cutoff_fn = spk.nn.CosineCutoff(cutoff)

    def forward(self, inputs):
        inputs = self.painn(inputs)
        x = inputs['scalar_representation']
        
        rij = inputs['_Rij']
        radial_ij_raw = torch.norm(rij, dim=-1)
        radial_ij = self.rbf(radial_ij_raw)
        sh = o3.SphericalHarmonics(range(self.lmax + 1), normalize=True, normalization='component')
        dir_ij = rij / (radial_ij_raw.unsqueeze(-1) + 1e-8)
        dir_ij_sh = sh(dir_ij)
        cutoff_ij = self.cutoff_fn(radial_ij_raw).unsqueeze(-1).unsqueeze(-1)
        idx_i = inputs['_idx_i']
        idx_j = inputs['_idx_j']
        n_atoms = inputs['_n_atoms'].sum().item()
        
        radial_weights = self.radial_net(radial_ij)
        radial_weights = radial_weights.view(-1, (self.lmax + 1) ** 2, self.n_atom_basis)
        
        for interaction in self.e3nn_interactions:
            neighbor_contrib = interaction(x[idx_j], dir_ij_sh)
            neighbor_contrib = neighbor_contrib.view(-1, (self.lmax + 1) ** 2, self.n_atom_basis)
            neighbor_contrib = neighbor_contrib * radial_weights * cutoff_ij
            neighbor_contrib = neighbor_contrib.view(-1, (self.lmax + 1) ** 2 * self.n_atom_basis)
            x = snn.scatter_add(neighbor_contrib, idx_i, dim_size=n_atoms, dim=0)
        
        x = self.reduce_layer(x)
        inputs['scalar_representation'] = x
        return inputs


class FusedHybridModel(nn.Module):
    def __init__(self, n_atom_basis, n_radial, cutoff, lmax=2):
        super(FusedHybridModel, self).__init__()
        self.n_atom_basis = n_atom_basis
        self.lmax = lmax
        self.rbf = spk.nn.GaussianRBF(n_rbf=n_radial, cutoff=cutoff)
        
        # PaiNN: 1 layer
        self.painn = spk.representation.PaiNN(
            n_atom_basis=n_atom_basis,
            n_interactions=1,
            radial_basis=self.rbf,
            cutoff_fn=spk.nn.CosineCutoff(cutoff)
        )
        
        # SchNet: 1 layer
        self.schnet = spk.representation.SchNet(
            n_atom_basis=n_atom_basis,
            n_interactions=1,
            radial_basis=self.rbf,
            cutoff_fn=spk.nn.CosineCutoff(cutoff)
        )
        
        # e3nn SO3: 1 layer
        irreps_in = o3.Irreps(f"{n_atom_basis}x0e")
        irreps_sh = o3.Irreps(" + ".join([f"1x{l}e" for l in range(lmax + 1)]))
        irreps_out = o3.Irreps(f"{n_atom_basis}x0e + {n_atom_basis}x1e + {n_atom_basis}x2e")
        instructions = [(0, l, l, "uvu", True) for l in range(lmax + 1)]
        self.e3nn_interaction = o3.TensorProduct(
            irreps_in1=irreps_in,
            irreps_in2=irreps_sh,
            irreps_out=irreps_out,
            instructions=instructions,
            internal_weights=True,
            shared_weights=True
        )
        
        # Embedding for e3nn path
        self.e3nn_embedding = nn.Embedding(100, n_atom_basis)
        
        self.e3nn_radial_net = FullyConnectedNet([n_radial, n_atom_basis * (lmax + 1) ** 2], act=torch.nn.ReLU())
        
        # Projection layers for fusion
        self.painn_proj = nn.Linear(n_atom_basis, n_atom_basis // 2)
        self.schnet_proj = nn.Linear(n_atom_basis, n_atom_basis // 2)
        self.e3nn_proj = nn.Linear(n_atom_basis * (lmax + 1) ** 2, n_atom_basis // 2)
        
        # Fusion layer
        self.fusion_layer = nn.Sequential(
            nn.Linear((n_atom_basis // 2) * 3, n_atom_basis),
            nn.ReLU(),
            nn.Linear(n_atom_basis, n_atom_basis)
        )
        
        self.register_buffer("cutoff", torch.tensor(cutoff, dtype=torch.float32))
        self.cutoff_fn = spk.nn.CosineCutoff(cutoff=self.cutoff)

    def forward(self, inputs):
        # Store original inputs for e3nn
        inputs_orig = inputs.copy()
        
        # PaiNN features
        inputs_painn = self.painn(inputs)
        painn_features = inputs_painn['scalar_representation']  # [n_atoms, 256]
        painn_features = self.painn_proj(painn_features)  # [n_atoms, 128]
        
        # SchNet features
        inputs_schnet = inputs_orig.copy()
        inputs_schnet = self.schnet(inputs_schnet)
        schnet_features = inputs_schnet['scalar_representation']  # [n_atoms, 256]
        schnet_features = self.schnet_proj(schnet_features)  # [n_atoms, 128]
        
        # e3nn SO3 features
        x = self.e3nn_embedding(inputs_orig['_atomic_numbers'])  # [n_atoms, 256]
        rij = inputs_orig['_Rij']
        radial_ij_raw = torch.norm(rij, dim=-1)
        radial_ij = self.rbf(radial_ij_raw)
        sh = o3.SphericalHarmonics(range(self.lmax + 1), normalize=True, normalization='component')
        dir_ij = rij / (radial_ij_raw.unsqueeze(-1) + 1e-8)
        dir_ij_sh = sh(dir_ij)
        cutoff_ij = self.cutoff_fn(radial_ij_raw).unsqueeze(-1).unsqueeze(-1)
        idx_i = inputs_orig['_idx_i']
        idx_j = inputs_orig['_idx_j']
        n_atoms = inputs_orig['_n_atoms'].sum().item()
        
        radial_weights = self.e3nn_radial_net(radial_ij)
        radial_weights = radial_weights.view(-1, (self.lmax + 1) ** 2, self.n_atom_basis)
        
        neighbor_contrib = self.e3nn_interaction(x[idx_j], dir_ij_sh)
        neighbor_contrib = neighbor_contrib.view(-1, (self.lmax + 1) ** 2, self.n_atom_basis)
        neighbor_contrib = neighbor_contrib * radial_weights * cutoff_ij
        neighbor_contrib = neighbor_contrib.view(-1, (self.lmax + 1) ** 2 * self.n_atom_basis)
        e3nn_features = snn.scatter_add(neighbor_contrib, idx_i, dim_size=n_atoms, dim=0)  # [n_atoms, 2304]
        e3nn_features = self.e3nn_proj(e3nn_features)  # [n_atoms, 128]
        
        # Fuse features
        fused_features = torch.cat([painn_features, schnet_features, e3nn_features], dim=-1)  # [n_atoms, 128*3]
        x = self.fusion_layer(fused_features)  # [n_atoms, 256]
        
        inputs['scalar_representation'] = x
        return inputs


#  **************************************************************************** put class hehre from another tab


class EquiSeqHybrid(nn.Module):
    def __init__(self, n_atom_basis, n_radial, cutoff, lmax=2, n_interactions=3):
        super(EquiSeqHybrid, self).__init__()
        self.n_atom_basis = n_atom_basis
        self.lmax = lmax
        self.rbf = spk.nn.GaussianRBF(n_rbf=n_radial, cutoff=cutoff)
        
        # PaiNN: 1 layer
        self.painn = spk.representation.PaiNN(
            n_atom_basis=n_atom_basis,
            n_interactions=1,
            radial_basis=self.rbf,
            cutoff_fn=spk.nn.CosineCutoff(cutoff)
        )
        
        # Equivariant Attention: 1 layer
        self.attention = EquivariantAttention(n_atom_basis, lmax)
        
        # e3nn SO3: 1 layer
        irreps_in = o3.Irreps(f"{n_atom_basis}x0e")
        irreps_sh = o3.Irreps(" + ".join([f"1x{l}e" for l in range(lmax + 1)]))
        irreps_out = o3.Irreps(f"{n_atom_basis}x0e + {n_atom_basis}x1e + {n_atom_basis}x2e")
        instructions = [(0, l, l, "uvu", True) for l in range(lmax + 1)]
        self.e3nn_interaction = o3.TensorProduct(
            irreps_in1=irreps_in,
            irreps_in2=irreps_sh,
            irreps_out=irreps_out,
            instructions=instructions,
            internal_weights=True,
            shared_weights=True
        )
        
        self.e3nn_radial_net = FullyConnectedNet([n_radial, n_atom_basis * (lmax + 1) ** 2], act=torch.nn.ReLU())
        self.reduce_layer = nn.Linear(n_atom_basis * (lmax + 1) ** 2, n_atom_basis)
        self.register_buffer("cutoff", torch.tensor(cutoff, dtype=torch.float32))
        self.cutoff_fn = spk.nn.CosineCutoff(cutoff=self.cutoff)

    def forward(self, inputs):
        # PaiNN
        inputs = self.painn(inputs)
        x = inputs['scalar_representation']  # [n_atoms, n_atom_basis]
        
        # Prepare for attention and e3nn
        rij = inputs['_Rij']
        radial_ij_raw = torch.norm(rij, dim=-1)
        radial_ij = self.rbf(radial_ij_raw)
        sh = o3.SphericalHarmonics(range(self.lmax + 1), normalize=True, normalization='component')
        dir_ij = rij / (radial_ij_raw.unsqueeze(-1) + 1e-8)
        dir_ij_sh = sh(dir_ij)
        cutoff_ij = self.cutoff_fn(radial_ij_raw).unsqueeze(-1).unsqueeze(-1)
        idx_i = inputs['_idx_i']
        idx_j = inputs['_idx_j']
        n_atoms = inputs['_n_atoms'].sum().item()
        
        # Equivariant Attention
        x = self.attention(x, dir_ij_sh, idx_i, idx_j, n_atoms)  # [n_atoms, n_atom_basis]
        
        # e3nn SO3
        radial_weights = self.e3nn_radial_net(radial_ij)
        radial_weights = radial_weights.view(-1, (self.lmax + 1) ** 2, self.n_atom_basis)
        neighbor_contrib = self.e3nn_interaction(x[idx_j], dir_ij_sh)
        neighbor_contrib = neighbor_contrib.view(-1, (self.lmax + 1) ** 2, self.n_atom_basis)
        neighbor_contrib = neighbor_contrib * radial_weights * cutoff_ij
        neighbor_contrib = neighbor_contrib.view(-1, (self.lmax + 1) ** 2 * self.n_atom_basis)
        x = snn.scatter_add(neighbor_contrib, idx_i, dim_size=n_atoms, dim=0)
        
        # Reduce
        x = self.reduce_layer(x)
        inputs['scalar_representation'] = x
        return inputs




class FusedNequIPHybrid(nn.Module):
    def __init__(self, n_atom_basis, n_radial, cutoff, lmax=2):
        super(FusedNequIPHybrid, self).__init__()
        self.n_atom_basis = n_atom_basis
        self.lmax = lmax
        self.rbf = spk.nn.GaussianRBF(n_rbf=n_radial, cutoff=cutoff)
        
        # PaiNN: 1 layer
        self.painn = spk.representation.PaiNN(
            n_atom_basis=n_atom_basis,
            n_interactions=1,
            radial_basis=self.rbf,
            cutoff_fn=spk.nn.CosineCutoff(cutoff)
        )
        
        # SchNet: 1 layer
        self.schnet = spk.representation.SchNet(
            n_atom_basis=n_atom_basis,
            n_interactions=1,
            radial_basis=self.rbf,
            cutoff_fn=spk.nn.CosineCutoff(cutoff)
        )
        
        # NequIP-like e3nn: 1 layer with multi-body equivariance
        irreps_in = o3.Irreps(f"{n_atom_basis}x0e")
        irreps_sh = o3.Irreps(" + ".join([f"1x{l}e" for l in range(lmax + 1)]))
        irreps_out = o3.Irreps(f"{n_atom_basis}x0e + {n_atom_basis}x1e + {n_atom_basis}x2e")
        instructions = [(0, l, l, "uvu", True) for l in range(lmax + 1)]
        self.nequip_interaction = o3.TensorProduct(
            irreps_in1=irreps_in,
            irreps_in2=irreps_sh,
            irreps_out=irreps_out,
            instructions=instructions,
            internal_weights=True,
            shared_weights=True
        )
        
        self.nequip_embedding = nn.Embedding(100, n_atom_basis)
        self.nequip_radial_net = FullyConnectedNet([n_radial, n_atom_basis * (lmax + 1) ** 2], act=torch.nn.ReLU())
        
        # Projection layers for fusion
        proj_dim = n_atom_basis // 2  # e.g., 96
        self.painn_proj = nn.Linear(n_atom_basis, proj_dim)
        self.schnet_proj = nn.Linear(n_atom_basis, proj_dim)
        self.nequip_proj = nn.Linear(n_atom_basis * (lmax + 1) ** 2, proj_dim)
        
        # Fusion layer
        self.fusion_layer = nn.Sequential(
            nn.Linear(proj_dim * 3, n_atom_basis),
            nn.ReLU(),
            nn.Linear(n_atom_basis, n_atom_basis)
        )
        
        self.register_buffer("cutoff", torch.tensor(cutoff, dtype=torch.float32))
        self.cutoff_fn = spk.nn.CosineCutoff(cutoff=self.cutoff)

    def forward(self, inputs):
        inputs_orig = inputs.copy()
        
        # PaiNN
        inputs_painn = self.painn(inputs)
        painn_features = inputs_painn['scalar_representation']  # [n_atoms, 192]
        painn_features = self.painn_proj(painn_features)  # [n_atoms, 96]
        
        # SchNet
        inputs_schnet = inputs_orig.copy()
        inputs_schnet = self.schnet(inputs_schnet)
        schnet_features = inputs_schnet['scalar_representation']  # [n_atoms, 192]
        schnet_features = self.schnet_proj(schnet_features)  # [n_atoms, 96]
        
        # NequIP-like e3nn
        x = self.nequip_embedding(inputs_orig['_atomic_numbers'])  # [n_atoms, 192]
        rij = inputs_orig['_Rij']
        radial_ij_raw = torch.norm(rij, dim=-1)
        radial_ij = self.rbf(radial_ij_raw)
        sh = o3.SphericalHarmonics(range(self.lmax + 1), normalize=True, normalization='component')
        dir_ij = rij / (radial_ij_raw.unsqueeze(-1) + 1e-8)
        dir_ij_sh = sh(dir_ij)
        cutoff_ij = self.cutoff_fn(radial_ij_raw).unsqueeze(-1).unsqueeze(-1)
        idx_i = inputs_orig['_idx_i']
        idx_j = inputs_orig['_idx_j']
        n_atoms = inputs_orig['_n_atoms'].sum().item()
        
        radial_weights = self.nequip_radial_net(radial_ij)
        radial_weights = radial_weights.view(-1, (self.lmax + 1) ** 2, self.n_atom_basis)
        neighbor_contrib = self.nequip_interaction(x[idx_j], dir_ij_sh)
        neighbor_contrib = neighbor_contrib.view(-1, (self.lmax + 1) ** 2, self.n_atom_basis)
        neighbor_contrib = neighbor_contrib * radial_weights * cutoff_ij
        neighbor_contrib = neighbor_contrib.view(-1, (self.lmax + 1) ** 2 * self.n_atom_basis)
        nequip_features = snn.scatter_add(neighbor_contrib, idx_i, dim_size=n_atoms, dim=0)  # [n_atoms, 1728]
        nequip_features = self.nequip_proj(nequip_features)  # [n_atoms, 96]
        
        # Fuse
        fused_features = torch.cat([painn_features, schnet_features, nequip_features], dim=-1)  # [n_atoms, 96*3]
        x = self.fusion_layer(fused_features)  # [n_atoms, 192]
        
        inputs['scalar_representation'] = x
        return inputs


# updated
class MACEInspiredLayer(nn.Module):
    def __init__(self, n_atom_basis, n_radial, lmax=2):
        super(MACEInspiredLayer, self).__init__()
        self.n_atom_basis = n_atom_basis
        self.lmax = lmax
        
        irreps_in = o3.Irreps(f"{n_atom_basis}x0e")
        irreps_sh = o3.Irreps(" + ".join([f"1x{l}e" for l in range(lmax + 1)]))
        irreps_out = o3.Irreps(" + ".join([f"{n_atom_basis}x{l}e" for l in range(lmax + 1)]))
        
        # Calculate number of paths for TensorProduct (one weight per irrep)
        n_paths = (lmax + 1) ** 2  # Number of irreps (e.g., 9 for lmax=2, 16 for lmax=3)
        
        self.message_tp = o3.TensorProduct(
            irreps_in1=irreps_in,
            irreps_in2=irreps_sh,
            irreps_out=irreps_out,
            instructions=[(0, l, l, "uvu", True) for l in range(lmax + 1)],
            internal_weights=True,
            shared_weights=True
        )
        
        self.radial_net = FullyConnectedNet([n_radial, 64, n_paths], act=torch.nn.ReLU())
        self.reduce_layer = nn.Linear(n_atom_basis * (lmax + 1) ** 2, n_atom_basis)
        self.update_linear = nn.Linear(n_atom_basis * (lmax + 1) ** 2, n_atom_basis * (lmax + 1) ** 2)

    def forward(self, x, dir_ij_sh, radial_ij, idx_i, idx_j, n_atoms):
        device = x.device
        radial_weights = self.radial_net(radial_ij)  # [n_edges, (lmax + 1)^2]
        radial_weights = radial_weights.view(-1, (self.lmax + 1) ** 2, 1)  # [n_edges, (lmax + 1)^2, 1]
        
        # Message passing
        message = self.message_tp(x[idx_j], dir_ij_sh)  # [n_edges, n_atom_basis * (lmax + 1)^2]
        message = message.view(-1, (self.lmax + 1) ** 2, self.n_atom_basis)  # [n_edges, (lmax + 1)^2, n_atom_basis]
        message = message * radial_weights  # Broadcasting: [n_edges, (lmax + 1)^2, n_atom_basis] * [n_edges, (lmax + 1)^2, 1]
        message = message.view(-1, self.n_atom_basis * (self.lmax + 1) ** 2)
        aggregated = snn.scatter_add(message, idx_i, dim_size=n_atoms, dim=0)  # [n_atoms, n_atom_basis * (lmax + 1)^2]
        
        # Update with linear layer
        x = self.update_linear(aggregated)  # [n_atoms, n_atom_basis * (lmax + 1)^2]
        x = torch.relu(x)
        x = self.reduce_layer(x)  # [n_atoms, n_atom_basis]
        return x

class MACEInspiredSequential(nn.Module):
    def __init__(self, n_atom_basis, n_radial, cutoff, lmax=2, n_interactions=3):
        super(MACEInspiredSequential, self).__init__()
        self.n_atom_basis = n_atom_basis
        self.lmax = lmax
        self.rbf = spk.nn.GaussianRBF(n_rbf=n_radial, cutoff=cutoff)
        
        # PaiNN: 1 layer
        self.painn = spk.representation.PaiNN(
            n_atom_basis=n_atom_basis,
            n_interactions=1,
            radial_basis=self.rbf,
            cutoff_fn=spk.nn.CosineCutoff(cutoff)
        )
        
        # MACE-like e3nn: 2 layers
        self.mace_layers = nn.ModuleList([
            MACEInspiredLayer(n_atom_basis, n_radial, lmax)
            for _ in range(2)
        ])
        
        self.register_buffer("cutoff", torch.tensor(cutoff, dtype=torch.float32))
        self.cutoff_fn = spk.nn.CosineCutoff(cutoff=self.cutoff)

    def forward(self, inputs):
        # PaiNN
        inputs = self.painn(inputs)
        x = inputs['scalar_representation']  # [n_atoms, n_atom_basis]
        
        # Prepare
        rij = inputs['_Rij']
        radial_ij_raw = torch.norm(rij, dim=-1)
        radial_ij = self.rbf(radial_ij_raw)
        sh = o3.SphericalHarmonics(range(self.lmax + 1), normalize=True, normalization='component')
        dir_ij = rij / (radial_ij_raw.unsqueeze(-1) + 1e-8)
        dir_ij_sh = sh(dir_ij)
        idx_i = inputs['_idx_i']
        idx_j = inputs['_idx_j']
        n_atoms = inputs['_n_atoms'].sum().item()
        
        # MACE-like layers
        for layer in self.mace_layers:
            x = layer(x, dir_ij_sh, radial_ij, idx_i, idx_j, n_atoms)  # [n_atoms, n_atom_basis]
        
        inputs['scalar_representation'] = x
        return inputs


# ******************************************************
class EquivariantAttention(nn.Module):
    def __init__(self, n_atom_basis, lmax):
        super(EquivariantAttention, self).__init__()
        self.n_atom_basis = n_atom_basis
        self.lmax = lmax
        irreps_in = o3.Irreps(f"{n_atom_basis}x0e")
        irreps_sh = o3.Irreps(" + ".join([f"1x{l}e" for l in range(lmax + 1)]))
        irreps_out = o3.Irreps(f"{n_atom_basis}x0e + {n_atom_basis}x1e + {n_atom_basis}x2e")
        
        instructions = [(0, l, l, "uvu", True) for l in range(lmax + 1)]
        
        self.query_tp = o3.TensorProduct(
            irreps_in1=irreps_in,
            irreps_in2=irreps_sh,
            irreps_out=irreps_out,
            instructions=instructions,
            internal_weights=True,
            shared_weights=True
        )
        self.key_tp = o3.TensorProduct(
            irreps_in1=irreps_in,
            irreps_in2=irreps_sh,
            irreps_out=irreps_out,
            instructions=instructions,
            internal_weights=True,
            shared_weights=True
        )
        self.value_tp = o3.TensorProduct(
            irreps_in1=irreps_in,
            irreps_in2=irreps_sh,
            irreps_out=irreps_out,
            instructions=instructions,
            internal_weights=True,
            shared_weights=True
        )
        
        self.scale = 1.0 / (n_atom_basis ** 0.5)
        self.scalar_proj = nn.Linear(n_atom_basis * (lmax + 1) ** 2, n_atom_basis)

    def forward(self, x, dir_ij_sh, idx_i, idx_j, n_atoms):
        query = self.query_tp(x[idx_i], dir_ij_sh)  # [n_neighbors, n_atom_basis * (lmax+1)^2]
        key = self.key_tp(x[idx_j], dir_ij_sh)     # [n_neighbors, n_atom_basis * (lmax+1)^2]
        value = self.value_tp(x[idx_j], dir_ij_sh) # [n_neighbors, n_atom_basis * (lmax+1)^2]
        
        query_scalar = self.scalar_proj(query)
        key_scalar = self.scalar_proj(key)
        
        scores = torch.einsum("ni,nj->n", query_scalar, key_scalar) * self.scale
        scores = torch.softmax(scores, dim=0)
        
        weighted_values = value * scores.unsqueeze(-1)
        output = snn.scatter_add(weighted_values, idx_i, dim_size=n_atoms, dim=0)
        output = self.scalar_proj(output)  # [n_atoms, n_atom_basis]
        return output

class AttentionFusionHybrid(nn.Module):
    def __init__(self, n_atom_basis, n_radial, cutoff, lmax=2):
        super(AttentionFusionHybrid, self).__init__()
        self.n_atom_basis = n_atom_basis
        self.lmax = lmax
        self.rbf = spk.nn.GaussianRBF(n_rbf=n_radial, cutoff=cutoff)
        
        # PaiNN: 1 layer
        self.painn = spk.representation.PaiNN(
            n_atom_basis=n_atom_basis,
            n_interactions=1,
            radial_basis=self.rbf,
            cutoff_fn=spk.nn.CosineCutoff(cutoff)
        )
        
        # SchNet: 1 layer
        self.schnet = spk.representation.SchNet(
            n_atom_basis=n_atom_basis,
            n_interactions=1,
            radial_basis=self.rbf,
            cutoff_fn=spk.nn.CosineCutoff(cutoff)
        )
        
        # Equivariant Attention + e3nn
        self.attention = EquivariantAttention(n_atom_basis, lmax)
        irreps_in = o3.Irreps(f"{n_atom_basis}x0e")
        irreps_sh = o3.Irreps(" + ".join([f"1x{l}e" for l in range(lmax + 1)]))
        irreps_out = o3.Irreps(f"{n_atom_basis}x0e + {n_atom_basis}x1e + {n_atom_basis}x2e")
        instructions = [(0, l, l, "uvu", True) for l in range(lmax + 1)]
        self.e3nn_interaction = o3.TensorProduct(
            irreps_in1=irreps_in,
            irreps_in2=irreps_sh,
            irreps_out=irreps_out,
            instructions=instructions,
            internal_weights=True,
            shared_weights=True
        )
        
        self.e3nn_embedding = nn.Embedding(100, n_atom_basis)
        self.e3nn_radial_net = FullyConnectedNet([n_radial, n_atom_basis * (lmax + 1) ** 2], act=torch.nn.ReLU())
        
        # Projection layers for fusion
        proj_dim = n_atom_basis // 2  # e.g., 96
        self.painn_proj = nn.Linear(n_atom_basis, proj_dim)
        self.schnet_proj = nn.Linear(n_atom_basis, proj_dim)
        self.attention_proj = nn.Linear(n_atom_basis, proj_dim)
        
        # Fusion layer
        self.fusion_layer = nn.Sequential(
            nn.Linear(proj_dim * 3, n_atom_basis),
            nn.ReLU(),
            nn.Linear(n_atom_basis, n_atom_basis)
        )
        
        self.register_buffer("cutoff", torch.tensor(cutoff, dtype=torch.float32))
        self.cutoff_fn = spk.nn.CosineCutoff(cutoff=self.cutoff)

    def forward(self, inputs):
        inputs_orig = inputs.copy()
        
        # PaiNN
        inputs_painn = self.painn(inputs)
        painn_features = inputs_painn['scalar_representation']  # [n_atoms, 192]
        painn_features = self.painn_proj(painn_features)  # [n_atoms, 96]
        
        # SchNet
        inputs_schnet = inputs_orig.copy()
        inputs_schnet = self.schnet(inputs_schnet)
        schnet_features = inputs_schnet['scalar_representation']  # [n_atoms, 192]
        schnet_features = self.schnet_proj(schnet_features)  # [n_atoms, 96]
        
        # Attention + e3nn
        x = self.e3nn_embedding(inputs_orig['_atomic_numbers'])  # [n_atoms, 192]
        rij = inputs_orig['_Rij']
        radial_ij_raw = torch.norm(rij, dim=-1)
        radial_ij = self.rbf(radial_ij_raw)
        sh = o3.SphericalHarmonics(range(self.lmax + 1), normalize=True, normalization='component')
        dir_ij = rij / (radial_ij_raw.unsqueeze(-1) + 1e-8)
        dir_ij_sh = sh(dir_ij)
        idx_i = inputs_orig['_idx_i']
        idx_j = inputs_orig['_idx_j']
        n_atoms = inputs_orig['_n_atoms'].sum().item()
        
        attention_features = self.attention(x, dir_ij_sh, idx_i, idx_j, n_atoms)  # [n_atoms, 192]
        attention_features = self.attention_proj(attention_features)  # [n_atoms, 96]
        
        # Fuse
        fused_features = torch.cat([painn_features, schnet_features, attention_features], dim=-1)  # [n_atoms, 96*3]
        x = self.fusion_layer(fused_features)  # [n_atoms, 192]
        
        inputs['scalar_representation'] = x
        return inputs




class EquivariantTransformer(nn.Module):
    def __init__(self, n_atom_basis, lmax, num_heads=4):
        super(EquivariantTransformer, self).__init__()
        self.n_atom_basis = n_atom_basis
        self.lmax = lmax
        self.num_heads = num_heads
        irreps_in = o3.Irreps(f"{n_atom_basis}x0e")
        irreps_sh = o3.Irreps(" + ".join([f"1x{l}e" for l in range(lmax + 1)]))
        irreps_out = o3.Irreps(f"{n_atom_basis}x0e + {n_atom_basis}x1e + {n_atom_basis}x2e")
        
        instructions = [(0, l, l, "uvu", True) for l in range(lmax + 1)]
        
        # Multi-head attention
        self.query_tps = nn.ModuleList([
            o3.TensorProduct(
                irreps_in1=irreps_in,
                irreps_in2=irreps_sh,
                irreps_out=irreps_out,
                instructions=instructions,
                internal_weights=True,
                shared_weights=True
            ) for _ in range(num_heads)
        ])
        self.key_tps = nn.ModuleList([
            o3.TensorProduct(
                irreps_in1=irreps_in,
                irreps_in2=irreps_sh,
                irreps_out=irreps_out,
                instructions=instructions,
                internal_weights=True,
                shared_weights=True
            ) for _ in range(num_heads)
        ])
        self.value_tps = nn.ModuleList([
            o3.TensorProduct(
                irreps_in1=irreps_in,
                irreps_in2=irreps_sh,
                irreps_out=irreps_out,
                instructions=instructions,
                internal_weights=True,
                shared_weights=True
            ) for _ in range(num_heads)
        ])
        
        self.scale = 1.0 / (n_atom_basis ** 0.5)
        self.head_proj = nn.Linear(n_atom_basis * (lmax + 1) ** 2, n_atom_basis // num_heads)
        self.concat_proj = nn.Linear(n_atom_basis, n_atom_basis)

    def forward(self, x, dir_ij_sh, idx_i, idx_j, n_atoms):
        head_outputs = []
        for h in range(self.num_heads):
            query = self.query_tps[h](x[idx_i], dir_ij_sh)  # [n_neighbors, n_atom_basis * (lmax+1)^2]
            key = self.key_tps[h](x[idx_j], dir_ij_sh)     # [n_neighbors, n_atom_basis * (lmax+1)^2]
            value = self.value_tps[h](x[idx_j], dir_ij_sh) # [n_neighbors, n_atom_basis * (lmax+1)^2]
            
            query_scalar = self.head_proj(query)  # [n_neighbors, n_atom_basis // num_heads]
            key_scalar = self.head_proj(key)
            
            scores = torch.einsum("ni,nj->n", query_scalar, key_scalar) * self.scale
            scores = torch.softmax(scores, dim=0)
            
            weighted_values = value * scores.unsqueeze(-1)
            head_output = snn.scatter_add(weighted_values, idx_i, dim_size=n_atoms, dim=0)  # [n_atoms, n_atom_basis * (lmax+1)^2]
            head_output = self.head_proj(head_output)  # [n_atoms, n_atom_basis // num_heads]
            head_outputs.append(head_output)
        
        # Concatenate heads
        output = torch.cat(head_outputs, dim=-1)  # [n_atoms, n_atom_basis]
        output = self.concat_proj(output)  # [n_atoms, n_atom_basis]
        return output

class CustomEquiTransformer(nn.Module):
    def __init__(self, n_atom_basis, n_radial, cutoff, lmax=2):
        super(CustomEquiTransformer, self).__init__()
        self.n_atom_basis = n_atom_basis
        self.lmax = lmax
        self.rbf = spk.nn.GaussianRBF(n_rbf=n_radial, cutoff=cutoff)
        
        # PaiNN: 1 layer
        self.painn = spk.representation.PaiNN(
            n_atom_basis=n_atom_basis,
            n_interactions=1,
            radial_basis=self.rbf,
            cutoff_fn=spk.nn.CosineCutoff(cutoff)
        )
        
        # Equivariant Transformer: 1 layer
        self.transformer = EquivariantTransformer(n_atom_basis, lmax, num_heads=4)
        
        # e3nn SO3: 1 layer
        irreps_in = o3.Irreps(f"{n_atom_basis}x0e")
        irreps_sh = o3.Irreps(" + ".join([f"1x{l}e" for l in range(lmax + 1)]))
        irreps_out = o3.Irreps(f"{n_atom_basis}x0e + {n_atom_basis}x1e + {n_atom_basis}x2e")
        instructions = [(0, l, l, "uvu", True) for l in range(lmax + 1)]
        self.e3nn_interaction = o3.TensorProduct(
            irreps_in1=irreps_in,
            irreps_in2=irreps_sh,
            irreps_out=irreps_out,
            instructions=instructions,
            internal_weights=True,
            shared_weights=True
        )
        
        self.e3nn_radial_net = FullyConnectedNet([n_radial, n_atom_basis * (lmax + 1) ** 2], act=torch.nn.ReLU())
        self.reduce_layer = nn.Linear(n_atom_basis * (lmax + 1) ** 2, n_atom_basis)
        
        self.register_buffer("cutoff", torch.tensor(cutoff, dtype=torch.float32))
        self.cutoff_fn = spk.nn.CosineCutoff(cutoff=self.cutoff)

    def forward(self, inputs):
        # PaiNN
        inputs_painn = self.painn(inputs)
        x = inputs_painn['scalar_representation']  # [n_atoms, n_atom_basis]
        
        # Prepare
        rij = inputs['_Rij']
        radial_ij_raw = torch.norm(rij, dim=-1)
        radial_ij = self.rbf(radial_ij_raw)
        sh = o3.SphericalHarmonics(range(self.lmax + 1), normalize=True, normalization='component')
        dir_ij = rij / (radial_ij_raw.unsqueeze(-1) + 1e-8)
        dir_ij_sh = sh(dir_ij)
        idx_i = inputs['_idx_i']
        idx_j = inputs['_idx_j']
        n_atoms = inputs['_n_atoms'].sum().item()
        
        # Transformer
        x = self.transformer(x, dir_ij_sh, idx_i, idx_j, n_atoms)  # [n_atoms, n_atom_basis]
        
        # e3nn SO3
        radial_weights = self.e3nn_radial_net(radial_ij).view(-1, (self.lmax + 1) ** 2, self.n_atom_basis)
        neighbor_contrib = self.e3nn_interaction(x[idx_j], dir_ij_sh)
        neighbor_contrib = neighbor_contrib.view(-1, (self.lmax + 1) ** 2, self.n_atom_basis)
        neighbor_contrib = neighbor_contrib * radial_weights * self.cutoff_fn(radial_ij_raw).unsqueeze(-1).unsqueeze(-1)
        neighbor_contrib = neighbor_contrib.view(-1, self.n_atom_basis * (self.lmax + 1) ** 2)
        x = snn.scatter_add(neighbor_contrib, idx_i, dim_size=n_atoms, dim=0)  # [n_atoms, n_atom_basis * (lmax+1)^2]
        
        # Reduce
        x = self.reduce_layer(x)  # [n_atoms, n_atom_basis]
        inputs['scalar_representation'] = x
        return inputs


class EnhancedTunedHybridPaiNNE3nn(nn.Module):
    def __init__(self, n_atom_basis, n_radial, cutoff, lmax=2, n_interactions=3):
        super(EnhancedTunedHybridPaiNNE3nn, self).__init__()
        self.n_atom_basis = n_atom_basis
        self.lmax = lmax
        self.rbf = spk.nn.GaussianRBF(n_rbf=n_radial, cutoff=cutoff)
        
        # PaiNN: 1 layer
        self.painn = spk.representation.PaiNN(
            n_atom_basis=n_atom_basis,
            n_interactions=1,
            radial_basis=self.rbf,
            cutoff_fn=spk.nn.CosineCutoff(cutoff)
        )
        
        # e3nn SO3: 2 layers
        self.n_e3nn = max(0, n_interactions - 1)
        irreps_in = o3.Irreps(f"{n_atom_basis}x0e")
        irreps_sh = o3.Irreps(" + ".join([f"1x{l}e" for l in range(lmax + 1)]))
        irreps_out = o3.Irreps(f"{n_atom_basis}x0e + {n_atom_basis}x1e + {n_atom_basis}x2e")
        instructions = [(0, l, l, "uvu", True) for l in range(lmax + 1)]
        
        self.e3nn_interactions = nn.ModuleList([
            o3.TensorProduct(
                irreps_in1=irreps_in if i == 0 else irreps_out,
                irreps_in2=irreps_sh,
                irreps_out=irreps_out,
                instructions=instructions,
                internal_weights=True,
                shared_weights=True
            )
            for i in range(self.n_e3nn)
        ])
        
        self.radial_net = FullyConnectedNet([n_radial, 64, n_atom_basis * (lmax + 1) ** 2], act=torch.nn.ReLU())
        self.reduce_layer = nn.Linear(n_atom_basis * (lmax + 1) ** 2, n_atom_basis)
        self.register_buffer("cutoff", torch.tensor(cutoff, dtype=torch.float32))
        self.cutoff_fn = spk.nn.CosineCutoff(cutoff=self.cutoff)

    def forward(self, inputs):
        # PaiNN
        inputs = self.painn(inputs)
        x = inputs['scalar_representation']  # [n_atoms, n_atom_basis]
        
        # Prepare
        rij = inputs['_Rij']
        radial_ij_raw = torch.norm(rij, dim=-1)
        radial_ij = self.rbf(radial_ij_raw)
        sh = o3.SphericalHarmonics(range(self.lmax + 1), normalize=True, normalization='component')
        dir_ij = rij / (radial_ij_raw.unsqueeze(-1) + 1e-8)
        dir_ij_sh = sh(dir_ij)
        cutoff_ij = self.cutoff_fn(radial_ij_raw).unsqueeze(-1).unsqueeze(-1)
        idx_i = inputs['_idx_i']
        idx_j = inputs['_idx_j']
        n_atoms = inputs['_n_atoms'].sum().item()
        
        # e3nn SO3 layers
        radial_weights = self.radial_net(radial_ij)
        radial_weights = radial_weights.view(-1, (self.lmax + 1) ** 2, self.n_atom_basis)
        
        for interaction in self.e3nn_interactions:
            neighbor_contrib = interaction(x[idx_j], dir_ij_sh)
            neighbor_contrib = neighbor_contrib.view(-1, (self.lmax + 1) ** 2, self.n_atom_basis)
            neighbor_contrib = neighbor_contrib * radial_weights * cutoff_ij
            neighbor_contrib = neighbor_contrib.view(-1, (self.lmax + 1) ** 2 * self.n_atom_basis)
            x = snn.scatter_add(neighbor_contrib, idx_i, dim_size=n_atoms, dim=0)
        
        # Reduce
        x = self.reduce_layer(x)
        inputs['scalar_representation'] = x
        return inputs




class EquivariantAttention(nn.Module):
    def __init__(self, n_atom_basis, lmax, num_heads=2):
        super(EquivariantAttention, self).__init__()
        self.n_atom_basis = n_atom_basis
        self.lmax = lmax
        self.num_heads = num_heads
        irreps_in = o3.Irreps(f"{n_atom_basis}x0e")
        irreps_sh = o3.Irreps(" + ".join([f"1x{l}e" for l in range(lmax + 1)]))
        irreps_out = o3.Irreps(f"{n_atom_basis}x0e + {n_atom_basis}x1e + {n_atom_basis}x2e")
        
        # Instructions for tensor product
        instructions = [(0, l, l, "uvu", True) for l in range(lmax + 1)]
        
        # Multi-head attention
        self.query_tps = nn.ModuleList([
            o3.TensorProduct(
                irreps_in1=irreps_in,
                irreps_in2=irreps_sh,
                irreps_out=irreps_out,
                instructions=instructions,
                internal_weights=True,
                shared_weights=True
            ) for _ in range(num_heads)
        ])
        self.key_tps = nn.ModuleList([
            o3.TensorProduct(
                irreps_in1=irreps_in,
                irreps_in2=irreps_sh,
                irreps_out=irreps_out,
                instructions=instructions,
                internal_weights=True,
                shared_weights=True
            ) for _ in range(num_heads)
        ])
        self.value_tps = nn.ModuleList([
            o3.TensorProduct(
                irreps_in1=irreps_in,
                irreps_in2=irreps_sh,
                irreps_out=irreps_out,
                instructions=instructions,
                internal_weights=True,
                shared_weights=True
            ) for _ in range(num_heads)
        ])
        
        self.scale = 1.0 / (n_atom_basis ** 0.5)
        self.head_proj = nn.Linear(n_atom_basis * (lmax + 1) ** 2, n_atom_basis // num_heads)
        self.concat_proj = nn.Linear(n_atom_basis, n_atom_basis)

    def forward(self, x, dir_ij_sh, idx_i, idx_j, n_atoms):
        head_outputs = []
        for h in range(self.num_heads):
            query = self.query_tps[h](x[idx_i], dir_ij_sh)  # [n_neighbors, n_atom_basis * (lmax+1)^2]
            key = self.key_tps[h](x[idx_j], dir_ij_sh)     # [n_neighbors, n_atom_basis * (lmax+1)^2]
            value = self.value_tps[h](x[idx_j], dir_ij_sh) # [n_neighbors, n_atom_basis * (lmax+1)^2]
            
            query_scalar = self.head_proj(query)  # [n_neighbors, n_atom_basis // num_heads]
            key_scalar = self.head_proj(key)
            
            scores = torch.einsum("ni,nj->n", query_scalar, key_scalar) * self.scale
            scores = torch.softmax(scores, dim=0)
            
            weighted_values = value * scores.unsqueeze(-1)
            head_output = snn.scatter_add(weighted_values, idx_i, dim_size=n_atoms, dim=0)
            head_output = self.head_proj(head_output)  # [n_atoms, n_atom_basis // num_heads]
            head_outputs.append(head_output)
        
        output = torch.cat(head_outputs, dim=-1)  # [n_atoms, n_atom_basis]
        output = self.concat_proj(output)  # [n_atoms, n_atom_basis]
        return output

class OptimizedAttentionFusionHybrid(nn.Module):
    def __init__(self, n_atom_basis, n_radial, cutoff, lmax=2):
        super(OptimizedAttentionFusionHybrid, self).__init__()
        self.n_atom_basis = n_atom_basis
        self.lmax = lmax
        self.rbf = spk.nn.GaussianRBF(n_rbf=n_radial, cutoff=cutoff)
        
        # PaiNN: 1 layer
        self.painn = spk.representation.PaiNN(
            n_atom_basis=n_atom_basis,
            n_interactions=1,
            radial_basis=self.rbf,
            cutoff_fn=spk.nn.CosineCutoff(cutoff)
        )
        
        # SchNet: 1 layer
        self.schnet = spk.representation.SchNet(
            n_atom_basis=n_atom_basis,
            n_interactions=1,
            radial_basis=self.rbf,
            cutoff_fn=spk.nn.CosineCutoff(cutoff)
        )
        
        # Equivariant Attention
        self.attention = EquivariantAttention(n_atom_basis, lmax, num_heads=2)
        
        # Projection layers for fusion
        proj_dim = n_atom_basis // 2  # e.g., 64
        self.painn_proj = nn.Linear(n_atom_basis, proj_dim)
        self.schnet_proj = nn.Linear(n_atom_basis, proj_dim)
        self.attention_proj = nn.Linear(n_atom_basis, proj_dim)
        
        # Fusion layer
        self.fusion_layer = nn.Sequential(
            nn.Linear(proj_dim * 3, n_atom_basis),
            nn.ReLU(),
            nn.Linear(n_atom_basis, n_atom_basis)
        )
        
        self.register_buffer("cutoff", torch.tensor(cutoff, dtype=torch.float32))
        self.cutoff_fn = spk.nn.CosineCutoff(cutoff=self.cutoff)

    def forward(self, inputs):
        inputs_orig = inputs.copy()
        
        # PaiNN
        inputs_painn = self.painn(inputs)
        painn_features = inputs_painn['scalar_representation']  # [n_atoms, 128]
        painn_features_proj = self.painn_proj(painn_features)  # [n_atoms, 64]
        
        # SchNet
        inputs_schnet = inputs_orig.copy()
        inputs_schnet = self.schnet(inputs_schnet)
        schnet_features = inputs_schnet['scalar_representation']  # [n_atoms, 128]
        schnet_features_proj = self.schnet_proj(schnet_features)  # [n_atoms, 64]
        
        # Attention
        rij = inputs_orig['_Rij']
        radial_ij_raw = torch.norm(rij, dim=-1)
        radial_ij = self.rbf(radial_ij_raw)
        sh = o3.SphericalHarmonics(range(self.lmax + 1), normalize=True, normalization='component')
        dir_ij = rij / (radial_ij_raw.unsqueeze(-1) + 1e-8)
        dir_ij_sh = sh(dir_ij)
        idx_i = inputs_orig['_idx_i']
        idx_j = inputs_orig['_idx_j']
        n_atoms = inputs_orig['_n_atoms'].sum().item()
        
        attention_features = self.attention(painn_features, dir_ij_sh, idx_i, idx_j, n_atoms)  # [n_atoms, 128]
        attention_features_proj = self.attention_proj(attention_features)  # [n_atoms, 64]
        
        # Fuse
        fused_features = torch.cat([painn_features_proj, schnet_features_proj, attention_features_proj], dim=-1)  # [n_atoms, 64*3]
        x = self.fusion_layer(fused_features)  # [n_atoms, 128]
        
        inputs['scalar_representation'] = x
        return inputs


class FinalTunedHybridPaiNNE3nn(nn.Module):
    def __init__(self, n_atom_basis, n_radial, cutoff, lmax=2, n_interactions=3):
        super(FinalTunedHybridPaiNNE3nn, self).__init__()
        self.n_atom_basis = n_atom_basis
        self.lmax = lmax
        self.rbf = spk.nn.GaussianRBF(n_rbf=n_radial, cutoff=cutoff)
        
        # PaiNN: 1 layer
        self.painn = spk.representation.PaiNN(
            n_atom_basis=n_atom_basis,
            n_interactions=1,
            radial_basis=self.rbf,
            cutoff_fn=spk.nn.CosineCutoff(cutoff)
        )
        
        # e3nn SO3: 2 layers
        self.n_e3nn = max(0, n_interactions - 1)
        irreps_in = o3.Irreps(f"{n_atom_basis}x0e")
        irreps_sh = o3.Irreps(" + ".join([f"1x{l}e" for l in range(lmax + 1)]))
        irreps_out = o3.Irreps(f"{n_atom_basis}x0e + {n_atom_basis}x1e + {n_atom_basis}x2e")
        instructions = [(0, l, l, "uvu", True) for l in range(lmax + 1)]
        
        self.e3nn_interactions = nn.ModuleList([
            o3.TensorProduct(
                irreps_in1=irreps_in if i == 0 else irreps_out,
                irreps_in2=irreps_sh,
                irreps_out=irreps_out,
                instructions=instructions,
                internal_weights=True,
                shared_weights=True
            )
            for i in range(self.n_e3nn)
        ])
        
        self.radial_net = FullyConnectedNet([n_radial, 32, n_atom_basis * (lmax + 1) ** 2], act=torch.nn.ReLU())
        self.reduce_layer = nn.Linear(n_atom_basis * (lmax + 1) ** 2, n_atom_basis)
        self.register_buffer("cutoff", torch.tensor(cutoff, dtype=torch.float32))
        self.cutoff_fn = spk.nn.CosineCutoff(cutoff=self.cutoff)

    def forward(self, inputs):
        # PaiNN
        inputs = self.painn(inputs)
        x = inputs['scalar_representation']  # [n_atoms, n_atom_basis]
        
        # Prepare
        rij = inputs['_Rij']
        radial_ij_raw = torch.norm(rij, dim=-1)
        radial_ij = self.rbf(radial_ij_raw)
        sh = o3.SphericalHarmonics(range(self.lmax + 1), normalize=True, normalization='component')
        dir_ij = rij / (radial_ij_raw.unsqueeze(-1) + 1e-8)
        dir_ij_sh = sh(dir_ij)
        cutoff_ij = self.cutoff_fn(radial_ij_raw).unsqueeze(-1).unsqueeze(-1)
        idx_i = inputs['_idx_i']
        idx_j = inputs['_idx_j']
        n_atoms = inputs['_n_atoms'].sum().item()
        
        # e3nn SO3 layers
        radial_weights = self.radial_net(radial_ij)
        radial_weights = radial_weights.view(-1, (self.lmax + 1) ** 2, self.n_atom_basis)
        
        for interaction in self.e3nn_interactions:
            neighbor_contrib = interaction(x[idx_j], dir_ij_sh)
            neighbor_contrib = neighbor_contrib.view(-1, (self.lmax + 1) ** 2, self.n_atom_basis)
            neighbor_contrib = neighbor_contrib * radial_weights * cutoff_ij
            neighbor_contrib = neighbor_contrib.view(-1, (self.lmax + 1) ** 2 * self.n_atom_basis)
            x = snn.scatter_add(neighbor_contrib, idx_i, dim_size=n_atoms, dim=0)
        
        # Reduce
        x = self.reduce_layer(x)
        inputs['scalar_representation'] = x
        return inputs


class TunedOptimizedHybridPaiNNE3nn(nn.Module):
    def __init__(self, n_atom_basis, n_radial, cutoff, lmax=2, n_interactions=3):
        super(TunedOptimizedHybridPaiNNE3nn, self).__init__()
        self.n_atom_basis = n_atom_basis
        self.lmax = lmax
        self.rbf = spk.nn.GaussianRBF(n_rbf=n_radial, cutoff=cutoff)
        
        # PaiNN: 1 layer
        self.painn = spk.representation.PaiNN(
            n_atom_basis=n_atom_basis,
            n_interactions=1,
            radial_basis=self.rbf,
            cutoff_fn=spk.nn.CosineCutoff(cutoff)
        )
        
        # e3nn SO3: 2 layers
        self.n_e3nn = max(0, n_interactions - 1)
        irreps_in = o3.Irreps(f"{n_atom_basis}x0e")
        irreps_sh = o3.Irreps(" + ".join([f"1x{l}e" for l in range(lmax + 1)]))
        irreps_out = o3.Irreps(f"{n_atom_basis}x0e + {n_atom_basis}x1e + {n_atom_basis}x2e")
        instructions = [(0, l, l, "uvu", True) for l in range(lmax + 1)]
        
        self.e3nn_interactions = nn.ModuleList([
            o3.TensorProduct(
                irreps_in1=irreps_in if i == 0 else irreps_out,
                irreps_in2=irreps_sh,
                irreps_out=irreps_out,
                instructions=instructions,
                internal_weights=True,
                shared_weights=True
            )
            for i in range(self.n_e3nn)
        ])
        
        self.radial_net = FullyConnectedNet([n_radial, 64, n_atom_basis * (lmax + 1) ** 2], act=torch.nn.ReLU())
        self.reduce_layer = nn.Linear(n_atom_basis * (lmax + 1) ** 2, n_atom_basis)
        self.register_buffer("cutoff", torch.tensor(cutoff, dtype=torch.float32))
        self.cutoff_fn = spk.nn.CosineCutoff(cutoff=self.cutoff)

    def forward(self, inputs):
        # PaiNN
        inputs = self.painn(inputs)
        x = inputs['scalar_representation']  # [n_atoms, n_atom_basis]
        
        # Prepare
        rij = inputs['_Rij']
        radial_ij_raw = torch.norm(rij, dim=-1)
        radial_ij = self.rbf(radial_ij_raw)
        sh = o3.SphericalHarmonics(range(self.lmax + 1), normalize=True, normalization='component')
        dir_ij = rij / (radial_ij_raw.unsqueeze(-1) + 1e-8)
        dir_ij_sh = sh(dir_ij)
        cutoff_ij = self.cutoff_fn(radial_ij_raw).unsqueeze(-1).unsqueeze(-1)
        idx_i = inputs['_idx_i']
        idx_j = inputs['_idx_j']
        n_atoms = inputs['_n_atoms'].sum().item()
        
        # e3nn SO3 layers
        radial_weights = self.radial_net(radial_ij)
        radial_weights = radial_weights.view(-1, (self.lmax + 1) ** 2, self.n_atom_basis)
        
        for interaction in self.e3nn_interactions:
            neighbor_contrib = interaction(x[idx_j], dir_ij_sh)
            neighbor_contrib = neighbor_contrib.view(-1, (self.lmax + 1) ** 2, self.n_atom_basis)
            neighbor_contrib = neighbor_contrib * radial_weights * cutoff_ij
            neighbor_contrib = neighbor_contrib.view(-1, (self.lmax + 1) ** 2 * self.n_atom_basis)
            x = snn.scatter_add(neighbor_contrib, idx_i, dim_size=n_atoms, dim=0)
        
        # Reduce
        x = self.reduce_layer(x)
        inputs['scalar_representation'] = x
        return inputs


class EquivariantAttention(nn.Module):
    def __init__(self, n_atom_basis, lmax):
        super(EquivariantAttention, self).__init__()
        self.n_atom_basis = n_atom_basis
        self.lmax = lmax
        irreps_in = o3.Irreps(f"{n_atom_basis}x0e")
        irreps_sh = o3.Irreps(" + ".join([f"1x{l}e" for l in range(lmax + 1)]))
        irreps_out = o3.Irreps(f"{n_atom_basis}x0e + {n_atom_basis}x1e + {n_atom_basis}x2e")
        
        instructions = [(0, l, l, "uvu", True) for l in range(lmax + 1)]
        
        self.query_tp = o3.TensorProduct(
            irreps_in1=irreps_in,
            irreps_in2=irreps_sh,
            irreps_out=irreps_out,
            instructions=instructions,
            internal_weights=True,
            shared_weights=True
        )
        self.key_tp = o3.TensorProduct(
            irreps_in1=irreps_in,
            irreps_in2=irreps_sh,
            irreps_out=irreps_out,
            instructions=instructions,
            internal_weights=True,
            shared_weights=True
        )
        self.value_tp = o3.TensorProduct(
            irreps_in1=irreps_in,
            irreps_in2=irreps_sh,
            irreps_out=irreps_out,
            instructions=instructions,
            internal_weights=True,
            shared_weights=True
        )
        
        self.scale = 1.0 / (n_atom_basis ** 0.5)
        self.scalar_proj = nn.Linear(n_atom_basis * (lmax + 1) ** 2, n_atom_basis)

    def forward(self, x, dir_ij_sh, idx_i, idx_j, n_atoms):
        query = self.query_tp(x[idx_i], dir_ij_sh)
        key = self.key_tp(x[idx_j], dir_ij_sh)
        value = self.value_tp(x[idx_j], dir_ij_sh)
        
        query_scalar = self.scalar_proj(query)
        key_scalar = self.scalar_proj(key)
        
        scores = torch.einsum("ni,nj->n", query_scalar, key_scalar) * self.scale
        scores = torch.softmax(scores, dim=0)
        
        weighted_values = value * scores.unsqueeze(-1)
        output = snn.scatter_add(weighted_values, idx_i, dim_size=n_atoms, dim=0)
        output = self.scalar_proj(output)
        return output

class SuperHybridPaiNNE3nn(nn.Module):
    def __init__(self, n_atom_basis, n_radial, cutoff, lmax=2, n_interactions=3):
        super(SuperHybridPaiNNE3nn, self).__init__()
        self.n_atom_basis = n_atom_basis
        self.lmax = lmax
        self.rbf = spk.nn.GaussianRBF(n_rbf=n_radial, cutoff=cutoff)
        
        # PaiNN: 1 layer
        self.painn = spk.representation.PaiNN(
            n_atom_basis=n_atom_basis,
            n_interactions=1,
            radial_basis=self.rbf,
            cutoff_fn=spk.nn.CosineCutoff(cutoff)
        )
        
        # SchNet: 1 layer
        self.schnet = spk.representation.SchNet(
            n_atom_basis=n_atom_basis,
            n_interactions=1,
            radial_basis=self.rbf,
            cutoff_fn=spk.nn.CosineCutoff(cutoff)
        )
        
        # Single-head attention
        self.attention = EquivariantAttention(n_atom_basis, lmax)
        
        # e3nn SO3: 2 layers
        irreps_in = o3.Irreps(f"{n_atom_basis}x0e")
        irreps_sh = o3.Irreps(" + ".join([f"1x{l}e" for l in range(lmax + 1)]))
        irreps_out = o3.Irreps(f"{n_atom_basis}x0e + {n_atom_basis}x1e + {n_atom_basis}x2e")
        instructions = [(0, l, l, "uvu", True) for l in range(lmax + 1)]
        
        self.e3nn_interactions = nn.ModuleList([
            o3.TensorProduct(
                irreps_in1=irreps_in if i == 0 else irreps_out,
                irreps_in2=irreps_sh,
                irreps_out=irreps_out,
                instructions=instructions,
                internal_weights=True,
                shared_weights=True
            )
            for i in range(2)
        ])
        
        self.e3nn_radial_net = FullyConnectedNet([n_radial, 64, n_atom_basis * (lmax + 1) ** 2], act=torch.nn.ReLU())
        
        # Projection layers for fusion
        proj_dim = n_atom_basis // 2  # e.g., 128
        self.painn_proj = nn.Linear(n_atom_basis, proj_dim)
        self.schnet_proj = nn.Linear(n_atom_basis, proj_dim)
        self.attention_proj = nn.Linear(n_atom_basis, proj_dim)
        self.e3nn_proj = nn.Linear(n_atom_basis * (lmax + 1) ** 2, proj_dim)
        
        # Fusion layer
        self.fusion_layer = nn.Sequential(
            nn.Linear(proj_dim * 4, n_atom_basis),
            nn.ReLU(),
            nn.Linear(n_atom_basis, n_atom_basis)
        )
        
        self.register_buffer("cutoff", torch.tensor(cutoff, dtype=torch.float32))
        self.cutoff_fn = spk.nn.CosineCutoff(cutoff=self.cutoff)

    def forward(self, inputs):
        inputs_orig = inputs.copy()
        
        # PaiNN
        inputs_painn = self.painn(inputs)
        painn_features = inputs_painn['scalar_representation']  # [n_atoms, 256]
        painn_features_proj = self.painn_proj(painn_features)  # [n_atoms, 128]
        
        # SchNet
        inputs_schnet = inputs_orig.copy()
        inputs_schnet = self.schnet(inputs_schnet)
        schnet_features = inputs_schnet['scalar_representation']  # [n_atoms, 256]
        schnet_features_proj = self.schnet_proj(schnet_features)  # [n_atoms, 128]
        
        # Attention
        rij = inputs_orig['_Rij']
        radial_ij_raw = torch.norm(rij, dim=-1)
        radial_ij = self.rbf(radial_ij_raw)
        sh = o3.SphericalHarmonics(range(self.lmax + 1), normalize=True, normalization='component')
        dir_ij = rij / (radial_ij_raw.unsqueeze(-1) + 1e-8)
        dir_ij_sh = sh(dir_ij)
        idx_i = inputs_orig['_idx_i']
        idx_j = inputs_orig['_idx_j']
        n_atoms = inputs_orig['_n_atoms'].sum().item()
        
        attention_features = self.attention(painn_features, dir_ij_sh, idx_i, idx_j, n_atoms)  # [n_atoms, 256]
        attention_features_proj = self.attention_proj(attention_features)  # [n_atoms, 128]
        
        # e3nn SO3
        x = painn_features  # Start with PaiNN features
        radial_weights = self.e3nn_radial_net(radial_ij).view(-1, (self.lmax + 1) ** 2, self.n_atom_basis)
        for interaction in self.e3nn_interactions:
            neighbor_contrib = interaction(x[idx_j], dir_ij_sh)
            neighbor_contrib = neighbor_contrib.view(-1, (self.lmax + 1) ** 2, self.n_atom_basis)
            neighbor_contrib = neighbor_contrib * radial_weights * self.cutoff_fn(radial_ij_raw).unsqueeze(-1).unsqueeze(-1)
            neighbor_contrib = neighbor_contrib.view(-1, self.n_atom_basis * (self.lmax + 1) ** 2)
            x = snn.scatter_add(neighbor_contrib, idx_i, dim_size=n_atoms, dim=0)  # [n_atoms, 256 * 9]
        e3nn_features_proj = self.e3nn_proj(x)  # [n_atoms, 128]
        
        # Fuse
        fused_features = torch.cat([painn_features_proj, schnet_features_proj, attention_features_proj, e3nn_features_proj], dim=-1)  # [n_atoms, 128*4]
        x = self.fusion_layer(fused_features)  # [n_atoms, 256]
        
        inputs['scalar_representation'] = x
        return inputs



# ignore time 

class GatedE3nnBlock(nn.Module):
    def __init__(self, n_atom_basis, lmax):
        super(GatedE3nnBlock, self).__init__()
        self.n_atom_basis = n_atom_basis
        self.lmax = lmax
        irreps_in = o3.Irreps(f"{n_atom_basis}x0e")
        irreps_sh = o3.Irreps(" + ".join([f"1x{l}e" for l in range(lmax + 1)]))
        irreps_out = o3.Irreps(f"{n_atom_basis}x0e + {n_atom_basis}x1e + {n_atom_basis}x2e")
        
        instructions = [(0, l, l, "uvu", True) for l in range(lmax + 1)]
        
        self.tp = o3.TensorProduct(
            irreps_in1=irreps_in,
            irreps_in2=irreps_sh,
            irreps_out=irreps_out,
            instructions=instructions,
            internal_weights=True,
            shared_weights=True
        )
        
        self.gate = nn.Sequential(
            nn.Linear(n_atom_basis * (lmax + 1) ** 2, n_atom_basis * (lmax + 1) ** 2),
            nn.Sigmoid()
        )

    def forward(self, x, dir_ij_sh, idx_i, idx_j, n_atoms):
        neighbor_contrib = self.tp(x[idx_j], dir_ij_sh)
        neighbor_contrib = neighbor_contrib.view(-1, (self.lmax + 1) ** 2, self.n_atom_basis)
        neighbor_contrib = neighbor_contrib.view(-1, self.n_atom_basis * (self.lmax + 1) ** 2)
        output = snn.scatter_add(neighbor_contrib, idx_i, dim_size=n_atoms, dim=0)
        gate_weights = self.gate(output)
        output = output * gate_weights
        return output

class EquivariantAttention(nn.Module):
    def __init__(self, n_atom_basis, lmax):
        super(EquivariantAttention, self).__init__()
        self.n_atom_basis = n_atom_basis
        self.lmax = lmax
        irreps_in = o3.Irreps(f"{n_atom_basis}x0e")
        irreps_sh = o3.Irreps(" + ".join([f"1x{l}e" for l in range(lmax + 1)]))
        irreps_out = o3.Irreps(f"{n_atom_basis}x0e + {n_atom_basis}x1e + {n_atom_basis}x2e")
        
        instructions = [(0, l, l, "uvu", True) for l in range(lmax + 1)]
        
        self.query_tp = o3.TensorProduct(
            irreps_in1=irreps_in,
            irreps_in2=irreps_sh,
            irreps_out=irreps_out,
            instructions=instructions,
            internal_weights=True,
            shared_weights=True
        )
        self.key_tp = o3.TensorProduct(
            irreps_in1=irreps_in,
            irreps_in2=irreps_sh,
            irreps_out=irreps_out,
            instructions=instructions,
            internal_weights=True,
            shared_weights=True
        )
        self.value_tp = o3.TensorProduct(
            irreps_in1=irreps_in,
            irreps_in2=irreps_sh,
            irreps_out=irreps_out,
            instructions=instructions,
            internal_weights=True,
            shared_weights=True
        )
        
        self.scale = 1.0 / (n_atom_basis ** 0.5)
        self.scalar_proj = nn.Linear(n_atom_basis * (lmax + 1) ** 2, n_atom_basis)
        self.gate = nn.Sequential(
            nn.Linear(n_atom_basis, n_atom_basis),
            nn.Sigmoid()
        )

    def forward(self, x, dir_ij_sh, idx_i, idx_j, n_atoms):
        query = self.query_tp(x[idx_i], dir_ij_sh)
        key = self.key_tp(x[idx_j], dir_ij_sh)
        value = self.value_tp(x[idx_j], dir_ij_sh)
        
        query_scalar = self.scalar_proj(query)
        key_scalar = self.scalar_proj(key)
        
        scores = torch.einsum("ni,nj->n", query_scalar, key_scalar) * self.scale
        scores = torch.softmax(scores, dim=0)
        
        weighted_values = value * scores.unsqueeze(-1)
        output = snn.scatter_add(weighted_values, idx_i, dim_size=n_atoms, dim=0)
        output = self.scalar_proj(output)
        gate_weights = self.gate(output)
        output = output * gate_weights
        return output

class UltraHybridPaiNNE3nn(nn.Module):
    def __init__(self, n_atom_basis, n_radial, cutoff, lmax=2):
        super(UltraHybridPaiNNE3nn, self).__init__()
        self.n_atom_basis = n_atom_basis
        self.lmax = lmax
        self.rbf = spk.nn.GaussianRBF(n_rbf=n_radial, cutoff=cutoff)
        
        # PaiNN: 1 layer
        self.painn = spk.representation.PaiNN(
            n_atom_basis=n_atom_basis,
            n_interactions=1,
            radial_basis=self.rbf,
            cutoff_fn=spk.nn.CosineCutoff(cutoff)
        )
        
        # SchNet: 1 layer
        self.schnet = spk.representation.SchNet(
            n_atom_basis=n_atom_basis,
            n_interactions=1,
            radial_basis=self.rbf,
            cutoff_fn=spk.nn.CosineCutoff(cutoff)
        )
        
        # Gated e3nn block
        self.gated_e3nn = GatedE3nnBlock(n_atom_basis, lmax)
        
        # Single-head attention
        self.attention = EquivariantAttention(n_atom_basis, lmax)
        
        # e3nn SO3: 3 layers
        irreps_in = o3.Irreps(f"{n_atom_basis}x0e")
        irreps_sh = o3.Irreps(" + ".join([f"1x{l}e" for l in range(lmax + 1)]))
        irreps_out = o3.Irreps(f"{n_atom_basis}x0e + {n_atom_basis}x1e + {n_atom_basis}x2e")
        instructions = [(0, l, l, "uvu", True) for l in range(lmax + 1)]
        
        self.e3nn_interactions = nn.ModuleList([
            o3.TensorProduct(
                irreps_in1=irreps_in if i == 0 else irreps_out,
                irreps_in2=irreps_sh,
                irreps_out=irreps_out,
                instructions=instructions,
                internal_weights=True,
                shared_weights=True
            )
            for i in range(3)
        ])
        
        self.e3nn_radial_net = FullyConnectedNet([n_radial, 64, n_atom_basis * (lmax + 1) ** 2], act=torch.nn.ReLU())
        
        # Projection layers for fusion
        proj_dim = n_atom_basis // 2  # e.g., 150
        self.painn_proj = nn.Linear(n_atom_basis, proj_dim)
        self.schnet_proj = nn.Linear(n_atom_basis, proj_dim)
        self.gated_e3nn_proj = nn.Linear(n_atom_basis * (lmax + 1) ** 2, proj_dim)
        self.attention_proj = nn.Linear(n_atom_basis, proj_dim)
        self.e3nn_seq_proj = nn.Linear(n_atom_basis * (lmax + 1) ** 2, proj_dim)
        
        # Gated fusion
        self.fusion_gate = nn.Sequential(
            nn.Linear(proj_dim * 5, proj_dim * 5),  # Output matches fused_features
            nn.Sigmoid()
        )
        self.fusion_layer = nn.Sequential(
            nn.Linear(proj_dim * 5, n_atom_basis),
            nn.ReLU(),
            nn.Linear(n_atom_basis, n_atom_basis)
        )
        
        self.register_buffer("cutoff", torch.tensor(cutoff, dtype=torch.float32))
        self.cutoff_fn = spk.nn.CosineCutoff(cutoff=self.cutoff)

    def forward(self, inputs):
        inputs_orig = inputs.copy()
        
        # PaiNN
        inputs_painn = self.painn(inputs)
        painn_features = inputs_painn['scalar_representation']  # [n_atoms, 300]
        painn_features_proj = self.painn_proj(painn_features)  # [n_atoms, 150]
        
        # SchNet
        inputs_schnet = inputs_orig.copy()
        inputs_schnet = self.schnet(inputs_schnet)
        schnet_features = inputs_schnet['scalar_representation']  # [n_atoms, 300]
        schnet_features_proj = self.schnet_proj(schnet_features)  # [n_atoms, 150]
        
        # Gated e3nn
        rij = inputs_orig['_Rij']
        radial_ij_raw = torch.norm(rij, dim=-1)
        radial_ij = self.rbf(radial_ij_raw)
        sh = o3.SphericalHarmonics(range(self.lmax + 1), normalize=True, normalization='component')
        dir_ij = rij / (radial_ij_raw.unsqueeze(-1) + 1e-8)
        dir_ij_sh = sh(dir_ij)
        idx_i = inputs_orig['_idx_i']
        idx_j = inputs_orig['_idx_j']
        n_atoms = inputs_orig['_n_atoms'].sum().item()
        
        gated_e3nn_features = self.gated_e3nn(painn_features, dir_ij_sh, idx_i, idx_j, n_atoms)  # [n_atoms, 300 * 9]
        gated_e3nn_features_proj = self.gated_e3nn_proj(gated_e3nn_features)  # [n_atoms, 150]
        
        # Attention
        attention_features = self.attention(painn_features, dir_ij_sh, idx_i, idx_j, n_atoms)  # [n_atoms, 300]
        attention_features_proj = self.attention_proj(attention_features)  # [n_atoms, 150]
        
        # e3nn SO3 sequential
        x = painn_features
        radial_weights = self.e3nn_radial_net(radial_ij).view(-1, (self.lmax + 1) ** 2, self.n_atom_basis)
        for interaction in self.e3nn_interactions:
            neighbor_contrib = interaction(x[idx_j], dir_ij_sh)
            neighbor_contrib = neighbor_contrib.view(-1, (self.lmax + 1) ** 2, self.n_atom_basis)
            neighbor_contrib = neighbor_contrib * radial_weights * self.cutoff_fn(radial_ij_raw).unsqueeze(-1).unsqueeze(-1)
            neighbor_contrib = neighbor_contrib.view(-1, self.n_atom_basis * (self.lmax + 1) ** 2)
            x = snn.scatter_add(neighbor_contrib, idx_i, dim_size=n_atoms, dim=0)
        e3nn_seq_features_proj = self.e3nn_seq_proj(x)  # [n_atoms, 150]
        
        # Gated fusion
        fused_features = torch.cat([painn_features_proj, schnet_features_proj, gated_e3nn_features_proj, attention_features_proj, e3nn_seq_features_proj], dim=-1)  # [n_atoms, 150*5]
        gate_weights = self.fusion_gate(fused_features)  # [n_atoms, 150*5]
        fused_features = fused_features * gate_weights
        x = self.fusion_layer(fused_features)  # [n_atoms, 300]
        
        inputs['scalar_representation'] = x
        return inputs



class PrimeHybridPaiNNE3nn(nn.Module):
    def __init__(self, n_atom_basis, n_radial, cutoff, lmax=2):
        super(PrimeHybridPaiNNE3nn, self).__init__()
        self.n_atom_basis = n_atom_basis
        self.lmax = lmax
        self.rbf = spk.nn.GaussianRBF(n_rbf=n_radial, cutoff=cutoff)
        
        # PaiNN: 1 layer
        self.painn = spk.representation.PaiNN(
            n_atom_basis=n_atom_basis,
            n_interactions=1,
            radial_basis=self.rbf,
            cutoff_fn=spk.nn.CosineCutoff(cutoff)
        )
        
        # NequIP-like e3nn: 1 layer
        irreps_in = o3.Irreps(f"{n_atom_basis}x0e")
        irreps_sh = o3.Irreps(" + ".join([f"1x{l}e" for l in range(lmax + 1)]))
        irreps_out = o3.Irreps(f"{n_atom_basis}x0e + {n_atom_basis}x1e + {n_atom_basis}x2e")
        instructions = [(0, l, l, "uvu", True) for l in range(lmax + 1)]
        
        self.nequip_e3nn = o3.TensorProduct(
            irreps_in1=irreps_in,
            irreps_in2=irreps_sh,
            irreps_out=irreps_out,
            instructions=instructions,
            internal_weights=True,
            shared_weights=True
        )
        
        # MACE-like e3nn: 1 layer
        self.mace_e3nn = o3.TensorProduct(
            irreps_in1=irreps_out,
            irreps_in2=irreps_sh,
            irreps_out=irreps_out,
            instructions=instructions,
            internal_weights=True,
            shared_weights=True
        )
        
        self.e3nn_radial_net = FullyConnectedNet([n_radial, 64, n_atom_basis * (lmax + 1) ** 2], act=torch.nn.ReLU())
        
        # Fusion
        self.fusion_layer = nn.Sequential(
            nn.Linear(n_atom_basis + n_atom_basis * (lmax + 1) ** 2, n_atom_basis),
            nn.ReLU(),
            nn.Linear(n_atom_basis, n_atom_basis)
        )
        
        self.register_buffer("cutoff", torch.tensor(cutoff, dtype=torch.float32))
        self.cutoff_fn = spk.nn.CosineCutoff(cutoff=self.cutoff)

    def forward(self, inputs):
        inputs_orig = inputs.copy()
        
        # PaiNN
        inputs_painn = self.painn(inputs)
        painn_features = inputs_painn['scalar_representation']  # [n_atoms, 192]
        
        # Prepare
        rij = inputs_orig['_Rij']
        radial_ij_raw = torch.norm(rij, dim=-1)
        radial_ij = self.rbf(radial_ij_raw)
        sh = o3.SphericalHarmonics(range(self.lmax + 1), normalize=True, normalization='component')
        dir_ij = rij / (radial_ij_raw.unsqueeze(-1) + 1e-8)
        dir_ij_sh = sh(dir_ij)
        idx_i = inputs_orig['_idx_i']
        idx_j = inputs_orig['_idx_j']
        n_atoms = inputs_orig['_n_atoms'].sum().item()
        
        # NequIP-like e3nn
        radial_weights = self.e3nn_radial_net(radial_ij).view(-1, (self.lmax + 1) ** 2, self.n_atom_basis)
        x = painn_features
        neighbor_contrib = self.nequip_e3nn(x[idx_j], dir_ij_sh)
        neighbor_contrib = neighbor_contrib.view(-1, (self.lmax + 1) ** 2, self.n_atom_basis)
        neighbor_contrib = neighbor_contrib * radial_weights * self.cutoff_fn(radial_ij_raw).unsqueeze(-1).unsqueeze(-1)
        neighbor_contrib = neighbor_contrib.view(-1, self.n_atom_basis * (self.lmax + 1) ** 2)
        nequip_features = snn.scatter_add(neighbor_contrib, idx_i, dim_size=n_atoms, dim=0)  # [n_atoms, 192 * 9]
        
        # MACE-like e3nn
        neighbor_contrib = self.mace_e3nn(nequip_features[idx_j], dir_ij_sh)
        neighbor_contrib = neighbor_contrib.view(-1, (self.lmax + 1) ** 2, self.n_atom_basis)
        neighbor_contrib = neighbor_contrib * radial_weights * self.cutoff_fn(radial_ij_raw).unsqueeze(-1).unsqueeze(-1)
        neighbor_contrib = neighbor_contrib.view(-1, self.n_atom_basis * (self.lmax + 1) ** 2)
        mace_features = snn.scatter_add(neighbor_contrib, idx_i, dim_size=n_atoms, dim=0)  # [n_atoms, 192 * 9]
        
        # Fusion
        fused_features = torch.cat([painn_features, mace_features], dim=-1)  # [n_atoms, 192 + 192 * 9]
        x = self.fusion_layer(fused_features)  # [n_atoms, 192]
        
        inputs['scalar_representation'] = x
        return inputs








# After Meeting


class E3nnSchNetParallelFusion(nn.Module):
    def __init__(self, n_atom_basis, n_radial, cutoff, lmax=2):
        super(E3nnSchNetParallelFusion, self).__init__()
        self.n_atom_basis = n_atom_basis
        self.lmax = lmax
        self.rbf = spk.nn.GaussianRBF(n_rbf=n_radial, cutoff=cutoff)
        
        # Embedding layer for e3nn
        self.embedding = nn.Embedding(100, n_atom_basis)
        
        # e3nn SO3: 1 layer
        irreps_in = o3.Irreps(f"{n_atom_basis}x0e")
        irreps_sh = o3.Irreps(" + ".join([f"1x{l}e" for l in range(lmax + 1)]))
        irreps_out = o3.Irreps(f"{n_atom_basis}x0e + {n_atom_basis}x1e + {n_atom_basis}x2e")
        instructions = [(0, l, l, "uvu", True) for l in range(lmax + 1)]
        
        self.e3nn = o3.TensorProduct(
            irreps_in1=irreps_in,
            irreps_in2=irreps_sh,
            irreps_out=irreps_out,
            instructions=instructions,
            internal_weights=True,
            shared_weights=True
        )
        
        self.e3nn_radial_net = FullyConnectedNet([n_radial, 64, n_atom_basis * (lmax + 1) ** 2], act=torch.nn.ReLU())
        self.e3nn_proj = nn.Linear(n_atom_basis * (lmax + 1) ** 2, n_atom_basis // 2)
        
        # SchNet: 1 layer
        self.schnet = spk.representation.SchNet(
            n_atom_basis=n_atom_basis,
            n_interactions=1,
            radial_basis=self.rbf,
            cutoff_fn=spk.nn.CosineCutoff(cutoff)
        )
        
        self.schnet_proj = nn.Linear(n_atom_basis, n_atom_basis // 2)
        
        # Fusion
        self.fusion_layer = nn.Sequential(
            nn.Linear(n_atom_basis, n_atom_basis),
            nn.ReLU(),
            nn.Linear(n_atom_basis, n_atom_basis)
        )
        
        self.register_buffer("cutoff", torch.tensor(cutoff, dtype=torch.float32))
        self.cutoff_fn = spk.nn.CosineCutoff(cutoff=self.cutoff)

    def forward(self, inputs):
        inputs_orig = inputs.copy()
        
        # Ensure all input tensors are on the same device as the model
        device = next(self.parameters()).device
        inputs_orig['_atomic_numbers'] = inputs_orig['_atomic_numbers'].to(device)
        inputs_orig['_Rij'] = inputs_orig['_Rij'].to(device)
        inputs_orig['_idx_i'] = inputs_orig['_idx_i'].to(device)
        inputs_orig['_idx_j'] = inputs_orig['_idx_j'].to(device)
        inputs_orig['_n_atoms'] = inputs_orig['_n_atoms'].to(device)
        
        # e3nn SO3
        rij = inputs_orig['_Rij']
        radial_ij_raw = torch.norm(rij, dim=-1)
        radial_ij = self.rbf(radial_ij_raw).to(device)
        sh = o3.SphericalHarmonics(range(self.lmax + 1), normalize=True, normalization='component')
        dir_ij = rij / (radial_ij_raw.unsqueeze(-1) + 1e-8)
        dir_ij_sh = sh(dir_ij).to(device)
        idx_i = inputs_orig['_idx_i']
        idx_j = inputs_orig['_idx_j']
        n_atoms = inputs_orig['_n_atoms'].sum().item()
        
        # Embedding with device handling
        x = inputs_orig['_atomic_numbers'].unsqueeze(-1).to(device)
        x = self.embedding(x.long()).squeeze(1)  # [n_atoms, 192]
        
        radial_weights = self.e3nn_radial_net(radial_ij).view(-1, (self.lmax + 1) ** 2, self.n_atom_basis)
        neighbor_contrib = self.e3nn(x[idx_j], dir_ij_sh)
        neighbor_contrib = neighbor_contrib.view(-1, (self.lmax + 1) ** 2, self.n_atom_basis)
        neighbor_contrib = neighbor_contrib * radial_weights * self.cutoff_fn(radial_ij_raw).unsqueeze(-1).unsqueeze(-1)
        neighbor_contrib = neighbor_contrib.view(-1, self.n_atom_basis * (self.lmax + 1) ** 2)
        e3nn_features = snn.scatter_add(neighbor_contrib, idx_i, dim_size=n_atoms, dim=0)  # [n_atoms, 192 * 9]
        e3nn_features = self.e3nn_proj(e3nn_features)  # [n_atoms, 96]
        
        # SchNet
        inputs_schnet = inputs_orig.copy()
        inputs_schnet = self.schnet(inputs_schnet)
        schnet_features = inputs_schnet['scalar_representation']  # [n_atoms, 192]
        schnet_features = self.schnet_proj(schnet_features)  # [n_atoms, 96]
        
        # Fusion
        fused_features = torch.cat([e3nn_features, schnet_features], dim=-1)  # [n_atoms, 96 * 2 = 192]
        x = self.fusion_layer(fused_features)  # [n_atoms, 192]
        
        inputs['scalar_representation'] = x
        return inputs



class E3nnSO3ConvSchNetFusion(nn.Module):
    def __init__(self, n_atom_basis, n_radial, cutoff, lmax=2, n_interactions_e3nn=1, n_interactions_schnet=1):
        super(E3nnSO3ConvSchNetFusion, self).__init__()
        self.n_atom_basis = n_atom_basis
        self.lmax = lmax
        self.rbf = spk.nn.GaussianRBF(n_rbf=n_radial, cutoff=cutoff)
        
        # e3nn SO3Conv: Configurable interactions
        self.e3nn = E3nnSO3ConvWrapper(
            n_atom_basis=n_atom_basis,
            n_radial=n_radial,
            cutoff=cutoff,
            lmax=lmax,
            n_interactions=n_interactions_e3nn
        )
        
        self.e3nn_proj = nn.Linear(n_atom_basis, n_atom_basis // 2)
        
        # SchNet: Configurable interactions
        self.schnet = spk.representation.SchNet(
            n_atom_basis=n_atom_basis,
            n_interactions=n_interactions_schnet,
            radial_basis=self.rbf,
            cutoff_fn=spk.nn.CosineCutoff(cutoff)
        )
        
        self.schnet_proj = nn.Linear(n_atom_basis, n_atom_basis // 2)
        
        # Fusion
        self.fusion_layer = nn.Sequential(
            nn.Linear(n_atom_basis, n_atom_basis),
            nn.ReLU(),
            nn.Linear(n_atom_basis, n_atom_basis)
        )
        
        self.register_buffer("cutoff", torch.tensor(cutoff, dtype=torch.float32))
        self.cutoff_fn = spk.nn.CosineCutoff(cutoff=self.cutoff)

    def forward(self, inputs):
        inputs_orig = inputs.copy()
        
        # Ensure all input tensors are on the same device as the model
        device = next(self.parameters()).device
        inputs_orig['_atomic_numbers'] = inputs_orig['_atomic_numbers'].to(device)
        inputs_orig['_Rij'] = inputs_orig['_Rij'].to(device)
        inputs_orig['_idx_i'] = inputs_orig['_idx_i'].to(device)
        inputs_orig['_idx_j'] = inputs_orig['_idx_j'].to(device)
        inputs_orig['_n_atoms'] = inputs_orig['_n_atoms'].to(device)
        
        # e3nn SO3Conv
        inputs_e3nn = self.e3nn(inputs_orig)
        e3nn_features = inputs_e3nn['scalar_representation']  # [n_atoms, 192]
        e3nn_features = self.e3nn_proj(e3nn_features)  # [n_atoms, 96]
        
        # SchNet
        inputs_schnet = inputs_orig.copy()
        inputs_schnet = self.schnet(inputs_schnet)
        schnet_features = inputs_schnet['scalar_representation']  # [n_atoms, 192]
        schnet_features = self.schnet_proj(schnet_features)  # [n_atoms, 96]
        
        # Fusion
        fused_features = torch.cat([e3nn_features, schnet_features], dim=-1)  # [n_atoms, 96 * 2 = 192]
        x = self.fusion_layer(fused_features)  # [n_atoms, 192]
        
        inputs['scalar_representation'] = x
        return inputs

class E3nnSO3TensorSchNetFusion(nn.Module):
    def __init__(self, n_atom_basis, n_radial, cutoff, lmax=0):
        super(E3nnSO3TensorSchNetFusion, self).__init__()
        self.n_atom_basis = n_atom_basis
        self.lmax = lmax
        self.rbf = spk.nn.GaussianRBF(n_rbf=n_radial, cutoff=cutoff)
        
        # e3nn SO3Tensor: 1 interaction
        self.e3nn = E3nnSO3TensorWrapper(
            n_atom_basis=n_atom_basis,
            n_radial=n_radial,
            cutoff=cutoff,
            lmax=lmax,
            n_interactions=1
        )
        
        self.e3nn_proj = nn.Linear(n_atom_basis, n_atom_basis // 2)
        
        # SchNet: 1 layer
        self.schnet = spk.representation.SchNet(
            n_atom_basis=n_atom_basis,
            n_interactions=1,
            radial_basis=self.rbf,
            cutoff_fn=spk.nn.CosineCutoff(cutoff)
        )
        
        self.schnet_proj = nn.Linear(n_atom_basis, n_atom_basis // 2)
        
        # Fusion
        self.fusion_layer = nn.Sequential(
            nn.Linear(n_atom_basis, n_atom_basis),
            nn.ReLU(),
            nn.Linear(n_atom_basis, n_atom_basis)
        )
        
        self.register_buffer("cutoff", torch.tensor(cutoff, dtype=torch.float32))
        self.cutoff_fn = spk.nn.CosineCutoff(cutoff=self.cutoff)

    def forward(self, inputs):
        inputs_orig = inputs.copy()
        
        # Ensure all input tensors are on the same device as the model
        device = next(self.parameters()).device
        inputs_orig['_atomic_numbers'] = inputs_orig['_atomic_numbers'].to(device)
        inputs_orig['_Rij'] = inputs_orig['_Rij'].to(device)
        inputs_orig['_idx_i'] = inputs_orig['_idx_i'].to(device)
        inputs_orig['_idx_j'] = inputs_orig['_idx_j'].to(device)
        inputs_orig['_n_atoms'] = inputs_orig['_n_atoms'].to(device)
        
        # e3nn SO3Tensor
        inputs_e3nn = self.e3nn(inputs_orig)
        e3nn_features = inputs_e3nn['scalar_representation']  # [n_atoms, 192]
        e3nn_features = self.e3nn_proj(e3nn_features)  # [n_atoms, 96]
        
        # SchNet
        inputs_schnet = inputs_orig.copy()
        inputs_schnet = self.schnet(inputs_schnet)
        schnet_features = inputs_schnet['scalar_representation']  # [n_atoms, 192]
        schnet_features = self.schnet_proj(schnet_features)  # [n_atoms, 96]
        
        # Fusion
        fused_features = torch.cat([e3nn_features, schnet_features], dim=-1)  # [n_atoms, 96 * 2 = 192]
        x = self.fusion_layer(fused_features)  # [n_atoms, 192]
        
        inputs['scalar_representation'] = x
        return inputs



class NequIPMACEFusion(nn.Module):
    def __init__(self, n_atom_basis, n_radial, cutoff, lmax=2, n_interactions_nequip=1, n_interactions_mace=1):
        super(NequIPMACEFusion, self).__init__()
        self.n_atom_basis = n_atom_basis
        self.lmax = lmax
        self.rbf = spk.nn.GaussianRBF(n_rbf=n_radial, cutoff=cutoff)
        
        # Embedding for MACE-inspired branch
        self.embedding = nn.Embedding(100, n_atom_basis)
        
        # NequIP-like e3nn: Configurable interactions
        self.nequip = E3nnSO3ConvWrapper(
            n_atom_basis=n_atom_basis,
            n_radial=n_radial,
            cutoff=cutoff,
            lmax=lmax,
            n_interactions=n_interactions_nequip
        )
        
        self.nequip_proj = nn.Linear(n_atom_basis, n_atom_basis // 2)
        
        # MACE-inspired: Configurable interactions
        self.mace = nn.ModuleList([
            MACEInspiredLayer(
                n_atom_basis=n_atom_basis,
                n_radial=n_radial,
                lmax=lmax
            ) for _ in range(n_interactions_mace)
        ])
        
        self.mace_proj = nn.Linear(n_atom_basis, n_atom_basis // 2)
        
        # Fusion
        self.fusion_layer = nn.Sequential(
            nn.Linear(n_atom_basis, n_atom_basis),
            nn.ReLU(),
            nn.Linear(n_atom_basis, n_atom_basis)
        )
        
        self.register_buffer("cutoff", torch.tensor(cutoff, dtype=torch.float32))
        self.cutoff_fn = spk.nn.CosineCutoff(cutoff=self.cutoff)

    def forward(self, inputs):
        inputs_orig = inputs.copy()
        
        # Ensure all input tensors are on the same device as the model
        device = next(self.parameters()).device
        inputs_orig['_atomic_numbers'] = inputs_orig['_atomic_numbers'].to(device)
        inputs_orig['_Rij'] = inputs_orig['_Rij'].to(device)
        inputs_orig['_idx_i'] = inputs_orig['_idx_i'].to(device)
        inputs_orig['_idx_j'] = inputs_orig['_idx_j'].to(device)
        inputs_orig['_n_atoms'] = inputs_orig['_n_atoms'].to(device)
        
        # NequIP-like e3nn
        inputs_nequip = self.nequip(inputs_orig)
        nequip_features = inputs_nequip['scalar_representation']  # [n_atoms, 192]
        nequip_features = self.nequip_proj(nequip_features)  # [n_atoms, 96]
        
        # MACE-inspired
        rij = inputs_orig['_Rij']
        radial_ij_raw = torch.norm(rij, dim=-1)
        radial_ij = self.rbf(radial_ij_raw).to(device)
        sh = o3.SphericalHarmonics(range(self.lmax + 1), normalize=True, normalization='component')
        dir_ij = rij / (radial_ij_raw.unsqueeze(-1) + 1e-8)
        dir_ij_sh = sh(dir_ij).to(device)
        idx_i = inputs_orig['_idx_i']
        idx_j = inputs_orig['_idx_j']
        n_atoms = inputs_orig['_n_atoms'].sum().item()
        
        x = inputs_orig['_atomic_numbers'].unsqueeze(-1).to(device)
        x = self.embedding(x.long()).squeeze(1)  # [n_atoms, 192]
        for layer in self.mace:
            x = layer(x, dir_ij_sh, radial_ij, idx_i, idx_j, n_atoms)  # [n_atoms, 192]
        mace_features = x
        mace_features = self.mace_proj(mace_features)  # [n_atoms, 96]
        
        # Fusion
        fused_features = torch.cat([nequip_features, mace_features], dim=-1)  # [n_atoms, 96 * 2 = 192]
        x = self.fusion_layer(fused_features)  # [n_atoms, 192]
        
        inputs['scalar_representation'] = x
        return inputs




class NequIPMACEEmbeddingPaiNNFusion(nn.Module):
    def __init__(self, n_atom_basis, n_radial, cutoff, lmax=1, n_interactions_painn=1):
        super(NequIPMACEEmbeddingPaiNNFusion, self).__init__()
        self.n_atom_basis = n_atom_basis
        self.lmax = lmax
        self.rbf = spk.nn.GaussianRBF(n_rbf=n_radial, cutoff=cutoff)
        
        # NequIP embedding
        self.nequip_embedding = nn.Embedding(100, n_atom_basis)
        
        # MACE embedding
        self.mace_embedding = nn.Embedding(100, n_atom_basis)
        
        # Fusion layer
        self.fusion_embedding = nn.Linear(2 * n_atom_basis, n_atom_basis)
        
        # PaiNN: Configurable interactions
        self.painn = spk.representation.PaiNN(
            n_atom_basis=n_atom_basis,
            n_interactions=n_interactions_painn,
            radial_basis=self.rbf,
            cutoff_fn=spk.nn.CosineCutoff(cutoff)
        )
        
        self.register_buffer("cutoff", torch.tensor(cutoff, dtype=torch.float32))
        self.cutoff_fn = spk.nn.CosineCutoff(cutoff=self.cutoff)

    def forward(self, inputs):
        inputs_orig = inputs.copy()
        
        # Ensure all input tensors are on the same device as the model
        device = next(self.parameters()).device
        inputs_orig['_atomic_numbers'] = inputs_orig['_atomic_numbers'].to(device)
        inputs_orig['_Rij'] = inputs_orig['_Rij'].to(device)
        inputs_orig['_idx_i'] = inputs_orig['_idx_i'].to(device)
        inputs_orig['_idx_j'] = inputs_orig['_idx_j'].to(device)
        inputs_orig['_n_atoms'] = inputs_orig['_n_atoms'].to(device)
        
        # NequIP embedding
        nequip_emb = self.nequip_embedding(inputs_orig['_atomic_numbers'])  # [n_atoms, 192]
        
        # MACE embedding
        mace_emb = self.mace_embedding(inputs_orig['_atomic_numbers'])  # [n_atoms, 192]
        
        # Fuse embeddings
        fused_emb = torch.cat([nequip_emb, mace_emb], dim=-1)  # [n_atoms, 384]
        fused_emb = self.fusion_embedding(fused_emb)  # [n_atoms, 192]
        
        # PaiNN
        inputs_painn = inputs_orig.copy()
        inputs_painn['scalar_representation'] = fused_emb
        inputs_painn = self.painn(inputs_painn)
        painn_features = inputs_painn['scalar_representation']  # [n_atoms, 192]
        
        inputs['scalar_representation'] = painn_features
        return inputs

# updated
class NequIPMACEInteractionFusion(nn.Module):
    def __init__(self, n_atom_basis, n_radial, cutoff, lmax=2, n_interactions_nequip=1, n_interactions_mace=1, n_interactions_painn=1):
        super(NequIPMACEInteractionFusion, self).__init__()
        self.n_atom_basis = n_atom_basis
        self.lmax = lmax
        self.rbf = spk.nn.GaussianRBF(n_rbf=n_radial, cutoff=cutoff)
        
        # NequIP: Configurable interactions
        self.nequip = E3nnSO3ConvWrapper(
            n_atom_basis=n_atom_basis,
            n_radial=n_radial,
            cutoff=cutoff,
            lmax=lmax,
            n_interactions=n_interactions_nequip
        )
        
        # MACE: Configurable interactions
        self.mace = nn.ModuleList([
            MACEInspiredLayer(
                n_atom_basis=n_atom_basis,
                n_radial=n_radial,
                lmax=lmax
            ) for _ in range(n_interactions_mace)
        ])
        
        # MACE embedding
        self.mace_embedding = nn.Embedding(100, n_atom_basis)
        
        # Fusion layer
        self.fusion_embedding = nn.Linear(2 * n_atom_basis, n_atom_basis)
        
        # PaiNN: Configurable interactions
        self.painn = spk.representation.PaiNN(
            n_atom_basis=n_atom_basis,
            n_interactions=n_interactions_painn,
            radial_basis=self.rbf,
            cutoff_fn=spk.nn.CosineCutoff(cutoff)
        )
        
        self.register_buffer("cutoff", torch.tensor(cutoff, dtype=torch.float32))
        self.cutoff_fn = spk.nn.CosineCutoff(cutoff=self.cutoff)

    def forward(self, inputs):
        inputs_orig = inputs.copy()
        
        # Ensure all input tensors are on the same device as the model
        device = next(self.parameters()).device
        inputs_orig['_atomic_numbers'] = inputs_orig['_atomic_numbers'].to(device)
        inputs_orig['_Rij'] = inputs_orig['_Rij'].to(device)
        inputs_orig['_idx_i'] = inputs_orig['_idx_i'].to(device)
        inputs_orig['_idx_j'] = inputs_orig['_idx_j'].to(device)
        inputs_orig['_n_atoms'] = inputs_orig['_n_atoms'].to(device)
        
        # NequIP interaction embedding
        inputs_nequip = self.nequip(inputs_orig)
        nequip_emb = inputs_nequip['scalar_representation']  # [n_atoms, 192]
        
        # MACE interaction embedding
        rij = inputs_orig['_Rij']
        radial_ij_raw = torch.norm(rij, dim=-1)
        radial_ij = self.rbf(radial_ij_raw).to(device)
        sh = o3.SphericalHarmonics(range(self.lmax + 1), normalize=True, normalization='component')
        dir_ij = rij / (radial_ij_raw.unsqueeze(-1) + 1e-8)
        dir_ij_sh = sh(dir_ij).to(device)
        idx_i = inputs_orig['_idx_i']
        idx_j = inputs_orig['_idx_j']
        n_atoms = inputs_orig['_n_atoms'].sum().item()
        
        x = inputs_orig['_atomic_numbers'].unsqueeze(-1).to(device)
        x = self.mace_embedding(x.long()).squeeze(1)  # [n_atoms, 192]
        for layer in self.mace:
            x = layer(x, dir_ij_sh, radial_ij, idx_i, idx_j, n_atoms)  # [n_atoms, 192]
        mace_emb = x  # [n_atoms, 192]
        
        # Fuse embeddings
        fused_emb = torch.cat([nequip_emb, mace_emb], dim=-1)  # [n_atoms, 384]
        fused_emb = self.fusion_embedding(fused_emb)  # [n_atoms, 192]
        
        # PaiNN
        inputs_painn = inputs_orig.copy()
        inputs_painn['scalar_representation'] = fused_emb
        inputs_painn = self.painn(inputs_painn)
        painn_features = inputs_painn['scalar_representation']  # [n_atoms, 192]
        
        inputs['scalar_representation'] = painn_features
        return inputs


class NequIPMACESchNetFusion(nn.Module):
    def __init__(self, n_atom_basis, n_radial, cutoff, lmax=2, n_interactions_nequip=1, n_interactions_mace=1, n_interactions_schnet=2):
        super(NequIPMACESchNetFusion, self).__init__()
        self.n_atom_basis = n_atom_basis
        self.lmax = lmax
        self.rbf = spk.nn.GaussianRBF(n_rbf=n_radial, cutoff=cutoff)
        
        # NequIP: Configurable interactions
        self.nequip = E3nnSO3ConvWrapper(
            n_atom_basis=n_atom_basis,
            n_radial=n_radial,
            cutoff=cutoff,
            lmax=lmax,
            n_interactions=n_interactions_nequip
        )
        
        # MACE: Configurable interactions
        self.mace = nn.ModuleList([
            MACEInspiredLayer(
                n_atom_basis=n_atom_basis,
                n_radial=n_radial,
                lmax=lmax
            ) for _ in range(n_interactions_mace)
        ])
        
        # MACE embedding
        self.mace_embedding = nn.Embedding(100, n_atom_basis)
        
        # SchNet: Configurable interactions
        self.schnet = spk.representation.SchNet(
            n_atom_basis=n_atom_basis,
            n_interactions=n_interactions_schnet,
            radial_basis=self.rbf,
            cutoff_fn=spk.nn.CosineCutoff(cutoff)
        )
        
        # Fusion layer
        self.fusion_embedding = nn.Linear(2 * n_atom_basis, n_atom_basis)
        
        self.register_buffer("cutoff", torch.tensor(cutoff, dtype=torch.float32))
        self.cutoff_fn = spk.nn.CosineCutoff(cutoff=self.cutoff)

    def forward(self, inputs):
        device = next(self.parameters()).device
        inputs_orig = inputs.copy()
        inputs_orig['_atomic_numbers'] = inputs_orig['_atomic_numbers'].to(device)
        inputs_orig['_Rij'] = inputs_orig['_Rij'].to(device)
        inputs_orig['_idx_i'] = inputs_orig['_idx_i'].to(device)
        inputs_orig['_idx_j'] = inputs_orig['_idx_j'].to(device)
        inputs_orig['_n_atoms'] = inputs_orig['_n_atoms'].to(device)
        
        # NequIP interaction embedding
        inputs_nequip = self.nequip(inputs_orig)
        nequip_emb = inputs_nequip['scalar_representation']  # [n_atoms, 192]
        
        # MACE interaction embedding
        rij = inputs_orig['_Rij']
        radial_ij_raw = torch.norm(rij, dim=-1)
        radial_ij = self.rbf(radial_ij_raw).to(device)
        sh = o3.SphericalHarmonics(range(self.lmax + 1), normalize=True, normalization='component')
        dir_ij = rij / (radial_ij_raw.unsqueeze(-1) + 1e-8)
        dir_ij_sh = sh(dir_ij).to(device)
        idx_i = inputs_orig['_idx_i']
        idx_j = inputs_orig['_idx_j']
        n_atoms = inputs_orig['_n_atoms'].sum().item()
        
        x = inputs_orig['_atomic_numbers'].unsqueeze(-1).to(device)
        x = self.mace_embedding(x.long()).squeeze(1)  # [n_atoms, 192]
        for layer in self.mace:
            x = layer(x, dir_ij_sh, radial_ij, idx_i, idx_j, n_atoms)
        mace_emb = x  # [n_atoms, 192]
        
        # SchNet interaction embedding
        inputs_schnet = inputs_orig.copy()
        inputs_schnet = self.schnet(inputs_schnet)
        schnet_emb = inputs_schnet['scalar_representation']  # [n_atoms, 192]
        
        # Fuse embeddings
        fused_emb = torch.cat([nequip_emb, mace_emb], dim=-1)  # [n_atoms, 384]
        fused_emb = self.fusion_embedding(fused_emb)  # [n_atoms, 192]
        
        # Combine with SchNet embedding
        final_emb = torch.cat([fused_emb, schnet_emb], dim=-1)  # [n_atoms, 384]
        final_emb = self.fusion_embedding(final_emb)  # [n_atoms, 192]
        
        inputs['scalar_representation'] = final_emb
        return inputs


class NequIPMACESO3TensorFusion(nn.Module):
    def __init__(self, n_atom_basis, n_radial, cutoff, lmax=3, n_interactions_nequip=1, n_interactions_mace=1):
        super(NequIPMACESO3TensorFusion, self).__init__()
        self.n_atom_basis = n_atom_basis
        self.lmax = lmax
        self.rbf = spk.nn.GaussianRBF(n_rbf=n_radial, cutoff=cutoff)
        
        self.nequip = E3nnSO3ConvWrapper(n_atom_basis, n_radial, cutoff, lmax, n_interactions_nequip)
        self.mace = nn.ModuleList([MACEInspiredLayer(n_atom_basis, n_radial, lmax) for _ in range(n_interactions_mace)])
        self.mace_embedding = nn.Embedding(100, n_atom_basis)
        
        self.fusion_embedding = nn.Linear(2 * n_atom_basis, n_atom_basis)
        self.so3_tensor = o3.TensorProduct(
            irreps_in1=f"{n_atom_basis}x0e",
            irreps_in2=" + ".join([f"1x{l}e" for l in range(lmax + 1)]),
            irreps_out=" + ".join([f"{n_atom_basis}x{l}e" for l in range(lmax + 1)]),
            instructions=[(0, l, l, "uvu", True) for l in range(lmax + 1)],
            internal_weights=True,
            shared_weights=True
        )
        self.reduce_layer = nn.Linear(n_atom_basis * (lmax + 1) ** 2, n_atom_basis)
        self.register_buffer("cutoff", torch.tensor(cutoff, dtype=torch.float32))
        self.cutoff_fn = spk.nn.CosineCutoff(cutoff=self.cutoff)

    def forward(self, inputs):
        device = next(self.parameters()).device
        inputs_orig = inputs.copy()
        inputs_orig['_atomic_numbers'] = inputs_orig['_atomic_numbers'].to(device)
        inputs_orig['_Rij'] = inputs_orig['_Rij'].to(device)
        inputs_orig['_idx_i'] = inputs_orig['_idx_i'].to(device)
        inputs_orig['_idx_j'] = inputs_orig['_idx_j'].to(device)
        inputs_orig['_n_atoms'] = inputs_orig['_n_atoms'].to(device)
        
        inputs_nequip = self.nequip(inputs_orig)
        nequip_emb = inputs_nequip['scalar_representation']  # [n_atoms, 192]
        
        rij = inputs_orig['_Rij']
        radial_ij_raw = torch.norm(rij, dim=-1)
        radial_ij = self.rbf(radial_ij_raw).to(device)
        sh = o3.SphericalHarmonics(range(self.lmax + 1), normalize=True, normalization='component')
        dir_ij = rij / (radial_ij_raw.unsqueeze(-1) + 1e-8)
        dir_ij_sh = sh(dir_ij).to(device)
        idx_i = inputs_orig['_idx_i']
        idx_j = inputs_orig['_idx_j']
        n_atoms = inputs_orig['_n_atoms'].sum().item()
        
        x = inputs_orig['_atomic_numbers'].unsqueeze(-1).to(device)
        x = self.mace_embedding(x.long()).squeeze(1)  # [n_atoms, 192]
        for layer in self.mace:
            x = layer(x, dir_ij_sh, radial_ij, idx_i, idx_j, n_atoms)
        mace_emb = x  # [n_atoms, 192]
        
        fused_emb = torch.cat([nequip_emb, mace_emb], dim=-1)  # [n_atoms, 384]
        fused_emb = self.fusion_embedding(fused_emb)  # [n_atoms, 192]
        
        tensor_emb = self.so3_tensor(fused_emb[idx_i], dir_ij_sh)  # [n_edges, n_atom_basis * (lmax + 1)^2]
        tensor_emb = tensor_emb.view(-1, (self.lmax + 1) ** 2, self.n_atom_basis)
        tensor_emb = snn.scatter_add(tensor_emb, idx_i, dim_size=n_atoms, dim=0)  # [n_atoms, (lmax + 1)^2, n_atom_basis]
        tensor_emb = tensor_emb.view(-1, self.n_atom_basis * (self.lmax + 1) ** 2)
        tensor_emb = self.reduce_layer(tensor_emb)  # [n_atoms, n_atom_basis]
        
        inputs['scalar_representation'] = tensor_emb
        return inputs



class MultiModelFusion(nn.Module):
    def __init__(self, n_atom_basis, n_radial, cutoff, lmax=3, n_interactions_nequip=1, n_interactions_mace=1, n_interactions_schnet=1):
        super(MultiModelFusion, self).__init__()
        self.n_atom_basis = n_atom_basis
        self.lmax = lmax
        self.rbf = spk.nn.GaussianRBF(n_rbf=n_radial, cutoff=cutoff)
        
        self.nequip = E3nnSO3ConvWrapper(n_atom_basis, n_radial, cutoff, lmax, n_interactions_nequip)
        self.mace = nn.ModuleList([MACEInspiredLayer(n_atom_basis, n_radial, lmax) for _ in range(n_interactions_mace)])
        self.mace_embedding = nn.Embedding(100, n_atom_basis)
        self.schnet = spk.representation.SchNet(n_atom_basis, n_interactions_schnet, radial_basis=self.rbf, cutoff_fn=spk.nn.CosineCutoff(cutoff))
        self.raw_embedding = nn.Embedding(100, n_atom_basis)
        
        self.fusion_linear = nn.Linear(4 * n_atom_basis, n_atom_basis)
        self.tensor_product = o3.TensorProduct(
            irreps_in1=f"{n_atom_basis}x0e",
            irreps_in2=" + ".join([f"1x{l}e" for l in range(lmax + 1)]),
            irreps_out=" + ".join([f"{n_atom_basis}x{l}e" for l in range(lmax + 1)]),
            instructions=[(0, l, l, "uvu", True) for l in range(lmax + 1)],
            internal_weights=True,
            shared_weights=True
        )
        self.reduce_layer = nn.Linear(n_atom_basis * (lmax + 1) ** 2, n_atom_basis)
        self.register_buffer("cutoff", torch.tensor(cutoff, dtype=torch.float32))
        self.cutoff_fn = spk.nn.CosineCutoff(cutoff=self.cutoff)

    def forward(self, inputs):
        device = next(self.parameters()).device
        inputs_orig = inputs.copy()
        inputs_orig['_atomic_numbers'] = inputs_orig['_atomic_numbers'].to(device)
        inputs_orig['_Rij'] = inputs_orig['_Rij'].to(device)
        inputs_orig['_idx_i'] = inputs_orig['_idx_i'].to(device)
        inputs_orig['_idx_j'] = inputs_orig['_idx_j'].to(device)
        inputs_orig['_n_atoms'] = inputs_orig['_n_atoms'].to(device)
        
        inputs_nequip = self.nequip(inputs_orig)
        nequip_emb = inputs_nequip['scalar_representation']  # [n_atoms, 192]
        
        rij = inputs_orig['_Rij']
        radial_ij_raw = torch.norm(rij, dim=-1)
        radial_ij = self.rbf(radial_ij_raw).to(device)
        sh = o3.SphericalHarmonics(range(self.lmax + 1), normalize=True, normalization='component')
        dir_ij = rij / (radial_ij_raw.unsqueeze(-1) + 1e-8)
        dir_ij_sh = sh(dir_ij).to(device)
        idx_i = inputs_orig['_idx_i']
        idx_j = inputs_orig['_idx_j']
        n_atoms = inputs_orig['_n_atoms'].sum().item()
        
        x = inputs_orig['_atomic_numbers'].unsqueeze(-1).to(device)
        x = self.mace_embedding(x.long()).squeeze(1)  # [n_atoms, 192]
        for layer in self.mace:
            x = layer(x, dir_ij_sh, radial_ij, idx_i, idx_j, n_atoms)
        mace_emb = x  # [n_atoms, 192]
        
        inputs_schnet = inputs_orig.copy()
        inputs_schnet = self.schnet(inputs_schnet)
        schnet_emb = inputs_schnet['scalar_representation']  # [n_atoms, 192]
        
        raw_emb = self.raw_embedding(inputs_orig['_atomic_numbers'].long()).squeeze(1)  # [n_atoms, 192]
        
        fused_emb = torch.cat([nequip_emb, mace_emb, schnet_emb, raw_emb], dim=-1)  # [n_atoms, 768]
        fused_emb = self.fusion_linear(fused_emb)  # [n_atoms, 192]
        
        # Apply tensor_product to edge-level features
        fused_emb_edges = fused_emb[idx_j]  # [n_edges, 192]
        tensor_emb = self.tensor_product(fused_emb_edges, dir_ij_sh)  # [n_edges, n_atom_basis * (lmax + 1)^2]
        tensor_emb = tensor_emb.view(-1, (self.lmax + 1) ** 2, self.n_atom_basis)  # [n_edges, (lmax + 1)^2, n_atom_basis]
        tensor_emb = snn.scatter_add(tensor_emb, idx_i, dim_size=n_atoms, dim=0)  # [n_atoms, (lmax + 1)^2, n_atom_basis]
        tensor_emb = tensor_emb.view(-1, self.n_atom_basis * (self.lmax + 1) ** 2)  # [n_atoms, n_atom_basis * (lmax + 1)^2]
        tensor_emb = self.reduce_layer(tensor_emb)  # [n_atoms, n_atom_basis]
        
        inputs['scalar_representation'] = tensor_emb
        return inputs
        

class HierarchicalFusion(nn.Module):
    def __init__(self, n_atom_basis, n_radial, cutoff, lmax=3, n_interactions_nequip=1, n_interactions_mace=1, n_interactions_schnet=1, n_interactions_painn=1):
        super(HierarchicalFusion, self).__init__()
        self.n_atom_basis = n_atom_basis
        self.lmax = lmax
        self.rbf = spk.nn.GaussianRBF(n_rbf=n_radial, cutoff=cutoff)
        
        self.nequip = E3nnSO3ConvWrapper(n_atom_basis, n_radial, cutoff, lmax, n_interactions_nequip)
        self.mace = nn.ModuleList([MACEInspiredLayer(n_atom_basis, n_radial, lmax) for _ in range(n_interactions_mace)])
        self.mace_embedding = nn.Embedding(100, n_atom_basis)
        self.schnet = spk.representation.SchNet(n_atom_basis, n_interactions_schnet, radial_basis=self.rbf, cutoff_fn=spk.nn.CosineCutoff(cutoff))
        self.painn = spk.representation.PaiNN(n_atom_basis, n_interactions_painn, radial_basis=self.rbf, cutoff_fn=spk.nn.CosineCutoff(cutoff))
        self.raw_embedding = nn.Embedding(100, n_atom_basis)
        
        self.fusion_ab = nn.Linear(2 * n_atom_basis, n_atom_basis)
        self.fusion_d = nn.Linear(2 * n_atom_basis, n_atom_basis)
        self.tensor_product = o3.TensorProduct(
            irreps_in1=f"{n_atom_basis}x0e",
            irreps_in2=" + ".join([f"1x{l}e" for l in range(lmax + 1)]),
            irreps_out=" + ".join([f"{n_atom_basis}x{l}e" for l in range(lmax + 1)]),
            instructions=[(0, l, l, "uvu", True) for l in range(lmax + 1)],
            internal_weights=True,
            shared_weights=True
        )
        self.reduce_layer = nn.Linear(n_atom_basis * (lmax + 1) ** 2, n_atom_basis)
        self.register_buffer("cutoff", torch.tensor(cutoff, dtype=torch.float32))
        self.cutoff_fn = spk.nn.CosineCutoff(cutoff=self.cutoff)

    def forward(self, inputs):
        device = next(self.parameters()).device
        inputs_orig = inputs.copy()
        inputs_orig['_atomic_numbers'] = inputs_orig['_atomic_numbers'].to(device)
        inputs_orig['_Rij'] = inputs_orig['_Rij'].to(device)
        inputs_orig['_idx_i'] = inputs_orig['_idx_i'].to(device)
        inputs_orig['_idx_j'] = inputs_orig['_idx_j'].to(device)
        inputs_orig['_n_atoms'] = inputs_orig['_n_atoms'].to(device)
        
        inputs_nequip = self.nequip(inputs_orig)
        nequip_emb = inputs_nequip['scalar_representation']  # [n_atoms, 192]
        
        rij = inputs_orig['_Rij']
        radial_ij_raw = torch.norm(rij, dim=-1)
        radial_ij = self.rbf(radial_ij_raw).to(device)
        sh = o3.SphericalHarmonics(range(self.lmax + 1), normalize=True, normalization='component')
        dir_ij = rij / (radial_ij_raw.unsqueeze(-1) + 1e-8)
        dir_ij_sh = sh(dir_ij).to(device)
        idx_i = inputs_orig['_idx_i']
        idx_j = inputs_orig['_idx_j']
        n_atoms = inputs_orig['_n_atoms'].sum().item()
        
        x = inputs_orig['_atomic_numbers'].unsqueeze(-1).to(device)
        x = self.mace_embedding(x.long()).squeeze(1)  # [n_atoms, 192]
        for layer in self.mace:
            x = layer(x, dir_ij_sh, radial_ij, idx_i, idx_j, n_atoms)
        mace_emb = x  # [n_atoms, 192]
        
        inputs_schnet = inputs_orig.copy()
        inputs_schnet = self.schnet(inputs_schnet)
        schnet_emb = inputs_schnet['scalar_representation']  # [n_atoms, 192]
        
        inputs_painn = inputs_orig.copy()
        inputs_painn = self.painn(inputs_painn)
        painn_emb = inputs_painn['scalar_representation']  # [n_atoms, 192]
        
        raw_emb = self.raw_embedding(inputs_orig['_atomic_numbers'].long()).squeeze(1)  # [n_atoms, 192]
        
        ab_emb = torch.cat([nequip_emb, mace_emb], dim=-1)  # [n_atoms, 384]
        ab_emb = self.fusion_ab(ab_emb)  # [n_atoms, 192]
        bc_emb = torch.cat([schnet_emb, painn_emb], dim=-1)  # [n_atoms, 384]
        bc_emb = self.fusion_ab(bc_emb)  # [n_atoms, 192]
        d_emb = torch.cat([ab_emb, bc_emb], dim=-1)  # [n_atoms, 384]
        d_emb = self.fusion_d(d_emb)  # [n_atoms, 192]
        final_emb = torch.cat([d_emb, raw_emb], dim=-1)  # [n_atoms, 384]
        final_emb = self.fusion_d(final_emb)  # [n_atoms, 192]
        
        # Apply tensor_product to edge-level features
        final_emb_edges = final_emb[idx_j]  # [n_edges, 192]
        tensor_emb = self.tensor_product(final_emb_edges, dir_ij_sh)  # [n_edges, n_atom_basis * (lmax + 1)^2]
        tensor_emb = tensor_emb.view(-1, (self.lmax + 1) ** 2, self.n_atom_basis)  # [n_edges, (lmax + 1)^2, n_atom_basis]
        tensor_emb = snn.scatter_add(tensor_emb, idx_i, dim_size=n_atoms, dim=0)  # [n_atoms, (lmax + 1)^2, n_atom_basis]
        tensor_emb = tensor_emb.view(-1, self.n_atom_basis * (self.lmax + 1) ** 2)  # [n_atoms, n_atom_basis * (lmax + 1)^2]
        tensor_emb = self.reduce_layer(tensor_emb)  # [n_atoms, n_atom_basis]
        
        inputs['scalar_representation'] = tensor_emb
        return inputs


class CustomAtomwise(spk.atomistic.Atomwise):
    def __init__(self, n_in, output_key, n_layers=2, n_neurons=None, activation=nn.ReLU(), dropout_rate=0.2):
        super(CustomAtomwise, self).__init__(n_in=n_in, output_key=output_key)
        if n_neurons is None:
            n_neurons = [n_in] * (n_layers - 1)
        elif len(n_neurons) != n_layers - 1:
            raise ValueError(f"n_neurons must have {n_layers - 1} elements for {n_layers} layers")
        self.n_layers = n_layers
        self.dropout_rate = dropout_rate

        layers = []
        prev_size = n_in
        for size in n_neurons:
            layers.append(nn.Linear(prev_size, size))
            layers.append(activation)
            layers.append(nn.Dropout(p=dropout_rate))
            prev_size = size
        layers.append(nn.Linear(prev_size, 1))
        self.outnet = nn.Sequential(*layers)

    def forward(self, inputs):
        feats = inputs['scalar_representation']
        if isinstance(feats, tuple):
            feats = feats[0]
        elif len(feats.shape) == 3:
            feats = feats.sum(dim=1)
        
        atomwise_out = self.outnet(feats)
        n_atoms = inputs['_n_atoms']
        atom_indices = torch.repeat_interleave(torch.arange(len(n_atoms), device=feats.device), n_atoms)
        system_out = torch.zeros(len(n_atoms), 1, device=feats.device).scatter_add_(0, atom_indices.unsqueeze(1), atomwise_out)
        inputs[self.output_key] = system_out.squeeze(1)
        return inputs

def setup_model(config):
    model_type = config['settings']['model'].get('model_type', 'schnet')
    cutoff = config['settings']['model']['cutoff']
    n_rbf = config['settings']['model']['n_rbf']
    n_atom_basis = config['settings']['model']['n_atom_basis']
    #n_interactions = config['settings']['model']['n_interactions']
    n_interactions = config['settings']['model'].get('n_interactions', 3) if model_type.lower() != 'fused_hybrid' else None
    dropout_rate = config['settings']['model'].get('dropout_rate', 0.2)
    n_layers = config['settings']['model'].get('n_layers', 2)
    n_neurons = config['settings']['model'].get('n_neurons', None)
    lmax = config['settings']['model'].get('lmax', 2)

    
    n_interactions_e3nn = config['settings']['model'].get('n_interactions_e3nn', 1)
    n_interactions_schnet = config['settings']['model'].get('n_interactions_schnet', 1)
    n_interactions_nequip = config['settings']['model'].get('n_interactions_nequip', 1)
    n_interactions_mace = config['settings']['model'].get('n_interactions_mace', 1)
    n_interactions_painn = config['settings']['model'].get('n_interactions_painn', 1)
    n_interactions_schnet = config['settings']['model'].get('n_interactions_schnet', 1)
    
    pairwise_distance = spk.atomistic.PairwiseDistances()
    precompute_harmonics = PrecomputeSphericalHarmonics(lmax=lmax)
    radial_basis = spk.nn.GaussianRBF(n_rbf=n_rbf, cutoff=cutoff)

    if model_type.lower() == 'schnet':
        representation = spk.representation.SchNet(
            n_atom_basis=n_atom_basis,
            n_interactions=n_interactions,
            radial_basis=radial_basis,
            cutoff_fn=spk.nn.CosineCutoff(cutoff)
        )
        input_modules = [pairwise_distance]
        print("Using SchNet Model")
    elif model_type.lower() == 'painn':
        representation = spk.representation.PaiNN(
            n_atom_basis=n_atom_basis,
            n_interactions=n_interactions,
            radial_basis=radial_basis,
            cutoff_fn=spk.nn.CosineCutoff(cutoff),
        )
        input_modules = [pairwise_distance]
        print("Using PaiNN Model")
    elif model_type.lower() == 'schnet_so3conv':
        representation = SchnetSO3ConvWrapper(
            lmax=lmax,
            n_radial=n_rbf,
            n_atom_basis=n_atom_basis,
            cutoff=cutoff,
            n_interactions=n_interactions
        )
        input_modules = [pairwise_distance, precompute_harmonics]
        print("Using SchNet SO3 Convolution Model")
    elif model_type.lower() == 'schnet_so3tensor':
        representation = SchnetSO3TensorWrapper(
            n_atom_basis=n_atom_basis,
            n_radial=n_rbf,
            cutoff=cutoff,
            lmax=lmax,
            n_interactions=n_interactions
        )
        input_modules = [pairwise_distance, precompute_harmonics]
        print("Using SchNet SO3 Tensor Product Model")
    elif model_type.lower() == 'e3nn_so3conv':
        representation = E3nnSO3ConvWrapper(
            n_atom_basis=n_atom_basis,
            n_radial=n_rbf,
            cutoff=cutoff,
            lmax=lmax,
            n_interactions=n_interactions
        )
        input_modules = [pairwise_distance]
        print("Using e3nn SO3 Convolution Model")
    elif model_type.lower() == 'e3nn_so3tensor':
        representation = E3nnSO3TensorWrapper(
            n_atom_basis=n_atom_basis,
            n_radial=n_rbf,
            cutoff=cutoff,
            lmax=lmax,
            n_interactions=n_interactions
        )
        input_modules = [pairwise_distance]
        print("Using e3nn SO3 Tensor Product Model")
        
        
    elif model_type.lower() == 'hybrid_painn_e3nn':
        representation = HybridPaiNNE3nnSO3(
            n_atom_basis=n_atom_basis,
            n_radial=n_rbf,
            cutoff=cutoff,
            lmax=lmax,
            n_interactions=n_interactions  # Total interactions: 1 PaiNN + (n_interactions-1) e3nn
        )
        input_modules = [pairwise_distance]
        print("Using Hybrid PaiNN + e3nn SO3 Model")
        
        
    elif model_type.lower() == 'hybrid_painn_schnet_e3nn':
        representation = HybridPaiNNSchNetE3nnSO3(
            n_atom_basis=n_atom_basis,
            n_radial=n_rbf,
            cutoff=cutoff,
            lmax=lmax,
            n_interactions=n_interactions  # 1 PaiNN + 1 SchNet + (n_interactions-2) e3nn
        )
        input_modules = [pairwise_distance]
        print("Using Hybrid PaiNN + SchNet + e3nn SO3 Model")
        
        
    elif model_type.lower() == 'optimized_hybrid_painn_e3nn':
        representation = OptimizedHybridPaiNNE3nn(
            n_atom_basis=n_atom_basis,
            n_radial=n_rbf,
            cutoff=cutoff,
            lmax=lmax,
            n_interactions=n_interactions  # 1 PaiNN + (n_interactions-1) e3nn
        )
        input_modules = [pairwise_distance]
        print("Using Optimized Hybrid PaiNN + e3nn SO3 Model")
        
        
    elif model_type.lower() == 'fused_hybrid':
        representation = FusedHybridModel(
            n_atom_basis=n_atom_basis,
            n_radial=n_rbf,
            cutoff=cutoff,
            lmax=lmax
        )
        input_modules = [pairwise_distance]
        print("Using Fused Hybrid Model")
    
    elif model_type.lower() == 'equi_seq_hybrid':
        representation = EquiSeqHybrid(
            n_atom_basis=n_atom_basis,
            n_radial=n_rbf,
            cutoff=cutoff,
            lmax=lmax,
            n_interactions=n_interactions
        )
        input_modules = [pairwise_distance]
        print("Using EquiSeqHybrid Model")    
        
    elif model_type.lower() == 'fused_nequip_hybrid':
        representation = FusedNequIPHybrid(
            n_atom_basis=n_atom_basis,
            n_radial=n_rbf,
            cutoff=cutoff,
            lmax=lmax
        )
        input_modules = [pairwise_distance]
        print("Using FusedNequIPHybrid Model")
    
    elif model_type.lower() == 'mace_inspired_sequential':
        representation = MACEInspiredSequential(
            n_atom_basis=n_atom_basis,
            n_radial=n_rbf,
            cutoff=cutoff,
            lmax=lmax,
            n_interactions=n_interactions
        )
        input_modules = [pairwise_distance]
        print("Using MACEInspiredSequential Model")
        
    elif model_type.lower() == 'attention_fusion_hybrid':
        representation = AttentionFusionHybrid(
            n_atom_basis=n_atom_basis,
            n_radial=n_rbf,
            cutoff=cutoff,
            lmax=lmax
        )
        input_modules = [pairwise_distance]
        print("Using AttentionFusionHybrid Model")
        
        
    elif model_type.lower() == 'custom_equi_transformer':
        representation = CustomEquiTransformer(
            n_atom_basis=n_atom_basis,
            n_radial=n_rbf,
            cutoff=cutoff,
            lmax=lmax
        )
        input_modules = [pairwise_distance]
        print("Using CustomEquiTransformer Model")
    
    elif model_type.lower() == 'enhanced_tuned_hybrid_painn_e3nn':
        representation = EnhancedTunedHybridPaiNNE3nn(
            n_atom_basis=n_atom_basis,
            n_radial=n_rbf,
            cutoff=cutoff,
            lmax=lmax,
            n_interactions=n_interactions
        )
        input_modules = [pairwise_distance]
        print("Using EnhancedTunedHybridPaiNNE3nn Model")
    
    
    elif model_type.lower() == 'optimized_attention_fusion_hybrid':
        representation = OptimizedAttentionFusionHybrid(
            n_atom_basis=n_atom_basis,
            n_radial=n_rbf,
            cutoff=cutoff,
            lmax=lmax
        )
        input_modules = [pairwise_distance]
        print("Using OptimizedAttentionFusionHybrid Model")
    
    elif model_type.lower() == 'final_tuned_hybrid_painn_e3nn':
        representation = FinalTunedHybridPaiNNE3nn(
            n_atom_basis=n_atom_basis,
            n_radial=n_rbf,
            cutoff=cutoff,
            lmax=lmax,
            n_interactions=n_interactions
        )
        input_modules = [pairwise_distance]
        print("Using FinalTunedHybridPaiNNE3nn Model")
    
    
    elif model_type.lower() == 'tuned_optimized_hybrid_painn_e3nn':
        representation = TunedOptimizedHybridPaiNNE3nn(
            n_atom_basis=n_atom_basis,
            n_radial=n_rbf,
            cutoff=cutoff,
            lmax=lmax,
            n_interactions=n_interactions
        )
        input_modules = [pairwise_distance]
        print("Using TunedOptimizedHybridPaiNNE3nn Model")
    
    
    elif model_type.lower() == 'super_hybrid_painn_e3nn':
        representation = SuperHybridPaiNNE3nn(
            n_atom_basis=n_atom_basis,
            n_radial=n_rbf,
            cutoff=cutoff,
            lmax=lmax,
            n_interactions=n_interactions
        )
        input_modules = [pairwise_distance]
        print("Using SuperHybridPaiNNE3nn Model")
    
    elif model_type.lower() == 'ultra_hybrid_painn_e3nn':
        representation = UltraHybridPaiNNE3nn(
            n_atom_basis=n_atom_basis,
            n_radial=n_rbf,
            cutoff=cutoff,
            lmax=lmax
        )
        input_modules = [pairwise_distance]
        print("Using UltraHybridPaiNNE3nn Model")
    

    elif model_type.lower() == 'prime_hybrid_painn_e3nn':
        representation = PrimeHybridPaiNNE3nn(
            n_atom_basis=n_atom_basis,
            n_radial=n_rbf,
            cutoff=cutoff,
            lmax=lmax
        )
        input_modules = [pairwise_distance]
        print("Using PrimeHybridPaiNNE3nn Model")
        
    elif model_type.lower() == 'e3nn_schnet_parallel_fusion':
        representation = E3nnSchNetParallelFusion(
            n_atom_basis=n_atom_basis,
            n_radial=n_rbf,
            cutoff=cutoff,
            lmax=lmax
        )
        input_modules = [pairwise_distance]
        print("Using E3nnSchNetParallelFusion Model")
    
    elif model_type.lower() == 'e3nn_so3conv_schnet_fusion':
        representation = E3nnSO3ConvSchNetFusion(
            n_atom_basis=n_atom_basis,
            n_radial=n_rbf,
            cutoff=cutoff,
            lmax=lmax,
            n_interactions_e3nn=n_interactions_e3nn,
            n_interactions_schnet=n_interactions_schnet
        )
        input_modules = [pairwise_distance]
        print("Using E3nnSO3ConvSchNetFusion Model")
        
    elif model_type.lower() == 'e3nn_so3tensor_schnet_fusion':
        representation = E3nnSO3TensorSchNetFusion(
            n_atom_basis=n_atom_basis,
            n_radial=n_rbf,
            cutoff=cutoff,
            lmax=0  # Scalar-only interactions
        )
        input_modules = [pairwise_distance]
        print("Using E3nnSO3TensorSchNetFusion Model")
        
    elif model_type.lower() == 'nequip_mace_fusion':
        representation = NequIPMACEFusion(
            n_atom_basis=n_atom_basis,
            n_radial=n_rbf,
            cutoff=cutoff,
            lmax=lmax,
            n_interactions_nequip=n_interactions_nequip,
            n_interactions_mace=n_interactions_mace
        )
        input_modules = [pairwise_distance]
        print(f"Using NequIPMACEFusion Model with {n_interactions_nequip} NequIP interactions and {n_interactions_mace} MACE interactions")
        
    elif model_type.lower() == 'nequip_mace_embedding_painn_fusion':
        representation = NequIPMACEEmbeddingPaiNNFusion(
            n_atom_basis=n_atom_basis,
            n_radial=n_rbf,
            cutoff=cutoff,
            lmax=1,  # PaiNN default
            n_interactions_painn=n_interactions_painn
        )
        input_modules = [pairwise_distance]
        print(f"Using NequIPMACEEmbeddingPaiNNFusion Model with {n_interactions_painn} PaiNN interactions")
        
    elif model_type.lower() == 'nequip_mace_interaction_fusion':
        representation = NequIPMACEInteractionFusion(
            n_atom_basis=n_atom_basis,
            n_radial=n_rbf,
            cutoff=cutoff,
            lmax=lmax,
            n_interactions_nequip=n_interactions_nequip,
            n_interactions_mace=n_interactions_mace,
            n_interactions_painn=n_interactions_painn
        )
        input_modules = [pairwise_distance]
        print(f"Using NequIPMACEInteractionFusion Model with {n_interactions_nequip} NequIP interactions, {n_interactions_mace} MACE interactions, and {n_interactions_painn} PaiNN interactions")
    
    elif model_type.lower() == 'nequip_mace_schnet_fusion':
        representation = NequIPMACESchNetFusion(
            n_atom_basis=n_atom_basis,
            n_radial=n_rbf,
            cutoff=cutoff,
            lmax=lmax,
            n_interactions_nequip=n_interactions_nequip,
            n_interactions_mace=n_interactions_mace,
            n_interactions_schnet=n_interactions_schnet
        )
        input_modules = [pairwise_distance]
        print(f"Using NequIPMACESchNetFusion Model with {n_interactions_nequip} NequIP interactions, {n_interactions_mace} MACE interactions, and {n_interactions_schnet} SchNet interactions")
    
    elif model_type.lower() == 'nequip_mace_so3tensor_fusion':
        representation = NequIPMACESO3TensorFusion(
            n_atom_basis=n_atom_basis,
            n_radial=n_rbf,
            cutoff=cutoff,
            lmax=lmax,
            n_interactions_nequip=n_interactions_nequip,
            n_interactions_mace=n_interactions_mace
        )
        input_modules = [pairwise_distance]
        print(f"Using NequIPMACESO3TensorFusion Model with {n_interactions_nequip} NequIP interactions, {n_interactions_mace} MACE interactions")
    
    elif model_type.lower() == 'multi_model_fusion':
        representation = MultiModelFusion(
            n_atom_basis=n_atom_basis,
            n_radial=n_rbf,
            cutoff=cutoff,
            lmax=lmax,
            n_interactions_nequip=n_interactions_nequip,
            n_interactions_mace=n_interactions_mace,
            n_interactions_schnet=n_interactions_schnet
        )
        input_modules = [pairwise_distance]
        print(f"Using MultiModelFusion Model with {n_interactions_nequip} NequIP interactions, {n_interactions_mace} MACE interactions, and {n_interactions_schnet} SchNet interactions")
 
    elif model_type.lower() == 'hierarchical_fusion':
        representation = HierarchicalFusion(
            n_atom_basis=n_atom_basis,
            n_radial=n_rbf,
            cutoff=cutoff,
            lmax=lmax,
            n_interactions_nequip=n_interactions_nequip,
            n_interactions_mace=n_interactions_mace,
            n_interactions_schnet=n_interactions_schnet,
            n_interactions_painn=n_interactions_painn
        )
        input_modules = [pairwise_distance]
        print(f"Using MultiModelFusion Model with {n_interactions_nequip} NequIP interactions, {n_interactions_mace} MACE interactions")
        print(f" and {n_interactions_schnet} SchNet interactions and {n_interactions_painn} PaiNN interactions")
     
    
    else:
        raise ValueError(f"Invalid model_type '{model_type}'. Choose from: schnet, painn, schnet_so3conv, schnet_so3tensor, e3nn_so3conv, e3nn_so3tensor")

    output_modules = [
        CustomAtomwise(
            n_in=n_atom_basis,
            output_key='energy',
            n_layers=n_layers,
            n_neurons=n_neurons if n_neurons else [n_atom_basis] * (n_layers - 1),
            activation=nn.ReLU(),
            dropout_rate=dropout_rate
        ),
        spk.atomistic.Forces(energy_key='energy', force_key='forces')
    ]

    postprocessors = [trn.CastTo64(), trn.AddOffsets("energy", add_mean=True, add_atomrefs=False)]

    nnpot = spk.model.NeuralNetworkPotential(
        representation=representation,
        input_modules=input_modules,
        output_modules=output_modules,
        postprocessors=postprocessors,
    )

    output_energy = spk.task.ModelOutput(
        name="energy",
        loss_fn=torch.nn.MSELoss(),
        loss_weight=config["settings"]["outputs"]["energy"]["loss_weight"],
        metrics={"MAE": torchmetrics.MeanAbsoluteError()},
    )

    output_forces = spk.task.ModelOutput(
        name="forces",
        loss_fn=torch.nn.MSELoss(),
        loss_weight=config["settings"]["outputs"]["forces"]["loss_weight"],
        metrics={"MAE": torchmetrics.MeanAbsoluteError()},
    )

    outputs = [output_energy, output_forces]
    return nnpot, outputs

if __name__ == "__main__":
    import yaml
    with open("input.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    nnpot, outputs = setup_model(config)
    print("Model setup complete!")

