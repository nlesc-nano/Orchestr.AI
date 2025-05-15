import torch
import torch.nn as nn
import schnetpack as spk
import schnetpack.transform as trn
from torch.cuda.amp import autocast
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
import torchmetrics

class CustomAtomwise(spk.atomistic.Atomwise):
    def __init__(self, n_in, output_key, n_layers=1, n_neurons=None, activation=nn.ReLU(), dropout_rate=0.1):
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
        self.outnet = nn.Sequential(*layers)  # Rename to outnet

    def forward(self, inputs):
        feats = inputs['scalar_representation']
        #print(f"Input shape: {feats.shape}")
        atomwise_out = self.outnet(feats)  # Use outnet
        n_atoms = inputs['_n_atoms']
        atom_indices = torch.repeat_interleave(torch.arange(len(n_atoms), device=feats.device), n_atoms.long())
        with torch.cuda.amp.autocast(enabled=False):
           atomwise_out32 = atomwise_out.float()
           system_out = torch.zeros(len(n_atoms), 1, device=feats.device, dtype=torch.float32)
           system_out = system_out.scatter_add_(0, atom_indices.unsqueeze(1), atomwise_out32)

        inputs[self.output_key] = system_out.squeeze(1)
        return inputs

def setup_model(config):
    cutoff = config['settings']['model']['cutoff']
    n_rbf = config['settings']['model']['n_rbf']
    n_atom_basis = config['settings']['model']['n_atom_basis']
    n_interactions = config['settings']['model']['n_interactions']
    dropout_rate = config['settings']['model'].get('dropout_rate', 0.1)
    n_layers = config['settings']['model'].get('n_layers', 1)
    n_neurons = config['settings']['model'].get('n_neurons', None)

    pairwise_distance = spk.atomistic.PairwiseDistances()
    radial_basis = spk.nn.GaussianRBF(n_rbf=n_rbf, cutoff=cutoff)

    schnet = spk.representation.SchNet(
        n_atom_basis=n_atom_basis,
        n_interactions=n_interactions,
        radial_basis=radial_basis,
        cutoff_fn=spk.nn.CosineCutoff(cutoff)
    )

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

    postprocessors = [trn.CastTo64(), trn.AddOffsets('energy', add_mean=True, add_atomrefs=False)]

    nnpot = spk.model.NeuralNetworkPotential(
        representation=schnet,
        input_modules=[pairwise_distance],
        output_modules=output_modules,
        postprocessors=postprocessors
    )

    output_energy = spk.task.ModelOutput(
        name='energy',
        loss_fn=torch.nn.MSELoss(),
        loss_weight=config['settings']['outputs']['energy']['loss_weight'],
        metrics={"MAE": torchmetrics.MeanAbsoluteError()}
    )

    output_forces = spk.task.ModelOutput(
        name='forces',
        loss_fn=torch.nn.MSELoss(),
        loss_weight=config['settings']['outputs']['forces']['loss_weight'],
        metrics={"MAE": torchmetrics.MeanAbsoluteError()}
    )

    outputs = [output_energy, output_forces]
    return nnpot, outputs
