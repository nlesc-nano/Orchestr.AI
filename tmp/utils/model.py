import torch
import schnetpack as spk
import schnetpack.transform as trn
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
import torchmetrics

def setup_model(config):
    cutoff = config['settings']['model']['cutoff']
    n_rbf = config['settings']['model']['n_rbf']
    n_atom_basis = config['settings']['model']['n_atom_basis']
    n_interactions = config['settings']['model']['n_interactions']
    pairwise_distance = spk.atomistic.PairwiseDistances()
    radial_basis = spk.nn.GaussianRBF(n_rbf=n_rbf, cutoff=cutoff)

    schnet = spk.representation.SchNet(
        n_atom_basis=n_atom_basis, n_interactions=n_interactions,
        radial_basis=radial_basis,
        cutoff_fn=spk.nn.CosineCutoff(cutoff)
    )

    output_modules = [
        spk.atomistic.Atomwise(n_in=n_atom_basis, output_key='energy'),
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