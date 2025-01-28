import torch
import schnetpack as spk
import schnetpack.transform as trn
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
import torchmetrics

def setup_model(config, include_homo_lumo, include_bandgap, include_eigenvalues_vector):
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

    if include_homo_lumo:
        output_modules.extend([
            spk.atomistic.Atomwise(n_in=n_atom_basis, output_key='homo'),
            spk.atomistic.Atomwise(n_in=n_atom_basis, output_key='lumo')
        ])

    if include_bandgap:
        output_modules.append(spk.atomistic.Atomwise(n_in=n_atom_basis, output_key='bandgap'))

    if include_eigenvalues_vector:
        n_out = len(config['settings']['data']['eigenvalue_labels'])
        output_modules.append(spk.atomistic.Atomwise(n_in=n_atom_basis, output_key='eigenvalues_vector', n_out=n_out))

    postprocessors = [trn.CastTo64(), trn.AddOffsets('energy', add_mean=True, add_atomrefs=False)]
    if include_homo_lumo:
        postprocessors.append(trn.AddOffsets('homo', add_mean=True, add_atomrefs=False))
        postprocessors.append(trn.AddOffsets('lumo', add_mean=True, add_atomrefs=False))
    if include_bandgap:
        postprocessors.append(trn.AddOffsets('bandgap', add_mean=True, add_atomrefs=False))

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

    if include_homo_lumo:
        outputs.extend([
            spk.task.ModelOutput(name='homo', loss_fn=torch.nn.MSELoss(), loss_weight=config['settings']['outputs']['homo']['loss_weight'], metrics={"MAE": torchmetrics.MeanAbsoluteError()}),
            spk.task.ModelOutput(name='lumo', loss_fn=torch.nn.MSELoss(), loss_weight=config['settings']['outputs']['lumo']['loss_weight'], metrics={"MAE": torchmetrics.MeanAbsoluteError()})
        ])

    if include_bandgap:
        outputs.append(spk.task.ModelOutput(
            name='bandgap', loss_fn=torch.nn.MSELoss(),
            loss_weight=config['settings']['outputs']['gap']['loss_weight'],
            metrics={"MAE": torchmetrics.MeanAbsoluteError()}
        ))

    if include_eigenvalues_vector:
        outputs.append(spk.task.ModelOutput(
            name='eigenvalues_vector', loss_fn=homo_lumo_loss_fn,
            loss_weight=config['settings']['outputs']['eigenvalues_vector']['loss_weight'],
            metrics={"MAE": torchmetrics.MeanAbsoluteError()}
        ))

    return nnpot, outputs