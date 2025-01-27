
import schnetpack as spk
import torchmetrics

def build_model(config, transformations):
    schnet = spk.representation.SchNet(
        n_atom_basis=config['settings']['model']['n_atom_basis'],
        n_interactions=config['settings']['model']['n_interactions'],
        radial_basis=spk.nn.GaussianRBF(
            n_rbf=config['settings']['model']['n_rbf'],
            cutoff=config['settings']['model']['cutoff']
        ),
        cutoff_fn=spk.nn.CosineCutoff(config['settings']['model']['cutoff'])
    )

    return spk.model.NeuralNetworkPotential(
        representation=schnet,
        input_modules=transformations
    )

def get_task(model, config):
    outputs = [
        spk.task.ModelOutput(
            name='energy',
            loss_fn=torch.nn.MSELoss(),
            metrics={"MAE": torchmetrics.MeanAbsoluteError()}
        )
    ]
    return spk.task.AtomisticTask(
        model=model,
        outputs=outputs
    )
