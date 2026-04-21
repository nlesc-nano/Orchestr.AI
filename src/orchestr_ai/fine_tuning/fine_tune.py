import os
import logging
import torch
import numpy as np

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping, Callback
from pytorch_lightning.loggers import CSVLogger 

import schnetpack as spk

from mlff_qd.training.trainer_utils import setup_task_and_trainer
from mlff_qd.utils.data_processing import ( load_data, preprocess_data, setup_logging_and_dataset,
        prepare_transformations, setup_data_module, show_dataset_info )
from mlff_qd.utils.model import setup_model
from mlff_qd.utils.helpers import load_config, parse_args, get_optimizer_class, get_scheduler_class
from mlff_qd.utils.logging_utils import setup_logging

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def freeze_layers(model, freeze_embedding=True, freeze_interactions_up_to=0, freeze_all_representation=False):
    schnet = model.representation
    num_interactions = len(schnet.interactions)
    
    if freeze_all_representation:
        for param in schnet.parameters():
            param.requires_grad = False
        logging.info("Froze all representation layers (embedding and interactions)")
        for module in model.output_modules:
            for param in module.parameters():
                param.requires_grad = True
        logging.info("Output modules remain trainable")
    else:
        if freeze_embedding:
            for param in schnet.embedding.parameters():
                param.requires_grad = False
            logging.info("Froze embedding layer")
        if freeze_interactions_up_to > 0:
            actual_frozen = min(freeze_interactions_up_to, num_interactions)
            for i in range(actual_frozen):
                for param in schnet.interactions[i].parameters():
                    param.requires_grad = False
            logging.info(f"Froze first {actual_frozen} of {num_interactions} interaction blocks")
            if freeze_interactions_up_to > num_interactions:
                logging.warning(f"Requested to freeze {freeze_interactions_up_to} blocks, but model has only {num_interactions}")

def log_trainable_layers(model):
    logging.info("Trainable layers after freezing:")
    for name, param in model.named_parameters():
        logging.info(f"{name}: {param.requires_grad}")

def print_layer_info(model):
    print("Layer Information:")
    print(f"{'Layer Name':<50} {'Shape':<20} {'Trainable':<10}")
    print("-" * 80)
    for name, param in model.named_parameters():
        print(f"{name:<50} {str(param.shape):<20} {param.requires_grad:<10}")

class SaveBestModelPt(Callback):
    def __init__(self, save_dir):
        super().__init__()
        self.save_dir = save_dir
        self.best_val_loss = float('inf')

    def on_validation_end(self, trainer, pl_module):
        current_val_loss = trainer.callback_metrics.get("val_loss", float('inf'))
        if current_val_loss < self.best_val_loss:
            self.best_val_loss = current_val_loss
            save_path = os.path.join(self.save_dir, "model.pt")
            torch.save(pl_module.model.state_dict(), save_path)
            logging.info(f"Saved best model (.pt) with val_loss={current_val_loss:.4f} to {save_path}")

def check_architecture_compatibility(state_dict, model): 
    pretrained_keys = set(state_dict.keys())
    model_keys = set(dict(model.named_parameters()).keys())
    missing_in_model = pretrained_keys - model_keys
    extra_in_model = model_keys - pretrained_keys
    
    if missing_in_model:
        logging.warning(f"Pre-trained model has keys not in fine-tuning model: {missing_in_model}")
    if extra_in_model:
        logging.warning(f"Fine-tuning model has extra keys not in pre-trained model: {extra_in_model}")
    return len(missing_in_model) == 0 and len(extra_in_model) == 0

def main(args):
    config = load_config(args.config)
    set_seed(config['general']['seed'])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Using device: {device}")

    folder = os.path.abspath(config['logging']['folder'])
    logging.info(f"Using output directory: {folder}")

    # some permision error, Verify folder exists and is writable
    try:
        os.makedirs(folder, exist_ok=True)
        logging.info(f"Verified or created directory: {folder}")
    except PermissionError as e:
        logging.error(f"Permission denied for {folder}: {e}")
        raise
    except OSError as e:
        logging.error(f"Error creating {folder}: {e}")
        raise

    data = load_data(config)
    use_last_n = config['data'].get('use_last_n', None) # to select sample of data
    atoms_list, property_list = preprocess_data(data)
    new_dataset, property_units = setup_logging_and_dataset(config, atoms_list, property_list)
    show_dataset_info(new_dataset)
    
    transformations = prepare_transformations(config,"train")
    custom_data = setup_data_module(
        config,
        os.path.join(folder, config['general']['database_name']),
        transformations,
        property_units
    )
    
    nnpot, outputs = setup_model(config)
    
    # Make sure fodler has checkpoints
    fine_tune_checkpoint = config['fine_tuning'].get('pretrained_checkpoint')
    if not fine_tune_checkpoint or not os.path.exists(fine_tune_checkpoint):
        raise FileNotFoundError(f"Pre-trained checkpoint not found at {fine_tune_checkpoint}")
    
    logging.info(f"Loading pre-trained model from {fine_tune_checkpoint}")
    checkpoint = torch.load(fine_tune_checkpoint, map_location=device)
    state_dict = checkpoint['state_dict']
     
    for key in state_dict:
        if 'postprocessors.1.mean' in key:
            if state_dict[key].shape == torch.Size([]):
                state_dict[key] = state_dict[key].reshape(1)
                logging.info(f"Reshaped {key} from scalar to torch.Size([1])")

    
    optimizer_name = config['training']['optimizer']['type']
    scheduler_name = config['training']['scheduler']['type']
    optimizer_cls = get_optimizer_class(optimizer_name)
    scheduler_cls = get_scheduler_class(scheduler_name)
    fine_tune_lr = config['fine_tuning'].get('lr', 1e-4) 
    
    # preapre the same architecture  
    task = spk.task.AtomisticTask(
        model=nnpot,
        outputs=outputs,
        optimizer_cls=optimizer_cls,
        optimizer_args={"lr": fine_tune_lr},
        scheduler_cls=scheduler_cls,
        scheduler_args={
            "mode": "min",
            "factor": config['training']['scheduler']['factor'],
            "patience": config['training']['scheduler']['patience'],
            "verbose": config['training']['scheduler']['verbose']
        },
        scheduler_monitor=config['logging']['monitor']
    )
    
    if not check_architecture_compatibility(state_dict, task.model):
        logging.warning("Architecture mismatch detected. Fine-tuning may be suboptimal.")
    
    task.load_state_dict(state_dict, strict=False)
    task.to(device)
    print_layer_info(task.model)
    
    freeze_embedding = config['fine_tuning'].get('freeze_embedding', True)
    freeze_interactions_up_to = config['fine_tuning'].get('freeze_interactions_up_to', 0)
    freeze_all_representation = config['fine_tuning'].get('freeze_all_representation', False)
    freeze_layers(task.model, freeze_embedding, freeze_interactions_up_to, freeze_all_representation)
    log_trainable_layers(task.model) # if True mean that we are training that layers
    
    # Use YAML-configured subdirectories merged with logging.folder
    folder = config['logging']['folder']
    best_model_subdir = config['fine_tuning'].get('best_model_dir', "fine_tuned_best_model")
    checkpoint_subdir = config['fine_tuning'].get('checkpoint_dir', "fine_tuned_checkpoints")
    log_name = config['fine_tuning'].get('log_name', "fine_tune_logs")
    
    best_model_dir = os.path.join(folder, best_model_subdir)
    checkpoint_dir = os.path.join(folder, checkpoint_subdir)
    
    os.makedirs(best_model_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    callbacks = [
        ModelCheckpoint(
            dirpath=checkpoint_dir,
            save_top_k=1,
            monitor=config['logging']['monitor'],
            filename="fine_tuned-{epoch}-{val_loss:.4f}"
        ),
        SaveBestModelPt(save_dir=best_model_dir),
        LearningRateMonitor(logging_interval='epoch')
    ]
    
    early_stopping_patience = config['fine_tuning'].get('early_stopping_patience', 0)
    if early_stopping_patience > 0:
        callbacks.append(
            EarlyStopping(
                monitor="val_loss",
                patience=early_stopping_patience,
                mode="min",
                verbose=True
            )
        )
        logging.info(f"Enabled EarlyStopping with patience={early_stopping_patience}")
    
    # Add CSVLogger like in trainer_utils.py
    csv_logger = CSVLogger(save_dir=folder, name="csv_logs", version="")
    tensorboard_logger = pl.loggers.TensorBoardLogger(save_dir=folder, name=log_name)
    logger = [tensorboard_logger, csv_logger]

    trainer = pl.Trainer(
        callbacks=callbacks,
        logger=logger,  # same as we do in training
        default_root_dir=folder,
        max_epochs=config['training']['max_epochs'],
        accelerator=config['training']['accelerator'],
        precision=config['training']['precision'],
        devices=config['training']['devices']
    )
    
    logging.info("Starting fine-tuning")
    trainer.fit(task, datamodule=custom_data)

if __name__ == '__main__':
    args = parse_args()
    setup_logging()
    logging.info(f"{'*' * 30} Fine-Tuning Started {'*' * 30}")
    main(args)
