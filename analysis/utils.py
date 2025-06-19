# utils.py
import pandas as pd
from io import BytesIO

def validate_columns(df, required_cols, platform):
    """
    Check if the DataFrame has all required columns. Returns True if valid, False with a message if not.
    """
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        return False, f"CSV is missing required columns for {platform}: {', '.join(missing)}. Please upload a valid CSV."
    return True, ""

def get_sample_schnet_csv():
    data = {
        'Step': [1, 2, 3, 1, 2, 3, 1, 2, 3],
        'Metric': ['train_loss', 'train_loss', 'train_loss', 'val_loss', 'val_loss', 'val_loss', 'train_energy_MAE', 'train_energy_MAE', 'train_energy_MAE'],
        'Value': [0.5, 0.4, 0.3, 0.6, 0.5, 0.4, 0.1, 0.09, 0.08]
    }
    df = pd.DataFrame(data)
    buffer = BytesIO()
    df.to_csv(buffer, index=False)
    buffer.seek(0)
    return buffer

def get_sample_nequip_csv():
    data = {
        'epoch': [1, 2, 3],
        'training_loss': [0.5, 0.4, 0.3],
        'validation_loss': [0.6, 0.5, 0.4],
        'training_f_mae': [0.1, 0.09, 0.08],
        'validation_f_mae': [0.11, 0.10, 0.09],
        'training_e_mae': [0.05, 0.04, 0.03],
        'validation_e_mae': [0.06, 0.05, 0.04]
    }
    df = pd.DataFrame(data)
    buffer = BytesIO()
    df.to_csv(buffer, index=False)
    buffer.seek(0)
    return buffer