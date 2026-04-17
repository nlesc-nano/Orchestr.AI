import os
import glob
import shutil
import pandas as pd
import math
import re
from datetime import datetime
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

def extract_metrics(engine_dir, engine, engine_cfg):
    seed = engine_cfg.get('training', {}).get('seed', 42)
    standard_metrics_list = [
        'total_energy_rmse', 'total_energy_mae',
        'per_atom_energy_rmse', 'per_atom_energy_mae',
        'forces_rmse', 'forces_mae', 'weighted_sum'
    ]
    
    try:
        if engine in ['nequip', 'allegro', 'schnet', 'painn', 'fusion']:
            # Generic glob for all: lightning_logs/**/events.* (recursive, catches tutorial_log or not)
            event_files = glob.glob(os.path.join(engine_dir, 'lightning_logs/**/events.*'), recursive=True)
            # Sort by modification time (oldest first = train)
            event_files = sorted(event_files, key=os.path.getmtime)
            print(f"Debug: Found {len(event_files)} event files for {engine}, sorted by time: {event_files}")
            if not event_files:
                print(f"Warning: No event files found for {engine}")
                rows = build_placeholder_rows(engine, standard_metrics_list)
                return pd.DataFrame(rows)
            
            # Extract DFs based on count
            dfs = []
            max_files = 3 if engine in ['nequip', 'allegro'] else 2
            for i, event_file in enumerate(event_files[:max_files]):
                include_wall = (i == 0)  # Wall time for first (train)
                df = extract_metrics_data(event_file, include_wall_time=include_wall)
                dfs.append(df)
            
            if not dfs:
                rows = build_placeholder_rows(engine, standard_metrics_list)
                return pd.DataFrame(rows)
            
            # Common training time from first DF
            train_time = dfs[0]['WallTime'].max() - dfs[0]['WallTime'].min() if 'WallTime' in dfs[0].columns and not dfs[0].empty else float('nan')
            
            summaries = {}
            if engine in ['nequip', 'allegro']:
                # 3 files: train, val, test
                final_train = pd.Series()
                if dfs:
                    train_df = dfs[0][dfs[0]['Metric'].str.startswith('train_metric_epoch/')]
                    train_pivot = train_df.pivot(index='Step', columns='Metric', values='Value').reset_index(drop=True)
                    final_train = train_pivot.iloc[-1] if not train_pivot.empty else pd.Series()
                
                final_val = pd.Series()
                if len(dfs) > 1:
                    val_df = dfs[1][dfs[1]['Metric'].str.startswith('val0_epoch/')]
                    val_pivot = val_df.pivot(index='Step', columns='Metric', values='Value').reset_index(drop=True)
                    final_val = val_pivot.iloc[0] if not val_pivot.empty else pd.Series()
                
                final_test = pd.Series()
                if len(dfs) > 2:
                    test_df = dfs[2][dfs[2]['Metric'].str.startswith('test0_epoch/')]
                    test_pivot = test_df.pivot(index='Step', columns='Metric', values='Value').reset_index(drop=True)
                    final_test = test_pivot.iloc[0] if not test_pivot.empty else pd.Series()
                
                summaries = {'train': final_train, 'val': final_val, 'test': final_test}
                prefix_map = {'train': 'train_metric_epoch/', 'val': 'val0_epoch/', 'test': 'test0_epoch/'}
                metric_map = {m: m for m in standard_metrics_list}
                split_remap = {'train': 'train', 'val': 'val', 'test': 'test'}
            
            else:  # schnet/painn/fusion (2 files: train + metrics)
                schnet_pivot = pd.Series()
                if len(dfs) > 1:
                    pivot_df = dfs[1].pivot(index='Step', columns='Metric', values='Value').reset_index(drop=True)
                    schnet_pivot = pivot_df.iloc[0] if not pivot_df.empty else pd.Series()
                
                summaries = {}
                for sch_split in ['train', 'validation', 'testing']:
                    sch_filtered = {k: v for k, v in schnet_pivot.items() if k.startswith(sch_split + '/')}
                    summaries[sch_split] = sch_filtered
                
                metric_map = {
                    'total_energy_rmse': 'energy_rmse',
                    'total_energy_mae': 'energy_mae',
                    'per_atom_energy_rmse': None,
                    'per_atom_energy_mae': 'energy_mae_per_atom',
                    'forces_rmse': 'forces_rmse',
                    'forces_mae': 'forces_mae',
                    'weighted_sum': None
                }
                split_remap = {'train': 'train', 'validation': 'val', 'testing': 'test'}
            
            # Build rows (common for both)
            rows = []
            for orig_split, summary in summaries.items():
                row = {'model': engine, 'split': split_remap[orig_split]}
                for std in standard_metrics_list:
                    orig = metric_map.get(std)
                    if orig is not None:
                        key = f'{orig_split}/{orig}' if engine in ['schnet', 'painn', 'fusion'] else f'{prefix_map[orig_split]}{orig}'
                        row[std] = summary.get(key, float('nan'))
                    else:
                        row[std] = float('nan')
                row['training_time_seconds'] = train_time
                rows.append(row)
            
            return pd.DataFrame(rows)
        
        elif engine == 'mace':
            log_files = glob.glob(os.path.join(engine_dir, f'logs/mace_cdsecl_model_run-{seed}.log'))
            print(f"Debug: Found {len(log_files)} log files for mace: {log_files}")
            if not log_files:
                print(f"Warning: No MACE log file found for seed {seed}")
                rows = build_placeholder_rows(engine, standard_metrics_list)
                return pd.DataFrame(rows)
            
            mace_file = log_files[0]
            mace_train, mace_val, mace_train_time = parse_mace_log(mace_file)
            summaries = {'train': mace_train, 'val': mace_val, 'test': {}}
            
            mace_map = {m: m for m in standard_metrics_list}
            split_remap = {'train': 'train', 'val': 'val', 'test': 'test'}
            
            rows = []
            for orig_split, summary in summaries.items():
                row = {'model': engine, 'split': split_remap[orig_split]}
                for std in standard_metrics_list:
                    orig = mace_map.get(std)
                    if orig is not None:
                        row[std] = summary.get(orig, float('nan'))
                    else:
                        row[std] = float('nan')
                row['training_time_seconds'] = mace_train_time
                rows.append(row)
            
            return pd.DataFrame(rows)
        
        else:
            print(f"Warning: Unsupported engine {engine}")
            rows = build_placeholder_rows(engine, standard_metrics_list)
            return pd.DataFrame(rows)
    
    except Exception as e:
        print(f"Error in {engine}: {str(e)}")
        rows = build_placeholder_rows(engine, standard_metrics_list)
        return pd.DataFrame(rows)

def build_placeholder_rows(engine, standard_metrics_list):
    rows = []
    for s in ['train', 'val', 'test']:
        row = {'model': engine, 'split': s}
        for std in standard_metrics_list:
            row[std] = float('nan')
        row['training_time_seconds'] = float('nan')
        rows.append(row)
    return rows

def extract_metrics_data(event_file, include_wall_time=False):
    if not os.path.exists(event_file):
        raise FileNotFoundError(f"Event file not found at path: {event_file}")
    try:
        event_acc = EventAccumulator(event_file)
        event_acc.Reload()
        tags = event_acc.Tags().get('scalars', [])
        if not tags:
            return pd.DataFrame(columns=['Step', 'Metric', 'Value', 'WallTime'] if include_wall_time else ['Step', 'Metric', 'Value'])
        metrics_data = []
        for tag in tags:
            events = event_acc.Scalars(tag)
            for event in events:
                row = [event.step, tag, event.value]
                if include_wall_time:
                    row.append(event.wall_time)
                metrics_data.append(row)
        columns = ['Step', 'Metric', 'Value']
        if include_wall_time:
            columns.append('WallTime')
        df = pd.DataFrame(metrics_data, columns=columns)
        return df
    except Exception as e:
        raise RuntimeError(f"Error processing event file {event_file}: {e}")

def parse_mace_log(file_path):
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Extract start and end times for training duration
    start_match = re.search(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\.\d{3} INFO: Started training', content)
    end_match = re.search(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\.\d{3} INFO: Training complete', content)
    
    training_time = float('nan')
    if start_match and end_match:
        start_str = start_match.group(1)
        end_str = end_match.group(1)
        start_dt = datetime.strptime(start_str, '%Y-%m-%d %H:%M:%S')
        end_dt = datetime.strptime(end_str, '%Y-%m-%d %H:%M:%S')
        training_time = (end_dt - start_dt).total_seconds()
    
    # Extract loss from the last Epoch line (on val)
    loss_match = re.search(r"Epoch \d+: .*loss=([\d.]+)", content)
    val_loss = float(loss_match.group(1)) if loss_match else float('nan')
    
    # Extract table values
    train_per_atom_mae = float('nan')
    train_forces_mae = float('nan')
    val_per_atom_mae = float('nan')
    val_forces_mae = float('nan')
    
    train_line = re.search(r"\| train_Default \|([\s\d.]+)\|([\s\d.]+)\|", content)
    if train_line:
        train_per_atom_mae = float(train_line.group(1).strip()) / 1000  # meV to eV
        train_forces_mae = float(train_line.group(2).strip()) / 1000
    
    val_line = re.search(r"\| valid_Default \|([\s\d.]+)\|([\s\d.]+)\|", content)
    if val_line:
        val_per_atom_mae = float(val_line.group(1).strip()) / 1000
        val_forces_mae = float(val_line.group(2).strip()) / 1000
    
    mace_train = {
        'per_atom_energy_mae': train_per_atom_mae,
        'forces_mae': train_forces_mae
    }
    mace_val = {
        'per_atom_energy_mae': val_per_atom_mae,
        'forces_mae': val_forces_mae,
        'weighted_sum': val_loss
    }
    
    return mace_train, mace_val, training_time


def post_process_benchmark(benchmark_results_dir='./benchmark_results', engines=None):
    if engines is None:
        engines = ['schnet', 'painn', 'fusion', 'nequip', 'allegro', 'mace']
    
    results = []
    for engine in engines:
        engine_dir = os.path.join(benchmark_results_dir, engine)
        if not os.path.exists(engine_dir):
            print(f"Warning: No dir for {engine}")
            continue
        
        # Load engine_cfg from engine_yaml/engine_{engine}.yaml (for seed)
        yaml_path = glob.glob(os.path.join(engine_dir, 'engine_yaml/*.yaml'))
        engine_cfg = {}
        if yaml_path:
            with open(yaml_path[0], 'r') as f:
                engine_cfg = yaml.safe_load(f)
        
        engine_df = extract_metrics(engine_dir, engine, engine_cfg)
        results.append(engine_df)
    
    if results:
        combined_df = pd.concat(results, ignore_index=True)
        print("\nBenchmark Summary:\n")
        print(combined_df.to_markdown(index=False))
        combined_df.to_csv(os.path.join(benchmark_results_dir, 'benchmark_summary.csv'), index=False)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--post-process", action="store_true")
    parser.add_argument("--results-dir", default="./benchmark_results")
    args = parser.parse_args()
    if args.post_process:
        post_process_benchmark(args.results_dir)