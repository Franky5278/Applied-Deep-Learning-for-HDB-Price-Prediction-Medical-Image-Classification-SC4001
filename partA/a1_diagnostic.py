import csv
import hashlib
import inspect
import json
import os
import time
from pathlib import Path

PART_A = Path(__file__).resolve().parent
PROJECT = PART_A.parent
os.environ.setdefault('MPLCONFIGDIR', str(PROJECT / '.runtime/matplotlib'))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter
import pandas as pd
import torch
from torch.utils.data import DataLoader

from common.data import encode, read_partitions
from common.model import PriceModel
from common.train import evaluate, read_json, set_seed


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def plot_curve(rows, path):
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'axes.labelsize': 11, 'axes.titlesize': 11,
                         'legend.fontsize': 9, 'axes.spines.top': False,
                         'axes.spines.right': False})
    figure, axis = plt.subplots(figsize=(6.4, 4.2), constrained_layout=True)
    epochs = [row['epoch'] for row in rows]
    axis.plot(epochs, [row['train_rmse_sgd'] for row in rows], color='#0072B2',
              marker='o', linewidth=1.8, markersize=5, label='Train (2017–2021)')
    axis.plot(epochs, [row['test_rmse_sgd'] for row in rows], color='#D55E00',
              marker='s', linestyle='--', linewidth=1.8, markersize=5,
              label='Test (2022, visualization only)')
    axis.set(xlabel='Epoch', ylabel='RMSE (SGD)', xticks=epochs, xlim=(0.8, 6.2),
             title='A1 final fit: diagnostic replay')
    axis.yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
    axis.grid(axis='y', color='#D8D8D8', linewidth=0.6, alpha=0.7)
    axis.set_axisbelow(True)
    axis.legend(frameon=False, loc='best')
    with path.open('xb') as handle:
        figure.savefig(handle, format='png', dpi=450, facecolor='white')
    plt.close(figure)


def main():
    directory = PART_A / 'results/a1'
    figure_path = directory / 'a1_final_train_test_rmse_curve.png'
    csv_path = directory / 'a1_final_train_test_rmse_curve_diagnostic.csv'
    audit_path = PROJECT / 'logs/a1_diagnostic_replay_audit.json'
    for path in (figure_path, csv_path, audit_path):
        if path.exists():
            raise FileExistsError(f'已有文件不覆盖：{path}')
    protected = {path: digest(path) for folder in ('results', 'results_smoke')
                 for path in (PART_A / folder).rglob('*') if path.is_file()}
    began = time.perf_counter()
    reference = read_json(directory / 'final_seed42/result.json')
    config = reference['config']
    assert (config['width'], config['embedding_dim'], config['seed'], config['max_epochs']) == (256, 4, 42, 6)
    assert config['variant'] == 'baseline' and not config['early_stopping']
    assert config['learning_rate'] == 0.01 and config['batch_size'] == 512
    checkpoint = torch.load(directory / 'final_seed42/last.pt', map_location='cpu', weights_only=False)
    assert checkpoint['config'] == config and checkpoint['preprocessing'] == reference['preprocessing']
    data_path = PROJECT / 'data/hdb_price_prediction.csv'
    assert digest(data_path) == read_json(PART_A / 'results/metadata.json')['dataset_sha256']
    parts = read_partitions(data_path)
    preprocessing = reference['preprocessing']  # 原样读取最终运行的预处理。
    combined = pd.concat([parts['train'], parts['validation']])
    assert len(combined) == preprocessing['fit_rows'] == 116427
    train = encode(combined, preprocessing)
    test = encode(parts['test'], preprocessing)
    torch.set_num_threads(1)
    set_seed(config['seed'], 'cpu')
    model = PriceModel(preprocessing, config['width'], config['embedding_dim'])
    group, = checkpoint['optimiser']['param_groups']
    accepted = inspect.signature(torch.optim.Adam).parameters
    adam_settings = {key: value for key, value in group.items() if key != 'params' and key in accepted}
    optimiser = torch.optim.Adam(model.parameters(), **adam_settings)  # Match saved Adam settings.
    generator = torch.Generator().manual_seed(config['seed'])
    loader = DataLoader(train, batch_size=config['batch_size'], shuffle=True, generator=generator)
    rows = []
    with csv_path.open('x', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=['epoch', 'train_rmse_sgd', 'test_rmse_sgd'])
        writer.writeheader()
        for epoch in range(1, 7):  # 固定六轮，不设早停。
            model.train()
            for categorical, continuous, target in loader:
                optimiser.zero_grad(set_to_none=True)
                prediction = model(categorical, continuous)
                loss = torch.nn.functional.mse_loss(prediction, (target - model.target_mean) / model.target_std)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f'epoch {epoch}: loss 非有限值')
                loss.backward()
                optimiser.step()
            train_rmse = evaluate(model, train, config['batch_size'], 'cpu')['rmse']
            with torch.random.fork_rng(devices=[]):  # Test evaluation leaves training RNG unchanged.
                test_rmse = evaluate(model, test, config['batch_size'], 'cpu')['rmse']
            row = {'epoch': epoch, 'train_rmse_sgd': train_rmse, 'test_rmse_sgd': test_rmse}
            rows.append(row)
            writer.writerow(row)  # 每轮保存诊断值。
            handle.flush()
            print(f'epoch={epoch}/6 train_rmse={train_rmse:.6f} test_rmse={test_rmse:.6f}', flush=True)
    assert all(torch.equal(value, checkpoint['model'][name]) for name, value in model.state_dict().items())
    assert all(row['train_rmse_sgd'] == old['train_rmse'] for row, old in zip(rows, reference['history']))
    assert rows[-1]['test_rmse_sgd'] == reference['test']['rmse']
    current_state = optimiser.state_dict()['state']
    for parameter, old_state in checkpoint['optimiser']['state'].items():
        for key, value in old_state.items():
            assert torch.equal(current_state[parameter][key], value)
    plot_curve(rows, figure_path)
    assert all(path.exists() and digest(path) == old for path, old in protected.items())  # Verify original files byte for byte.
    audit = {'status': 'passed', 'kind': 'diagnostic_replay', 'device': 'cpu',
             'torch': torch.__version__, 'seed': 42, 'hidden_width': 256, 'embedding_dim': 4,
             'epochs': 6, 'train_samples': len(train), 'test_samples': len(test),
             'preprocessing_source': str(directory / 'final_seed42/result.json'),
             'adam_settings': adam_settings, 'early_stopping': False,
             'test_usage': 'visualization only; no effect on training, selection, stopping or checkpoints',
             'saved_checkpoints': False, 'original_final_weights_match': True,
             'original_adam_state_matches': True, 'original_training_rmse_matches_all_epochs': True,
             'protected_existing_files': len(protected), 'existing_file_hashes_unchanged': True,
             'diagnostic_csv': str(csv_path), 'figure': str(figure_path),
             'elapsed_seconds': time.perf_counter() - began}
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    with audit_path.open('x', encoding='utf-8') as handle:
        json.dump(audit, handle, ensure_ascii=False, indent=2)
    print(json.dumps(audit, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
