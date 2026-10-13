import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from .model import PriceModel


def atomic_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_checkpoint(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(data, temporary)
    os.replace(temporary, path)


def set_seed(seed, device="cpu"):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if str(device).startswith("cuda"):
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


@torch.no_grad() #@符号是Python中的装饰器语法，用于在函数定义之前应用装饰器。在这里，@torch.no_grad() 装饰器用于修饰 evaluate 函数，表示在该函数内部的所有操作都不需要计算梯度，从而节省内存和计算资源。
#torch.no_grad()现在只做 forward prediction，不需要构造 backpropagation graph。
def evaluate(model, dataset, batch_size, device):
    model.eval()
    predictions, targets = [], []
    for categorical, continuous, target in DataLoader(dataset, batch_size=batch_size):
        predictions.append(model.predict_price(categorical.to(device), continuous.to(device)).cpu())
        targets.append(target)
    prediction = torch.cat(predictions).double()
    target = torch.cat(targets).double()
    squared_error = (prediction - target).square().sum().item()
    total_variation = (target - target.mean()).square().sum().item()
    return {"rmse": (squared_error / len(target)) ** 0.5, "r2": 1 - squared_error / total_variation}


def train_run(directory, preprocessing, datasets, config, device="cpu", force=False):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    completed_path = directory / "result.json"
    checkpoint_path = directory / "last.pt"
    if completed_path.exists() and not force:
        result = read_json(completed_path)
        if result["config"] != config or result["preprocessing"] != preprocessing:
            raise ValueError(f"已有结果的配置不匹配：{directory}")
        return result
    set_seed(config["seed"], device)
    model = PriceModel(
        preprocessing, config["width"], config["embedding_dim"], config.get("variant", "baseline"), config.get("lambda", 0)
    ).to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=config.get("learning_rate", 0.01))
    #Adam optimizer 是一种自适应学习率优化算法，它结合了动量法和RMSProp的优点。它通过计算梯度的一阶矩估计（均值）和二阶矩估计（未中心化的方差）来调整每个参数的学习率，从而在训练过程中实现更快的收敛和更稳定的优化。
    #SGD 大家基本共用learning rate，Adam每个参数r 根据过去 gradient 的大小和方向，得到更 adaptive 的learning rate
    '''Adam：- 仍然是 gradient-based optimizer；
    - 仍然一个 mini-batch update 一次；
    - backprop 本身没有变；
    - 改的是 有了 gradient 之后，怎么更新 parameters。'''
    generator = torch.Generator().manual_seed(config["seed"])
    train_loader = DataLoader(datasets["train"], batch_size=config.get("batch_size", 512), shuffle=True, generator=generator)
    history = []
    best_rmse = float("inf")
    monitoring_best = float("inf")
    best_epoch = 0
    best_weights = None
    failures = 0
    start_epoch = 1
    early_stopping = config.get("early_stopping", True)
    monitor_validation = early_stopping or config.get("monitor_validation", False)
    if checkpoint_path.exists() and not force:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if checkpoint["config"] != config or checkpoint["preprocessing"] != preprocessing:
            raise ValueError(f"断点配置或数据预处理不匹配：{directory}")
        model.load_state_dict(checkpoint["model"])
        optimiser.load_state_dict(checkpoint["optimiser"])
        for state in optimiser.state.values():
            for name, value in state.items():
                if isinstance(value, torch.Tensor):
                    state[name] = value.to(device)
        history = checkpoint["history"]
        best_rmse, best_epoch = checkpoint["best_rmse"], checkpoint["best_epoch"]
        monitoring_best = checkpoint.get("monitoring_best", best_rmse)
        best_weights, failures = checkpoint["best_weights"], checkpoint["failures"]
        generator.set_state(checkpoint["generator_rng"])
        torch.set_rng_state(checkpoint["torch_rng"])
        np.random.set_state(checkpoint["numpy_rng"])
        random.setstate(checkpoint["python_rng"])
        if str(device).startswith("cuda") and checkpoint.get("cuda_rng") is not None:
            torch.cuda.set_rng_state_all(checkpoint["cuda_rng"])
        start_epoch = checkpoint["epoch"] + 1
    max_epochs = config["max_epochs"]
    if early_stopping and failures >= 4: # 如果在早停的情况下，连续4个epoch验证集的RMSE没有改善，则停止训练。
        start_epoch = max_epochs + 1 # 意思是如果已经连续4个epoch验证集的RMSE没有改善，那么就不再进行训练，直接将start_epoch设置为max_epochs + 1，从而跳过训练循环。
    for epoch in range(start_epoch, max_epochs + 1):
        began = time.perf_counter() #time.perf_counter()是Python标准库time模块中的一个函数，用于返回一个高精度的计时器值，表示从某个固定时间点（通常是程序启动时）到当前时刻的经过时间，单位为秒。它适用于测量代码执行时间，因为它提供了比time.time()更高的分辨率和精度。
        model.train() #train()是PyTorch中nn.Module类的一个方法，用于将模型设置为训练模式。在训练模式下，模型会启用一些特定的行为，例如启用dropout层和batch normalization层的训练行为，以便在训练过程中进行正则化和归一化操作。调用model.train()后，模型会根据训练数据进行前向传播和反向传播，从而更新模型的参数。
        for categorical, continuous, target in train_loader:
            categorical, continuous, target = categorical.to(device), continuous.to(device), target.to(device)
            optimiser.zero_grad(set_to_none=True) #zero_grad()是PyTorch中优化器（optimizer）对象的一个方法，用于将模型参数的梯度清零。
            #在每次反向传播之前，通常需要调用optimiser.zero_grad()来清除上一次迭代中计算得到的梯度，以避免梯度累积对当前迭代产生影响。set_to_none=True表示将梯度设置为None，而不是将其置为零，这样可以节省内存并提高性能。
            prediction = model(categorical, continuous) #forward propagation
            loss = nn.functional.mse_loss(prediction, (target - model.target_mean) / model.target_std)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"损失非有限数值：{directory} epoch={epoch}")
            loss.backward() #loss.backward()是PyTorch中用于计算反向传播的函数。它会根据当前计算图中的损失值（loss）自动计算模型参数的梯度，并将这些梯度存储在每个参数的.grad属性中。
            optimiser.step() #optimiser.step()是PyTorch中优化器（optimizer）对象的一个方法，用于根据计算得到的梯度更新模型参数。在调用loss.backward()计算出梯度后，调用optimiser.step()会使用这些梯度来调整模型的权重，从而使模型在训练数据上更好地拟合目标函数。
        row = {"epoch": epoch}
        row.update({f"train_{key}": value for key, value in evaluate(model, datasets["train"], config.get("batch_size", 512), device).items()})
        if monitor_validation:
            row.update({f"validation_{key}": value for key, value in evaluate(model, datasets["validation"], config.get("batch_size", 512), device).items()})
        if early_stopping:
            if best_weights is None or row["validation_rmse"] < best_rmse:
                best_rmse, best_epoch = row["validation_rmse"], epoch
                best_weights = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
            if row["validation_rmse"] < monitoring_best - 1e-4:
                monitoring_best, failures = row["validation_rmse"], 0
            else:
                failures += 1
        else:
            best_epoch = epoch
            best_weights = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
        row["epoch_seconds"] = time.perf_counter() - began
        history.append(row)
        save_checkpoint(checkpoint_path, {
            "model": model.state_dict(), "optimiser": optimiser.state_dict(), "epoch": epoch,
            "config": config, "preprocessing": preprocessing, "history": history,
            "best_rmse": best_rmse, "best_epoch": best_epoch, "best_weights": best_weights, "failures": failures,
            "monitoring_best": monitoring_best,
            "generator_rng": generator.get_state(), "torch_rng": torch.get_rng_state(),
            "numpy_rng": np.random.get_state(), "python_rng": random.getstate(),
            "cuda_rng": torch.cuda.get_rng_state_all() if str(device).startswith("cuda") else None,
        })
        atomic_json(directory / "history.json", history)
        print(f"{directory.name} epoch={epoch} train_rmse={row['train_rmse']:.3f}" + (f" validation_rmse={row['validation_rmse']:.3f}" if monitor_validation else ""), flush=True)
        if early_stopping and failures >= 4:
            break
    if best_weights is None:
        raise RuntimeError("未得到可用模型")
    model.load_state_dict(best_weights)
    save_checkpoint(directory / "model.pt", {"model": best_weights, "config": config, "preprocessing": preprocessing})
    result = {"config": config, "preprocessing": preprocessing, "best_epoch": best_epoch, "epochs_run": len(history), "history": history, "model_path": str((directory / "model.pt").resolve())}
    if monitor_validation:
        result["best_validation_rmse"] = best_rmse if early_stopping else min(row["validation_rmse"] for row in history)
        result["validation"] = evaluate(model, datasets["validation"], config.get("batch_size", 512), device)
    if config.get("evaluate_test", not monitor_validation):
        evaluation_path = directory / "test_metrics.json"
        if evaluation_path.exists() and not force:
            result["test"] = read_json(evaluation_path)
        else:
            result["test"] = evaluate(model, datasets["test"], config.get("batch_size", 512), device)
            atomic_json(evaluation_path, result["test"])
    result["train"] = evaluate(model, datasets["train"], config.get("batch_size", 512), device)
    atomic_json(completed_path, result)
    return result


def load_model(path, device="cpu"):
    saved = torch.load(path, map_location="cpu", weights_only=False)
    config = saved["config"]
    model = PriceModel(saved["preprocessing"], config["width"], config["embedding_dim"], config.get("variant", "baseline"), config.get("lambda", 0)).to(device)
    model.load_state_dict(saved["model"])
    model.eval()
    return model, saved["preprocessing"]
