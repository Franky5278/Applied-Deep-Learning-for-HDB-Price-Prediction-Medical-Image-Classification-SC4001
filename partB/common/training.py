import os
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from .models import MODELS
from .runtime import atomic_json, load_json, seed_everything, write_csv


def atomic_checkpoint(path, state):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, temporary)
    os.replace(temporary, path)  # 临时文件写完再替换断点。


def rng_state():
    # Capture all RNG states used in training.
    return {
        "python": random.getstate(), "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def classification_metrics(labels, logits, loss):
    actual = np.asarray(labels, dtype=np.int64)
    predicted = np.asarray(logits) >= 0  # logit 零阈值对应概率 0.5。
    tp = int(((actual == 1) & predicted).sum())
    tn = int(((actual == 0) & ~predicted).sum())
    fp = int(((actual == 0) & predicted).sum())
    fn = int(((actual == 1) & ~predicted).sum())
    accuracy = (tp + tn) / len(actual)
    return {
        "loss": float(loss), "accuracy": accuracy, "classification_error": 1.0 - accuracy,
        "sensitivity": tp / (tp + fn) if tp + fn else None,
        "specificity": tn / (tn + fp) if tn + fp else None,
        "tp": tp, "tn": tn, "fp": fp, "fn": fn, "samples": len(actual),
    }


@torch.no_grad()
def evaluate(model, dataset, batch_size, device):
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    criterion = nn.BCEWithLogitsLoss(reduction="sum")  # Average over samples after summing loss.
    labels, logits, total_loss = [], [], 0.0
    for images, targets in loader:
        images, targets = images.to(device), targets.to(device)
        output = model(images)
        total_loss += float(criterion(output, targets).item())
        labels.extend(targets.cpu().tolist())
        logits.extend(output.cpu().tolist())
    return classification_metrics(labels, logits, total_loss / len(dataset)), np.asarray(logits, dtype=np.float32)


def train_model(model_name, data, directory, learning_rate, batch_size, seed, device,
                max_epochs=20, patience=4, min_delta=1e-4, final_epochs=None, force=False):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    seed_everything(seed)
    model = MODELS[model_name]().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)  # 按题目使用 Adam。
    criterion = nn.BCEWithLogitsLoss()
    generator = torch.Generator().manual_seed(seed)
    dataset = data.datasets["trainval" if final_epochs is not None else "train"]  # Final fits use train + validation.
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, generator=generator, num_workers=0)
    config = {
        "model": model_name, "learning_rate": learning_rate, "batch_size": batch_size,
        "seed": seed, "max_epochs": max_epochs, "patience": patience, "min_delta": min_delta,
        "final_epochs": final_epochs, "training_mean": data.mean, "training_std": data.std,
        "train_samples": len(dataset), "validation_samples": len(data.datasets["val"]),
    }
    state_path = directory / "last.pt"
    history, best_epoch, best_metrics = [], 0, None
    monitoring_best, bad_epochs, elapsed = float("inf"), 0, 0.0
    start_epoch = 1
    finished = False
    if state_path.exists() and not force:
        state = torch.load(state_path, map_location="cpu", weights_only=False)
        if state["config"] != config:
            raise ValueError(f"续跑配置与 checkpoint 不同：{directory}；请使用不同 output-dir 或 --force")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        generator.set_state(state["loader_generator"])  # 恢复数据打乱的随机状态。
        restore_rng(state["rng"])
        history, best_epoch, best_metrics = state["history"], state["best_epoch"], state["best_metrics"]
        monitoring_best, bad_epochs, elapsed = state["monitoring_best"], state["bad_epochs"], state["elapsed_seconds"]
        start_epoch, finished = state["epoch"] + 1, state["finished"]
        print(f"续跑 {directory.name}，已完成 {state['epoch']} 个 epoch", flush=True)
    limit = final_epochs if final_epochs is not None else max_epochs
    for epoch in range(start_epoch, limit + 1):
        if finished:
            break
        if device.type == "cuda":
            torch.cuda.synchronize(device)  # Wait for CUDA before starting the timer.
        began = time.perf_counter()
        model.train()
        labels, logits, total_loss = [], [], 0.0
        for images, targets in loader:
            images, targets = images.to(device), targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            output = model(images)
            loss = criterion(output, targets)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"训练 loss 非有限值：{directory.name}, epoch={epoch}")
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * len(images)  # Weight each batch by its sample count.
            labels.extend(targets.detach().cpu().tolist())
            logits.extend(output.detach().cpu().tolist())
        train_metrics = classification_metrics(labels, logits, total_loss / len(dataset))
        row = {"epoch": epoch, **{f"train_{key}": value for key, value in train_metrics.items()}}
        is_best = False
        if final_epochs is None:
            validation, _ = evaluate(model, data.datasets["val"], batch_size, device)
            if not np.isfinite(validation["loss"]):
                raise FloatingPointError(f"验证 loss 非有限值：{directory.name}, epoch={epoch}")
            row.update({f"val_{key}": value for key, value in validation.items()})
            if best_metrics is None or validation["loss"] < best_metrics["loss"]:
                # 最优断点取最低验证损失。
                best_epoch, best_metrics, is_best = epoch, validation, True
            if validation["loss"] < monitoring_best - min_delta:
                # min_delta controls the patience counter.
                monitoring_best, bad_epochs = validation["loss"], 0
            else:
                bad_epochs += 1
            finished = bad_epochs >= patience or epoch == limit
        else:
            finished = epoch == limit
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        row["epoch_seconds"] = time.perf_counter() - began
        elapsed += row["epoch_seconds"]
        history.append(row)
        state = {
            "config": config, "epoch": epoch, "model": model.state_dict(),
            "optimizer": optimizer.state_dict(), "rng": rng_state(), "loader_generator": generator.get_state(),
            "history": history, "best_epoch": best_epoch, "best_metrics": best_metrics,
            "monitoring_best": monitoring_best, "bad_epochs": bad_epochs, "elapsed_seconds": elapsed,
            "finished": finished,
        }
        if is_best:
            atomic_checkpoint(directory / "best.pt", state)
        atomic_checkpoint(state_path, state)
        write_csv(directory / "history.csv", history)
        atomic_json(directory / "history.json", history)
        message = f"{model_name} lr={learning_rate:g} batch={batch_size} epoch={epoch}/{limit} train_loss={train_metrics['loss']:.5f}"
        if final_epochs is None:
            message += f" val_loss={validation['loss']:.5f} val_acc={validation['accuracy']:.4f}"
        message += f" seconds={row['epoch_seconds']:.2f}"
        print(message, flush=True)
    if final_epochs is not None:
        final_path = directory / "final.pt"
        atomic_checkpoint(final_path, {
            "model": model.state_dict(), "config": config, "epochs": limit,
            "normalization": {"mean": data.mean, "std": data.std},
        })
        result = {**config, "epochs": limit, "checkpoint": str(final_path.resolve()), "elapsed_seconds": elapsed}
    else:
        result = {
            **config, "best_epoch": best_epoch,
            **{f"validation_{key}": value for key, value in best_metrics.items()},
            "epochs_run": len(history), "checkpoint": str((directory / "best.pt").resolve()),
            "elapsed_seconds": elapsed,
            "checkpoint_selection": "最低验证 BCE loss；min_delta 仅控制早停 patience 的计数",
        }
    atomic_json(directory / "result.json", result)
    return result


def final_fit_and_test(model_name, data, directory, learning_rate, batch_size, epochs, seed, device, force=False):
    directory = Path(directory)
    test_path = directory / "test_result.json"
    if test_path.exists() and not force:
        # 已完成的测试结果直接读取。
        return load_json(test_path)
    result = train_model(model_name, data, directory, learning_rate, batch_size, seed, device,
                         final_epochs=epochs, force=force)
    state = torch.load(result["checkpoint"], map_location="cpu", weights_only=False)
    model = MODELS[model_name]().to(device)
    model.load_state_dict(state["model"])
    test_metrics, logits = evaluate(model, data.datasets["test"], batch_size, device)
    result["test"] = test_metrics
    result["protocol"] = "新模型在 train+validation 固定 epoch 训练；测试集仅用于最终评估一次"
    atomic_json(test_path, result)
    write_csv(directory / "test_predictions.csv", [
        {"original_test_index": int(index), "label": int(label), "logit": float(logit),
         "probability": float(torch.sigmoid(torch.tensor(logit)).item()), "prediction": int(logit >= 0)}
        for index, label, logit in zip(data.indices["test"], data.labels["test"], logits)
    ])
    return result
