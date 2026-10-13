import hashlib
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import TensorDataset


def find_data(data_dir):
    directory = Path(data_dir).resolve()
    candidates = [directory] if directory.is_file() else [directory / "chestxray_binary_64.npz"]
    candidates.extend([directory.parent / "chestxray_binary_64.npz", directory / "data" / "chestxray_binary_64.npz"])
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"找不到 chestxray_binary_64.npz，指定路径为 {directory}")


def _array(archive, split, label=False):
    # Accept common NPZ key names.
    names = [f"y_{split}", f"Y_{split}", f"{split}_labels", f"{split}_y"] if label else [f"X_{split}", f"x_{split}", f"{split}_images", f"{split}_x"]
    for name in names:
        if name in archive:
            return archive[name]
    raise KeyError(f"数据缺少 {split} {'标签' if label else '图像'}；现有键：{archive.files}")


def _subset(labels, count, seed):
    rng = np.random.default_rng(seed)
    selected = []
    for label in (0, 1):
        # Smoke 子集按类别抽样。
        choices = np.flatnonzero(labels == label)
        number = min(len(choices), count // 2)
        selected.extend(rng.choice(choices, size=number, replace=False).tolist())
    return np.asarray(sorted(selected), dtype=np.int64)


class ChestData:
    def __init__(self, data_dir, smoke=False, seed=42):
        self.path = find_data(data_dir)
        self.images = {}
        self.labels = {}
        with np.load(self.path, allow_pickle=False) as archive:
            for split in ("train", "val", "test"):
                images = np.asarray(_array(archive, split))
                labels = np.asarray(_array(archive, split, label=True)).reshape(-1)
                if images.ndim == 4 and images.shape[-1] == 1:
                    images = images[..., 0]
                if images.ndim == 4 and images.shape[1] == 1:
                    images = images[:, 0]
                if images.ndim != 3 or images.shape[1:] != (64, 64):
                    raise ValueError(f"{split} 图像形状异常：{images.shape}")
                if len(images) != len(labels) or not np.isin(labels, [0, 1]).all():
                    raise ValueError(f"{split} 标签异常")
                if images.dtype != np.uint8:
                    raise ValueError(f"预期 uint8 灰度图，实际为 {images.dtype}")
                self.images[split] = images.astype(np.float32) / 255.0  # Scale intensities to [0, 1].
                self.labels[split] = labels.astype(np.float32)
        self.original_shapes = {split: list(images.shape) for split, images in self.images.items()}
        self.mean = float(self.images["train"].mean(dtype=np.float64))  # 统计量只取原始训练分区。
        self.std = float(self.images["train"].std(dtype=np.float64))
        if self.std <= 0:
            raise ValueError("训练集标准差为零")
        self.indices = {split: np.arange(len(images), dtype=np.int64) for split, images in self.images.items()}  # Keep original indices for Grad-CAM.
        if smoke:
            # 缩小样本不重新计算统计量。
            for split, count in (("train", 96), ("val", 32), ("test", 32)):
                chosen = _subset(self.labels[split], count, seed)
                self.indices[split] = chosen
                self.images[split] = self.images[split][chosen]
                self.labels[split] = self.labels[split][chosen]
        self.datasets = {}
        for split in self.images:
            normalized = (self.images[split] - self.mean) / self.std
            self.datasets[split] = TensorDataset(torch.from_numpy(normalized[:, None]), torch.from_numpy(self.labels[split]))
        self.datasets["trainval"] = TensorDataset(
            # 最终重训合并训练和验证集。
            torch.cat([self.datasets["train"].tensors[0], self.datasets["val"].tensors[0]]),
            torch.cat([self.datasets["train"].tensors[1], self.datasets["val"].tensors[1]]),
        )

    def description(self):
        # Record data hashes and preprocessing.
        digest = hashlib.sha256()
        with self.path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return {
            "file": str(self.path), "sha256": digest.hexdigest(),
            "original_shapes": self.original_shapes,
            "used_shapes": {split: list(images.shape) for split, images in self.images.items()},
            "class_counts": {split: {"healthy": int((labels == 0).sum()), "pneumonia": int((labels == 1).sum())} for split, labels in self.labels.items()},
            "training_mean": self.mean, "training_std": self.std,
            "normalization_source": "完整原始训练集；验证集与测试集不参与统计量计算",
            "label_mapping": {"0": "healthy", "1": "pneumonia"},
            "used_original_indices": {split: indices.tolist() for split, indices in self.indices.items()},
        }
