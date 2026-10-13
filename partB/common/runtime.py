import csv
import importlib.metadata
import json
import os
import platform
import random
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch


def atomic_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    os.replace(temporary, path)  # 临时文件写完再替换。


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_csv(path, rows):
    if not rows:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    fields = list(dict.fromkeys(key for row in rows for key in row))  # Keep column order across rows.
    with temporary.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def seed_everything(seed):
    # 设置各个随机源。
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False  # Avoid algorithm benchmarking.
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)


def choose_device(name):
    if name == "auto":
        name = "cuda:0" if torch.cuda.is_available() else "cpu"
    if name.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("指定了 CUDA，但当前环境没有可用 CUDA。")
    return torch.device(name)


def metadata(args):
    # 记录依赖版本和设备信息。
    versions = {}
    for name in ("torch", "numpy", "matplotlib", "captum"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return {
        "time_utc": datetime.now(timezone.utc).isoformat(),
        "command": [sys.executable, *sys.argv],
        "arguments": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "python": sys.version,
        "platform": platform.platform(),
        "versions": versions,
        "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "deterministic_algorithms": "warn_only",
    }


class StageStore:
    def __init__(self, root, force=False):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.path = self.root / "status_summary.json"
        self.force = force
        self.stages = {} if force or not self.path.exists() else load_json(self.path).get("stages", {})

    def save(self):
        atomic_json(self.path, {
            "stages": self.stages,
            "completed": [key for key, value in self.stages.items() if value["status"] == "completed"],
            "failed": [key for key, value in self.stages.items() if value["status"] == "failed"],
            "pending": [key for key, value in self.stages.items() if value["status"] in ("pending", "running")],
        })

    def register(self, names):
        # Declare expected stages before starting.
        for name in names:
            self.stages.setdefault(name, {"status": "pending"})
        self.save()

    def run(self, name, function):
        prior = self.stages.get(name, {})
        if prior.get("status") == "completed" and not self.force:
            # 续跑跳过已完成阶段。
            print(f"跳过已完成阶段：{name}", flush=True)
            return prior["result"]
        self.stages[name] = {"status": "running"}
        self.save()
        print(f"开始阶段：{name}", flush=True)
        try:
            result = function()
        except Exception as error:
            self.stages[name] = {"status": "failed", "error": repr(error), "traceback": traceback.format_exc()}
            self.save()
            with (self.root / "exceptions.log").open("a", encoding="utf-8") as handle:  # Append errors to the existing log.
                handle.write(f"\n{name}\n{traceback.format_exc()}")
            raise
        self.stages[name] = {"status": "completed", "result": result}
        self.save()
        return result
