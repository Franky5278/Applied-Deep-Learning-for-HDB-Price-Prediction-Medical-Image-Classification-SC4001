import argparse
import hashlib
import json
import os
import platform
import sys
import time
import traceback
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / ".runtime" / "matplotlib"))

import numpy as np
import pandas as pd
import torch

from common.data import read_partitions
from common.train import atomic_json, read_json


def arguments():
    parser = argparse.ArgumentParser(description="SC4001 Part A 可恢复实验")
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).resolve().parents[1] / "data")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--stage", choices=("all", "a1", "a2", "a3", "a4"), default="all")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()
    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        parser.error("CUDA 不可用，请使用 --device cpu")
    if args.workers < 1 or args.threads < 1:
        parser.error("workers 和 threads 必须为正整数")
    args.csv_path = (args.data_dir / "hdb_price_prediction.csv").resolve()
    if not args.csv_path.exists():
        parser.error(f"数据文件不存在：{args.csv_path}")
    if args.output_dir is None:
        # Keep smoke outputs separate.
        args.output_dir = Path(__file__).resolve().parent / ("results_smoke" if args.smoke_test else "results")
    args.output_dir = args.output_dir.resolve()
    return args


def save_status(path, stages, smoke):
    # 各阶段状态即时保存。
    atomic_json(path, {
        "smoke_test": smoke, "stages": stages,
        "completed": [name for name, state in stages.items() if state["status"] == "completed"],
        "failed": [name for name, state in stages.items() if state["status"] == "failed"],
        "pending": [name for name, state in stages.items() if state["status"] in ("pending", "blocked", "running")],
    })


def report_notes(output, stages, results, smoke):
    # Summarize saved results only.
    lines = [
        "# Part A 实验记录", "", "这些记录仅从已保存结果生成。数值不能代替学生对模型和图的核查。",
        "", f"运行模式：{'小规模冒烟测试，不能作为正式结果' if smoke else '正式实验'}。",
        "", "搜索预处理仅拟合2017–2020训练数据；最终模型在2017–2021合并数据上重新拟合相同编码与标准化方法。2023数据被排除。",
        "", "A1最终训练按验证搜索选出的固定轮数执行，测试仅在最终模型上评估一次；最终训练图的测试点不是逐轮测试曲线，此处遵循TA澄清。",
        "", "A2的逐轮RMSE/R²图来自独立训练/验证诊断实验，测试集没有参与曲线或选择。五种子的标准差使用ddof=1。",
        "", "A3最终轮数采用选定lambda的最佳验证轮数。应讨论A1/A3轮数可能不同这一比较限制。",
        "", "A4类别消融到合并训练集众数，连续变量消融到合并训练集均值。结果依赖这些参照值，不能解读为因果影响。",
        "", "报告开头须按作业要求声明GenAI辅助代码生成和调试，并由学生如实填写人工检查过程。", "",
    ]
    for name, state in stages.items():
        lines.append(f"{name.upper()}：{state['status']}。")
        if state["status"] == "completed":
            result = results[name]
            if name == "a1":
                lines.append(f"宽度={result['hidden_width']}，嵌入维数={result['embedding_dim']}，最佳轮数={result['best_epoch']}；最终测试 RMSE={result['final']['test']['rmse']:.6f} SGD，R²={result['final']['test']['r2']:.6f}。")
            elif name in ("a2", "a3"):
                if name == "a3":
                    lines.append(f"lambda={result['lambda']}，最佳轮数={result['best_epoch']}。")
                for row in result["summary"]:
                    lines.append(f"{row['model']}：测试RMSE均值={row['test_rmse_mean']:.6f}，样本标准差={row['test_rmse_sample_std']:.6f}；测试R²均值={row['test_r2_mean']:.6f}，样本标准差={row['test_r2_sample_std']:.6f}。")
            elif name == "a4":
                lines.append(f"基线前三特征：{', '.join(result['baseline']['top_three'])}；宽深模型前三特征：{', '.join(result['wide_deep']['top_three'])}。")
        elif state.get("error"):
            lines.append(f"错误：{state['error']}")
        lines.append("")
    (Path(output) / "report_notes_partA.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    args = arguments()
    torch.set_num_threads(args.threads)
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    dataset_hash = hashlib.sha256(args.csv_path.read_bytes()).hexdigest()  # 校验续跑使用的数据。
    identity = {"smoke_test": args.smoke_test, "dataset_sha256": dataset_hash, "seed": args.seed}
    metadata_path = output / "metadata.json"
    if metadata_path.exists() and not args.force:
        # Reject a different run in the same directory.
        old = read_json(metadata_path)
        if any(old.get(key) != value for key, value in identity.items()):
            raise ValueError("输出目录已用于其他数据、种子或运行模式，请更换 --output-dir")
    parts = read_partitions(args.csv_path, args.smoke_test, args.seed)
    metadata = dict(identity, command_line=sys.argv, python=platform.python_version(), torch=torch.__version__, cuda=torch.version.cuda, numpy=np.__version__, pandas=pd.__version__, device=args.device, dataset_path=str(args.csv_path), dataset_rows={name: len(rows) for name, rows in parts.items()}, epochs_limit=2 if args.smoke_test else 20, early_stopping_patience=4, minimum_improvement=1e-4)
    atomic_json(metadata_path, metadata)
    print(json.dumps(metadata, ensure_ascii=False), flush=True)
    import a1
    import a2
    import a3
    import a4

    stages = {name: {"status": "pending"} for name in ("a1", "a2", "a3", "a4")}
    results = {}
    status_path = output / "status_summary.json"
    requested = list(stages) if args.stage == "all" else [args.stage]
    dependencies = {"a1": [], "a2": ["a1"], "a3": ["a1", "a2"], "a4": ["a1", "a2", "a3"]}  # 后续阶段依赖前序结果。
    for name in stages:
        if name not in requested:
            result_path = output / f"{name}_stage_result.json"
            if result_path.exists():
                results[name] = read_json(result_path)
                stages[name] = {"status": "completed", "loaded_previous_result": True}
    for name in requested:
        missing = [dependency for dependency in dependencies[name] if dependency not in results]
        if missing:
            stages[name] = {"status": "blocked", "error": f"缺少已完成的依赖阶段：{missing}"}
            save_status(status_path, stages, args.smoke_test)
            continue
        stages[name] = {"status": "running"}
        save_status(status_path, stages, args.smoke_test)
        start = time.monotonic()  # Time each stage separately.
        try:
            if name == "a1":
                result = a1.run(args, parts, output)
            elif name == "a2":
                result = a2.run(args, parts, output, results["a1"])
            elif name == "a3":
                result = a3.run(args, parts, output, results["a1"], results["a2"])
            else:
                result = a4.run(args, parts, output, results["a1"], results["a2"], results["a3"])
            results[name] = result
            atomic_json(output / f"{name}_stage_result.json", result)
            stages[name] = {"status": "completed", "elapsed_seconds": time.monotonic() - start}
        except Exception as error:
            # 记录异常，保留已完成结果。
            stages[name] = {"status": "failed", "error": str(error), "elapsed_seconds": time.monotonic() - start}
            (output / f"{name}_error.log").write_text(traceback.format_exc(), encoding="utf-8")
            traceback.print_exc()
        save_status(status_path, stages, args.smoke_test)
        report_notes(output, stages, results, args.smoke_test)
    print(json.dumps({"output": str(output), "stages": stages}, ensure_ascii=False, indent=2), flush=True)
    if any(stages[name]["status"] != "completed" for name in requested):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
