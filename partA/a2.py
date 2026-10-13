from pathlib import Path

import numpy as np

from common.data import prepare_final, prepare_search
from common.plots import comparison_plot, diagnostic_plot, write_csv
from common.train import atomic_json, train_run


SEEDS = (42, 43, 44, 45, 46)  # Fixed seeds for all variants.


def summarize(rows):
    # 按模型汇总五种子结果。
    summary = []
    for name in dict.fromkeys(row["model"] for row in rows):
        group = [row for row in rows if row["model"] == name]
        result = {"model": name, "seeds": len(group)}
        for metric in ("rmse", "r2"):
            values = [row[f"test_{metric}"] for row in group]
            result[f"test_{metric}_mean"] = float(np.mean(values))
            result[f"test_{metric}_sample_std"] = float(np.std(values, ddof=1))  # Sample standard deviation.
        summary.append(result)
    return summary


def run(args, parts, output, a1):
    directory = Path(output) / "a2"
    directory.mkdir(parents=True, exist_ok=True)
    prefix = "smoke_a2" if args.smoke_test else "a2"
    seeds = SEEDS[:2] if args.smoke_test else SEEDS
    preprocessing, datasets = prepare_final(parts)  # 合并训练和验证分区。
    rows = []
    models = {}
    for variant in ("baseline", "continuous_only", "sigmoid"):
        for seed in seeds:
            config = {
                "width": a1["hidden_width"], "embedding_dim": a1["embedding_dim"], "seed": seed,
                "variant": variant, "max_epochs": a1["best_epoch"], "early_stopping": False,
                "learning_rate": 0.01, "batch_size": 512, "smoke_test": args.smoke_test,
            }
            if variant == "baseline" and seed == args.seed:
                run_directory = Path(output) / "a1" / f"final_seed{seed}"
            else:
                run_directory = directory / "final" / f"{variant}_seed{seed}"
            reuse_a1 = variant == "baseline" and seed == args.seed  # Reuse the matching A1 final fit.
            result = train_run(run_directory, preprocessing, datasets, config, args.device, args.force and not reuse_a1)
            rows.append({"model": variant, "seed": seed, "epochs": a1["best_epoch"], "test_rmse": result["test"]["rmse"], "test_r2": result["test"]["r2"]})
            models[f"{variant}_seed{seed}"] = result["model_path"]
            write_csv(directory / "a2_seed_results.csv", rows)
    summary = summarize(rows)
    write_csv(directory / "a2_summary.csv", summary)
    atomic_json(directory / "models.json", models)
    comparison_plot(summary, directory / f"{prefix}_test_rmse_comparison.png")
    comparison_plot(summary, directory / f"{prefix}_test_r2_comparison.png", "r2")
    search_preprocessing, diagnostic_datasets = prepare_search(parts)  # 诊断曲线不使用测试集。
    diagnostics = {}
    for variant in ("baseline", "continuous_only", "sigmoid"):
        config = {
            "width": a1["hidden_width"], "embedding_dim": a1["embedding_dim"], "seed": args.seed,
            "variant": variant, "max_epochs": a1["best_epoch"], "early_stopping": False,
            "monitor_validation": True, "evaluate_test": False,
            "learning_rate": 0.01, "batch_size": 512, "smoke_test": args.smoke_test,
        }
        diagnostics[variant] = train_run(directory / "diagnostics" / f"{variant}_seed{args.seed}", search_preprocessing, diagnostic_datasets, config, args.device, args.force)
    diagnostic_plot(diagnostics, directory / f"{prefix}_diagnostic_rmse.png", "rmse")
    diagnostic_plot(diagnostics, directory / f"{prefix}_diagnostic_r2.png", "r2")
    return {"summary": summary, "seeds": list(seeds), "models": models, "diagnostic_protocol": "2017-2020 train / 2021 validation; no test metrics"}
