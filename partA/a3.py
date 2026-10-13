from pathlib import Path

from a2 import SEEDS, summarize
from common.data import prepare_final, prepare_search
from common.plots import comparison_plot, learning_curve, write_csv
from common.train import atomic_json, train_run


def run(args, parts, output, a1, a2):
    directory = Path(output) / "a3"
    directory.mkdir(parents=True, exist_ok=True)
    prefix = "smoke_a3" if args.smoke_test else "a3"
    search_preprocessing, search_datasets = prepare_search(parts)
    lambdas = [0, 0.5] if args.smoke_test else [0, 0.25, 0.5, 1, 2]  # λ 网格以正式题目为准。
    trials = []
    for lam in lambdas:
        # Each lambda has its own early stopping.
        config = {
            "width": a1["hidden_width"], "embedding_dim": a1["embedding_dim"], "seed": args.seed,
            "variant": "wide_deep", "lambda": lam, "max_epochs": 2 if args.smoke_test else 20,
            "early_stopping": True, "learning_rate": 0.01, "batch_size": 512, "smoke_test": args.smoke_test,
        }
        result = train_run(directory / "search" / f"lambda{lam}_seed{args.seed}", search_preprocessing, search_datasets, config, args.device, args.force)
        trials.append(result)
        rows = [{"lambda": trial["config"]["lambda"], "best_validation_rmse": trial["best_validation_rmse"], "best_epoch": trial["best_epoch"], "epochs_run": trial["epochs_run"]} for trial in trials]
        write_csv(directory / "a3_search_results.csv", rows)
    selected = min(trials, key=lambda trial: (trial["best_validation_rmse"], trial["config"]["lambda"]))  # 按验证 RMSE 选择。
    chosen = {
        "lambda": selected["config"]["lambda"], "best_epoch": selected["best_epoch"],
        "best_validation_rmse": selected["best_validation_rmse"], "smoke_test": args.smoke_test,
        "epoch_protocol": "fresh train+validation fit for the selected lambda's best validation epoch count",
    }
    atomic_json(directory / "a3_selected.json", chosen)
    learning_curve(selected["history"], directory / f"{prefix}_selected_search_rmse.png", title=f"A3 selected lambda={chosen['lambda']}")
    preprocessing, datasets = prepare_final(parts)  # Refit preprocessing on train + validation.
    seeds = SEEDS[:2] if args.smoke_test else SEEDS
    rows, models = [], {}
    for seed in seeds:
        # Reinitialize for each seed.
        config = dict(selected["config"], seed=seed, max_epochs=chosen["best_epoch"], early_stopping=False)
        result = train_run(directory / "final" / f"wide_deep_lambda{chosen['lambda']}_seed{seed}", preprocessing, datasets, config, args.device, args.force)
        rows.append({"model": "wide_deep", "seed": seed, "lambda": chosen["lambda"], "epochs": chosen["best_epoch"], "test_rmse": result["test"]["rmse"], "test_r2": result["test"]["r2"]})
        models[str(seed)] = result["model_path"]
        write_csv(directory / "a3_seed_results.csv", rows)
    summary = summarize(rows)
    comparison = [row for row in a2["summary"] if row["model"] == "baseline"] + summary  # 对照 A2 的五种子基线。
    write_csv(directory / "a3_summary.csv", summary)
    write_csv(directory / "baseline_wide_deep_comparison.csv", comparison)
    comparison_plot(comparison, directory / f"{prefix}_test_rmse_comparison.png")
    comparison_plot(comparison, directory / f"{prefix}_test_r2_comparison.png", "r2")
    chosen.update({"summary": summary, "models": models, "seeds": list(seeds)})
    atomic_json(directory / "a3_selected.json", chosen)
    return chosen
