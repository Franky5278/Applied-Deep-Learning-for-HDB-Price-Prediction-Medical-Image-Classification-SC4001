from pathlib import Path
import os
import secrets

import torch

from common.data import prepare_final, prepare_search, read_partitions
from common.model import PriceModel, parameter_breakdown
from common.plots import architecture_plot, learning_curve, write_csv
from common.train import atomic_json, read_json, train_run


def trial_name(width, embedding_dim, seed):
    return f"width{width}_embedding{embedding_dim}_seed{seed}"


def _ray_trial(config, data_path, smoke, seed, directory, device, force, max_epochs, threads):
    from ray import tune

    torch.set_num_threads(threads)
    parts = read_partitions(data_path, smoke, seed)
    preprocessing, datasets = prepare_search(parts)
    training_config = {
        "width": config["width"], "embedding_dim": config["embedding_dim"], "seed": seed,
        "variant": "baseline", "max_epochs": max_epochs, "early_stopping": True,
        "learning_rate": 0.01, "batch_size": 512, "smoke_test": smoke,
    }
    result = train_run(Path(directory) / trial_name(config["width"], config["embedding_dim"], seed), preprocessing, datasets, training_config, device, force)
    tune.report({"best_validation_rmse": result["best_validation_rmse"], "best_epoch": result["best_epoch"]})


def run(args, parts, output):#自己定义的普通 Python function parameters
    #args是一个包含命令行参数的对象，parts是一个包含数据集划分的字典，output是输出目录的路径。
    #这三个参数来自于命令行参数解析和数据准备阶段，分别用于指定运行配置、提供数据集划分信息以及指定输出结果的存储位置。
    #他们源于lib吗
    runtime = Path(__file__).resolve().parents[1] / ".runtime"
    runtime.mkdir(exist_ok=True)
    token_path = runtime / "ray_auth_token"
    if not token_path.exists():
        token_path.write_text(secrets.token_hex(32), encoding="utf-8")
    os.environ.setdefault("RAY_AUTH_TOKEN_PATH", str(token_path.resolve()))
    os.environ.setdefault("RAY_AUTH_MODE", "token")
    from ray import tune
    import ray

    directory = Path(output) / "a1"
    directory.mkdir(parents=True, exist_ok=True)
    prefix = "smoke_a1" if args.smoke_test else "a1"
    widths = [16, 32] if args.smoke_test else [16, 32, 64, 128, 256]
    embedding_dims = [2] if args.smoke_test else [2, 4, 8, 12, 16]
    #ray是一个用于分布式计算的开源框架，它提供了简单易用的API来实现并行和分布式计算任务。Ray可以在单机、多机甚至云环境中运行，支持多种计算任务类型，如强化学习、超参数搜索、分布式训练等。
    #它通过将任务划分为小的子任务，并在多个工作节点上并行执行，从而提高计算效率和资源利用率。
    #Ray 里面专门做：hyperparameter tuning / experiment orchestration
    expected = [directory / "trials" / trial_name(width, embedding, args.seed) / "result.json" for width in widths for embedding in embedding_dims]
    if args.force or not all(path.exists() for path in expected):
        ray.init(num_cpus=args.workers, num_gpus=0 if args.device == "cpu" else 1,
                 include_dashboard=False, log_to_driver=True, ignore_reinit_error=True,
                 object_store_memory=512 * 1024 ** 2, _node_ip_address="127.0.0.1",
                 _temp_dir=str((runtime / "ray").resolve()))
        try:
            tune.run(
                tune.with_parameters(
                    _ray_trial, data_path=str(args.csv_path), smoke=args.smoke_test, seed=args.seed,
                    directory=str(directory / "trials"), device=args.device, force=args.force,
                    max_epochs=2 if args.smoke_test else 20, threads=args.threads,
                ),
                config={"width": tune.grid_search(widths), "embedding_dim": tune.grid_search(embedding_dims)},
                resources_per_trial={"cpu": 1, "gpu": 1 if args.device.startswith("cuda") else 0},
                metric="best_validation_rmse", mode="min", storage_path=str((directory / "ray").resolve()),
                name="grid", verbose=1, raise_on_failed_trial=False,
                trial_dirname_creator=lambda trial: f"w{trial.config['width']}_e{trial.config['embedding_dim']}",
            )
        finally:
            ray.shutdown()
    missing = [str(path) for path in expected if not path.exists()]
    if missing:
        raise RuntimeError(f"Ray 搜索未完成，缺少 {len(missing)} 个结果：{missing}")
    trials = [read_json(path) for path in expected]
    rows = [{"hidden_width": trial["config"]["width"], "embedding_dim": trial["config"]["embedding_dim"], "seed": args.seed, "best_validation_rmse": trial["best_validation_rmse"], "best_epoch": trial["best_epoch"], "epochs_run": trial["epochs_run"]} for trial in trials]
    write_csv(directory / "a1_search_results.csv", rows)
    selected = min(trials, key=lambda trial: (trial["best_validation_rmse"], trial["config"]["width"], trial["config"]["embedding_dim"]))
    #lambda函数是一个匿名函数，用于在min函数中指定比较的关键字。它接受一个trial对象作为输入，并返回一个元组，包含三个元素：trial["best_validation_rmse"]、trial["config"]["width"]和trial["config"]["embedding_dim"]。
    #min函数会根据这个元组的顺序进行比较，首先比较best_validation_rmse，如果相等，则比较width，如果仍然相等，则比较embedding_dim。最终返回的selected是具有最小best_validation_rmse的trial对象，如果有多个trial具有相同的best_validation_rmse，则选择具有最小width和embedding_dim的trial。
    #trials是一个包含所有搜索结果的列表，每个元素都是一个trial对象，包含了对应的配置和结果信息。selected是从trials中选出的最佳trial对象，具有最小的best_validation_rmse。
    #trial里的[]的比较顺序是什么?
    #min(..., key=...) 里的 key不是 dictionary 的 key:value，它是 Python 内置函数 min() 的一个参数，意思是：“比较每个元素时，用什么规则算它的比较值？”
    '''先trial["config"]得到：{
    "width": 256,
    "embedding_dim": 4
    }
    再["width"]得到256
    所以trial["config"]["width"]就是：先进入 config,再取 width

    lambda最终返回一个tuple:
    (
    validation_rmse,
    width,
    embedding_dim
    )
    python比 tuple 是从左到右比较。先比较best_validation_rmse
    如果 RMSE 完全相同，再比 width
    如果 width 也相同，再比 embedding_dim
    '''

    config = selected["config"]
    chosen = {
        "hidden_width": config["width"], "embedding_dim": config["embedding_dim"],
        "best_epoch": selected["best_epoch"], "best_validation_rmse": selected["best_validation_rmse"],
        "seed": args.seed, "search_backend": "ray_tune_grid_search", "trial_count": len(trials),
        "smoke_test": args.smoke_test,
    }
    atomic_json(directory / "a1_selected.json", chosen)
    learning_curve(selected["history"], directory / f"{prefix}_selected_search_rmse.png", title="A1 selected search configuration")
    preprocessing, datasets = prepare_final(parts)
    final_config = dict(config, max_epochs=chosen["best_epoch"], early_stopping=False)
    final = train_run(directory / f"final_seed{args.seed}", preprocessing, datasets, final_config, args.device, args.force)
    chosen["final"] = {"model_path": final["model_path"], "test": final["test"], "train": final["train"]}
    atomic_json(directory / "a1_selected.json", chosen)
    learning_curve(final["history"], directory / f"{prefix}_final_train_test_rmse.png", title="A1 final fit; test evaluated once", final_test=final["test"])
    search_preprocessing, _ = prepare_search(parts)
    original = PriceModel(search_preprocessing, 64, 8)
    selected_model = PriceModel(preprocessing, config["width"], config["embedding_dim"])
    counts = {"original_baseline": parameter_breakdown(original), "selected_final_model": parameter_breakdown(selected_model)}
    atomic_json(directory / "parameter_breakdown.json", counts)
    architecture_plot(search_preprocessing, 64, 8, counts["original_baseline"]["total"], directory / f"{prefix}_original_baseline_diagram.png")
    architecture_plot(preprocessing, config["width"], config["embedding_dim"], counts["selected_final_model"]["total"], directory / f"{prefix}_selected_model_diagram.png")
    return chosen
