import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / ".runtime" / "matplotlib"))

import torch

from common.data import ChestData
from common.explain import gradcam
from common.models import MODELS, complexity
from common.plots import search_curves
from common.runtime import StageStore, atomic_json, choose_device, load_json, metadata, write_csv
from common.training import final_fit_and_test, train_model


LEARNING_RATES = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]  # 学习率网格来自 B1。
BATCH_SIZES = [16, 32, 64, 128, 256]  # Batch-size grid from B2.


def parse_args():
    parser = argparse.ArgumentParser(description="SC4001 Part B：搜索、重新训练及 Grad-CAM")
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).resolve().parents[1] / "data")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "cuda:0"], default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--stage", choices=["all", "b1", "b2", "b3", "b4"], default="all")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--gradcam-layer-b1", default=None)
    parser.add_argument("--gradcam-layer-b2", default=None)
    parser.add_argument("--gradcam-layer-b3", default=None)
    return parser.parse_args()


def choose_trial(results):
    # 错误率相同，再比较损失和轮数。
    return min(results, key=lambda row: (row["validation_classification_error"], row["validation_loss"], row["best_epoch"]))


def search(model_name, values, variable, other, args, data, output_dir, device, store):
    rows = []
    for value in values:
        config = {**other, variable: value}
        suffix = f"lr_{value:g}" if variable == "learning_rate" else f"batch_{value}"
        directory = output_dir / model_name / "search" / f"{suffix}_seed_{args.seed}"
        stage = f"{model_name}.search.{suffix}"
        # Bind the current config before scheduling.
        result = store.run(stage, lambda config=config, directory=directory: train_model(
            model_name, data, directory, seed=args.seed, device=device, force=args.force,
            max_epochs=2 if args.smoke_test else 20, patience=4, min_delta=1e-4, **config,
        ))
        rows.append(result)
        write_csv(output_dir / f"{model_name}_search_results.csv", rows)
    selected = dict(choose_trial(rows))
    selected["hyperparameter_selection"] = "各配置最低验证 loss checkpoint 的分类错误率最低；并列按 loss、best_epoch、列表顺序决定"
    atomic_json(output_dir / f"{model_name}_selected.json", selected)
    search_curves(rows, output_dir / model_name, variable, smoke=args.smoke_test)
    return selected


def final_model(model_name, selected, args, data, output_dir, device, store):
    # 新模型按最佳验证轮数重训。
    directory = output_dir / model_name / "final" / f"seed_{args.seed}_epochs_{selected['best_epoch']}"
    return store.run(f"{model_name}.final", lambda: final_fit_and_test(
        model_name, data, directory, selected["learning_rate"], selected["batch_size"],
        selected["best_epoch"], args.seed, device, force=args.force,
    ))


def compare(finals, output_dir):
    rows = []
    detailed = {}
    for name in ("b1", "b2", "b3"):
        structural = complexity(MODELS[name]())  # Use the same FLOP convention for all models.
        detailed[name] = structural
        rows.append({"model": name, **finals[name]["test"],
                     "learnable_parameters": structural["learnable_parameters"], "flops": structural["flops"],
                     "macs": structural["macs"], "learning_rate": finals[name]["learning_rate"],
                     "batch_size": finals[name]["batch_size"], "epochs": finals[name]["epochs"]})
    write_csv(output_dir / "b1_b3_comparison.csv", rows)
    atomic_json(output_dir / "complexity_details.json", detailed)
    return {"rows": rows, "flop_convention": detailed["b1"]["convention"]}


def report_notes(output_dir, args, selections, finals, comparison=None):
    # 文字摘要只读取已保存结果。
    lines = ["# Part B 报告事实记录", "",
             "冒烟测试结果只验证代码流程，不能用于作业报告。" if args.smoke_test else "以下数值直接读取本次保存结果；报告中的解释仍需结合曲线和图像人工检查。",
             "", "报告开头须按作业要求声明生成式 AI 的使用及人工验证过程。", "",
             "原图除以 255 后归一化；均值和标准差仅来自原始训练集。标签 0 为健康，1 为肺炎。",
             "优化器为 Adam；正式搜索最多 20 个 epoch，patience=4，min_delta=1e-4；最终模型从头在训练加验证集训练固定 epoch。",
             "搜索 checkpoint 按最低验证 BCE loss 保存，超参数按这些 checkpoint 的验证分类错误率选择。最终输出 raw logit，使用 BCEWithLogitsLoss。",
             "B1 隐藏全连接层各用 dropout=0.5；B2/B3 无 dropout。根据作业“每个卷积后 BN+ReLU”的说明，B3 主路第二卷积与投影 shortcut 也采用 BN+ReLU。", ""]
    for name in ("b1", "b2", "b3"):
        if name not in selections:
            continue
        selected = selections[name]
        lines.append(f"{name.upper()}：学习率 {selected['learning_rate']:g}，batch size {selected['batch_size']}，最低验证 loss 位于 epoch {selected['best_epoch']}。")
        if name in finals:
            test = finals[name]["test"]
            lines.append(f"测试 accuracy={test['accuracy']:.6f}，sensitivity={test['sensitivity']:.6f}，specificity={test['specificity']:.6f}，classification error={test['classification_error']:.6f}。")
        lines.append("")
    if comparison:
        lines.extend([comparison["flop_convention"], ""])
    b4_path = output_dir / "b4" / "result.json"
    if b4_path.exists():
        b4 = load_json(b4_path)
        lines.extend([f"Grad-CAM 原始测试索引：{b4['original_test_indices']}。",
                      f"目标层：{json.dumps(b4['target_layers'], ensure_ascii=False)}。",
                      "三种模型使用同样五张肺炎图像；热图按每张图的正归因最大值归一化，因此颜色强度不能直接比较跨图绝对归因。", ""])
    lines.extend(["需要人工完成的讨论：学习率/批大小对收敛的影响；类别比例与敏感性/特异性；模型计算成本的优缺点；热图是否落在可疑肺部区域。",
                  "Grad-CAM 是模型归因可视化，不能据此证明识别了临床病灶。64×64 分辨率及每模型一次随机初始化限制了结论。", ""])
    (output_dir / "report_notes_partB.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    args = parse_args()
    if args.threads < 1:
        raise ValueError("threads 必须至少为 1")
    torch.set_num_threads(args.threads)
    device = choose_device(args.device)
    default_name = "results_smoke" if args.smoke_test else "results"
    output_dir = (args.output_dir or Path(__file__).resolve().parent / default_name).resolve()
    if args.smoke_test and "smoke" not in output_dir.name.lower():
        output_dir = output_dir / "results_smoke"
    output_dir.mkdir(parents=True, exist_ok=True)
    data = ChestData(args.data_dir, smoke=args.smoke_test, seed=args.seed)
    description = data.description()
    # Guard against mixing datasets or run modes.
    manifest = {"smoke_test": args.smoke_test, "seed": args.seed, "data_sha256": description["sha256"],
                "data_shapes": description["used_shapes"], "protocol_version": 1}
    manifest_path = output_dir / "run_manifest.json"
    if manifest_path.exists() and load_json(manifest_path) != manifest:
        raise ValueError("当前 output-dir 的数据、seed 或 smoke 模式不同；请另选输出目录。")
    atomic_json(manifest_path, manifest)
    atomic_json(output_dir / "metadata.json", metadata(args))
    atomic_json(output_dir / "data_preprocessing.json", description)
    print(f"设备={device}，torch={torch.__version__}，CUDA={torch.version.cuda}，seed={args.seed}", flush=True)
    print(f"数据原始形状={data.original_shapes}；本次形状={description['used_shapes']}；输出={output_dir}", flush=True)
    if args.smoke_test:
        print("当前为小样本两轮冒烟测试，全部输出仅用于验证运行流程。", flush=True)
    store = StageStore(output_dir, force=args.force)
    expected = [f"b1.search.lr_{value:g}" for value in LEARNING_RATES]
    expected += ["b1.final"] + [f"b2.search.batch_{value}" for value in BATCH_SIZES]
    expected += ["b2.final", "b3.search", "b3.final", "comparison", "b4"]
    store.register(expected)  # 登记搜索、重训和归因阶段。
    selections, finals, comparison = {}, {}, None
    # B1 fixes the batch size at 128.
    selections["b1"] = search("b1", LEARNING_RATES, "learning_rate", {"batch_size": 128}, args, data, output_dir, device, store)
    finals["b1"] = final_model("b1", selections["b1"], args, data, output_dir, device, store)
    report_notes(output_dir, args, selections, finals)
    if args.stage == "b1":
        return
    # B2 沿用 B1 选定的学习率。
    selections["b2"] = search("b2", BATCH_SIZES, "batch_size", {"learning_rate": selections["b1"]["learning_rate"]}, args, data, output_dir, device, store)
    finals["b2"] = final_model("b2", selections["b2"], args, data, output_dir, device, store)
    report_notes(output_dir, args, selections, finals)
    if args.stage == "b2":
        return
    selections["b3"] = store.run("b3.search", lambda: train_model(
        "b3", data, output_dir / "b3" / "search" / f"seed_{args.seed}",
        selections["b1"]["learning_rate"], selections["b2"]["batch_size"], args.seed, device,
        max_epochs=2 if args.smoke_test else 20, force=args.force,
    ))
    atomic_json(output_dir / "b3_selected.json", selections["b3"])
    search_curves([selections["b3"]], output_dir / "b3", "learning_rate", smoke=args.smoke_test)
    finals["b3"] = final_model("b3", selections["b3"], args, data, output_dir, device, store)
    comparison = store.run("comparison", lambda: compare(finals, output_dir))
    report_notes(output_dir, args, selections, finals, comparison)
    if args.stage == "b3":
        return
    target_layers = {name: getattr(args, f"gradcam_layer_{name}") for name in MODELS if getattr(args, f"gradcam_layer_{name}")}  # Allow explicit Grad-CAM layer overrides.
    store.run("b4", lambda: gradcam(data, finals, output_dir / "b4", device, args.seed, target_layers=target_layers, force=args.force, smoke=args.smoke_test))
    report_notes(output_dir, args, selections, finals, comparison)
    summary = {"completed": [name for name, stage in store.stages.items() if stage["status"] == "completed"],
               "failed": [name for name, stage in store.stages.items() if stage["status"] == "failed"],
               "b1_learning_rate": selections["b1"]["learning_rate"], "b2_batch_size": selections["b2"]["batch_size"],
               "best_epochs": {name: selected["best_epoch"] for name, selected in selections.items()}, "output_dir": str(output_dir)}
    atomic_json(output_dir / "completion_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
