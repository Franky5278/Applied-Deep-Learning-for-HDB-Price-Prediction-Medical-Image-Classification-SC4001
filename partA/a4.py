from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from common.data import FEATURES, encode
from common.plots import importance_plot, write_csv
from common.train import atomic_json, load_model, read_json


def explain(model_path, frame, directory, args, label):
    from captum.attr import FeatureAblation

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    prefix = f"{'smoke_' if args.smoke_test else ''}a4_{directory.name}"
    model, preprocessing = load_model(model_path, args.device)
    dataset = encode(frame, preprocessing)
    explainer = FeatureAblation(model.predict_price)  # 对新加坡元尺度的预测归因。
    pieces = []
    for index, (categorical, continuous, _) in enumerate(DataLoader(dataset, batch_size=128)):
        piece_path = directory / "batches" / f"batch{index:04d}.npy"  # Cache each attribution batch.
        if piece_path.exists() and not args.force:
            values = np.load(piece_path)
        else:
            categorical, continuous = categorical.to(args.device), continuous.to(args.device)
            baseline = torch.tensor(preprocessing["category_baselines"], device=args.device).reshape(1, 4)  # 类别参照为合并训练集众数。
            masks = (torch.arange(4, device=args.device).reshape(1, 4), torch.arange(4, 10, device=args.device).reshape(1, 6))  # One ID per original feature.
            with torch.no_grad():
                attributions = explainer.attribute(
                    inputs=(categorical, continuous), baselines=(baseline, torch.zeros(1, 6, device=args.device)),
                    feature_mask=masks, perturbations_per_eval=1,
                )
            values = torch.cat(attributions, dim=1).cpu().numpy()
            if values.shape != (len(categorical), 10) or not np.isfinite(values).all():
                raise ValueError("FeatureAblation 输出形状或数值异常")
            piece_path.parent.mkdir(parents=True, exist_ok=True)
            with piece_path.with_suffix(".tmp").open("wb") as handle:
                np.save(handle, values)
            piece_path.with_suffix(".tmp").replace(piece_path)
        pieces.append(values)
    values = np.concatenate(pieces)
    if values.shape != (len(frame), 10):
        raise ValueError("归因总样本数不正确")
    np.savez_compressed(directory / "test_attributions.npz", indices=frame.index.to_numpy(), attributions=values, features=np.asarray(FEATURES))
    importance = np.abs(values).mean(axis=0)  # 全局重要性取绝对归因均值。
    order = np.argsort(-importance)  # Rank features in descending order.
    rows = [{"feature": FEATURES[index], "mean_absolute_attribution_sgd": float(importance[index])} for index in order]
    write_csv(directory / "feature_importance.csv", rows)
    importance_plot(FEATURES, importance, directory / f"{prefix}_feature_importance.png", label)
    result = {
        "top_three": [FEATURES[index] for index in order[:3]], "importance": rows, "test_samples": len(frame),
        "model_path": str(model_path), "categorical_baseline": "most frequent category in final train+validation data",
        "continuous_baseline": "final train+validation feature mean (zero after standardization)",
        "units": "SGD", "feature_count": 10,
    }
    atomic_json(directory / "summary.json", result)
    return result


def run(args, parts, output, a1, a2, a3):
    directory = Path(output) / "a4"
    directory.mkdir(parents=True, exist_ok=True)
    seed = 42  # 两个模型使用相同训练种子。
    baseline_path = a2["models"][f"baseline_seed{seed}"]
    wide_path = a3["models"][str(seed)]
    baseline = explain(baseline_path, parts["test"], directory / "baseline", args, "A1 baseline")
    baseline_model, baseline_preprocessing = load_model(baseline_path, args.device)
    wide_model, wide_preprocessing = load_model(wide_path, args.device)
    same_weights = all(torch.equal(value.cpu(), wide_model.state_dict().get(name, torch.empty(0)).cpu()) for name, value in baseline_model.state_dict().items())
    identical = a3["lambda"] == 0 and baseline_preprocessing == wide_preprocessing and same_weights
    if identical:
        # Reuse attribution only when weights also match.
        wide = dict(baseline, model_path=wide_path, reused_identical_baseline=True)
        atomic_json(directory / "wide_deep" / "summary.json", wide)
    else:
        wide = explain(wide_path, parts["test"], directory / "wide_deep", args, f"A3 wide-and-deep, lambda={a3['lambda']}")
    result = {"seed": seed, "baseline": baseline, "wide_deep": wide, "identical_at_lambda_zero": identical}
    atomic_json(directory / "a4_summary.json", result)
    return result
