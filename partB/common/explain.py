from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from .models import DEFAULT_TARGETS, MODELS
from .runtime import atomic_json, load_json, write_csv


def gradcam(data, finals, output_dir, device, seed, target_layers=None, force=False, smoke=False):
    from captum.attr import LayerAttribution, LayerGradCam

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = "smoke_b4" if smoke else "b4"
    indices_path = output_dir / "selected_indices.json"
    if indices_path.exists() and not force:
        selection = load_json(indices_path)
        original_indices = selection["original_test_indices"]
        lookup = {int(original): local for local, original in enumerate(data.indices["test"])}
        local_indices = [lookup[index] for index in original_indices]
    else:
        candidates = np.flatnonzero(data.labels["test"] == 1)
        if len(candidates) < 5:
            raise ValueError("测试集的肺炎样本不足五张")
        local_indices = np.random.default_rng(seed).choice(candidates, size=5, replace=False).tolist()  # Sample cases without replacement.
        selection = {
            "seed": seed, "local_test_indices": local_indices,
            "original_test_indices": data.indices["test"][local_indices].tolist(),
            "selection": "在所有肺炎测试样本中等概率抽取五张，不根据预测结果筛选",
        }
        atomic_json(indices_path, selection)
    layers = dict(DEFAULT_TARGETS)  # 默认取最后的主路空间卷积。
    layers.update(target_layers or {})
    manifest, panel = [], plt.figure(figsize=(11, 13))
    for row, local in enumerate(local_indices):
        original = int(data.indices["test"][local])
        image = data.images["test"][local]
        plt.imsave(output_dir / f"{prefix}_original_{original}.png", image, cmap="gray", vmin=0, vmax=1)
        axis = panel.add_subplot(5, 4, row * 4 + 1)
        axis.imshow(image, cmap="gray", vmin=0, vmax=1)
        axis.set_title(f"Test {original}")
        axis.axis("off")
    for column, name in enumerate(("b1", "b2", "b3"), start=1):
        model = MODELS[name]().to(device)
        checkpoint = torch.load(finals[name]["checkpoint"], map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model"])
        model.eval()  # Disable dropout; use running BN stats.
        modules = dict(model.named_modules())
        if layers[name] not in modules or not isinstance(modules[layers[name]], torch.nn.Conv2d):
            raise ValueError(f"{name} Grad-CAM 层必须是空间卷积层：{layers[name]}")
        method = LayerGradCam(model, modules[layers[name]])
        model_dir = output_dir / name
        model_dir.mkdir(parents=True, exist_ok=True)
        for row, local in enumerate(local_indices):
            original = int(data.indices["test"][local])
            image = data.images["test"][local]
            inputs = data.datasets["test"].tensors[0][local:local + 1].to(device).requires_grad_(True)
            attribution = method.attribute(inputs, target=None, relu_attributions=True)  # 对肺炎原始 logit 求归因。
            enlarged = LayerAttribution.interpolate(attribution, (64, 64), interpolate_mode="bilinear")
            raw = enlarged.detach().cpu().numpy()[0, 0]
            maximum = float(raw.max())
            heatmap = raw / maximum if maximum > 0 else np.zeros_like(raw)  # Normalize each image independently.
            np.save(model_dir / f"attribution_{original}.npy", raw)  # 保留未经缩放的归因数组。
            np.save(model_dir / f"heatmap_{original}.npy", heatmap)
            plt.imsave(model_dir / f"{prefix}_{name}_heatmap_{original}.png", heatmap, cmap="jet", vmin=0, vmax=1)
            figure, axis = plt.subplots(figsize=(3, 3))
            axis.imshow(image, cmap="gray", vmin=0, vmax=1)
            axis.imshow(heatmap, cmap="jet", vmin=0, vmax=1, alpha=0.4)  # 固定透明度叠加原图。
            axis.axis("off")
            figure.savefig(model_dir / f"{prefix}_{name}_overlay_{original}.png", dpi=180, bbox_inches="tight", pad_inches=0)
            plt.close(figure)
            axis = panel.add_subplot(5, 4, row * 4 + column + 1)
            axis.imshow(image, cmap="gray", vmin=0, vmax=1)
            axis.imshow(heatmap, cmap="jet", vmin=0, vmax=1, alpha=0.4)
            axis.set_title(name.upper())
            axis.axis("off")
            with torch.no_grad():
                probability = float(torch.sigmoid(model(inputs)).item())
            # Record probability and attribution scale.
            manifest.append({"model": name, "original_test_index": original, "target_layer": layers[name],
                             "pneumonia_probability": probability, "raw_attribution_max": maximum})
    panel.tight_layout()
    panel.savefig(output_dir / f"{prefix}_gradcam_comparison.png", dpi=180)
    plt.close(panel)
    write_csv(output_dir / "gradcam_manifest.csv", manifest)
    result = {
        **selection, "target_layers": layers,
        "attribution_target": "肺炎原始 logit，不对 sigmoid 概率求梯度",
        "implementation": "Captum LayerGradCam, relu_attributions=True, bilinear interpolation",
        "heatmap_scale": "每张图以正归因的最大值归一化；零归因保存全零 heatmap",
        "overlay_alpha": 0.4, "manifest": manifest,
    }
    atomic_json(output_dir / "result.json", result)
    return result
