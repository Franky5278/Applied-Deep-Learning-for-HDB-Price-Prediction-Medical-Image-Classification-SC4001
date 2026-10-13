from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset


CATEGORICAL = ("month", "town", "flat_model_type", "storey_range")
CONTINUOUS = (
    "dist_to_nearest_stn",
    "dist_to_dhoby",
    "degree_centrality",
    "eigenvector_centrality",
    "remaining_lease_years",
    "floor_area_sqm",
)
FEATURES = CATEGORICAL + CONTINUOUS


def read_partitions(path, smoke=False, seed=42):
    #SEED是一个整数值，用于初始化随机数生成器，以确保在不同的运行中生成相同的随机数序列，从而保证实验的可重复性。
    #smoke = True表示进行快速测试，通常会使用较小的数据集或较少的训练轮数，以便快速验证代码的正确性和功能。
    #smoke = False表示进行完整的训练和评估，使用完整的数据集和训练轮数，以获得更准确的模型性能评估。
    frame = pd.read_csv(Path(path))
    required = set(FEATURES) | {"year", "resale_price"}
    if not required.issubset(frame.columns):
        raise ValueError(f"CSV 缺少字段：{sorted(required - set(frame.columns))}")
    if frame[list(required)].isna().any().any():
        raise ValueError("模型字段中存在缺失值")
    if not np.isfinite(frame[list(CONTINUOUS) + ["resale_price"]].to_numpy()).all():
        raise ValueError("连续字段中存在非有限数值")
    parts = {
        "train": frame.loc[frame.year.between(2017, 2020)].copy(),
        "validation": frame.loc[frame.year == 2021].copy(),
        "test": frame.loc[frame.year == 2022].copy(),
    }
    for name, rows in parts.items():
        if rows.empty:
            raise ValueError(f"{name} 数据为空")
        if smoke:
            count = 512 if name == "train" else 128
            parts[name] = rows.sample(min(count, len(rows)), random_state=seed).sort_index()
    return parts


def fit_preprocessor(frame):
#this funcion is to fit the preprocessor on the training data,
# and return a dictionary containing the fitted parameters.
    category_maps = {}
    category_baselines = []
    for name in CATEGORICAL:
        values = frame[name].astype(str)
        ordered = [str(value) for value in sorted(frame[name].unique().tolist())]
        category_maps[name] = {value: index + 1 for index, value in enumerate(ordered)}
        category_baselines.append(category_maps[name][values.mode().iloc[0]])
    continuous = frame[list(CONTINUOUS)].to_numpy(dtype=np.float32)
    target = frame.resale_price.to_numpy(dtype=np.float32)
    return {
        "category_maps": category_maps,
        "cardinalities": [len(category_maps[name]) + 1 for name in CATEGORICAL],
        "category_baselines": category_baselines,
        "continuous_mean": continuous.mean(axis=0).tolist(),#tolist（）是numpy数组的一个方法，用于将numpy数组转换为Python的列表。它会返回一个包含相同元素的列表。
        "continuous_std": np.maximum(continuous.std(axis=0, ddof=1), 1e-8).tolist(),
        "target_mean": float(target.mean()),
        "target_std": max(float(target.std(ddof=0)), 1e-8),
        "fit_rows": len(frame),
        "categorical_features": list(CATEGORICAL),
        "continuous_features": list(CONTINUOUS),
    }


def encode(frame, preprocessing):
    categorical = np.column_stack(
        [
            frame[name].astype(str).map(preprocessing["category_maps"][name]).fillna(0)
            for name in CATEGORICAL
        ]
    ).astype(np.int64)
    continuous = frame[list(CONTINUOUS)].to_numpy(dtype=np.float32)
    continuous = (continuous - np.asarray(preprocessing["continuous_mean"], dtype=np.float32)) / np.asarray(
        preprocessing["continuous_std"], dtype=np.float32
    )
    target = frame.resale_price.to_numpy(dtype=np.float32)
    return TensorDataset(
        torch.from_numpy(categorical), torch.from_numpy(continuous), torch.from_numpy(target)
    )


def prepare_search(parts):
    preprocessing = fit_preprocessor(parts["train"])
    return preprocessing, {name: encode(rows, preprocessing) for name, rows in parts.items()}
#语法上，encode(rows, preprocessing)的意思是调用encode函数，将rows和preprocessing作为参数传入。这里的rows是一个DataFrame，表示数据集中的一部分（train、validation或test），而preprocessing是一个字典，包含了预处理所需的参数和信息。
#name：encode 结构是一个字典推导式，用于创建一个字典，其中键是数据集的名称（train、validation或test），值是对应的数据集经过编码后的TensorDataset对象。最终返回的字典将包含三个键值对，分别对应train、validation和test数据集。
#encode（）对应rows吗？
# .items是什么？.items()是Python字典对象的一个方法，用于返回字典中的键值对。它返回一个可迭代的视图对象，其中每个元素都是一个包含键和值的元组（key, value）。在这个上下文中，parts.items()会返回一个包含("train", train_rows)、("validation", validation_rows)和("test", test_rows)的可迭代对象。

#在search阶段，train → 学 parameters
#validation → 选择 hyperparameters
#一旦 hidden width、embedding dimension、best epoch 都选完，validation 就不再需要承担“裁判”的作用了
#合起来，给最终模型更多 training examples。这样通常能够：更充分利用数据；
#更准确估计 weights / embeddings；
#降低因为训练样本有限带来的 variance；
#有机会改善最终 generalization。
#而 test 必须继续完全锁住，因为我们还需要一个没有参与任何选择或训练的数据集来估计最终 generalization。
def prepare_final(parts): #this function is to prepare the final dataset for training and testing, using the fitted preprocessor from the search phase.
# retrain is done on the combined training and validation data, and the test data is encoded using the same preprocessor.
    combined = pd.concat([parts["train"], parts["validation"]])
    #pd.concat()是pandas库中的一个函数，用于将多个DataFrame对象沿指定轴进行拼接或连接。它可以将多个DataFrame按行（axis=0）或按列（axis=1）进行合并，生成一个新的DataFrame。
    preprocessing = fit_preprocessor(combined)
    return preprocessing, {"train": encode(combined, preprocessing), "test": encode(parts["test"], preprocessing)}
#return 返回的是preprocessing和一个字典，字典中包含两个键值对，分别对应训练集和测试集的数据集。
