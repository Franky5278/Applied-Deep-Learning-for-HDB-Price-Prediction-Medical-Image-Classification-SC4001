import torch
from torch import nn


class PriceModel(nn.Module):
    def __init__(self, preprocessing, width=64, embedding_dim=8, variant="baseline", lam=0.0):
        super().__init__() #这行代码调用了父类 nn.Module 的构造函数，确保 PriceModel 类正确继承了 nn.Module 的属性和方法。
        #super() 函数用于调用父类的方法，而 __init__() 是父类的构造函数。
        #通过调用 super().__init__()，PriceModel 类可以继承 nn.Module 的初始化逻辑，从而能够使用 PyTorch 提供的神经网络功能，如参数管理、前向传播等。
        self.variant = variant #这一行代码将传入的 variant 参数赋值给 PriceModel 类的实例属性 self.variant。
        # variant 参数用于指定模型的变体类型，例如 "baseline"、"wide_deep" 或 "continuous_only"。
        # 通过将其存储为实例属性，模型可以在后续的前向传播和训练过程中根据不同的变体类型执行相应的逻辑。
        self.lam = float(lam) #lam 是一个浮点数，表示 wide 部分的权重系数。它用于在 wide_deep 模型中控制 wide 部分对最终输出的贡献。
        self.target_mean = float(preprocessing["target_mean"])
        self.target_std = float(preprocessing["target_std"])
        self.embeddings = nn.ModuleList() # i change the code here coz A2 NIDS continuous_only
        if variant != "continuous_only":
            self.embeddings.extend(
                nn.Embedding(cardinality, embedding_dim)
                for cardinality in preprocessing["cardinalities"]
            )
        input_width = 6 + (4 * embedding_dim if self.embeddings else 0)
        activation = nn.Sigmoid() if variant == "sigmoid" else nn.ReLU()
        # Relu是Rectified Linear Unit的缩写，是一种常用的激活函数。它的定义是f(x) = max(0, x)，即当输入x大于0时，输出为x本身；当输入x小于等于0时，输出为0。
        # Relu的优点是计算简单、收敛速度快，并且在深度神经网络中能够有效缓解梯度消失问题。
        #两个 linear layers 连起来，还是一个 linear transformation。加上ReLu，才变成了非线性 transformation。从而呢能拟合复杂的的房价关系
        self.mlp = nn.Sequential( #sequential是一个容器，里面的层会按照顺序依次执行
            nn.Linear(input_width, width), nn.LayerNorm(width), activation, nn.Linear(width, 1)
        )
        #layernorm 把当前 hidden vector 的数值 normalize。
        #layernorm 是network内部的一层，其对这一层sample的64个hidden values计算均值和方差，
        #让这一层的 values 数值尺度更规整，通常有利于稳定 optimization。
        #别和data processing搞混了
        #layernorm默认有两个learnable parameters，分别是scale和shift，分别对应每个hidden value的scale和shift
        #比如nn.LayerNorm(64)，两个参数加起来一共是64+64=128个learnable parameters
        #理解为normalize 完以后，模型自己还能学习“每个 hidden dimension 应该缩放多少、平移多少”

        #最后一层nn.Linear(width, 1)是输出层，输出一个标量，表示房价的预测值。参数的话就是weights64个+bias1个=65个learnable parameters
        self.wide = nn.Linear(input_width, 1) if variant == "wide_deep" else None
        #input_width表示送进hidden layer之前，一套房子最终有多少个数值features
        #width表示hidden layer有多少个neurons
        #activation对 normalize 后的每个 hidden value 做：h=g（z）
    def forward(self, categorical, continuous):
        embedded = [table(categorical[:, index].long()) for index, table in enumerate(self.embeddings)]
        features = torch.cat(embedded + [continuous.float()], dim=1)
        output = self.mlp(features).squeeze(1)
        if self.wide is not None:
            output = output + self.lam * self.wide(features).squeeze(1)
        return output

    def predict_price(self, categorical, continuous):
        return self(categorical, continuous) * self.target_std + self.target_mean


def parameter_breakdown(model):
    rows = []
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            rows.append({"name": name, "shape": list(parameter.shape), "count": parameter.numel()})
    return {"parameters": rows, "total": sum(row["count"] for row in rows)}
