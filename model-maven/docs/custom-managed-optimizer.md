# 受管步骤中的自定义优化器

受管步骤默认使用 AdamW；也可通过 `--optimizer-factory module:callable` 选择项目或环境中的 Python 工厂。工厂接收模型作为第一个参数，并接收 `--optimizer-param NAME=JSON` 指定的关键字参数，返回 `torch.optim.Optimizer`。例如：

```python
import torch

def make_optimizer(model, lr=0.01):
    return torch.optim.SGD(model.parameters(), lr=lr)
```

在已配置模型、批次和训练步骤的项目中，可在试跑命令追加 `--optimizer-factory my_optim:make_optimizer --optimizer-param lr=0.01`。统一工作区的受管步骤也提供此选择。工厂会在所选 Python 和运行时路径中导入并执行；参数和检查点可能被记录，不要传入密钥。

工具检查优化器只包含模型的可训练参数，允许选择子集并报告覆盖数量；它会检查梯度、参数和支持的优化器状态的有限性。工厂必须返回正常支持 `zero_grad(set_to_none=True)` 和无参数 `step()` 的 PyTorch 优化器；需要闭包或自行控制反向的算法应使用用户完整循环。工具不能证明任意优化器的数值正确性。

正式训练检查点记录工厂、参数及参数组身份，续训沿用原选择并检查工厂参数、优化器类型和参数组；源码快照和环境对照的限制与其他训练运行相同。已安装依赖的文件内容并未全部哈希，导入或工厂执行也可能产生不能回滚的副作用。自定义优化器源码或环境变化时不借用其他 callable 的状态迁移授权。

手动步骤可以在用户的训练代码中登记不同的工厂，例如 `context.optimizer("encoder", model.encoder.parameters(), factory=torch.optim.SGD, factory_params={"lr": 0.01})`。工厂接收参数列表和 JSON 关键字参数，必须返回 PyTorch 优化器，参数组按登记顺序覆盖所登记的参数。使用返回对象的 `zero_grad()` 和无闭包的 `step()`，并由步骤代码管理反向和更新次序；登记后不能改变参数分组。工厂须有稳定的模块及限定名称，检查点保存其名称、参数、类型与分组；参数会原样进入检查点，不能包含密钥。不同具名优化器不能重复登记同一参数。需要不满足这些约束的更新策略时，可自行管理完整训练循环。

一个可直接放入项目的手动步骤示例：

```python
import torch

def make_sgd(parameters, lr=0.1, momentum=0.9):
    return torch.optim.SGD(parameters, lr=lr, momentum=momentum)

def manual_step(model, batch, context):
    optimizer = context.optimizer(
        "main", model.parameters(), factory=make_sgd,
        factory_params={"lr": 0.1, "momentum": 0.9},
    )
    optimizer.zero_grad()
    loss = model(batch).square().mean()
    context.backward(loss)
    optimizer.step()
    return {"loss": loss.detach()}

manual_step.MODEL_MAVEN_STATELESS = True
```

将实际项目的批次与损失替换到步骤中，选择 `--training-mode manual-step --step my_train:manual_step`。此步骤本身无跨步状态；真实批次工厂的状态需要独立声明或实现保存和恢复，不能因示例的步骤无状态就假设批次也无状态。仅当用户确认批次状态允许重置时，才可在续训时显式使用 `--allow-state-reset`。

本机 CPU、PyTorch 2.10.0 的一次手动 SGD 运行完成两步并保存检查点，明确接受示例批次重置后从第 2 步续至第 3 步；其他优化器、硬件和批次状态契约尚未由此验证。
