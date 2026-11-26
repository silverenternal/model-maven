# 从空项目组装双输入模型

这是工程示例，不是模型效果或产品验收结论。需要已有可用的 Python、PyTorch 和统一工作区；输入与目标均为骨架生成的合成数据。操作会执行用户 Python，组件预览和提交前请查看界面的影响说明。

1. 在空目录打开工作区，按 `n` 创建模型。第一个输入填写 `left=4`，附加输入填写 `right=6`，具名输出填写 `score=1:mse`。查看生成的配置和源码后确认创建。
2. 复制源码仓库或 npm 安装中的 `examples/from_scratch_components.py` 到新项目根目录，保持文件名不变。npm 安装可用 `m-model-tui --examples-dir` 查找示例目录；这是 npm 启动器的选项。在编辑器中查看并保存该文件。它提供保留特征宽度的可训练编码器和可训练的拼接融合层；这些只是可替换的普通 Python 代码。
3. 按 `p` 使用合成输入请求首次前向。前向必须由用户确认，实际结果以运行报告为准；尚未前向不能宣称模型可用。
4. 按 `A`，选择“逐步组合多项修改”，三次选择“设置或替换组件”，分别填写下面的接缝、工厂和参数。选择“逐项填写参数”后，依次输入参数名和值，参数名留空表示完成；也可选择输入完整 JSON。每次完成后回到草稿菜单。工厂参数是用户可编辑的，不需要修改工具代码。

   | 接缝 | 工厂 | 参数名和值 |
   | --- | --- | --- |
   | `left_encoder` | `from_scratch_components:vector_encoder` | `width` → `4` |
   | `right_encoder` | `from_scratch_components:vector_encoder` | `width` → `6` |
   | `fusion` | `from_scratch_components:gated_concat` | `inputs` → `2` |

   也可选择“同时修改多项”，一次输入以下 JSON：

   ```json
   {"sets":{"left_encoder":"from_scratch_components:vector_encoder","right_encoder":"from_scratch_components:vector_encoder","fusion":"from_scratch_components:gated_concat"},"params":{"left_encoder.width":4,"right_encoder.width":6,"fusion.inputs":2}}
   ```

5. 在草稿菜单选择“预览装配后的模型”，审阅结果后决定是否提交，并选择保存时如何核对输入。提交会重新执行项目代码并更新配置；随后再次前向，检查实际输出及调用记录。可以直接修改组件源码，再保存并重新观察；对不支持的自动编辑继续使用原生源码编辑。
6. 可选择骨架生成的 `step` 做试跑，但示例目标不代表真实任务。接入实际数据、损失或训练循环前，需自行核对形状、数据类型和业务语义。

默认骨架的线性头按声明输入宽度计算；若自定义编码器改变输出宽度或返回嵌套结构，需要同时提供匹配的融合与预测头，或直接修改源码。骨架对动态轴求平均时保留浮点或复数精度，整数会转换为 PyTorch 默认浮点类型；默认融合也使用该默认类型。示例自定义融合层不强制转换输入 dtype；请根据自己的数据、参数精度和设备检查类型兼容性。动态导入、运行环境、数据和非 Python 资源不因这份示例而固定。

本机 Linux、Neovim 0.12.6、Python 3.12 / PyTorch 2.10.0 的一次走查完成了上述逐步装配、保存和独立前向，输出 `score` 的形状为 `[2, 1]`；随后逐项参数入口用一个编码器完成了真实预览并取消。逐项值按 JSON 解析，不能解析的内容作为普通文本；不传入的参数由工厂决定默认值。这不代表真实数据、其他环境或模型效果已验证。
