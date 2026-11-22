# Python 环境材料与离线复现

Model-Maven 将三类环境材料分开管理：

| 材料 | 用途 | 能证明什么 |
| --- | --- | --- |
| `requirements.txt` | 用户声明的直接依赖 | 用户希望安装哪些依赖 |
| `requirements.lock.txt` 与 `.model-maven/environment.lock.json` | 已解析的 wheel 版本、SHA-256 和 Python ABI | 当前锁定依赖闭包及其哈希 |
| `.model-maven/wheelhouse/<lock-sha256>/` | 对应锁的实际 wheel 文件和 `index.json` | 可用这些归档文件在兼容 ABI 上离线安装 |

## 项目环境策略

默认策略创建隔离 venv，只安装锁定文件中的包。已有基础 Python 中的 PyTorch/CUDA/ROCm 等运行库不会自动继承。用户可明确选择共享基础 Python 的 distribution：

```sh
m-model env configure --system-site-packages
m-model env setup
```

这会把策略写入 `.model-maven/environment.json`，只影响后续新建的项目 venv。策略不可原地改变现有环境；若要换策略，指定尚不存在的新路径，例如 `m-model env configure --isolate-system-site-packages --venv .venv-isolated`。原环境不会被改写或删除。

启用继承后，锁元数据记录基础解释器身份、distribution 名称/版本清单、指纹及当时已有的 `pip check` 问题。`env check/setup` 会检查继承包是否漂移，并拒绝新增依赖冲突；已存在的基础环境警告会明确报告。项目 requirements 中声明的包仍以锁版本和 wheel SHA-256 安装到 venv，并覆盖同名继承包。继承策略提升了依赖系统 Python 包的便利性，但可移植性取决于接收机器上相同的基础 Python 包和系统运行库。

## Wheel archive 的范围

Archive 只为生成依赖锁时的 Python 实现、完整版本、平台、机器架构和 SOABI 采集文件。收集时使用当前 pip 配置的索引或 `find-links`，并要求每个 wheel 的名称、版本和 SHA-256 与锁完全一致。每个条目只记录 `configured-pip-source` 这一来源类别，不保存索引 URL、用户名、密码、token、pip 配置或网络响应内容。

当前方案不会把 OS 库、GPU 驱动、CUDA/ROCm 运行时、数据集、模型权重或外部服务归档进 wheelhouse。Python/ABI 不匹配时拒绝重放；同 ABI 也不意味着硬件驱动或加速器环境可移植。环境快照仍是已安装 distribution 的观察记录，不等同于带 artifact 哈希的 lock 或 wheel archive。

归档索引和归档文件分开检查：索引有效只说明清单可读；只有每个锁定包的 wheel 都存在、哈希和文件大小都匹配，且 Python ABI 与锁一致时，archive 才能用于离线安装。缺包或损坏会在安装开始前失败，不会返回成功状态。

## 离线 wheelhouse 命令

`m-model env wheelhouse archive`、`check` 与 `replay` 已连接到 `python/model_maven_worker/environment/wheelhouse.py` 中的 `archive_environment_wheels`、`inspect_wheel_archive` 和 `replay_wheel_archive` 服务。离线重放通过 pip 的 isolated、no-index 模式从归档目录安装锁定文件，再执行包清单、版本和依赖一致性核对。

建议交互形式：

```text
m-model env wheelhouse archive
m-model env wheelhouse check
m-model env wheelhouse replay --venv .venv-offline
```

命令应显示材料目录、包数量、ABI、缺失/损坏文件和下一步修复操作。`check` 必须允许单独报告 `index_available`、`archive_available` 与 `compatible`，避免把“索引存在”误报成“可离线重放”。
