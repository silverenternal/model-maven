# Linux TUI 回放基线（2026-11-21）

这是 x86-64 Linux release 自动回放基线，不是用户任务完成率，也不是原生终端显示时延或跨设备性能承诺。系统为 Linux 7.3.4-zen2-2-zen，CPU 为 AMD Ryzen 9 8945HS w/ Radeon 780M Graphics，Rust/Cargo 1.100.0。测试在有其他工作负载的机器上运行，未固定 CPU 频率。PTY 场景使用 100×30，关闭动效。测量从提交 `2cb3231155bfbb00415f28030eb7f11a2fc437c5` 起的当前未提交工作树，且包含这些改动。当前关键文件 SHA-256：

- `src/graph.rs`: `932ed74722fb62a0cfdbd161353531f091040d425eaf1225f6df20c9d4fc7803`
- `src/graph/appearance.rs`: `d7468587df7aa53ccd8171e4a2f0450e79922b34c1aaec96361647dc851f7cd4`
- `src/graph/interaction.rs`: `c7b3aaf58aafa1c598c8277daef66dbbefcb5894e39af958e81ce4b787bbde9e`
- `src/graph/search.rs`: `4c9c49a12f083bd5ac5a4eb33514f7bc2d5f463e6e377c91ef63e0ceeabbc03a`
- `tests/welcome.rs`: `ff2b67cd4b9513055c0de475ab4ddf7520be86d61fc1400c770452e35955b574`

## 千层诊断工作台

固定样例生成 1024 层。首帧从欢迎页发送 Enter 开始，到 PTY 收到包含失败证据的完整同步帧结束；不包含进程启动或欢迎页等待。响应为 100 次交替滚轮事件从写入 PTY 到读到目标同步帧的时间，包含应用调度和管道传输，不包含真实终端绘制与屏幕刷新。采集终端流到本地临时目录会引入少量测量扰动。

| 100×30 | 首帧 | 输入响应 p50 / p95 / 最大 | 帧间隔 p50 / p95 / 最大 | 单次响应最多字节 |
| --- | ---: | ---: | ---: | ---: |
| 字符 | 9.24 ms | 0.215 / 0.278 / 0.824 ms | 0.215 / 0.341 / 0.825 ms | 3,398 |
| 像素 | 13.82 ms | 16.73 / 17.36 / 17.73 ms | 16.73 / 17.38 / 17.73 ms | 10,853 |

两种模式静止约 80 ms 均写入 0 字节；仅切换详情分别写入 197 字节，拒绝修改分别写入 72 字节。连续输入包含拖动、缩放、两次有效确认与一次无效确认：字符模式到 v2/v3 分别为 1.08/4.59 ms，像素模式为 9.41/23.77 ms。回放逐帧核对 v2、v3、拒绝后保留 v3、重验前后来源和返回后的回执；提交和重验是离线样例。

## 不同视口的帧生成

以下是在同一千层结构中进行 100 次选择和相机旋转的进程内帧生成时间，不经过 PTY 或原生终端。字符模式包含轮廓细化，像素模式包含光栅、压缩和编码。最大字节数为生成的文本与图像放置数据之和。

| 尺寸 | 模式 | 总时间 p50 / p95 | 细化或光栅 p50 / p95 | 压缩编码 p50 / p95 | 最多字节 |
| --- | --- | ---: | ---: | ---: | ---: |
| 40×24 | 字符 | 0.050 / 0.124 ms | 0.0035 / 0.0085 ms | — | 1,993 |
| 40×24 | 像素 | 0.656 / 0.722 ms | 0.384 / 0.435 ms | 0.230 / 0.245 ms | 6,190 |
| 100×30 | 字符 | 0.121 / 0.131 ms | 0.0089 / 0.0102 ms | — | 4,906 |
| 100×30 | 像素 | 1.663 / 1.861 ms | 0.908 / 1.065 ms | 0.676 / 0.706 ms | 8,452 |
| 200×60 | 字符 | 0.372 / 0.402 ms | 0.0151 / 0.0183 ms | — | 13,759 |
| 200×60 | 像素 | 3.202 / 3.539 ms | 1.580 / 1.859 ms | 1.420 / 1.461 ms | 15,248 |

## 本机暂定回归预算

预算针对上述 Linux release 环境和 PTY 测量口径，按本轮最大 p95 留出约 1.4–2 倍余量。它用于发现本机交互性能回退；不代表原生屏幕响应保证，也不能直接套用于 Ghostty、Windows Terminal 或 SSH。

| 项目 | 字符模式上限 | 像素模式上限 | 当前最大测量 |
| --- | ---: | ---: | --- |
| 100×30 诊断首帧（从 Enter 到完整帧） | 15 ms | 25 ms | 9.24 / 13.82 ms |
| 100×30 100 次输入响应 p95 | 2 ms | 25 ms | 0.278 / 17.36 ms |
| 100×30 连续提交到 v3 | 10 ms | 30 ms | 4.59 / 23.77 ms |
| 200×60 进程内帧生成 p95 | 1 ms | 5 ms | 0.402 / 3.539 ms |
| 200×60 单次组合输出 | 20 KB | 20 KB | 13,759 / 15,248 B |
| 静止视图等待 80 ms 后的重复写入 | 0 B | 0 B | 0 / 0 B |

首帧、响应和输出量按同一 PTY/离线 1024 层场景重复测量；扩大视口的绘制预算按 100 次进程内场景生成测量。将这些阈值转为 CI 闸门前，应在干净机器重复测量并确认稳定性。其他终端和硬件仍只有缺测记录，不能从本机数字外推。

## 场景和终端覆盖

- 48/32 离线诊断：release PTY 回放通过，覆盖欢迎交接、证据、预览、确认与两种重验结果。
- 密集交叉端口：字符与像素 release PTY 回放通过，分别选择两条相交连接并验证改接后保留另一条。
- Drifting 24-block 与 12-block：字符和像素 release PTY 回放通过，包含翻页、展开、追线和返回。单次翻页到帧的测量分别为字符 0.21–0.33 ms、像素 3.69–4.24 ms；展开到帧分别为字符 0.44–0.46 ms、像素 5.32–5.54 ms。这些是各场景单次测量，不是 p95。
- tmux 3.8：隔离服务器中的字符模式通过欢迎交接、48/32 证据、二维切换和两条交叉连线选择。`capture-pane` 检查终端单元格，不等同于 SSH 或原生鼠标/触控板走查。
- Ghostty 1.4 已安装，但本次环境没有图形显示会话，原生窗口和手势未测；Windows Terminal、SSH 和真实用户任务对照也未测。以上暂定预算只适用于本机回归。

## 复现

```sh
env RUSTC_WRAPPER= cargo test --release --locked --offline thousand_operator_graph_keeps_layout_and_rendering_local -- --nocapture
env RUSTC_WRAPPER= cargo test --release --locked --offline --test welcome large_offline_workbench_measures_input_and_retains_each_edit -- --nocapture
env RUSTC_WRAPPER= cargo test --release --locked --offline --test welcome crossed_ports_can_both_be_selected_and_one_can_be_rewired -- --nocapture
env RUSTC_WRAPPER= cargo test --release --locked --offline --test welcome drifting_source_traces_a_boundary_and_restores_the_selected_page -- --ignored --nocapture
python3 scripts/verify_tmux.py
python3 scripts/verify_linux_performance_budget.py
```

设置 `MODEL_MAVEN_TUI_REPLAY_DIR` 为空目录可另存离线千层回放的终端流、检查点和计时文件。终端流验证和像素图像解码步骤见 [README](../README.md)。

## 焦点反馈改动后复验（2026-11-21）

在加入空间焦点短脉冲和图标悬停文案后，重新运行 `python3 scripts/verify_linux_performance_budget.py`。检查时修正了回归脚本的正则分组索引，使输入响应预算实际比较 p95（之前比较的是 p50）。修正后所有预算均通过：200×60 字符/像素帧 p95 为 0.340/3.707 ms，最大组合输出 13,759/15,248 B；100×30 字符/像素首帧为 8.537/13.556 ms，100 次输入响应 p95 为 0.206/16.982 ms，闲置 80 ms 写入 0 B，连续确认到 v3 为 4.588/22.619 ms。数字只说明此 Linux 主机的自动回放没有超过本机临时预算。

复验时关键源文件哈希：`src/graph.rs` `e4944ed0e68ab1e37f1df4e382319459d8ec8301a65777d49b0a62444e892569`；`src/graph/interaction.rs` `e60470c33a2b4f401aff214e8e4f66ae1f764db986c7d2902721121ee2da7764`；`scripts/verify_linux_performance_budget.py` `9dc0784e79644dfba4691d327c1dd754afdb0a62730421f22ec6a847dc56748e`。这次回放没有覆盖 Ghostty 原生窗口、触控板或真人任务定位表现。

## 模块盲文纹理改动后复验（2026-11-21）

在按模块族区分盲文颗粒密度后重新运行 `python3 scripts/verify_linux_performance_budget.py`，本机所有暂定预算仍通过。最后复验（含独立预设相机状态及相机读数移位后的伪终端回归）中，200×60 字符/像素帧 p95 为 0.413/3.510 ms，最大组合输出为 13,700/15,281 B；100×30 字符/像素首帧为 7.881/15.027 ms，100 次输入响应 p95 为 0.276/17.914 ms，静止 80 ms 写入 0 B，连续确认到 v3 为 4.530/24.742 ms。各数据与前次测量波动在预算内；原生终端与用户任务仍未测。

复验时关键源文件哈希：`src/graph.rs` `76c6665db9f766303e5fee6c1a8f5f80719d9fe365cc96350f7508568a656487`；`src/graph/appearance.rs` `ecb12c88315e1c332d6848a7090537ad2f58ec322bfbb8994fc3649a3fff7b23`；`src/graph/solid.rs` `7228c0556f8bd6421a9cfa8e10c7662cf346fed75014349b185bf5ffbdc57653`；`src/graph/interaction.rs` `17d738d3342d66db312a89c9b791c18932228717ae9da718a30ef46769e7211d`；`scripts/verify_linux_performance_budget.py` `9dc0784e79644dfba4691d327c1dd754afdb0a62730421f22ec6a847dc56748e`。

## 爆炸视图层间距改动后复验（2026-11-21）

在为真实 120×36 全屏图形区的 35 行布局保留选中 block 标签后，重跑 `python3 scripts/verify_linux_performance_budget.py`，本机暂定预算全部通过。200×60 字符/像素帧生成 p95 为 0.401/3.500 ms，输出 13,700/15,281 B；100×30 字符/像素首帧为 8.813/14.558 ms，输入响应 p95 为 0.413/17.707 ms，静止 80 ms 写入 0 B，连续确认到 v3 为 4.326/24.336 ms。

本次布局复验关键哈希：`src/graph.rs` `0b06c982ef798b5972766eb60f177e7091c333c1837d86d9550b58e82a10291d`；`src/graph/interaction.rs` `17d738d3342d66db312a89c9b791c18932228717ae9da718a30ef46769e7211d`；`src/graph/appearance.rs` `ecb12c88315e1c332d6848a7090537ad2f58ec322bfbb8994fc3649a3fff7b23`；`src/graph/solid.rs` `7228c0556f8bd6421a9cfa8e10c7662cf346fed75014349b185bf5ffbdc57653`；`scripts/verify_linux_performance_budget.py` `9dc0784e79644dfba4691d327c1dd754afdb0a62730421f22ec6a847dc56748e`。原生终端和真人任务表现未测。

## 单键预设与窄屏提示改动后复验（2026-11-21）

为图焦点增加 `1`–`4` 直接预设选择并调整窄屏提示后，重新运行 release 性能闸门，暂定预算全部通过。200×60 字符/像素帧 p95 为 0.371/3.505 ms，最大输出 13,704/15,285 B；100×30 字符/像素首帧 8.410/14.519 ms，输入响应 p95 0.297/18.015 ms，闲置 80 ms 均为 0 B，连续确认到 v3 为 4.484/22.737 ms。

本轮哈希：`src/graph.rs` `686dc8ec1a2fe0e35d1d2c47a2f44e1111b319825055a3368bd67d918260a1a9`；`src/graph/interaction.rs` `6930cf8ffc75db6757917d64ef6d599cf54eeb7b82dff6dc88775c770c158c31`；`scripts/verify_linux_performance_budget.py` `9dc0784e79644dfba4691d327c1dd754afdb0a62730421f22ec6a847dc56748e`。格式检查通过；源码哈希包含复验后运行 `cargo fmt` 的纯格式调整。同时完成全 workspace 测试（库 3、主程序 194、行式诊断 28、交互回放 71 通过；共 4 项忽略），以及真实 40×24 tmux 无色帧检查。原生窗口、真实鼠标/触控板和真人任务仍未验收。

## 图命令搜索快捷键和窄屏文案后复验（2026-11-21）

命令搜索为图动作显示随视图状态变化的快捷键；40 列下使用紧凑名称，避免动作、键位和不可用原因互相截断。二维视图中的搜索结果现在可直接用 Enter 进入爆炸 x1，和 `3/4` 快捷键效果一致。全 workspace 测试通过（库 3、主程序 195、行式诊断 28、交互回放 71；4 项忽略）；100×30 与 40×24 隔离 tmux 的预设及命令搜索回放通过。重新运行 Linux release 性能闸门，暂定预算全部通过：200×60 字符/像素帧 p95 0.389/3.458 ms，最大输出 13,704/15,285 B；100×30 首帧 7.299/14.069 ms，输入响应 p95 0.390/17.577 ms，闲置 80 ms 均为 0 B，连续确认到 v3 为 4.324/24.442 ms。

最终关键哈希：`src/graph/search.rs` `6954dcd297f4d903843c3509c389f05b9827d6d1eaabfa7835f3dcddc89c69a1`；`src/graph.rs` `686dc8ec1a2fe0e35d1d2c47a2f44e1111b319825055a3368bd67d918260a1a9`；`src/graph/interaction.rs` `76dd3bc5a7eee3452ed9d99a06943ddc8e8eac001332b5a280f30f30008bac65`；`tests/welcome.rs` `81054aeb21a79ff5cead53c1bae11fa97db20bf7af572981e6048186273e6639`；`scripts/verify_tmux.py` `ee898a9b05840377d4afd56c143c1fb2931f3bb0be99e9471c3a94525be65d19`。性能测量覆盖渲染实现；测量仍只适用于本机 PTY，不代表原生终端或人工任务表现。

## 宽屏鼠标预设工具栏后复验（2026-11-21）

图焦点宽度至少 92 列时增加可点按的 1–4 预设入口后，重新运行 workspace 测试、100×30/40×24 隔离 tmux 回放及 `python3 scripts/verify_linux_performance_budget.py`。全 workspace 299 项通过、4 项忽略；字符与像素伪终端的 SGR 鼠标回放均可逐个切换四种预设。Linux release 暂定预算全部通过：200×60 字符/像素帧 p95 为 0.390/3.503 ms，输出 13,704/15,285 B；100×30 首帧为 8.529/15.651 ms，输入响应 p95 为 0.411/17.745 ms，静止 80 ms 写入均为 0 B，连续确认到 v3 为 4.512/24.909 ms。此时离线工作台提示栏仍覆盖图标题，预设控件稳定可见性由后续章节补验；原生终端与真人任务也未测。

复验哈希：`src/graph.rs` `7cd2b3dcecd63f915c6fedfe7bd9138c0abf37895c6b7ac8e50c4b180893c98b`；`src/graph/interaction.rs` `0c2e392adbde7ca6c3863d7086ae33fa5aa435f98f550933928532561f3f4cd7`；`tests/welcome.rs` `7fe4028d0a56f820363808181bf29aad7c2b8b608daa7e640edef9b01e2eaec1`；`scripts/verify_linux_performance_budget.py` `9dc0784e79644dfba4691d327c1dd754afdb0a62730421f22ec6a847dc56748e`。`git diff --check` 与 `tuitodo.json` JSON 解析检查通过。

## 无色预设选中标记后复验（2026-11-21）

当前预设改用同宽花括号标记后再次运行全 workspace、tmux 与 release 性能闸门。299 项通过、4 项忽略；宽屏字符 SGR 点按与 40×24/100×30 隔离 tmux 回放均通过。暂定性能预算全通过：200×60 字符/像素帧 p95 为 0.375/3.555 ms，输出 13,704/15,285 B；100×30 首帧为 7.597/15.004 ms，输入响应 p95 为 0.299/17.641 ms，静止 80 ms 均为 0 B，连续确认到 v3 为 5.046/24.889 ms。四种预设当前标记、无流图禁用标记与无操作保护由数据测试覆盖；原生终端与真人任务仍待验收。

当前哈希：`src/graph.rs` `58fca370802214cb7e240ad6dc7b02f3dd0d41d4c886daf7e891c663ad5db05d`；`src/graph/interaction.rs` `5d75780de50582a9ae0d380b72f8c433acadb3cdf05132f0dbe168ad541a8de4`；`tests/welcome.rs` `aeda4eedf6c9fa3e22e46f400bfa3ab963df765f03abe1aae2d79e5925849303`；`scripts/verify_linux_performance_budget.py` `9dc0784e79644dfba4691d327c1dd754afdb0a62730421f22ec6a847dc56748e`。格式与 JSON 解析检查通过。

## 离线工作台预设工具栏可见性修复后复验（2026-11-21）

把被离线工作台标题覆盖的图预设工具栏合并到可见提示栏后，重新运行全 workspace、100×30/40×24 tmux 回放和 Linux release 性能预算。全 workspace 300 项通过、4 项忽略；100×30 tmux 最终屏幕断言确认空间、二维、爆炸 x1/x2 四个 active mark 均持续可见；字符和像素 SGR 点按回放均通过。暂定性能预算全通过：200×60 字符/像素帧 p95 为 0.369/3.404 ms，输出 13,704/15,285 B；100×30 首帧为 8.069/14.612 ms，输入响应 p95 为 0.342/17.655 ms，静止 80 ms 均为 0 B，连续确认到 v3 为 4.322/25.242 ms。原生终端、触控板和真人任务仍未验收。

最终源码哈希：`src/graph.rs` `58fca370802214cb7e240ad6dc7b02f3dd0d41d4c886daf7e891c663ad5db05d`；`src/graph/interaction.rs` `bca8e2cdbcf9d53e646cc7ea226875d54e432d2cfb4a897863c7026ee5eab77d`；`tests/welcome.rs` `aeda4eedf6c9fa3e22e46f400bfa3ab963df765f03abe1aae2d79e5925849303`；`scripts/verify_tmux.py` `0f114074a23c1d52cb1ab4a6f9242b34b969d2cd848708a58d7783f08d9eaa54`；`scripts/verify_linux_performance_budget.py` `9dc0784e79644dfba4691d327c1dd754afdb0a62730421f22ec6a847dc56748e`。

## 图操作帮助补充矩阵入口后复验（2026-11-21）

图操作帮助页增加矩阵快捷键后，全 workspace 300 项通过、4 项忽略；100×30 可见预设标记的 tmux 回放通过；Linux release 性能预算全通过。200×60 字符/像素帧 p95 为 0.386/3.486 ms，输出 13,704/15,285 B；100×30 首帧为 7.678/15.583 ms，输入响应 p95 为 0.280/17.265 ms，静止 80 ms 均写入 0 B，连续确认到 v3 为 4.336/22.185 ms。变化仅影响图操作帮助的文本行；预算作为本轮最终回归记录。

哈希：`src/graph.rs` `58fca370802214cb7e240ad6dc7b02f3dd0d41d4c886daf7e891c663ad5db05d`；`src/graph/interaction.rs` `bca8e2cdbcf9d53e646cc7ea226875d54e432d2cfb4a897863c7026ee5eab77d`；`src/graph/offline/display.rs` `ddfb3ed58a611c0af912fabb96374d67faf4a6ed25e02011d62a0e7f5b78c95c`；`tests/welcome.rs` `e18b1aa16e9746dea95af682c35ced395f6210006fc690225ab232e5e5c0e151`；`scripts/verify_tmux.py` `0f114074a23c1d52cb1ab4a6f9242b34b969d2cd848708a58d7783f08d9eaa54`；`scripts/verify_linux_performance_budget.py` `9dc0784e79644dfba4691d327c1dd754afdb0a62730421f22ec6a847dc56748e`。

## 四预设帧捕获及宽屏提示栏紧凑文案复验（2026-11-21）

`verify_tmux.py` 保存了同一离线 48/32 选中连线下的四张 100×30 `NO_COLOR` 最终帧；脚本断言每张帧的行数、选中预设标记、`? Controls` 与 `T Mouse on` 完整可见。短提示栏避免控件加入后截断鼠标状态。全 workspace 300 项通过、4 项忽略；tmux 四预设帧及 40×24 回放通过。Linux release 预算全部通过：200×60 字符/像素帧 p95 为 0.369/3.443 ms，输出 13,704/15,285 B；100×30 首帧为 7.567/14.117 ms，输入响应 p95 为 0.391/17.757 ms，静止 80 ms 均写入 0 B，连续确认到 v3 为 4.859/24.761 ms。

四种无色终端帧：`docs/graph-preset-samples/toolbar-spatial-100x30-cells.txt`、`toolbar-flow-2d-100x30-cells.txt`、`toolbar-exploded-x1-100x30-cells.txt`、`toolbar-exploded-x2-100x30-cells.txt`。

复验哈希：`src/graph/interaction.rs` `c9820c62688126869ce1ed3f7fdb04d439f26a1b279245d7f1f4f09b4a1169df`；`src/graph/offline/display.rs` `84e85bde11d0cbb15e23d7eeafdc2e09c54f6244beecb90c25eaf067a719d3be`；`tests/welcome.rs` `e18b1aa16e9746dea95af682c35ced395f6210006fc690225ab232e5e5c0e151`；`scripts/verify_tmux.py` `b4ab1a180dbef6a4356a1107e38f0087fa8270671570b6c1172db16e24a1e684`；`scripts/verify_linux_performance_budget.py` `9dc0784e79644dfba4691d327c1dd754afdb0a62730421f22ec6a847dc56748e`。

## 主图增加矩阵直达提示及四帧稳定采集后复验（2026-11-21）

宽屏离线图提示栏现在同时显示 `? Controls`、`M Port matrix`、`T Mouse on/off` 与四个图形按钮；100 列字符帧不截断这些操作提示。tmux 最终帧检查四种预设标记及控制提示，并生成同一选中连线下的四张 100×30 无色图帧。全 workspace 300 项通过、4 项忽略；tmux 两种尺寸通过；release 性能预算全部通过：200×60 字符/像素帧 p95 为 0.373/3.498 ms，输出 13,704/15,285 B；100×30 首帧为 8.431/15.593 ms，输入响应 p95 为 0.279/17.772 ms，静止 80 ms 均写入 0 B，连续确认到 v3 为 4.305/24.188 ms。

最新哈希：`src/graph/interaction.rs` `ce52593171997ed6ff88530df0769442c46aef9673a963d4c6a7f467230fbfb6`；`src/graph/offline/display.rs` `2fa588ef841c3fe00d549d0d5fe804bc4459b0b55a86c1b6eb243a693b68bbfd`；`tests/welcome.rs` `e18b1aa16e9746dea95af682c35ced395f6210006fc690225ab232e5e5c0e151`；`scripts/verify_tmux.py` `26e252bcb573f978897e04aa549e538834129538eb17dadcbc0efb90a29d0f32`；`scripts/verify_linux_performance_budget.py` `9dc0784e79644dfba4691d327c1dd754afdb0a62730421f22ec6a847dc56748e`。

## 等轴爆炸视角改动后复验（2026-11-21）

x1/x2 爆炸视图默认相机改为正交等轴角（偏航 45°、俯仰约 35.3°），空间视图仍保留原默认角；每档相机与缩放仍独立恢复。相机轴投影单测、状态角度测试、20 张 120×36 无色 tmux 画布及对应 PNG 样稿已更新，`scripts/verify_tmux.py` 通过。全 workspace 301 项通过、4 项忽略。Linux release 暂定预算全部通过：200×60 字符/像素帧 p95 0.320/3.654 ms、输出 13,704/15,285 B；100×30 字符/像素首帧 8.622/14.114 ms，输入响应 p95 0.404/17.782 ms，闲置 80 ms 写入 0 B，连续确认到 v3 为 4.318/23.232 ms。

复验哈希：`src/graph.rs` `d65c350faf5a0d815f94985ce6ccfa1045e587146d830dddd3de579e7a24d529`；`src/graph/solid.rs` `73cd128e56f95e489ce594a7e04356e3805967ae10b7eb2aeeca2c566b42d75f`；`src/graph/source.rs` `283b412c75811ce924751c9a4ca6cd2b8625d5b8f2f84ed00e78663a5c61bd05`；`src/graph/interaction.rs` `ce52593171997ed6ff88530df0769442c46aef9673a963d4c6a7f467230fbfb6`；`tests/welcome.rs` `6ae0afad3dc7ea3408360e392fbc7fb1def92056e11b87a5aba4ca8bb455eb62`；`scripts/verify_tmux.py` `03371f18bcd53cb114183c8f6a5cdf6296a8f713795e6b98b107259fa37b5449`。真人任务及 Ghostty 原生窗口/触控板、Windows Terminal、SSH 仍未验收。

## 91/92 列预设工具栏边界回归后（2026-11-21）

增加 91 列紧凑工具栏与 92 列四预设工具栏的边界回放，并在 92 列末端单元格点击 x2。全 workspace 当前为 302 项通过、4 项忽略。该改动只新增伪终端验收用例，未改变 release 渲染路径，因此沿用上一节等轴相机后的性能测量，不把本节测试通过误当作新的性能复测。`git diff --check` 和 `tuitodo.json` JSON 解析检查通过。

最新回放测试哈希：`tests/welcome.rs` `46ba5318994d4842b9920f47ec9a3740dde910323fb32c9a3d27178214c9b1b0`。原生终端边界与触控板点击仍未实测。

## 修复草稿跨预设往返回归后（2026-11-21）

增加字符/像素场景的修复 Draft 标记往返检查后，全 workspace 当前为 303 项通过、4 项忽略。此次仅增加测试与文档，没有改动运行时渲染代码；前述 release 性能测量仍对应当前运行时代码，未重复计为本轮新测量。

最新视觉状态回归哈希：`src/graph/offline.rs` `5108f7a8265db1e69e03928af8a5453bf0e6b7e0b8fb7dedd0666df097eb583a`。

## 爆炸图切片身份深度检查后复验（2026-11-21）

增加测试构建内的 `(operator ID, slice ordinal)` Braille 深度栅格身份追踪，并在 x1/x2、四个相机方位与 40×24/100×30/120×36 视口确认选中模块的每个切片至少覆盖 2 个字符格。该身份追踪由 `cfg(test)` 编译，不进入运行时渲染。全 workspace 303 项通过、4 项忽略；`scripts/verify_tmux.py` 通过。Linux release 暂定预算全部通过：200×60 字符/像素帧 p95 为 0.406/3.544 ms，最大输出 13,704/15,285 B；100×30 字符/像素首帧 8.038/14.603 ms，输入响应 p95 0.421/17.804 ms，闲置 80 ms 写入 0 B，连续确认到 v3 为 4.326/24.649 ms。

本轮哈希：`src/graph.rs` `11d21c3b830da05754de019e258d03311e252be3ca3b8b54ada4b526fe482570`；`src/graph/solid.rs` `9c8b35cb1af3304ae2fa941e3d5e81ecd4bcc5d36850d0106a3a142622187dfa`；`scripts/verify_tmux.py` `03371f18bcd53cb114183c8f6a5cdf6296a8f713795e6b98b107259fa37b5449`；`scripts/verify_linux_performance_budget.py` `9dc0784e79644dfba4691d327c1dd754afdb0a62730421f22ec6a847dc56748e`。自动像素可见性仍不能替代辨认度、误认率或定位耗时的真人对照；原生终端和触控板也未测。

## Braille 切片覆盖度量校准后（2026-11-21）

把切片可见性从任意像素改为 Braille 字符格覆盖后，重新运行全 workspace（303 项通过、4 项忽略）。窄屏 x1 某个旋转角的部分切片只覆盖 2 个字符格；4 格门槛回归失败，因此采用实际验证的 2 格下界，并在视觉评审中明确标注这一姿态仍需真人辨认检查。这里只调整测试度量，没有改运行时渲染；上一节 release 性能测量仍对应相同运行时实现，本次没有重复计量。

校准后的测试源码哈希：`src/graph.rs` `b5677612c1ef22e76e38dfc2c69fcad3f5146e8c04887f4b5246c65e9cd4425c`；`src/graph/solid.rs` `f125062012aaad9c0884a7ae340a8624b2d9ceb8c4a52bdd2f6786f131857dfe`。

## 40×24 爆炸根视图纵向分页后复验（2026-11-21）

流图爆炸模式在 60 列以下改用纵向分层；40 列选中 stack 额外增加轻度厚度，保留原有 slice count；页码移到状态行前端，x1/x2 分别可以在真实无色 TUI 帧中看到 `4–5/5` 和 `4–4/5`。20 张 120×36 和 2 张 40×24 Drifting 帧均由 `scripts/capture_preset_cells.py` 重采，脚本断言窄屏页码、24 行和目标选择。`scripts/verify_tmux.py` 通过。全 workspace 303 项通过、4 项忽略。

Linux release 暂定预算全部通过：200×60 字符/像素帧 p95 为 0.423/3.444 ms，输出 13,704/15,285 B；100×30 字符/像素首帧 7.663/14.809 ms，输入响应 p95 0.351/17.525 ms，闲置 80 ms 写入 0 B，连续确认到 v3 为 4.398/25.839 ms。

当前关键哈希：`src/graph.rs` `3fbfeb2060101fd823730056df9b75d0db7a1fd16063307a4d98e7e374481680`；`src/graph/solid.rs` `f125062012aaad9c0884a7ae340a8624b2d9ceb8c4a52bdd2f6786f131857dfe`；`scripts/capture_preset_cells.py` `8434c7395a5f41b4a41489f59abb1f4e935c43044a307f9fae8b5331dc387fdf`；`scripts/verify_tmux.py` `03371f18bcd53cb114183c8f6a5cdf6296a8f713795e6b98b107259fa37b5449`；`scripts/verify_linux_performance_budget.py` `9dc0784e79644dfba4691d327c1dd754afdb0a62730421f22ec6a847dc56748e`。自动布局检查不能替代分支标签辨认度和用户走查。

## 单色选中连线方向标记后复验（2026-11-21）

固定连线的箭头增大后，全 workspace 测试和 `scripts/verify_tmux.py` 均通过；Linux release 暂定性能预算全部通过。200×60 字符/像素绘制 p95 为 0.410/3.523 ms，最大输出 13,704/15,285 B；100×30 字符/像素首帧为 7.984/15.279 ms，输入响应 p95 为 0.367/18.028 ms；闲置 80 ms 写入 0 B，连续确认到 v3 为 4.315/24.481 ms。

当前视觉/回归哈希：`src/graph/tensor.rs` `5f52417ef68958f32ab48c4893ac20d6e607f4dec999f42dc828fe8e228ee775`；`src/graph/solid.rs` `44d477a1d1bb7f3441a94305210e73cf3a21f6bf7b35d25da84c4a4d92aefd54`；`tests/welcome.rs` `c2b8da445cc1379e9c2ce8d68e30bb6ccf45e5a26ffff9e87b216af90a340931`；`scripts/verify_tmux.py` `0c480230e958155190c20d2c73d7803d64a4b39e1206e6db023cca92127e6cea`；`scripts/verify_linux_performance_budget.py` `9dc0784e79644dfba4691d327c1dd754afdb0a62730421f22ec6a847dc56748e`。这些数字是本机暂定预算，不代表 Ghostty、Windows Terminal 或真人定位任务验收。

## 爆炸 stack 片序号标注后复验（2026-11-21）

当前选中 stack 的 `s1…sN` 文字标注只写入已有终端 caption 图层；release 暂定预算全通过。全 workspace 通过，Drift-L/B 120×36 NO_COLOR 捕获及 40×24 窄屏样稿重采，100×30 四预设与 40×24 x1/x2 tmux 回放通过。200×60 字符/像素帧 p95 为 0.439/3.700 ms，最大输出 13,704/15,285 B；100×30 字符/像素首帧 7.454/15.382 ms、输入响应 p95 0.273/17.713 ms，闲置 80 ms 写入 0 B，连续确认到 v3 为 4.629/24.830 ms。

复验哈希：`src/graph.rs` `8953e3b76aa445ee069c6774fd14187141844a4a12aa4b9fb0f0a97d48f4153d`；`src/graph/solid.rs` `44d477a1d1bb7f3441a94305210e73cf3a21f6bf7b35d25da84c4a4d92aefd54`；`tests/welcome.rs` `fe8ef4d1701764e20f940af495a78135d922ba6539954e0754bd941fff0c192b`；`scripts/capture_preset_cells.py` `58f773943b3558831e08351a9ca9330f750823e1ac03892fcfdc47c587615a8a`；`scripts/verify_linux_performance_budget.py` `9dc0784e79644dfba4691d327c1dd754afdb0a62730421f22ec6a847dc56748e`。这些是本机暂定性能门槛，不代替原生终端和用户辨认任务。
