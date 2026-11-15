# 开发与协作指南

先按 [README](../README.md) 安装，再阅读 [架构说明](ARCHITECTURE.md)。开发入口分成摄像头主线、新感知协议和早期 MVP 三部分；修改前先确认自己的变更属于哪一部分，避免在旧入口修了逻辑却以为新页面已更新。

## 1. 第一次接手的顺序

1. 运行 README 的合成演示和下面的最小回归，确认 Python 环境及规则可用。这一步不需要视频、摄像头或模型权重。
2. 读 `PerceptionChain` 的输出合同与对应测试，分清当前动作、过去窗口类别、过程证据、事件和通知。
3. 运行资源检查，根据自己要做的任务准备资源。先实现接口或单元逻辑的同学无需下载全部模型。
4. 做视觉效果验证时再启动摄像头服务或使用授权录像，保存基线配置、源时间和结果。历史实验脚本的默认路径不能当成新机器的运行条件。

常用位置：

| 任务 | 首先阅读 |
| --- | --- |
| 网页显示、录制、输入输出 | [assets/webcam](../assets/webcam)、[webcam_pose_demo.py](../scripts/webcam_pose_demo.py) |
| 人体身份、机位、观测覆盖 | [webcam_target.py](../src/fallpre/weekly/webcam_target.py)、[webcam_fall_target.py](../src/fallpre/weekly/webcam_fall_target.py)、[webcam_quality.py](../src/fallpre/weekly/webcam_quality.py) |
| ST 输入与异步历史 | [stgcn_temporal.py](../src/fallpre/weekly/stgcn_temporal.py)、[webcam_temporal.py](../src/fallpre/weekly/webcam_temporal.py) |
| 动作与跌倒融合 | [stgcn_fusion.py](../src/fallpre/weekly/stgcn_fusion.py)、[stgcn_fall_evidence.py](../src/fallpre/weekly/stgcn_fall_evidence.py)、[fall_event_fusion.py](../src/fallpre/weekly/fall_event_fusion.py) |
| 新感知协议与回放 | [perception_chain.py](../src/fallpre/weekly/perception_chain.py)、[replay_perception_chain.py](../scripts/replay_perception_chain.py) |
| 后续智能体与持久执行 | [agent.py](../src/fallpre/agent.py)、[tools.py](../src/fallpre/tools.py)、[service.py](../src/fallpre/service.py)、[execution.py](../src/fallpre/execution.py) |

## 2. 测试分层

以下命令从仓库根目录运行，并使用已安装项目的虚拟环境。激活 `.venv` 后可用 `python`；未激活时改为 `.venv/bin/python`。

### 最小可分享回归：无需本地实验数据

推荐一次运行完整的可分享测试组：`python scripts/check_portable.py`。它包含合同回归、时间窗口、二分类证据、观测语义、持久事件和单 Agent 闭环，以及资源管理与源码交接测试；加 `--with-ui` 可运行无需浏览器的 Node 录制测试。

```bash
python -m pytest -q \
  tests/test_end_to_end.py \
  tests/test_policy_tools.py \
  tests/test_cli_configuration.py \
  tests/test_perception_chain.py \
  tests/test_stgcn_fusion.py \
  tests/test_stgcn_fall_evidence.py \
  tests/test_fall_event_fusion.py \
  tests/test_webcam_fall_handoff.py
```

这些测试使用合成观测、假时序分数和临时目录，覆盖旧 MVP 的模拟执行，以及新协议的当前动作、观测未知、多人阻断、未来/失效时序、事件加速和通知去重。它们不加载私有 V7 权重，不运行真实 YOLO 或 ST 模型，也不读取个人录像。不要把通过这些测试表述为“真实视频已改善”。

### 时间窗口与异步服务

```bash
python -m pytest -q tests/test_stgcn_temporal.py tests/test_webcam_temporal.py
```

默认测试以合成骨架和替身运行。`test_stgcn_temporal.py` 有两类可选检查：原开发机的 `linear_probe.py` 不存在时跳过历史实现比对；实际 ST 权重检查仅在设置 `FALLPRE_TEST_STGCN_MODEL=1` 时运行。启用它之前先准备完整的 ST 权重、配置、标签和 Python 依赖；测试会核对权重和预处理，不会自动完成安装。

### 模型和完整服务测试

资源状态由工具检查：

```bash
python scripts/project_resources.py doctor --profile synthetic
python scripts/project_resources.py doctor --profile camera-rgb
python scripts/project_resources.py doctor --profile temporal
python scripts/project_resources.py doctor --profile perception-chain
```

`doctor` 用于确认所选层需要的资源；实际安装和导入步骤以 README 为准。不要在测试开始后才猜某个历史目录为什么不存在。

完整 `python -m pytest` 还包含历史模型回归和服务测试。例如 `test_webcam_service.py`、`test_webcam_actions.py`、`test_webcam_target.py`、`test_webcam_diagnostics.py`、`test_webcam_risk_integration.py` 及相关服务测试需要本地 V7 动作模型。部分测试直接使用原实验路径或共享其他测试中的 fixture。GitHub 最小代码包不含这些训练产物，不能承诺在空白克隆上直接跑完整套件。

文件里出现 `runs/` 不一定代表依赖个人数据：`test_fall_candidate_review.py` 在 `tmp_path` 下构造临时目录，`test_webcam_rgb_model.py` 构造假的模型文件。判断可移植性要检查实际读取的位置，不能仅凭字符串批量删除测试。

### 真实录像的事件回归

只有视觉推理、融合或质量逻辑变更需要这一层。使用已有授权视频和人工复核标签；个人视频及其帧图不提交到 Git。先固定视频清单、模型哈希、参数、事件起点定义与排除原因，再比较基线和候选。

每次报告至少给出：

- 真实事件数、漏检数及匹配方法；不可观测时发生的事件也单独报告，不能悄悄从分母删去。
- 误报警数和有效监测时长；同时给出总时长与不可观测时长，避免靠弃权降低表面误报率。
- 以相同事件起点和源时间计算的报警延迟；不要用脚本执行时间或文件处理速度替代。
- 重复报警、错误恢复、旧事件拖尾和事件键冲突。
- 按人物、机位和视频分组的结果；同一视频的相邻窗口不能一边训练一边充当独立测试。

基础动作混淆用于解释事件错误。一个动作标签的变化若没有改善跌倒事件，就不作为扩大融合权限的充分理由。历史小样本回归只能说明该批素材上的表现；上线前仍需独立视频和长时间正常活动评估。

## 3. 两人如何分工

适合的初始划分是：一人负责视觉、骨架窗口和融合证据；另一人负责工具协议、回放/展示和后续事件消费。双方共同审查事件状态与数据合同的改动。也可以按功能交换，但同一轮尽量避免同时修改 `webcam_pose_demo.py` 和 `perception_chain.py` 的相同接口。

合并前约定这些边界：

1. **输入负责人**保证源时间、目标阶段、骨架置信度和质量字段的含义；缺失字段不能被填成“正常”来通过下游判断。
2. **时序负责人**保证训练和推理预处理一致，输出带窗口、身份、代次、模型版本和有效性；不把旧结果重新盖上当前时间戳。
3. **事件负责人**保留基线与候选、明确报警权限、处理去重和恢复；不得把模型类别分数直接当成事件命令。
4. **界面与智能体负责人**分别展示/消费动作、质量、过程和事件；只消费符合合同的新事件，并保留证据关联。

开始智能体适配时先使用 `notification` 的幂等键做模拟消费。新链路的去重目前仅在内存中，需补持久化和会话命名后再讨论真实外部执行。旧 MVP 的工具可以复用设计，但必须显式转换合同，不能假定已经接通。

## 4. 改动与提交约定

- 一个分支解决一个可以解释的问题。PR 写清楚触发场景、原行为、新行为、验证命令及尚未验证的边界。
- 改动作补充时必须检查当前动作原本已知、时序冲突、缺骨架、换目标、多人、机位变动和过期结果；改事件时另查重复确认、目标消失、恢复后再发和跨流事件键。
- 不为提覆盖率屏蔽固定房间区域或降低多人保护；让无法判断的原因可见，再定位观测损失。
- 不在重构时删除 `action_v2` 等旧文件，除非已确认依赖替代、行为回归及历史复现影响。先用代码搜索查导入关系。
- 不把下载脚本改为默默复制全部录像或模型。资源使用 [runtime-resources.json](../configs/runtime-resources.json) 和外部路径，校验后按需获取；保留现有软链接及其目标。
- `runs/`、模型权重、数据集、录制包、数据库、个人标签和含隐私的报告留在本地。共享结果优先提供不含个人画面和绝对路径的汇总。
- GitHub 版本以 [发布准备说明](GITHUB_PREPARATION.md) 为准。公开权重有独立的来源和许可，安装来源不等于允许把权重复制进仓库。

本机页面验证遵守项目的 Safari 偏好。没有可用的 Safari 自动化时，做不依赖浏览器的检查并注明限制；不要把其他浏览器的结果写成 Safari 验证。
