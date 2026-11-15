# FallPreAgent

面向养老场景的跌倒事件识别开发项目：用 YOLO 提取人体和骨架，用连续规则与 ST-GCN++ 提供时序证据，再由事件管理处理确认、去重和恢复。感知事件也可交给单 Agent 执行模拟提醒与报警。

**当前是研究与开发版本。** 已有摄像头/视频演示、受限时序融合、持久事件查询和单 Agent 模拟闭环。最新末段 ST＋几何头用于影子对照，真实部署的长期误报和迁移表现仍待评估。

## 从哪里开始

| 想做什么 | 入口 | 是否需要模型/录像 |
| --- | --- | --- |
| 先跑通环境、了解工具调用 | `fallpre demo`、合成测试 | 不需要 |
| 看当前视觉系统识别摄像头/本地视频 | `scripts/webcam_pose_demo.py` | 需要视觉模型；视频可自行选择 |
| 开发新的动作/质量/过程/事件输出协议 | `PerceptionChain`、`scripts/replay_perception_chain.py` | 合成测试不需要；实际推理按需准备资源 |
| 接手智能体开发 | [单 Agent 交接](docs/care-agent-20261110.md)、`scripts/replay_care_agent.py --demo` | 需要已安装的 Python 依赖；无需模型或录像 |

代码与数据流见 [架构说明](docs/ARCHITECTURE.md)，协作与测试见 [开发指南](docs/DEVELOPMENT.md)。先读这两份，不必先翻历史实验脚本。

## 1. 安装与第一次运行

推荐 Python **3.11**，项目支持范围是 3.11–3.12；需要 Git 和 `uv`。当前开发及实机检查在 macOS Apple Silicon 上进行，尚未完成 Windows/Linux 全链路验证。以下命令均在仓库根目录执行。

```bash
# pyproject 使用固定版本的本地 Ultralytics 源码，因此先取源码再装依赖。
python3 scripts/fetch_sources.py
uv sync --python 3.11 --locked

# 合成观测 + 规则 + 模拟工具调用；不访问摄像头、不发外部报警。
.venv/bin/fallpre demo

# 单 Agent 合成闭环，写出一次模拟提醒、一次模拟报警及执行回执。
.venv/bin/python scripts/replay_care_agent.py --demo --output runs/care-agent-demo

# 不需要个人数据或模型的合同、时序、事件和资源回归。
.venv/bin/python scripts/check_portable.py
```

`demo` 属于早期 MVP，用于验证工具与执行链路，**不代表运行了 YOLO 或 ST-GCN++**。结果写入本地 `runs/`。可用 `.venv/bin/fallpre tools-schema --output runs/tools.schema.json` 查看旧 MVP 工具定义。

`uv.lock` 固定基础依赖；当前默认下载索引配置在 `pyproject.toml`。网络失败时先检查 GitHub 与该索引是否可达，不要通过随意升级依赖解决。`fetch_sources.py` 遇到已有源码版本不符或本地修改会停止，保留原目录。

需要录制模块回归且本机有 Node.js 时：

```bash
.venv/bin/python scripts/check_portable.py --with-ui
```

这组 JavaScript 测试不启动浏览器。完整 `pytest` 含历史模型回归，另需本地产物；详见开发指南。

## 2. 准备视觉和时序资源

仓库只分享源码、配置、资源清单和测试，**不包含个人视频、数据集和模型文件**。资源名称、固定大小、SHA256、来源及规范路径记录在 [runtime-resources.json](configs/runtime-resources.json)。

```bash
# 只检查，不下载；新克隆缺模型时显示具体缺项并返回非零状态。
python3 scripts/project_resources.py doctor --profile camera-rgb

# 显式下载当前视觉入口的公开权重，不下载视频。
python3 scripts/project_resources.py fetch-public --profile camera-rgb

# V7 是团队训练产物，请向维护者单独获取校验一致的文件。
python3 scripts/project_resources.py import action-v7 /path/to/model.json
python3 scripts/project_resources.py doctor --profile camera-rgb
```

`/path/to/...` 是需要替换的文件位置。公开下载命令会提示团队产物仍缺失，不能代替最终 `doctor` 检查。已有校验正确的资源会复用；不会覆盖现有文件或写穿软链接。原开发机旧 `runs/` 路径继续只读兼容，不必移动或重复复制。

选择资源层级：

| profile | 用途 |
| --- | --- |
| `synthetic` | 无模型接口开发 |
| `camera` | YOLO 骨架 + V7，供显式 `--fall-mode pose` 使用 |
| `camera-rgb` | 当前视觉入口，包含 RGB 倒卧候选，暂不加载 ST |
| `temporal` | 视觉入口 + ST-GCN++ 60 类窗口推理 |
| `perception-chain` | 完整资源清单，再包含实验 C/D 二分类头 |

启用 ST-GCN++ 前，额外准备固定的 MMAction2 源码、公开权重及隔离依赖：

```bash
python3 scripts/project_resources.py fetch-sources
python3 scripts/project_resources.py fetch-public --profile temporal
uv pip install --python .venv/bin/python -r requirements-weekly.txt
uv pip install --python .venv/bin/python --target .runtime/action-deps \
  --no-deps -r requirements-weekly-action.txt
python3 scripts/project_resources.py doctor --profile temporal
```

附加依赖清单独立于基础 `uv.lock`。ST 依赖优先使用 `.runtime/action-deps`，也可用 `FALLPRE_ACTION_DEPS` 指向已准备的目录；兼容原开发机旧隔离目录。`doctor` 校验资源，不验证 Python 推理环境；实际模型检查可运行：

```bash
FALLPRE_TEST_STGCN_MODEL=1 .venv/bin/python -m pytest -q \
  tests/test_stgcn_temporal.py -k actual
```

## 3. 摄像头与本地视频演示

只准备 `camera-rgb` 资源时，先运行视觉基线：

```bash
.venv/bin/python scripts/webcam_pose_demo.py --port 8780 --no-temporal
```

准备好 `temporal` 资源后，运行带 ST 窗口结果的版本：

```bash
.venv/bin/python scripts/webcam_pose_demo.py --port 8780
```

在 **Safari** 打开 `http://127.0.0.1:8780/`，选择实时摄像头或本地视频，再开始分析。服务只监听本机；端口已被使用时换一个端口，不终止已有实验服务。YOLO 默认 `yolo26n-pose`、CPU；ST-GCN++ 当前也使用 CPU。

三个融合开关职责不同：`--fusion-shadow` 显示动作补充候选；`--fall-temporal-shadow` 输出跌倒时序证据；`--fall-event-fusion` 才启用现有受限事件确认加速，并保留基线对照。它们都要求 ST 已启用，不能与 `--no-temporal` 并用。新机器先复现基线，再按任务开启；实验二分类头不会因此获得报警权限。

## 4. 新感知链路的离线回放

这是接下来开发智能体输入协议的主要入口。已有完整观测 JSONL 时：

```bash
.venv/bin/python scripts/replay_perception_chain.py \
  --input-jsonl /path/to/observations.jsonl \
  --output runs/perception-replay-001 --stream-id camera-a-session-001
```

默认复用输入中的 `result.temporal`，不会加载模型。输入每行的结构为：

```text
{
  "source_time_s": 0.1,
  "result": {
    "width": ..., "height": ..., "persons": [...],
    "quality": {"camera_stable": ..., "reason": "..."},
    "action": {"label": ..., "target_index": ..., "selection": {...}},
    "fall": {...}, "fall_mode": "rgb",
    "pose_fall_observation": {"person": ..., "selection": {"target_epoch": ...}},
    "fall_observation": {...},
    "rgb_raw_candidates": {"persons": [...], "fallen": [...]}, "temporal": {...}
  }
}
```

上面是字段示意，省略号不能作为实际 JSON。完整合成例子见 [test_perception_chain.py](tests/test_perception_chain.py) 与 [test_stgcn_fall_evidence.py](tests/test_stgcn_fall_evidence.py)。时间必须递增；不同视频/会话使用不同 `stream-id`，不要拼接成同一目标历史。网页导出的标量诊断不含完整骨架，不能直接当作此输入。

输出目录必须是新目录，生成 `pipeline.jsonl` 和 `summary.json`，分别保留每时刻的合同输出和计数/通知汇总，不复制视频、不发送外部通知。这些计数不等于有人工标签的准确率。

要重新计算 ST 和实验判别头，先由维护者单独提供 `fall-head-c`、`fall-head-d` 并导入，再加 `--infer-temporal`：

```bash
python3 scripts/project_resources.py import fall-head-c /path/to/head-c.npz
python3 scripts/project_resources.py import fall-head-d /path/to/head-d.npz
python3 scripts/project_resources.py doctor --profile perception-chain
```

重算模式按实测推理耗时延后提供结果，不借用未来帧；它模拟离线结果可用时刻，不能替代实机端到端延迟测量。C/D 头仍是实验诊断，不直接确认跌倒报警。

## 5. 目录与开发边界

```text
src/fallpre/weekly/       当前视觉、动作、时序和事件主线（沿用历史目录名）
src/fallpre/             早期 MVP、工具、SQLite、策略；含新的资源解析模块
scripts/                 运行入口、资源管理、回放、发布检查及历史实验工具
assets/webcam/           当前浏览器页面与录制代码
configs/                 规则/站点样例、资源清单、研究协议和数据元信息
tests/                  合成、模型、历史与界面测试（按开发指南分层运行）
docs/                    架构、协作和发布说明
models/ third_party/     本地模型与上游源码，不提交
runs/ tmp/ output/       本地实验、缓存、导出，不提交
```

`configs/` 中历史数据清单是研究元信息，不是启动依赖。旧版本动作模块仍有实际导入关系；本轮保留它们，没有通过删文件来“整理”主线。原 README 实验时间线已保存在本地 `docs/local-history/README-before-github-20261108.md`，不进入分享包。

当前最重要的开发约束是：观测不足不等于非跌倒；过去 2.5 秒的类别不等于当前姿态；规则与 ST 共享骨架，不能把一致当成独立双重保证。单 Agent 的事件存储按生产会话、视频流和上游事件键持久去重。上游重启或事件计数重置时须使用新的生产会话标识。

## 6. 协作与 GitHub 准备

提交前运行源码边界检查：

```bash
python3 scripts/prepare_github.py
```

它检查 Git 可见文件、常见凭据格式、软链接、二进制和大文件，清单写入本地 `output/github-preparation/manifest.json`；不会执行 `git add`、提交或推送。还可用 `--export output/fallpre-source.zip` 导出同一文件范围。

具体范围和交接步骤见 [GitHub 准备说明](docs/GITHUB_PREPARATION.md)。新增顶层目录要显式更新 `.gitignore` 的发布边界，并复查清单。项目自身许可证尚未选定；第三方代码、模型和训练来源的已知条款见 [THIRD_PARTY.md](THIRD_PARTY.md)，不能默认整个仓库适用 MIT。
