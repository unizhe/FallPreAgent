# FallPreAgent

面向养老场景的跌倒事件识别开发项目：用 YOLO 提取人体和骨架，用连续规则与 ST-GCN++ 提供时序证据，再由事件管理处理确认、去重和恢复。感知事件也可交给单 Agent 执行模拟提醒与报警。

**当前是研究与开发版本。** 主线支持摄像头与本地视频、时序证据、持久 Agent 决策及已分析片段回看。线上模式可显式接入阿里云，离线模式可用合成输入复现决策闭环。

## 从哪里开始

| 想做什么 | 入口 | 是否需要模型/录像 |
| --- | --- | --- |
| 先跑通环境、了解工具调用 | `fallpre demo`、合成测试 | 不需要 |
| 看当前视觉系统识别摄像头/本地视频 | `scripts/webcam_pose_demo.py` | 需要视觉模型；视频可自行选择 |
| 开发新的动作/质量/过程/事件输出协议 | `PerceptionChain`、`scripts/replay_perception_chain.py` | 合成测试不需要；实际推理按需准备资源 |
| 接手智能体开发 | [单 Agent 交接](docs/care-agent-20261110.md)、`scripts/replay_care_agent.py --demo` | 需要已安装的 Python 依赖；无需模型或录像 |
| 验证现在的跌倒决策主线 | [本地持久化决策回放](#5-本地持久化决策回放与真实决策模型验证p1p2) | 合成回放无需模型、录像或云凭据 |

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

# 新的跌倒决策合同、任务去重和 SQLite 闭环；离线合成输入。
.venv/bin/python scripts/replay_decision_agent.py \
  --demo --provider mock --output runs/decision-agent-demo

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

### 视频 Agent 与已分析片段回看

Agent 复用同一网页视频解码和 `/infer`。先按第 2 节准备 ST-GCN++，再下载 `perception-chain` profile 的公开资源，并通过资源工具导入维护者授权提供的 V7、C、D 头文件。以下是 C/D 的导入与最终校验步骤：

```bash
python3 scripts/project_resources.py fetch-public --profile perception-chain
python3 scripts/project_resources.py import fall-head-c /path/to/head-c-st.npz
python3 scripts/project_resources.py import fall-head-d /path/to/head-d-geometry.npz
python3 scripts/project_resources.py doctor --profile perception-chain
```

启动本地服务，显式开启 Agent API。默认 provider 为 `off`：

```bash
.venv/bin/python scripts/webcam_pose_demo.py --port 8780 --fall-mode rgb \
  --agent-enabled --provider off --agent-db runs/decision-stage04-live/decision-agent.sqlite3
```

在 Safari 选本地视频或摄像头，再点 **开始 Agent**。服务端从 YOLO、ST-GCN++ 和 D 头结果生成证据。页面展示跌倒、证据充分性和下一步处理三组模型概率，并标记结果来源窗口。证据不完整、过期或质量不足时，当前概率为空。旧结果按历史展示。

本地视频开始分析后，拖动进度条可回看已完成的帧和该时刻覆盖的历史决策。回看使用本页缓存的观测和结果，不发起新的推理或云请求。点 **回到分析进度** 切回最新画面。已分析片段可在停止后回看，浏览器刷新或开始下一次分析时重建本页回看缓存。

要验证真实 Aliyun 决策，可参考 `.env.example` 配置本机忽略的 `.env`，或通过进程环境提供凭据，然后显式启用网络：

```bash
.venv/bin/python scripts/webcam_pose_demo.py --port 8780 --fall-mode rgb \
  --agent-enabled --provider aliyun --enable-network \
  --agent-http-budget 6 --agent-db runs/decision-aliyun-demo/decision-agent.sqlite3
```

HTTP 请求额度按数据库累计，含重试。默认 6 次，启动参数允许 0–1000 次。要为现有数据库提高已获授权的额度，指定新的 `--agent-http-budget`，同时加上 `--raise-agent-http-budget`，此前用量会保留。服务或页面重启不会重置已用计数。

`/api/v1/agent/runs` 创建新 run，`GET .../{run_id}/latest` 查当前投影，`GET .../{run_id}/decisions?after_id=...&limit=...` 按结果游标查历史，`POST .../{run_id}/stop` 停止，`POST /api/v1/agent/decisions/{decision_id}/feedback` 追加带版本校验的反馈。帧按 run、源时间、JPEG 摘要和推理参数去重。只使用本地授权视频，云端接收通过校验的结构化摘要。

### 跟踪一次 RGB 视频决策

在启动服务的命令中加上 `--agent-trace runs/debug-session/server.jsonl`，即可记录帧去重、观测登记、任务领取、云请求与响应、结果写入和当前值查询。日志路径须为新文件。可在同一服务上使用 Safari，也可用脚本逐帧走实际的 `/infer` 接口。

```bash
.venv/bin/python scripts/debug_video_agent.py /path/to/video.mp4 \
  --port 8780 --out runs/debug-session/video
```

脚本按视频时间取帧，在可决策窗口等待异步结果。打开输出目录的 `result.html` 查看概率，`result.json` 保存在运行期间读取的当前值和历史结果。用其中的 `decision_id` 在客户端 `trace.jsonl` 和服务端日志中查找同一次决策。服务端日志的 `provider_dispatch` 表示尝试调用 transport，实际请求次数查看 `http_budget.used`。脚本仅在真实 Aliyun 的 HTTP 200 结果与当前决策 ID、概率一致时以 0 退出。

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

## 5. 本地持久化决策回放与真实决策模型验证（P1/P2）

P1 接受 schema v1 感知对象外包一层 `agent-observation-envelope.v1`，示例见
[完整合成 envelope](docs/agent-observation-envelope-v1.example.json)。输入可来自完整合成 fixture 或明确授权的结构化 JSONL；缺少目标 epoch、generation、窗口或资源版本时会保存观测和缺项，决策保持 `null`。

```bash
# P1 完整合成闭环；默认不联网
.venv/bin/python scripts/replay_decision_agent.py \
  --demo --provider mock --output runs/decision-stage02-demo

# 每行一个 envelope 的离线回放
.venv/bin/python scripts/replay_decision_agent.py \
  --input-jsonl /path/to/agent-observations.jsonl \
  --provider mock --output runs/decision-stage02-jsonl

# 仅在明确恢复既有数据库时复用；consumer 重启沿用生产 session
.venv/bin/python scripts/replay_decision_agent.py \
  --demo --provider mock --resume runs/decision-stage02-demo/decision-agent.sqlite3

# P2 显式网络 smoke：三个合格合成场景、质量不足零请求、重复输入去重
# 每个场景最多重试一次，全批最多六个 HTTP 请求；只读取服务端 .env。
.venv/bin/python scripts/replay_decision_agent.py \
  --synthetic-scenarios --provider aliyun --enable-network \
  --output runs/decision-stage03-manual-smoke

# 可复现的五个 SQLite/provider 崩溃窗口；不联网
.venv/bin/python scripts/demo_decision_recovery.py \
  --output runs/decision-stage02-recovery
```

回放会生成 `input-snapshot.jsonl`、`trace.jsonl`、`decisions.jsonl`、`summary.json` 和 SQLite 数据库。off/mock 默认不读取 API Key；模板和 CLI 默认 provider 均为 `off`。真实调用必须同时指定 `--provider aliyun --enable-network`，且该入口仅接受内置合成场景。合成输入会标记 `input_origin=synthetic_fixture`、`simulated=true`；真实模型调用另标 `provider=aliyun`、`model_called=true`，概率类型为 `uncalibrated_model_estimate`。质量不足仍保留本地结果且不请求云端。
同一 producer session 在 consumer 重启后复用；producer 重启、时间重置或换视频时应提供新的 session。
原 EventStore/CareLoop 继续管理主事件确认、恢复及本地模拟报警，决策日志不拥有这些权限。

## 6. 目录与开发边界

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

## 7. 协作与 GitHub 准备

提交前运行源码边界检查：

```bash
python3 scripts/prepare_github.py
```

它检查 Git 可见文件、常见凭据格式、软链接、二进制和大文件，清单写入本地 `output/github-preparation/manifest.json`；不会执行 `git add`、提交或推送。还可用 `--export output/fallpre-source.zip` 导出同一文件范围。

具体范围和交接步骤见 [GitHub 准备说明](docs/GITHUB_PREPARATION.md)。新增顶层目录要显式更新 `.gitignore` 的发布边界，并复查清单。项目自身许可证尚未选定；第三方代码、模型和训练来源的已知条款见 [THIRD_PARTY.md](THIRD_PARTY.md)，不能默认整个仓库适用 MIT。
