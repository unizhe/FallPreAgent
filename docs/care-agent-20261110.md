# 单 Agent 双策略模拟闭环

本轮实现单个 FallPreventionAgent，连接现有感知事件、EventInterface 和 ExecutionService。报警策略只消费上游已确认事件，不执行旧 MVP 跌倒规则，不改变多人保护、确认条件、恢复条件或 ST 影子权限。执行器强制 dry_run，无外部通知或音频播放。

## 学弟第一次运行

按 [README](../README.md) 安装 Python 依赖后，运行以下命令。输出目录每次使用一个新路径。

```sh
.venv/bin/python scripts/replay_care_agent.py --demo --output runs/care-agent-demo
```

样例提供 7 条合成感知消息，产生 1 次模拟起身提醒、1 次模拟跌倒报警。脚本把每条消息重放一次以检查去重。查看 `summary.json`、`trace.jsonl` 和 `jev-requests.jsonl` 可核对计数、回执与建议请求。无需个人录像、模型权重、摄像头或 JEV 凭据。[单 Agent 与工具层架构图](assets/single-agent-tools-20261111.html)

## 感知与 Agent 接口

| 字段 | 用途 |
| --- | --- |
| `schema_version=1`、`stream_id`、`source_time_s` | 指定协议、视频流和观测时点 |
| `identity.action_epoch`、`observation.observable/event_decidable` | 区分目标连续性、骨架动作可见性和事件可判断性 |
| `current_action.label` | 提醒策略看到的当前动作 |
| `event.state/event_key/active_event_key/transitions` | 主感知判断、事件身份和状态变化 |
| `event.recovery_status/recovery_confirmed` | 上游提供的恢复证据 |
| `notification.emit` | 上游新事件提示，持久去重由 EventStore 完成 |
| `fall_process` | 时序判别的诊断证据，最新候选为影子输出 |

Python 集成使用 `FallPreventionAgent(CareLoop(event_store, execution_store))` 和 `agent.observe(session_id, perception)`。已有 PoseEngine 原始结果可调用 `loop.observe_live(session_id, stream_id, result)`。`agent.describe_tools()` 列出查询工具，`agent.invoke("get_event", {"event_key": key})` 查询已存事件，用 ingest 返回的持久事件键原样查询。需要记录人工接手或完成时，业务代码调用 `EventInterface.set_processing`，提供当前版本和请求 ID。旧 MVP 的 `inspect_person` 属于另一后端。

`--input` 接受一个 JSONL 文件或一个生产会话的 JSONL 目录。逐行支持直接的 schema v1 对象、最新版影子日志里的 `perception`、PerceptionChain 回放里的 `pipeline`，以及历史带 `primary` 的嵌套回放。旧版只有 `primary_state` 等诊断字段的扁平影子日志缺少完整的观测与事件信息，需要从新版入口重新导出。`--session` 对应生产会话，消费者重启沿用原标识，生产者重启或时钟/事件计数重置时换新标识。一个日志若横跨生产会话，先按会话拆分再回放。

## 可选的末段 ST＋几何影子入口

只做 Agent 开发时使用合成样例即可。要看最新影子候选，先依 [README](../README.md) 准备 temporal 资源，再向维护者获取登记的检查点。

```sh
.venv/bin/python scripts/project_resources.py import fall-head-tail-joint /path/to/tail-joint-fold-03.pt
.venv/bin/python scripts/project_resources.py doctor --profile tail-shadow
.venv/bin/python scripts/webcam_tail_shadow.py --log runs/camera-shadow.jsonl
.venv/bin/python scripts/replay_care_agent.py --input runs/camera-shadow.jsonl --output runs/camera-care
```

影子服务启动后在 Safari 打开 `http://127.0.0.1:8781/`，在 RGB 跌倒模式下采集日志，再回放该日志。若原服务已经启用既有受限事件融合，启动影子服务时加上 `--fall-event-fusion`。默认检查点经清单校验，显式 `--head` 保留给实验回归。检查点采用 286 维特征，包含末段冻结 ST 特征和三段几何，留出组为 11 月 8 日采集会话。它用于开发观察，各组的历史回归使用各自的留出检查点。[资源清单](../configs/runtime-resources.json)

## 提醒场景

首个场景是“已观察到坐姿后开始起身，一次站稳提示”。它是普通动作的模拟提示，不是失衡识别，不宣称预测跌倒，也不作为需要预警的人工真值。

条件为唯一可见姿态目标、相机稳定、时序可用、同一目标阶段、连续两次观测间隔不超过 0.35 秒、原事件状态 normal，坐姿转入 standing_up。每流冷却 30 源时间秒，冷却不限制报警。质量或身份中断清空动作锚点，不以物品所在区域过滤人员。

提示后的最多 5 秒观察记录为 observed_standing、observation_lost、observation_timeout 或 fall_confirmed。观察到站立不等于风险已解除，更不证明提示导致改善。文件结尾未完成的观察保留 pending，不伪造反馈。

## 回归结果

历史开发回归结果保存在维护者本机的 `runs/care-agent-20261110/final`。

- 复用 17 段旧视频的 7,893 条已保存实时后端记录，没有重跑视觉模型或复制视频。来源路径和 SHA256 见 summary.json。
- 14 个权威事件生成 14 次模拟报警；与原 new_event 的视频流和源时间逐一相同，无新增、丢失或提前/推后。
- 每条记录重复输入一次，共检查 7,893 次重试，无重复执行。
- 原 20 段正常片段沿用原回归，桥接层没有新增报警。本轮不独立重估视觉准确率、长期误报率或端到端墙钟延迟。
- 模拟提示 2 次：old-clip05 的 15.300 秒；old-old 的 57.015 秒。随后在 15.976、57.549 秒观察到站立。两者都不是已经标注的跌倒前失衡事件。
- 18 条 JEV 建议请求：2 条提醒、16 条疑似跌倒进入点；疑似候选不是 16 次确认跌倒。历史最多 2.5 秒，保留质量、动作、事件状态和影子头的实验属性。
- 40 项新桥接/事件接口测试、102 项原策略/事件融合/转换点/输出语义测试通过；Ruff 通过。

## JEV 状态与运行

适配器依据 TypeSafe System One 文档实现 `state + questions` 的 noul 判断、类型/有限数值检查、超时和错误降级：https://docs.typesafe.ai/api/system-one 。分数不视为校准后的跌倒概率。未额外引入传统 LLM。

历史回归时未配置 TYPESAFE_API_KEY，18 条真实调用前检查均返回 UNAVAILABLE，没有向服务发送数据；不能声称已有 JEV 收益。单元测试中的 transport 是明确模拟，只验证协议。

在本机配置 TYPESAFE_API_KEY（不要写进仓库或聊天记录）后运行：

```sh
.venv/bin/python scripts/compare_jev_suggestions.py \
  --input runs/care-agent-20261110/final/jev-requests.jsonl \
  --output runs/care-agent-20261110/final/jev-comparison-live.jsonl \
  --enable-network
```

该命令仅向 TypeSafe 发送已导出的结构化状态，不发送视频、图片、本地路径或姓名。JEV_MODEL 可固定具体模型版本；结果保留服务返回的模型名称。输出仅用于历史建议对照，任何迟到建议均不会直接执行。

重新验证闭环使用新输出目录：

```sh
.venv/bin/python scripts/replay_care_agent.py \
  --input runs/realtime-tail-shadow-20261110/live \
  --output runs/care-agent-next
```

接下来先取得真实建议，检查正常起身是否被不必要地建议提醒，以及疑似跌倒建议与人工事件边界是否一致，再判断是否值得有限接入。当前可证明的是接口与模拟闭环可运行，尚不能证明提醒减少跌倒或 JEV 降低误报。不需要补录即可继续该对照。
