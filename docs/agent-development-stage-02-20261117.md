# 第二阶段开发任务：P1 本地持久化闭环

日期：2026-11-17。前置条件：P0 复审通过，记录位于 `output/agent-review/stage-01/reviewer-approval-revision-01.md`。本阶段状态：P1 修订一复审通过，可进入 P2。首轮问题记录保留在 `output/agent-review/stage-02/reviewer-review.md`；通过记录为 `output/agent-review/stage-02/reviewer-approval-revision-01.md`。下一阶段见 [P2 开发任务](agent-development-stage-03-20261117.md)。

## 给开发窗口的提示词

请在仓库根目录 实施 P1。阅读用户 AGENTS.md 要求、架构与开发指南、总方案 `docs/agent-decision-loop-plan-20261117.md`、本任务及 P0 通过记录。保留现有修改、凭据和实验产物。

本阶段目标：用离线合成观测/已有授权结构化 JSONL，完成“观测入库 → 校验与调度 → mock 判断 → 结果入库和查询 → 新观测有限重评”的可恢复闭环。实现本地 Python 接口和 CLI 回放，不接摄像头/视频推理、不实现真实阿里云网络调用、不做网页改造、不进入 P2。

### 1. 输入边界与证据适配

- 新增 `decision_evidence.py`，定义回放输入 envelope，包含生产 session、stream、源时间、原 schema v1 perception 及可选的决策证据摘要。提供完整合成 fixture 和 schema 示例。
- 保留原感知 payload，明确映射事件、恢复、身份、窗口及证据来源；来源字段不够时，记录具体缺项，不伪造 epoch、generation、窗口、分数或头哈希。正常窗口也能参与决策，不限于 confirmed。
- 当原感知只有简化字段或完全缺失目标/窗口，使用 envelope 的不足状态与 `decision=null` 表达并持久化原因，不强行构造 P0 所要求的完整 DecisionEvidence。后续有完整观测再建立决策。
- 所有进入 provider 和存储的合同按 P0 校验。不能用 `model_construct` 或未经验证的 `model_copy(update=...)` 绕过跨字段验证；存储时复制/序列化快照，调用方后续修改对象不得改变已入库证据。
- 身份、源时间和原事件键由观测生产者提供；decision_id、任务摘要等由编排器生成/校验，不信任调用方摘要。合成资源的版本/哈希需标为 fixture，不能冒充真实权重。

### 2. 持久化、幂等和恢复

- 新增 `decision_store.py`。使用 SQLite、版本化且可重复执行的建表/迁移，保存 run/session、原观测、调度登记、任务/attempt、不可变 provider 结果、父子重评关系及反馈。
- 观测唯一键为 `(session_id, stream_id, source_time_s)`。同键同内容返回原 receipt，同键不同内容报冲突且无副作用。新源时间观测不能因同属一个事件而被吞掉。
- 决策去重摘要由规范化证据、provider/model、头及预处理版本、prompt/policy/eligibility 版本构成；包含会话/目标/代次和源窗口。排除随机 ID、入库时间、耗时和摘要自身，禁止 NaN。保持原证据摘要与决策请求摘要职责清楚。
- 数据库唯一约束处理重复调度和竞争。接受快照与登记工作项原子提交；若与现有 EventStore 跨事务，设置可恢复的桥接登记并扫描补齐，不能在崩溃窗口丢观测或漏任务。
- 区分任务生命周期与模型结果状态。任务支持 pending/running/completed/cancelled 等、attempt、租约和有界重试；陈旧租约回收使用可跨进程恢复的时间设计，不持久化单调时钟值后跨进程相减。
- 模型调用在数据库事务外执行；SQLite 连接只由所属线程使用，worker 通过队列交回结果。先持久提交结果再供查询，同一逻辑任务最多接受一个最终结果；记录重复/迟到 attempt，不承诺未来云端 exactly-once。
- consumer 重启沿用 session；producer 重启、时间重置、换视频必须新 session。旧时间观测可以审计保存，但不回退当前状态或触发当前任务。

### 3. 本地编排与有限重评

- 新增 `decision_loop.py`，提供 observe / accept_result / latest / list_decisions / stop / feedback 等本地接口，参数、返回和错误合同可测试。保留 off/mock，aliyun 仍明确 unavailable，不联网。
- 确定性质量门不能由模型绕过。mock 支持注入固定场景，展示 observe_more 后确实等待新观测，而非拿旧证据反复询问。
- 一个流最多一个在途任务；普通窗口按配置最小源时间间隔调度，待处理窗口可合并为最新一条。关键事件转移单独保留并可追溯，不因普通窗口合并而丢失；任务队列有界。
- 至多一次附加观察（首版默认值），新证据同一目标/代次、源时间至少前进约 1 秒；记录 parent_decision_id 和消耗预算。换目标/失去连续性、EOF、stop、约 5 秒墙钟等待超时均结束等待，保留明确终态与原因。
- 结束补采时不得仍提示无限 observe_more；增加独立 workflow 状态表达 finished/request_review/cancelled，并保留原 provider 建议作为历史，不改写它。off 模式不自动补采。
- 用可注入单调时钟测试进程内 deadline/等待；源时间只用于视频时间轴和采样间隔。快速离线回放不伪装成真实网络实时性能。

### 4. 历史结果与当前视图

- ProviderResult 为不可变历史判断；latest 根据最新观测和配置投影。检查同 run/session、stream、目标/代次、结果次序、关键状态转移、观测质量，以及源时间/墙钟年龄。
- P0 的 ST 0.6 秒校验约束输入相对触发观测；它不同于已完成决策的默认 2.5 秒源年龄和 5 秒墙钟年龄。分别测试并保留时间来源。
- 换人、质量丢失、过期、stop、恢复/新关键转移或迟到乱序结果，当前概率为空；历史结果仍可分页查询且保持原窗口和原概率来源。重启后无法确认时效的旧结果只供历史查看，不能恢复成 current ready。
- 仅生成新视图，不原地修改旧结果。若沿用 DecisionView，则非当前结果投影为明确 stale/cancelled 等并清空当前概率；原 ready 结果保留在历史。
- P0 把 skipped 限制为 provider=off。本阶段调度抑制/背压优先放入独立 task/workflow reason，不能伪造 provider=off。确需扩展合同，须记录理由和兼容测试。

### 5. 事件、反馈及现有闭环兼容

- 复用 EventStore 的原权威事件键及既有 CareLoop/ExecutionService 模拟执行语义。模型结果不修改主事件确认、恢复或权限；一个事件多个决策 revision 不造成重复模拟报警。
- 可在集成测试中组合原 CareLoop 和新 DecisionLoop，采用单拥有者或明确桥接，不必重写旧组件。未修改权威事件的前提下，验证同一上游 confirmed 及重复输入仍只产生一次既有模拟执行。
- 人工反馈另表保存 fall/non_fall/uncertain、request_id、expected_version 与历史。重试幂等、版本冲突不覆盖，不把人工处理完成当成身体恢复，也不自动训练模型。反馈正文不进入 provider state。

### 6. 回放与验收场景

提供 `scripts/replay_decision_agent.py --demo --provider mock --output <新目录>`，以及读取明确格式 JSONL 的入口。演示输出输入快照、trace.jsonl、decisions.jsonl、summary.json 和 SQLite；明确 simulated=true、model_called=false，不覆盖旧目录。重启恢复通过显式 resume 参数复用数据库，与新建输出行为区分。

至少覆盖以下可重复场景：

1. 完整正常窗口 → 一次 ready/mock/synthetic_fixture；质量不足 → null。
2. observe_more → 新窗口 → 一次子决策；同一窗口重试不增加预算，重评最多一次。
3. EOF/暂停超时/stop/换目标中止等待，旧概率不继续显示当前。
4. 同观测重试、同时间冲突、同事件新窗口、多流相同局部 ID、producer 新 session。
5. 结果乱序、过期/旧 generation、旧结果跨重启、恢复后迟到结果，历史保留但当前视图不倒退。
6. 两个调度者竞争唯一任务；在接受后调度前、运行中、provider 返回后写库前、提交后回执前注入失败并重启，证明不漏逻辑任务、不重复接受最终结果。
7. 一个上游 confirmed 的多个窗口/重复输入/消费者重启仍只产生一次既有模拟报警；模型低分、失败或 stale 不撤销上游事件。
8. 反馈重试与版本冲突；off/mock 全程禁网，配置/日志/数据库及导出不包含真实 Key。

使用确定性假时钟、替身 provider 和临时数据库，避免依赖真实 sleep 或本机模型。将回归纳入 check_portable.py；运行新增测试、Ruff、便携回归和发布边界检查。保留所有 P0 校验；若调整 schema，提供兼容策略和原始观测不变的验证。

### 7. 交付审核

交付代码、正式测试、可复现演示和 `output/agent-review/stage-02/report.md`。报告列出实际命令、退出码、测试计数、数据库与 trace 的绝对路径，并逐项解释去重、失败恢复和补采预算证据。不要把 mock 数据称为真实跌倒准确率或模型调用成功。

根目录 .env 保持现状且不输出；合成测试不读取真实 Key。P1 无需浏览器；后续需要浏览器时使用 Safari。完成后只提交本窗口审核，不自动进入 P2。
