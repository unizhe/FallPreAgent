# 第三阶段开发任务：P2 阿里云真实决策接口

日期：2026-11-17。前置：P1 修订一复审通过。当前状态：P2 修订二复审通过，见 `output/agent-review/stage-03/reviewer-approval-revision-02.md`；可按 `docs/agent-development-stage-04-20261117.md` 进入 P3。此前不通过记录保留在 stage-03 审核目录。以下保留原阶段任务；前置依据为总方案及 `output/agent-review/stage-02/reviewer-approval-revision-01.md`。

## 给开发窗口的提示词

请在仓库根目录 实施 P2。先阅读用户 AGENTS.md、架构/开发指南、总方案、本任务和 P0/P1 审核记录，保留现有修改及旧实验。目标是把真实 Aliyun provider 接入已通过的持久化闭环，并用有限的合成证据完成真实服务验证。视频推理、摄像头、网页和模型校准留到后续阶段。

### 1. 凭据与调用协议

根目录 .env 已由用户提供的 CSV 填写 Key、workspace 和北京地域。本阶段可在显式真实调用入口读取这些配置；不要再次索取 Key，不输出、不覆盖、不提交 .env，不把密钥作为命令行参数或写入数据库。模板和全局默认 provider 保持 off；单次验证用 CLI 显式覆盖。

2026-11-17 复核的官方协议为 POST `https://{WorkspaceId}.cn-beijing.maas.aliyuncs.com/compatible-mode/v1/systemone`，Bearer 鉴权，请求体为 model/state/questions；模型为 decision-model-preview。`noul` 给出 P(yes)，`choice` 提供选项、概率分布和置信度。这是专用决策接口，不使用 chat/completions，也不由模型调用任意工具。[阿里云 API 文档](https://help.aliyun.com/zh/model-studio/decision-model-api)

实现 `adapters/aliyun_decision.py`，使用可注入 HTTP transport。可沿用标准库或增加必要且锁定的 HTTP 依赖；无需通用 Agent 框架。workspace 必须验证为合法域名标签，region/model 使用明确允许值。固定 HTTPS 网关，禁止任意 endpoint 及跨域重定向转发凭据，开启 TLS 验证。

真实调用同时要求 provider=aliyun 和明确的网络启用参数，例如 `--enable-network`；缺一项不能联网。off/mock 即使存在真实配置也不读取/使用 Key，不静默切换 provider。

### 2. 证据构建和三个问题

新增独立的 state 构建器和版本化问题定义。state 从已校验合同按白名单提取，包括目标连续性、窗口、真实观测数、质量、动作/过程、ST 分数、头原始分数、几何/冲突、事件与恢复语义、证据共享关系及必要版本。

现有 P0/P1 合同尚未携带总方案中的所有动作/几何摘要。需要新增字段时显式定义可选项和来源映射，保留缺失；同步 schema/迁移兼容和源数据一致性测试。不得根据高分反推已观测下降或补造几何。初次真实验证使用完整、明确标记的合成证据。

不发送视频、图片、完整骨架、路径、姓名、反馈备注、完整 envelope 或任意自由文本。原始 perception 保留本地；问句由代码固定，不由输入内容覆盖。上一轮模型概率不能充当新的视觉证据。

一次请求的三个问题沿用总方案：

- `fall_occurred`：noul，仅判断指定目标在指定窗口内是否发生跌倒过程。
- `evidence_sufficiency`：choice，sufficient/insufficient/conflicting。
- `next_step`：choice，emit_result/observe_more/request_review。

prompt_version 和 state_schema_version 固定并进入决策去重材料。问题选择、质量门和模型输出解析分别实现，避免把未知默认成正常。

### 3. 响应合同和降级

验证三个答案的存在、类型、枚举、完整分布、有限数值、[0,1] 范围和总和容差，拒绝 bool 冒充数字。校验服务 model 回显，保留 request_id、input_tokens、服务 latency_ms；noul 不假定存在 confidence。异常响应正文不直接打印或持久化，错误只保留脱敏类别和可安全记录的 request_id。

只有质量合格、模型答案充分且允许发布时，fall_probability=fall_probability_raw=fall_occurred.noul，probability_kind=uncalibrated_model_estimate；calibrated 为 null。不存在“达到固定准确率”验收门槛。

不充分/冲突/回答组合不一致时，当前概率为空，raw 可作审计但不冒充当前概率；保留模型原建议与本地工作流最终处置。继续观察或复核按总方案及 P0 合同处理，不将异常回答强制转换成 ready。

质量门不合格时不发云请求；记录本地不足/过期结果并按 P1 继续观察。缺少配置、超时、401/403、429、5xx、无效 JSON/合同分别给出原因，失败不能回退成 mock 或 ST 概率。主事件不受云模型失败或低分影响。

### 4. 接入 P1，统一时限与重试

接通 provider factory、DecisionLoop、worker 和 replay/独立诊断 CLI；不要只新增绕开持久链路的 curl 演示。身份、源窗口、版本、任务 attempt 和服务 request_id 均可追溯，worker 共享服务启动标识。

现有 P1 给每次 provider 调用重新计算 deadline，并对异常立即重试。P2 必须明确一个逻辑任务的总网络预算：默认 3 秒，包含网络、等待和重试；最多一次重试。不要同时在 HTTP adapter 与 durable task 层各自重试而倍增调用。建议 transport 每 attempt 至多一次发送，由持久调度层统一管理 attempt 与剩余时间。

网络预算从首次执行开始，而非视频源时间；同进程执行用单调时钟，重启时只能保守恢复剩余预算，不能每次重启重新获得完整预算。连接/读取/响应大小均有限制；仅设置 socket 单次读取 timeout 不足以保证整体 deadline，需测试慢响应/分块读取耗时。

可恢复连接错误、429、部分 5xx 才重试；401/403、参数/合同错误不反复请求。退避与 Retry-After 必须受剩余预算约束，过长直接结束任务。错误分级需与 P1 的 retry_pending/终态/子 workflow 收束一致；结束失败应能在查询中看到 unavailable 及原因，而不只有没有结果的 cancelled task。保留 R1–R5 修复的测试。

停止、过期、换人、迟到响应不更新当前视图。调用在数据库事务之外；持久化之后才发布。重试/崩溃造成外部请求可能重复需诚实记录，仍只接受一个逻辑最终结果。新增持久字段应有迁移和旧库兼容测试。

### 5. 无网络测试先行

新增 transport 替身测试：成功、三个问题缺项/类型错误、NaN/Inf/bool、未知选项/分布错误、答案冲突、非 JSON、响应过大、401/403/429/5xx、Retry-After 超预算、连接/读取超时、慢响应、截止前后返回、重启预算恢复、无 Key、off/mock 禁网、重定向拒绝和日志/导出脱敏。

用假的 Key 标记证明它不出现在日志、异常、数据库、产物中；不要把用户真实 Key 当测试 fixture。断言具体发送次数和 attempt 数，防止“测试通过但重试翻倍”。P0/P1 与便携回归默认零真实网络，真调用 smoke test 只通过独立显式入口执行。

### 6. 有限真实调用验收

完成上述实现和无网络测试后，使用现有 .env，通过显式 `--provider aliyun --enable-network` 入口执行小批合成状态验证。这是本阶段任务的一部分，不需要再询问已经提供的密钥。

- 准备三个完整合格的合成窗口：正常活动、跌倒过程证据、存在语义歧义但未触犯确定性质量门。每个场景最多一个逻辑任务，所有重试合计最多 6 次 HTTP 发送；到达预算停止。
- 再做一个本地质量不足场景，证明其 HTTP 发送次数为 0。
- 至少一个真实请求要走 observe → durable task → worker/provider → 保存结果 → 查询的完整 P1 路径；同输入重放不能多发请求。
- 真实响应不要求恰好符合预设分数，也不保证三者可排序。验证协议、权限、字段、时延和可追踪性；模型判断异常保留为实验结果，不改提示词反复试到满意。
- 区分 `input_origin=synthetic_fixture` 与 `provider=aliyun`、`model_called=true`；真实模型对合成输入的输出不是 synthetic_fixture 概率。可以保留 simulated 标志，但要解释它仅指输入/执行情境，避免与“未调用模型”混淆。
- 记录每场景的脱敏 state、问题版本、task/attempt、服务 request_id、合法答案、HTTP 状态、客户端耗时、服务 latency 和 token 用量，不记录 Authorization。把 raw 结果标为未校准估计。
- 若权限、地域、密钥或服务不可用，记录失败类别与 request_id（如有）并提交审核，不能声称 P2 真调用通过；不要静默换模型/地域，也不要无限重试。

如默认 3 秒预算导致真实环境稳定超时，先记录真实超时事实；可在报告中提出明确预算调整建议供审核，不暗改默认值或用超时输出冒充成功。

### 7. 交付

交付适配器、state/questions/解析合同、P1 集成、诊断入口、测试、迁移说明和 `output/agent-review/stage-03/report.md`。产物放新建 `runs/decision-stage03-*`，保留旧记录。

报告分别列明：模拟 transport 测试、P0/P1 回归、真实服务调用，附命令、退出码、计数、产物绝对路径、调用数、request_id、用量及当前限制。执行阶段 Ruff、check_portable.py、prepare_github.py、git diff --check。敏感错误正文不能为了“完整审计”原样导出。

P2 不改变主事件确认/恢复/报警权限，不发送实际通知，不加载视频模型，也不进入 P3。完成后提交本窗口审核；P2 通过后才给视频贯通提示词。
