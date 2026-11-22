# 第一阶段开发任务：P0 配置、合同与 off/mock

日期：2026-11-17。当前状态：P0 修订一复审通过，可进入 P1。总方案为 [视频决策闭环方案](agent-decision-loop-plan-20261117.md)。首轮问题与失败证据保留在 `output/agent-review/stage-01/reviewer-review.md`；通过记录为 `output/agent-review/stage-01/reviewer-approval-revision-01.md`。下一阶段见 [P1 开发任务](agent-development-stage-02-20261117.md)。

## 可直接交给开发窗口的提示词

请在仓库根目录 完成智能体建设的第一阶段 P0。先阅读用户提供的 AGENTS.md 要求、`docs/ARCHITECTURE.md`、`docs/DEVELOPMENT.md`、`docs/agent-decision-loop-plan-20261117.md` 和本任务。检查现有 Git 状态，保留方案文档、配置模板及其他人的修改。

本阶段目标是建立可验证的配置与数据合同，提供无需模型、视频或网络的 off/mock 调用路径。实际持久闭环在 P1，阿里云真实调用在 P2，视频接入在 P3；本阶段完成后提交审核，不自行开始下一阶段。

### 1. 配置加载

- 新增 `src/fallpre/decision_config.py`。显式加载项目根目录 `.env`，优先级为进程环境变量 > `.env` > 非秘密默认值；不得使用 shell source 执行配置。支持测试注入独立 env 文件路径及环境映射，测试不能依赖或修改开发机真实配置。
- 本机 `.env` 已由审核窗口导入真实 `DASHSCOPE_API_KEY`、`DASHSCOPE_WORKSPACE_ID`、`DASHSCOPE_REGION=cn-beijing`。不要覆盖、重写或打印这个文件；不要复制 CSV 到源码目录。`.env.example` 的 Key 和空间 ID 继续为空，provider 默认 off。
- 对照模板支持全部配置项：provider/model/head、timeout、max_retries、min_interval、source/wall age、max_reobservations、DB 路径。校验类型和范围，拒绝非法 provider、地域、空必要项、NaN/Inf、负数及 bool 冒充数值；时限与间隔要求正数，retry/reobservations 要求非负整数并设置合理上限。
- off/mock 在没有 Key 或空间 ID 时仍能工作；aliyun 配置模式检查必要项，但 P0 不实现网络调用。API Key 是不透明秘密字符串，不强制假定以 `sk-` 开头。
- 使用秘密类型或等价保护，使 repr、异常、配置 JSON、日志和测试失败信息不泄漏 Key。对外配置状态仅返回 configured、provider、model、region 和缺失项名称。
- 如需 python-dotenv，按项目依赖方式新增并更新锁文件，不批量升级无关依赖。不要安装阿里云 SDK 或引入额外 Agent 框架。

### 2. 数据合同与 provider 抽象

- 新增 `src/fallpre/decision_contracts.py`，定义 `DecisionEvidence`、`ProviderResult`、`DecisionView` 和必要枚举。依据总方案第 5–6 节，明确 session/stream、epoch/generation、snapshot/decision、source_window、质量、模型/头版本、概率来源、状态及 reason_codes。
- 明确既有感知 schema v1 和新增 decision-evidence.v1 / agent-decision.v1 是不同合同，不改写原感知 payload，不改变旧 MVP 合同。
- 概率目标固定 `observed_fall_in_window`；区分 raw、calibrated、当前可用概率。当前可用概率只能是有限的 [0,1] 数值或 null。不可用、过期和观测不足时为 null；禁止用 0 代替未知。
- ready 必须有符合身份、窗口和质量合同的结果；mock 必须为 provider=mock、probability_kind=synthetic_fixture。真实来源与已校准状态只预留定义，P0 不宣称实现校准；使用 calibrated 状态必须要求校准版本。
- 定义 `DecisionProvider.assess(evidence, deadline=...)` 协议；deadline 表示单调时钟截止时点，避免与视频源时间混用。允许可注入时钟做确定性测试。
- 本阶段为概率评估，`changes_alarm` 和 `external_action_performed` 固定 false；不提供修改主事件确认、恢复或执行外部动作的权限。

### 3. off/mock 与最小演示

- 提供 Disabled/Mock provider。off 返回 skipped；mock 从明确的合成 fixture 返回确定结果，不联网、不调用 YOLO/ST、不伪装真实模型输出。
- 至少覆盖三个示例：证据完整返回合成概率；证据不足返回 null 和 observe_more；provider off 返回 skipped。mock 的 observe_more 只返回建议，本阶段不实现采样或重评循环。
- aliyun provider 在 P0 必须明确返回 not_implemented/unavailable 或专门错误，不能静默替换成 mock；不得因为本机已有 Key 自动发请求。
- 新增可执行的最小演示入口，例如 `scripts/demo_decision_contract.py --provider mock --output runs/decision-stage01`。演示仅验证新合同，可使用独立合成证据 fixture。输出显式声明 simulated=true、model_called=false，并保留输入与结果关联。
- 输出路径已存在时避免覆盖旧实验，使用新目录或明确失败。演示不输出原始配置对象或秘密。

### 4. 测试与提交审核材料

新增有意义的配置和合同测试，覆盖：环境覆盖顺序；off/mock 缺密钥；aliyun 配置缺项；秘密不出现在 repr/JSON/异常；非法数值；身份或窗口必需项缺失；未来/倒置窗口；raw 与对外概率语义；mock 标记；非 ready 当前概率为空；off/mock 即使外界存在真实 Key 也不会发出网络请求。用测试替身在触发网络时直接失败，不读取用户真实 Key 充当测试 fixture。

允许 score=0 或 1；拒绝 bool、NaN/Inf、越界和不符合合同的枚举。时间检查只在相关对象间比较，不能以源时间和 UTC/单调时钟相减判断年龄。不得为通过测试放松既有事件或质量约束。

运行新增测试、变更文件 Ruff、`scripts/check_portable.py` 和 `scripts/prepare_github.py`；按项目约定把新增可移植测试纳入相应检查入口。保留命令、退出码、结果摘要与可复现步骤。P0 无需浏览器；以后需要页面验证默认使用 Safari，不启动 Chrome/Chromium。

交付：

1. 实际代码与测试；清楚列出修改文件、依赖变化及设计选择。
2. 三个场景的合成输入和输出，放在被 Git 忽略的 `runs/decision-stage01-*` 等新目录。
3. 审核报告 `output/agent-review/stage-01/report.md`，列出复现命令、退出码、测试计数、已知限制、产物绝对路径；不要包含密钥、CSV 内容或完整 `.env`。
4. 最终回复说明“P0 已提交审核，等待审核窗口结论”，并给出报告路径。不要把 mock 结果描述成真实模型、视频闭环或准确率验证。

## 审核窗口的通过标准

审核时读取实际差异和产物，并复跑必要命令，不能只依据开发窗口自述。逐项检查：

| 检查项 | 通过要求 |
| --- | --- |
| 配置与凭据 | 原 `.env` 被保留且仍被忽略；模板无真实秘密；覆盖顺序、错误处理和脱敏有测试证据 |
| 数据合同 | 窗口、身份、概率语义与未知状态无歧义；原感知合同与主事件权限不变 |
| off/mock | 三种场景可复现、mock 标签明显、不会联网、不会自动切换 provider |
| 工程回归 | 新增测试、Ruff、可分享回归、发布边界检查完成；失败项有真实原因，不能静默跳过 |
| 交付真实性 | 报告对应实际代码与产物，未夸大模型/视频/校准验证 |

审核结论采用“通过 / 未通过”。未通过时给出问题位置、影响、修改建议和复验要求，继续留在 P0；通过后由审核窗口给出 P1 持久化本地闭环的具体提示词。P0 通过不代表真实 API Key 或模型权限已经验证。
