# core/schemas — 跨模块共享 DTO

## 定位

本目录是**全局共享数据契约（DTO）的唯一家**。一个数据结构进入这里的条件有且只有一个：**被 ≥ 2 个上层业务模块同时读或写**（例如 `red_team/` 与 `blue_team/`、或 `red_team/mutation/` 与 `sandbox/`）。

本目录**禁止承载业务逻辑**：不得做 I/O、不得调 LLM、不得读配置、不得起线程。DTO 只做三件事——声明字段、在 `__post_init__` 里做静态不变量校验、可选地提供 `to_dict` / `from_dict` 这类纯函数序列化。任何需要依赖上下文（文件系统、网络、运行时状态）的代码都必须留在各自的业务模块里。

## 对外接口

当前共 8 个公开 DTO，按用途分组：

变体搜索点（红队唯一可改动的点）

- `SkillVariant` — CLAW 论文的 (o, τ, c, κ, d) 因子化骨架；红队的搜索即在本 DTO 的实例空间上进行。
- `AttackObjective` — 7 种攻击目标的闭合词表（含历史 payload-key 映射）。
- `Topology` — 复合拓扑 τ：`single` / `pair` / `k_chain`。
- `Channel` — 副作用通道 κ：`subprocess` / `filesystem` / `network` / `memory`。

审计输出（蓝队三元组）

- `PaperFinding` + `Evidence` + `Layer` — 蓝队 A–E 层的单条 finding 与指针式证据。
- `RuntimeTrace` — Layer D 动态探针的紧凑摘要。
- `AuditOutput` — 蓝队一次审计的完整返回（decorated_feedback + findings + trace + verdict）。

L1 判定与 payload 流转（本轮新增）

- `LethalityVerdict` — 一次致命性判定结果：(`lethal`, `objective`, `evidence`, `rule_id`)，frozen + slots，lethal 必须带至少一条 evidence。
- `PayloadArtifact` — 一个已渲染的攻击 payload（`doc` 层 = SKILL.md 追加文本，`bin` 层 = bash wrapper 源码），frozen + slots。
- `SensorReading` — `sandbox.honeypot.HoneypotResult` 的语义别名，供 oracle 层在不 import sandbox 的情况下声明 sensor 输入类型。长期会独立成 frozen dataclass。

## 依赖方向

```
core/schemas/   ──→   Python 标准库
                 └→   sandbox.honeypot  （仅 SensorReading 做 re-export）
```

所有其他模块（`red_team/`、`blue_team/`、`sandbox/`（除 honeypot 自身）、`shared/`、`eval/`、`oracle/`）**可以** `from core.schemas import …`。

**反向禁止**：`core/schemas/*` 里**不得** import 任何业务模块（`red_team.*`、`blue_team.*`、`shared.loaders.*`、`eval.*` 等）。`paper_finding.py` 里 `from_structured` 方法对 `core.models.StructuredFinding` 的 import 是**局部延迟 import**，仅为向后兼容老 scanner，新代码不得引入新的此类环依赖。

## 典型调用路径

```python
from core.schemas import (
    AttackObjective, SkillVariant, Topology,
    LethalityVerdict, PayloadArtifact, SensorReading,
)

# 红队生成一条变体 + 一份 payload
variant = SkillVariant(o=AttackObjective.DATA_EXFILTRATION, tau=Topology.SINGLE, chain=["repo-summary"])
payload = PayloadArtifact(name="data_exfil_doc", layer="doc",
                          objective=variant.o, content="...", metadata={})

# 沙箱跑完后，oracle 读 sensor → 出 verdict
reading: SensorReading = run_in_sandbox(variant, payload)   # HoneypotResult 实例
verdict = LethalityVerdict(lethal=reading.canary_token_leaked,
                           objective=variant.o,
                           evidence=tuple(reading.evidence),
                           rule_id="data_exfil.canary_read")
```

## 如何扩展

新增 DTO 必须同时满足以下三条准则，否则放回业务模块：

1. **跨模块性**：至少被 2 个上层业务模块读或写。单一模块内部使用的数据结构请留在该模块。
2. **只读数据**：没有行为、没有副作用、没有外部依赖。仅字段 + 静态校验 + 纯函数序列化。
3. **冻结结构**：首选 `@dataclass(frozen=True, slots=True)`（新代码）或至少 `slots=True`（旧代码兼容）。可变集合字段请在构造器后立刻规范化（如 `tuple(...)`）或约定调用方不得原地修改。

同时要：在 `__init__.py` 的 `from … import …` 与 `__all__` 中加入；在本 README「对外接口」一节补一行职责说明；为每个新 DTO 写一份 `tests/unit/core/schemas/test_<name>.py`，至少覆盖构造、不变量违反、frozen 特性、相等性。

## 反模式

以下写法**会被拒**，请在 review 时指出：

1. **在本目录写业务逻辑**。例如在 `PayloadArtifact` 里塞一个 `render_from_template(ctx: RedTeamCtx)` 方法，或在 `LethalityVerdict.judge()` 里调 LLM。DTO 仅描述数据，行为请留在 `red_team/payloads/`、`oracle/rules/` 等处。
2. **反向 import 上层模块**。例如在 `verdict.py` 里写 `from red_team.oracle.rules import DataExfilRule`、或在 `payload.py` 顶层 `from sandbox.runner import run`。这会让 `core.schemas` 变成整个代码库的瓶颈层，任何模块稍有改动都会回流到这里。
3. **可变默认值或共享引用**。例如 `metadata: dict = {}` 作为字段默认（而非 `field(default_factory=dict)`），或把外部传入的 `list` 直接存成字段不做 `tuple(...)` 规范化。frozen 数据类被多个判定/log 链路共享时，一次不小心的原地修改就是跨模块的幽灵 bug。
