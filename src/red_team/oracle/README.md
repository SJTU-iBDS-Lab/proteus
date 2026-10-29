# `red_team.oracle` — L1 判定层

## 1. 定位

L1 判定层：给定一条 `SkillVariant` 与一次 sandbox 执行后的 `SensorReading`（即 `sandbox.honeypot.HoneypotResult`），判定该 variant 声明的 `AttackObjective` 是否在观测证据里得到兑现。

本层与 blue team 平行：red team 在 mutation 层产出 bypass 候选，blue team 的 scanner / veto 决定能否拦下；只有当 blue team 未拦下 **且** 本层判定为 lethal 时，mutation 才算成功，驱动器才会把这条路径写入 verbal-gradient log 并结束该分支的进一步演化。

规则的灵魂是「保守」：含糊信号（比如仅仅出现了网络关键字但没有任何 canary 泄漏）不构成 lethal。缺规则的 objective 失败闭合（返回 False），不能借由 blue team 单边通过就声称攻击成功。

## 2. 对外接口

本模块对外只暴露三个名字：

| 名字 | 角色 |
| --- | --- |
| `lethal(variant, hp, o) -> bool` | 兼容入口。历史代码（`evolution/driver.py` 等）不改一行就可复用 |
| `ObjectiveJudge` | `runtime_checkable` Protocol，规定一个 judge 必须有 `objective: AttackObjective` 属性与 `judge(variant, reading) -> LethalityVerdict` 方法 |
| `REGISTRY` | `dict[AttackObjective, ObjectiveJudge]`，7 个 objective 对应的单例 judge |

新代码应优先拿到 `LethalityVerdict` 结构体（含 `lethal / objective / evidence / rule_id`），因为驱动层的 verbal-gradient log 依赖 `evidence` 和 `rule_id` 分组归因；只有把返回值压缩成布尔的老调用方才应该继续走 `lethal()`。

## 3. 依赖方向

本模块只依赖：

- `core.schemas`：`AttackObjective` / `LethalityVerdict` / `SensorReading` / `SkillVariant`。
- `sandbox.honeypot`：通过 `core.schemas.SensorReading` 的别名间接依赖 `HoneypotResult`。

禁止方向（一律算回环依赖，不可引入）：

- 从 `oracle` 里 `import red_team.evolution.*`
- 从 `oracle` 里 `import red_team.campaign.*` / `red_team.agent.*`
- 从 `oracle` 里读写 `variant` 上除 `variant.o / variant.chain / variant.tau` 之外的字段

调用方向始终是 `evolution / campaign → oracle → core.schemas / sandbox.honeypot`。

## 4. 典型调用路径

**旧 API（`evolution/driver.py` 现用写法，零改动）：**

```python
from red_team.oracle import lethal

if not blue_verdict.vetoed and lethal(variant, honeypot_result, variant.o):
    record_successful_mutation(variant)
```

**新 Protocol（拿到完整 verdict，写日志用）：**

```python
from red_team.oracle import REGISTRY

judge = REGISTRY[variant.o]
verdict = judge.judge(variant, honeypot_result)
if verdict.lethal:
    verbal_gradient.log(
        rule_id=verdict.rule_id,
        evidence=verdict.evidence,
        objective=verdict.objective,
    )
```

## 5. 如何扩展一个新 objective

1. **新增枚举值**：在 `core.schemas.attack_objective.AttackObjective` 里加一条（全大写 + 下划线，value 用小写）。
2. **新建 judge 文件**：在 `red_team/oracle/objectives/` 下新建 `xxx.py`，照 `data_exfiltration.py` 的形态写一个 `@dataclass(frozen=True)` 的 `XxxJudge`，类属性 `objective` 指向新枚举值，`judge()` 返回 `LethalityVerdict`。把该 judge 独有的常量（阈值 / 目标子串）定义成模块级 `_PRIVATE` 常量，不要上升到共享模块。
3. **注册到 REGISTRY**：在 `red_team/oracle/objectives/__init__.py` 的 `REGISTRY` 字典里添加一行 `AttackObjective.XXX: XxxJudge()`，并在 `__all__` 里把类名加上。
4. **补测**：在 `tests/unit/red_team/oracle/test_objectives.py` 里为新 judge 加至少一个 lethal 分支和一个 non-lethal 分支；参数化 `REGISTRY` 的不变式测试会自动覆盖新增 judge。

`lethal()` 分发器无需改动——新 judge 一旦在 REGISTRY 里出现，旧入口自动支持该 objective。

## 6. 反模式

以下写法会破坏 L1 判定层的边界，代码评审应直接拒绝：

- **在 judge 里读 `variant.o` 以外的语义字段**（如 `variant.c["scripts"]` / `variant.d` 的正文），会让 judge 同时承担扫描器的职责，侵蚀 blue team 的边界。判定只应基于 `SensorReading`。
- **在 judge 里对 `variant` 做写改**（任何 `variant.xxx = ...` 或 `variant.metadata.update(...)`）。judge 必须是纯函数；有状态改写归驱动层。
- **在 judge 内部 `import red_team.evolution` / `red_team.campaign` / `red_team.agent`**。反向依赖等于把 L1 判定层拉下水。
- **在 judge 里写「空 evidence 但 lethal=True」**。`LethalityVerdict.__post_init__` 会直接抛 ValueError；若触发该分支说明规则写错了，必须回到规则本身。
- **跨 judge 共享可变状态**（module-level `list` / `dict` 反复 append）。每个 judge 的私有常量用 `tuple` / 字面量，保持 frozen 语义。
- **用 `S1/S2/S3` 之类的抽象编号命名 judge 或规则 id**。`rule_id` 必须是人类可读的 dotted 标识（例：`data_exfil.canary_read_or_leak`）。
