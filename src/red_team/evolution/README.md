# red_team.evolution — Algorithm 1 的三层演化

## 定位

本包实现论文 §4.3 / Algorithm 1 的演化循环：三层搜索空间 + 两类横向扩展的工程骨架。

```
evasion            ── 仅 d 可变          （paper §4.3 aka S1）
path_expansion     ── c / κ / d / chain  （paper §4.3 aka S2，需家族枚举）
surface_expansion  ── 全部字段可变       （paper §4.3 aka S3，需 shared_motif 2/3 命中）
```

本包**只**定义"什么样的 child 相对 parent 是合法的"、"家族枚举怎么筛 top-k"、"surface ideation 怎么选 motif"。具体"这一轮花多少钱、怎么落盘、ablation 开关怎么翻译"归 `campaign/`；"这一轮造出一个 variant"归 `agent/` + `mutation/`；"这一个 variant 该不该算成功"归 `oracle/`。

## 对外接口

```python
from red_team.evolution import (
    SearchSpace,            # @dataclass(frozen=True); level + mask() + validate()
    InvariantViolation,     # child 逃出 mask → 抛这个
    shared_motif,           # (parent, child, metadata) -> bool; 2/3 轴命中
)

from red_team.evolution.evasion import step_s1
from red_team.evolution.path_expansion import path_expand, PathExpandResult
from red_team.evolution.surface_expansion import surface_expand, SurfaceFunnel
from red_team.evolution.horizontal_expand import horizontal_expand, HorizontalExpansion
from red_team.evolution.family_enumeration import (
    enumerate_family,       # 3 阶段：brainstorm → filter → score
    FamilyCandidate,
)
from red_team.evolution.driver import run_evolution    # 待 W5 搬去 campaign
```

`SearchSpace` 方法：

```python
SearchSpace(level="evasion").mask()         # dict: {"frozen": [...], "mutable": [...]}
SearchSpace(level="evasion").validate(parent, child)  # 违规 → InvariantViolation
```

`SearchSpace.level` 接收**语义名**；传入 `"S1"`/`"S2"`/`"S3"` 会被 `_normalize_level` 映射并发出一条 `DeprecationWarning`（向后兼容）。

## 依赖方向

```
red_team.campaign       ──┐
red_team.experiments    ──┼──► red_team.evolution
other top-level drivers ──┘
                              │
                              ├─► red_team.agent.runtime         （evasion 委托给 agent_run）
                              ├─► red_team.mutation              （W5 之后，目前走 agent_run）
                              ├─► red_team.oracle.lethality      （driver 用，横向扩展里不直调）
                              ├─► red_team.baselines.random      （ablation.tool_use_off）
                              ├─► red_team.log.*                 （落盘）
                              ├─► llm                             （family / surface LLM 调用）
                              └─► core.schemas / core.models / sandbox.honeypot
```

**禁止**（本包 → 上层）：

- `import red_team.campaign.*`——campaign 单向依赖 evolution。
- 在 `search_space.py` 里 `import red_team.agent` / `import red_team.oracle`——search-space 是 L1 不变量，不得被业务层污染。

`driver.py` 是历史遗留的顶层编排入口（W5-driver-split 会搬到 `campaign/runner.py`）。在迁移完成前，`driver.py` 会 import `campaign.io`，那是下行迁移的过渡，属于已登记的 W5 临时状态。

## 典型调用路径

```python
from red_team.evolution import SearchSpace, InvariantViolation
from red_team.evolution.evasion import step_s1

# 合法性校验（L1 不变量）
try:
    SearchSpace(level="evasion").validate(parent, child)
except InvariantViolation as exc:
    drop_variant(child, reason=str(exc))

# 一轮 evasion（委托给 agent_run；search_space 用 driver 现有的 "S1" 别名）
result = step_s1(
    parent_variant=parent,
    findings=blue_findings,
    repo=repo,
    target_model="glm-5",
    budget_usd=5.0,
    attack_log_path=attack_log_path,
    ablation={"memory_enabled": True},
    round_index=t,
    agent_run=default_agent_run,   # 依赖注入，测试可替换
)

# 横向扩展（仅在某轮 bypass + lethal 后触发）
expansion = horizontal_expand(
    parent=result.variant,
    findings=blue_findings,
    repo=repo,
    ablation={"disable_expansion": False},
    # ... path_expand_fn / surface_expand_fn 可注入
)
```

## 如何扩展

加一层新的搜索级别（假设叫 `deep_mutation`）：

1. 在 `search_space.py` 的 `_LEVEL_SPECS` 表里加一行（frozen 字段列表 + mutable 字段列表）。
2. 在 `SearchSpace.validate` 里针对新 level 加 invariant branch（大概率只是走 mask diff，不需要新代码）。
3. 在 `evolution/` 下新建同名模块（`deep_mutation.py`），对外暴露 `deep_mutate(parent, ...)`，签名向 evasion/path_expansion 看齐。
4. `search_space_rules_text`（在 `agent/prompts.py`）补一条分支，告诉 LLM 新级别的 mask。
5. `campaign/runner.py` 按 ablation 开关编排，不要在本包内直接写"如何调度"。
6. 测试：`tests/unit/red_team/evolution/test_search_space.py` 参数化新 level 的 invariant 测试；新模块自己写 unit + integration。

## 反模式

- **在 `validate` 里放语言模型调用或 I/O**——`SearchSpace` 是纯函数 + frozen dataclass，副作用归 driver / stage。
- **在 stage 模块（evasion / path_expansion / surface_expansion）里 `json.dumps` 写日志**。写盘必须经 `campaign/io.py` + `log/*`，本包只返回结构体。
- **跨 stage 互相 import**（例如 `surface_expansion` 里 `from red_team.evolution.path_expansion import path_expand`）。顺序由 `horizontal_expand` 编排；同级 stage 必须平行，不形成依赖链。
- **新代码里直接把 `search_space="S2"` 写死**。对外一律用语义名；向 `agent_run` 转发时 `driver.py` / baseline 适配器会翻译——新模块不得引入新的翻译点。
- **在 `family_enumeration.py` 或 `surface_expansion.py` 里写 `docker exec`**。这两层只做 ideation + filter + score；跑沙箱的是调用方。
