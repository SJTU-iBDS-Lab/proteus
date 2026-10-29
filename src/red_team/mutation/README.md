# red_team.mutation 模块

## 1. 定位

本模块是红队 mutation 策略的**接口层**——L2 抽象。它回答的是
"**造一个** variant 的策略长什么样"。至于"一轮造**几个**"、"不同
strategy 怎么组合"、"什么时候停"——那是 evolution / campaign 的职责
(L3)，不归这一层。

Wave 3 只交付骨架：`MutationStrategy` Protocol + `MutationContext` /
`MutationResult` 两个 DTO。下一波会把现有的四类策略（tool-use agent、
random、zeroshot、blackbox）全部搬进来作为具体实现，见第 5 节。

| 层 | 职责 | 举例 |
| :-- | :-- | :-- |
| L3 evolution | 决定造几个、怎么组合、预算怎么分 | `stages/evasion.py` |
| **L2 mutation** | **造一个 variant 的策略接口** | **本模块** |
| L1 核心 | 不变量、DTO、io | `core.schemas.SkillVariant` |

## 2. 对外接口

### base.py — 2 个 DTO + 1 个 Protocol

| 符号 | 角色 | 约束 |
| :-- | :-- | :-- |
| `MutationContext` | 给 strategy 的只读输入包 | `frozen=True, slots=True`；`search_space` 必须是语义名（见反模式） |
| `MutationResult`  | strategy 的产出包        | `frozen=True, slots=True`；`variant=None` 表示 strategy 放弃；`backend` 纯字符串，不走枚举 |
| `MutationStrategy`| 造一个 variant 的 Protocol | `runtime_checkable`；要求 `name: str` + `mutate(ctx) -> MutationResult` |

`MutationContext` 字段：

| 字段 | 类型 | 说明 |
| :-- | :-- | :-- |
| `seed` | `SkillVariant` | 作为变异出发点的 variant |
| `findings` | `tuple[PaperFinding, ...]` | 上一轮蓝队 findings；空轮用 `()` |
| `repo` | `RepositorySlice` | 宿主 repo 上下文 |
| `search_space` | `str` | `"evasion"` / `"path_expansion"` / `"surface_expansion"` |
| `budget_usd` | `float=50.0` | 本次 mutate 的预算上限 |
| `max_tool_calls` | `int=10` | Tool-use agent 的调用次数上限 |
| `target_model` | `str="glm-5"` | 受害者模型 id |
| `extra` | `dict[str, Any]` | 策略私有参数出口 |

`MutationResult` 字段：

| 字段 | 类型 | 说明 |
| :-- | :-- | :-- |
| `variant` | `SkillVariant \| None` | `None` 表示 strategy 失败 |
| `tool_chain` | `tuple[str, ...]` | 调用的工具名序列；baseline 可空 |
| `stop_reason` | `str` | `submitted` / `budget` / `max_tools` / `refusal` / `invariant_violation` / `programmatic` |
| `cost_usd` | `float` | 真实花销（USD） |
| `backend` | `str` | `tool_use_agent` / `random` / `zeroshot` / `blackbox` |
| `extra` | `dict[str, Any]` | 策略私有产出 |

## 3. 依赖方向

```
red_team.evolution.stages.*   ──┐
red_team.campaign.*           ──┼──> red_team.mutation ──> core.schemas + core.models + stdlib
(未来) red_team.mutation.strategies.* ──┘
```

- 本模块**只**依赖 `core.schemas` / `core.models` + stdlib
  （`dataclasses`、`typing`）。
- 调用方（evolution stages、campaign runner）单向依赖本模块，反向依赖
  禁止——strategy 不许 `import red_team.evolution` / `import
  red_team.campaign`。
- 红队其他既有模块（agent / baselines）当前**不**强制依赖本模块；等
  下一波搬 agent/runtime.py 和 baselines/ 时再统一切过来，保持 Wave 3
  纯 additive。

## 4. 典型调用路径

evolution 的 evasion stage 调本模块、读写 log：

```python
from red_team.mutation import MutationContext, MutationStrategy

def run_one_evasion_round(strategy: MutationStrategy, seed, findings, repo):
    ctx = MutationContext(
        seed=seed,
        findings=tuple(findings),
        repo=repo,
        search_space="evasion",        # 语义名；S1/S2/S3 会 raise
    )
    result = strategy.mutate(ctx)      # 造一个 variant
    if result.variant is None:
        return None                    # refusal / budget / invariant
    return result                      # 上层写 campaign log + 送 oracle
```

## 5. 如何扩展

新增一个策略（例如 `tree_search`）的步骤：

1. 在 `src/red_team/mutation/strategies/<name>.py` 新建实现类（下一波
   建目录；当前骨架只暴露 base）。
2. 类定义必须满足 `MutationStrategy`：
   - 类属性或实例属性 `name: str`（与 `MutationResult.backend` 对齐）；
   - `mutate(self, ctx: MutationContext) -> MutationResult` 方法。
3. 在 `tests/unit/red_team/mutation/` 下加对应测试：
   - 典型输入产出 variant、`stop_reason='submitted'`；
   - 预算打爆 → `variant=None, stop_reason='budget'`；
   - Tool-use agent 要验 `tool_chain` 顺序。
4. 让 evolution 的对应 stage 通过依赖注入拿到这个 strategy；**不要**
   在 stage 里 hardcode 类名，用 config 选择。

## 6. 反模式

- 禁止在 strategy 里直接写日志——落日志归 campaign；strategy 只返回
  `MutationResult`，上层按 `backend` / `cost_usd` / `stop_reason` 做
  账本。
- 禁止 strategy 之间互相调用；需要"复合策略"时，在 evolution stage 里
  顺序调多次 `strategy.mutate(ctx)`，不在 strategy 内部递归。
- 禁止 strategy 反向 `import red_team.evolution` / `import
  red_team.campaign`——单向依赖必须守住。
- 禁止在 `MutationContext.search_space` 使用 `S1` / `S2` / `S3` 等位置
  代号，只接受 `evasion` / `path_expansion` / `surface_expansion`；
  构造期直接 raise。
- 禁止在 `MutationContext` / `MutationResult` 里塞可变状态或 IO；DTO
  只做不变量校验，不做落盘、不做 LLM 调用。
- 禁止给 `MutationResult.backend` 加 Enum 守卫——保持纯 `str`，新策略
  加入时不必改本模块。
