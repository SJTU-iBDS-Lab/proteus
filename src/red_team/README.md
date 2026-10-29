# red_team — 红队总导览

## 定位

本包是 CLAW 攻击侧的全部实现。它把论文 §4 描述的三件事合成一条单向的管道：

1. 在 `SkillVariant` 的 `(o, τ, c, κ, d, chain)` 搜索空间里**变异**出一条候选（`mutation/` + `agent/`）。
2. 按 Algorithm 1 的三层（evasion / path_expansion / surface_expansion）**演化**这条候选（`evolution/` + `campaign/`）。
3. 把候选打包成可执行 payload 落到沙箱里，并由 oracle 判 lethal（`payloads/` + `oracle/`）。

整条流水线的事件流全部由 `log/` 落盘，不在任何 runner 里内联 `json.dumps`。

每个子模块只负责**自己那一层的抽象**——本文件不展开具体 API，请看子模块 README。

## 对外接口

上层（`experiments/`、`eval/`、论文分析脚本）只应从以下入口调红队：

```python
from red_team.campaign.runner import run_evolution   # Algorithm 1 主入口
from red_team.evolution import SearchSpace, InvariantViolation
from red_team.mutation import MutationStrategy, MutationContext, MutationResult
from red_team.oracle import lethal, REGISTRY as ORACLE_REGISTRY
from red_team.payloads.registry import REGISTRY as PAYLOAD_REGISTRY, get, by_objective, by_layer
from red_team.log import (
    CampaignLogRow, AttackLogRow, SummaryRow,
    campaign_log_path, attack_log_path, summary_path,
)
```

**不建议**直接 import 的私有面：

- `red_team.agent.tools.*` 的内部常量（`_PREVIEW_LIMIT` 等），除 `TOOLS` / `TOOL_NAMES` / `_dispatch_tool` 外。
- `red_team.evolution.driver._*`、`red_team.campaign.io._*`。
- `red_team.baselines.*`——由 `mutation/strategies/*` 适配后暴露，直接 import 绕过了语义名翻译。

## 依赖方向

子模块的分层严格自顶向下，**禁止反向**：

```
campaign     （顶层编排：I/O、runner、ablation）
   │
evolution    （Algorithm 1 三层 + 家族/表面枚举）
   │
mutation     （L2 策略接口：MutationStrategy / MutationContext / MutationResult）
   │
agent        （单轮 LLM tool-use loop + prompts）
   │
oracle       （L1 判定：LethalityVerdict）
   │
log          （日志 DTO + 路径函数 + BM25 检索）
   │
payloads     （叶子弹药库：doc / bin plugins）
   │
sandbox ──► llm ──► core.schemas
```

`import-linter` 的 `红队分层禁反向` contract 在 CI 中逐条强制。任何"我这层就是想调一下上面那个工具函数"都要被当作架构信号处理——通常意味着抽象被切错了位置。

## 典型调用路径

```python
from pathlib import Path
from red_team.campaign.runner import run_evolution
from red_team.log.paths import run_root

run_id = "exp-42"
outcome = run_evolution(
    run_id=run_id,
    seed_variant=seed,
    rounds=20,
    repo=repo_slice,
    target_model="glm-5",
    budget_usd=50.0,
    ablation={"search_space": "evasion"},   # 语义名，非 S1/S2/S3
)

# 全部落盘：outputs/evolution/<run_id>/{log.jsonl, attack_log.jsonl, summary.json}
for path in run_root(run_id).iterdir():
    print(path.name, path.stat().st_size)
```

## 如何扩展

加一种新的**变异策略**：走 `mutation/` 的 "如何扩展" 清单，不要直接改 `agent/runtime.py`。

加一种新的**攻击目标（objective）**：走 `oracle/` 的 "如何扩展" 清单；`payloads/` 会通过 `by_objective` 自动接住新枚举。

加一种新的**日志流**：走 `log/` 的 "如何扩展" 清单；不要在 `campaign/io.py` 里新拼 dict。

加一种新的**演化阶段**：先在 `docs/` 的 impl 文档里描述清楚 ablation 开关，再在 `evolution/` 下新建一个并列于 `evasion.py` / `path_expansion.py` / `surface_expansion.py` 的模块，最后由 `campaign/runner.py` 选择性编排——不要把新阶段塞进现有阶段的 helper 里。

## 反模式

- **跨层直调**。例如在 `oracle/objectives/*.py` 里 `from red_team.agent.tools import get_binary`，或在 `payloads/doc/*.py` 里 `from red_team.evolution.driver import run_evolution`。reverse-dependency 一出现，CI 的 import-linter 就会 fail。
- **用 `S1` / `S2` / `S3` 作为新代码的标识符**。论文的位置代号只在向后兼容 shim（`_LEGACY_LEVELS`、baseline.random_mutation）里保留，语义名 `evasion` / `path_expansion` / `surface_expansion` 是唯一允许写在新 DTO / 新公共 API / 新日志字段上的词。docstring 里写 "paper §4.3 aka S1" 作为注释是 OK 的，把 `S1` 塞进 `CampaignLogRow.stage` 或 `MutationContext.search_space` 会 raise。
- **在叶子模块里做 I/O 或 LLM 调用**。`payloads/` / `oracle/` / `log.schema` 都是纯数据 / 纯函数，违反这一点就会把测试拖进 docker 与代理依赖。
- **在 campaign / evolution 里 `json.dumps` 到文件**。所有落盘必须走 `red_team.log.campaign_log.write_row` / `summary.write_summary` / `attack_log.append`，不然 schema 漂移没人能审计。
- **绕过 search-space 校验直接构造 child variant**。所有变异必须经 `red_team.agent.mutator.build_mutation_from_llm_output` 或 baseline 自身的 `_build_candidate`，这两条路径都会过 `SearchSpace.validate`。
