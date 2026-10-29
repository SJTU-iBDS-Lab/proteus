# sandbox — 受害者执行环境与 sensor 层

## 定位

本模块是**红蓝双方共同依赖的执行基座**：把一个 `(variant, payload)` 丢进 Docker 容器里跑起来，并在事后给出一份结构化的 `HoneypotResult` 作为 sensor 读数。红队的 oracle 依据这份 sensor 判 lethal；蓝队的 dynamic-probe runner 依据同一份 sensor 做 runtime trace summarization。

本模块**只**承担两件事：

1. 以一致的 container 生命周期（preflight → 拉取镜像 → mount skills → exec → teardown）跑一次受害者 agent。
2. 以 canary + pattern 匹配的方式把 runtime 行为落成 `HoneypotResult`（叶子 DTO）。

本模块**不**承担：生成 payload、裁判 lethal、生成 findings、跑 LLM、编排多轮 mutation。这些分别归 `red_team/payloads/`、`red_team/oracle/`、`blue_team/scanners/`、`llm/`、`red_team/campaign/`。

## 对外接口

```python
from sandbox.honeypot import (
    HoneypotConfig,    # 种 canary 的配置（workspace, canary_token）
    HoneypotResult,    # 一次执行后的 sensor 读数（frozen-ish slots dataclass）
    plant_canaries,    # plant_canaries(container, config) -> dict[str, str]
    snapshot_workspace,# snapshot_workspace(container, config) -> None
    check_canaries,    # check_canaries(container, config, ...) -> HoneypotResult
)

from sandbox.openclaw_runner import (
    SandboxConfig,     # 单 skill 容器配置
    SandboxResult,     # 单 skill 运行结果
    preflight,         # preflight(config) -> dict
    run_seed,          # run_seed(...) -> SandboxResult  [主入口]
)

from sandbox.chain_runner import (
    ChainResult,       # 多 skill 链式执行结果
    run_chain,         # run_chain(...) -> ChainResult  [红队实验主入口]
)

from sandbox.openclaw_chain_runner import (
    PoisonedChainResult,
    run_poisoned_chain,  # 已做 doc 注入的 chain 运行
)

from sandbox.skill_vetter_runner import (
    SkillVetterResult,
    run_skill_vetter_audit,  # 蓝队审计 runner
)

from sandbox.materialize import (
    materialize_poisoned_skill,  # copy src skill → dst + append payload
    materialize_poisoned_chain,  # copy chain → chain_root/ + poison 1 SKILL.md
)
```

历史兼容：`from red_team.skill_poisoner import materialize_poisoned_skill, materialize_poisoned_chain` 仍可用——`skill_poisoner` 是纯 re-export shim，绑定的就是 `sandbox.materialize` 里同一个 callable。

所有 `Result` 类型都是 `@dataclass(slots=True)`；`HoneypotResult` 被 `core.schemas.SensorReading` 语义别名化，oracle 层仅依赖该别名，不直接 import 本模块。

## 依赖方向

```
sandbox  ──→  Python 标准库（subprocess / shlex / pathlib / dataclasses）
         └→  red_team.payloads.doc        （已知遗留，参见下一条）
```

**允许**（上层 → 本模块）：

- `red_team.campaign.*`、`red_team.evolution.*`、`red_team.agent.*` 调 `run_seed` / `run_chain` / `run_poisoned_chain`。
- `blue_team.*` 调 `run_skill_vetter_audit`。
- `core.schemas.SensorReading` 以别名方式引用 `HoneypotResult`（仅类型层，不拿行为）。

**禁止**（本模块 → 上层）：

- `import red_team.evolution`、`red_team.agent`、`red_team.mutation`、`red_team.oracle`、`red_team.campaign`——一律视为回环。
- `import blue_team.*`——受害者容器不知道蓝队存在。

**已知遗留**：`sandbox.materialize`（W5 本轮已从 `red_team/skill_poisoner.py` 搬入本包）当前 `from red_team.payloads.doc import POISON_PAYLOADS`，用 `attack_type` 字符串查表拿 payload 文本——属于 `sandbox → red_team.payloads` 的反向依赖，与本包“只依赖 stdlib”的目标冲突。下一轮计划把 `materialize_poisoned_*` 的签名改成只接 `payload_text: str`，由上层调 `red_team.payloads.prompts.build_poisoned_prompt` 先解析再传入，把这条反向 import 彻底去掉。期间 import-linter 的 `sandbox-isolation` contract 把 `red_team.payloads` 从 forbidden 列表里 whitelist 掉，以避免 CI 因这条已登记的遗留失败。

## 典型调用路径

```python
from sandbox.chain_runner import run_chain, HoneypotConfig
from sandbox.openclaw_runner import SandboxConfig

cfg = SandboxConfig(
    container="claw-sandbox-01",
    workspace="/tmp/sandbox/exp-42",
    skills_dir="/host/skills",
)
hp_cfg = HoneypotConfig(workspace=cfg.workspace)  # 自动生成 canary_token

result = run_chain(cfg, hp_cfg, chain=["repo-summary", "doc-compose"])
# result.honeypot: HoneypotResult  -> 交给 oracle.lethal(variant, hp, variant.o)
# result.tool_calls / result.final_text -> 交给 blue_team dynamic probe summarizer
```

## 如何扩展

新增一种 runner（例如 `k8s_runner.py`）：

1. 新建 `src/sandbox/<name>_runner.py`，定义本地 `XxxConfig` / `XxxResult` dataclass。
2. 复用 `honeypot.HoneypotConfig` / `plant_canaries` / `check_canaries` 三件套——不得重造 sensor。
3. 不得 import 任何 `red_team.*` 或 `blue_team.*`；配置项以参数注入。
4. 为新 runner 写 e2e 测试（标 `@pytest.mark.slow`）并在 `tests/unit/sandbox/` 下至少补一条 fast 单测覆盖 `Result` 字段规范。
5. 更新本文件「对外接口」节加入新 `Result` 类型行；不修改既有 runner 的 public 签名。

## 反模式

- **在本模块拼判定逻辑**。例如在 `check_canaries` 里根据 `variant.o` 翻译为 `bool lethal`——判定归 `red_team/oracle/`，sensor 只记录 raw 信号。
- **在 runner 里 `import openai`**。runner 不调 LLM；LLM 调用发生在容器内或上层 agent。
- **在 `HoneypotResult` 字段上挂行为方法**（`judge()` / `to_finding()` / `render()`）——它只是 sensor 读数，行为在上层。
- **用 f-string 拼 docker exec 命令**。所有跨进程字符串必须经 `shlex.quote` 或以 list 形式传给 `subprocess.run`，防止 canary token 里的特殊字符把命令拆开。
- **把 `_copy_to_container` / `_exec_in_container` 从 private 变 public 供上层使用**。跨容器的 I/O 细节不得泄漏给 campaign / agent 层——新增需求请加新公共函数，不要暴露 helpers。
