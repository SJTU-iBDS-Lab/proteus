# red_team.agent — 单轮红队 Agent

## 定位

本包实现论文 §4.2 描述的红队 agent：一个带 OpenAI function-calling 的 tool-use LLM 循环，**一次调用** = **一轮变异**。

它把下面的事情收敛到一个稳定 runtime：

- 连接 LiteLLM 代理（`llm.LiteLLMClient`）；
- 把 6 个工具 schema 暴露给 LLM（`tools/`）；
- 维护轮内 message history、retrieval-first 不变量、预算 / turn 账本；
- 在 LLM 触发 `submit_mutation` 时，把参数交给 `agent.mutator` 做搜索空间校验，产出 `SkillVariant`；
- 把全部思考过程（tool_calls + thinking）原样喂给 `red_team.log.attack_log.append`，供下一轮 BM25 检索。

本包**不**负责：决定这一轮是 evasion 还是 path_expansion（那是 `evolution/` 的事），也不负责跨轮编排（那是 `campaign/` 的事）。agent 只认 `search_space: str` 参数。

## 对外接口

```python
from red_team.agent import (
    SYSTEM_RED_TEAM,            # str  —— 系统提示词常量
    ROUND_USER_TEMPLATE,        # str  —— 每轮 user 消息模板
    MUTATION_OPERATORS,         # tuple[str, ...]  —— prompt 里列出的合法 operator
    render_round_user,          # (variant, findings, hints, round_index, ...) -> str
    render_findings_block,      # (findings) -> str
    c_summary_from_variant,     # (variant) -> str
    search_space_rules_text,    # (level: "evasion"|"path_expansion"|"surface_expansion") -> str
    truncate_d,                 # (d: str, limit: int = 2000) -> str
    TOOLS,                      # tuple[dict, ...]  —— 6 个 OpenAI function schema
    TOOL_NAMES,                 # tuple[str, ...]   —— TOOLS 对应名字
    _dispatch_tool,             # (name, args, seed, findings, repo, attack_log_path) -> dict
)

from red_team.agent.runtime import agent_run, AgentRunResult
from red_team.agent.mutator import build_mutation_from_llm_output, InvariantViolation
from red_team.agent.attack_log import append, search          # W2 shim，转发到 red_team.log
```

`agent_run` 签名（摘主要字段）：

```python
def agent_run(
    *,
    seed: SkillVariant,
    findings: list[PaperFinding],
    repo: RepositorySlice,
    search_space: str,                # "evasion" / "path_expansion" / "surface_expansion"
    target_model: str,
    budget_usd: float = 50.0,
    max_tool_calls: int = 10,
    attack_log_path: Path | None = None,
    ablation: dict[str, Any] | None = None,
    round_index: int = 0,
) -> AgentRunResult: ...
```

`tools/` 子包的 API 由 `tools/README.md` 详细描述（6 个工具 + `AgentTool` Protocol）。

## 依赖方向

```
red_team.agent ──► red_team.agent.tools      （包内）
               ├─► red_team.agent.prompts    （包内）
               ├─► red_team.agent.mutator    （包内）
               ├─► red_team.agent.attack_log （shim → red_team.log.attack_log）
               ├─► red_team.baselines.base   （AgentRunResult + estimate_cost）
               ├─► red_team.payloads         （仅在 tools.get_binary 里懒 import）
               ├─► llm                        （未来统一走 LLMClient）
               └─► core.schemas / core.models / stdlib
```

**禁止**（本包 → 上层）：

- `import red_team.evolution`、`red_team.campaign`、`red_team.mutation` 均视为回环。`search_space` 字符串参数足够——不需要反过来拿 `SearchSpace` 对象。
- `import blue_team.*`——红队 agent 不读蓝队代码，只读蓝队 `PaperFinding` 列表（已在 `core.schemas`）。

## 典型调用路径

```python
from red_team.agent.runtime import agent_run

result = agent_run(
    seed=parent_variant,
    findings=blue_findings,
    repo=repo_slice,
    search_space="evasion",     # 语义名
    target_model="glm-5",
    budget_usd=5.0,
    max_tool_calls=8,
    attack_log_path=Path("outputs/evolution/exp-42/attack_log.jsonl"),
    ablation={"memory_enabled": True},
    round_index=3,
)

if result.stop_reason == "submitted":
    child_variant = result.variant            # 已过 SearchSpace.validate
else:
    # budget / max_tools / refusal / invariant_violation / programmatic_fallback
    ...
```

## 如何扩展

加新 operator 或新 prompt 段：在 `prompts.py` 修改常量；`MUTATION_OPERATORS` 更新后配对 update `tests/unit/red_team/agent/test_prompts.py`。

加新工具（**先过论文审查**，参见 `tools/README.md`）：走 tools 子包的 5 步清单；不要改 `runtime.py`，dispatcher 是位置无关的。

加新 ablation：在 `runtime.py` 的 ablation 字典里加 key，文档化语义，加单测。**不要**在 `agent_run` 签名里加新关键字参数——会污染所有调用方；复用 `ablation: dict` 出口。

替换 LLM provider：全走 `llm.LLMClient` 协议。`runtime._make_client` 的 `LITELLM_URL` / `LITELLM_KEY` fallback 是 Wave 2 之前的过渡；Wave 2 把 `_make_client` 迁到 `llm/`。

## 反模式

- **在 `agent_run` 里直接 `json.dump` 到 `attack_log.jsonl`**。所有落日志必须经 `red_team.log.attack_log.append`；`red_team.agent.attack_log` 只是转发 shim。
- **把 `search_space` 硬编码成 `"S1"` / `"S2"` / `"S3"`**。用语义名；baseline 适配器 (`mutation/strategies/random.py`) 是唯一允许出现 `_SEMANTIC_TO_LEGACY` 字典的地方。
- **在工具 dispatch 里 `import red_team.evolution` 或 `red_team.campaign`**——立刻产生回环。
- **把 `submit_mutation` 以外的工具也标 `terminal=True`**。终止契约只允许一条；其他工具必须返回 JSON-serializable dict 继续 loop。
- **在 `runtime.py` 里自己 parse `ChatCompletion` 的细节**。`llm.LLMClient.chat` 约定原样返回上游对象；response shape 的变化必须在 `llm/` 层隔离，别让 `runtime.py` 跟着漂。
