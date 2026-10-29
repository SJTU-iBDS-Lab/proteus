# red_team.campaign

## 1. 模块定位

`campaign` 封装**一个 `run_id` 的完整生命周期**——从 seed 出发，按
Algorithm 1 的 S1 轮次循环直到 bypass+lethal（或用尽 rounds），并
最终写出 `log.jsonl` / `attack_log.jsonl` / `summary.json`。

原 `red_team.evolution.driver` 把四件事耦合在 773 行里：

| 职责 | 拆分位置 |
|------|----------|
| 状态机（`for t` / 门控 / expansion 触发 / stop_reason） | `runner.py` |
| I/O（log 行组装、JSONL 写入、summary 组装、mutation md、chain 快照） | `io.py` |
| Seams（5 个默认依赖注入点：agent/sandbox/blue/oracle/expansion） | `seams.py` |
| Ablations（9 条 A1…A9 的开关分支） | `ablations/<name>.py` |

Wave 5 后每个文件的职责是**正交的**：runner 不知道具体消融，
ablation 不知道 I/O，I/O 不跑业务逻辑。

## 2. 对外接口

```python
from red_team.campaign import run_evolution

run_dir = run_evolution(
    seed,                         # SkillVariant
    repo,                         # RepositorySlice
    *,
    target_model="glm-5",
    mutator_model="claude-sonnet-4-6",
    rounds=20,
    free_expansions=3,
    surface_k=10, surface_k_prime=3,
    ablation=None,                # dict，如 {"tool_use_off": True}
    blue_team="ours",             # or "skillvetter"
    sandbox_container="skillrt-openclaw-00",
    run_id=None,                  # 默认 {seed_id}_{repo_id}_{model}_{hex}
    output_root="outputs/evolution",
    # 5 个注入 seam（测试专用，生产不用传）:
    blue_team_fn=None,
    sandbox_run_fn=None,
    agent_run_fn=None,
    oracle_fn=None,
    horizontal_expand_fn=None,
) -> Path                         # 返回 run_dir
```

向后兼容：`from red_team.evolution.driver import run_evolution` 继续
工作——`driver.py` 退化为一行 re-export 的 shim。

## 3. 依赖方向

只允许 `campaign → {evolution.stages, oracle, log, sandbox, baselines,
agent}`。**反向禁止**。

* ✅ `campaign/runner.py` 调 `evolution.evasion.step_s1`（S1 agent）。
* ✅ `campaign/seams.py` 调 `oracle.lethality.lethal`、
  `agent.runtime.agent_run`、`sandbox.honeypot.HoneypotResult`。
* ✅ `campaign/io.py` 调 `log.campaign_log` / `log.summary` / `log.attack_log`。
* ❌ `evolution.*` **不得** import `campaign.*`。旧代码
  （`evolution/driver.py`）现在是 campaign 的 shim，不算反向调用。
* ❌ `oracle/*`、`sandbox/*`、`log/*` **不得** import `campaign.*`。

违反方向的 import 会被 CI 的 import-linter 挡住。

## 4. 典型调用路径

```python
# 1) 准备种子 + 仓库切片
seed = SkillVariant(o=PERSISTENCE, tau=SINGLE, chain=["skill-x"], ...)
repo = RepositorySlice.from_dir("/repos/foo")

# 2) 跑 20 轮，启用 A8（禁止 kappa 编辑）
run_dir = run_evolution(seed, repo, rounds=20,
                        ablation={"forbid_channel_edit": True})

# 3) 读日志 + 摘要
import json
for line in (run_dir / "log.jsonl").read_text().splitlines():
    row = json.loads(line)
    # row: {round, bypassed, lethal, backend, phase, ...}
summary = json.loads((run_dir / "summary.json").read_text())
# summary: {rounds_ran, first_bypass_round, total_bypass, total_lethal, ...}
```

内部流水线（runner.run_evolution 的实际调用顺序）：

```
seams.call_blue_team  → F_t
ablations.apply_feedback_hooks  → agent 的 findings（A9 归零）
runner._run_single_round        → step_s1 或 RandomMutation（A4）
ablations.check_mutation_invariants → A8 可能 raise
ablations.apply_sandbox_rewriters   → A7 truncate chain
io.persist_mutation_md              → 写 SKILL.md 快照
seams.sandbox_run                   → 真 OpenClaw 或 seam stub
oracle_fn(s', hp, o)                → lethal 判定
io.build_log_row + io.write_log_row → 落盘一行
# bypass && lethal:
ablations.is_expansion_enabled (A5) → True 时跑 S2/S3
seams.default_horizontal_expand     → S2 path + S3 surface
io.build_summary + io.write_summary → 收尾
```

## 5. 如何扩展

**加一条新消融**（假设叫 A10 `disable_attack_log`）：

1. 新建 `ablations/disable_attack_log.py`：
   ```python
   def apply(entry: dict, ablation: dict) -> dict | None:
       if ablation.get("disable_attack_log"):
           return None  # 用 None 约定“跳过写入”
       return entry
   ```
2. 在 `ablations/__init__.py` 的 hook 注册表里追加：
   ```python
   from red_team.campaign.ablations.disable_attack_log import apply as _skip
   _ATTACK_LOG_HOOKS.append(_skip)
   ```
3. 在 `runner.py` 的 `_io.append_attack_log` 调用前过一遍
   `_ATTACK_LOG_HOOKS`——**只新增一行，不新增 if 分支**。
4. `tests/unit/red_team/campaign/test_ablations.py` 加单测
   `test_a10_skips_attack_log`。

**加一个新 seam**（假设要注入自定义 oracle 解释器）：

1. 在 `seams.py` 顶层 expose `default_oracle_explainer = ...`。
2. 在 `run_evolution` 签名加 `oracle_explainer_fn=None`。
3. runner 里一行 `oracle_explainer_fn = oracle_explainer_fn or _seams.default_oracle_explainer`。

## 6. 反模式（PR reviewer 请拦截）

* 🚫 **在 `runner.py` 写 `if ablation.get("<name>"):` 针对具体消融**。
  这是拆分前 773 行的老坑——任何分支都必须落在 `ablations/<name>.py`
  里并通过 hook 注册表暴露。
* 🚫 **在 `ablations/<name>.py` 里写业务逻辑**（调 sandbox、读文件、
  query LLM）。ablation 模块必须是**纯变换**：输入 → 输出，零副作用。
* 🚫 **在 `io.py` 里做状态机逻辑**（比如判断 bypass/lethal、维护
  `current_seed`、决定是否 break）。io 只负责 dict ↔ 磁盘。
* 🚫 **在 `seams.py` 里放日志组装**或回调 `io.*`。seam 只转发到真实实现
  或测试 stub。
* 🚫 **从 `evolution.*` 或 `oracle.*` 反向 import `campaign.*`**。campaign
  在依赖链下游，反向调用必须走事件 / 结果对象。
* 🚫 **在新代码里用 `S1/S2/S3` 作为标识符名**——这些字符串仅用于
  日志 `phase` 字段（老测试受约束），代码里用 `evasion` /
  `path_expansion` / `surface_expansion` 这类语义名。
