# red_team/payloads — 红队攻击弹药库

## 1. 定位

本模块是红队的**统一攻击弹药库**，把此前散落的两类 payload 收敛到一处：

- **doc 层** —— 文本注入，来源是历史 `red_team.skill_poisoner.POISON_PAYLOADS` 字典（12 条），用于追加到目标技能的 `SKILL.md`，让被攻击 agent 的 system prompt 里长出一段“必须执行的前置步骤”。
- **bin 层** —— 编译产物，来源是 `red_team.payloads.binary_catalog.BINARY_CATALOG`（6 条 Go 写的原语），每次 render 会 `go build` 出 Linux ELF、base64 嵌入一段 Bash 自解压 wrapper，直接塞进 `variant.c.scripts`。

两层对外看起来是同一种东西：一个 `Payload` 插件 `render()` 出一个 `PayloadArtifact`。下游消费者（sandbox runner、attack log、blue-team reproducer）只依赖这个统一的 DTO。

## 2. 对外接口

```python
from red_team.payloads.base import Payload            # Protocol
from red_team.payloads.registry import (
    REGISTRY,         # dict[str, Payload]，doc + bin 合并视图
    get,              # get(name) -> Payload
    by_objective,     # by_objective(AttackObjective) -> list[Payload]
    by_layer,         # by_layer("doc" | "bin") -> list[Payload]
)
from core.schemas import AttackObjective, PayloadArtifact
```

`Payload` 是 `@runtime_checkable` 的 Protocol，三字段 + 一方法：

| 字段/方法 | 类型 | 约定 |
|-----------|------|------|
| `name` | `str` | 稳定 id，等于注册进 `REGISTRY` 的 key |
| `layer` | `str` | `"doc"` 或 `"bin"` |
| `objective` | `AttackObjective` | 本 payload 推进的攻击目标 |
| `render(**kwargs)` | `PayloadArtifact` | 纯函数渲染 |

`render` 必须是纯函数：**不准**在插件里做文件写入、网络请求或全局状态改动。落盘给 `SKILL.md` 或 `variant.c.scripts` 是调用方的职责。

## 3. 依赖方向

- 依赖 `core.schemas`（`AttackObjective`、`PayloadArtifact`）和 Python 标准库。
- bin/ 子包**内部**复用 `binary_catalog.py` 的 `get_binary()`——Go build + base64 + wrapper 全部复用，不重写。
- doc/ 子包**内部** `from red_team.skill_poisoner import POISON_PAYLOADS`——payload 文本单一事实源仍在 `skill_poisoner`（Wave 4 会反过来，把 `skill_poisoner` 退化成 shim）。
- **禁止反向依赖**：`payloads/` 里绝对不准 `import red_team.evolution`、`red_team.campaign`、`red_team.agent` 等上层模块。
- `red_team.payloads.binary_catalog` 保留为 shim，继续 re-export `BINARY_CATALOG` / `BinarySpec` / `get_binary` / `build_binary` / `build_root` / `list_binary_specs` / `summarize_catalog` / `BinaryBuildError` / `BinaryNotFoundError`，用以支撑既有 `from red_team.payloads import ...` 的调用点。

## 4. 典型调用路径

```python
from red_team.payloads.registry import get, by_objective
from core.schemas import AttackObjective

# doc 用法：取 payload -> 渲染 -> 拿 content 追加到 SKILL.md
doc_plugin = get("data_exfiltration")
art = doc_plugin.render()
assert art.layer == "doc"
# art.content 可直接 append 到 SKILL.md 末尾

# bin 用法：取 payload -> 传 env-var override -> 渲染 -> content 是 bash 脚本
bin_plugin = get("honeypot_beacon")
art = bin_plugin.render(
    HONEYPOT_URL="http://127.0.0.1:9999/collect",
    LABEL="exp-42",
)
assert art.layer == "bin"
# art.content 可直接塞进 variant.c.scripts[art.metadata["filename"]]

# 按目标检索
for plugin in by_objective(AttackObjective.PERSISTENCE):
    ...

# 按层检索（例如只要生成 variant.c.scripts 的那些）
for plugin in by_layer("bin"):
    ...
```

`PayloadArtifact.metadata` 约定：

- doc 层：`{"source": "skill_poisoner.POISON_PAYLOADS", "key": <插件名>}`
- bin 层：`{"binary_id", "filename", "language", "description", "covers_objectives", "params_used"}`（直接来自 `get_binary()` 返回字典，去掉 `content` 后给 metadata）

## 5. 如何扩展

加一条新 payload（4 步走）：

1. **挑层**：纯文本追加到 `SKILL.md` → doc；要落下可执行脚本 → bin。
2. **写插件文件**：在 `payloads/doc/<name>.py` 或 `payloads/bin/<name>.py` 里放一个 `@dataclass(frozen=True)` 的类，三字段（`name` / `layer` / `objective`）+ `render(**kwargs) -> PayloadArtifact`。
   - doc 新增文本：当前 Wave 仍把文本放在 `skill_poisoner.POISON_PAYLOADS`，Wave 4 迁移后会改成在插件里直接写文本常量。
   - bin 新增二进制：先到 `payloads/binaries/<go_package>/main.go` 写 Go 源码，再在 `binary_catalog.BINARY_CATALOG` 加一个 `BinarySpec` 行，最后写插件。
3. **注册进子包 `__init__.py`**：把新类 import 进来并加到该子包的 `REGISTRY` 字典（key 就是插件 `name`）。
4. **写契约测试**：直接由 `tests/unit/red_team/payloads/test_doc_plugins.py` 或 `test_bin_plugins.py` 的 `parametrize` 自动覆盖；跑 `uv run pytest tests/unit/red_team/payloads/ -q` 验证。

## 6. 反模式

**禁止**在插件内做如下事情——任何一条出现就是 bug：

- 在 `render()` 里做文件 I/O、网络调用或 `subprocess.run`（bin 插件的 `go build` 是 `binary_catalog.build_binary` 管的，插件本身只是数据）。
- 在插件里 `import red_team.evolution` / `red_team.campaign` / `red_team.agent.*`——弹药库必须是叶子模块，依赖方向反了就会产生循环。
- `doc/` 和 `bin/` 下用**同名 key**。`registry.REGISTRY` 是 `{**doc, **bin}` 扁平合并，重名会静默丢掉一个，契约测试 `test_doc_and_bin_names_do_not_collide` 会失败。
- 把可变状态挂到插件实例上。插件都是 `@dataclass(frozen=True)`，保持纯数据对象。
- 跨层复用：bin 插件不要去 `import red_team.payloads.doc`，反之亦然。需要联动时，让调用方在上层组装。
