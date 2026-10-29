# red_team.agent.tools 工具插件集

## 定位

红队 agent 在每一轮中调用的工具插件集合，对应论文 §4.2 "Red Team:
Self-Evolving Agent" 所约束的封闭工具空间。每个工具是一个独立文件
（一个工具 = 一个模块 + 一个 `AgentTool` 实现类 + 一个 `_SCHEMA`
字典），由包的 `__init__.py` 统一注册为一个只读 tuple。

本目录 **替换** 了旧的单文件 `agent/tools.py`，但对外接口 100% 保持
向后兼容：所有旧的导入路径继续工作。

## 对外接口

包根暴露以下符号（全部在 `__all__` 中）：

| 符号 | 类型 | 说明 |
|------|------|------|
| `TOOLS` | `tuple[dict, ...]` | 6 个 OpenAI function-calling schema，顺序冻结 |
| `TOOL_NAMES` | `tuple[str, ...]` | `TOOLS` 对应的工具名，用于 prompt 里定位枚举 |
| `_dispatch_tool(name, args, seed, findings, repo, attack_log_path)` | function | runtime 调度每个 `tool_calls[*]` 的唯一入口 |
| `AgentTool` | `Protocol` | 插件契约：`schema` + `name` + `terminal` + `dispatch` |
| `_PREVIEW_LIMIT` / `_BENIGN_SAMPLE_SIZE` / `_VALID_REPO_SECTIONS` / `_VALID_SKILL_SPLITS` / `_ENTRY_SCRIPT_SUFFIXES` | constants | 旧 monolith 的共享常量，原位 re-export |

`submit_mutation` 是 **唯一** `terminal=True` 的工具；其 `dispatch`
返回 sentinel `{"_terminal": True, "args": <defensive-copy>}`，由
runtime 转交 `build_mutation_from_llm_output` 做搜索空间校验，本包不碰
`SkillVariant`。

## 依赖方向

```
red_team.agent.tools ──┬── red_team.agent.tools.base          (Protocol + 常量)
                       ├── red_team.agent.tools._data_root    (文件系统 helper)
                       ├── red_team.agent.attack_log          (lazy import, 仅 read_attack_log)
                       ├── red_team.payloads                  (lazy import, 仅 get_binary)
                       ├── core.schemas                       (间接, 通过 mutator)
                       └── stdlib: json / os / random / pathlib / dataclasses / typing
```

`runtime.py` 调度本包，反向依赖不存在；本包 **不** 直接 import
`red_team.evolution.*` 或 `red_team.campaign.*`。`CLAW_DATA_ROOT` 的
解析发生在每次 `_dispatch_tool` 调用时（而非 import 时），因为测试
fixture 的 env var 是在 fixture body 里设置的。

## 典型调用路径

```python
# runtime.py 里一轮 agent 调度（简化）
from red_team.agent.tools import TOOLS, _dispatch_tool

# 1) LLM 收到 TOOLS 作为 function schema
response = client.chat.completions.create(..., tools=TOOLS)

# 2) 对每个 tool_call 同步调用 _dispatch_tool
for call in response.choices[0].message.tool_calls:
    result = _dispatch_tool(
        call.function.name,
        json.loads(call.function.arguments),
        seed, findings, repo,          # 预留上下文
        attack_log_path,               # 仅 read_attack_log 真正用到
    )
    # 3) submit_mutation 的 sentinel 触发终止
    if result.get("_terminal"):
        mutation = build_mutation_from_llm_output(result["args"], ...)
        break
    # 4) 其它工具把 result json.dumps 后回灌为 tool message
    messages.append({"role": "tool", "content": json.dumps(result), ...})
```

## 如何扩展

**警告**：加新工具前先读 `docs/red-team-impl.zh.md` §4.2。
当前论文严格限定红队 agent 的 tool space 为 6 个（5 个检索 / 终止
原语 + `get_binary`）。继续扩展必须先在论文 / 实验设计里明确允许，
否则会破坏 §4.2 的 action-space 声明与实验可复现性。

若论文更新后允许新增工具，按以下步骤：

1. 新建 `src/red_team/agent/tools/<name>.py`，定义 `_SCHEMA` dict
   与 `<Name>Tool` dataclass（实现 `AgentTool` 协议，即 `schema` /
   `name` / `terminal` / `dispatch`）。
2. 在 `__init__.py` 的 `_TOOL_INSTANCES` 中按 **正确位置** 插入实例——
   位置决定 `TOOL_NAMES` 与 prompt 里的编号。
3. 更新 `__init__.py` 开头的 import-time `assert` 检查工具数量
   与 `TOOL_NAMES` 列表。
4. 扩写 `tests/unit/red_team/agent/tools/test_registry.py` 里的计数
   assertion，再补充针对新工具的 dispatcher 行为测试。
5. 若新工具需要新的共享常量，加到 `base.py`；不要在 `__init__.py`
   里塞常量。

## 反模式

- 加 `web_search` / `browse` / `fetch_url` 之类的公网访问工具——
  论文明确禁止；schema 层已保证不会意外引入，请勿绕过。
- 在某个工具的 `dispatch` 内部 import `red_team.evolution` 或
  `red_team.campaign`——会制造 tools ↔ evolution 的双向依赖并破坏
  分层。
- 工具之间互相调用：每个工具必须独立，runtime 负责编排。
- 在 `__init__.py` 里重新实现 tool 逻辑：`_dispatch_tool` 的职责仅
  是 "按名字找到实例、把 args 转过去"，不做任何业务判断。
- 依赖 import-time 的 `_data_root`：所有文件系统访问都必须走
  `_data_root()` 函数，以便测试 fixture 能在 import 之后注入
  `CLAW_DATA_ROOT`。
