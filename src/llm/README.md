# `src/llm/` — 统一 LLM 客户端

## 定位

CLAW 里**所有** LLM 调用的唯一入口（红队 agent + 蓝队 scanner 共用）。此模块不承担任何业务逻辑，仅仅把"指向 LiteLLM 代理的 OpenAI 兼容 chat-completions 客户端"这件事抽成一个结构化协议。

目标是在 Wave 2 把 `src/red_team/agent/runtime.py::_make_client` 和蓝队分散的 `openai.OpenAI(...)` 构造点全部迁移到这里，并为将来直连其他 provider（Anthropic、本地 llama.cpp 等）留出扩展位。

## 对外接口

- `llm.LLMClient`（`Protocol`，`@runtime_checkable`）—— 结构化类型；任何暴露同名 `chat(...)` 方法的类都视为合法实现。
- `llm.LiteLLMClient` —— 默认实现，指向本地 LiteLLM 代理。
  - 构造参数：`base_url: str | None = None`，`api_key: str | None = None`；均可省略，退回到 `LITELLM_URL` / `LITELLM_KEY` 环境变量，最终退回到 `http://127.0.0.1:4000` / `sk-claw-local`。
  - 唯一方法：`chat(*, model, messages, tools=None, tool_choice=None, temperature=0.0, max_tokens=None) -> Any`；原样返回上游的 `ChatCompletion` 对象，不做二次解析。

## 依赖方向

- 本模块**只**依赖 stdlib 与 `openai`（且 `openai` 是**懒 import**，只在首次 `chat(...)` 时触发）。
- 不反向依赖 `src/red_team/` 或 `src/blue_team/`——这是 Wave 1 的硬约束。
- 所有上层模块通过 `llm.LLMClient` 协议依赖本模块，不要直接 `import openai`。

```
  red_team / blue_team  ───►  llm.LLMClient  ───►  llm.LiteLLMClient  ───(lazy)──►  openai
```

## 典型调用路径

```python
from llm import LiteLLMClient

client = LiteLLMClient()  # env-var fallback; no OpenAI import yet
resp = client.chat(
    model="claude-sonnet-4-6",
    messages=[{"role": "user", "content": "hello"}],
    tools=TOOLS, tool_choice="auto",
)
```

注入/替换（单测与多 provider 并存）：

```python
def run(client: LLMClient) -> None: ...
run(LiteLLMClient(base_url="http://proxy:4000", api_key="sk-..."))
```

## 如何扩展

新加一个 provider（例：Anthropic 直连）：

1. 新建 `src/llm/anthropic.py`，在其中实现 `AnthropicClient`，对外暴露同名的 `chat(*, model, messages, tools=None, tool_choice=None, temperature=0.0, max_tokens=None) -> Any`。
2. 只要结构匹配，`isinstance(AnthropicClient(), LLMClient)` 自动为真，调用方无需改动。
3. 把真正的 SDK import 放在一个独立的 `_make_xxx_client` 工厂里，保持懒加载；在 `src/llm/__init__.py` 按需 re-export。

不需要修改 `client.py` 也不需要改任何调用方。

## 反模式

- 不要在上层模块里 hard-code `from openai import OpenAI` + `OpenAI(base_url=..., api_key=...)`——统一走 `LiteLLMClient()`。
- 不要在本模块里做预算追踪 / retry / 工具分发 / 响应解析——这些归调用方（`runtime.py`、scanners 等）。本模块只负责"把 chat-completions 包一层可替换的接口"。
- 不要在 `__init__.py` 或 `client.py` 顶部 `import openai`——懒 import 是硬约束，否则单测环境无法在没有代理的机器上跑起来。
- 不要 `except` 掉上游异常后返回空对象；把原始响应原样交给调用方。
