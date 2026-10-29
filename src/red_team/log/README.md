# red_team.log 模块

## 1. 定位

本模块是红队所有日志的**唯一出口**。红队运行期产出三条日志流，
全部落在 `outputs/evolution/<run_id>/` 下：

| 流 | 物理文件 | 角色 |
| :-- | :-- | :-- |
| campaign log | `log.jsonl` | 每轮事件流（evolution driver 写入） |
| attack log | `attack_log.jsonl` | 跨轮记忆，供 BM25 检索 |
| summary | `summary.json` | 运行终态汇总 |

本模块锁死 **schema** 与 **路径**，任何写入方都必须经过这里的 DTO
和路径函数；禁止手拼字符串路径或直接 `json.dump` 业务结构体。

Wave 1 只交付骨架（DTO + 路径）。Wave 2 会把
`agent/attack_log.py` 的 append + BM25 逻辑搬到 `campaign_log.py`
/ `attack_log.py`，外部接口不变。

## 2. 对外接口

### schema.py — 3 个 DTO

| DTO | 对应文件 | 说明 |
| :-- | :-- | :-- |
| `CampaignLogRow` | `log.jsonl` 一行 | `stage` 必须是语义名（见反模式） |
| `AttackLogRow` | `attack_log.jsonl` 一行 | `findings_summary` 是 BM25 的输入文本 |
| `SummaryRow` | `summary.json` 整体 | 终态；只落一次 |

所有 DTO 都是 `frozen=True, slots=True`，构造后不可改。

### paths.py — 5 个路径函数

| 函数 | 返回 |
| :-- | :-- |
| `run_root(run_id, root=None)` | `<root>/<run_id>/` |
| `campaign_log_path(run_id, root=None)` | `<root>/<run_id>/log.jsonl` |
| `attack_log_path(run_id, root=None)` | `<root>/<run_id>/attack_log.jsonl` |
| `summary_path(run_id, root=None)` | `<root>/<run_id>/summary.json` |
| `mutation_dir(run_id, root=None)` | `<root>/<run_id>/mutations` |

`root` 默认 `outputs/evolution/`，测试里可传 `tmp_path` 覆盖。

## 3. 依赖方向

```
red_team.campaign.runner  ──┐
red_team.agent.loop       ──┼──>  red_team.log  ──>  stdlib + core.schemas
red_team.evolution.*      ──┘
```

- 本模块**只**依赖 stdlib（`dataclasses`、`pathlib`、`typing`）
  以及必要时的 `core.schemas`。
- 调用方（campaign runner、agent loop、evolution driver）单向依赖
  本模块，反向依赖禁止。
- 跨模块共享的枚举（比如 attack objective）放 `core.schemas/`，不要
  在这里重新定义。

## 4. 典型调用路径

### 写一行 campaign log（Wave 2 之后会直接调本模块的 append）

```python
from red_team.log import CampaignLogRow, campaign_log_path

row = CampaignLogRow(
    run_id=run_id,
    round=current_round,
    stage="evasion",                 # 语义名；S1/S2/S3 会 raise
    is_expansion=False,
    timestamp=_now_iso(),
    seed_id=variant.metadata["seed_id"],
    repo_id=repo.id,
    target_model="claude-sonnet-4-5",
    bypass=verdict.bypassed,
    lethal=verdict.lethal,
)
path = campaign_log_path(run_id)     # outputs/evolution/<run_id>/log.jsonl
# Wave 2 会引入 campaign_log.append(path, row)；当前骨架只提供 DTO + path
```

### 读 attack_log（Wave 2 之后会直接调本模块的 search）

```python
from red_team.log import attack_log_path

path = attack_log_path(run_id)
# Wave 2 之前继续用 red_team.agent.attack_log.search(path, query)
# Wave 2 之后改为 red_team.log.attack_log.search(path, query)，签名不变
```

## 5. 如何扩展

新增一类日志（例如探针日志 `ProbeLogRow`）的步骤：

1. 在 `schema.py` 增加一个 `frozen` dataclass，列齐所有字段，写好 docstring。
2. 在 `paths.py` 增加 `probe_log_path(run_id, root=None)`。
3. 在 `__init__.py` 的 `__all__` 中导出新符号。
4. Wave 2 完成后，在 `campaign_log.py` / 对应文件里加 append / read 函数。
5. 在 `tests/unit/red_team/log/` 加对应测试：类型、frozen、路径约定。

**不要**在其他模块定义新日志格式；所有日志 DTO 都必须进入本模块。

## 6. 反模式

- 禁止在其他模块直接 `json.dump(dict, open(...))` 落日志——必须过 DTO
  + 本模块提供的写入函数（Wave 2 上线）。
- 禁止用 f-string 拼 `outputs/evolution/<run_id>/...`——必须调
  `paths.py` 的函数；否则路径 schema 变更时漏改。
- 禁止在 `CampaignLogRow.stage` 使用 `S1` / `S2` / `S3` 等位置代号；
  只接受 `evasion` / `path_expansion` / `surface_expansion`。
- 禁止在 DTO 里塞业务逻辑；DTO 只做不变量校验（构造期），不做 IO。
- 禁止在本模块引入 `rank_bm25`、`httpx` 等重依赖——保持 stdlib only。
  Wave 2 搬 BM25 索引逻辑时，依赖放到 `attack_log.py` 单文件里隔离。

## W2 补充：attack_log 实现层

Wave 2 把原先散落在 `src/red_team/agent/attack_log.py` 的 append + BM25
检索代码整体迁移进本模块，所有对外签名保持不变。

### 新布局

```
src/red_team/log/
├── schema.py               # DTO（W1）
├── paths.py                # 路径函数（W1）
├── attack_log.py           # W2：append / read_all / search / row_to_document
└── retrieval/              # W2：可插拔检索后端
    ├── base.py             #   Retriever Protocol
    └── bm25.py             #   BM25Retriever（默认实现）
```

### 兼容策略

- `src/red_team/agent/attack_log.py` 现为**转发 shim**，只做 re-export；
  历史调用方 `from red_team.agent.attack_log import append, search`
  继续可用，不需要改动。
- 新代码请直接 `from red_team.log.attack_log import append, search`，
  或从包根 `from red_team.log import append, search`。
- `append()` 同时接受老的 `dict` 和新 `AttackLogRow` DTO；DTO 序列化时
  `tool_chain` 元组会被转成 list，其余字段直接走 `asdict()`。
- 字段兼容：`attack_log.jsonl` 的历史行 schema（`bypassed` / `reason` /
  `reason_codes_evaded` / `tool_chain_summary`）与 Wave 1 DTO
  （`bypass` / `findings_summary` / `tool_chain`）**并存**。老写入
  路径继续写老 schema；新写入路径直接传 `AttackLogRow` DTO 写新 schema。
  `search()` / BM25 只读老字段（`reason` / `reason_codes_evaded` /
  `tool_chain_summary`），DTO 行不会被索引。升级计划见下一节。

### retrieval/ 子包：如何加新后端

`retrieval/base.py` 定义 `Retriever` Protocol：

```python
class Retriever(Protocol):
    def search(self, query: str, rows: list[dict[str, Any]], k: int) -> list[dict[str, Any]]: ...
```

新增一个检索后端（例如 dense embedding）的步骤：

1. 在 `retrieval/` 下新建 `<backend>.py`，实现一个类满足
   `Retriever` 协议（`isinstance(obj, Retriever)` 返回 True）。
2. 不要写入全局可变状态；若需要缓存，keyed by `(path, mtime_ns)` 的
   缓存放在 `attack_log.py` 的 `_CORPUS_CACHE` 层面。
3. 在 `retrieval/__init__.py` 中导出新类。
4. 在 `tests/unit/red_team/log/` 下加对应测试：空语料 / top-k 截断 /
   无匹配 query 返回子集 / 排序顺序符合预期。
5. 需要切换默认检索器时，改 `attack_log.py` 顶部的
   `_DEFAULT_RETRIEVER`；不要暴露全局 setter。

### 缓存语义

`attack_log.search()` 在 `_CORPUS_CACHE` 中按 `(绝对路径, mtime_ns)`
缓存整棵语料 + 索引，文件一旦重写（mtime 变化）缓存自动失效。
同一个 run 内多次 search 命中缓存，零 I/O。

## W3 补充：campaign_log 和 summary

Wave 3 把原先内联在 `src/red_team/evolution/driver.py` 里的 `log.jsonl`
与 `summary.json` 写入逻辑抽到本模块，driver.py 不再直接 `json.dumps`
到文件，只改写入**通道**，**行字段 schema 完全不动**（byte-equal）。

### 写入层职责

```
src/red_team/log/
├── campaign_log.py    # W3：now_iso / write_row / read_rows  → log.jsonl
└── summary.py         # W3：write_summary / read_summary     → summary.json
```

`campaign_log.write_row` 和 `summary.write_summary` 都接受两种输入：

- **dict**（主通道，向后兼容）—— driver.py 当前写入的超集 schema。
  `dict(row)` 保序 → `json.dumps(..., ensure_ascii=False, default=str)` →
  和旧代码 byte-equal。
- **DTO**（推荐新路径）—— 传入 `CampaignLogRow` / `SummaryRow`，
  内部走 `asdict()`；目前仅作为可选通道开放，driver.py 继续写 dict，
  论文分析脚本继续 diff 不到格式变化。

### dict-as-row vs DTO-as-row 兼容期策略

driver.py 写 `log.jsonl` 的 dict 字段（`variant_hash` / `o` / `tau` /
`chain` / `bypassed` / `lethal` / `backend` / `phase` / `chain_original`
/ `chain_used` / `hp` …）是 Wave 1 `CampaignLogRow` DTO 的**严格超集**。
如果强行让 driver.py 切 DTO，要么把这些字段塞进 `extra`（DTO 语义变糊），
要么给 DTO 加十几个字段（把 driver 实现细节外推进 log 层）。两条路都
违反"data structures first / 每层 own 自己的抽象层级"。

因此 **W3 不切 DTO**：
- driver.py 通过 `campaign_log.write_row(path, dict_row)` 写原有 schema；
- 未来某个 Wave 若有独立模块想写 campaign log，可以直接传 DTO
  —— 两种通道并存，写入层统一。

### 示例调用

```python
from red_team.log import campaign_log, summary as summary_mod
from red_team.log.paths import campaign_log_path, summary_path

# 每轮一行 JSON（driver.py 现在走这条路）
campaign_log.write_row(
    campaign_log_path(run_id),
    {
        "round": t,
        "variant_hash": variant.content_hash(),
        "bypassed": True,
        "lethal": False,
        "timestamp": campaign_log.now_iso(),
        # ... 其它 driver.py 的超集字段
    },
)

# 终态一次 overwrite
summary_mod.write_summary(
    summary_path(run_id),
    {"run_id": run_id, "rounds_ran": 7, "total_cost_usd": 0.42},
)
```

### 反模式

- 禁止在 driver（或任何调用方）里再 `json.dumps(..., indent=2)` 到
  `summary.json` —— 必须走 `summary.write_summary`，否则 `indent` /
  `ensure_ascii` / `default` 参数会在多个地方漂移。
- 禁止在 driver 里再 `with path.open("a") as f: f.write(json.dumps(...))`
  写 `log.jsonl` —— 必须走 `campaign_log.write_row`。
- 禁止改 driver.py 的 log 行 dict 字段名或顺序（dict 保序意味着
  字节级兜底），要新增字段请先讨论是否升到 DTO。
