"""红队 mutation 策略的接口层 —— L2 抽象。

"造一个 variant" 的不同策略（tool-use agent / random / zeroshot / blackbox）
共享这个接口。evolution stages (L3) 调本模块，反向禁止。

MutationContext 是给 strategy 的只读输入包；MutationResult 是产出——
包含 variant 本身和生产过程的元数据（tool_chain / cost / stop_reason）
供上层 campaign 落日志、做预算追踪。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from core.models import RepositorySlice
from core.schemas import PaperFinding, SkillVariant


@dataclass(frozen=True, slots=True)
class MutationContext:
    """Strategy 的只读输入。

    ``search_space`` 必须使用语义名（``evasion`` / ``path_expansion`` /
    ``surface_expansion``）；历史上的位置代号 ``S1`` / ``S2`` / ``S3``
    在构造期直接 raise，避免缩写沿用到新代码路径。
    """

    seed: SkillVariant                  # 作为变异出发点的 variant
    findings: tuple[PaperFinding, ...]  # 上一轮的蓝队 findings（供反馈学习）
    repo: RepositorySlice               # 宿主 repo 上下文
    search_space: str                   # 语义名："evasion" / "path_expansion" / "surface_expansion"
    budget_usd: float = 50.0
    max_tool_calls: int = 10
    target_model: str = "glm-5"
    extra: dict[str, Any] = field(default_factory=dict)

    _VALID_SPACES = frozenset({"evasion", "path_expansion", "surface_expansion"})

    def __post_init__(self) -> None:
        if self.search_space not in self._VALID_SPACES:
            raise ValueError(
                f"invalid search_space {self.search_space!r}; "
                f"expected one of {sorted(self._VALID_SPACES)}"
            )


@dataclass(frozen=True, slots=True)
class MutationResult:
    """Strategy 的产出包。

    ``variant=None`` 表示 strategy 主动放弃（refusal / budget 打爆 /
    invariant 违反），上层 campaign 照常落日志，但不会把它喂给 oracle。
    ``backend`` 保持纯字符串、不走枚举，为未来的新 strategy（tree_search
    等）预留扩展空间，无需改本模块。
    """

    variant: SkillVariant | None        # None 表示 strategy 失败
    tool_chain: tuple[str, ...]         # 依次调用的工具名（baseline 可空）
    stop_reason: str                    # 'submitted' | 'budget' | 'max_tools' | 'refusal' | 'invariant_violation' | 'programmatic'
    cost_usd: float
    backend: str                        # 'tool_use_agent' | 'random' | 'zeroshot' | 'blackbox'
    extra: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class MutationStrategy(Protocol):
    """造一个 variant 的策略接口。

    实现类必须暴露一个 ``name`` 属性（与 :class:`MutationResult.backend`
    对齐）以及 ``mutate(ctx)`` 方法；缺任何一个都会让
    ``isinstance(obj, MutationStrategy)`` 返回 False。
    """

    name: str

    def mutate(self, ctx: MutationContext) -> MutationResult:
        """给定 context，产出 MutationResult。"""
        ...


__all__ = ["MutationContext", "MutationResult", "MutationStrategy"]
