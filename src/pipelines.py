"""流水线模式（scope）定义 —— 「走哪些步骤」的单一数据源。

拆成两个正交维度：

- 流程范围 scope：full（需求分析→代码开发→测试验证→文档撰写）
                  lite（代码开发→测试验证）
- 是否迭代：由 Web 层的 base_run_id 决定，与 scope 无关。

产出文档名由「步骤下标 + 基础名」派生（见 numbered_name），因此 full 仍逐字得到
01_requirements.md ~ 04_final_report.md，与历史任务完全一致。

本模块只做纯数据与命名派生，不读写文件、不依赖 crewai。
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class StepSpec:
    """单个步骤。"""

    key: str            # analyze / develop / test / document
    role: str           # 对应 build_agents 返回字典的键
    title: str          # 步骤显示名，直接给前端进度条
    output_file: str    # 产出文档基础名，序号前缀由 numbered_name 派生
    deps: tuple[str, ...]  # 依赖的前序步骤 key（线性前驱）


@dataclass(frozen=True)
class ScopeSpec:
    """一个流程范围。"""

    key: str
    title: str
    steps: tuple[StepSpec, ...]
    # 迭代时复制到新沙箱的步骤 key，顺序即复制顺序（会进入迭代提示词文本）
    base_doc_keys: tuple[str, ...]


# ---- full：与改造前的四步流水线逐字对齐（零回归锚点）----------------------

FULL = ScopeSpec(
    key="full",
    title="复杂任务",
    steps=(
        StepSpec("analyze", "analyst", "需求分析", "requirements.md", ()),
        StepSpec("develop", "developer", "代码开发", "implementation.md", ("analyze",)),
        StepSpec(
            "test", "tester", "测试验证", "test_report.md", ("analyze", "develop")
        ),
        StepSpec(
            "document",
            "writer",
            "文档撰写",
            "final_report.md",
            ("analyze", "develop", "test"),
        ),
    ),
    # 与改造前的 _BASE_DOC_MAP 一致：01、03、04（有意不含 02_implementation）
    base_doc_keys=("analyze", "test", "document"),
)

# ---- lite：只做开发与测试 --------------------------------------------------

LITE = ScopeSpec(
    key="lite",
    title="轻量任务",
    steps=(
        StepSpec("develop", "developer", "代码开发", "implementation.md", ()),
        StepSpec("test", "tester", "测试验证", "test_report.md", ("develop",)),
    ),
    base_doc_keys=("develop", "test"),
)

SCOPES: dict[str, ScopeSpec] = {FULL.key: FULL, LITE.key: LITE}

# 向后兼容的默认值：未指定 scope 时一律按 full 处理（历史任务回放依赖此行为）
DEFAULT_KEY = FULL.key

# base_*.md 在迭代提示词里的中文说明
_BASE_DOC_LABELS = {
    "analyze": "原需求分析",
    "develop": "原实现说明",
    "test": "原测试报告",
    "document": "原最终报告",
}


def get_scope(key: str | None) -> ScopeSpec:
    """把外部传入的 scope 字符串归一化；空值回退 full，非法值抛 ValueError。"""
    if key is None or not str(key).strip():
        return FULL
    normalized = str(key).strip().lower()
    scope = SCOPES.get(normalized)
    if scope is None:
        raise ValueError(
            f"未知的流程范围 '{key}'，可选值：{'、'.join(SCOPES)}"
        )
    return scope


def numbered_name(index: int, step: StepSpec) -> str:
    """产出文档的完整文件名：序号前缀 + 基础名。"""
    return f"{index + 1:02d}_{step.output_file}"


def stage_names(scope: ScopeSpec) -> list[str]:
    """前端进度条的步骤名列表。"""
    return [step.title for step in scope.steps]


def output_names(scope: ScopeSpec) -> list[str]:
    """该范围的全部产出文档名（按步骤顺序）。"""
    return [numbered_name(i, step) for i, step in enumerate(scope.steps)]


def base_doc_map(scope: ScopeSpec) -> dict[str, str]:
    """迭代复制映射：基线产出文件名 -> 新沙箱内的参考文件名。

    仅同范围迭代，故基线文件名与本范围的编号规则一致。
    """
    mapping: dict[str, str] = {}
    for index, step in enumerate(scope.steps):
        if step.key in scope.base_doc_keys:
            mapping[numbered_name(index, step)] = f"base_{step.output_file}"
    return mapping


def base_doc_note(scope: ScopeSpec) -> str:
    """迭代提示词里对 base_*.md 的括注说明。

    full 的返回值必须与改造前 tasks.py 中的字面量逐字一致。
    """
    items = [
        f"base_{step.output_file}={_BASE_DOC_LABELS[step.key]}"
        for step in scope.steps
        if step.key in scope.base_doc_keys
    ]
    return "（" + "，".join(items) + "）。"


def tester_report_name(scope: ScopeSpec) -> str | None:
    """承载验收结论的测试报告文件名（按角色 tester 定位）；该范围无测试步骤时为 None。"""
    for index, step in enumerate(scope.steps):
        if step.role == "tester":
            return numbered_name(index, step)
    return None


def primary_doc_name(scope: ScopeSpec) -> str | None:
    """前端默认展示的产出文档：该范围的最后一步产出。"""
    if not scope.steps:
        return None
    last = len(scope.steps) - 1
    return numbered_name(last, scope.steps[last])