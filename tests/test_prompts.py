"""提示词与角色契约。

分三层，越靠前越值钱（前两轮真实事故都是"改一处漏改另一处"，而不是"提示词写错"）：
1. 跨模块契约：占位符、角色/步骤对应关系；
2. 文本冻结快照：full 四个任务描述与 base_doc_note 的 sha256 锚点；
3. 结构化断言：lite 各形态必须自洽，不得引用流程中不存在的上游。

⚠️ sha256 是**当前冻结快照**，不是"改造前基线"：把代码迁到 src/ + tests/ 约定时，
02/03/04 三条描述已按计划改写过，只有 01_requirements.md 与改造前逐字相同。
有意修改提示词时，需要同步更新本文件的常量并在提交说明里写清原因。
"""

from __future__ import annotations

import hashlib
import os
import string

import pytest

import src.config as config
import src.tasks as tasks_mod
import web_server as ws
from crewai import Crew, LLM
from src import pipelines as pl

# ---- 冻结快照（sha256 of task.description，full 流程、全新模式）-------------
FROZEN_PROMPTS = {
    "01_requirements.md":
        "4507f10c95c9c2926f855a250d077db95a963d88a1417ba97692c61b559b5334",
    "02_implementation.md":
        "b6090cf7149b64970ce968e718dc4a940930a0c62dc09c58bec5dfeb9d6417c8",
    "03_test_report.md":
        "d4b90c8f1dd488c11f8ce76118c861762f556f7201122da1d36dcf6da9a0ccc4",
    "04_final_report.md":
        "7031dc0f326b0fb19e6fbdbccdd4fe68f286844e10b86f18bea37258564af6cb",
}

EXPECTED_FIELDS = {"requirement", "sandbox_dir"}
LITE_MUST_HAVE = "本流程为精简流程"
LITE_MUST_NOT_HAVE = (
    "需求分析师",
    "base_requirements",
    "base_final_report",
    "整合前三份产出",
)


def _placeholders(text: str) -> set[str]:
    names = set()
    for _, field_name, _, _ in string.Formatter().parse(text):
        if field_name:
            names.add(field_name.split(".")[0].split("[")[0])
    return names


def _all_task_descriptions(agents) -> list[tuple[str, str]]:
    """返回 (标签, description)，覆盖 full/lite × 全新/迭代 四种形态。

    迭代上下文的 copied_docs 必须按 scope 真实派生（base_doc_map），
    不能塞一个该范围不会产生的文档名，否则测的是不存在的组合。
    """
    from pathlib import Path

    out: list[tuple[str, str]] = []
    for scope in (pl.FULL, pl.LITE):
        info_base = {
            "base_run_id": "20260924_193214",
            "copied_src": ["main.py"],
            "copied_tests": ["test_main.py"],
            "copied_docs": sorted(pl.base_doc_map(scope).values()),
        }
        for label, info in (("全新", None), ("迭代", info_base)):
            for task in tasks_mod.build_tasks(
                agents, _probe_dir(), scope, iteration_info=info
            ):
                out.append((f"{scope.key}/{label}/{Path(task.output_file).name}",
                            task.description))
    return out


def _probe_dir():
    from pathlib import Path

    return Path(__file__).resolve().parents[1] / "workspace" / "outputs" / "_probe"


class TestMissingPlaceholderContract:
    def test_descriptions_only_expose_requirement_and_sandbox_dir(self, agents):
        for tag, description in _all_task_descriptions(agents):
            assert _placeholders(description) == EXPECTED_FIELDS, (
                f"{tag} 暴露了意外的占位符："
                f"{_placeholders(description) ^ EXPECTED_FIELDS}"
            )


class TestRoleAndStepContract:
    def test_every_step_role_exists_in_build_agents(self, agents):
        roles = {step.role for step in pl.FULL.steps + pl.LITE.steps}
        assert roles <= set(agents), f"缺少角色的步骤：{roles - set(agents)}"

    def test_every_step_key_has_expected_output(self):
        assert {step.key for step in pl.FULL.steps} == set(tasks_mod._EXPECTED_OUTPUT)
        assert {step.key for step in pl.LITE.steps} <= set(tasks_mod._EXPECTED_OUTPUT)

    def test_full_step_dependencies_are_linear_and_valid(self):
        keys = [step.key for step in pl.FULL.steps]
        for index, step in enumerate(pl.FULL.steps):
            assert all(dep in keys[:index] for dep in step.deps), step


class TestFrozenFullPrompts:
    def test_full_descriptions_match_frozen_snapshot(self, agents):
        tasks = tasks_mod.build_tasks(agents, _probe_dir(), pl.FULL)
        for task in tasks:
            name = task.output_file.rsplit("\\", 1)[-1].rsplit("/", 1)[-1]
            digest = hashlib.sha256(task.description.encode("utf-8")).hexdigest()
            expected = FROZEN_PROMPTS[name]
            assert digest == expected, (
                f"{name} 提示词已变化：\n"
                f"  期望 {expected}\n  实际 {digest}\n"
                f"  长度 {len(task.description)}（对比见提交说明）\n"
                f"  开头 {task.description[:40]!r}"
            )

    def test_frozen_hashes_are_mutually_distinct(self):
        # 防复制粘贴事故：四条描述不可能一样
        assert len(set(FROZEN_PROMPTS.values())) == len(FROZEN_PROMPTS)

    def test_base_doc_note_full_is_frozen(self):
        assert pl.base_doc_note(pl.FULL) == (
            "（base_requirements.md=原需求分析，base_test_report.md=原测试报告，"
            "base_final_report.md=原最终报告）。"
        )


class TestLiteIsSelfConsistent:
    def test_all_lite_forms_avoid_missing_upstream(self, agents):
        lite_forms = [
            (tag, text)
            for tag, text in _all_task_descriptions(agents)
            if tag.startswith("lite/")
        ]
        assert len(lite_forms) == 4, lite_forms
        for tag, text in lite_forms:
            assert LITE_MUST_HAVE in text, tag
            for forbidden in LITE_MUST_NOT_HAVE:
                assert forbidden not in text, f"{tag} 引用了不存在的上游：{forbidden}"

    def test_lite_points_at_new_directory_layout(self, agents):
        from pathlib import Path

        for task in tasks_mod.build_tasks(agents, _probe_dir(), pl.LITE):
            name = Path(task.output_file).name
            assert "src/main.py" in task.description, name
            if name.endswith("test_report.md"):
                # 只有测试任务需要知道正式测试的落点
                assert "tests/test_main.py" in task.description, name


class TestIterationBriefIsConditional:
    INFO_WITH_TESTS = {
        "base_run_id": "20260924_193214",
        "copied_src": ["main.py", "quotes_parser.py"],
        "copied_tests": ["test_main.py"],
        "copied_docs": ["base_test_report.md"],
    }

    def test_with_baseline_tests(self):
        brief = tasks_mod._iteration_brief("tester", self.INFO_WITH_TESTS, pl.FULL)
        assert "tests/ 下已有基线的 test_main.py" in brief

    def test_without_baseline_tests(self):
        info = dict(self.INFO_WITH_TESTS, copied_tests=[])
        brief = tasks_mod._iteration_brief("tester", info, pl.FULL)
        assert "基线未提供可复用的测试文件" in brief
        assert "tests/ 下已有基线" not in brief

    def test_src_count_is_reported_not_listed(self):
        info = dict(self.INFO_WITH_TESTS, copied_src=[f"m{i}.py" for i in range(30)])
        brief = tasks_mod._iteration_brief("developer", info, pl.FULL)
        assert "共 30 个文件" in brief
        assert "m29.py" not in brief  # 不把长文件列表塞进提示词


class TestSafetyFuse:
    """反向自检：确保"测试绝不会真实调用 LLM"这条防线没被静默拆掉。"""

    def test_dotenv_is_disabled_so_real_key_never_enters_process(self):
        assert os.environ.get("PYTHON_DOTENV_DISABLED") == "1"
        # 刻意比长度而不是比字符串：断言失败时 pytest 会打印比较值，
        # 直接比 "" 会把真实密钥原文泄进测试输出
        assert len(config.API_KEY) == 0, "真实密钥不应进入测试进程（值已隐去）"
        assert ws.is_api_key_configured() is False

    def test_get_llm_is_fused(self):
        with pytest.raises(AssertionError):
            ws.get_llm()
        with pytest.raises(AssertionError):
            config.get_llm()

    def test_crew_kickoff_is_fused(self):
        with pytest.raises(AssertionError):
            Crew.kickoff(None)

    def test_llm_call_is_fused(self, dummy_llm):
        with pytest.raises(AssertionError):
            dummy_llm.call("hello")