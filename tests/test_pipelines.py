"""流程范围（src/pipelines.py）的纯数据与派生命名契约。

这里断言的每个名字都是 web_server 与前端实际消费的文件名/步骤名，
回退一次就是全线断链，所以用逐字量断言而不是"看起来对就行"。
"""

from __future__ import annotations

import pytest

# 注意：这里刻意用"模块限定"引用而不是 from ... import 名字。
# 因为 pytest 会把测试模块命名空间里所有 `test*` 开头的函数当用例收集，
# 直接 import tester_report_name 会被误收集成一个需要 scope 参数的假用例。
from src import pipelines as pl

FULL_OUTPUTS = [
    "01_requirements.md",
    "02_implementation.md",
    "03_test_report.md",
    "04_final_report.md",
]
LITE_OUTPUTS = ["01_implementation.md", "02_test_report.md"]
FULL = pl.FULL
LITE = pl.LITE


class TestDerivedNames:
    def test_full_output_names_are_unchanged(self):
        assert pl.output_names(FULL) == FULL_OUTPUTS

    def test_lite_output_names(self):
        assert pl.output_names(LITE) == LITE_OUTPUTS

    def test_full_stage_names_are_unchanged(self):
        assert pl.stage_names(FULL) == ["需求分析", "代码开发", "测试验证", "文档撰写"]

    def test_lite_stage_names(self):
        assert pl.stage_names(LITE) == ["代码开发", "测试验证"]

    def test_numbered_name_prefix_format(self):
        assert pl.numbered_name(0, FULL.steps[0]) == "01_requirements.md"
        assert pl.numbered_name(9, FULL.steps[0]) == "10_requirements.md"

    def test_tester_report_name(self):
        assert pl.tester_report_name(FULL) == "03_test_report.md"
        assert pl.tester_report_name(LITE) == "02_test_report.md"

    def test_primary_doc_name(self):
        assert pl.primary_doc_name(FULL) == "04_final_report.md"
        assert pl.primary_doc_name(LITE) == "02_test_report.md"


class TestBaseDocs:
    def test_full_base_doc_map_unchanged(self):
        # 有意不含 02_implementation（旧代码已复制，实现说明冗余）
        assert pl.base_doc_map(FULL) == {
            "01_requirements.md": "base_requirements.md",
            "03_test_report.md": "base_test_report.md",
            "04_final_report.md": "base_final_report.md",
        }

    def test_full_base_doc_map_order_is_01_03_04(self):
        # 顺序会进入迭代提示词文本，不能变
        assert list(pl.base_doc_map(FULL)) == [
            "01_requirements.md",
            "03_test_report.md",
            "04_final_report.md",
        ]

    def test_lite_base_doc_map(self):
        assert pl.base_doc_map(LITE) == {
            "01_implementation.md": "base_implementation.md",
            "02_test_report.md": "base_test_report.md",
        }

    def test_full_base_doc_keys_excludes_develop(self):
        assert FULL.base_doc_keys == ("analyze", "test", "document")
        assert "develop" not in FULL.base_doc_keys

    def test_base_doc_note_lists_declared_docs_only(self):
        note = pl.base_doc_note(LITE)
        assert "base_implementation.md" in note
        assert "base_test_report.md" in note
        assert "base_requirements.md" not in note
        assert "base_final_report.md" not in note


class TestGetScope:
    def test_none_and_blank_fall_back_to_full(self):
        assert pl.get_scope(None) is FULL
        assert pl.get_scope("") is FULL
        assert pl.get_scope("   ") is FULL

    def test_normalizes_case_and_whitespace(self):
        assert pl.get_scope("LITE") is LITE
        assert pl.get_scope(" lite ") is LITE
        assert pl.get_scope("full") is FULL

    def test_unknown_scope_raises(self):
        with pytest.raises(ValueError) as exc:
            pl.get_scope("bogus")
        assert "bogus" in str(exc.value)

    def test_default_key_is_full(self):
        assert pl.DEFAULT_KEY == "full"
        assert set(pl.SCOPES) == {"full", "lite"}