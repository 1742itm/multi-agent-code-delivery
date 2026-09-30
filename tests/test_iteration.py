"""迭代桥接：静态 import 闭包、目录树复制、脚本列表。

背景：历史扁平沙箱根目录混着大量探测脚本（实测某任务 33 个 .py 里 18 个是脏文件），
用文件名前缀猜法系统性失效。现在改为按 import 关系反推业务文件。
"""

from __future__ import annotations

import json

import web_server as ws
from src import pipelines as pl

MAIN = (
    "import quotes_parser\n"
    "import json\n"
    "import os\n\n"
    "def run():\n"
    "    return quotes_parser.parse('x')\n"
)
PARSER = "def parse(s):\n    return s\n"
JUNK = "print('i am a probe')\n"


def _flat_base(root, run_id="20260101_000000", with_test=True):
    sandbox = root / "sandbox" / run_id
    sandbox.mkdir(parents=True, exist_ok=True)
    (sandbox / "main.py").write_text(MAIN, encoding="utf-8")
    (sandbox / "quotes_parser.py").write_text(PARSER, encoding="utf-8")
    (sandbox / "run_tests.py").write_text(JUNK, encoding="utf-8")
    (sandbox / "path_check1.py").write_text(JUNK, encoding="utf-8")
    (sandbox / "write_main.py").write_text(JUNK, encoding="utf-8")
    if with_test:
        (sandbox / "test_main.py").write_text(
            "import main\n", encoding="utf-8")
    return sandbox


class TestImportClosure:
    def test_includes_imported_siblings_only(self, tmp_path):
        base = _flat_base(tmp_path)
        closure = ws._import_closure(base, ["main.py"])
        assert closure == ["main.py", "quotes_parser.py"]

    def test_is_transitive_until_fixpoint(self, tmp_path):
        (tmp_path / "a.py").write_text("import b\n", encoding="utf-8")
        (tmp_path / "b.py").write_text("import c\n", encoding="utf-8")
        (tmp_path / "c.py").write_text("x = 1\n", encoding="utf-8")
        assert ws._import_closure(tmp_path, ["a.py"]) == ["a.py", "b.py", "c.py"]

    def test_unparsable_file_is_kept_without_guessing(self, tmp_path):
        (tmp_path / "broken.py").write_text("def broken(:\n", encoding="utf-8")
        (tmp_path / "other.py").write_text("x = 1\n", encoding="utf-8")
        assert ws._import_closure(tmp_path, ["broken.py"]) == ["broken.py"]

    def test_unimported_local_files_are_not_included(self, tmp_path):
        (tmp_path / "m.py").write_text("import nothing_local\n", encoding="utf-8")
        (tmp_path / "unrelated.py").write_text("x = 1\n", encoding="utf-8")
        assert ws._import_closure(tmp_path, ["m.py"]) == ["m.py"]

    def test_local_file_shadowing_stdlib_is_included(self, tmp_path):
        # 与标准库同名的本地文件会被纳入：运行时它确实会遮蔽标准库，
        # 带着它迭代才不会让基线行为在迁移后悄悄变化。
        # （提示词里已禁止在 src/ 下起这类名字，这里固化的是实际语义。）
        (tmp_path / "json.py").write_text("x = 1\n", encoding="utf-8")
        (tmp_path / "m.py").write_text("import json\n", encoding="utf-8")
        assert ws._import_closure(tmp_path, ["m.py"]) == ["json.py", "m.py"]


class TestCopyTree:
    def test_copies_single_level_py_and_returns_names(self, tmp_path):
        src = tmp_path / "src"
        src.mkdir()
        (src / "main.py").write_text("x=1\n", encoding="utf-8")
        (src / "helper.py").write_text("x=2\n", encoding="utf-8")
        (src / "notes.txt").write_text("ignore", encoding="utf-8")
        dst = tmp_path / "dst"
        assert ws._copy_tree(src, dst) == ["helper.py", "main.py"]
        assert not (dst / "notes.txt").exists()

    def test_missing_source_returns_empty(self, tmp_path):
        assert ws._copy_tree(tmp_path / "nope", tmp_path / "dst") == []


class TestBridgeFlatBaseline:
    def test_only_business_closure_crosses_over(self, tmp_path):
        base = _flat_base(tmp_path)
        new_sandbox = tmp_path / "new"
        new_sandbox.mkdir()
        copied_src, copied_tests = ws._bridge_flat_baseline(base, new_sandbox)
        assert copied_src == ["main.py", "quotes_parser.py"]
        assert copied_tests == ["test_main.py"]
        assert (new_sandbox / "src" / "main.py").is_file()
        assert (new_sandbox / "src" / "quotes_parser.py").is_file()
        assert (new_sandbox / "tests" / "test_main.py").is_file()
        for junk in ("run_tests.py", "path_check1.py", "write_main.py"):
            assert not (new_sandbox / "src" / junk).exists(), junk

    def test_flat_base_without_tests_yields_empty_tests(self, tmp_path):
        base = _flat_base(tmp_path, with_test=False)
        new_sandbox = tmp_path / "new"
        new_sandbox.mkdir()
        _, copied_tests = ws._bridge_flat_baseline(base, new_sandbox)
        assert copied_tests == []
        assert not (new_sandbox / "tests").exists()

    def test_backup_and_check_tests_are_not_carried(self, tmp_path):
        base = _flat_base(tmp_path)
        (base / "test_main_backup.py").write_text("x=1\n", encoding="utf-8")
        (base / "test_script_check.py").write_text("x=1\n", encoding="utf-8")
        new_sandbox = tmp_path / "new"
        new_sandbox.mkdir()
        _, copied_tests = ws._bridge_flat_baseline(base, new_sandbox)
        assert copied_tests == ["test_main.py"]


class TestPrepareIteration:
    def test_new_layout_base_copies_src_and_tests_trees(self, tmp_roots):
        base_id = "20260101_000001"
        base = tmp_roots.sandbox / base_id
        (base / "src").mkdir(parents=True)
        (base / "tests").mkdir()
        (base / "src" / "main.py").write_text("x=1\n", encoding="utf-8")
        (base / "tests" / "test_main.py").write_text("x=1\n", encoding="utf-8")
        (base / "scratch").mkdir()
        (base / "scratch" / "probe.py").write_text("x=1\n", encoding="utf-8")
        # 产出文档同步存在，验证 base_*.md 复制
        outputs = tmp_roots.outputs / base_id
        outputs.mkdir(parents=True)
        for name in pl.output_names(pl.FULL):
            (outputs / name).write_text("doc", encoding="utf-8")

        new_sandbox = tmp_roots.root / "new"
        new_sandbox.mkdir()
        info = ws._prepare_iteration(base_id, new_sandbox, pl.FULL)
        assert info["copied_src"] == ["main.py"]
        assert info["copied_tests"] == ["test_main.py"]
        assert info["copied_docs"] == [
            "base_requirements.md", "base_test_report.md", "base_final_report.md",
        ]
        assert not (new_sandbox / "scratch").exists()
        # 迭代上下文必须是提示词能消费的键
        assert {"copied_src", "copied_tests", "copied_docs"} <= set(info)

    def test_flat_base_is_bridged_into_new_layout(self, tmp_roots):
        base_id = "20260101_000002"
        base = _flat_base(tmp_roots.root, run_id=base_id)
        assert base.is_dir()
        new_sandbox = tmp_roots.root / "new2"
        new_sandbox.mkdir()
        info = ws._prepare_iteration(base_id, new_sandbox, pl.LITE)
        assert info["copied_src"] == ["main.py", "quotes_parser.py"]
        assert (new_sandbox / "src" / "main.py").is_file()


class TestListPyFiles:
    def test_excludes_scratch_and_puts_delivery_first(self, tmp_path):
        for sub in ("src", "tests", "scratch"):
            (tmp_path / sub).mkdir()
        (tmp_path / "src" / "main.py").write_text("x=1\n", encoding="utf-8")
        (tmp_path / "tests" / "test_main.py").write_text("x=1\n", encoding="utf-8")
        (tmp_path / "scratch" / "probe_a.py").write_text("x=1\n", encoding="utf-8")
        (tmp_path / "scratch" / "probe_b.py").write_text("x=1\n", encoding="utf-8")
        files = ws._list_py_files(tmp_path)
        assert files[0] == "src/main.py"
        assert "tests/test_main.py" in files
        assert not [f for f in files if f.startswith("scratch/")]

    def test_legacy_flat_sandbox_listing_is_unchanged(self, tmp_path):
        (tmp_path / "main.py").write_text("x=1\n", encoding="utf-8")
        (tmp_path / "run_tests.py").write_text("x=1\n", encoding="utf-8")
        files = ws._list_py_files(tmp_path)
        assert files == ["main.py", "run_tests.py"]

    def test_paths_are_posix_relative(self, tmp_path):
        (tmp_path / "src").mkdir()
        (tmp_path / "src" / "main.py").write_text("x=1\n", encoding="utf-8")
        assert all("\\" not in f for f in ws._list_py_files(tmp_path))