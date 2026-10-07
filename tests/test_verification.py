"""验收层：主脚本/测试脚本定位、8 类判定分支、run.json 读写与结论回放。

这里守的是本项目最核心的承诺：**"服务端独立复跑"说了算，而不是测试工程师的自述**，
以及**历史任务的回放结论逐字不变、且回放绝不重跑**。
"""

from __future__ import annotations

import json

import pytest

import web_server as ws
from src import pipelines as pl

MAIN_OK = "def add(a, b):\n    return a + b\n"
TEST_OK = (
    "import unittest\nimport main\n\n"
    "class T(unittest.TestCase):\n"
    "    def test_add(self):\n"
    "        self.assertEqual(main.add(2, 3), 5)\n\n"
    "if __name__ == '__main__':\n"
    "    unittest.main()\n"
)
TEST_FAIL = TEST_OK.replace("5)", "6)")
TEST_EMPTY = "import unittest\n\nif __name__ == '__main__':\n    unittest.main()\n"
REPORT_PASS = "# 测试报告\n\n结论：全部用例通过，通过率 100%。\n"
REPORT_FAIL = "# 测试报告\n\n结论：本次交付未通过验收。\n"

LEGACY_PASS_REASON = "验收通过：交付脚本 main.py 已落盘，测试报告未报告失败结论。"


def _build(root, main_code=MAIN_OK, test_code=TEST_OK, report=REPORT_PASS,
           layout="new", run_id="19000101_000000"):
    """在给定根目录下造一个 run 的沙箱与产出。"""
    sandbox = root / "sandbox" / run_id
    outputs = root / "outputs" / run_id
    sandbox.mkdir(parents=True, exist_ok=True)
    outputs.mkdir(parents=True, exist_ok=True)
    main_rel = "src/main.py" if layout == "new" else "main.py"
    test_rel = "tests/test_main.py" if layout == "new" else "test_main.py"
    if main_code is not None:
        (sandbox / main_rel).parent.mkdir(parents=True, exist_ok=True)
        (sandbox / main_rel).write_text(main_code, encoding="utf-8")
    if test_code is not None:
        (sandbox / test_rel).parent.mkdir(parents=True, exist_ok=True)
        (sandbox / test_rel).write_text(test_code, encoding="utf-8")
    if report is not None:
        (outputs / "03_test_report.md").write_text(report, encoding="utf-8")
    return sandbox, outputs


class TestLocateDelivery:
    def test_prefers_src_main(self, tmp_path):
        (tmp_path / "src").mkdir()
        (tmp_path / "src" / "main.py").write_text(MAIN_OK, encoding="utf-8")
        (tmp_path / "main.py").write_text(MAIN_OK, encoding="utf-8")
        assert ws._locate_delivery(tmp_path)[0] == "src/main.py"

    def test_falls_back_to_legacy_root_main(self, tmp_path):
        (tmp_path / "main.py").write_text(MAIN_OK, encoding="utf-8")
        assert ws._locate_delivery(tmp_path)[0] == "main.py"

    def test_single_src_candidate_is_accepted(self, tmp_path):
        (tmp_path / "src").mkdir()
        (tmp_path / "src" / "tool.py").write_text(MAIN_OK, encoding="utf-8")
        assert ws._locate_delivery(tmp_path)[0] == "src/tool.py"

    def test_multiple_src_candidates_is_an_error(self, tmp_path):
        (tmp_path / "src").mkdir()
        for name in ("alpha.py", "beta.py"):
            (tmp_path / "src" / name).write_text(MAIN_OK, encoding="utf-8")
        delivery, reason = ws._locate_delivery(tmp_path)
        assert delivery is None
        assert "多个业务脚本" in reason

    def test_probe_only_sandbox_is_an_error(self, tmp_path):
        (tmp_path / "probe_x.py").write_text("print(1)\n", encoding="utf-8")
        delivery, reason = ws._locate_delivery(tmp_path)
        assert delivery is None
        assert "未找到开发交付的主脚本" in reason

    def test_missing_dir_is_an_error(self, tmp_path):
        delivery, reason = ws._locate_delivery(tmp_path / "nope")
        assert delivery is None
        assert "沙箱目录不存在" in reason


class TestLocateTestEntry:
    def test_prefers_tests_test_main(self, tmp_path):
        (tmp_path / "tests").mkdir()
        (tmp_path / "tests" / "test_main.py").write_text("x=1\n", encoding="utf-8")
        (tmp_path / "tests" / "test_other.py").write_text("x=1\n", encoding="utf-8")
        assert ws._locate_test_entry(tmp_path) == "tests/test_main.py"

    def test_excludes_backup_and_check_files(self, tmp_path):
        (tmp_path / "tests").mkdir()
        (tmp_path / "tests" / "test_main_backup.py").write_text("x=1\n",
                                                               encoding="utf-8")
        (tmp_path / "tests" / "test_script_check.py").write_text("x=1\n",
                                                                 encoding="utf-8")
        assert ws._locate_test_entry(tmp_path) is None

    def test_falls_back_to_legacy_root(self, tmp_path):
        (tmp_path / "test_main.py").write_text("x=1\n", encoding="utf-8")
        assert ws._locate_test_entry(tmp_path) == "test_main.py"

    def test_returns_none_when_absent(self, tmp_path):
        (tmp_path / "main.py").write_text(MAIN_OK, encoding="utf-8")
        assert ws._locate_test_entry(tmp_path) is None


class TestRunVerificationBranches:
    def test_missing_sandbox_dir(self, tmp_path):
        ok, reason, ev = ws._run_verification(
            tmp_path / "nope", tmp_path, pl.FULL)
        assert ok is False and "沙箱目录不存在" in reason

    def test_missing_delivery(self, tmp_path):
        sandbox, outputs = _build(tmp_path, main_code=None)
        ok, reason, _ = ws._run_verification(sandbox, outputs, pl.FULL)
        assert ok is False and "未找到开发交付的主脚本" in reason

    def test_empty_delivery(self, tmp_path):
        sandbox, outputs = _build(tmp_path, main_code="")
        ok, reason, _ = ws._run_verification(sandbox, outputs, pl.FULL)
        assert ok is False and "为空文件" in reason

    def test_missing_test_entry(self, tmp_path):
        sandbox, outputs = _build(tmp_path, test_code=None)
        ok, reason, _ = ws._run_verification(sandbox, outputs, pl.FULL)
        assert ok is False and "未找到可独立运行的正式测试脚本" in reason

    def test_zero_tests_is_invalid_evidence(self, tmp_path):
        sandbox, outputs = _build(tmp_path, test_code=TEST_EMPTY)
        ok, reason, ev = ws._run_verification(sandbox, outputs, pl.FULL)
        assert ok is False and "Ran 0 tests" in reason
        assert ev["exit_code"] == 0

    def test_failing_tests_turn_red(self, tmp_path):
        sandbox, outputs = _build(tmp_path, test_code=TEST_FAIL)
        ok, reason, ev = ws._run_verification(sandbox, outputs, pl.FULL)
        assert ok is False and "测试未全部通过" in reason
        assert ev["exit_code"] == 1

    def test_timeout_turns_red(self, tmp_path, monkeypatch):
        sandbox, outputs = _build(
            tmp_path, test_code="import time\ntime.sleep(60)\n")
        monkeypatch.setattr(ws, "_VERIFY_TIMEOUT", 2)
        ok, reason, ev = ws._run_verification(sandbox, outputs, pl.FULL)
        assert ok is False and "被强制终止" in reason
        assert ev["timed_out"] is True

    def test_report_failure_signal_turns_red(self, tmp_path):
        sandbox, outputs = _build(tmp_path, report=REPORT_FAIL)
        ok, reason, _ = ws._run_verification(sandbox, outputs, pl.FULL)
        assert ok is False and "与独立复跑结果不一致" in reason

    def test_pass_reports_evidence(self, tmp_path):
        sandbox, outputs = _build(tmp_path)
        ok, reason, ev = ws._run_verification(sandbox, outputs, pl.FULL)
        assert ok is True
        assert "服务端独立复跑 tests/test_main.py 退出码 0" in reason
        assert ev["tests_file"] == "tests/test_main.py"
        assert ev["delivery_file"] == "src/main.py"
        assert ev["exit_code"] == 0
        assert ev["timed_out"] is False
        assert "checked_at" in ev

    def test_pass_survives_huge_stdout(self, tmp_path):
        # 判据是退出码，不是输出内容：脚本狂打印也不能影响验收结论
        noisy = TEST_OK.replace(
            "class T(unittest.TestCase):",
            "for i in range(50000):\n    print('noise')\n\n\nclass T(unittest.TestCase):",
        )
        sandbox, outputs = _build(tmp_path, test_code=noisy)
        ok, reason, ev = ws._run_verification(sandbox, outputs, pl.FULL)
        assert ok is True, reason
        assert ev["exit_code"] == 0
        assert len(ev["stdout_tail"]) <= 2000

    def test_legacy_layout_still_verifiable(self, tmp_path):
        sandbox, outputs = _build(tmp_path, layout="legacy")
        ok, reason, ev = ws._run_verification(sandbox, outputs, pl.FULL)
        assert ok is True, reason
        assert ev["delivery_file"] == "main.py"
        assert ev["tests_file"] == "test_main.py"


class TestRunJson:
    def test_write_merges_and_preserves_existing_fields(self, tmp_roots):
        run_id = "19000101_000000"
        outputs = tmp_roots.outputs / run_id
        outputs.mkdir(parents=True)
        (outputs / "run.json").write_text(
            json.dumps({"scope": "lite", "requirement": "原需求",
                        "created_at": "T0"}, ensure_ascii=False),
            encoding="utf-8",
        )
        ws._write_run_json(run_id, verification={"ok": True, "reason": "r"})
        data = ws._read_run_json(run_id)
        assert data["scope"] == "lite"
        assert data["requirement"] == "原需求"
        assert data["created_at"] == "T0"
        assert data["verification"]["ok"] is True

    def test_write_swallows_io_error(self, tmp_roots):
        # 产出目录不存在也不能抛异常：任何异常都不得影响流水线结论
        ws._write_run_json("19990101_000000", verification={"ok": True})

    def test_read_returns_empty_on_bad_json(self, tmp_roots):
        outputs = tmp_roots.outputs / "19000101_000000"
        outputs.mkdir(parents=True)
        (outputs / "run.json").write_text("{ not json", encoding="utf-8")
        assert ws._read_run_json("19000101_000000") == {}

    def test_load_meta_tolerates_bad_json_and_bogus_scope(self, tmp_roots):
        outputs = tmp_roots.outputs / "19000101_000001"
        outputs.mkdir(parents=True)
        (outputs / "run.json").write_text('{"scope": "bogus"}', encoding="utf-8")
        # 非法 scope 应被吞掉并回退推断，而不是抛 ValueError
        assert ws._load_run_meta("19000101_000001")["scope"] in {"full", "lite"}

    def test_verification_must_be_a_dict(self, tmp_roots):
        outputs = tmp_roots.outputs / "19000101_000002"
        outputs.mkdir(parents=True)
        (outputs / "run.json").write_text(
            json.dumps({"scope": "full", "verification": ["not", "dict"]}),
            encoding="utf-8",
        )
        assert "verification" not in ws._load_run_meta("19000101_000002")


class TestInferScope:
    def test_lite_report_without_requirements_is_lite(self, tmp_roots):
        outputs = tmp_roots.outputs / "19000101_000003"
        outputs.mkdir(parents=True)
        (outputs / "02_test_report.md").write_text(REPORT_PASS, encoding="utf-8")
        assert ws._infer_scope("19000101_000003") == "lite"

    def test_requirements_present_is_full(self, tmp_roots):
        outputs = tmp_roots.outputs / "19000101_000004"
        outputs.mkdir(parents=True)
        (outputs / "01_requirements.md").write_text("x", encoding="utf-8")
        (outputs / "02_test_report.md").write_text("x", encoding="utf-8")
        assert ws._infer_scope("19000101_000004") == "full"

    def test_empty_dir_defaults_to_full(self, tmp_roots):
        (tmp_roots.outputs / "19000101_000005").mkdir(parents=True)
        assert ws._infer_scope("19000101_000005") == "full"


class TestVerdictFor:
    """回放契约：有持久化结论就用它（绝不重跑）；没有就回退旧判据。"""

    def test_uses_persisted_verification_verbatim(self, tmp_roots):
        run_id = "19000101_000006"
        _build(tmp_roots.root, run_id=run_id)
        persisted = {"ok": False, "reason": "持久化的红灯理由", "exit_code": 1}
        (tmp_roots.outputs / run_id / "run.json").write_text(
            json.dumps({"scope": "full", "verification": persisted},
                       ensure_ascii=False),
            encoding="utf-8",
        )
        ok, reason = ws._verdict_for(run_id)
        assert ok is False
        assert reason == "持久化的红灯理由"

    def test_falls_back_to_legacy_without_verification(self, tmp_roots):
        run_id = "19000101_000007"
        _build(tmp_roots.root, run_id=run_id, layout="legacy")
        # 没有 run.json → 走旧判据，理由必须逐字等于历史任务的措辞
        ok, reason = ws._verdict_for(run_id)
        assert ok is True
        assert reason == LEGACY_PASS_REASON

    def test_legacy_failure_signal_is_red(self, tmp_roots):
        run_id = "19000101_000008"
        _build(tmp_roots.root, run_id=run_id, layout="legacy", report=REPORT_FAIL)
        ok, reason = ws._verdict_for(run_id)
        assert ok is False
        assert "未通过验收" in reason

    def test_replay_does_not_write_anything(self, tmp_roots):
        run_id = "19000101_000009"
        _build(tmp_roots.root, run_id=run_id, layout="legacy")
        before = sorted(
            (p.relative_to(tmp_roots.root).as_posix(), p.stat().st_size,
             p.stat().st_mtime_ns)
            for p in tmp_roots.root.rglob("*") if p.is_file()
        )
        ws._verdict_for(run_id)
        after = sorted(
            (p.relative_to(tmp_roots.root).as_posix(), p.stat().st_size,
             p.stat().st_mtime_ns)
            for p in tmp_roots.root.rglob("*") if p.is_file()
        )
        assert before == after


class TestScanRuns:
    def test_history_statuses_and_scope_field(self, tmp_roots):
        _build(tmp_roots.root, run_id="19000101_000010")
        _build(tmp_roots.root, run_id="19000101_000011", layout="legacy",
               report=REPORT_FAIL)
        runs = {r["run_id"]: r for r in ws._scan_runs()}
        assert runs["19000101_000010"]["status"] == "archived"
        assert runs["19000101_000011"]["status"] == "rejected"
        for info in runs.values():
            assert info["scope"] in {"full", "lite"}
            assert ws._RUN_ID_RE.match(info["run_id"])