"""本机专属：对 workspace 下的真实历史任务断言"回放结论不变、且回放零写回"。

CI 上 workspace/ 不存在（被 gitignore），整个模块会 skip；
本机运行时会真正执行，是历史行为最直接的守卫。

刻意**不使用** tmp_roots fixture —— 这里要测的就是真实目录。
"""

from __future__ import annotations

import pytest

import web_server as ws
from src import pipelines as pl

pytestmark = pytest.mark.local

# 冻结见证表：任务还在 → 结论必须不变；被 clean_history 清理掉 → skip 而不是误报
FROZEN_WITNESS = {
    "20260924_193214": True,
    "20260924_195530": True,
    "20260924_204645": True,
    "20260924_211307": True,
    "20260926_235231": True,
    "20260926_235351": True,
    "20260926_235622": True,
    "20260927_001142": True,
    "20260929_221502": True,
    "20260929_221833": True,
    "20260929_221939": True,
    "20260929_222119": True,
}


def _run_ids() -> list[str]:
    ids = set()
    for root in (ws.SANDBOX_ROOT, ws.OUTPUTS_ROOT):
        if root.is_dir():
            ids |= {p.name for p in root.iterdir()
                    if p.is_dir() and ws._RUN_ID_RE.match(p.name)}
    return sorted(ids)


@pytest.fixture(scope="module")
def require_workspace():
    if not ws.SANDBOX_ROOT.is_dir() and not ws.OUTPUTS_ROOT.is_dir():
        pytest.skip("本机没有 workspace 历史任务目录（CI 环境正常现象）")


def _snapshot(root) -> list[tuple[str, int, int]]:
    return sorted(
        (p.relative_to(root).as_posix(), p.stat().st_size, p.stat().st_mtime_ns)
        for p in root.rglob("*") if p.is_file()
    )


class TestReplayIsSelfAnchored:
    """不依赖硬编码 id，对当前存在的每个任务都成立的不变量。"""

    def test_every_run_has_a_boolean_verdict(self, require_workspace):
        ids = _run_ids()
        assert ids, "workspace 存在却一个 run 目录都没有"
        for rid in ids:
            ok, reason = ws._verdict_for(rid)
            assert isinstance(ok, bool), rid
            assert isinstance(reason, str), rid

    def test_persisted_verification_is_replayed_verbatim(self, require_workspace):
        checked = 0
        for rid in _run_ids():
            verification = ws._load_run_meta(rid).get("verification")
            if not isinstance(verification, dict):
                continue
            checked += 1
            ok, reason = ws._verdict_for(rid)
            assert ok == bool(verification.get("ok")), rid
            assert reason == str(verification.get("reason") or ""), rid
        assert checked >= 1, "本机应至少有一个带 verification 的任务"

    def test_without_verification_it_equals_recomputed_legacy_verdict(
        self, require_workspace
    ):
        checked = 0
        for rid in _run_ids():
            if "verification" in ws._load_run_meta(rid):
                continue
            checked += 1
            expected = ws._evaluate_delivery(
                ws.SANDBOX_ROOT / rid,
                ws.OUTPUTS_ROOT / rid,
                pl.get_scope(ws._load_run_meta(rid)["scope"]),
            )
            assert ws._verdict_for(rid) == expected, rid
        assert checked >= 1, "本机应至少有一个走旧判据的任务"

    def test_delivery_layout_contract(self, require_workspace):
        for rid in _run_ids():
            sandbox = ws.SANDBOX_ROOT / rid
            if not sandbox.is_dir():
                continue
            delivery, _ = ws._locate_delivery(sandbox)
            if (sandbox / "src" / "main.py").is_file():
                assert delivery == "src/main.py", rid
            elif (sandbox / "main.py").is_file():
                assert delivery == "main.py", rid

    def test_replay_writes_nothing(self, require_workspace):
        roots = [r for r in (ws.SANDBOX_ROOT, ws.OUTPUTS_ROOT) if r.is_dir()]
        before = {r: _snapshot(r) for r in roots}
        for rid in _run_ids():
            ws._verdict_for(rid)
        for root in roots:
            assert before[root] == _snapshot(root), f"{root} 在回放后被改动"

    def test_scan_runs_reports_only_terminal_states(self, require_workspace):
        for info in ws._scan_runs():
            assert info["status"] in {"archived", "rejected"}, info
            assert info["scope"] in {"full", "lite"}, info


class TestFrozenWitness:
    """逐条见证：任务还在就必须仍是原来的结论。"""

    @pytest.mark.parametrize("run_id,expected_ok", sorted(FROZEN_WITNESS.items()))
    def test_known_run_verdict_unchanged(self, require_workspace, run_id, expected_ok):
        if not (ws.SANDBOX_ROOT / run_id).is_dir():
            pytest.skip(f"{run_id} 已被清理，跳过见证（不算回归）")
        ok, _reason = ws._verdict_for(run_id)
        assert ok is expected_ok, f"{run_id} 的结论从 {expected_ok} 变成了 {ok}"