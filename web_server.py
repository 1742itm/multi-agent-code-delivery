"""多角色协作任务 Agent —— 简易 Web 服务。

职责：
1. 接收前端提交的需求与流程范围（复杂/轻量），后台串行启动 CrewAI 多角色流水线；
2. 通过阶段回调向前端提供实时进度（前端轮询）；
3. 列出/查看历史任务的脚本与产出文档；
4. 在与 Agent 完全相同的安全沙箱中运行任务脚本（支持 stdin 输入）；
5. 基于历史任务迭代修改（仅限同范围）。

仅绑定 127.0.0.1，不对外暴露。启动：

    .venv\\Scripts\\python.exe web_server.py
"""

from __future__ import annotations

import ast
import os
import re
import json
import shutil
import threading
import traceback
import subprocess
import webbrowser
from datetime import datetime
from pathlib import Path

# 关闭 CrewAI 遥测，必须在导入 crewai 相关模块前设置
os.environ.setdefault("OTEL_SDK_DISABLED", "true")

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from crewai import Crew, Process

from src.agents import build_agents
from src.config import get_llm, is_api_key_configured
from src.pipelines import (
    DEFAULT_KEY,
    FULL,
    LITE,
    ScopeSpec,
    base_doc_map,
    get_scope,
    numbered_name,
    primary_doc_name,
    stage_names,
    tester_report_name,
)
from src.tasks import build_tasks
from src.tools import execute_sandboxed, is_safe_filename, static_check

PROJECT_ROOT = Path(__file__).resolve().parent
STATIC_DIR = PROJECT_ROOT / "static"
WORKSPACE_ROOT = PROJECT_ROOT / "workspace"
SANDBOX_ROOT = WORKSPACE_ROOT / "sandbox"
OUTPUTS_ROOT = WORKSPACE_ROOT / "outputs"

_RUN_ID_RE = re.compile(r"^\d{8}_\d{6}$")
_STDIN_LIMIT = 4000
_OUTPUT_LIMIT = 20000

app = FastAPI(title="多角色协作任务 Agent")


# ---------------------------------------------------------------------------
# 任务管理（内存态；同一时刻只允许一个 Crew 运行）
# ---------------------------------------------------------------------------

_jobs_lock = threading.Lock()
_jobs: dict[str, dict] = {}


def _is_busy() -> bool:
    return any(job["status"] == "running" for job in _jobs.values())


def _new_stage_states(scope: ScopeSpec) -> list[dict]:
    stages = [
        {"name": name, "status": "pending", "output": ""} for name in stage_names(scope)
    ]
    stages[0]["status"] = "running"
    return stages


# 探测/调试/测试类文件名前缀，不作为“开发交付主脚本”
_NON_DELIVERY_PREFIXES = ("probe", "inspect", "test_", "conclude", "read_", "extra_")


def _load_lineage(run_id: str) -> dict:
    """读取任务血缘文件（服务重启后仍可获知其基于哪个历史任务迭代）。"""
    path = OUTPUTS_ROOT / run_id / "iteration.json"
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def _infer_scope(run_id: str) -> str:
    """run.json 缺失时的回退推断：无需求分析文档但有轻量流程的测试报告，判为 lite。

    否则会把轻量任务按 full 回放 —— 找不到测试报告就跳过失败信号检查，
    让实际失败的任务被回放成绿灯。
    """
    out_dir = OUTPUTS_ROOT / run_id
    full_first = numbered_name(0, FULL.steps[0])
    lite_report = tester_report_name(LITE)
    if (
        not (out_dir / full_first).is_file()
        and lite_report
        and (out_dir / lite_report).is_file()
    ):
        return LITE.key
    return FULL.key


def _load_run_meta(run_id: str) -> dict:
    """读取任务元信息（run.json）；缺失或损坏时回退推断（老任务一律 full）。

    返回 dict，可能含 scope 与 verification 两个键。
    """
    path = OUTPUTS_ROOT / run_id / "run.json"
    if path.is_file():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            meta = {"scope": get_scope(data.get("scope")).key}
            if isinstance(data.get("verification"), dict):
                meta["verification"] = data["verification"]
            return meta
        except (json.JSONDecodeError, OSError, ValueError):
            pass
    return {"scope": _infer_scope(run_id)}


def _scope_key_of(run_id: str) -> str:
    """任务所属流程范围：运行中的任务取内存记录，历史任务读 run.json。"""
    job = _jobs.get(run_id)
    if job:
        return job.get("scope", DEFAULT_KEY)
    return _load_run_meta(run_id)["scope"]


def _prepare_iteration(
    base_run_id: str, new_sandbox: Path, scope: ScopeSpec
) -> dict:
    """把历史任务的代码与文档复制到新任务沙箱，返回迭代上下文。

    - 基线已是 src/ + tests/ 结构：直接复制这两棵子树；
    - 基线是历史扁平结构：按**静态 import 闭包**挑选业务文件归入 src/，
      把正式测试归入 tests/，探测/调试类脚本一律不带过去；
    - 该范围声明为 base 的产出文档复制为 base_*.md（放在沙箱内，Agent 可直接读取）。

    仅同范围迭代，故基线产出文件名与当前范围的编号规则一致。
    """
    base_sandbox = SANDBOX_ROOT / base_run_id
    base_outputs = OUTPUTS_ROOT / base_run_id

    copied_src: list[str] = []
    copied_tests: list[str] = []

    if (base_sandbox / "src").is_dir():
        copied_src = _copy_tree(base_sandbox / "src", new_sandbox / "src")
        copied_tests = _copy_tree(base_sandbox / "tests", new_sandbox / "tests")
    elif base_sandbox.is_dir():
        copied_src, copied_tests = _bridge_flat_baseline(
            base_sandbox, new_sandbox
        )

    copied_docs: list[str] = []
    if base_outputs.is_dir():
        for old_name, new_name in base_doc_map(scope).items():
            src = base_outputs / old_name
            if src.is_file():
                shutil.copy2(src, new_sandbox / new_name)
                copied_docs.append(new_name)

    return {
        "copied_src": copied_src,
        "copied_tests": copied_tests,
        "copied_docs": copied_docs,
    }


def _copy_tree(src_dir: Path, dst_dir: Path) -> list[str]:
    """复制某个单层目录下的全部 .py（只取单层，与沙箱命名规则一致）。"""
    if not src_dir.is_dir():
        return []
    copied: list[str] = []
    for p in sorted(src_dir.iterdir()):
        if not (p.is_file() and p.suffix == ".py" and is_safe_filename(p.name)):
            continue
        dst_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, dst_dir / p.name)
        copied.append(p.name)
    return copied


def _import_closure(base_sandbox: Path, entries: list[str]) -> list[str]:
    """静态解析入口文件的 import，收集沙箱根目录下的同名 .py，迭代到不动点。

    历史扁平沙箱里混杂着大量探测脚本，用前缀猜法挑不干净；按 import 关系
    反推业务文件更准。解析失败的文件直接保留自身，不做猜测。
    """
    seen: list[str] = []
    pending = list(entries)
    while pending:
        rel = pending.pop()
        if rel in seen:
            continue
        seen.append(rel)
        path = base_sandbox / rel
        if not path.is_file():
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
        except (SyntaxError, OSError):
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [alias.name.split(".")[0] for alias in node.names]
            elif (
                isinstance(node, ast.ImportFrom)
                and node.level == 0
                and node.module
            ):
                names = [node.module.split(".")[0]]
            else:
                continue
            for name in names:
                cand = f"{name}.py"
                if (base_sandbox / cand).is_file() and cand not in seen:
                    pending.append(cand)
    return sorted(seen)


def _bridge_flat_baseline(
    base_sandbox: Path, new_sandbox: Path
) -> tuple[list[str], list[str]]:
    """把历史扁平基线桥接成 src/ + tests/ 结构，返回 (业务文件, 测试文件)。"""
    delivery, _ = _locate_delivery(base_sandbox)
    test_entry = _locate_test_entry(base_sandbox)

    # 正式测试：入口测试 + 其余正式 test_*.py（排除过程性/备份文件）
    test_names = sorted(
        p.name
        for p in base_sandbox.iterdir()
        if p.is_file()
        and p.name.startswith("test_")
        and p.suffix == ".py"
        and not p.name.endswith(_TEST_ENTRY_EXCLUDE)
    )

    closure = _import_closure(
        base_sandbox, [n for n in (delivery, test_entry) if n]
    )
    src_names = [n for n in closure if n not in test_names]

    copied_src: list[str] = []
    for name in src_names:
        src = base_sandbox / name
        if not (src.is_file() and is_safe_filename(name)):
            continue
        (new_sandbox / "src").mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, new_sandbox / "src" / name)
        copied_src.append(name)

    copied_tests: list[str] = []
    for name in test_names:
        src = base_sandbox / name
        if not (src.is_file() and is_safe_filename(name)):
            continue
        (new_sandbox / "tests").mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, new_sandbox / "tests" / name)
        copied_tests.append(name)

    return copied_src, copied_tests

# 测试报告中代表验收失败的明确信号
_FAIL_SIGNALS = ("未通过验收", "全部阻塞", "通过率 0.0%", "通过率:0.0%")

# 服务端独立复跑正式测试的配置
_VERIFY_TIMEOUT = 60
_VERIFY_TAIL = 2000
# unittest 明确跑 0 个用例的信号（属无效证据，不能算通过）
_RAN_ZERO_SIGNAL = "Ran 0 tests"


# 正式测试脚本的排除后缀（过程性/备份文件，不作为独立复跑的入口）
_TEST_ENTRY_EXCLUDE = ("_backup.py", "_check.py")


def _rel_py_files(sandbox_dir: Path, subdir: str) -> list[str]:
    """列出沙箱下某个单层子目录的 .py（相对 POSIX 路径，已排序）。"""
    target = sandbox_dir / subdir if subdir else sandbox_dir
    if not target.is_dir():
        return []
    return sorted(
        f"{subdir}/{p.name}" if subdir else p.name
        for p in target.iterdir()
        if p.is_file() and p.suffix == ".py"
    )


def _is_delivery_name(rel_path: str) -> bool:
    """是否可能是开发交付的业务脚本（只看文件名前缀，不看所在目录）。"""
    return not Path(rel_path).name.startswith(_NON_DELIVERY_PREFIXES)


def _locate_delivery(sandbox_dir: Path) -> tuple[str | None, str]:
    """定位开发交付的主脚本，返回 (相对路径, 失败原因)；成功时原因为空串。

    优先新目录约定 src/，再兼容历史扁平结构（历史任务的结论必须逐字不变，
    因此根目录回退分支保持改造前的行为）。
    """
    if not sandbox_dir.is_dir():
        return None, "验收失败：任务沙箱目录不存在。"

    if (sandbox_dir / "src" / "main.py").is_file():
        return "src/main.py", ""
    if (sandbox_dir / "main.py").is_file():
        return "main.py", ""

    # 新约定回退：src/ 下若只有一个业务脚本才认为它是交付，多于一个宁可判红
    src_candidates = [
        f for f in _rel_py_files(sandbox_dir, "src") if _is_delivery_name(f)
    ]
    if len(src_candidates) == 1:
        return src_candidates[0], ""
    if len(src_candidates) > 1:
        names = "、".join(Path(f).name for f in src_candidates)
        return None, (
            f"验收失败：未找到 src/main.py，且 src/ 下存在多个业务脚本（{names}），"
            "无法确定交付主脚本。"
        )

    # 历史扁平结构回退：根目录非探测类第一个（与改造前一致）
    root_files = _rel_py_files(sandbox_dir, "")
    root_candidates = [n for n in root_files if _is_delivery_name(n)]
    if not root_candidates:
        return None, (
            "验收失败：沙箱目录中未找到开发交付的主脚本（main.py 或其他业务 .py），"
            f"仅有 {len(root_files)} 个探测/测试类文件，开发产出未真实落盘。"
        )
    return root_candidates[0], ""


def _locate_test_entry(sandbox_dir: Path) -> str | None:
    """定位正式测试脚本（相对路径）；找不到返回 None。"""

    def _pick(files: list[str]) -> str | None:
        if not files:
            return None
        for rel in files:
            if Path(rel).name == "test_main.py":
                return rel
        return files[0]

    for subdir in ("tests", ""):
        files = [
            rel
            for rel in _rel_py_files(sandbox_dir, subdir)
            if Path(rel).name.startswith("test_")
            and not Path(rel).name.endswith(_TEST_ENTRY_EXCLUDE)
        ]
        picked = _pick(files)
        if picked:
            return picked
    return None


def _verdict_for(run_id: str) -> tuple[bool, str]:
    """任务验收结论：优先用 run.json 里持久化的复跑结果，否则回退旧判据。

    历史任务没有 verification 字段，一律走旧判据且**不执行任何代码** ——
    这是历史回放结论逐字不变、且沙箱零写回的保证。
    """
    meta = _load_run_meta(run_id)
    verification = meta.get("verification")
    if isinstance(verification, dict):
        return bool(verification.get("ok")), str(verification.get("reason") or "")
    return _evaluate_delivery(
        SANDBOX_ROOT / run_id, OUTPUTS_ROOT / run_id, get_scope(meta["scope"])
    )


def _evaluate_delivery(
    sandbox_dir: Path, output_dir: Path, scope: ScopeSpec | None = None
) -> tuple[bool, str]:
    """旧版（报告信号）客观验收 —— 供历史任务回放使用。

    两层判定：
    1. 沙箱中必须真实存在开发交付的主脚本（src/main.py 或历史结构的 main.py）；
    2. 该流程范围的测试报告中不得出现明确的验收失败结论。
    返回 (是否通过, 原因说明)。

    scope 不给定时按 full 处理（历史任务回放依赖此默认值）。
    """
    scope = scope if scope is not None else FULL

    delivery, reason = _locate_delivery(sandbox_dir)
    if delivery is None:
        return False, reason

    # 主脚本为空文件也视为无效交付
    if (sandbox_dir / delivery).stat().st_size == 0:
        return False, f"验收失败：交付脚本 {delivery} 为空文件。"

    report_name = tester_report_name(scope)
    test_report = output_dir / report_name if report_name else None
    if test_report is not None and test_report.is_file():
        content = test_report.read_text(encoding="utf-8", errors="replace")
        hit = next((sig for sig in _FAIL_SIGNALS if sig in content), None)
        if hit:
            return False, (
                f"验收失败：测试工程师在测试报告中给出未通过结论（命中信号“{hit}”），"
                f"交付脚本 {delivery} 未通过用例验收，请查看测试报告了解详情。"
            )

    return True, f"验收通过：交付脚本 {delivery} 已落盘，测试报告未报告失败结论。"


def _read_run_json(run_id: str) -> dict:
    path = OUTPUTS_ROOT / run_id / "run.json"
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}
    return data if isinstance(data, dict) else {}


def _write_run_json(run_id: str, **fields) -> None:
    """读-改-写合并 run.json（保留 scope/requirement/created_at）。

    任何异常都不得影响流水线结果，因此静默吞掉 IO 错误。
    """
    path = OUTPUTS_ROOT / run_id / "run.json"
    try:
        data = _read_run_json(run_id)
        data.update(fields)
        path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    except OSError:
        pass


def _run_verification(
    sandbox_dir: Path, output_dir: Path, scope: ScopeSpec
) -> tuple[bool, str, dict]:
    """服务端独立复跑正式测试，返回 (是否通过, 原因说明, 证据)。

    只在流水线结束时调用一次（历史回放走 `_verdict_for` 读持久化结论，不经过这里）。
    判定优先级：主脚本缺失/为空 > 找不到测试入口 > 空跑（Ran 0 tests）>
    复跑超时 > 退出码非 0 > 报告命中失败信号 > 通过。
    """
    evidence: dict = {"checked_at": datetime.now().isoformat(timespec="seconds")}

    delivery, reason = _locate_delivery(sandbox_dir)
    evidence["delivery_file"] = delivery
    if delivery is None:
        return False, reason, evidence
    if (sandbox_dir / delivery).stat().st_size == 0:
        return False, f"验收失败：交付脚本 {delivery} 为空文件。", evidence

    test_entry = _locate_test_entry(sandbox_dir)
    evidence["tests_file"] = test_entry
    if test_entry is None:
        return False, (
            "验收失败：沙箱中未找到可独立运行的正式测试脚本"
            "（新约定为 tests/test_main.py，历史结构为根目录 test_main.py），"
            f"无法独立复跑验证交付脚本 {delivery}。"
        ), evidence

    try:
        proc = execute_sandboxed(
            sandbox_dir, sandbox_dir / test_entry, timeout=_VERIFY_TIMEOUT
        )
    except subprocess.TimeoutExpired:
        evidence.update({"timed_out": True, "exit_code": None})
        return False, (
            f"验收失败：服务端独立复跑 {test_entry} 超过 {_VERIFY_TIMEOUT} 秒被强制终止。"
            "正式测试必须是非交互、可独立快速跑完的脚本。"
        ), evidence

    stdout = proc.stdout or ""
    stderr = proc.stderr or ""
    combined = stdout + "\n" + stderr
    evidence.update(
        {
            "timed_out": False,
            "exit_code": proc.returncode,
            "stdout_tail": stdout[-_VERIFY_TAIL:],
            "stderr_tail": stderr[-_VERIFY_TAIL:],
        }
    )

    if _RAN_ZERO_SIGNAL in combined:
        return False, (
            f"验收失败：服务端独立复跑 {test_entry} 时显示「Ran 0 tests」，"
            "未真实执行任何用例，属无效证据（测试脚本需真正运行断言）。"
        ), evidence

    if proc.returncode != 0:
        tail = (stderr or stdout).strip().splitlines()[-6:]
        detail = " / ".join(line.strip() for line in tail) or "（无输出）"
        return False, (
            f"验收失败：服务端独立复跑 {test_entry} 退出码为 {proc.returncode}，"
            f"测试未全部通过，交付脚本 {delivery} 未通过验收。输出摘要：{detail}"
        ), evidence

    report_name = tester_report_name(scope)
    test_report = output_dir / report_name if report_name else None
    if test_report is not None and test_report.is_file():
        content = test_report.read_text(encoding="utf-8", errors="replace")
        hit = next((sig for sig in _FAIL_SIGNALS if sig in content), None)
        if hit:
            return False, (
                f"验收失败：测试报告中给出未通过结论（命中信号“{hit}”），"
                f"与独立复跑结果不一致，请检查交付脚本 {delivery}。"
            ), evidence

    return True, (
        f"验收通过：服务端独立复跑 {test_entry} 退出码 0，"
        f"交付脚本 {delivery} 已落盘，测试报告未报告失败结论。"
    ), evidence


def _run_crew(
    run_id: str,
    requirement: str,
    sandbox_dir: Path,
    output_dir: Path,
    scope: ScopeSpec,
    iteration_info: dict | None = None,
) -> None:
    """后台线程：构建并运行 Crew，通过回调更新阶段状态。"""
    job = _jobs[run_id]

    def on_stage(index: int, name: str, task_output) -> None:
        with _jobs_lock:
            stages = job["stages"]
            stages[index]["status"] = "done"
            raw = str(getattr(task_output, "raw", task_output))
            stages[index]["output"] = raw[:8000]
            if index + 1 < len(stages):
                stages[index + 1]["status"] = "running"

    try:
        # Agent 工具通过该环境变量定位本次沙箱目录（串行执行，无并发冲突）
        os.environ["SANDBOX_TASK_DIR"] = str(sandbox_dir)

        llm = get_llm()
        agents = build_agents(llm)
        tasks = build_tasks(
            agents,
            output_dir,
            scope,
            stage_callback=on_stage,
            iteration_info=iteration_info,
        )

        crew = Crew(
            agents=[agents[step.role] for step in scope.steps],
            tasks=tasks,
            process=Process.sequential,
            verbose=False,
            cache=False,
        )
        crew.kickoff(
            inputs={
                "requirement": requirement,
                "sandbox_dir": str(sandbox_dir),
            }
        )
        # 流程走完 ≠ 交付成功：服务端独立复跑正式测试后给出结论，并持久化证据
        accepted, reason, evidence = _run_verification(sandbox_dir, output_dir, scope)
        _write_run_json(run_id, verification={"ok": accepted, "reason": reason, **evidence})
        with _jobs_lock:
            job["verdict_reason"] = reason
            if accepted:
                job["status"] = "done"
            else:
                job["status"] = "failed"
                job["error"] = reason
    except Exception:  # noqa: BLE001 - 任何失败都要回传给前端
        with _jobs_lock:
            job["status"] = "failed"
            job["error"] = traceback.format_exc()[-4000:]


# ---------------------------------------------------------------------------
# 请求模型
# ---------------------------------------------------------------------------


class RequirementBody(BaseModel):
    requirement: str = Field(..., min_length=1, max_length=4000)
    base_run_id: str | None = Field(default=None, max_length=20)
    scope: str | None = Field(default=None, max_length=20)


class ExecBody(BaseModel):
    filename: str = Field(..., min_length=1, max_length=100)
    stdin: str = Field(default="", max_length=_STDIN_LIMIT)
    timeout: int = Field(default=60, ge=1, le=60)


# ---------------------------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------------------------


def _validate_run_id(run_id: str) -> Path:
    if not _RUN_ID_RE.match(run_id):
        raise HTTPException(status_code=400, detail="非法的任务ID格式")
    sandbox_dir = SANDBOX_ROOT / run_id
    if not sandbox_dir.is_dir():
        raise HTTPException(status_code=404, detail="任务沙箱目录不存在")
    return sandbox_dir


def _scan_runs() -> list[dict]:
    """配对扫描历史任务（含本次服务内运行中的任务）。"""
    sandbox_ids = {
        p.name
        for p in SANDBOX_ROOT.iterdir()
        if p.is_dir() and _RUN_ID_RE.match(p.name)
    } if SANDBOX_ROOT.is_dir() else set()
    output_ids = {
        p.name
        for p in OUTPUTS_ROOT.iterdir()
        if p.is_dir() and _RUN_ID_RE.match(p.name)
    } if OUTPUTS_ROOT.is_dir() else set()

    run_ids = sorted(sandbox_ids | output_ids | set(_jobs), reverse=True)
    result = []
    for rid in run_ids:
        job = _jobs.get(rid)
        if job:
            status = job["status"]
            requirement = job["requirement"]
            base_run_id = job.get("base_run_id")
            scope_key = job.get("scope", DEFAULT_KEY)
        else:
            # 服务重启后的历史任务：优先用持久化的复跑结论回放，绝不重跑
            scope_key = _load_run_meta(rid)["scope"]
            accepted, _ = _verdict_for(rid)
            status = "archived" if accepted else "rejected"
            requirement = ""
            base_run_id = _load_lineage(rid).get("base_run_id")
        result.append(
            {
                "run_id": rid,
                "status": status,
                "requirement": requirement,
                "base_run_id": base_run_id,
                "scope": scope_key,
                "has_sandbox": rid in sandbox_ids,
                "has_outputs": rid in output_ids,
            }
        )
    return result


def _list_py_files(sandbox_dir: Path) -> list[str]:
    """列出沙箱内可供调试运行的脚本（相对 POSIX 路径），交付主脚本排最前。

    只遍历沙箱根目录与 src/、tests/ 三个位置，不递归，因此不会带出
    __pycache__ 等目录。**刻意排除 scratch/**：那里按约定只放探测/临时脚本，
    列出来会把下拉框塞满（实测某迭代任务 37 项里有 33 项是探测脚本）。
    """
    files: list[str] = []
    for subdir in ("src", "tests", ""):
        for rel in _rel_py_files(sandbox_dir, subdir):
            if is_safe_filename(rel) and rel not in files:
                files.append(rel)
    delivery, _ = _locate_delivery(sandbox_dir)
    if delivery and delivery in files:
        files.remove(delivery)
        files.insert(0, delivery)
    return files


def _list_md_files(run_id: str) -> list[str]:
    out_dir = OUTPUTS_ROOT / run_id
    if not out_dir.is_dir():
        return []
    return sorted(p.name for p in out_dir.iterdir() if p.is_file() and p.suffix == ".md")


def _truncate(text: str) -> str:
    if len(text) <= _OUTPUT_LIMIT:
        return text
    return text[:_OUTPUT_LIMIT] + f"\n...（输出已截断，共 {len(text)} 字符）"


# ---------------------------------------------------------------------------
# API 路由
# ---------------------------------------------------------------------------


@app.get("/")
def index():
    # 开发期单文件前端，禁止浏览器缓存，避免改完页面仍加载旧 JS
    return FileResponse(
        STATIC_DIR / "index.html",
        headers={"Cache-Control": "no-store"},
    )


@app.get("/api/runs")
def list_runs():
    return {"runs": _scan_runs()}


@app.post("/api/jobs")
def create_job(body: RequirementBody):
    requirement = body.requirement.strip()
    if not requirement:
        raise HTTPException(status_code=400, detail="需求内容不能为空")
    if not is_api_key_configured():
        raise HTTPException(
            status_code=400,
            detail="尚未在 .env 中配置有效的 OPENAI_API_KEY（DeepSeek 密钥），"
            "请先填写密钥并重启本服务。",
        )

    # 迭代模式：校验基线任务存在（至少有沙箱或产出目录之一）
    base_run_id = (body.base_run_id or "").strip() or None
    if base_run_id is not None:
        if not _RUN_ID_RE.match(base_run_id):
            raise HTTPException(status_code=400, detail="非法的基线任务ID格式")
        if not (SANDBOX_ROOT / base_run_id).is_dir() and not (
            OUTPUTS_ROOT / base_run_id
        ).is_dir():
            raise HTTPException(status_code=404, detail="基线任务不存在，无法迭代")

    # 流程范围校验：必须在建目录之前完成，避免留下空目录变成僵尸任务
    try:
        scope = get_scope(body.scope)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    # 仅支持同范围迭代：跨范围会缺少对应范围的基线文档，提示词必然自相矛盾
    if base_run_id is not None:
        base_scope = get_scope(_scope_key_of(base_run_id))
        if base_scope.key != scope.key:
            raise HTTPException(
                status_code=400,
                detail=f"当前仅支持基于同范围任务的迭代：基线任务为「{base_scope.title}」，"
                f"本次选择的是「{scope.title}」，请改用与基线一致的范围。",
            )

    with _jobs_lock:
        if _is_busy():
            raise HTTPException(status_code=409, detail="已有任务正在运行，请等待完成后再提交")

        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        # 极端情况下秒级冲突，追加序号后缀会破坏 run_id 正则，这里直接拒绝重试即可
        if (SANDBOX_ROOT / run_id).exists() or run_id in _jobs:
            raise HTTPException(status_code=409, detail="同一秒内已有任务，请一秒后重试")

        sandbox_dir = SANDBOX_ROOT / run_id
        output_dir = OUTPUTS_ROOT / run_id
        sandbox_dir.mkdir(parents=True, exist_ok=True)
        output_dir.mkdir(parents=True, exist_ok=True)

        # 迭代模式：复制基线代码与参考文档，构造 Crew 迭代上下文
        iteration_info = None
        if base_run_id is not None:
            assets = _prepare_iteration(base_run_id, sandbox_dir, scope)
            if (
                not assets["copied_src"]
                and not assets["copied_tests"]
                and not assets["copied_docs"]
            ):
                # 基线目录存在但完全是空壳，拒绝并清理新建目录
                shutil.rmtree(sandbox_dir, ignore_errors=True)
                shutil.rmtree(output_dir, ignore_errors=True)
                raise HTTPException(
                    status_code=400,
                    detail="基线任务没有可继承的代码或文档，无法在此基础上迭代",
                )
            iteration_info = {
                "base_run_id": base_run_id,
                "change_request": requirement,
                **assets,
            }
            # 血缘持久化（服务重启后列表仍能展示“基于哪个任务迭代”）
            (output_dir / "iteration.json").write_text(
                json.dumps(
                    {
                        "base_run_id": base_run_id,
                        "change_request": requirement,
                        "created_at": datetime.now().isoformat(timespec="seconds"),
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )

        # 任务元信息持久化（服务重启后回放验收与列表展示都依赖它）
        (output_dir / "run.json").write_text(
            json.dumps(
                {
                    "scope": scope.key,
                    "requirement": requirement,
                    "created_at": datetime.now().isoformat(timespec="seconds"),
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        _jobs[run_id] = {
            "run_id": run_id,
            "requirement": requirement,
            "status": "running",
            "error": "",
            "verdict_reason": "",
            "base_run_id": base_run_id,
            "scope": scope.key,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "stages": _new_stage_states(scope),
        }

    thread = threading.Thread(
        target=_run_crew,
        args=(run_id, requirement, sandbox_dir, output_dir, scope, iteration_info),
        daemon=True,
    )
    thread.start()
    return {"run_id": run_id, "base_run_id": base_run_id, "scope": scope.key}


@app.get("/api/jobs/{run_id}")
def get_job(run_id: str):
    if not _RUN_ID_RE.match(run_id):
        raise HTTPException(status_code=400, detail="非法的任务ID格式")
    job = _jobs.get(run_id)
    if not job:
        # 可能是历史任务（服务重启后内存中无记录）：回放验收结论
        if (SANDBOX_ROOT / run_id).is_dir() or (OUTPUTS_ROOT / run_id).is_dir():
            scope_key = _scope_key_of(run_id)
            accepted, reason = _verdict_for(run_id)
            lineage = _load_lineage(run_id)
            return {
                "run_id": run_id,
                "status": "archived" if accepted else "rejected",
                "stages": [],
                "error": "" if accepted else reason,
                "verdict_reason": reason,
                "requirement": lineage.get("change_request", ""),
                "base_run_id": lineage.get("base_run_id"),
                "scope": scope_key,
            }
        raise HTTPException(status_code=404, detail="任务不存在")
    return job


@app.get("/api/runs/{run_id}/files")
def list_files(run_id: str):
    _validate_run_id(run_id)
    scope = get_scope(_scope_key_of(run_id))
    delivery, _ = _locate_delivery(SANDBOX_ROOT / run_id)
    return {
        "run_id": run_id,
        "scope": scope.key,
        "py_files": _list_py_files(SANDBOX_ROOT / run_id),
        "md_files": _list_md_files(run_id),
        "primary_doc": primary_doc_name(scope),
        "primary_script": delivery,
    }


@app.get("/api/runs/{run_id}/file")
def read_file(run_id: str, kind: str, name: str):
    """kind=sandbox 读取 .py 源码；kind=outputs 读取 .md 文档。"""
    _validate_run_id(run_id)

    if kind == "sandbox":
        if not is_safe_filename(name):
            raise HTTPException(status_code=400, detail="非法文件名")
        path = (SANDBOX_ROOT / run_id / name).resolve()
        allowed_root = (SANDBOX_ROOT / run_id).resolve()
    elif kind == "outputs":
        if not re.match(r"^[A-Za-z0-9_\-.]+\.md$", name):
            raise HTTPException(status_code=400, detail="非法文档名")
        path = (OUTPUTS_ROOT / run_id / name).resolve()
        allowed_root = (OUTPUTS_ROOT / run_id).resolve()
    else:
        raise HTTPException(status_code=400, detail="kind 只能是 sandbox 或 outputs")

    if not path.is_relative_to(allowed_root) or not path.is_file():
        raise HTTPException(status_code=404, detail="文件不存在或路径越界")

    return {"name": name, "kind": kind, "content": path.read_text(encoding="utf-8")}


@app.post("/api/runs/{run_id}/exec")
def exec_file(run_id: str, body: ExecBody):
    """在安全沙箱中运行该任务目录下的指定脚本（与 Agent 使用同一套审计钩子）。"""
    sandbox_dir = _validate_run_id(run_id)
    root = sandbox_dir.resolve()
    if not is_safe_filename(body.filename):
        raise HTTPException(
            status_code=400,
            detail="非法文件名，仅允许 .py，最多一层子目录（如 src/main.py）",
        )

    script_path = (sandbox_dir / body.filename).resolve()
    if not script_path.is_relative_to(root) or not script_path.is_file():
        raise HTTPException(status_code=404, detail="脚本不存在或路径越界")

    # 纵深防御：运行前再做一次 AST 静态检查（文件由 Agent 生成，防外部篡改）
    errors = static_check(script_path.read_text(encoding="utf-8"))
    if errors:
        return {
            "run_id": run_id,
            "filename": body.filename,
            "rejected": True,
            "reason": "静态安全检查未通过:\n" + "\n".join(f"- {e}" for e in errors),
        }

    try:
        proc = execute_sandboxed(
            sandbox_dir,
            script_path,
            timeout=body.timeout,
            stdin=body.stdin,
        )
    except subprocess.TimeoutExpired:
        return {
            "run_id": run_id,
            "filename": body.filename,
            "rejected": False,
            "timed_out": True,
            "returncode": None,
            "stdout": "",
            "stderr": f"执行超过 {body.timeout} 秒，已被沙箱强制终止",
        }

    return {
        "run_id": run_id,
        "filename": body.filename,
        "rejected": False,
        "timed_out": False,
        "returncode": proc.returncode,
        "stdout": _truncate(proc.stdout or ""),
        "stderr": _truncate(proc.stderr or ""),
    }


if __name__ == "__main__":
    import uvicorn

    port = 8000
    url = f"http://127.0.0.1:{port}"
    print("=" * 60)
    print(f"多角色协作任务 Agent Web 服务启动中：{url}")
    print("浏览器将自动打开；如未打开请手动访问上述地址。Ctrl+C 停止服务。")
    print("=" * 60)
    threading.Timer(1.5, lambda: webbrowser.open(url)).start()
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="info")
