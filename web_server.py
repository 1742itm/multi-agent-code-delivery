"""多角色协作任务 Agent —— 简易 Web 服务。

职责：
1. 接收前端提交的需求，后台串行启动 CrewAI 四角色流水线；
2. 通过阶段回调向前端提供实时进度（前端轮询）；
3. 列出/查看历史任务的脚本与产出文档；
4. 在与 Agent 完全相同的安全沙箱中运行任务脚本（支持 stdin 输入）。

仅绑定 127.0.0.1，不对外暴露。启动：

    .venv\\Scripts\\python.exe web_server.py
"""

from __future__ import annotations

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
_STAGE_NAMES = ["需求分析", "代码开发", "测试验证", "文档撰写"]

app = FastAPI(title="多角色协作任务 Agent")


# ---------------------------------------------------------------------------
# 任务管理（内存态；同一时刻只允许一个 Crew 运行）
# ---------------------------------------------------------------------------

_jobs_lock = threading.Lock()
_jobs: dict[str, dict] = {}


def _is_busy() -> bool:
    return any(job["status"] == "running" for job in _jobs.values())


def _new_stage_states() -> list[dict]:
    stages = [
        {"name": name, "status": "pending", "output": ""} for name in _STAGE_NAMES
    ]
    stages[0]["status"] = "running"
    return stages


# 探测/调试/测试类文件名前缀，不作为“开发交付主脚本”
_NON_DELIVERY_PREFIXES = ("probe", "inspect", "test_", "conclude", "read_", "extra_")

# 迭代复制时排除的过程性/探测脚本前缀（test_main.py 等正式测试保留，供回归）
_JUNK_PREFIXES = ("probe", "inspect", "read_", "extra_", "conclude")

# 迭代时复制进新沙箱的旧文档映射（旧产出名 -> 新沙箱内参考文件名）
_BASE_DOC_MAP = {
    "01_requirements.md": "base_requirements.md",
    "03_test_report.md": "base_test_report.md",
    "04_final_report.md": "base_final_report.md",
}


def _load_lineage(run_id: str) -> dict:
    """读取任务血缘文件（服务重启后仍可获知其基于哪个历史任务迭代）。"""
    path = OUTPUTS_ROOT / run_id / "iteration.json"
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def _prepare_iteration(base_run_id: str, new_sandbox: Path, new_outputs: Path) -> dict:
    """把历史任务的代码与文档复制到新任务沙箱，返回迭代上下文。

    - 业务 .py 与正式测试全部复制（探测脚本排除），开发可在旧代码上增量修改；
    - 旧需求/测试报告/最终报告复制为 base_*.md（放在沙箱内，Agent 可直接读取）；
    - 血缘写入 outputs/<new>/iteration.json 持久化。
    """
    base_sandbox = SANDBOX_ROOT / base_run_id
    base_outputs = OUTPUTS_ROOT / base_run_id

    copied_py: list[str] = []
    if base_sandbox.is_dir():
        for p in sorted(base_sandbox.iterdir()):
            if not (p.is_file() and p.suffix == ".py"):
                continue
            if p.name.startswith(_JUNK_PREFIXES):
                continue
            shutil.copy2(p, new_sandbox / p.name)
            copied_py.append(p.name)

    copied_docs: list[str] = []
    if base_outputs.is_dir():
        for old_name, new_name in _BASE_DOC_MAP.items():
            src = base_outputs / old_name
            if src.is_file():
                shutil.copy2(src, new_sandbox / new_name)
                copied_docs.append(new_name)

    return {"copied_py": copied_py, "copied_docs": copied_docs}

# 测试报告中代表验收失败的明确信号
_FAIL_SIGNALS = ("未通过验收", "全部阻塞", "通过率 0.0%", "通过率:0.0%")


def _evaluate_delivery(sandbox_dir: Path, output_dir: Path) -> tuple[bool, str]:
    """流水线结束后的客观交付验收。

    两层判定：
    1. 沙箱中必须真实存在开发交付的主脚本（main.py 优先，其次非探测类 .py）；
    2. 测试报告中不得出现明确的验收失败结论。
    返回 (是否通过, 原因说明)。
    """
    py_files = sorted(
        p.name
        for p in sandbox_dir.iterdir()
        if p.is_file() and p.suffix == ".py"
    ) if sandbox_dir.is_dir() else []

    main_file = sandbox_dir / "main.py"
    if main_file.is_file():
        delivery = "main.py"
    else:
        candidates = [
            name
            for name in py_files
            if not name.startswith(_NON_DELIVERY_PREFIXES)
        ]
        if not candidates:
            return False, (
                "验收失败：沙箱目录中未找到开发交付的主脚本（main.py 或其他业务 .py），"
                f"仅有 {len(py_files)} 个探测/测试类文件，开发产出未真实落盘。"
            )
        delivery = candidates[0]

    # 主脚本为空文件也视为无效交付
    if (sandbox_dir / delivery).stat().st_size == 0:
        return False, f"验收失败：交付脚本 {delivery} 为空文件。"

    test_report = output_dir / "03_test_report.md"
    if test_report.is_file():
        content = test_report.read_text(encoding="utf-8", errors="replace")
        hit = next((sig for sig in _FAIL_SIGNALS if sig in content), None)
        if hit:
            return False, (
                f"验收失败：测试工程师在测试报告中给出未通过结论（命中信号“{hit}”），"
                f"交付脚本 {delivery} 未通过用例验收，请查看测试报告了解详情。"
            )

    return True, f"验收通过：交付脚本 {delivery} 已落盘，测试报告未报告失败结论。"


def _run_crew(
    run_id: str,
    requirement: str,
    sandbox_dir: Path,
    output_dir: Path,
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
            stage_callback=on_stage,
            iteration_info=iteration_info,
        )

        crew = Crew(
            agents=[
                agents["analyst"],
                agents["developer"],
                agents["tester"],
                agents["writer"],
            ],
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
        # 流程走完 ≠ 交付成功：必须通过客观验收（主脚本落盘 + 测试报告无失败结论）
        accepted, reason = _evaluate_delivery(sandbox_dir, output_dir)
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
        else:
            # 服务重启后的历史任务：实时回放客观验收，失败任务保持红灯
            accepted, _ = _evaluate_delivery(SANDBOX_ROOT / rid, OUTPUTS_ROOT / rid)
            status = "archived" if accepted else "rejected"
            requirement = ""
            base_run_id = _load_lineage(rid).get("base_run_id")
        result.append(
            {
                "run_id": rid,
                "status": status,
                "requirement": requirement,
                "base_run_id": base_run_id,
                "has_sandbox": rid in sandbox_ids,
                "has_outputs": rid in output_ids,
            }
        )
    return result


def _list_py_files(sandbox_dir: Path) -> list[str]:
    return sorted(
        p.name
        for p in sandbox_dir.iterdir()
        if p.is_file() and is_safe_filename(p.name)
    )


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
            assets = _prepare_iteration(base_run_id, sandbox_dir, output_dir)
            if not assets["copied_py"] and not assets["copied_docs"]:
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

        _jobs[run_id] = {
            "run_id": run_id,
            "requirement": requirement,
            "status": "running",
            "error": "",
            "verdict_reason": "",
            "base_run_id": base_run_id,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "stages": _new_stage_states(),
        }

    thread = threading.Thread(
        target=_run_crew,
        args=(run_id, requirement, sandbox_dir, output_dir, iteration_info),
        daemon=True,
    )
    thread.start()
    return {"run_id": run_id, "base_run_id": base_run_id}


@app.get("/api/jobs/{run_id}")
def get_job(run_id: str):
    if not _RUN_ID_RE.match(run_id):
        raise HTTPException(status_code=400, detail="非法的任务ID格式")
    job = _jobs.get(run_id)
    if not job:
        # 可能是历史任务（服务重启后内存中无记录）：回放验收结论
        if (SANDBOX_ROOT / run_id).is_dir() or (OUTPUTS_ROOT / run_id).is_dir():
            accepted, reason = _evaluate_delivery(
                SANDBOX_ROOT / run_id, OUTPUTS_ROOT / run_id
            )
            lineage = _load_lineage(run_id)
            return {
                "run_id": run_id,
                "status": "archived" if accepted else "rejected",
                "stages": [],
                "error": "" if accepted else reason,
                "verdict_reason": reason,
                "requirement": lineage.get("change_request", ""),
                "base_run_id": lineage.get("base_run_id"),
            }
        raise HTTPException(status_code=404, detail="任务不存在")
    return job


@app.get("/api/runs/{run_id}/files")
def list_files(run_id: str):
    _validate_run_id(run_id)
    return {
        "run_id": run_id,
        "py_files": _list_py_files(SANDBOX_ROOT / run_id),
        "md_files": _list_md_files(run_id),
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

    if not str(path).startswith(str(allowed_root)) or not path.is_file():
        raise HTTPException(status_code=404, detail="文件不存在或路径越界")

    return {"name": name, "kind": kind, "content": path.read_text(encoding="utf-8")}


@app.post("/api/runs/{run_id}/exec")
def exec_file(run_id: str, body: ExecBody):
    """在安全沙箱中运行该任务目录下的指定脚本（与 Agent 使用同一套审计钩子）。"""
    sandbox_dir = _validate_run_id(run_id)
    if not is_safe_filename(body.filename):
        raise HTTPException(status_code=400, detail="非法文件名，仅允许 .py 且不含路径")

    script_path = (sandbox_dir / body.filename).resolve()
    if not str(script_path).startswith(str(sandbox_dir.resolve())) or not script_path.is_file():
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
