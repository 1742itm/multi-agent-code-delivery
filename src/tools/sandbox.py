"""沙箱 Python 代码执行工具（供开发 / 测试 Agent 使用）。

双重安全防护：
1. AST 静态扫描 —— 在代码进入子进程前，拒绝危险导入、危险内建函数与
   dunder 属性访问（经典沙箱逃逸路径）；
2. sandbox_runner.py 审计钩子 —— 运行时拦截进程创建、动态库、
   注册表、越界文件读写。

其他隔离措施：
- 独立子进程执行，超时（默认 60s，上限 60s）强杀；
- 清洗环境变量，不携带任何 API Key / 代理配置；
- 工作目录隔离（每次 Crew 运行独立时间戳目录）；
- stdout/stderr 截断回显。
"""

from __future__ import annotations

import ast
import os
import re
import subprocess
import sys
from pathlib import Path

from crewai.tools import BaseTool
from pydantic import BaseModel, Field

# ---- 安全策略常量 ----------------------------------------------------------

# 禁止导入的模块（根模块名命中即拒绝）
# 策略说明（2026-09-24 调整）：网络已放开（urllib/socket/requests/os/sys 可用），
# 沙箱边界改为“允许联网，但禁止改变本机配置/创建进程/越界文件写入”。
_BLOCKED_MODULES = frozenset(
    {
        # 进程创建（执行任意系统命令 = 可改变本机配置，必须禁止）
        "subprocess",
        "multiprocessing",
        "pty",
        # 递归删除/批量文件操作
        "shutil",
        # 路径与临时文件（文件读写边界由审计钩子统一管控）
        "pathlib",
        "tempfile",
        "glob",
        # 动态执行 / 反序列化 / 导入器（沙箱逃逸通道）
        "ctypes",
        "pickle",
        "marshal",
        "importlib",
        "builtins",
        "code",
        "codeop",
        # Windows 系统配置
        "winreg",
        "msvcrt",
        "platform",
    }
)

# 禁止以“裸名字”直接调用的内建函数
_BLOCKED_BUILTINS = frozenset(
    {
        "eval",
        "exec",
        "compile",
        "__import__",
        "breakpoint",
        "globals",
        "locals",
        "vars",
        "getattr",
        "setattr",
        "delattr",
        "memoryview",
    }
)

# 禁止以“任意对象属性”形式访问的名字（obj.attr 命中即拒绝，无视接收者）。
# 包含两类：
# 1. 沙箱逃逸经典 dunder 跳板；
# 2. 进程/动态执行类危险函数 —— 放开 import os 后，os.system/os.popen/
#    os.startfile(Windows，无可靠审计事件) 等必须在静态层封死，作为
#    运行时审计钩子之外的第二道防线；同时封死 builtins.eval 等经由
#    sys.modules['builtins'] 的间接访问路径。
_BLOCKED_ATTRS = frozenset(
    {
        # dunder 逃逸跳板
        "__subclasses__",
        "__globals__",
        "__builtins__",
        "__bases__",
        "__mro__",
        "__class__",
        "__base__",
        "__dict__",
        "__getattribute__",
        "__reduce__",
        "__reduce_ex__",
        # 进程 / 命令执行（os.system / os.popen / os.startfile / os.exec* 等）
        "system",
        "popen",
        "startfile",
        "fork",
        "forkpty",
        "execv",
        "execve",
        "execl",
        "execle",
        "execlp",
        "execvp",
        "execvpe",
        "spawnl",
        "spawnle",
        "spawnlp",
        "spawnlpe",
        "spawnv",
        "spawnve",
        "spawnvp",
        "spawnvpe",
        "kill",
        "killpg",
        # 动态库
        "dlopen",
        "dlclose",
        # 动态执行 / 反射（封死 builtins.eval、obj.getattr 等间接路径）
        "eval",
        "exec",
        "compile",
        "__import__",
        "getattr",
        "setattr",
        "delattr",
    }
)

_SEGMENT_RE = re.compile(r"^[A-Za-z0-9_\-.]+$")
_MAX_SUBDIR_DEPTH = 1
_OUTPUT_LIMIT = 8000

_DEFAULT_TASK_DIR = (
    Path(__file__).resolve().parents[2] / "workspace" / "sandbox" / "default"
)
_RUNNER_PATH = Path(__file__).resolve().parent / "sandbox_runner.py"

# 子进程最小环境（Windows 运行解释器所需）
_KEEP_ENV = ("SYSTEMROOT", "SYSTEMDRIVE", "COMSPEC", "TEMP", "TMP", "PATHEXT")

# 强制子进程使用 UTF-8，避免 Windows 默认 GBK 导致中文输出乱码
_FORCED_ENV = {"PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"}


class SandboxExecInput(BaseModel):
    code: str = Field(..., description="要在沙箱中执行的完整 Python 3 源代码")
    filename: str = Field(
        default="solution.py",
        description="代码保存的相对路径，仅允许字母数字下划线中划线，"
        "最多一层子目录（如 main.py、src/main.py、tests/test_main.py），"
        ".py 后缀，保存在当前沙箱目录内，可用于多文件协作",
    )
    timeout: int = Field(
        default=60, ge=1, le=60, description="执行超时秒数（1-60，默认 60）"
    )


class _SecurityChecker(ast.NodeVisitor):
    """对用户代码做静态安全检查。"""

    def __init__(self) -> None:
        self.errors: list[str] = []

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            root = alias.name.split(".")[0]
            if root in _BLOCKED_MODULES:
                self.errors.append(f"第 {node.lineno} 行: 禁止导入模块 '{alias.name}'")
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if node.module:
            root = node.module.split(".")[0]
            if root in _BLOCKED_MODULES:
                self.errors.append(f"第 {node.lineno} 行: 禁止从模块 '{node.module}' 导入")
        self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> None:
        if node.id in _BLOCKED_BUILTINS:
            self.errors.append(f"第 {node.lineno} 行: 禁止调用内建函数 '{node.id}'")
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if node.attr in _BLOCKED_ATTRS:
            self.errors.append(f"第 {node.lineno} 行: 禁止访问属性 '{node.attr}'")
        self.generic_visit(node)


def static_check(code: str) -> list[str]:
    """对源码做 AST 静态安全检查，返回错误信息列表（空列表表示通过）。"""
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return [f"代码存在语法错误: 第 {exc.lineno} 行: {exc.msg}"]
    checker = _SecurityChecker()
    checker.visit(tree)
    return checker.errors


def is_safe_filename(filename: str) -> bool:
    """校验脚本的相对路径是否符合沙箱命名规则。

    - 最多一层子目录（如 src/main.py、tests/test_main.py）
    - 各段仅允许字母/数字/下划线/中划线，不以点开头（顺带挡掉 . / .. 与隐藏文件）
    - 必须 .py 结尾；不接受绝对路径与反斜杠分隔
    """
    if not filename or "\\" in filename or filename.startswith("/"):
        return False
    parts = filename.split("/")
    if len(parts) > _MAX_SUBDIR_DEPTH + 1:
        return False
    if not parts[-1].endswith(".py"):
        return False
    return all(
        _SEGMENT_RE.match(part) and not part.startswith(".") for part in parts
    )


def build_clean_env() -> dict[str, str]:
    """构造沙箱子进程的最小环境（清洗密钥/代理，强制 UTF-8）。"""
    clean_env = {key: os.environ[key] for key in _KEEP_ENV if key in os.environ}
    clean_env.update(_FORCED_ENV)
    return clean_env


def execute_sandboxed(
    task_dir: Path,
    script_path: Path,
    timeout: int = 60,
    stdin: str = "",
) -> subprocess.CompletedProcess:
    """在隔离子进程中执行沙箱目录内的脚本（审计钩子 + 清洗环境 + 超时）。

    供 Agent 工具与 Web 运行接口共用。超时抛出 subprocess.TimeoutExpired。
    """
    return subprocess.run(
        [
            sys.executable,
            str(_RUNNER_PATH),
            str(task_dir),
            str(script_path),
        ],
        cwd=str(task_dir),
        env=build_clean_env(),
        timeout=timeout,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        input=stdin or "",
    )


class SafePythonExecTool(BaseTool):  # type: ignore[misc]
    name: str = "sandbox_python_exec"
    description: str = (
        "在安全沙箱中执行 Python 3 代码并返回 stdout/stderr 与退出码。"
        "允许正常网络访问（urllib/requests 等）与沙箱目录内文件读写；"
        "禁止进程创建、系统命令、动态库加载、注册表操作与沙箱目录外文件读写。"
        "代码按 filename 保存到沙箱目录后执行，filename 最多可带一层子目录"
        "（业务代码放 src/、正式测试放 tests/、临时脚本放 scratch/），"
        "可多次调用以实现多文件协作。"
    )
    args_schema: type[BaseModel] = SandboxExecInput

    def _run(self, code: str, filename: str = "solution.py", timeout: int = 60) -> str:
        # 1. 文件名安全校验
        if not is_safe_filename(filename):
            return (
                f"[沙箱拒绝] 非法文件名 '{filename}'：仅允许字母/数字/下划线/中划线，"
                "最多一层子目录（如 src/main.py），且必须以 .py 结尾"
            )

        # 2. AST 静态扫描
        errors = static_check(code)
        if errors:
            return "[沙箱拒绝] 静态安全检查未通过:\n" + "\n".join(
                f"- {e}" for e in errors
            )

        # 3. 定位本次任务的沙箱目录（main.py / web_server 通过环境变量注入）
        task_dir = Path(os.environ.get("SANDBOX_TASK_DIR", str(_DEFAULT_TASK_DIR)))
        task_dir.mkdir(parents=True, exist_ok=True)
        script_path = task_dir / filename
        # 支持 src/、tests/ 这类一层子目录：落盘前先建父目录
        script_path.parent.mkdir(parents=True, exist_ok=True)
        script_path.write_text(code, encoding="utf-8")

        # 4. 清洗环境变量后在隔离子进程中运行（强制 UTF-8，不携带任何密钥）
        try:
            proc = execute_sandboxed(task_dir, script_path, timeout=timeout)
        except subprocess.TimeoutExpired:
            return f"[沙箱超时] 代码执行超过 {timeout} 秒，已被强制终止"

        return _format_result(proc.returncode, proc.stdout, proc.stderr)


def _format_result(returncode: int, stdout: str, stderr: str) -> str:
    parts = [f"退出码: {returncode}"]
    if stdout.strip():
        parts.append("标准输出:\n" + _truncate(stdout.rstrip()))
    if stderr.strip():
        parts.append("错误输出:\n" + _truncate(stderr.rstrip()))
    if returncode == 0:
        parts.append("结论: 执行成功")
    else:
        parts.append("结论: 执行失败，请根据错误信息修正代码后重试")
    return "\n".join(parts)


def _truncate(text: str) -> str:
    if len(text) <= _OUTPUT_LIMIT:
        return text
    return text[:_OUTPUT_LIMIT] + f"\n...（输出已截断，共 {len(text)} 字符）"
