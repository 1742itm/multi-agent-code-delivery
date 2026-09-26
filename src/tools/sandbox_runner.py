"""沙箱运行器（在隔离子进程中执行）。

本脚本由 SafePythonExecTool 以子进程方式启动，先于用户代码安装审计钩子，
在运行时阻断高危行为：

- 进程创建：subprocess.Popen / os.system / os.exec* / os.spawn*
- 动态库加载：ctypes.dlopen / ctypes.dlclose
- 注册表操作：winreg.*
- 危险文件操作：shutil.rmtree
- 文件越界：写入只能发生在沙箱目录；读取仅限沙箱目录与 Python 解释器目录
  （保证正常 import 标准库 / 第三方库可用）

注意：AST 静态扫描（sandbox.py）是第一道防线，本审计钩子是第二道防线。
"""

from __future__ import annotations

import io
import os
import sys
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

# argv[1] = 沙箱工作目录，argv[2] = 用户代码文件
SANDBOX_DIR = Path(sys.argv[1]).resolve()
USER_FILE = Path(sys.argv[2]).resolve()

# 允许读取的根目录：沙箱目录、venv 目录、基础解释器目录（标准库）
_READ_ROOTS = (
    SANDBOX_DIR,
    Path(sys.prefix).resolve(),
    Path(sys.base_prefix).resolve(),
)

# 运行时禁止事件（网络已放开；进程创建 / 动态库 / 注册表 / 递归删除仍禁止）
_BLOCKED_EVENTS = frozenset(
    {
        "subprocess.Popen",
        "os.system",
        "os.popen",
        "os.startfile",
        "os.exec",
        "os.fork",
        "os.forkpty",
        "os.spawn",
        "os.kill",
        "os.killpg",
        "ctypes.dlopen",
        "ctypes.dlclose",
        "winreg.create_key",
        "winreg.set_value",
        "winreg.delete_key",
        "winreg.delete_value",
        "winreg.connect_registry",
        "shutil.rmtree",
    }
)

# 路径首参数必须落在沙箱目录内的事件（删除 / 改名 / 建目录）
_SANDBOX_ONLY_EVENTS = frozenset(
    {
        "os.remove",
        "os.unlink",
        "os.rmdir",
        "os.rename",
        "os.replace",
        "os.mkdir",
        "os.makedirs",
    }
)

# 路径首参数必须落在“可读根目录”内的事件（目录枚举）
_READABLE_PATH_EVENTS = frozenset({"os.listdir", "os.scandir", "os.chdir"})

_WRITE_FLAGS = ("w", "a", "x", "+")


def _is_under(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _audit_hook(event: str, args: tuple) -> None:  # noqa: C901
    # 精确命中的高危事件
    for blocked in _BLOCKED_EVENTS:
        if event == blocked or event.startswith(blocked + "."):
            raise PermissionError(f"沙箱已禁止高危操作: {event}")

    # 删除 / 改名 / 建目录：仅允许沙箱目录内
    if event in _SANDBOX_ONLY_EVENTS and args:
        # rename/replace 有两个路径参数；其余事件（remove/unlink/rmdir/
        # mkdir/makedirs）只有第一个是路径，后续参数是 mode/dir_fd 等整数，
        # 不能当路径解析（Windows 上会抛 TypeError 导致误拦正常建目录）
        if event in ("os.rename", "os.replace"):
            path_args = args[:2]
        else:
            path_args = args[:1]
        for raw_path in path_args:
            # 仅校验字符串/PathLike 路径参数；非路径参数（fd 等）交给操作系统
            if not isinstance(raw_path, (str, os.PathLike)):
                continue
            try:
                target = Path(raw_path).resolve()
            except (OSError, ValueError, TypeError):
                raise PermissionError("沙箱无法解析文件路径")
            if not _is_under(target, SANDBOX_DIR):
                raise PermissionError(f"沙箱已禁止操作沙箱外路径: {target}")

    # 目录枚举 / 切换工作目录：仅允许可读根目录
    if event in _READABLE_PATH_EVENTS and args:
        try:
            target = Path(args[0]).resolve()
        except (OSError, ValueError, TypeError):
            raise PermissionError("沙箱无法解析目录路径")
        if not any(_is_under(target, root) for root in _READ_ROOTS):
            raise PermissionError(f"沙箱已禁止访问沙箱外目录: {target}")

    # 文件读写边界控制
    if event == "open" and args:
        raw_path = args[0]
        mode = args[1] if len(args) > 1 else "r"
        if isinstance(raw_path, int):
            return  # 文件描述符形式（通常来自解释器内部）
        try:
            target = Path(raw_path).resolve()
        except (OSError, ValueError):
            raise PermissionError("沙箱无法解析文件路径")
        is_write = any(flag in str(mode) for flag in _WRITE_FLAGS)
        if is_write:
            allowed = _is_under(target, SANDBOX_DIR)
        else:
            allowed = any(_is_under(target, root) for root in _READ_ROOTS)
        if not allowed:
            action = "写入" if is_write else "读取"
            raise PermissionError(f"沙箱已禁止越界文件{action}: {target}")


def main() -> int:
    if not _is_under(USER_FILE, SANDBOX_DIR):
        print("安全错误: 待执行文件不在沙箱目录内", file=sys.stderr)
        return 2

    sys.addaudithook(_audit_hook)

    # 让用户脚本能 import 同一沙箱目录内的其他文件（多文件协作）
    sys.path.insert(0, str(USER_FILE.parent))

    source = USER_FILE.read_text(encoding="utf-8")
    compiled = compile(source, str(USER_FILE), "exec")

    user_globals = {
        "__name__": "__main__",
        "__file__": str(USER_FILE),
    }

    stdout_buf, stderr_buf = io.StringIO(), io.StringIO()
    try:
        with redirect_stdout(stdout_buf), redirect_stderr(stderr_buf):
            exec(compiled, user_globals)  # noqa: S102
    except SystemExit as exc:  # 用户代码调用 sys.exit 被禁后理论上到不了这里
        code = exc.code if isinstance(exc.code, int) else 1
        _dump(stdout_buf, stderr_buf)
        return int(code or 0)
    except BaseException:  # noqa: BLE001 - 沙箱需要回显全部异常给调用方
        import traceback

        traceback.print_exc(file=stderr_buf)
        _dump(stdout_buf, stderr_buf)
        return 1

    _dump(stdout_buf, stderr_buf)
    return 0


def _dump(stdout_buf: io.StringIO, stderr_buf: io.StringIO) -> None:
    out, err = stdout_buf.getvalue(), stderr_buf.getvalue()
    if out:
        sys.stdout.write(out)
    if err:
        sys.stderr.write(err)


if __name__ == "__main__":
    sys.exit(main())
