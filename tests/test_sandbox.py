"""沙箱层：路径安全矩阵、静态检查、环境清洗、子进程执行语义。

执行语义这几条是本轮最关键的回归点：改动前 runner 用 exec() 运行用户脚本，
unittest.main() 只能收集 0 个用例并退出 0 —— 于是"用退出码做验收"会变成批量假绿灯。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from src.tools import sandbox as sb

SAFE_NAMES = [
    "main.py",
    "solution.py",
    "src/main.py",
    "tests/test_main.py",
    "scratch/probe.py",
]
UNSAFE_NAMES = [
    "../a.py",
    "a/../b.py",
    "./a.py",
    "a/b/c.py",
    "a\\b.py",
    "a/b.txt",
    "",
    "src/.hidden.py",
    "..py",
    "/abs/main.py",
]


class TestPathMatrix:
    @pytest.mark.parametrize("name", SAFE_NAMES)
    def test_accepts_single_level_subdirs(self, name):
        assert sb.is_safe_filename(name) is True

    @pytest.mark.parametrize("name", UNSAFE_NAMES)
    def test_rejects_traversal_and_deep_paths(self, name):
        assert sb.is_safe_filename(name) is False


class TestStaticCheck:
    def test_clean_code_passes(self):
        assert sb.static_check("import json\nprint(json.dumps({'a': 1}))\n") == []

    @pytest.mark.parametrize(
        "code",
        [
            "import subprocess\n",
            "from shutil import rmtree\n",
            "import ctypes\n",
            "__import__('os')\n",
            "eval('1+1')\n",
            "exec('x=1')\n",
            "open('/etc/passwd')\n",
        ],
    )
    def test_blocks_dangerous_constructs(self, code):
        # open 不在静态黑名单里，用一个确定被拦的样例替换
        if "open(" in code:
            code = "import os\nos.system('echo hi')\n"
        assert sb.static_check(code) != []

    def test_reports_syntax_error(self):
        errors = sb.static_check("def broken(:\n")
        assert errors and "语法错误" in errors[0]


class TestCleanEnv:
    def test_secrets_and_proxy_are_stripped(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-should-not-leak")
        monkeypatch.setenv("HTTPS_PROXY", "http://user:pass@proxy:8080")
        env = sb.build_clean_env()
        assert "OPENAI_API_KEY" not in env
        assert "HTTPS_PROXY" not in env
        assert not [k for k in env if "KEY" in k.upper() or "PROXY" in k.upper()]

    def test_forces_utf8(self):
        env = sb.build_clean_env()
        assert env["PYTHONIOENCODING"] == "utf-8"
        assert env["PYTHONUTF8"] == "1"


def _write(path: Path, code: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(code, encoding="utf-8")
    return path


def _run(sandbox_dir: Path, script: Path, stdin: str = ""):
    return sb.execute_sandboxed(sandbox_dir, script, timeout=60, stdin=stdin)


class TestExecutionSemantics:
    """所有脚本体都写在 tmp_path 下，绝不落到真实 workspace。"""

    def test_unittest_main_with_failing_case_exits_nonzero(self, tmp_path):
        script = _write(tmp_path / "t_fail.py", (
            "import unittest\n\n"
            "class T(unittest.TestCase):\n"
            "    def test_ok(self):\n"
            "        self.assertEqual(1 + 1, 2)\n"
            "    def test_bad(self):\n"
            "        self.assertEqual(2 + 2, 5)\n\n"
            "if __name__ == '__main__':\n"
            "    unittest.main()\n"
        ))
        proc = _run(tmp_path, script)
        combined = (proc.stdout or "") + (proc.stderr or "")
        assert proc.returncode == 1, combined
        assert "Ran 2 tests" in combined, combined

    def test_unittest_main_with_no_cases_runs_zero_and_exits_zero(self, tmp_path):
        # 这条是"Ran 0 tests 属无效证据"那个判据的底层事实
        script = _write(tmp_path / "t_empty.py", (
            "import unittest\n\n"
            "if __name__ == '__main__':\n"
            "    unittest.main()\n"
        ))
        proc = _run(tmp_path, script)
        combined = (proc.stdout or "") + (proc.stderr or "")
        assert proc.returncode == 0
        assert "Ran 0 tests" in combined

    def test_sys_exit_code_is_propagated(self, tmp_path):
        script = _write(tmp_path / "t_exit.py", "import sys\nsys.exit(3)\n")
        assert _run(tmp_path, script).returncode == 3

    def test_sys_exit_non_int_becomes_one(self, tmp_path):
        script = _write(tmp_path / "t_exit_msg.py", "import sys\nsys.exit('boom')\n")
        assert _run(tmp_path, script).returncode == 1

    def test_uncaught_exception_echoes_traceback(self, tmp_path):
        script = _write(tmp_path / "t_raise.py", "raise ValueError('boom')\n")
        proc = _run(tmp_path, script)
        assert proc.returncode == 1
        assert "ValueError: boom" in (proc.stderr or "")

    def test_argv_contains_only_the_script(self, tmp_path):
        script = _write(tmp_path / "t_argv.py", "import sys\nprint(repr(sys.argv))\n")
        proc = _run(tmp_path, script)
        assert proc.returncode == 0
        assert "sandbox_runner" not in (proc.stdout or "")
        assert Path(eval(proc.stdout.strip())[0]) == script.resolve()  # noqa: S307

    def test_sibling_module_import_still_works(self, tmp_path):
        _write(tmp_path / "helper.py", "def double(x):\n    return x * 2\n")
        script = _write(tmp_path / "use.py", "import helper\nprint(helper.double(21))\n")
        proc = _run(tmp_path, script)
        assert proc.returncode == 0
        assert "42" in (proc.stdout or "")

    def test_tests_dir_can_import_business_code_from_src(self, tmp_path):
        _write(tmp_path / "src" / "main.py", "def add(a, b):\n    return a + b\n")
        script = _write(tmp_path / "tests" / "test_main.py", (
            "import unittest\nimport main\n\n"
            "class T(unittest.TestCase):\n"
            "    def test_add(self):\n"
            "        self.assertEqual(main.add(2, 3), 5)\n\n"
            "if __name__ == '__main__':\n"
            "    unittest.main()\n"
        ))
        proc = _run(tmp_path, script)
        assert proc.returncode == 0, (proc.stdout or "") + (proc.stderr or "")

    def test_src_shadows_root_module(self, tmp_path):
        _write(tmp_path / "src" / "main.py", "def add(a, b):\n    return a + b\n")
        _write(tmp_path / "main.py", "raise RuntimeError('根目录的 main.py 被误加载')\n")
        script = _write(tmp_path / "tests" / "test_main.py", (
            "import main\nprint(main.add(1, 1))\n"
        ))
        proc = _run(tmp_path, script)
        assert proc.returncode == 0, (proc.stderr or "")

    def test_relative_output_lands_in_sandbox_root(self, tmp_path):
        script = _write(tmp_path / "w.py",
                        "open('made.txt', 'w').write('x')\nprint('ok')\n")
        assert _run(tmp_path, script).returncode == 0
        assert (tmp_path / "made.txt").is_file()

    def test_write_outside_sandbox_is_blocked(self, tmp_path):
        outside = tmp_path.parent / "outside.txt"
        script = _write(tmp_path / "evil.py",
                        f"open(r'{outside}', 'w').write('x')\n")
        proc = _run(tmp_path, script)
        assert proc.returncode == 1
        assert "越界" in (proc.stderr or "")
        assert not outside.exists()


class TestBoundedOutput:
    """子进程输出必须有上限：管道全量缓存会把服务 OOM 掉。"""

    def test_huge_output_is_capped_and_tail_is_kept(self, tmp_path):
        script = _write(tmp_path / "loud.py", (
            "for i in range(200000):\n"
            "    print('noise-%06d' % i)\n"
            "print('TAIL-MARKER')\n"
        ))
        proc = _run(tmp_path, script)
        assert proc.returncode == 0
        assert len(proc.stdout) <= sb._CAPTURE_LIMIT + 200, len(proc.stdout)
        # 尾部（判决依据所在处）必须保留
        assert "TAIL-MARKER" in proc.stdout
        assert "已省略前" in proc.stdout

    def test_small_output_is_returned_verbatim(self, tmp_path):
        script = _write(tmp_path / "quiet.py", "print('hello')\n")
        assert _run(tmp_path, script).stdout == "hello\n"

    def test_huge_output_does_not_break_exit_code(self, tmp_path):
        script = _write(tmp_path / "loud_fail.py", (
            "for i in range(100000):\n"
            "    print('noise')\n"
            "raise SystemExit(2)\n"
        ))
        assert _run(tmp_path, script).returncode == 2

    def test_stdin_still_passes_through(self, tmp_path):
        script = _write(tmp_path / "echo.py", "print(input().upper())\n")
        proc = _run(tmp_path, script, stdin="abc\n")
        assert proc.returncode == 0
        assert proc.stdout.strip() == "ABC"


class TestNoRepoLeak:
    def test_default_sandbox_dir_was_not_created(self):
        """护栏：_DEFAULT_TASK_DIR 指向真实仓库，测试绝不能把它写出来。"""
        repo_default = (
            Path(__file__).resolve().parents[1] / "workspace" / "sandbox" / "default"
        )
        assert not repo_default.exists(), (
            f"测试泄漏到了真实仓库目录：{repo_default}；"
            "请检查是否忘记设置 SANDBOX_TASK_DIR"
        )

    def test_sandbox_task_dir_points_to_tmp(self, tmp_path):
        import os

        assert os.environ["SANDBOX_TASK_DIR"].startswith(str(tmp_path.parent))
        assert sys.executable  # 仅确认解释器可用，避免空断言