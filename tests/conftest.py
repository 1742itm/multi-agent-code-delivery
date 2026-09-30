"""测试公共配置：环境隔离、联网保险丝、共享 fixture。

顺序很重要：**本文件顶部的环境变量必须在导入任何项目模块之前生效**，
否则 .env 里的真实密钥会先被 python-dotenv 读进进程，遥测开关也可能来不及设。
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

# ---------------------------------------------------------------------------
# 1) 环境隔离：必须早于任何项目 import（含 src.*）
# ---------------------------------------------------------------------------

# 让 python-dotenv 完全失效，真实密钥不会进入测试进程
os.environ["PYTHON_DOTENV_DISABLED"] = "1"
# 遥测/版本检查全部关掉（crewai 首次 import 前生效）
os.environ["OTEL_SDK_DISABLED"] = "true"
os.environ["CREWAI_DISABLE_TELEMETRY"] = "true"
os.environ["CREWAI_DISABLE_TRACKING"] = "true"
os.environ["CREWAI_DISABLE_VERSION_CHECK"] = "1"
os.environ["ANONYMIZED_TELEMETRY"] = "False"
os.environ["CI"] = "true"
# crewai 构造 Crew 时会往用户目录写库，且路径随 CWD 变化 —— 指到临时目录
_TEST_STORAGE = Path(tempfile.mkdtemp(prefix="agentdemo-tests-"))
os.environ["CREWAI_STORAGE_DIR"] = str(_TEST_STORAGE)

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    # 兜底：主机制是 pytest.ini 的 pythonpath = .
    sys.path.insert(0, str(ROOT))

import pytest  # noqa: E402

import src.config as config  # noqa: E402
import web_server as ws  # noqa: E402
from crewai import LLM, Crew  # noqa: E402
from src.agents import build_agents  # noqa: E402

import main as cli_entry  # noqa: E402  （只为把它的 get_llm 绑定也纳入保险丝）

_FUSE_MESSAGE = "测试环境禁止真实调用 LLM / 访问网络"


def _boom(*args, **kwargs):
    raise AssertionError(_FUSE_MESSAGE)


def _patch_if_present(monkeypatch, target, name, replacement) -> None:
    """只在属性存在时替换，避免 crewai 版本变动导致 conftest 直接报错。"""
    if hasattr(target, name):
        monkeypatch.setattr(target, name, replacement)


# ---------------------------------------------------------------------------
# 2) 联网保险丝 + 全局态清理（autouse）
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _fuse_and_clean_state(monkeypatch, tmp_path):
    """任何真实联网出口都换成"调用即失败"，并清理会跨用例污染的全局态。

    保险丝有冗余层次，任一层都足以拦住"意外烧真钱"：
    get_llm 入口（三个模块各绑了一份）、Crew.kickoff 系列、LLM.call 系列、
    以及真正的出口 openai 的 chat completions create。
    """
    monkeypatch.setattr(config, "get_llm", _boom)
    monkeypatch.setattr(ws, "get_llm", _boom)
    monkeypatch.setattr(cli_entry, "get_llm", _boom)

    for name in ("kickoff", "akickoff", "kickoff_async", "kickoff_for_each"):
        _patch_if_present(monkeypatch, Crew, name, _boom)
    for name in ("call", "acall"):
        _patch_if_present(monkeypatch, LLM, name, _boom)

    try:
        from openai.resources.chat.completions import (
            AsyncCompletions,
            Completions,
        )

        monkeypatch.setattr(Completions, "create", _boom)
        monkeypatch.setattr(AsyncCompletions, "create", _boom)
    except ImportError:  # pragma: no cover - openai 是 crewai 的依赖，正常都在
        pass

    # 模块级任务表：不清会被上一个用例的 running 状态传染（_is_busy）
    monkeypatch.setattr(ws, "_jobs", {})
    # 沙箱工具默认回退到真实仓库的 workspace/sandbox/default，必须改指临时目录
    monkeypatch.setenv("SANDBOX_TASK_DIR", str(tmp_path / "sandbox"))

    yield


# ---------------------------------------------------------------------------
# 3) 显式 fixture（刻意不做 autouse：test_local_workspace 需要真实目录）
# ---------------------------------------------------------------------------


@pytest.fixture
def dummy_llm():
    """离线可用的假 LLM：只构造对象，绝不发起请求。

    base_url 用保留域名 example.invalid，即使保险丝失效也连不通。
    """
    return LLM(
        model="deepseek-chat",
        base_url="https://example.invalid/v1",
        api_key="sk-dummy",
        custom_openai=True,
        temperature=0.3,
        max_tokens=8192,
    )


@pytest.fixture
def agents(dummy_llm):
    return build_agents(dummy_llm)


@pytest.fixture
def tmp_roots(monkeypatch, tmp_path):
    """把 web_server 的沙箱/产出根目录改指到临时目录。

    web_server 的全部文件访问都经这两个模块常量，因此这一个 fixture
    就能隔离所有读写的副作用（不会碰到真实 workspace）。
    刻意不做成 autouse：test_local_workspace 需要真实目录。
    """
    from types import SimpleNamespace

    sandbox = tmp_path / "sandbox"
    outputs = tmp_path / "outputs"
    sandbox.mkdir()
    outputs.mkdir()
    monkeypatch.setattr(ws, "SANDBOX_ROOT", sandbox)
    monkeypatch.setattr(ws, "OUTPUTS_ROOT", outputs)
    return SimpleNamespace(sandbox=sandbox, outputs=outputs, root=tmp_path)


@pytest.fixture
def make_sandbox(tmp_path):
    """在临时目录里造一个沙箱 + 产出目录，用于直接调用验收相关函数。"""

    def _make(
        main_code: str | None = "def add(a, b):\n    return a + b\n",
        test_code: str | None = None,
        report: str | None = None,
    ):
        sandbox = tmp_path / "sandbox" / "19000101_000000"
        outputs = tmp_path / "outputs" / "19000101_000000"
        sandbox.mkdir(parents=True, exist_ok=True)
        outputs.mkdir(parents=True, exist_ok=True)
        if main_code is not None:
            (sandbox / "src").mkdir(exist_ok=True)
            (sandbox / "src" / "main.py").write_text(main_code, encoding="utf-8")
        if test_code is not None:
            (sandbox / "tests").mkdir(exist_ok=True)
            (sandbox / "tests" / "test_main.py").write_text(
                test_code, encoding="utf-8"
            )
        if report is not None:
            (outputs / "03_test_report.md").write_text(report, encoding="utf-8")
        return sandbox, outputs

    return _make