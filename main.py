"""多角色协作任务 Agent 入口。

流水线：需求分析师 -> 代码开发工程师 -> 测试工程师 -> 文档撰写员
所有代码执行均在安全沙箱中完成（AST 静态扫描 + 审计钩子 + 隔离子进程）。

用法（在项目根目录、已激活 .venv 的终端中）：

    # 直接传入需求
    python main.py "实现一个支持加减乘除与除零保护的命令行计算器"

    # 从文件读取需求
    python main.py --file requirement.txt

    # 不传参数则进入交互输入（输入空行结束）
    python main.py
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

from crewai import Crew, Process

from src.agents import build_agents
from src.config import PROJECT_ROOT, get_llm, is_api_key_configured
from src.tasks import build_tasks

WORKSPACE_ROOT = PROJECT_ROOT / "workspace"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="多角色协作任务 Agent（需求分析/开发/测试/文档）"
    )
    parser.add_argument("requirement", nargs="?", help="用户的原始需求文本")
    parser.add_argument("--file", "-f", help="从文本文件读取需求")
    return parser.parse_args()


def resolve_requirement(args: argparse.Namespace) -> str:
    if args.file:
        path = Path(args.file)
        if not path.is_absolute():
            path = Path.cwd() / path
        if not path.is_file():
            print(f"错误：需求文件不存在: {path}", file=sys.stderr)
            sys.exit(1)
        return path.read_text(encoding="utf-8").strip()

    if args.requirement:
        return args.requirement.strip()

    print("请输入需求（可输入多行，单独一行输入 END 结束）：")
    lines: list[str] = []
    for line in sys.stdin:
        if line.strip() == "END":
            break
        lines.append(line.rstrip("\n"))
    return "\n".join(lines).strip()


def main() -> None:
    args = parse_args()
    requirement = resolve_requirement(args)

    if not requirement:
        print("错误：需求内容为空，已退出。", file=sys.stderr)
        sys.exit(1)

    if not is_api_key_configured():
        print(
            "错误：未在 .env 中配置有效的 OPENAI_API_KEY（DeepSeek 密钥）。\n"
            f"请编辑 {PROJECT_ROOT / '.env'} 后重试。",
            file=sys.stderr,
        )
        sys.exit(1)

    # 本轮运行的隔离目录：沙箱代码 + 产出文档
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    sandbox_dir = WORKSPACE_ROOT / "sandbox" / run_id
    output_dir = WORKSPACE_ROOT / "outputs" / run_id
    sandbox_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 沙箱工具通过该环境变量定位本轮工作目录
    os.environ["SANDBOX_TASK_DIR"] = str(sandbox_dir)

    print("=" * 70)
    print(f"需求：{requirement}")
    print(f"沙箱目录：{sandbox_dir}")
    print(f"产出目录：{output_dir}")
    print("=" * 70)

    llm = get_llm()
    agents = build_agents(llm)
    tasks = build_tasks(agents, output_dir)

    crew = Crew(
        agents=[
            agents["analyst"],
            agents["developer"],
            agents["tester"],
            agents["writer"],
        ],
        tasks=tasks,
        process=Process.sequential,
        verbose=True,
        cache=False,
    )

    result = crew.kickoff(
        inputs={
            "requirement": requirement,
            "sandbox_dir": str(sandbox_dir),
        }
    )

    print("\n" + "=" * 70)
    print("协作流程已完成")
    print(f"沙箱代码目录：{sandbox_dir}")
    print(f"产出文档目录：{output_dir}")
    for doc in sorted(output_dir.glob("*.md")):
        print(f"  - {doc.name}")
    print("=" * 70)
    print(result)


if __name__ == "__main__":
    main()
