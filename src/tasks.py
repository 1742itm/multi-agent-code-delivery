"""四个顺序衔接的 Task 定义。

流水线：需求分析 -> 代码开发（沙箱自测）-> 测试验证（沙箱）-> 最终报告。
每个 Task 的产出同时写入 workspace/outputs/<run_id>/ 下的 Markdown 文件。
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

from crewai import Task

# 阶段回调签名：(阶段序号 0-3, 阶段名称, TaskOutput) -> None
StageCallback = Callable[[int, str, "object"], None]


def _iteration_brief(role: str, info: dict) -> str:
    """迭代模式下注入到各角色任务描述前的指引。

    info 字段均来自服务端校验/复制结果（base_run_id 经正则校验，
    文件名为沙箱安全文件名），不含用户自由文本，可安全直接拼接；
    用户的“修改要求”仍通过 {requirement} 占位符注入，避免花括号插值问题。
    """
    py_list = "、".join(info.get("copied_py") or []) or "（无）"
    doc_list = "、".join(info.get("copied_docs") or []) or "（无）"

    common = (
        "【迭代任务模式 · 最高优先级背景】\n"
        f"本次不是从零开发，而是在历史任务 {info['base_run_id']} 的交付物基础上"
        "进行修改迭代。下文“用户需求”的内容是针对已有版本的【修改要求】，"
        "请务必结合基线资产工作，而非无视旧代码重新实现。\n"
        f"- 基线代码已复制到当前沙箱目录：{py_list}；\n"
        f"- 基线文档已复制到沙箱目录供阅读：{doc_list}"
        "（base_requirements.md=原需求分析，base_test_report.md=原测试报告，"
        "base_final_report.md=原最终报告）。\n"
        "- 开始前必须先用 sandbox_python_exec 工具（open 读取）查阅上述基线代码与文档，"
        "基于真实旧代码开展工作。\n\n"
    )

    role_hints = {
        "analyst": (
            "你的任务是【变更影响分析】，输出文档须包含：\n"
            "1. 修改要求概述；\n"
            "2. 基线现状：阅读 base_requirements.md 与沙箱内旧代码，"
            "概述已有功能与函数结构；\n"
            "3. 变更影响分析：逐条说明修改要求影响哪些文件/函数/数据结构/"
            "验收条目，哪些原有功能保持不变（必须保留的回归面）；\n"
            "4. 修改后技术方案：给出需要新增/改动的函数签名与实现思路，"
            "优先复用旧代码结构；\n"
            "5. 验收标准：新增/变更的验收条目，并明确列出必须继续通过的"
            "原有回归验收条目；\n"
            "6. 范围外说明。\n\n"
        ),
        "developer": (
            "【在现有代码上增量修改】\n"
            "1. 先通读沙箱中已有的 main.py 等基线代码，理解原有结构；\n"
            "2. 直接在现有文件上修改并通过 sandbox_python_exec 覆盖保存，"
            "只改动实现修改要求所必需的部分，保持原有功能与接口向后兼容，"
            "禁止推倒重写、禁止把旧代码改名旁路；\n"
            "3. 自测必须同时覆盖：原有功能的回归验证 + 本次新增/变更逻辑；\n"
            "4. 实现说明中需单列“本次变更清单”（改了哪些函数/新增了什么）"
            "与“回归自测结果”。\n\n"
        ),
        "tester": (
            "【回归测试 + 增量测试】\n"
            "1. 沙箱中已有基线的 test_main.py，请先阅读并保留其全部原有用例作为"
            "回归用例（可按新接口调整断言，但不得删除对原有功能的覆盖）；\n"
            "2. 针对本次修改要求新增正常/边界/异常用例；\n"
            "3. 在沙箱实际运行完整测试，报告中须分“回归用例结果”与"
            "“新增用例结果”两组统计；任何回归失败都属于阻断性问题，如实记录。\n\n"
        ),
        "writer": (
            "【版本演进报告】\n"
            f"本报告是历史任务 {info['base_run_id']} 的迭代版本，除原有章节外须包含：\n"
            "1. 版本血缘：明确说明基于哪个历史任务迭代、本次修改要求是什么；\n"
            "2. 变更前后对比：修改点、新增点、保持不变的部分；\n"
            "3. 回归测试结论：原有功能是否保持正常；\n"
            "4. 交付结论对照增量验收标准与回归验收标准逐条标注。\n\n"
        ),
    }
    return common + role_hints[role]


def build_tasks(
    agents: dict,
    output_dir: Path,
    stage_callback: StageCallback | None = None,
    iteration_info: dict | None = None,
) -> list[Task]:
    """构建四个任务。output_dir 为本轮产出目录（已存在）。

    stage_callback：每个任务完成后被调用，用于 Web 端实时更新阶段进度；
    命令行入口 main.py 不传，行为与之前完全一致。

    iteration_info：为 None 时是全新开发；传入基线信息字典时切换为迭代模式
    （基线代码/文档已由 Web 层复制进沙箱）。
    """

    requirements_doc = output_dir / "01_requirements.md"
    implementation_doc = output_dir / "02_implementation.md"
    test_doc = output_dir / "03_test_report.md"
    final_doc = output_dir / "04_final_report.md"

    def _cb(index: int, name: str):
        if stage_callback is None:
            return None

        def _invoke(task_output) -> None:
            stage_callback(index, name, task_output)

        return _invoke

    def _iter(role: str) -> str:
        # 非迭代模式返回空前缀，任务描述与历史完全一致
        return _iteration_brief(role, iteration_info) if iteration_info else ""

    # ---- Task 1：需求分析 ---------------------------------------------------
    analyze = Task(
        description=(
            _iter("analyst")
            + "请对下面的用户原始需求进行结构化拆解。\n\n"
            "用户需求：\n{requirement}\n\n"
            "沙箱代码目录：{sandbox_dir}\n\n"
            "输出要求（Markdown）：\n"
            "1. 需求概述：用 2-3 句话重述需求本质；\n"
            "2. 功能点清单：逐条列出功能点，每条包含输入、处理逻辑、输出；\n"
            "3. 边界与异常场景：至少识别空输入、非法输入、极端值等情况；\n"
            "4. 技术方案：模块/函数划分（给出建议的函数名、参数、返回值）、"
            "核心数据结构、实现思路；\n"
            "5. 验收标准：可被测试工程师直接引用的、无歧义的验收条目；\n"
            "6. 范围外说明：明确本次不做什么，避免过度设计。\n"
            "技术约束：沙箱允许正常网络访问（urllib/requests 等）与文件读写，"
            "但禁止创建子进程、执行系统命令、操作注册表和沙箱目录外文件，"
            "方案必须在这些约束内可落地、可运行验证。"
        ),
        expected_output=(
            "一份中文 Markdown 需求分析文档，包含需求概述、功能点清单、"
            "边界异常场景、技术方案（含函数签名）、验收标准、范围外说明六个章节。"
        ),
        agent=agents["analyst"],
        output_file=str(requirements_doc),
        callback=_cb(0, "需求分析"),
    )

    # ---- Task 2：代码开发（沙箱自测）---------------------------------------
    develop = Task(
        description=(
            _iter("developer")
            + "请依据需求分析师的方案完成 Python 编码，并在沙箱中自测通过。\n\n"
            "用户需求：\n{requirement}\n\n"
            "沙箱代码目录：{sandbox_dir}\n\n"
            "执行要求：\n"
            "1. 使用 sandbox_python_exec 工具，将业务代码保存为 main.py；"
            "需要时可拆分为多个 .py 文件（通过 filename 参数指定）；\n"
            "2. 每个函数实现后立即编写自测调用并运行，覆盖正常值、边界值、异常输入；\n"
            "3. 发现报错必须分析工具返回的错误输出，修改后重新执行，直到退出码 0；\n"
            "4. 严禁通过任何沙箱外方式运行代码，严禁虚构执行结果；\n"
            "5. 完成后输出实现说明（Markdown）：文件清单及职责、核心函数说明"
            "（名称/参数/返回值/关键逻辑）、实际执行过的自测命令与真实输出摘要、"
            "遗留限制（如有）。"
        ),
        expected_output=(
            "代码已保存在沙箱目录且自测全部退出码 0；同时输出一份中文 Markdown "
            "实现说明，包含文件清单、核心函数说明、真实自测输出摘要与遗留限制。"
        ),
        agent=agents["developer"],
        context=[analyze],
        output_file=str(implementation_doc),
        callback=_cb(1, "代码开发"),
    )

    # ---- Task 3：测试验证（沙箱）-------------------------------------------
    test = Task(
        description=(
            _iter("tester")
            + "请依据需求分析师的验收标准与开发工程师的实现，在沙箱中独立编写并运行测试。\n\n"
            "用户需求：\n{requirement}\n\n"
            "沙箱代码目录：{sandbox_dir}（开发产出的 main.py 已在此目录）\n\n"
            "执行要求：\n"
            "1. 使用 sandbox_python_exec 工具新建 test_main.py，通过 import main "
            "引入被测代码，使用 assert 或 unittest 编写测试，禁止复制被测代码；\n"
            "2. 用例必须覆盖每条验收标准，并额外包含边界值与异常输入用例；\n"
            "3. 在沙箱中实际运行 test_main.py；若失败，记录失败用例、复现代码、"
            "实际输出与预期输出，并将问题如实写入报告（不得修改后谎报）；\n"
            "4. 可重新运行验证，但最终结论必须以最后一次真实执行输出为准；\n"
            "5. 输出测试报告（Markdown）：用例清单（编号/对应用例/类型："
            "正常-边界-异常）、执行结果统计表（通过数/失败数/通过率）、"
            "失败详情与复现步骤、整体质量结论。"
        ),
        expected_output=(
            "test_main.py 已在沙箱中实际运行；输出一份中文 Markdown 测试报告，"
            "包含用例清单、执行结果统计表、失败详情与客观的质量结论。"
        ),
        agent=agents["tester"],
        context=[analyze, develop],
        output_file=str(test_doc),
        callback=_cb(2, "测试验证"),
    )

    # ---- Task 4：最终报告 ---------------------------------------------------
    document = Task(
        description=(
            _iter("writer")
            + "请整合前三份产出，生成面向交付的最终报告。\n\n"
            "原始用户需求：\n{requirement}\n\n"
            "沙箱代码目录：{sandbox_dir}\n\n"
            "报告要求（Markdown）：\n"
            "1. 项目概述与原始需求；\n"
            "2. 功能点与技术方案摘要（以需求分析产出为准）；\n"
            "3. 代码结构说明：文件/函数职责，以及在沙箱目录中的位置；\n"
            "4. 核心代码片段：摘录关键函数代码（保持与沙箱中实际文件一致）；\n"
            "5. 测试结果汇总：通过率、通过/失败用例数、已知缺陷"
            "（严格引用测试工程师的真实结果，不得美化）；\n"
            "6. 运行方式说明：如何在沙箱目录运行 main.py 与 test_main.py；\n"
            "7. 交付结论：对照验收标准逐条标注达成情况。\n"
            "全部内容必须来自前三份产出与真实执行结果，禁止虚构。"
        ),
        expected_output=(
            "一份结构完整的中文 Markdown 最终报告，涵盖需求、方案、代码结构、"
            "核心代码、测试结果、运行方式与验收结论，内容与上游产出一致。"
        ),
        agent=agents["writer"],
        context=[analyze, develop, test],
        output_file=str(final_doc),
        callback=_cb(3, "文档撰写"),
    )

    return [analyze, develop, test, document]
