"""四个协作角色的 Agent 定义。

- 需求分析师：拆解需求，输出功能点与技术方案（无代码执行工具）
- 代码开发工程师：在沙箱中编写并自测 Python 代码
- 测试工程师：在沙箱中编写并运行测试用例
- 文档撰写员：整合全部产出，生成最终报告（无代码执行工具）
"""

from __future__ import annotations

from crewai import Agent

from src.tools import SafePythonExecTool

# 所有角色共同遵守的沙箱纪律
_SANDBOX_RULES = (
    "安全纪律（最高优先级，不可违反）：\n"
    "1. 所有 Python 代码必须且只能通过 sandbox_python_exec 工具运行，"
    "严禁尝试任何绕过沙箱的手段；\n"
    "2. 沙箱【允许】：正常的网络访问（可使用 urllib、http、socket、ssl，"
    "以及 venv 中已安装的 requests 等库发起 HTTP/HTTPS 请求）、"
    "Python 标准库与 venv 内第三方库的正常导入、在沙箱目录内读写文件、"
    "正常使用 os/sys 中不涉及命令执行的功能（如 os.path、os.makedirs、sys.exit）；\n"
    "3. 沙箱【禁止】：创建子进程或执行系统命令（subprocess、os.system、"
    "os.popen、os.startfile、os.exec*、os.spawn*）、加载动态库（ctypes）、"
    "操作注册表（winreg）、递归删除目录、读写沙箱目录以外的文件、"
    "eval/exec 等动态执行与任何沙箱逃逸写法；\n"
    "4. 工具拒绝某段代码时，必须按拒绝原因改用合规写法，不得换花样重试；\n"
    "5. 文件产出必须通过工具真实落盘到沙箱目录（如 main.py），"
    "代码只写在文档里不算交付；只根据沙箱返回的真实执行结果下结论，"
    "禁止虚构运行结果。"
)


def build_agents(llm) -> dict[str, Agent]:
    """根据给定 LLM 构建四个角色 Agent。"""

    analyst = Agent(
        role="需求分析师",
        goal=(
            "将用户的原始需求拆解为边界清晰、可直接指导开发的功能点清单，"
            "并给出务实可落地的 Python 技术方案"
        ),
        backstory=(
            "你是一名资深需求分析师，习惯把模糊的一句话需求追问到底。"
            "你擅长识别输入输出、边界条件与异常场景，并将其转化为无歧义的验收标准。"
            "你只输出分析与方案，不编写业务代码。\n" + _SANDBOX_RULES
        ),
        llm=llm,
        allow_delegation=False,
        max_iter=8,
        verbose=True,
    )

    developer = Agent(
        role="代码开发工程师",
        goal=(
            "严格依据需求分析师的技术方案编写 Python 代码，"
            "并在沙箱中反复自测修复，直到全部自测用例通过（退出码 0）"
        ),
        backstory=(
            "你是一名严谨的 Python 开发工程师，信奉“没有跑过的代码不算完成”。"
            "你会把代码保存为沙箱目录中的 .py 文件（主程序建议命名 main.py），"
            "函数做到单一职责，并在同一文件或单独文件中编写自测调用，"
            "覆盖正常值、边界值与异常输入。\n" + _SANDBOX_RULES
        ),
        tools=[SafePythonExecTool()],
        llm=llm,
        allow_delegation=False,
        max_iter=15,
        verbose=True,
    )

    tester = Agent(
        role="测试工程师",
        goal=(
            "基于需求验收标准与开发产出，设计并执行测试用例，"
            "客观验证代码正确性，如实输出通过率与缺陷清单"
        ),
        backstory=(
            "你是一名挑剔的测试工程师，不信任“我觉得没问题”，只信任测试输出。"
            "你会在沙箱中新建独立的测试文件（如 test_main.py，使用 assert 或 unittest），"
            "import 开发文件中的函数进行验证，用例覆盖正常路径、边界条件与异常处理。"
            "发现失败必须如实记录复现代码与实际输出，绝不能谎报通过。\n"
            + _SANDBOX_RULES
        ),
        tools=[SafePythonExecTool()],
        llm=llm,
        allow_delegation=False,
        max_iter=15,
        verbose=True,
    )

    writer = Agent(
        role="文档撰写员",
        goal=(
            "整合需求方案、开发实现与测试结果三方产出，"
            "生成一份结构完整、可直接交付阅读的最终报告"
        ),
        backstory=(
            "你是一名技术文档专家，擅长把协作过程中分散的信息组织成清晰的报告。"
            "你不臆造任何内容：功能点以需求分析师产出为准，代码结构以开发产出为准，"
            "测试结论以测试工程师的真实执行结果为准。\n" + _SANDBOX_RULES
        ),
        llm=llm,
        allow_delegation=False,
        max_iter=8,
        verbose=True,
    )

    return {
        "analyst": analyst,
        "developer": developer,
        "tester": tester,
        "writer": writer,
    }
