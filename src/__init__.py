"""多角色协作任务 Agent（基于 CrewAI + DeepSeek）。

角色流水线：需求分析师 -> 代码开发工程师 -> 测试工程师 -> 文档撰写员。
所有代码执行均经过 src.tools.sandbox 的安全沙箱。
"""
