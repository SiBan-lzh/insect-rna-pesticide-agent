"""
agents —— LangGraph ReAct Agent 实现

每个 Agent 是一个编译后的 StateGraph，特性：
  - 工具白名单隔离（Harness 权限管控）
  - DeepSeek 作为 LLM 后端（OpenAI 兼容 API）
  - MemorySaver 对话持久化
  - 手动 StateGraph（非 create_react_agent），便于后续扩展

用法:
    from agents import dsrna_designer_agent

    result = dsrna_designer_agent.invoke(
        {"messages": [HumanMessage(content="设计靶向序列的 dsRNA...")]},
        config={"configurable": {"thread_id": "session-1"}}
    )
"""

from .dsrna_designer import build_dsrna_designer_agent, dsrna_designer_agent
from .safety_inspector import (
    build_safety_inspector_graph,
    build_species_analysis_subgraph,
    safety_inspector_graph,
)

__all__ = [
    "build_dsrna_designer_agent",
    "dsrna_designer_agent",
    "build_safety_inspector_graph",
    "build_species_analysis_subgraph",
    "safety_inspector_graph",
]
