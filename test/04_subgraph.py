"""
04_subgraph.py —— 子图封装 + 记忆 + 上下文隔离 + Debug

四个知识点一次覆盖：
1. 子图封装：每个子智能体 = 一个独立编译的 StateGraph
2. Checkpointer（SQLite）：持久化会话状态，中断后可恢复
3. 上下文隔离：子图内部的消息循环不外泄，只回传结果摘要
4. Debug：logging 日志 + interrupt_before 断点

场景：
  用户要求"设计靶向 V-ATPase 的 RNAi 农药"
  Supervisor 拆成 2 个任务并行派发：
    子图A（文献检索员）：查文献 → 调 BLAST → 返回摘要
    子图B（靶标设计员）：设计靶标 → 调 oligowalk → 返回候选序列

架构：
  ┌─────────── 总图 (SupervisorGraph) ───────────┐
  │  Supervisor         派发                       │
  │    → Send → 文献检索子图 ─┐                    │
  │    → Send → 靶标设计子图 ─┤ 并行               │
  │                           ↓                    │
  │  operator.add 合并结果 → 汇总                  │
  └───────────────────────────────────────────────┘

  子图A (LiteratureSubGraph)      子图B (TargetDesignSubGraph)
  agent → tool → finalize         agent → tool → finalize
  内部消息不外泄                   内部消息不外泄
"""

import logging
import os
import sqlite3
from typing import TypedDict, Annotated
import operator
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Send
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.tools import tool


# ============================================================
# 第〇步：日志配置（Debug 基础）
# ============================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-5s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ============================================================
# 第一步：定义两个模拟工具
# ============================================================
@tool
def insectbase_blast(gene_name: str) -> str:
    """模拟在靶标昆虫基因组中 BLAST 搜索同源基因。"""
    logger.info(f"  🔬 [BLAST] 搜索基因: {gene_name}")
    return f"找到 {gene_name} 在 scaffold_12 上的匹配，E-value=1e-52，identity=98.7%"


@tool
def oligowalk_score(sequence: str) -> str:
    """模拟对 siRNA 序列做热力学综合评分。"""
    logger.info(f"  🧪 [OligoWalk] 评分序列: {sequence[:20]}...")
    return f"序列 {sequence[:20]}... 热力学评分: 0.87/1.0 (优秀)"


# ============================================================
# 第二步：定义 State
# ============================================================

# --- 子图内部 State ---
class SubState(TypedDict):
    """子图内部状态：每个子智能体独立拥有

    关键设计：
    - messages 用 add_messages，子图内部多轮对话都在这里
    - 这些 messages 不会泄漏到总图 State 中（因为总图没有 messages 字段）
    - completed_results 与总图共享 reducer → 子图结果自动回传 ⭐
    """
    task_id: str                                    # 任务 ID
    task_description: str                           # 任务描述
    messages: Annotated[list, add_messages]         # 内部对话，不外泄 ⭐
    completed_results: Annotated[list[dict], operator.add]  # ⭐ 与总图共享此字段


# --- 总图 State ---
class SubTask(TypedDict):
    """总图中的子任务结构"""
    id: str
    description: str
    subgraph: str              # 路由到哪个子图


class SupervisorState(TypedDict):
    """总图全局状态"""
    user_query: str
    tasks: list[SubTask]                             # 拆解后的任务列表
    completed_results: Annotated[list[dict], operator.add]  # ⭐ 并行结果合并
    final_summary: str


# ============================================================
# 第三步：构建子图
# ============================================================

def create_literature_subgraph() -> StateGraph:
    """文献检索子图

    对总图来说它就是一个普通节点。
    但内部有自己的 StateGraph、自己的 State、自己的工具。
    总图不知道也不关心它内部怎么干活。
    """
    builder = StateGraph(SubState)

    # ⭐ 工具白名单：文献检索员只能用 BLAST
    tools = [insectbase_blast]

    def agent_node(state: SubState) -> dict:
        """模拟 LLM 接收任务，决定查什么"""
        logger.info(f"  📚 [文献] agent 接收任务: {state['task_description']}")
        return {
            "messages": [AIMessage(content=f"收到任务，需要查询 V-ATPase 基因相关信息")]
        }

    def tool_node(state: SubState) -> dict:
        """调用 BLAST 工具"""
        logger.info(f"  📚 [文献] 调用 BLAST 工具")
        result = insectbase_blast.invoke({"gene_name": "V-ATPase"})
        return {"messages": [AIMessage(content=f"BLAST结果: {result}")]}

    def finalize_node(state: SubState) -> dict:
        """提取摘要 —— 内部 N 条消息 → 只返回 1 条结构化结果"""
        logger.info(f"  📚 [文献] finalize: 内部共 {len(state['messages'])} 条消息")

        # ⭐⭐⭐ 上下文隔离的核心：
        # 子图内部的 agent→tool→agent 循环消息全部在这里终止
        # 通过 completed_results（与总图共享字段）回传结构化结果
        # messages（内部对话）不会泄漏到总图（总图没有 messages 字段）
        return {
            "completed_results": [{
                "agent": "literature",
                "task_id": state["task_id"],
                "summary": "文献检索摘要: V-ATPase 在鳞翅目中高度保守，已有 12 篇研究报道其 RNAi 致死效果，推荐作为靶标"
            }]
        }

    builder.add_node("agent", agent_node)
    builder.add_node("tools", tool_node)
    builder.add_node("finalize", finalize_node)

    # 子图内部流转: START → agent → tools → finalize → END
    builder.add_edge(START, "agent")
    builder.add_edge("agent", "tools")
    builder.add_edge("tools", "finalize")
    builder.add_edge("finalize", END)

    return builder.compile()


def create_target_design_subgraph() -> StateGraph:
    """靶标设计子图 —— 结构与文献子图完全相同，但工具白名单不同"""
    builder = StateGraph(SubState)

    # ⭐ 工具白名单：靶标设计员只能用 OligoWalk
    tools = [oligowalk_score]

    def agent_node(state: SubState) -> dict:
        logger.info(f"  🎯 [靶标] agent 接收任务: {state['task_description']}")
        target_seq = "AUGCGUACGUUGACAGUCAC"
        return {
            "messages": [
                AIMessage(content=f"设计候选靶标序列: {target_seq}"),
                AIMessage(content=f"待评分序列: {target_seq}"),
            ]
        }

    def tool_node(state: SubState) -> dict:
        logger.info(f"  🎯 [靶标] 调用 OligoWalk 工具")
        # 从消息中提取序列
        for msg in reversed(state["messages"]):
            if hasattr(msg, "content") and "待评分序列:" in msg.content:
                seq = msg.content.split("待评分序列:")[1].strip()
                result = oligowalk_score.invoke({"sequence": seq})
                return {"messages": [AIMessage(content=f"OligoWalk结果: {result}")]}
        return {"messages": [AIMessage(content="错误: 未找到序列")]}

    def finalize_node(state: SubState) -> dict:
        logger.info(f"  🎯 [靶标] finalize: 内部共 {len(state['messages'])} 条消息")

        # ⭐ 同样：内部消息不外泄，通过共享字段回传结果
        return {
            "completed_results": [{
                "agent": "target_design",
                "task_id": state["task_id"],
                "summary": "靶标设计摘要: 候选序列 AUGCGUACGUUGACAGUCAC，热力学评分 0.87/1.0，GC含量 48%，推荐使用"
            }]
        }

    builder.add_node("agent", agent_node)
    builder.add_node("tools", tool_node)
    builder.add_node("finalize", finalize_node)

    builder.add_edge(START, "agent")
    builder.add_edge("agent", "tools")
    builder.add_edge("tools", "finalize")
    builder.add_edge("finalize", END)

    return builder.compile()


# ============================================================
# 第四步：定义总图节点
# ============================================================

def supervisor_node(state: SupervisorState) -> dict:
    """Supervisor：拆解用户需求为子任务"""
    logger.info(f"[Supervisor] 收到: {state['user_query']}")

    tasks: list[SubTask] = [
        {"id": "lit-1", "description": "检索 V-ATPase RNAi 文献", "subgraph": "literature"},
        {"id": "tar-1", "description": "设计靶标序列并评分", "subgraph": "target_design"},
    ]

    logger.info(f"[Supervisor] 拆解: {len(tasks)} 个任务，并行执行")
    return {"tasks": tasks}


def dispatch_to_subgraphs(state: SupervisorState) -> list[Send]:
    """条件边函数：并行派发到各子图"""
    sends = []
    for task in state["tasks"]:
        target = f"subgraph_{task['subgraph']}"  # 子图节点名
        sends.append(
            Send(
                target,  # ⭐ 目标 = 不同的子图节点（不是同一个 worker 了）
                {
                    "task_id": task["id"],
                    "task_description": task["description"],
                    "messages": [HumanMessage(content=task["description"])],
                }
            )
        )
    logger.info(f"[Supervisor] 派发 {len(sends)} 个任务到子图")
    return sends


def summarize_node(state: SupervisorState) -> dict:
    """汇总各子图的结果"""
    results = state["completed_results"]
    logger.info(f"[汇总] 收到 {len(results)} 个子任务结果")
    lines = ["## 执行汇总"]
    for r in results:
        lines.append(f"  - {r['agent']}: {r['summary']}")
    return {"final_summary": "\n".join(lines)}


# ============================================================
# 第五步：组装总图
# ============================================================

def build_graph():
    """组装总图，把编译好的子图直接当普通节点插入"""
    builder = StateGraph(SupervisorState)

    # 总图自身节点
    builder.add_node("supervisor", supervisor_node)
    builder.add_node("summarize", summarize_node)

    # ⭐ 把编译好的子图当作普通节点插入总图
    builder.add_node("subgraph_literature", create_literature_subgraph())
    builder.add_node("subgraph_target_design", create_target_design_subgraph())

    # 边
    builder.add_edge(START, "supervisor")

    # supervisor → 条件边 → 并行派发到两个不同子图
    builder.add_conditional_edges(
        "supervisor",
        dispatch_to_subgraphs,
        ["subgraph_literature", "subgraph_target_design"]
    )

    # 两个子图都完成后 → 汇合
    builder.add_edge("subgraph_literature", "summarize")
    builder.add_edge("subgraph_target_design", "summarize")
    builder.add_edge("summarize", END)

    return builder  # 不 compile，留给调用方加 checkpointer


# ============================================================
# 第六步：主程序 —— 演示三种运行方式
# ============================================================

def run_basic():
    """方式一：基础运行 —— 无记忆，无断点"""
    print("=" * 60)
    print("方式一：基础运行（无 Checkpointer）")
    print("=" * 60)

    graph = build_graph().compile()
    result = graph.invoke({"user_query": "为棉铃虫设计靶向 V-ATPase 的 RNAi 农药"})
    print(f"\n最终输出:\n{result['final_summary']}")
    print(f"\n子图回传的结果数: {len(result.get('completed_results', []))}")
    for r in result.get("completed_results", []):
        print(f"  [{r['agent']}] {r['summary'][:60]}...")


def run_with_memory():
    """方式二：带 Checkpointer —— 持久化记忆"""
    print("\n" + "=" * 60)
    print("方式二：带 Checkpointer（SQLite 持久化）")
    print("=" * 60)

    # 创建 SQLite 数据库连接
    db_path = "/home/lizonghuan/langgraph/test/checkpoints.db"
    conn = sqlite3.connect(db_path, check_same_thread=False)

    # ⭐ SqliteSaver: 每次节点执行后自动保存 State 快照
    checkpointer = SqliteSaver(conn)
    graph = build_graph().compile(checkpointer=checkpointer)

    # 配置：指定线程 ID（同一线程内共享记忆）
    config = {"configurable": {"thread_id": "session-001"}}

    # 第一次请求
    print("\n--- 第一次请求 ---")
    result1 = graph.invoke(
        {"user_query": "为棉铃虫设计靶向 V-ATPase 的 RNAi 农药"},
        config=config
    )
    print(f"结果: {result1['final_summary'][:100]}...")

    # 第二次请求（同一个 thread_id）
    print("\n--- 第二次请求（同一会话，保留上下文）---")
    result2 = graph.invoke(
        {"user_query": "在上次的基础上，增加生态安全评估"},
        config=config
    )
    print(f"结果: {result2['final_summary'][:100]}...")

    # 查看 State 快照
    print(f"\n数据库文件: {db_path}")
    print(f"大小: {os.path.getsize(db_path)} bytes")

    # 清理
    conn.close()
    if os.path.exists(db_path):
        os.remove(db_path)


def run_with_interrupt():
    """方式三：带断点 —— Debug 用"""
    print("\n" + "=" * 60)
    print("方式三：interrupt_before 断点调试")
    print("=" * 60)

    graph = build_graph().compile(
        checkpointer=SqliteSaver(sqlite3.connect(":memory:", check_same_thread=False)),
        # ⭐ 在指定节点执行前暂停，等待人工确认
        interrupt_before=["subgraph_literature", "subgraph_target_design"]
    )

    config = {"configurable": {"thread_id": "debug-001"}}

    # 第一次 invoke: 执行到子图前会暂停
    print("\n--- 第一步：执行到 supervisor 后暂停 ---")
    result = graph.invoke(
        {"user_query": "设计 RNAi 农药"},
        config=config
    )

    # 获取当前状态
    current_state = graph.get_state(config)
    print(f"\n当前暂停位置: {current_state.next}")
    print(f"已完成的任务数: {len(result.get('tasks', []))}")
    print("Supervisor 已拆解任务，即将进入子图...")

    # 人工确认后继续 —— 传入 None 即可
    print("\n--- 第二步：继续执行 ---")
    result = graph.invoke(
        None,  # ⭐ 传入 None 表示继续上次暂停的执行
        config=config
    )
    print(f"\n最终结果: {result['final_summary']}")


if __name__ == "__main__":
    run_basic()
    run_with_memory()
    run_with_interrupt()
