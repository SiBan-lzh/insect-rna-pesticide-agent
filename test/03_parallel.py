"""
03_parallel.py —— Send() API 并行执行

核心知识点：
1. Send() — 从条件边返回 Send 列表，动态创建 N 个并行分支
2. operator.add reducer — 多个并行分支的结果自动合并
3. 不同 State 的传参 — Send 时只用传子状态需要的字段
4. 汇合点 — 所有并行分支完成后自动进入下一个节点

对照你的 RPA 项目：
  Supervisor 拆解任务 → Send("literature_search", ...)  ─┐
                         Send("kinship_analysis", ...)   ─┤  并行执行
                         Send("target_design", ...)      ─┘
                              ↓
                         全部完成后 → 汇合到 experiment_design
"""

import time
import random
from typing import TypedDict, Annotated
import operator
from langgraph.graph import StateGraph, START, END
from langgraph.types import Send


# ============================================================
# 第一步：定义 State
# ============================================================

class SubTask(TypedDict):
    """子任务结构"""
    id: str
    description: str
    agent: str                # 由哪个子智能体执行
    can_parallel: bool        # 是否可以与其他任务并行


class OverallState(TypedDict):
    """全局状态"""
    user_query: str                                    # 用户原始需求
    tasks: list[SubTask]                               # 拆解后的子任务列表
    # ⭐ operator.add: 多个并行分支的结果自动追加到这个列表
    completed_results: Annotated[list[dict], operator.add]
    final_summary: str                                 # 最终汇总


# 每个并行分支的输入状态（只需要任务信息）
class WorkerState(TypedDict):
    task_id: str
    task_description: str
    agent: str


# ============================================================
# 第二步：模拟三个子智能体（对应你的 literature_search, kinship_analysis, target_design）
# ============================================================

def literature_search_agent(task_id: str, description: str) -> dict:
    """文献检索智能体（模拟）"""
    print(f"  📚 [文献检索] 开始: {description}")
    time.sleep(random.uniform(0.5, 1.5))  # 模拟耗时操作（如 HTTP 调用 BLAST）
    result = {
        "task_id": task_id,
        "agent": "literature_search",
        "result": f"找到 15 篇相关文献，其中 3 篇直接涉及 {description}",
        "status": "done",
    }
    print(f"  📚 [文献检索] 完成: {task_id}")
    return result


def kinship_analysis_agent(task_id: str, description: str) -> dict:
    """亲缘关系分析智能体（模拟）"""
    print(f"  🧬 [亲缘关系] 开始: {description}")
    time.sleep(random.uniform(0.5, 1.5))
    result = {
        "task_id": task_id,
        "agent": "kinship_analysis",
        "result": f"基于 ETE3 物种树分析，{description} 的最近缘物种为 Helicoverpa zea",
        "status": "done",
    }
    print(f"  🧬 [亲缘关系] 完成: {task_id}")
    return result


def target_design_agent(task_id: str, description: str) -> dict:
    """靶标设计智能体（模拟）"""
    print(f"  🎯 [靶标设计] 开始: {description}")
    time.sleep(random.uniform(0.5, 1.5))
    result = {
        "task_id": task_id,
        "agent": "target_design",
        "result": f"设计出 3 个候选靶标序列，GC 含量 42%-52%，E-value < 1e-40",
        "status": "done",
    }
    print(f"  🎯 [靶标设计] 完成: {task_id}")
    return result


# 子智能体注册表
AGENT_REGISTRY = {
    "literature_search": literature_search_agent,
    "kinship_analysis": kinship_analysis_agent,
    "target_design": target_design_agent,
}


# ============================================================
# 第三步：定义节点
# ============================================================

def supervisor_node(state: OverallState) -> dict:
    """Supervisor：拆解用户需求为子任务列表（对应你的 workflow_decomposer）"""
    print("\n" + "=" * 60)
    print(f"[Supervisor] 收到需求: {state['user_query']}")
    print("=" * 60)

    # 实际项目中这里由 LLM + with_structured_output 生成
    # 这里模拟拆解结果
    tasks: list[SubTask] = [
        {
            "id": "task-lit",
            "description": "检索 V-ATPase RNAi 在鳞翅目害虫中的已有研究",
            "agent": "literature_search",
            "can_parallel": True,
        },
        {
            "id": "task-kin",
            "description": "分析棉铃虫 V-ATPase 基因的物种亲缘关系",
            "agent": "kinship_analysis",
            "can_parallel": True,
        },
        {
            "id": "task-tar",
            "description": "设计靶向 V-ATPase 的 dsRNA 序列",
            "agent": "target_design",
            "can_parallel": True,
        },
    ]

    print(f"[Supervisor] 拆解出 {len(tasks)} 个子任务，全部可并行")
    return {"tasks": tasks, "completed_results": []}


# ⭐ 关键函数：返回 Send 列表 → LangGraph 并行执行
def dispatch_parallel_tasks(state: OverallState) -> list[Send]:
    """条件边函数：为每个可并行任务生成一个 Send

    这就是你项目中 Sub-Agent 级并行的核心机制：
    Supervisor 派发 → 多个子智能体同时执行
    """
    sends = []
    for task in state["tasks"]:
        if task["can_parallel"]:
            sends.append(
                Send(
                    "worker",  # 目标节点名（所有并行任务进入同一个 worker 节点）
                    {          # 传入的子状态（只传需要的字段）
                        "task_id": task["id"],
                        "task_description": task["description"],
                        "agent": task["agent"],
                    }
                )
            )

    print(f"\n⚡ [Send API] 生成 {len(sends)} 个并行分支:")
    for s in sends:
        print(f"    → Send('worker', task_id='{s.arg['task_id']}', agent='{s.arg['agent']}')")

    return sends


def worker_node(state: WorkerState) -> dict:
    """Worker 节点：根据 agent 字段路由到对应的子智能体

    所有并行分支都进入这个节点，但携带不同的 state
    """
    agent_name = state["agent"]
    task_id = state["task_id"]
    description = state["task_description"]

    # 从注册表获取对应的子智能体函数
    agent_func = AGENT_REGISTRY.get(agent_name)
    if not agent_func:
        result = {"task_id": task_id, "agent": agent_name, "result": "未知智能体", "status": "error"}
    else:
        result = agent_func(task_id, description)

    # ⭐ 返回值通过 operator.add reducer 自动合并到 OverallState.completed_results
    return {"completed_results": [result]}


def summarize_node(state: OverallState) -> dict:
    """汇总节点：所有并行任务完成后，汇总结果"""
    print("\n" + "=" * 60)
    print("[Summarize] 全部子任务完成，开始汇总")
    print("=" * 60)

    results = state["completed_results"]
    lines = [f"## 执行汇总（共 {len(results)} 个任务）\n"]
    for r in results:
        status = "✅" if r["status"] == "done" else "❌"
        lines.append(f"- {status} [{r['agent']}] {r['result']}")

    summary = "\n".join(lines)
    print(summary)
    return {"final_summary": summary}


# ============================================================
# 第四步：构建 Graph
# ============================================================
def build_parallel_graph():
    """构建带并行派发的 Graph"""
    builder = StateGraph(OverallState)

    builder.add_node("supervisor", supervisor_node)
    builder.add_node("worker", worker_node)
    builder.add_node("summarize", summarize_node)

    # START → Supervisor
    builder.add_edge(START, "supervisor")

    # ⭐ Supervisor → 条件边返回 Send 列表 → 并行执行 worker
    builder.add_conditional_edges(
        "supervisor",
        dispatch_parallel_tasks,  # 返回 [Send("worker", ...), Send("worker", ...), ...]
        ["worker"]                # 所有 Send 的目标节点
    )

    # 所有 worker 并行完成 → 自动汇合 → Summarize
    builder.add_edge("worker", "summarize")
    builder.add_edge("summarize", END)

    return builder.compile()


# ============================================================
# 第五步：运行
# ============================================================
if __name__ == "__main__":
    graph = build_parallel_graph()

    print("=" * 60)
    print("03_parallel.py — Send() API 并行执行")
    print("=" * 60)
    print("\n图结构:")
    print("  START → Supervisor → [Send × N] → Worker (×N 并行)")
    print("                                    ↓")
    print("                              operator.add 合并")
    print("                                    ↓")
    print("                              Summarize → END\n")

    user_input = "为棉铃虫设计靶向 V-ATPase 的 RNAi 农药"

    start_time = time.time()
    result = graph.invoke({"user_query": user_input})
    elapsed = time.time() - start_time

    print(f"\n⏱️ 总耗时: {elapsed:.2f}s")
    print(f"   如果是串行执行，预计耗时约 {sum(random.uniform(0.5, 1.5) for _ in range(3)):.2f}s × 3 = ~3-4.5s")
    print(f"   并行执行仅需 ~{elapsed:.1f}s（三个任务中最慢的那个）\n")

    # 展示 Mermaid 图
    print("Graph Mermaid 结构（可在 https://mermaid.live 查看）:\n")
    print(graph.get_graph().draw_mermaid())
