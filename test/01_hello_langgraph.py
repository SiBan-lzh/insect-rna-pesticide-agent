"""
LangGraph 最小示例 —— 用一幅图理解核心概念

三个核心概念：
1. State  — 图中流转的"数据总线"，所有节点读写它
2. Node   — 处理函数，接收 State，返回部分更新的 dict
3. Edge   — 连接节点的箭头：普通边（固定）、条件边（分支）、Send（并行）
"""

from typing import TypedDict
from langgraph.graph import StateGraph, START, END

# ============================================================
# 第一步：定义 State（状态 = 图的"数据总线"）
# ============================================================
class MyState(TypedDict):
    messages: list[str]   # 消息列表
    counter: int          # 计数器


# ============================================================
# 第二步：定义 Node（节点 = 处理函数）
# 每个节点接收 (state, config)，返回部分更新的 dict
# ============================================================
def node_a(state: MyState) -> dict:
    """节点 A：追加一条消息，counter +1"""
    print(f"[Node A] 进入，当前 counter = {state['counter']}")
    return {
        "messages": state["messages"] + [f"A: counter became {state['counter'] + 1}"],
        "counter": state["counter"] + 1,
    }


def node_b(state: MyState) -> dict:
    """节点 B：追加一条消息，counter ×2"""
    print(f"[Node B] 进入，当前 counter = {state['counter']}")
    return {
        "messages": state["messages"] + [f"B: counter became {state['counter'] * 2}"],
        "counter": state["counter"] * 2,
    }


def node_c(state: MyState) -> dict:
    """节点 C：追加结束语"""
    print(f"[Node C] 进入，最终 counter = {state['counter']}")
    return {
        "messages": state["messages"] + [f"C: 流程结束，最终 counter = {state['counter']}"],
    }


# ============================================================
# 第三步：条件边 —— 根据 state 决定走哪条路
# ============================================================
def decide_route(state: MyState) -> str:
    """条件边函数：counter > 3 走 C，否则继续循环回 A"""
    if state["counter"] > 3:
        print(f"[Router] counter={state['counter']} > 3 → 走 C")
        return "node_c"
    else:
        print(f"[Router] counter={state['counter']} <= 3 → 回到 A")
        return "node_a"


# ============================================================
# 第四步：构建 Graph（图 = 节点 + 边的集合）
# ============================================================
builder = StateGraph(MyState)

# 添加节点
builder.add_node("node_a", node_a)
builder.add_node("node_b", node_b)
builder.add_node("node_c", node_c)

# 添加边
builder.add_edge(START, "node_a")    # 开始 → A
builder.add_edge("node_a", "node_b") # A → B（固定）
# B 之后 → 条件判断（大于3结束，否则回到A）
builder.add_conditional_edges("node_b", decide_route, ["node_a", "node_c"])
builder.add_edge("node_c", END)      # C → 结束

# 编译（把图"冻住"成可执行对象）
graph = builder.compile()

# ============================================================
# 第五步：运行
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("LangGraph Hello World")
    print("工作流: START → A → B → (counter>3? C:END : A→B→...)")
    print("=" * 60 + "\n")

    result = graph.invoke({"messages": [], "counter": 0})

    print("\n" + "=" * 60)
    print("最终结果:")
    for msg in result["messages"]:
        print(f"  {msg}")
    print(f"  counter = {result['counter']}")
    print("=" * 60)
