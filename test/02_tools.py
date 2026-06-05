"""
02_tools.py —— LangChain Tool 封装与工具调用

核心知识点：
1. @tool 装饰器：把普通 Python 函数变成 LangChain Tool
2. BaseTool 子类：适合复杂工具（异步、配置多、输入 Schema 复杂）
3. bind_tools：把工具"装"到 LLM 上，让 LLM 决定何时调用
4. ToolNode：LangGraph 内置的工具执行节点
5. ReAct 循环：LLM 思考 → 调用工具 → 观察结果 → 继续思考...

对照你的 RPA 项目：
  你的 7 个 FastAPI 生信服务 → 每个封装成一个 LangChain Tool
  insectbase_blast → BlastTool
  primer3          → Primer3Tool
  oligowalk        → OligoWalkTool
  nto_blast        → NTOBlastTool
  clustal          → ClustalTool
  fetchseq         → FetchSeqTool
  kinship          → KinshipTool
"""

import os
import json
from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_core.tools import tool, BaseTool
from pydantic import BaseModel, Field


# ============================================================
# 第一部分：两种 Tool 定义方式
# ============================================================

# --------------- 方式一：@tool 装饰器（简单函数）---------------

@tool
def insectbase_blast(gene_name: str, species: str) -> str:
    """在靶标昆虫基因组中搜索同源基因序列。
    返回匹配的 scaffold 位置、E-value 和序列相似度。

    Args:
        gene_name: 靶标基因名称，如 V-ATPase, COPI, Snf7
        species: 昆虫物种名，如 Helicoverpa armigera
    """
    # 模拟 BLAST 结果（实际中这里是 HTTP 调用你的 FastAPI 服务）
    mock_result = {
        "gene": gene_name,
        "species": species,
        "hits": [
            {"scaffold": "NC_045123.1", "evalue": "1e-45", "identity": 98.5},
            {"scaffold": "NC_045124.1", "evalue": "1e-30", "identity": 85.2},
        ],
        "best_hit": "NC_045123.1:12345-13456"
    }
    return json.dumps(mock_result, ensure_ascii=False, indent=2)


@tool
def check_gc_content(sequence: str) -> str:
    """计算 DNA 序列的 GC 含量。用于判断 dsRNA 设计质量。

    Args:
        sequence: DNA 序列字符串（大写），如 AGCTAGCTAGCT
    """
    sequence = sequence.upper()
    gc = sum(1 for c in sequence if c in "GC")
    total = len(sequence)
    gc_pct = gc / total * 100 if total > 0 else 0

    verdict = "✅ 适合" if 35 <= gc_pct <= 55 else "⚠️ 需优化"
    return f"序列: {sequence}\nGC含量: {gc_pct:.1f}%\n判定: {verdict} (建议范围 35%-55%)"


# --------------- 方式二：BaseTool 子类（复杂工具）---------------
# 适合你的 HTTP 服务封装：需要配置 API 地址、超时、重试等

class Primer3ToolInput(BaseModel):
    """primer3 工具的输入 Schema"""
    sequence: str = Field(description="模板 DNA 序列 (5'→3')")
    target_start: int = Field(default=1, description="靶标区域起始位置")
    target_length: int = Field(default=500, description="靶标区域长度")
    product_size_min: int = Field(default=80, description="最小产物大小")
    product_size_max: int = Field(default=150, description="最大产物大小")


class Primer3Tool(BaseTool):
    """通过 HTTP 调用 primer3 服务设计 PCR 引物。

    对应你项目中的 primer3 FastAPI 服务。
    """
    name: str = "primer3"
    description: str = "设计 PCR 引物。输入模板序列和产物参数，返回正向/反向引物对。"
    args_schema: type = Primer3ToolInput

    # ---- 配置项（实际中从环境变量或配置文件读取）----
    base_url: str = "http://localhost:8000/primer3"
    timeout: int = 30
    max_retries: int = 2

    def _run(self, sequence: str, target_start: int = 1, target_length: int = 500,
             product_size_min: int = 80, product_size_max: int = 150) -> str:
        """同步调用（LangChain 要求实现 _run）"""
        # 实际代码：
        # response = requests.post(
        #     self.base_url,
        #     json={...},
        #     timeout=self.timeout
        # )
        mock_result = {
            "forward_primer": "ATGCGTACGTTGACAGTCAC",
            "reverse_primer": "TCAGCCAGTAGCTACAAGGC",
            "tm_forward": 58.3,
            "tm_reverse": 59.1,
            "gc_forward": 50.0,
            "gc_reverse": 52.4,
            "product_size": 134,
            "status": "OK"
        }
        return json.dumps(mock_result, ensure_ascii=False, indent=2)


# ============================================================
# 第二部分：组装工具列表（这就是"工具白名单"）
# ============================================================
# 在你的 RPA 项目中，每个子智能体有独立的工具白名单：
# literature_agent → [web_search, vector_search]
# target_agent    → [insectbase_blast, insectbase_anno, primer3, oligowalk]
# safety_agent    → [nto_blast, clustal, kinship]
# ...

ALL_TOOLS = [insectbase_blast, check_gc_content, Primer3Tool()]

# 工具白名单示例（不同子智能体用不同的工具子集）
TARGET_DESIGN_TOOLS = [insectbase_blast, check_gc_content, Primer3Tool()]
SAFETY_CHECK_TOOLS = [insectbase_blast]  # 生态安全智能体只需要 BLAST


# ============================================================
# 第三部分：定义 State（消息列表 + 工具白名单）
# ============================================================
class ToolAgentState(TypedDict):
    messages: Annotated[list, add_messages]   # 对话历史（自动追加）
    allowed_tools: list                       # 当前子智能体的工具白名单 ⭐ Harness


# ============================================================
# 第四部分：构建 Agent 节点
# ============================================================

# 注意：这里用 DeepSeek 作为示例（你项目要求支持自定义 LLM 供应商）
# 如果本地没有 API Key，可以跳过实际运行，只看代码结构

def create_agent_node():
    """创建 Agent 节点 —— LLM + 工具 = 能干活"""

    # 支持任意 OpenAI 兼容的 API（DeepSeek / 智谱 / 硅基流动 / Ollama ...）
    llm = ChatOpenAI(
        model="deepseek-chat",                     # 或 gpt-4o / glm-4 / qwen
        base_url="https://api.deepseek.com/v1",    # 自定义 API 地址
        api_key=os.getenv("DEEPSEEK_API_KEY", "sk-placeholder"),
        temperature=0.3,
    )

    # 把工具"装"到 LLM 上
    llm_with_tools = llm.bind_tools(ALL_TOOLS)

    def agent_node(state: ToolAgentState) -> dict:
        """Agent 节点：调用 LLM，LLM 决定是否需要调用工具"""
        print(f"\n[Agent] 收到 {len(state['messages'])} 条消息")

        # LLM 会分析 messages，如果判断需要工具，会生成 tool_calls
        response = llm_with_tools.invoke(state["messages"])

        # 检查 LLM 是否想调用工具
        if hasattr(response, "tool_calls") and response.tool_calls:
            tool_names = [tc["name"] for tc in response.tool_calls]
            print(f"[Agent] LLM 决定调用工具: {tool_names}")

        return {"messages": [response]}

    return agent_node


# ============================================================
# 第五部分：路由决策
# ============================================================
def should_continue(state: ToolAgentState) -> str:
    """检查最后一条消息是否包含 tool_calls"""
    last_message = state["messages"][-1]

    # 如果 LLM 要求调用工具 → 去 ToolNode 执行
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"

    # 否则 → 结束（LLM 已给出最终回答）
    return END


# ============================================================
# 第六部分：构建 Graph
# ============================================================
def build_tool_agent():
    """构建带工具调用的 Agent Graph"""
    builder = StateGraph(ToolAgentState)

    # 创建节点
    agent_node = create_agent_node()
    tool_node = ToolNode(ALL_TOOLS)  # LangGraph 内置：自动执行工具并返回 ToolMessage

    builder.add_node("agent", agent_node)
    builder.add_node("tools", tool_node)

    # 边
    builder.add_edge(START, "agent")
    builder.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
    builder.add_edge("tools", "agent")     # 工具执行完 → 回到 agent（ReAct 循环）

    return builder.compile()


# ============================================================
# 第七部分：运行演示
# ============================================================
def run_demo():
    """演示工具调用流程"""
    print("=" * 60)
    print("02_tools.py — LangChain Tool 封装与 ReAct 循环")
    print("=" * 60)

    # 模拟用户输入
    user_query = "请在棉铃虫基因组中搜索 V-ATPase 基因，然后检查序列 ATGCGTACGTTGACAGTCAC 的 GC 含量"

    print(f"\n用户: {user_query}\n")

    # 检查 API Key
    if not os.getenv("DEEPSEEK_API_KEY") and not os.getenv("OPENAI_API_KEY"):
        print("\n⚠️ 未设置 DEEPSEEK_API_KEY 或 OPENAI_API_KEY")
        print("跳过 LLM 实际调用，仅展示 Graph 的 Mermaid 结构图:\n")
        graph = build_tool_agent()
        print(graph.get_graph().draw_mermaid())
        print("\n---")
        print("提示: 设置环境变量后即可实际运行:")
        print("  export DEEPSEEK_API_KEY=your-key")
        print("  python 02_tools.py")
        return

    graph = build_tool_agent()
    result = graph.invoke({
        "messages": [HumanMessage(content=user_query)],
        "allowed_tools": TARGET_DESIGN_TOOLS,  # ⭐ 传入工具白名单（Harness）
    })

    print("\n" + "=" * 60)
    print("完整对话历史:")
    for i, msg in enumerate(result["messages"]):
        role = type(msg).__name__
        content = msg.content if hasattr(msg, "content") else str(msg)
        if len(str(content)) > 200:
            content = str(content)[:200] + "...(省略)"
        print(f"  [{i}] {role}: {content}")


if __name__ == "__main__":
    run_demo()
