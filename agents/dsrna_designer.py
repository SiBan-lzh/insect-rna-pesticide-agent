"""
04_dsRNA_designer ReAct Agent —— 基于阶段状态机的 LangGraph 实现

架构（4 阶段工作流）:
    START
      ↓
    phase_scan:    LLM + [oligowalk]  →  仅可调用 oligowalk
      ↓ (oligowalk 完成后自动切换)
    phase_analyze: LLM + 无工具       →  纯推理，分析候选 + 设计片段
      ↓ (自动切换)
    phase_primers: LLM + [primer3]    →  仅可调用 primer3，可为多片段多次调用
      ↓ (primer3 完成后自动切换)
    phase_report:  LLM + 无工具       →  纯推理，综合输出最终报告
      ↓
    END

核心设计原则:
  - 每个阶段只暴露当前需要的工具（或不暴露任何工具）
  - 没有工具的阶段 LLM 根本不可能调用工具 → 从架构层面消除无限循环
  - 阶段切换由 state.phase 驱动，条件边自动判断
  - 每个阶段有专属的系统提示词，聚焦当前任务

用法:
    from agents import dsrna_designer_agent
    result = dsrna_designer_agent.invoke(
        {"messages": [HumanMessage(content="...")]},
        config={"configurable": {"thread_id": "1"}}
    )

    python agents/dsrna_designer.py
"""

import os
import sys
import logging
from pathlib import Path
from typing import Literal

# 确保项目根目录在 Python 路径中
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import (
    SystemMessage, HumanMessage, AIMessage, ToolMessage,
)
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver
from typing_extensions import TypedDict, Annotated

from tools import DSRNA_DESIGNER_TOOLS, oligowalk_tool, primer3_tool

# ============================================================
# 环境配置
# ============================================================
for _env_path in [
    _PROJECT_ROOT / ".env",
    _PROJECT_ROOT / "test" / "quickstart" / ".env",
]:
    if _env_path.exists():
        load_dotenv(_env_path, override=True)
        break
else:
    load_dotenv()

logger = logging.getLogger("RPA_Agent.DSRNA_Designer")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(message)s",
)

# ============================================================
# 阶段常量
# ============================================================
PHASE_SCAN = "scan"
PHASE_ANALYZE = "analyze"
PHASE_PRIMERS = "primers"
PHASE_REPORT = "report"

# ============================================================
# 阶段专属系统提示词
# ============================================================

_BASE_IDENTITY = (
    "你是一名专业的 dsRNA 设计专家。你的任务是生成可用于实验的 siRNA 候选列表，"
    "为扩增设计 dsRNA 片段，并提供兼容体外转录的 PCR 引物（附加 T7 启动子）。"
    "所有与用户的沟通请使用中文。"
)

SCAN_PROMPT = _BASE_IDENTITY + (
    "\n\n## 当前阶段：siRNA 扫描\n\n"
    "**直接调用 oligowalk 工具**扫描用户提供的靶标 mRNA 序列。\n\n"
    "⚠️ 关键规则：\n"
    "1. 使用 **function calling** 格式调用工具，**绝对不要**将工具调用写成文本、代码块或 XML。\n"
    "2. 如果序列需要清理（U→T, 小写→大写），在工具参数的 sequence 字段中直接使用清理后的序列，"
    "**不需要在调用前输出清理过程的文字说明**——先调工具，拿到结果后再简要总结。\n"
    "3. 参数建议：run_type='fast', oligo_length=21, top_n=10。"
)

ANALYZE_PROMPT = _BASE_IDENTITY + (
    "\n\n## 当前阶段：候选分析与片段设计\n\n"
    "oligowalk 扫描已完成，结果在上方消息中。现在你需要进行纯分析（本阶段没有任何工具可用）：\n\n"
    "1. **序列验证**：验证输入序列（仅 A/T/G/C），报告长度、GC%。\n"
    "2. **siRNA 排名**：根据热力学评分对候选 siRNA 排名，选出 Top 5，给出每个的理由。\n"
    "3. **dsRNA 片段设计**：设计覆盖 top siRNA 位点的 dsRNA 片段（400-500 bp），"
    "明确给出每个片段的起始-结束坐标（1-based）和序列。\n\n"
    "输出格式要求：对每个 dsRNA 片段，使用以下格式明确标注：\n"
    "```\n"
    "## 片段 1: 位置 150-600 (451bp)\n"
    "序列: ATCG...\n"
    "包含 siRNA: #1, #3\n"
    "```\n\n"
    "完成后请声明「片段设计完成，共 N 个片段待设计引物」，然后停止。"
)

PRIMERS_PROMPT = _BASE_IDENTITY + (
    "\n\n## 当前阶段：引物设计\n\n"
    "上一阶段已设计好 dsRNA 片段（含坐标和序列）。为每个片段调用 **primer3** 设计 PCR 引物。\n\n"
    "⚠️ 关键规则：\n"
    "1. 使用 **function calling** 格式调用工具，**绝对不要**将工具调用写成文本、代码块或 XML。\n"
    "2. 为每个 dsRNA 片段调用一次 primer3，传入该片段的序列（不是全长序列），"
    "可指定 target_start 和 target_len 限定扩增区域。\n"
    "3. 参数建议：num_return=3~5, opt_size=20, opt_tm=60, product_size_range=[100,300]。\n"
    "4. **先调工具，拿到所有结果后再一起总结**——不要在调用前输出过多分析。"
)

REPORT_PROMPT = _BASE_IDENTITY + (
    "\n\n## 当前阶段：综合报告\n\n"
    "所有工具调用已完成。请基于上方的 oligowalk 扫描结果和 primer3 引物结果，"
    "生成完整的 dsRNA 设计报告。\n\n"
    "**本阶段没有任何工具可用，请直接输出报告。**\n\n"
    "报告应包含：\n"
    "1. **序列 QC**：长度、GC%、低复杂度检查\n"
    "2. **Top 5 siRNA 候选表格**：排名 | 序列 | 位置 | 评分 | 备注\n"
    "3. **dsRNA 片段设计表格**：片段ID | 起始-结束 | 长度 | 包含的 top siRNA | 重叠\n"
    "4. **引物表格**（每个片段 ≥3 对）：排名 | 正向 | 反向 | T7-正向 | T7-反向 | Tm | GC% | 产物大小 | 罚分\n"
    "5. **PIS 干扰效率预测**：\n"
    "   - 权重A（siRNA 效能 60%）+ 权重B（扩增效率 30%）+ 权重C（序列背景 10%）\n"
    "   - 分级：高（>85%）、中等（60-85%）、低（<60%）\n"
    "6. **结论与推荐**\n\n"
    "完成后声明：「dsRNA 设计和引物集成已完成。」"
)

# ============================================================
# 状态定义
# ============================================================
class AgentState(TypedDict):
    """dsRNA 设计师工作流状态。

    phase 追踪当前工作流阶段，驱动条件边的路由决策。
    每个阶段有专属的工具集和系统提示词。
    """
    messages: Annotated[list, add_messages]
    phase: str          # PHASE_SCAN | PHASE_ANALYZE | PHASE_PRIMERS | PHASE_REPORT
    llm_calls: int      # 统计 LLM 调用次数（调试/监控用）
    retry_count: int    # 当前阶段的工具调用格式纠正重试次数


# ============================================================
# 工具函数
# ============================================================
def sanitize_text(text: str) -> str:
    """清洗非法 Unicode surrogate 字符（DeepSeek 防御）。"""
    return text.encode("utf-8", errors="replace").decode("utf-8")


def _has_tool_result(messages: list, tool_name: str) -> bool:
    """检查消息历史中是否已有指定工具的 ToolMessage（表示工具已被执行过）。"""
    for msg in reversed(messages):
        if isinstance(msg, ToolMessage) and getattr(msg, "name", None) == tool_name:
            return True
    return False


def _looks_like_tool_call_intent(content: str, tool_names: list[str]) -> bool:
    """检测 LLM 是否将工具调用写成了文本而非 function calling 格式。

    DeepSeek 偶发问题：模型在长文本推理后"忘记"使用 API 原生 tool_calls，
    而是把 <tool_calls> XML 或函数名写到 content 文本中。
    """
    if not content:
        return False
    content_lower = content.lower()
    for name in tool_names:
        if name in content_lower:
            # 有工具名 + 调用意图关键词 → 大概率是想调工具但格式错了
            intent_markers = [
                "调用", "扫描", "call", "invoke", "tool",
                "开始", "运行", "执行", "使用",
            ]
            if any(m in content_lower for m in intent_markers):
                return True
    return False


# ============================================================
# 工厂函数：构建阶段化 dsRNA designer Agent
# ============================================================
def build_dsrna_designer_agent(
    model_name: str = "deepseek-chat",
    base_url: str = "https://api.deepseek.com/v1",
    api_key: str | None = None,
    temperature: float = 0.3,
    checkpointer=None,
    interrupt_before: list[str] | None = None,
):
    """构建基于阶段状态机的 dsRNA 设计师 Agent。

    4 阶段工作流: SCAN → ANALYZE → PRIMERS → REPORT
    每个阶段绑定不同的工具集（0~1 个工具），从根本上防止无限循环。

    Args:
        model_name: LLM 模型标识符。
        base_url: API 基础 URL。
        api_key: API 密钥；None 时从环境变量读取。
        temperature: LLM 温度。
        checkpointer: LangGraph 检查点；默认 MemorySaver。
        interrupt_before: 调试用节点中断列表。

    Returns:
        编译后的 StateGraph。
    """
    # ---- LLM 配置 ----
    effective_api_key = api_key or os.getenv("DEEPSEEK_API_KEY", "")
    if not effective_api_key:
        logger.warning("DEEPSEEK_API_KEY 未设置")

    base_llm = ChatOpenAI(
        model=model_name,
        base_url=base_url,
        api_key=effective_api_key,
        temperature=temperature,
    )

    # 阶段 → (工具列表, 系统提示词)
    _PHASE_CONFIG = {
        PHASE_SCAN:    ([oligowalk_tool], SCAN_PROMPT),     # 仅 oligowalk
        PHASE_ANALYZE: ([],               ANALYZE_PROMPT),  # 无工具
        PHASE_PRIMERS: ([primer3_tool],   PRIMERS_PROMPT),  # 仅 primer3
        PHASE_REPORT: ([],                REPORT_PROMPT),   # 无工具
    }

    # ---- 节点: Agent (LLM 决策) ----
    def agent_node(state: AgentState) -> dict:
        """阶段感知的 LLM 决策节点。

        根据 state.phase 选择对应的工具绑定和系统提示词。
        - SCAN/PRIMERS 阶段：LLM 只能调用当前阶段的工具
        - ANALYZE/REPORT 阶段：LLM 无工具可用，强制纯推理输出
        """
        phase = state.get("phase", PHASE_SCAN)
        tools, system_prompt = _PHASE_CONFIG.get(phase, _PHASE_CONFIG[PHASE_SCAN])

        # 浅拷贝消息列表，避免修改状态中的原始对象
        msgs = list(state["messages"])
        for msg in msgs:
            if hasattr(msg, "content") and isinstance(msg.content, str):
                msg.content = sanitize_text(msg.content)

        # 根据阶段绑定工具
        if tools:
            llm = base_llm.bind_tools(tools)
        else:
            llm = base_llm  # 不绑定任何工具 → LLM 无法发起工具调用

        full_messages = [SystemMessage(content=system_prompt)] + msgs

        current_calls = state.get("llm_calls", 0)
        logger.info(
            "[%s] 调用 LLM (第%d次, 可用工具=%s)",
            phase, current_calls + 1,
            [t.name for t in tools] if tools else "无",
        )

        response = llm.invoke(full_messages)

        if hasattr(response, "content") and isinstance(response.content, str):
            response.content = sanitize_text(response.content)

        return {
            "messages": [response],
            "llm_calls": current_calls + 1,
        }

    # ---- 节点: Tools (执行工具调用) ----
    _standard_tool_node = ToolNode(DSRNA_DESIGNER_TOOLS)

    def tool_node(state: AgentState) -> dict:
        """执行工具调用并记录日志。"""
        last_msg = state["messages"][-1]
        if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
            names = [tc["name"] for tc in last_msg.tool_calls]
            logger.info("[%s] 执行工具: %s", state.get("phase", "?"), names)
        return _standard_tool_node.invoke(state)

    # ---- 阶段切换节点 ----
    def _to_analyze(state: AgentState) -> dict:
        logger.info("阶段切换: SCAN → ANALYZE")
        return {"phase": PHASE_ANALYZE, "retry_count": 0}

    def _to_primers(state: AgentState) -> dict:
        logger.info("阶段切换: ANALYZE → PRIMERS")
        return {"phase": PHASE_PRIMERS, "retry_count": 0}

    def _to_report(state: AgentState) -> dict:
        logger.info("阶段切换: PRIMERS → REPORT")
        return {"phase": PHASE_REPORT, "retry_count": 0}

    # ---- 重试节点: DeepSeek 文本格式工具调用纠正 ----
    MAX_RETRIES = 2

    def retry_node(state: AgentState) -> dict:
        """当 DeepSeek 将工具调用写成文本而非 function calling 格式时，
        注入纠正提示，要求其使用正确的 API 格式重新发起调用。"""
        phase = state.get("phase", PHASE_SCAN)
        content = ""
        last_msg = state["messages"][-1]
        if hasattr(last_msg, "content") and last_msg.content:
            content = last_msg.content[:200]

        logger.warning(
            "[%s] 检测到文本形式工具调用 (retry=%d), 内容预览: %s",
            phase, state.get("retry_count", 0) + 1, content,
        )

        return {
            "messages": [
                HumanMessage(
                    content=(
                        "⚠️ 你刚才将工具调用写成了文本格式，系统无法执行。\n"
                        "请使用 **function calling** 功能直接发起工具调用，"
                        "不要将工具调用写成代码块、XML 标签或文字描述。\n"
                        "现在请重新发起正确的函数调用。"
                    )
                ),
            ],
            "retry_count": state.get("retry_count", 0) + 1,
        }

    # ---- 条件路由 ----
    def route_after_agent(state: AgentState) -> Literal["tools", "retry", "to_analyze", "to_primers", "to_report", "__end__"]:
        """agent 节点的条件路由。

        规则（按优先级）：
        1. 有 tool_calls → 去 tools
        2. 无 tool_calls + 文本中包含工具调用意图 + 未超重试上限 → 去 retry
        3. 无 tool_calls + SCAN + oligowalk 已执行 → to_analyze
        4. 无 tool_calls + ANALYZE → to_primers
        5. 无 tool_calls + PRIMERS + primer3 已执行 → to_report
        6. 无 tool_calls + REPORT → END
        7. 其他 → END
        """
        last_msg = state["messages"][-1]
        phase = state.get("phase", PHASE_SCAN)
        messages = state["messages"]

        # 规则 1：有待执行的工具调用
        if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
            return "tools"

        # 规则 2：检测文本形式的工具调用（DeepSeek 偶发问题）
        if phase in (PHASE_SCAN, PHASE_PRIMERS):
            tools_for_phase = [oligowalk_tool] if phase == PHASE_SCAN else [primer3_tool]
            tool_names = [t.name for t in tools_for_phase]
            content = last_msg.content if hasattr(last_msg, "content") and last_msg.content else ""
            retry_count = state.get("retry_count", 0)
            if (
                retry_count < MAX_RETRIES
                and _looks_like_tool_call_intent(content, tool_names)
            ):
                logger.warning("[%s] 疑似文本工具调用 → 路由到 retry", phase)
                return "retry"

        # 规则 3-6：无工具调用，根据阶段决定下一步
        if phase == PHASE_SCAN:
            if _has_tool_result(messages, "oligowalk"):
                return "to_analyze"
            else:
                logger.info("[%s] 无工具执行记录，对话结束（可能是闲聊）", phase)
                return END

        elif phase == PHASE_ANALYZE:
            return "to_primers"

        elif phase == PHASE_PRIMERS:
            if _has_tool_result(messages, "primer3"):
                return "to_report"
            else:
                logger.info("[%s] 无 primer3 执行记录，对话结束", phase)
                return END

        elif phase == PHASE_REPORT:
            logger.info("对话结束 (共%d次LLM调用)", state.get("llm_calls", 0))
            return END

        return END

    # ---- 组装图 ----
    builder = StateGraph(AgentState)

    builder.add_node("agent", agent_node)
    builder.add_node("tools", tool_node)
    builder.add_node("retry", retry_node)
    builder.add_node("to_analyze", _to_analyze)
    builder.add_node("to_primers", _to_primers)
    builder.add_node("to_report", _to_report)

    builder.add_edge(START, "agent")

    builder.add_conditional_edges(
        "agent",
        route_after_agent,
        {
            "tools": "tools",
            "retry": "retry",
            "to_analyze": "to_analyze",
            "to_primers": "to_primers",
            "to_report": "to_report",
            END: END,
        },
    )

    # tools 总是回到 agent（当前阶段的 mini-ReAct 循环）
    builder.add_edge("tools", "agent")

    # retry 回到 agent（同一阶段重试，不切换阶段）
    builder.add_edge("retry", "agent")

    # 阶段切换节点总是回到 agent（以新阶段身份重新决策）
    builder.add_edge("to_analyze", "agent")
    builder.add_edge("to_primers", "agent")
    builder.add_edge("to_report", "agent")

    # ---- 编译 ----
    effective_checkpointer = checkpointer if checkpointer is not None else MemorySaver()
    return builder.compile(
        checkpointer=effective_checkpointer,
        interrupt_before=interrupt_before,
    )


# ============================================================
# 预编译单例
# ============================================================
dsrna_designer_agent = build_dsrna_designer_agent()


# ============================================================
# 交互式 CLI
# ============================================================
if __name__ == "__main__":
    agent = build_dsrna_designer_agent()
    # 每次对话使用独立的 thread_id，避免状态污染
    import uuid
    config = {"configurable": {"thread_id": f"dsrna-{uuid.uuid4().hex[:8]}"}}

    print("=" * 60)
    print("🧬 dsRNA Designer Agent（4 阶段工作流）")
    print(f"   SCAN → ANALYZE → PRIMERS → REPORT")
    print(f"   工具: {[t.name for t in DSRNA_DESIGNER_TOOLS]}")
    print(f"   模型: deepseek-chat")
    print("=" * 60)
    print("输入 DNA 序列开始 dsRNA 设计，或输入 'quit' 退出\n")

    # 初始问候
    try:
        greeting = agent.invoke(
            {
                "messages": [HumanMessage(content="你好，请简单介绍一下你能做什么。")],
                "phase": PHASE_SCAN,
            },
            config,
        )
        last_msg = greeting["messages"][-1]
        if hasattr(last_msg, "content"):
            print(f"🤖 Agent: {last_msg.content}\n")
    except Exception as e:
        logger.error("初始问候失败: %s", e)
        print(f"⚠️ 初始问候失败: {e}\n")

    while True:
        try:
            user_input = input("🧑 You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n👋 再见！")
            break

        if user_input.lower() in ("quit", "exit", "q"):
            print("👋 再见！")
            break
        if not user_input:
            continue

        try:
            # 每次用户输入都从 SCAN 阶段重新开始
            result = agent.invoke(
                {
                    "messages": [HumanMessage(content=user_input)],
                    "phase": PHASE_SCAN,
                },
                config,
            )
            last_msg = result["messages"][-1]
            if hasattr(last_msg, "content"):
                print(f"\n🤖 Agent: {last_msg.content}\n")
            else:
                print(f"\n🤖 Agent: [无文本内容]\n")
        except Exception as e:
            logger.exception("Agent 调用失败")
            print(f"\n❌ 错误: {e}\n")
            print("请重试或输入 'quit' 退出。")
