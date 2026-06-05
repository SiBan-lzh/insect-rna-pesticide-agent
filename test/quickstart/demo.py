import os
from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

from langchain_core.messages import AnyMessage, SystemMessage, ToolMessage
from typing_extensions import TypedDict, Annotated
import operator

from typing import Literal
from langgraph.graph import StateGraph, START, END

load_dotenv()  # 读取当前目录的 .env 文件

model = ChatOpenAI(
      model="deepseek-chat",
      base_url="https://api.deepseek.com/v1",
      api_key=os.getenv("DEEPSEEK_API_KEY"),
      temperature=0,
  )


# Define tools
@tool
def multiply(a: int, b: int) -> int:
    """Multiply `a` and `b`.

    Args:
        a: First int
        b: Second int
    """
    return a * b


@tool
def add(a: int, b: int) -> int:
    """Adds `a` and `b`.

    Args:
        a: First int
        b: Second int
    """
    return a + b


@tool
def divide(a: int, b: int) -> float:
    """Divide `a` and `b`.

    Args:
        a: First int
        b: Second int
    """
    return a / b

# messeges state
class MessagesState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]
    llm_calls: int

# Augment the LLM with tools
tools = [add, multiply, divide]
tools_by_name = {tool.name: tool for tool in tools}
model_with_tools = model.bind_tools(tools)

# llm node
def sanitize_text(text: str) -> str:
    """清洗非法 Unicode 字符（surrogate），防止 DeepSeek 偶发脏数据导致崩溃"""
    return text.encode("utf-8", errors="replace").decode("utf-8")


def llm_call(state: dict):
    """LLM decides whether to call a tool or not"""

    # 清洗每一条消息，防止累积的脏数据导致 JSON 序列化失败
    cleaned = []
    for msg in state["messages"]:
        if hasattr(msg, "content") and isinstance(msg.content, str):
            msg.content = sanitize_text(msg.content)
        cleaned.append(msg)

    return {
        "messages": [
            model_with_tools.invoke(
                [SystemMessage(
                    content="You are a helpful assistant tasked with performing arithmetic on a set of inputs."
                )]
                + cleaned
            )
        ],
        "llm_calls": state.get("llm_calls", 0) + 1,
    }

# tool node
def tool_node(state: dict):
    """Performs the tool call"""

    result = []
    for tool_call in state["messages"][-1].tool_calls:
        tool = tools_by_name[tool_call["name"]]
        observation = tool.invoke(tool_call["args"])
        result.append(ToolMessage(content=str(observation), tool_call_id=tool_call["id"]))
    return {"messages": result}

# how to end 
def should_continue(state: MessagesState) -> Literal["tool_node", END]:
    """Decide if we should continue the loop or stop based upon whether the LLM made a tool call"""

    messages = state["messages"]
    last_message = messages[-1]

    # If the LLM makes a tool call, then perform an action
    if last_message.tool_calls:
        return "tool_node"

    # Otherwise, we stop (reply to the user)
    return END


# Build workflow
agent_builder = StateGraph(MessagesState)

# Add nodes
agent_builder.add_node("llm_call", llm_call)
agent_builder.add_node("tool_node", tool_node)

# Add edges to connect nodes
agent_builder.add_edge(START, "llm_call")
agent_builder.add_conditional_edges(
    "llm_call",
    should_continue,
    ["tool_node", END]
)
agent_builder.add_edge("tool_node", "llm_call")

# Compile the agent
agent = agent_builder.compile()

# Show the agent (requires IPython/Jupyter — skip in terminal)
# from IPython.display import Image, display
# display(Image(agent.get_graph(xray=True).draw_mermaid_png()))

# 交互式对话
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver

# 用 MemorySaver 记住对话历史（当前会话内有效，关了就没了）
agent = agent_builder.compile(checkpointer=MemorySaver())
config = {"configurable": {"thread_id": "1"}}

print("=" * 50)
print("🤖 智能体已就绪！输入 'quit' 或 'exit' 退出")
print("=" * 50)

while True:
    user_input = input("\n🧑 You: ").strip()
    if user_input.lower() in ("quit", "exit", "q"):
        print("👋 再见！")
        break
    if not user_input:
        continue

    result = agent.invoke(
        {"messages": [HumanMessage(content=user_input)]},
        config,
    )

    # 只打印最后一条消息（AI 的回复）
    last_msg = result["messages"][-1]
    print(f"\n🤖 Agent: {last_msg.content}")
