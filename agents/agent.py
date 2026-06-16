"""
agent.py — Single ReAct Agent for RNAi pesticide design.

Architecture:
  One ReAct loop (create_react_agent) with all 13 tools bound.
  Skill documents injected into system prompt as domain knowledge.
  LLM decides which tools to call and in what order — no fixed pipeline.

Usage:
    from agents.agent import rnai_agent

    result = rnai_agent.invoke(
        {"messages": [("user", "Design dsRNA for sequence: ATGC...")]},
        config={"configurable": {"thread_id": "session-1"}}
    )
"""

from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from langchain_core.language_models import BaseChatModel
from langgraph.checkpoint.memory import MemorySaver
from langchain.agents import create_agent

from skill.skill_loader import build_skills
from tools import ALL_TOOLS

# ============================================================
# Role prompt
# ============================================================
AGENT_ROLE = """You are an RNAi pesticide design expert assistant.

You have access to 17 bioinformatics tools covering:
- Target discovery: BLAST against insect genomes, annotation lookup, CDS extraction, sequence clipping
- dsRNA design: siRNA thermodynamic scanning (OligoWalk), PCR primer design (Primer3)
- Safety assessment: BLAST against non-target organisms, sequence retrieval, pairwise alignment (Clustal)
- Literature: PubMed search, PubMed fetch, OpenAlex search
- Phylogenetics: Kinship analysis
- Knowledge: search_skills — retrieve domain-specific design guides on demand
- Data: lookup_data — check availability of locally stored genome databases

Guidelines:
1. Plan your approach before calling tools. Break complex tasks into steps.
2. Call tools one at a time or in parallel when independent.
3. Interpret tool outputs honestly — report what tools actually measured.
4. When making recommendations, use appropriate hedging language (suggest, may, candidate).
5. Never claim experimental validation from computational predictions alone.
6. IMPORTANT — Your tool list is FINITE and EXACT. There is NO "parallel" tool and NO "multi_tool_use.parallel" wrapper. The ability to call multiple tools in the same step is a built-in platform capability, not a separate tool. Never mention, list, or describe "parallel" as if it were a registered tool.
7. Use `search_skills` when you need detailed domain knowledge (e.g., dsRNA design parameters, primer design constraints, safety assessment protocols). Use `list_tools` when you need to discover available tools or inspect a tool's parameter schema.
"""


def build_rnai_agent(
    llm: BaseChatModel | None = None,
    checkpointer=None,
    skills: tuple[str, ...] | None = None,
):
    """Build a single ReAct agent with all tools bound.

    Behavioral skills (principles, evidence, tool, recommendation) are always
    injected into the system prompt. Domain-specific skills (dsrna_design, etc.)
    are retrieved on demand via the `search_skills` tool.

    Args:
        llm: Language model instance. Uses get_default_llm() if None.
        checkpointer: LangGraph checkpointer (default: MemorySaver).
        skills: Behavioral skill names to always inject into system prompt.
                Default: ("principles", "evidence", "tool", "recommendation").
                Domain skills (e.g., "dsrna_design") are NOT included here —
                the LLM retrieves them via search_skills tool when needed.

    Returns:
        Compiled LangGraph agent (create_react_agent).
    """
    if llm is None:
        from llm_config import get_default_llm
        llm = get_default_llm()

    if skills is None:
        skills = ("principles", "evidence", "tool", "recommendation")

    # Build system prompt: role + skill documents
    skill_content = build_skills(*skills)
    system_prompt = f"{AGENT_ROLE}\n\n{skill_content}"

    # Create the agent
    agent = create_agent(
        model=llm,
        tools=ALL_TOOLS,
        system_prompt=system_prompt,
        checkpointer=checkpointer or MemorySaver(),
    )

    return agent


# ============================================================
# Singleton
# ============================================================
from llm_config import get_default_llm
rnai_agent = build_rnai_agent(llm=get_default_llm())


# ============================================================
# CLI entry point
# ============================================================
def _strip_surrogates(text: str) -> str:
    """Remove surrogate characters that cause UTF-8 encoding errors."""
    return text.encode("utf-8", "replace").decode("utf-8")


def _sanitize_checkpoint(agent, config) -> None:
    """Strip surrogate chars from all messages in checkpoint to prevent next-call crash."""
    state = agent.get_state(config)
    if not state or not state.values:
        return
    messages = state.values.get("messages", [])
    modified = False
    for msg in messages:
        if hasattr(msg, "content") and isinstance(msg.content, str):
            cleaned = _strip_surrogates(msg.content)
            if cleaned != msg.content:
                msg.content = cleaned
                modified = True
    if modified:
        agent.update_state(config, {"messages": messages})


if __name__ == "__main__":
    import uuid

    thread_id = f"cli-{uuid.uuid4().hex[:8]}"
    config = {"configurable": {"thread_id": thread_id}}
    print("=" * 60)
    print("  RNAi Pesticide Design Agent")
    print("  Type 'exit' or 'quit' to end the session.")
    print("=" * 60)
    print()

    while True:
        try:
            user_input = input(">>> ")
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if user_input.strip().lower() in ("exit", "quit"):
            break
        if not user_input.strip():
            continue

        # Sanitize user input before sending
        user_input = _strip_surrogates(user_input)

        try:
            result = rnai_agent.invoke(
                {"messages": [("user", user_input)]},
                config=config,
            )
        except (UnicodeEncodeError, UnicodeDecodeError):
            # If crash due to surrogate chars in checkpoint, sanitize and retry once
            _sanitize_checkpoint(rnai_agent, config)
            result = rnai_agent.invoke(
                {"messages": [("user", user_input)]},
                config=config,
            )

        # Sanitize checkpoint to prevent next-round crash
        _sanitize_checkpoint(rnai_agent, config)

        # Print the last AI message
        for msg in reversed(result["messages"]):
            if msg.type == "ai" and msg.content.strip():
                print()
                print(msg.content)
                print()
                break
