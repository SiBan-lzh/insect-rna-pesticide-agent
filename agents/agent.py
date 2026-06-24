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

from skills import build_skills
from tools import SYSTEM_TOOLS, TOOL_REGISTRY

# ============================================================
# Role prompt
# ============================================================
AGENT_ROLE = """You are an RNAi pesticide design expert assistant.

You have 4 discovery tools — call these FIRST before using specialized tools:
- `list_tools` — browse available bioinformatics tools, their descriptions and parameters
- `list_skills` — list available skill documents (design guides, protocols)
- `list_ragbase` — list available knowledge bases (experimental records, literature)
- `list_database` — check available local genome data by species

After discovering what's available:
- Use `read_skill` to load full skill document content
- Use `search_knowledge` to search a specific knowledge base
- Use specialized tools (BLAST, OligoWalk, Primer3, etc.) for analysis

Guidelines:
1. Plan your approach. Use discovery tools first.
2. Call tools one at a time or in parallel when independent.
3. Interpret tool outputs honestly — report what tools actually measured.
4. When making recommendations, use appropriate hedging language (suggest, may, candidate).
5. Never claim experimental validation from computational predictions alone.
6. Your tool list is FINITE and EXACT. There is NO "parallel" tool wrapper.
   The ability to call multiple tools in the same step is a built-in platform capability.
7. Use `read_skill` when you need detailed domain knowledge. Use `list_tools` to inspect tool schemas.
8. Use memory tools proactively:
   - At conversation start, call `load_memory()` to check for saved context
     (user preferences, recent work, task progress).
   - Save important information via `save_memory(slot, key, value)` when you learn:
     * User preferences or settings → slot="preferences"
     * Ongoing task progress, decisions made → slot="task_progress"
     * Key project context that should persist → slot="recent_work"
   - The memory persists across sessions. Use it to maintain continuity.
9. Shell execution safety:
   - You have a `shell` tool that can run arbitrary commands on the host system.
   - Always ask yourself: is this command safe? .
   - The shell tool already asks the user "Proceed? [y/N]" before running — but you
     must still exercise judgment about what is reasonable to propose.
   - Prefer safe, read-only operations when possible.
"""


def build_rnai_agent(
    llm: BaseChatModel,
    checkpointer=None,
    skills: tuple[str, ...] | None = None,
):
    """Build a single ReAct agent with all tools bound.

    Behavioral skills are always injected into the system prompt.
    Domain-specific skills are retrieved on demand via the `read_skill` tool.

    Args:
        llm: Language model instance.
        checkpointer: LangGraph checkpointer (default: MemorySaver).
        skills: Behavioral skill names to always inject into system prompt.
                Default: ("behavior1",).
                Domain skills (e.g., "dsrna_design") are NOT included here —
                the LLM retrieves them via `read_skill` tool when needed.

    Returns:
        Compiled LangGraph agent (create_react_agent).
    """
    if skills is None:
        skills = ("behavior1",)

    # Build system prompt: role + skill documents
    skill_content = build_skills(*skills)
    system_prompt = f"{AGENT_ROLE}\n\n{skill_content}"

    # Create the agent
    all_tools = TOOL_REGISTRY + SYSTEM_TOOLS
    agent = create_agent(
        model=llm,
        tools=all_tools,
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

    from tool_config import validate_paths
    missing = validate_paths(verbose=False)
    if missing:
        print(f"⚠️  {len(missing)} data paths not found on disk:")
        for name in missing:
            print(f"     {name}")
        print()

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
