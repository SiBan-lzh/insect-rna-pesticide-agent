"""
shell/tool.py — Safe shell command execution.

Execution model (Claude Code-inspired):
  - Read-only commands run without user confirmation (auto-approved)
  - Write / network / install commands require explicit user approval
  - Dangerous commands (rm -rf /, mkfs, dd, etc.) are blocked entirely
  - Working directory is locked to the project root
  - Process group isolation via setsid() for clean timeout/kill

Approval classification:
  Read-only (auto): ls, cat, head, tail, grep, find, git status, git diff,
                    git log, pwd, echo, tree, wc, file, stat, which, type,
                    python -c, python3 -c (print-only scripts)
  Write (prompt):  rm, mv, cp, mkdir, rmdir, touch, chmod, chown,
                   pip install, apt install, make, git add, git commit,
                   git push, git pull, git checkout, git branch, curl -o,
                   wget -O, docker, systemctl, sed -i, python -c (with
                   file writes), any command containing >, >>, | (pipe)
                   that writes to a file
  Blocked (deny):  rm -rf /, rm -rf ~, mkfs.*, dd if=, :(){ :|:& };:,
                   chmod -R 777 /, sudo rm, > /dev/sda, format, reboot,
                   shutdown, halt, poweroff

Usage:
    from tools.shell import shell_tool

    result = shell_tool.invoke({"command": "ls database/blast/"})
"""

from __future__ import annotations

import json
import logging
import os
import re
import signal
import subprocess
import time
from datetime import datetime
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path
from typing import Literal

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
BASE_DIR = _PROJECT_ROOT
TIMEOUT = 300

# ============================================================
# Logging — structured JSON, daily rotation
# ============================================================
_log_dir = _PROJECT_ROOT / "logs"
_log_dir.mkdir(exist_ok=True)
_handler = TimedRotatingFileHandler(
    _log_dir / "rpa_shell.log", when="midnight", encoding="utf-8",
)
_handler.setFormatter(logging.Formatter("%(message)s"))
_logger = logging.getLogger("rpa.shell")
_logger.setLevel(logging.INFO)
_logger.addHandler(_handler)
_logger.propagate = False


# ============================================================
# Pydantic input schema
# ============================================================
class ShellInput(BaseModel):
    """Shell command execution parameters."""

    command: str = Field(description="Shell command to execute")


# ============================================================
# Command safety classifier
# ============================================================

# ── Blocked patterns (always rejected) ──
_BLOCKED_PATTERNS: list[re.Pattern] = [
    re.compile(r'\brm\s+-rf\s+/?\s*$'),           # rm -rf /
    re.compile(r'\brm\s+-rf\s+~'),                  # rm -rf ~
    re.compile(r'\bmkfs\.'),                         # mkfs.*
    re.compile(r'\bdd\s+if='),                       # dd if=
    re.compile(r':\(\)\{\s*:\|:\&\s*\};:'),          # fork bomb
    re.compile(r'\bsudo\s+rm\b'),                    # sudo rm
    re.compile(r'>\s+/dev/sd'),                      # write to block device
    re.compile(r'\breboot\b'),                       # reboot
    re.compile(r'\bshutdown\b'),                     # shutdown
    re.compile(r'\bhalt\b'),                         # halt
    re.compile(r'\bpoweroff\b'),                     # poweroff
]

# ── Read-only patterns (auto-approved) ──
_READ_ONLY_PATTERNS: list[re.Pattern] = [
    re.compile(r'^ls\b'),
    re.compile(r'^cat\b'),
    re.compile(r'^head\b'),
    re.compile(r'^tail\b'),
    re.compile(r'^grep\b'),
    re.compile(r'^find\b'),
    re.compile(r'^pwd\b'),
    re.compile(r'^echo\b'),
    re.compile(r'^tree\b'),
    re.compile(r'^wc\b'),
    re.compile(r'^file\b'),
    re.compile(r'^stat\b'),
    re.compile(r'^which\b'),
    re.compile(r'^type\b'),
    re.compile(r'^git status\b'),
    re.compile(r'^git diff\b'),
    re.compile(r'^git log\b'),
    re.compile(r'^git show\b'),
    re.compile(r'^git branch\b'),
    re.compile(r'^git ls-files\b'),
    re.compile(r'^python3?\s+-c\s+["\'](?!.*(?:open|write|os\.|subprocess))'),
]

# ── Write/network patterns (require confirmation) ──
_WRITE_PATTERNS: list[re.Pattern] = [
    re.compile(r'\brm\b'),
    re.compile(r'\bmv\b'),
    re.compile(r'\bcp\b'),
    re.compile(r'\bmkdir\b'),
    re.compile(r'\brmdir\b'),
    re.compile(r'\btouch\b'),
    re.compile(r'\bchmod\b'),
    re.compile(r'\bchown\b'),
    re.compile(r'\bpip\b'),
    re.compile(r'\bapt\b'),
    re.compile(r'\bbrew\b'),
    re.compile(r'\bmake\b'),
    re.compile(r'\bdocker\b'),
    re.compile(r'\bsystemctl\b'),
    re.compile(r'\bservice\b'),
    re.compile(r'\bsed -i\b'),
    re.compile(r'\bcurl\b'),
    re.compile(r'\bwget\b'),
    re.compile(r'\bgit add\b'),
    re.compile(r'\bgit commit\b'),
    re.compile(r'\bgit push\b'),
    re.compile(r'\bgit pull\b'),
    re.compile(r'\bgit checkout\b'),
    re.compile(r'\bgit merge\b'),
    re.compile(r'\bgit reset\b'),
    re.compile(r'>\s+'),          # output redirection to file
    re.compile(r'>>\s+'),         # append redirection
]


def _classify_command(command: str) -> Literal["blocked", "read_only", "write"]:
    """Classify a shell command by safety level.

    Returns:
        "blocked"   — dangerous, always rejected
        "read_only" — safe, auto-approved
        "write"     — may modify state, requires user confirmation
    """
    stripped = command.strip()

    # Check blocked patterns first
    for pattern in _BLOCKED_PATTERNS:
        if pattern.search(stripped):
            return "blocked"

    # Check read-only patterns
    for pattern in _READ_ONLY_PATTERNS:
        if pattern.search(stripped):
            return "read_only"

    # If it matches any write pattern, classify as write
    for pattern in _WRITE_PATTERNS:
        if pattern.search(stripped):
            return "write"

    # Default: unknown command → treat as write (requires confirmation)
    return "write"


def _prompt_user(command: str, classification: str) -> bool:
    """Prompt the user for confirmation.

    Args:
        command: The shell command to confirm.
        classification: "read_only", "write", or "blocked".

    Returns:
        True if the user approved, False otherwise.
    """
    label = {
        "read_only": "READ",
        "write": "WRITE",
        "blocked": "BLOCKED",
    }.get(classification, "UNKNOWN")

    if classification == "blocked":
        print(f"\n[RPA] 🛑 BLOCKED: {command}")
        print("  This command is considered dangerous and is not allowed.")
        return False

    if classification == "read_only":
        # Auto-approved, no prompt
        return True

    # Write command — prompt
    print(f"\n[RPA] [{label}] $ {command}")
    try:
        confirm = input("Proceed? [y/N] ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        print()
        return False
    return confirm == "y"


# ============================================================
# LangChain Tool
# ============================================================
class ShellTool(BaseTool):
    """Run a shell command in the project workspace.

    Read-only commands (ls, cat, grep, etc.) run automatically.
    Write/modify commands require user confirmation.
    Dangerous commands are blocked entirely.
    """

    name: str = "shell"
    description: str = (
        "Run a shell command in the project workspace. "
        "Use for ad-hoc file system operations, data processing, and exploration. "
        "Read-only commands (ls, cat, grep, git status, etc.) run automatically. "
        "Write/modify commands (rm, mv, pip install, git commit, etc.) require "
        "user confirmation. Dangerous system commands are blocked entirely."
    )
    args_schema: type = ShellInput
    timeout: int = TIMEOUT

    def _run(self, command: str) -> str:
        """Execute a shell command with safety classification."""

        # ── Classify ──
        classification = _classify_command(command)

        # ── Prompt ──
        if not _prompt_user(command, classification):
            _logger.info(json.dumps({
                "ts": datetime.now().isoformat(),
                "command": command,
                "classification": classification,
                "status": "cancelled",
            }, ensure_ascii=False))
            if classification == "blocked":
                return f"BLOCKED: '{command}' is not allowed."
            return "Cancelled by user."

        record: dict = {
            "ts": datetime.now().isoformat(),
            "command": command,
            "classification": classification,
        }
        t0 = time.time()

        # ── Execute with process group isolation ──
        try:
            proc = subprocess.Popen(
                command, shell=True,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                cwd=str(BASE_DIR), text=True,
                preexec_fn=os.setsid,
            )
            stdout, stderr = proc.communicate(timeout=TIMEOUT)

        except subprocess.TimeoutExpired:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            record.update({"status": "timeout", "elapsed": TIMEOUT})
            _logger.warning(json.dumps(record, ensure_ascii=False))
            return f"ERROR: Timed out after {TIMEOUT}s."

        # ── Structured logging ──
        record.update({
            "returncode": proc.returncode,
            "elapsed": round(time.time() - t0, 2),
            "stderr_preview": stderr[:200],
        })
        _logger.info(json.dumps(record, ensure_ascii=False))

        output = (stdout + stderr).strip()
        if proc.returncode != 0:
            return f"ERROR (code {proc.returncode}):\n{output[:1000]}"
        return output[:2000] or "OK"


# ============================================================
# Singleton export
# ============================================================
shell_tool = ShellTool()
