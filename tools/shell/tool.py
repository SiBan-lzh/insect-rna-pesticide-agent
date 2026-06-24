"""
shell/tool.py — Run general shell commands in the RPA sandbox workspace.

Provides a controlled shell execution environment with:
- Fixed working directory (sandbox/)
- User confirmation before execution
- Timeout and process group isolation
- Structured JSON logging with rotation
"""

from __future__ import annotations

import json
import logging
import os
import signal
import subprocess
import time
from datetime import datetime
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
BASE_DIR = _PROJECT_ROOT / "sandbox"
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
# LangChain Tool
# ============================================================
class ShellTool(BaseTool):
    """Run a shell command in the RPA sandbox workspace."""

    name: str = "shell"
    description: str = (
        "Run a general shell command in the RPA sandbox workspace. "
        "Use for ad-hoc file system operations, downloads, and data processing tasks "
        "that don't have a dedicated tool."
    )
    args_schema: type = ShellInput
    timeout: int = TIMEOUT

    def _run(self, command: str) -> str:
        """Execute a shell command with confirmation and timeout."""

        # --- User confirmation ---
        print(f"\n[RPA] $ {command}")
        try:
            confirm = input("Proceed? [y/N] ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print()
            return "Cancelled by user."
        if confirm != "y":
            _logger.info(json.dumps({
                "ts": datetime.now().isoformat(),
                "command": command,
                "status": "cancelled",
            }, ensure_ascii=False))
            return "Cancelled by user."

        record: dict = {"ts": datetime.now().isoformat(), "command": command}
        t0 = time.time()

        # --- Execute with process group isolation ---
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

        # --- Structured logging ---
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
