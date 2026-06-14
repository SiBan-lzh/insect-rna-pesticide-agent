"""
dsrna_designer.py — LangGraph dsRNA design workflow.

Graph:
    START -> sequence_validate -> oligowalk_scan -> fragment_design
          -> [Send(primer_design_subgraph) x N] -> pis_score
          -> report_generate -> END

PrimerDesignSubgraph:
    START -> primer3_design -> primer_validate -> END

LLM used in: fragment_design, pis_score, report_generate
Deterministic: sequence_validate
Tool-driven: oligowalk_scan, primer_design_subgraph
"""

from __future__ import annotations

import json
import logging
import operator
import re
import sys
from pathlib import Path
from typing import Any, Literal

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send
from typing_extensions import Annotated, TypedDict

from skill.skill_loader import build_skills
from tools import oligowalk_tool, primer3_tool

logger = logging.getLogger("RPA_Agent.DSRNADesigner")

MAX_TOOL_RETRIES = 3
def _merge_dicts(a: dict, b: dict) -> dict:
    """Merge two dicts for parallel-safe Annotated accumulator."""
    return {**(a or {}), **(b or {})}


# ============================================================
# Constants
# ============================================================
MIN_FRAGMENT_SIZE = 300
MAX_FRAGMENT_SIZE = 500
DEFAULT_FRAGMENT_SIZE = 400
MIN_PRIMER_PAIRS = 3
T7_PREFIX = "TAATACGACTCACTATAGGG"


# ============================================================
# State definitions
# ============================================================
class DSRNADesignState(TypedDict, total=False):
    sequence_input: str
    language: str
    sequence_clean: str
    sequence_valid: bool
    sequence_qc: dict
    oligowalk_mode: str
    oligowalk_result: dict
    retry_counts: Annotated[dict, _merge_dicts]
    fragment_proposals: list[dict]
    fragment_results: Annotated[list[dict], operator.add]
    pis_scores: list[dict]
    final_report: dict
    errors: Annotated[list[dict], operator.add]


class PrimerDesignState(TypedDict, total=False):
    fragment: dict
    primer3_result: dict
    retry_counts: dict
    primer_results: list[dict]
    fragment_result: dict
    fragment_results: Annotated[list[dict], operator.add]
    errors: Annotated[list[dict], operator.add]


# ============================================================
# Helpers
# ============================================================
def _json_loads_tool_result(raw: Any) -> dict:
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        return json.loads(raw)
    return {"status": "error", "error": "Unexpected tool result type", "details": str(type(raw))}


def _tool_error(node: str, tool_name: str, error: str, attempt: int) -> dict:
    return {"node": node, "tool": tool_name, "attempt": attempt, "error": error}


def _detect_language(text: str) -> str:
    if any("一" <= c <= "鿿" for c in text):
        return "Chinese"
    return "English"


def _normalize_sequence(seq: str) -> str:
    cleaned = "".join(str(seq or "").upper().split()).replace("U", "T")
    invalid = sorted(set(cleaned) - {"A", "T", "C", "G", "N"})
    if not cleaned:
        raise ValueError("sequence is empty")
    if invalid:
        raise ValueError(f"Invalid nucleotide characters: {invalid}")
    return cleaned


def _gc_percent(seq: str) -> float:
    if not seq:
        return 0.0
    gc = sum(1 for c in seq if c in "GC")
    return round(gc / len(seq) * 100, 2)


def _check_low_complexity(seq: str) -> list[str]:
    warnings = []
    seq_upper = seq.upper()
    for char in ("A", "T", "C", "G"):
        streak = 0
        max_streak = 0
        for c in seq_upper:
            streak = streak + 1 if c == char else 0
            max_streak = max(max_streak, streak)
        if max_streak >= 6:
            warnings.append(f"High {char} repetition: {max_streak} consecutive {char}s")
    return warnings


def _parse_oligowalk_candidates(result: dict, input_seq: str = "") -> list[dict]:
    """Extract siRNA candidates from OligoWalk tool result.

    The OligoWalk report may or may not include the Oligo sequence and %GC columns
    depending on binary version. To make downstream reporting robust, this function:
      1. Reads position, overall_score from the row (these are always present).
      2. Reads sequence/GC% from the row when available, else reconstructs them
         by slicing `input_seq` at the candidate position and computing GC%.
    """
    if result.get("status") != "success":
        return []
    candidates = []
    for row in (result.get("top_candidates", []) or []):
        try:
            pos_str = row.get("Pos.", "")
            pos = int(float(pos_str)) if pos_str else 0
            overall_str = row.get("Overall (kcal/mol)", "0")
            overall = float(overall_str)
        except (ValueError, TypeError):
            continue

        # Try row's own sequence first; fall back to slicing the input sequence.
        seq = (
            row.get("Oligo")
            or row.get("OligoSeq")
            or row.get("Sequence")
            or row.get("sequence")
            or ""
        )
        if not seq and input_seq and pos > 0:
            # OligoWalk positions are 1-based; slice [pos-1 : pos-1+oligo_length]
            oligo_length = int((result.get("parameters") or {}).get("oligo_length") or 21)
            start = pos - 1
            end = start + oligo_length
            if 0 <= start < end <= len(input_seq):
                seq = input_seq[start:end]

        # Compute GC% from the actual sequence (or accept row's value if present).
        gc = None
        for key in ("%GC", "GC%", "gc_percent", "GC_percent"):
            if key in row and row[key] not in (None, ""):
                try:
                    gc = float(row[key])
                    break
                except (ValueError, TypeError):
                    pass
        if gc is None and seq:
            gc = _gc_percent(seq)

        candidates.append({
            "position_1based": pos,
            "sequence": seq,
            "overall_score": overall,
            "gc_percent": gc,
        })
    return candidates


def _compact_fragment_result(fr: dict) -> dict:
    """Strip raw bulk data from a fragment result for LLM prompts.

    Keeps: fragment_id, coordinates, sequence summary, siRNA coverage, primer pairs, pis_data.
    Drops: fragment_result (nested copy), pis_data original if present.
    Full data stays in final_report for supervisor/evidence use.
    """
    return {
        "fragment_id": fr.get("fragment_id"),
        "fragment_start": fr.get("fragment_start"),
        "fragment_end": fr.get("fragment_end"),
        "fragment_length": fr.get("fragment_length"),
        "fragment_sequence": fr.get("fragment_sequence", "")[:20] + "..." if fr.get("fragment_sequence") else "",
        "covered_sirna": fr.get("covered_sirna", []),
        "primer_count": fr.get("primer_count", 0),
        "best_penalty": fr.get("best_penalty"),
        "primer_results": fr.get("primer_results", []),
        "pis_data": fr.get("pis_data", {}),
        "status": fr.get("status"),
    }



def _extract_json_from_llm_response(raw: str | None) -> dict | None:
    """Robustly extract JSON from LLM response text.

    Tries in order:
    1. Strip markdown fences (```json ... ```) and json.loads directly
    2. Find first {...} block via regex and json.loads
    3. Return None on failure (caller decides fallback)
    """
    if not raw:
        return None
    text = raw.strip()

    # Strategy 1: markdown fence
    if text.startswith("```"):
        parts = text.split("```")
        for part in parts[1::2]:  # even indices are fences, odd are content
            part = part.strip()
            if not part:
                continue
            if part.startswith("json"):
                part = part[4:].strip()
            try:
                return json.loads(part)
            except (json.JSONDecodeError, ValueError):
                continue

    # Strategy 2: first {...} block (handles "Here's the result: {...}" etc.)
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except (json.JSONDecodeError, ValueError):
            pass

    # Strategy 3: treat whole text as JSON
    try:
        return json.loads(text)
    except (json.JSONDecodeError, ValueError):
        pass

    return None


# ============================================================
# Role prompt
# ============================================================
DSRNA_DESIGNER_ROLE = """
Role: dsRNA Design Expert

You are a professional dsRNA design specialist for RNAi pesticide development.
Your mission is to design dsRNA fragments and PCR primers by integrating
bioinformatics tool outputs into a coherent, evidence-driven fragment selection strategy.

Workflow constraints:
1. Sequence Validation: normalized by deterministic rules (uppercase, U->T, whitespace stripped).
2. siRNA Scanning: OligoWalk thermodynamic scores are generated by tools and stored in state.
3. Fragment Design: you perform sliding-window analysis across the full sequence.
4. Primer Design: primer3 results are generated by tools and stored in state.
5. PIS Scoring: you receive the three evaluation dimensions and design a scoring formula suited
   to the current sequence context.
6. Final Report: all structured data is provided — generate a clear, actionable report.
""".strip()


# ============================================================
# PrimerDesignSubgraph
# ============================================================
def build_primer_design_subgraph(
    primer_tool: BaseTool = primer3_tool,
    debug_output: bool = False,
):
    def _inc_retry(state: PrimerDesignState, node_name: str) -> tuple[dict, int]:
        retry_counts = dict(state.get("retry_counts", {}) or {})
        attempt = int(retry_counts.get(node_name, 0)) + 1
        retry_counts[node_name] = attempt
        return retry_counts, attempt

    def primer3_design_node(state: PrimerDesignState) -> dict:
        fragment = state.get("fragment", {})
        retry_counts, attempt = _inc_retry(state, "primer3_design")
        payload = {
            "sequence": fragment.get("sequence", ""),
            "sequence_id": fragment.get("fragment_id", "fragment"),
            "num_return": 5,
            "primer_opt_size": 20,
            "primer_opt_tm": 60.0,
            "primer_min_tm": 57.0,
            "primer_max_tm": 63.0,
            "primer_product_size_range": [100, 300],
        }
        try:
            result = _json_loads_tool_result(primer_tool.invoke(payload))
        except Exception as exc:
            result = {"status": "error", "error": "tool invocation failed", "details": str(exc)}
        updates: dict = {"primer3_result": result, "retry_counts": retry_counts}
        if result.get("status") != "success":
            updates["errors"] = [
                _tool_error("primer3_design", getattr(primer_tool, "name", "primer3"),
                            result.get("error") or result.get("details") or "unknown error", attempt)
            ]
        return updates

    def route_after_primer3(state: PrimerDesignState) -> Literal["primer_validate", "primer3_design"]:
        if (state.get("primer3_result") or {}).get("status") == "success":
            return "primer_validate"
        if (state.get("retry_counts") or {}).get("primer3_design", 0) < MAX_TOOL_RETRIES:
            return "primer3_design"
        return "primer_validate"

    def primer_validate_node(state: PrimerDesignState) -> dict:
        fragment = state.get("fragment", {})
        primer3_result = state.get("primer3_result") or {}
        fragment_id = fragment.get("fragment_id", "fragment")
        errors = state.get("errors", []) or []

        primer_results = []
        for p in (primer3_result.get("primers", []) or []):
            primer_results.append({
                "fragment_id": fragment_id,
                "pair_index": p.get("pair_index"),
                "forward": p.get("forward"),
                "reverse": p.get("reverse"),
                "forward_with_T7": p.get("forward_with_T7"),
                "reverse_with_T7": p.get("reverse_with_T7"),
                "tm_left": p.get("tm_left"),
                "tm_right": p.get("tm_right"),
                "product_size": p.get("product_size"),
                "penalty": p.get("penalty"),
            })

        warnings = []
        if len(primer_results) < MIN_PRIMER_PAIRS:
            warnings.append(f"Only {len(primer_results)} primer pairs returned (minimum {MIN_PRIMER_PAIRS} recommended)")

        best_penalty = None
        if primer_results:
            best_penalty = min(p["penalty"] for p in primer_results)

        fragment_result = {
            "fragment_id": fragment_id,
            "fragment_start": fragment.get("start"),
            "fragment_end": fragment.get("end"),
            "fragment_sequence": fragment.get("sequence"),
            "fragment_length": fragment.get("length"),
            "covered_sirna": fragment.get("covered_sirna", []),
            "primer_count": len(primer_results),
            "best_penalty": best_penalty,
            "primer_results": primer_results,
            "warnings": warnings,
            "status": "success" if primer_results else "failed",
            "errors": errors,
        }
        if debug_output:
            fragment_result["primer3_raw"] = primer3_result

        return {"primer_results": primer_results, "fragment_result": fragment_result,
                "fragment_results": [fragment_result]}

    builder = StateGraph(PrimerDesignState)
    builder.add_node("primer3_design", primer3_design_node)
    builder.add_node("primer_validate", primer_validate_node)
    builder.add_edge(START, "primer3_design")
    builder.add_conditional_edges("primer3_design", route_after_primer3, {
        "primer_validate": "primer_validate",
        "primer3_design": "primer3_design",
    })
    builder.add_edge("primer_validate", END)
    return builder.compile()


# ============================================================
# DSRNADesignerGraph
# ============================================================
def build_dsrna_designer_graph(
    llm: BaseChatModel | None = None,
    checkpointer=None,
    interrupt_before: list[str] | None = None,
    oligowalk_tool_instance: BaseTool | None = None,
    primer3_tool_instance: BaseTool | None = None,
    primer_subgraph=None,
    debug_output: bool = False,
):
    if llm is None:
        from llm_config import get_default_llm
        llm = get_default_llm()

    _oligowalk = oligowalk_tool_instance or oligowalk_tool
    _primer3 = primer3_tool_instance or primer3_tool
    _primer_subgraph = (
        primer_subgraph
        or build_primer_design_subgraph(primer_tool=_primer3, debug_output=debug_output)
    )

    # ----- sequence_validate: normalize + QC -----
    def sequence_validate_node(state: DSRNADesignState) -> dict:
        raw = str(state.get("sequence_input", "")).strip()
        lang = _detect_language(raw)
        try:
            cleaned = _normalize_sequence(raw)
            warnings = _check_low_complexity(cleaned)
            qc = {
                "length": len(cleaned),
                "gc_percent": _gc_percent(cleaned),
                "warnings": warnings,
            }
            return {"sequence_clean": cleaned, "sequence_valid": True,
                    "sequence_qc": qc, "language": lang}
        except ValueError as exc:
            return {
                "sequence_clean": "",
                "sequence_valid": False,
                "sequence_qc": {"length": 0, "gc_percent": 0.0, "warnings": [str(exc)]},
                "language": lang,
                "errors": [{"stage": "sequence_validate", "error": str(exc)}],
            }

    def route_after_validate(state: DSRNADesignState) -> Literal["oligowalk_scan", "report_generate"]:
        return "oligowalk_scan" if state.get("sequence_valid") else "report_generate"

    # ----- oligowalk_scan: tool call -----
    def oligowalk_scan_node(state: DSRNADesignState) -> dict:
        retry_counts = dict(state.get("retry_counts", {}) or {})
        attempt = int(retry_counts.get("oligowalk_scan", 0)) + 1
        retry_counts["oligowalk_scan"] = attempt
        payload = {
            "sequence": state["sequence_clean"],
            "run_type": state.get("oligowalk_mode", "fast"),
            "oligo_length": 21,
            "top_n": 30,
        }
        try:
            result = _json_loads_tool_result(_oligowalk.invoke(payload))
        except Exception as exc:
            result = {"status": "error", "error": "tool invocation failed", "details": str(exc)}
        updates: dict = {"oligowalk_result": result, "retry_counts": retry_counts}
        if result.get("status") != "success":
            updates["errors"] = [
                _tool_error("oligowalk_scan", getattr(_oligowalk, "name", "oligowalk"),
                            result.get("error") or result.get("details") or "unknown error", attempt)
            ]
        return updates

    def route_after_oligowalk(state: DSRNADesignState) -> Literal["fragment_design", "oligowalk_scan", "report_generate"]:
        result = state.get("oligowalk_result") or {}
        if result.get("status") == "success":
            return "fragment_design"
        if (state.get("retry_counts") or {}).get("oligowalk_scan", 0) < MAX_TOOL_RETRIES:
            return "oligowalk_scan"
        return "report_generate"

    # ----- fragment_design: LLM sliding-window analysis -----
    def fragment_design_node(state: DSRNADesignState) -> dict:
        lang = state.get("language", "English")
        lang_instr = "Write all reasoning and output in Chinese." if lang == "Chinese" else "Write all reasoning and output in English."

        oligowalk_result = state.get("oligowalk_result") or {}
        seq = state.get("sequence_clean", "")
        candidates = _parse_oligowalk_candidates(oligowalk_result, input_seq=seq)
        qc = state.get("sequence_qc", {})

        candidates_json = json.dumps(candidates, ensure_ascii=False, indent=2)
        protocol = build_skills("principles", "evidence", "tool")

        prompt = (
            f"{DSRNA_DESIGNER_ROLE}\n\n"
            f"{protocol}\n\n"
            f"{lang_instr}\n\n"
            "You are tasked with designing optimal dsRNA fragments using a sliding-window strategy.\n\n"
            "Input data:\n"
            f"  Target sequence ({qc.get('length', len(seq))} nt, GC={qc.get('gc_percent', 0)}%):\n"
            f"  {seq}\n\n"
            f"  siRNA candidates from OligoWalk ({len(candidates)} total):\n{candidates_json}\n\n"
            "Constraints:\n"
            f"  - Fragment size: ideally 300-500 bp. If the target sequence is shorter than 300 bp, "
            f"adapt the window size to fit the available sequence (e.g., 100-200 bp for a 241 nt sequence).\n"
            "  - Target amplicon size for primer3: 100-300 bp\n"
            "  - Each fragment should cover as many high-scoring siRNA sites as possible\n"
            "  - Avoid fragments with extreme GC% (<30% or >65%)\n"
            "  - Prefer fragments where the 5' and 3' ends have reasonable GC content for primer binding\n\n"
            "Your task:\n"
            "  1. Identify siRNA hotspot regions — scan the target sequence using a sliding window "
            f"(step ~10-20 bp, window size 300-500 bp if sequence >= 300 nt; "
            f"otherwise use the full sequence length). "
            "Score each window by the sum of OligoWalk |Overall| scores of siRNAs inside it.\n"
            "  2. Rank windows by composite score: (a) siRNA coverage density (dominant); "
            "(b) GC% of the window; (c) 5'/3' end GC% for primer feasibility.\n"
            "  3. Design 1-3 dsRNA fragments — ONLY around high-scoring siRNA hotspot regions. "
            "It is perfectly acceptable to propose a single fragment if there is only one clear hotspot. "
            "Do NOT design fragments in regions with zero or negligible siRNA coverage — "
            "a fragment with no siRNA hits has PIS=0 and serves no purpose.\n"
            "     Explain your scoring logic — you may weight these factors as you see fit for this sequence.\n"
            "  4. If the sequence is short (e.g., <600 bp), a single fragment covering the entire "
            "sequence is the preferred approach. Multiple fragments are justified only when "
            "there are multiple distinct, well-separated hotspot regions.\n\n"
            "Return a JSON object with key 'fragment_proposals' — a list of candidates, each containing:\n"
            "  fragment_id, start (1-based), end (1-based), sequence, length,\n"
            "  covered_sirna (list of {position_1based, sequence, overall_score}),\n"
            "  composite_score, gc_percent, primer_feasibility_note, evidence_based_rationale.\n"
            "Return ONLY valid JSON, no explanation outside the JSON."
        )

        try:
            response = llm.invoke(prompt)
            raw = getattr(response, "content", str(response)).strip()
            parsed = _extract_json_from_llm_response(raw)
            if parsed is None:
                logger.warning("fragment_design: could not parse JSON from LLM response: %s", raw)
                proposals = []
            else:
                proposals = parsed.get("fragment_proposals", [])
        except Exception as exc:
            logger.exception("fragment_design LLM call failed")
            proposals = []

        return {"fragment_proposals": proposals}

    # ----- parallel_primer_design: Send dispatcher -----
    def parallel_primer_design(state: DSRNADesignState):
        proposals = state.get("fragment_proposals", []) or []
        if not proposals:
            return "pis_score"
        return [
            Send("primer_design_subgraph", {
                "fragment": p,
                "primer3_result": {},
                "primer_results": [],
                "fragment_result": {},
                "fragment_results": [],
                "retry_counts": {},
                "errors": [],
            })
            for p in proposals
        ]

    # ----- pis_score: LLM flexible scoring -----
    def pis_score_node(state: DSRNADesignState) -> dict:
        lang = state.get("language", "English")
        lang_instr = "Write all reasoning and output in Chinese." if lang == "Chinese" else "Write all reasoning and output in English."

        fragment_results = state.get("fragment_results", []) or []
        oligowalk_result = state.get("oligowalk_result") or {}
        seq = state.get("sequence_clean", "")
        candidates = _parse_oligowalk_candidates(oligowalk_result, input_seq=seq)

        # Build per-fragment siRNA coverage data
        fragment_sirna_map = {}
        for fr in fragment_results:
            frag_id = fr.get("fragment_id", "?")
            covered = fr.get("covered_sirna", []) or []
            frag_start = fr.get("fragment_start", 0)
            frag_end = fr.get("fragment_end", 0)
            # Filter candidates that fall within this fragment's range
            inside = [c for c in candidates if frag_start <= c["position_1based"] <= frag_end]
            fragment_sirna_map[frag_id] = {
                "fragment_id": frag_id,
                "fragment_start": frag_start,
                "fragment_end": frag_end,
                "fragment_length": fr.get("fragment_length", 0),
                "covered_sirna_count": len(inside),
                "covered_sirna_details": inside,
                "sirna_total_score": round(sum(abs(c["overall_score"]) for c in inside), 2),
                "best_penalty": fr.get("best_penalty"),
                "primer_count": fr.get("primer_count", 0),
                "gc_percent": _gc_percent(fr.get("fragment_sequence", "")),
            }

        protocol = build_skills("principles", "evidence")

        prompt = (
            f"{DSRNA_DESIGNER_ROLE}\n\n"
            f"{protocol}\n\n"
            f"{lang_instr}\n\n"
            "You are tasked with computing the Predicted Interference Score (PIS) for each dsRNA fragment.\n\n"
            "Three evaluation dimensions (you will receive actual data for each fragment):\n"
            "  A. siRNA Binding Score (suggested weight 50-70%): number and combined OligoWalk |ddG| values "
            "of siRNA sites within the fragment window.\n"
            "  B. Primer Quality Score (suggested weight 20-35%): based on primer3 best penalty and "
            "Tm deviation from the optimal 60°C.\n"
            "  C. Sequence Context Quality (suggested weight 5-20%): GC% deviation from ideal 50%, "
            "presence of low-complexity regions.\n\n"
            "Reference weights: 0.6A + 0.3B + 0.1C.\n"
            "You MAY adjust these weights based on the actual data quality you observe — "
            "if one dimension is clearly dominant or unreliable, explain your adjustment.\n\n"
            "Tier thresholds (adjustable with justification):\n"
            "  High   (>85): strong siRNA coverage + good primer quality + ideal GC\n"
            "  Medium (60-85): moderate coverage or some quality concerns\n"
            "  Low    (<60): poor coverage or significant design issues\n\n"
            "IMPORTANT — PIS is a composite QUALITY score, NOT a prediction of in vivo knockdown.\n"
            "It combines available computational metrics (Level 1) with heuristic weighting (Level 2).\n"
            "Do NOT present PIS as 'predicted interference efficiency' or 'knockdown percentage'.\n"
            "Do NOT use tier labels to imply experimental outcomes.\n\n"
            "Data for scoring:\n"
            f"{json.dumps(list(fragment_sirna_map.values()), ensure_ascii=False, indent=2)}\n\n"
            "Your task:\n"
            "  1. For each fragment, compute A, B, C sub-scores using formulas appropriate for this data.\n"
            "  2. Apply your chosen weights to get total PIS per fragment.\n"
            "  3. Explain any weight adjustments you made from the reference and why.\n"
            "  4. Assign tier labels.\n"
            "  5. Recommend the best fragment if multiple are provided.\n\n"
            "Return a JSON object with keys:\n"
            "  'pis_scores': list of {{fragment_id, score_A, score_B, score_C, pis_total, tier, weight_rationale}}\n"
            "  'best_fragment_id': fragment_id of the recommended fragment\n"
            "  'weight_explanation': your reasoning for any weight adjustments\n"
            "Return ONLY valid JSON, no explanation outside the JSON."
        )

        try:
            response = llm.invoke(prompt)
            raw = getattr(response, "content", str(response)).strip()
            parsed = _extract_json_from_llm_response(raw)
            if parsed is None:
                logger.warning("pis_score: could not parse JSON from LLM response: %s", raw)
                pis_scores = []
            else:
                pis_scores = parsed.get("pis_scores", [])
        except Exception as exc:
            logger.exception("pis_score LLM call failed")
            pis_scores = []

        return {"pis_scores": pis_scores}

    # ----- report_generate: LLM final report -----
    def report_generate_node(state: DSRNADesignState) -> dict:
        lang = state.get("language", "English")
        lang_instr = "Write the entire report in Chinese." if lang == "Chinese" else "Write the entire report in English."

        errors = state.get("errors", []) or []

        # Sequence QC summary
        qc = state.get("sequence_qc", {})
        oligowalk_result = state.get("oligowalk_result") or {}
        seq = state.get("sequence_clean", "")
        candidates = _parse_oligowalk_candidates(oligowalk_result, input_seq=seq)

        # If validation failed, generate error report
        if not state.get("sequence_valid"):
            prompt = (
                f"{DSRNA_DESIGNER_ROLE}\n\n"
                f"{build_skills('principles', 'tool')}\n\n"
                f"{lang_instr}\n\n"
                "Generate a dsRNA design error report.\n"
                f"Error: {errors}\n\n"
                "Output requirements:\n"
                "1. Identified the error (invalid sequence characters, empty input, etc.)\n"
                "2. Suggest how to fix it and re-submit\n"
            )
            try:
                response = llm.invoke(prompt)
                report_text = getattr(response, "content", str(response))
            except Exception:
                report_text = f"[Report generation failed: {errors}]"
            return {"final_report": {"status": "failed", "errors": errors, "report_text": report_text}}

        # Normal report
        fragment_results = state.get("fragment_results", []) or []
        pis_scores = state.get("pis_scores", []) or []

        # Enrich compact fragment summaries with pis data for LLM
        pis_map = {p.get("fragment_id"): p for p in pis_scores}
        compact_fragments = []
        for fr in fragment_results:
            frag_id = fr.get("fragment_id", "?")
            compact = _compact_fragment_result(fr)
            if frag_id in pis_map:
                compact["pis_data"] = pis_map[frag_id]
            compact_fragments.append(compact)

        protocol = build_skills("principles", "evidence", "tool", "recommendation")
        report_payload = {
            "sequence_qc": qc,
            "sirna_candidates_scanned": len(candidates),
            "fragments_designed": len(fragment_results),
            "top_sirna_candidates": candidates[:30],
            "fragments": compact_fragments,
            "pis_scores": pis_scores,
        }

        prompt = (
            f"{DSRNA_DESIGNER_ROLE}\n\n"
            f"{protocol}\n\n"
            f"{lang_instr}\n\n"
            "Generate the complete dsRNA Design Report from the structured data below.\n"
            "Output Requirements — your report MUST include all of the following sections:\n"
            "1. Sequence Quality Control: length, GC%, complexity warnings\n"
            "2. siRNA Candidate Summary: top 5 ranked by OligoWalk Overall score — include ALL available fields "
            "(sequence, position, overall_score/ddG, gc_percent). Do NOT omit any field.\n"
            "3. dsRNA Fragment Design: for each fragment — list all available fields from the payload "
            "(fragment_id, start, end, length, gc_percent, covered siRNA sites, composite_score, "
            "evidence_based_rationale). Do NOT omit any field.\n"
            "4. Primer Results: for each fragment — table of primer pairs with all fields "
            "(pair_index, forward, reverse, forward_with_T7, reverse_with_T7, tm_left, tm_right, "
            "product_size, penalty). Do NOT omit any field.\n"
            "5. PIS Scoring Summary: list all fields from pis_scores entries (fragment_id, score_A, score_B, "
            "score_C, pis_total, tier, weight_rationale). Do NOT omit any field.\n"
            "6. Conclusions and Recommendations: which dsRNA-fragment + primer pair to prioritize for synthesis\n\n"
            "Global rule: every section must render ALL fields present in the payload — never summarize away data.\n\n"
            "Structured Facts:\n"
            f"{json.dumps(report_payload, ensure_ascii=False, indent=2)}"
        )

        try:
            response = llm.invoke(prompt)
            report_text = getattr(response, "content", str(response))
        except Exception as exc:
            logger.exception("LLM report generation failed")
            report_text = f"[Report generation failed: {exc}]"

        return {
            "final_report": {
                "status": "success",
                "sequence_qc": qc,
                "sirna_candidates": candidates,
                "fragment_results": fragment_results,
                "pis_scores": pis_scores,
                "errors": errors,
                "report_text": report_text,
                "computed_by": {
                    "sequence_qc": "rule_engine",
                    "sirna_scan": f"oligowalk({state.get('oligowalk_mode', 'fast')})",
                    "primer_design": "primer3",
                    "fragment_design": "llm",
                    "pis_scoring": "llm",
                    "report_generation": "llm",
                },
            }
        }

    # ----- graph assembly -----
    builder = StateGraph(DSRNADesignState)
    builder.add_node("sequence_validate", sequence_validate_node)
    builder.add_node("oligowalk_scan", oligowalk_scan_node)
    builder.add_node("fragment_design", fragment_design_node)
    builder.add_node("primer_design_subgraph", _primer_subgraph)
    builder.add_node("pis_score", pis_score_node)
    builder.add_node("report_generate", report_generate_node)

    builder.add_edge(START, "sequence_validate")
    builder.add_conditional_edges("sequence_validate", route_after_validate, {
        "oligowalk_scan": "oligowalk_scan",
        "report_generate": "report_generate",
    })
    builder.add_conditional_edges("oligowalk_scan", route_after_oligowalk, {
        "fragment_design": "fragment_design",
        "oligowalk_scan": "oligowalk_scan",
        "report_generate": "report_generate",
    })
    builder.add_conditional_edges(
        "fragment_design",
        parallel_primer_design,
        {"primer_design_subgraph": "primer_design_subgraph", "pis_score": "pis_score"},
    )
    builder.add_edge("primer_design_subgraph", "pis_score")
    builder.add_edge("pis_score", "report_generate")
    builder.add_edge("report_generate", END)

    effective_checkpointer = checkpointer if checkpointer is not None else MemorySaver()
    return builder.compile(
        checkpointer=effective_checkpointer,
        interrupt_before=interrupt_before,
    )


# ============================================================
# Singleton and CLI smoke test
# ============================================================
from llm_config import get_default_llm
dsrna_designer_graph = build_dsrna_designer_graph(llm=get_default_llm())


if __name__ == "__main__":
    import uuid

    # Uncomment to suppress INFO logs:
    # for name in ("RPA_Agent", "RPA_Tools", "langchain_openai", "httpx"):
    #     logging.getLogger(name).setLevel(logging.WARNING)
    # logging.getLogger("langchain_openai").setLevel(logging.ERROR)

    test_seq = (
        "ATGGCTGCGACCTCTAAGGGTGTGCAAGGCGACACTGATGAGTCTGATTCCGAATACACCCCACTctatgacgatgatgatgtgGACGAAAGAACAGCACAGGAAACGAAAGGATGGAATTTGTTCCGAGAGATTCCTGTCAAGAAGGAGACTGGGTCTATGGCATCGGTACAATGGGTAGAATCGAGTGTAGTGATCCTGAAACTCTTAGCATATATCATCGTGTTCACTTTGGTGTTAGGAGCAGCCGTTATAGCAAAAGGCACTTTGCTTTTCATCACCTCACAGTTGAAAAAAGGTCGACAGATTACGCATTGCAACAGAGGATTAGGTTTGGATAAACAATACATAACACAGTTGACGCTCGACGAACGCGTGACATGGCTCTGGGCAGCTGTGATAGTGTATGGTGCTCCGGAACTTGGAGTCTTTCTAAGATCTGTTAGAATATGTTTCTTCAAAACCGCTAGGAAACCCACCGGCTTTCAGTTCTTATTCGCATTCTTTATAGAGACACTACAAGCGTTCGGAGTCGGGATTCTTGTCCTGGTAATCCTGCCAGAACTAGACGTGGTCAAGGGTGCTATGCTGATGAACGCCATGTGTATCGTCCCGGGTATTCTGGCAATGTTCACTAGAGATTTCACTGACTCCAGATACCACACGAAGATCCTGCTGGATGTGCTGGCGATATCTGCGCAGGCGACGGCGTTTGTTGCCTGGCCTCTGTACGATGGAACGGCAAAATTGTGGTGGATACCTTTTTCGTGTATATTTGTCTCCGTCGGCTGGTGGGAGAACTTCGTCAGTTTGGTTGAGAAGAATCCATCATCATTTGTCCTGTTCCTATTAGAAATCCGCGATGGCTTGCGGAAGACCCGCTACTTTACCATGAGAGCTCTGTCCCTCTGGAAGATAGTGGTCTTCGTGCTCTGTGCCATGATATCCCTTCATATGCAGAATGACTCAGCAGTTGCTTTCTTCACCCATGTTGCAGGAGCTTTTTCTGACAGAAACTACACTGTTTATGAGGTTCAAGTGTATATGCAAGATGCGTACGACGGCGTGTTGGCTTATACTGTAACTGGGGATATTATAGACAGCCTGCCGGCGTATTGGCCAGCCGCCCTTTGGGTCGCGCTGATTTCAGTCTGTGCAGCGTACATCTGCTTCGCCTGTGCTAAGTTTGCCTGCAAGATCCTCATACAAAACTTCAGTTTTACTTTCGCTCTAAGTCTAGTTGGACCGGTCACTATAAATATCTTGATAATACTATGCGGGATTAAAAATTCCAATCCGTGCGCTTTCAGGAGTATACCCAACTATTTGTTCTTCGACATTCCTCCAGTATACTACATTTGGCAATACGTTGGCCGTGAGATGGCGTGGGTGTGGCTGCTCTGGCTGTTGTCTCAAGGGTGGATCACGTACCACACGTGGCAGCCGCGTTGCGAGCGCCTAGCCGCTACAGAGAAGCTGTTTTCCAAGCCCTGGTACTGCGGACCACTGCTTGATCAAAGCCTGCTCTTGAACAGGACTAAGGACGACGATCATGATGTTACTCTtgagGATCTCAAAGACCTCGAAGATGACGCATCGATCAGCAGTGCCGAAAAGATGACGAACGTTAAGCCGAGTGATAGTATAACAagGATCTATATCTGCGCGACGATGTGGCACGAGACAAAAGACGAAATGATCGAGTTCTTAAAGTCGATCTTCCGGCTGGACGAGGATCAGAGCGCAAGGAGGGTTGCACAGAAATACCTCGGGGTTGTGGATCCTGACTACTACGAAATGGAGATTCACATTTTCATGGACGACGCCTTCGAAATATCGGATCACAGTTCGGAAGACTCGCAAGTCAACCGTTTCGTCAAATGTCTAGTGGACACAGTAGACGAGGCAGCATCTGAAGTGCACCTTACCAACGTCAAGCTACGTCCCCCCAAGAAGTTCCCGACGCCTTACGGAGGGAAACTGCAGTGGACTTTGCCGGGGAAAAACAAAATGATATGCCATTTGAAGGACAAGGCCAAGATAAGACATAGGAAGCGGTGGTCGCAGgtgATGTACATGTACTACTTCTTGGGTCATCGTctcatggacctccccctactGGTGGACCGTAAGGAAACTATAGCTGAGAACACTTACCTATTGGCTTTGGACGGGGACATTGATTTCAAGCCGCAAGCCGTGACGCTGCTCATTGACCTCatgaagaagaataagaatttAGGCGCGGCCTGTGGACG"
    )
    result = dsrna_designer_graph.invoke(
        {"sequence_input": test_seq},
        config={"configurable": {"thread_id": f"dsrna-{uuid.uuid4().hex[:8]}"}},
    )
    fr = result.get("final_report", {})
    print(f"Status: {fr.get('status', 'unknown')}")
    print(f"Fragments: {len(fr.get('fragment_results', []))}")
    print(f"Errors: {len(fr.get('errors', []))}")
    if fr.get("report_text"):
        print(fr['report_text'])