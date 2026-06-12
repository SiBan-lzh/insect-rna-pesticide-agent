"""
SafetyInspectorGraph — LangGraph ecological safety assessment workflow.

Tools produce facts; LangGraph controls routing, retries, fan-out/fan-in, and state.
LLM is only used for field-type classification and final report explanation.
All risk levels are computed by a deterministic rule engine.

Graph:
    START -> input_clean -> candidate_species_build
          -> [Send(species_analysis_subgraph) x N] -> risk_aggregate
          -> report_generate -> END

Subgraph:
    nto_blast -> risk_hit_evaluate -> sequence_fetch
              -> clustal -> risk_score -> END
"""

from __future__ import annotations

import json
import logging
import operator
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

from tool_config import NTO_BLAST_DB, NTOS_REFSEQ_DB
from skill.skill_loader import build_skills
from tools import clustal_tool, fetch_nto_seq_tool, nto_blast_tool

logger = logging.getLogger("RPA_Agent.SafetyInspector")

NTO_LIST_DIR = _PROJECT_ROOT / "database" / "NTOs_lists"
MAX_TOOL_RETRIES = 3


# ============================================================
# State definitions
# ============================================================
class SafetyState(TypedDict, total=False):
    """Main ecological safety assessment graph state."""

    field_type: str
    dsrna_sequence: Annotated[str, lambda a, b: a if a else b]
    field_types: list[str]
    candidate_species: list[dict]
    species_results: Annotated[list[dict], operator.add]
    high_risk_species: list[dict]
    medium_risk_species: list[dict]
    low_risk_species: list[dict]
    species_count: int
    high_count: int
    medium_count: int
    low_count: int
    error_count: int
    final_report: dict
    errors: Annotated[list[dict], operator.add]


class NTOAnalysisState(TypedDict, total=False):
    """Reusable single-NTO off-target analysis subgraph state."""

    species: dict
    dsrna_sequence: str
    blast_result: dict
    blast_summary: dict
    risk_hits: list[dict]
    risk_sequences: list[dict]
    alignment_results: list[dict]
    species_result: dict
    species_results: Annotated[list[dict], operator.add]
    errors: Annotated[list[dict], operator.add]
    retry_counts: dict


# ============================================================
# Rule engine
# ============================================================
def normalize_dsrna_sequence(seq: str) -> str:
    """Normalize a dsRNA/DNA sequence for downstream tools."""
    cleaned = "".join(str(seq or "").upper().split()).replace("U", "T")
    invalid = sorted(set(cleaned) - {"A", "T", "C", "G", "N"})
    if not cleaned:
        raise ValueError("dsRNA sequence is empty")
    if invalid:
        raise ValueError(f"Invalid nucleotide characters: {invalid}")
    return cleaned


def select_fetchable_hits(blast_result: dict) -> list[dict]:
    """Select BLAST hits that warrant sequence retrieval.

    Coverage is intentionally NOT used as a filter — a short alignment can
    still contain a dangerous >=21nt perfect match region that would be
    missed if coverage were a gate. Only identity_pct is used as a minimum
    quality gate to avoid fetching obviously non-homologous noise.
    """
    selected = []
    for hit in blast_result.get("results", []) or []:
        identity_pct = float(hit.get("identity_pct") or 0)
        if identity_pct > 80:
            selected.append({
                **hit,
                "filter_rule": "identity_pct > 80 (coverage intentionally excluded: irrelevant for 21nt match detection)",
            })
    return selected


def get_longest_match(alignment_results: list[dict]) -> int:
    """Return the maximum continuous perfect match across alignment results."""
    longest = 0
    for result in alignment_results or []:
        analysis = result.get("analysis", {}) or {}
        longest = max(longest, int(analysis.get("max_continuous_match") or 0))
    return longest


def score_species_risk(
    longest_match: int,
    has_fetchable_hits: bool,
    has_tool_error: bool = False,
) -> str:
    """Assign species risk using deterministic rules only."""
    if has_tool_error:
        return "error"
    if not has_fetchable_hits:
        return "negligible"
    if longest_match >= 21:
        return "high"
    if longest_match >= 19:
        return "medium"
    return "low"


def derive_overall_risk_level(species_results: list[dict]) -> str:
    """Derive overall risk level from species-level rule-engine results."""
    levels = [r.get("risk_level") for r in species_results or []]
    if "high" in levels:
        return "high"
    if "medium" in levels:
        return "medium"
    if "low" in levels:
        return "low"
    if "error" in levels:
        return "incomplete"
    return "negligible"


def group_species_by_risk(species_results: list[dict]) -> dict[str, list[dict]]:
    """Group species results into high/medium/low output buckets."""
    return {
        "high_risk_species": [r for r in species_results or [] if r.get("risk_level") == "high"],
        "medium_risk_species": [r for r in species_results or [] if r.get("risk_level") == "medium"],
        "low_risk_species": [
            r for r in species_results or []
            if r.get("risk_level") in {"low", "negligible"}
        ],
    }


def _species_name(species: dict | str) -> str:
    if isinstance(species, str):
        return species
    return (
        species.get("scientific_name")
        or species.get("species")
        or species.get("name")
        or "unknown_species"
    )


def _json_loads_tool_result(raw: Any) -> dict:
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        return json.loads(raw)
    return {"status": "error", "error": "Unexpected tool result type", "details": str(type(raw))}


def _tool_error(node: str, tool_name: str, species: str, error: str, attempt: int) -> dict:
    return {
        "node": node,
        "tool": tool_name,
        "species": species,
        "attempt": attempt,
        "error": error,
    }


# ============================================================
# NTO list retrieval (JSON-based)
# ============================================================
def read_nto_json(path: Path) -> list[dict]:
    """Read NTO species records from a JSON file."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return [
        {**org, "source_file": path.name}
        for org in data.get("organisms", [])
    ]


def has_local_nto_database(scientific_name: str) -> bool:
    """Return whether both local BLAST and refseq resources exist for a species."""
    blast_path = Path(NTO_BLAST_DB) / scientific_name / f"{scientific_name}.nsq"
    refseq_dir = Path(NTOS_REFSEQ_DB) / scientific_name
    return blast_path.exists() and refseq_dir.exists() and any(refseq_dir.glob("*.fa"))


def retrieve_nto_candidates(nto_files: list[str]) -> tuple[list[dict], list[dict]]:
    """Load NTO species from a list of JSON filenames (EPA always included by caller)."""
    warnings = []
    candidates = []

    for file_name in nto_files:
        path = NTO_LIST_DIR / file_name
        if not path.exists():
            warnings.append({
                "stage": "nto_retriever",
                "source_file": file_name,
                "warning": "NTO list file not found",
            })
            continue
        try:
            records = read_nto_json(path)
        except Exception as exc:
            warnings.append({
                "stage": "nto_retriever",
                "source_file": file_name,
                "warning": f"Failed to parse JSON: {exc}",
            })
            continue
        candidates.extend(records)

    deduped = []
    seen = {}
    for item in candidates:
        name = item.get("scientific_name")
        if not name:
            continue
        if name in seen:
            seen[name].setdefault("duplicate_sources", []).append(item.get("source_file"))
            continue
        copied = dict(item)
        copied["local_database_available"] = has_local_nto_database(name)
        copied["source_files"] = [copied.get("source_file")]
        seen[name] = copied
        deduped.append(copied)

    return deduped, warnings


# ============================================================
# Safety inspector role prompt
# ============================================================
SAFETY_INSPECTOR_ROLE = """
Role: Ecological Safety Inspector

You are a professional ecological safety risk assessment expert specializing in RNAi technology.
Your mission is to evaluate off-target risks of candidate dsRNA sequences in non-target organisms (NTOs)
by integrating large-scale bioinformatics data — BLAST homology searches, transcript/genomic sequence
retrieval, and pairwise alignment analyses — into rigorous, actionable ecological safety assessments.
This data-driven approach enables evidence-based risk evaluation at a scale that manual literature review
cannot achieve, forming the core advantage of AI-driven RNAi pesticide development.

Incentive: Exceptional, scientifically grounded risk assessments with clear evidence chains and
well-structured reports will be rewarded with $10,000.

Workflow constraints:
1. Field/Habitat Classification: interpret the user's field/habitat description and classify it into
   one or more types: rice_field, corn_field, wheat_field, tomatofield. Always append EPA.
2. Homology Search: BLAST facts are generated by tools and already stored in state.
3. Sequence Retrieval: homologous transcript/genomic sequences are generated by tools and stored in state.
4. Pairwise Alignment and Off-Target Detection: Clustal alignments are generated by tools and stored in state.
5. Risk Criterion: continuous perfect-match regions of >=21 nucleotides are the critical high-risk threshold.
6. Risk levels are computed by a deterministic rule engine. Do NOT change them.

Output Requirements — your final report MUST include all of the following sections in order:
1. Identified Species List: All NTO species analyzed, grouped by field type, with scientific name and category.
2. BLAST Results Summary: For each species, a table of subject sequence IDs, alignment scores, identity
   percentages, and genomic coordinates.
3. Sequence Retrieval Status: Which homologous sequences were successfully retrieved and which failed.
4. Pairwise Alignment and Off-Target Detection: For species with alignment data, present a visual
   alignment using | (pipe) symbols to mark continuous perfect-match regions. Highlight regions >=21nt
   as critical off-target risks. Include match coordinates and orientation for each alignment.
5. Final Risk Conclusion: A summary table with species name, assigned risk level, longest continuous match
   length, and rule-based justification (>=21nt = high; >=19nt = medium; <19nt = low; no BLAST hits =
   negligible). If the overall assessment is "high" or "medium", provide specific mitigation recommendations.
""".strip()


# ============================================================
# NTOAnalysisSubgraph
# ============================================================
def build_species_analysis_subgraph(
    blast_tool: BaseTool = nto_blast_tool,
    fetch_tool: BaseTool = fetch_nto_seq_tool,
    align_tool: BaseTool = clustal_tool,
    debug_output: bool = False,
):
    """Build reusable single-NTO off-target analysis subgraph.

    Flow (tool calls in brackets):
        nto_blast [nto_blast] -> risk_hit_evaluate -> sequence_fetch [fetch_nto_seq]
        -> clustal [clustal x N] -> risk_score -> END
    """
    def _increment_retry(state: NTOAnalysisState, node_name: str) -> tuple[dict, int]:
        retry_counts = dict(state.get("retry_counts", {}) or {})
        attempt = int(retry_counts.get(node_name, 0)) + 1
        retry_counts[node_name] = attempt
        return retry_counts, attempt

    # ----- nto_blast: homology search against one NTO genome -----
    def nto_blast_node(state: NTOAnalysisState) -> dict:
        species = state.get("species", {})
        species_name = _species_name(species)
        retry_counts, attempt = _increment_retry(state, "nto_blast")
        payload = {"sequence": state["dsrna_sequence"], "species": species_name}
        try:
            result = _json_loads_tool_result(blast_tool.invoke(payload))
        except Exception as exc:
            result = {"status": "error", "error": "tool invocation failed", "details": str(exc)}
        updates = {"blast_result": result, "retry_counts": retry_counts}
        if result.get("status") != "success":
            updates["errors"] = [
                _tool_error("nto_blast", getattr(blast_tool, "name", "nto_blast"), species_name,
                            result.get("error") or result.get("details") or "unknown error", attempt)
            ]
        return updates

    def route_after_blast(state: NTOAnalysisState) -> Literal["risk_hit_evaluate", "nto_blast", "risk_score"]:
        if (state.get("blast_result") or {}).get("status") == "success":
            return "risk_hit_evaluate"
        if (state.get("retry_counts") or {}).get("nto_blast", 0) < MAX_TOOL_RETRIES:
            return "nto_blast"
        return "risk_score"

    # ----- risk_hit_evaluate: select fetchable hits, compute blast_summary -----
    def risk_hit_evaluate_node(state: NTOAnalysisState) -> dict:
        risk_hits = select_fetchable_hits(state.get("blast_result", {}) or {})
        blast_result = state.get("blast_result", {}) or {}
        best_identity = 0.0
        for hit in blast_result.get("results", []) or []:
            best_identity = max(best_identity, float(hit.get("identity_pct") or 0))
        blast_summary = {
            "hits_count": blast_result.get("hits_count", 0),
            "best_identity_pct": round(best_identity, 2),
            "risk_hits_count": len(risk_hits),
        }
        return {"risk_hits": risk_hits, "blast_summary": blast_summary}

    def route_after_hit_evaluate(state: NTOAnalysisState) -> Literal["sequence_fetch", "risk_score"]:
        return "sequence_fetch" if state.get("risk_hits") else "risk_score"

    # ----- sequence_fetch: extract subject sequences for the risk hits -----
    def sequence_fetch_node(state: NTOAnalysisState) -> dict:
        species = state.get("species", {})
        species_name = _species_name(species)
        retry_counts, attempt = _increment_retry(state, "sequence_fetch")
        hits = [{"subject_id": h["subject_id"], "s_start": h["s_start"], "s_end": h["s_end"]}
                for h in state.get("risk_hits", [])]
        payload = {"species": species_name, "hits": hits}
        try:
            result = _json_loads_tool_result(fetch_tool.invoke(payload))
        except Exception as exc:
            result = {"status": "error", "error": "tool invocation failed", "details": str(exc)}

        updates = {"retry_counts": retry_counts}
        if result.get("status") == "success":
            updates["risk_sequences"] = result.get("results", []) or []
        else:
            updates["risk_sequences"] = []
            updates["errors"] = [
                _tool_error("sequence_fetch", getattr(fetch_tool, "name", "fetch_nto_seq"),
                            species_name, result.get("error") or result.get("details") or "unknown error", attempt)
            ]
        return updates

    def route_after_fetch(state: NTOAnalysisState) -> Literal["clustal", "sequence_fetch", "risk_score"]:
        if state.get("risk_sequences"):
            return "clustal"
        if (state.get("retry_counts") or {}).get("sequence_fetch", 0) < MAX_TOOL_RETRIES:
            return "sequence_fetch"
        return "risk_score"

    # ----- clustal: pairwise alignment, detect >=21nt perfect-match windows -----
    def clustal_node(state: NTOAnalysisState) -> dict:
        species = state.get("species", {})
        species_name = _species_name(species)
        retry_counts, attempt = _increment_retry(state, "clustal")
        alignment_results = []
        errors = []

        for idx, fetched in enumerate(state.get("risk_sequences", []) or [], start=1):
            target_name = f"{species_name}_{fetched.get('subject_id', 'hit')}_{idx}"
            payload = {
                "sequences": [
                    {"name": "dsRNA_query", "sequence": state["dsrna_sequence"]},
                    {"name": target_name, "sequence": fetched.get("sequence", "")},
                ],
                "window_size": 21,
            }
            try:
                result = _json_loads_tool_result(align_tool.invoke(payload))
            except Exception as exc:
                result = {"status": "error", "error": "tool invocation failed", "details": str(exc)}

            result = {**result, "subject_id": fetched.get("subject_id"),
                      "start": fetched.get("start"), "end": fetched.get("end")}
            if result.get("status") == "success":
                alignment_results.append(result)
            else:
                errors.append(
                    _tool_error("clustal", getattr(align_tool, "name", "clustal"), species_name,
                                result.get("error") or result.get("details") or "unknown error", attempt)
                )

        updates = {"alignment_results": alignment_results, "retry_counts": retry_counts}
        if errors:
            updates["errors"] = errors
        return updates

    def route_after_clustal(state: NTOAnalysisState) -> Literal["risk_score", "clustal"]:
        if state.get("alignment_results"):
            return "risk_score"
        if (state.get("retry_counts") or {}).get("clustal", 0) < MAX_TOOL_RETRIES:
            return "clustal"
        return "risk_score"

    # ----- risk_score: rule engine derives final risk level (no LLM, no tool) -----
    def risk_score_node(state: NTOAnalysisState) -> dict:
        species = state.get("species", {})
        species_name = _species_name(species)
        errors = state.get("errors", []) or []
        exhausted_error = any(
            state.get("retry_counts", {}).get(node, 0) >= MAX_TOOL_RETRIES
            for node in ("nto_blast", "sequence_fetch", "clustal")
        ) and bool(errors)
        risk_hits = state.get("risk_hits", []) or []
        alignment_results = state.get("alignment_results", []) or []
        longest_match = get_longest_match(alignment_results)
        risk_level = score_species_risk(
            longest_match=longest_match, has_fetchable_hits=bool(risk_hits),
            has_tool_error=exhausted_error,
        )

        blast_summary = state.get("blast_summary", {}) or {}
        alignment_visuals = [
            {
                "subject_id": ar.get("subject_id"),
                "start": ar.get("start"),
                "end": ar.get("end"),
                "alignment_fasta": ar.get("alignment", ""),
            }
            for ar in (alignment_results or [])
            if ar.get("status") == "success"
        ]
        species_result = {
            "species": species_name,
            "display_name": species_name.replace("_", " "),
            "category": species.get("category"),
            "chinese_name": species.get("chinese_name"),
            "english_name": species.get("english_name"),
            "scientific_name": species_name,
            "risk_level": risk_level,
            "longest_match": longest_match,
            "blast_summary": {**blast_summary, "alignments_count": len(alignment_results)},
            "alignment_visuals": alignment_visuals,
            "rule_trace": {
                "hit_filter": "identity_pct > 80 (coverage excluded)",
                "risk_rule": "no hit => negligible; >=21 => high; >=19 => medium; else low",
                "computed_by": "rule_engine",
            },
            "status": "error" if risk_level == "error" else ("negligible" if risk_level == "negligible" else "success"),
            "errors": errors,
        }
        if debug_output:
            species_result["evidence"] = {
                "risk_hits": risk_hits,
                "risk_sequences": state.get("risk_sequences", []) or [],
                "alignment_results": alignment_results,
            }
        return {"species_result": species_result, "species_results": [species_result]}

    # ----- graph assembly -----
    builder = StateGraph(NTOAnalysisState)
    builder.add_node("nto_blast", nto_blast_node)
    builder.add_node("risk_hit_evaluate", risk_hit_evaluate_node)
    builder.add_node("sequence_fetch", sequence_fetch_node)
    builder.add_node("clustal", clustal_node)
    builder.add_node("risk_score", risk_score_node)

    builder.add_edge(START, "nto_blast")
    builder.add_conditional_edges("nto_blast", route_after_blast, {
        "risk_hit_evaluate": "risk_hit_evaluate",
        "nto_blast": "nto_blast",
        "risk_score": "risk_score",
    })
    builder.add_conditional_edges("risk_hit_evaluate", route_after_hit_evaluate, {
        "sequence_fetch": "sequence_fetch",
        "risk_score": "risk_score",
    })
    builder.add_conditional_edges("sequence_fetch", route_after_fetch, {
        "clustal": "clustal",
        "sequence_fetch": "sequence_fetch",
        "risk_score": "risk_score",
    })
    builder.add_conditional_edges("clustal", route_after_clustal, {
        "risk_score": "risk_score",
        "clustal": "clustal",
    })
    builder.add_edge("risk_score", END)
    return builder.compile()


# ============================================================
# SafetyInspectorGraph
# ============================================================
def build_safety_inspector_graph(
    llm: BaseChatModel | None = None,
    checkpointer=None,
    interrupt_before: list[str] | None = None,
    species_subgraph=None,
    debug_output: bool = False,
):
    """Build the main ecological safety assessment graph.

    LLM is only used by `input_clean_node` (field-type classification) and
    `report_generate_node` (final report). When `llm` is None, the default
    LLM from `llm_config.get_default_llm()` is used. Full sequence/alignment
    evidence is included only when `debug_output=True`.

    Flow:
        input_clean -> candidate_species_build
                    -> [Send(species_analysis_subgraph) x N]
                    -> risk_aggregate -> report_generate -> END
    """
    if llm is None:
        from llm_config import get_default_llm
        llm = get_default_llm()

    compiled_species_subgraph = species_subgraph or build_species_analysis_subgraph(debug_output=debug_output)

    FIELD_TYPE_FILES = {
        "rice_field": "ricefield_NTOs.json",
        "corn_field": "cornfield_NTOs.json",
        "wheat_field": "wheatfield_NTOs.json",
        "tomatofield": "tomatofield_NTOs.json",
    }
    EPA_FILE = "EPA_NTOs.json"

    # ----- input_clean: dsRNA sanity check + LLM-driven field-type classification -----
    def input_clean_node(state: SafetyState) -> dict:
        """Clean dsRNA sequence and use LLM to classify field type(s) from user description."""
        user_input = str(state.get("field_type", "")).strip()
        try:
            sequence = normalize_dsrna_sequence(state.get("dsrna_sequence", ""))
        except ValueError as exc:
            return {"errors": [{"stage": "input_clean", "error": str(exc)}]}

        prompt = (
            "You are an ecological safety assessment expert.\n"
            "Classify the user's field/habitat description into one or more of the following types:\n"
            "  - rice_field: rice paddy and wetland ecosystems\n"
            "  - corn_field: corn/maize field ecosystems\n"
            "  - wheat_field: wheat field ecosystems\n"
            "  - tomatofield: tomato field and greenhouse ecosystems\n\n"
            f"User description: {user_input}\n\n"
            "Return a JSON object with a key 'field_types' holding a list of matched field type strings.\n"
            "Always include 'EPA' as a mandatory entry.\n"
            "If multiple field types apply, include all of them plus EPA.\n"
            "Return ONLY valid JSON, no explanation."
        )

        response = llm.invoke(prompt)
        raw = getattr(response, "content", str(response)).strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        parsed = json.loads(raw)
        field_types = parsed.get("field_types", [])
        if "EPA" not in field_types:
            field_types.append("EPA")
        return {
            "field_types": field_types,
            "dsrna_sequence": sequence,
        }

    # ----- candidate_species_build: resolve field types to NTO JSON files -----
    def candidate_species_build_node(state: SafetyState) -> dict:
        """Resolve field types to filenames and delegate to retrieve_nto_candidates()."""
        field_types = state.get("field_types", [])
        files = []
        for ft in field_types:
            if ft == "EPA":
                files.append(EPA_FILE)
            else:
                fname = FIELD_TYPE_FILES.get(ft)
                if fname:
                    files.append(fname)
        candidates, warnings = retrieve_nto_candidates(files)
        updates = {"candidate_species": candidates}
        if warnings:
            updates["errors"] = warnings
        return updates

    # ----- parallel_off_target_analyze: fan-out dispatcher for the NTO subgraph -----
    def parallel_off_target_analyze(state: SafetyState):
        """Fan-out: dispatch each candidate species into the NTO analysis subgraph."""
        candidates = state.get("candidate_species", []) or []
        if not candidates:
            return "risk_aggregate"
        return [
            Send("species_analysis_subgraph", {
                "species": species,
                "dsrna_sequence": state["dsrna_sequence"],
                "retry_counts": {},
                "species_results": [],
                "errors": [],
            })
            for species in candidates
        ]

    # ----- risk_aggregate: bucket species and compute summary counts -----
    def risk_aggregate_node(state: SafetyState) -> dict:
        results = state.get("species_results", []) or []
        groups = group_species_by_risk(results)
        return {
            **groups,
            "species_count": len(results),
            "high_count": len(groups["high_risk_species"]),
            "medium_count": len(groups["medium_risk_species"]),
            "low_count": len(groups["low_risk_species"]),
            "error_count": sum(1 for r in results if r.get("risk_level") == "error"),
        }

    # ----- report_generate: LLM produces a natural-language final report -----
    def _detect_language(text: str) -> str:
        """Heuristic language detection based on Chinese character presence."""
        if any("一" <= c <= "鿿" for c in text):
            return "Chinese"
        return "English"

    def _compact_species(species: dict) -> dict:
        """Strip a species result down to summary fields for the LLM prompt."""
        bs = species.get("blast_summary", {}) or {}
        return {
            "species": species["species"],
            "display_name": species.get("display_name", species["species"]),
            "risk_level": species["risk_level"],
            "longest_match": species.get("longest_match", 0),
            "best_identity_pct": bs.get("best_identity_pct", 0),
            "risk_hits_count": bs.get("risk_hits_count", 0),
            "alignments_count": bs.get("alignments_count", 0),
            "status": species.get("status", ""),
        }

    def report_generate_node(state: SafetyState) -> dict:
        overall = derive_overall_risk_level(state.get("species_results", []) or [])
        field_types = state.get("field_types", [])
        lang = _detect_language(str(state.get("field_type", "")))
        lang_instruction = "Write the entire report in Chinese." if lang == "Chinese" else "Write the entire report in English."

        species_results = state.get("species_results", []) or []
        summary_for_llm = [_compact_species(r) for r in species_results]

        # Extract alignment FASTA data for LLM visualization
        alignment_data_for_llm = {
            r["species"]: r.get("alignment_visuals", [])
            for r in species_results
            if r.get("alignment_visuals")
        }

        prompt = (
            f"{SAFETY_INSPECTOR_ROLE}\n\n"
            f"{build_skills('principles', 'evidence', 'tool')}\n\n"
            f"{lang_instruction}\n"
            "Generate the NTO Risk Assessment Report from the data below.\n"
            "Do NOT change any risk level — they are computed by the rule engine.\n\n"
            f"Overall risk: {overall}\n"
            f"Field types: {field_types}\n"
            f"Species analyzed: {len(species_results)}\n"
            f"High risk ({len(state.get('high_risk_species', []) or [])}): "
            f"{[s['species'] for s in state.get('high_risk_species', []) or []]}\n"
            f"Medium risk ({len(state.get('medium_risk_species', []) or [])}): "
            f"{[s['species'] for s in state.get('medium_risk_species', []) or []]}\n"
            f"Low risk ({len(state.get('low_risk_species', []) or [])}): "
            f"{[s['species'] for s in state.get('low_risk_species', []) or []]}\n\n"
            f"Species details:\n{json.dumps(summary_for_llm, ensure_ascii=False, indent=2)}\n\n"
            f"Alignment data (for visualization):\n"
            f"{json.dumps(alignment_data_for_llm, ensure_ascii=False, indent=2)}"
        )

        try:
            response = llm.invoke(prompt)
            report_text = getattr(response, "content", str(response))
        except Exception as exc:
            logger.exception("LLM report generation failed")
            report_text = f"[Report generation failed: {exc}]"

        return {
            "final_report": {
                "risk_level": overall,
                "field_types": field_types,
                "species_analyzed": len(species_results),
                "high_risk_species": state.get("high_risk_species", []) or [],
                "medium_risk_species": state.get("medium_risk_species", []) or [],
                "low_risk_species": state.get("low_risk_species", []) or [],
                "details": species_results,
                "errors": state.get("errors", []) or [],
                "report_text": report_text,
                "computed_by": {
                    "field_type_classification": "llm",
                    "nto_retrieval": "json_reader",
                    "homology_search": "nto_blast",
                    "sequence_fetch": "fetch_nto_seq",
                    "alignment": "clustal",
                    "risk_scoring": "rule_engine",
                    "risk_aggregation": "rule_engine",
                    "report_generation": "llm",
                },
            }
        }

    builder = StateGraph(SafetyState)
    builder.add_node("input_clean", input_clean_node)
    builder.add_node("candidate_species_build", candidate_species_build_node)
    builder.add_node("species_analysis_subgraph", compiled_species_subgraph)
    builder.add_node("risk_aggregate", risk_aggregate_node)
    builder.add_node("report_generate", report_generate_node)

    builder.add_edge(START, "input_clean")
    builder.add_edge("input_clean", "candidate_species_build")
    builder.add_conditional_edges(
        "candidate_species_build",
        parallel_off_target_analyze,
        ["species_analysis_subgraph", "risk_aggregate"],
    )
    builder.add_edge("species_analysis_subgraph", "risk_aggregate")
    builder.add_edge("risk_aggregate", "report_generate")
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
safety_inspector_graph = build_safety_inspector_graph(llm=get_default_llm())


if __name__ == "__main__":
    import uuid

    # Uncomment to suppress tool/library INFO logs for clean output:
    # for name in ("RPA_Agent", "RPA_Tools", "langchain_openai", "httpx"):
    #     logging.getLogger(name).setLevel(logging.WARNING)

    test_seq = (
        "AATAGTATAATGGCGAAAAGGAAGTCGCAGATGCAAACTTTAGGAGCAAATAACATTTACGGAGTGGCGGGCAATCCTTTGGGAATTCAAGCATCGAGAAGTAGCCAGATTTCTGTTAAGGATGTATTCGAAGGACACAGTGGTCATACTAACCCAGGCTACGAACCAGACGAAAATGCTGGAAACAATCTCACTTCACAGTTTAAGTCAGAATCCGTGGAGAGAACAGAATAATAtctga"
    )
    result = safety_inspector_graph.invoke(
        {"field_type": "rice_field", "dsrna_sequence": test_seq},
        config={"configurable": {"thread_id": f"safety-{uuid.uuid4().hex[:8]}"}},
    )
    print(json.dumps(result.get("final_report"), ensure_ascii=False, indent=2))
