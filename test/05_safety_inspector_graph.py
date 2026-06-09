"""
05_safety_inspector_graph.py —— SafetyInspectorGraph tests and demo.

Covers rule-engine boundaries, NTO JSON retrieval, reusable NTOAnalysisSubgraph,
retry exhaustion, and Send() aggregation with fake tools.

Usage:
    cd /home/lizonghuan/langgraph
    source langgraph_env/bin/activate
    python test/05_safety_inspector_graph.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agents.safety_inspector import (
    build_species_analysis_subgraph,
    compute_coverage_pct,
    derive_overall_risk_level,
    get_longest_match,
    read_nto_json,
    score_species_risk,
    select_fetchable_hits,
    NTOAnalysisState,
)


TEST_SEQ = (
    "ATGAAGCGGCAGAATGTACGAACATTGTCACTTGTGGTTTGCACTTTTACGTATCTT"
    "CTCATCGGAGCAGCGGTCTTTGATGCATTGGAGTCAGACACCGAAAGTAA"
)

PASS = 0
FAIL = 0


def check(name, fn):
    global PASS, FAIL
    print("-" * 60)
    print(f"[{name}]")
    try:
        fn()
    except Exception as exc:
        FAIL += 1
        print(f"❌ 失败: {exc}")
    else:
        PASS += 1
        print("✅ 通过")


class FakeTool:
    def __init__(self, name, outputs=None, fail=False):
        self.name = name
        self.outputs = list(outputs or [])
        self.fail = fail
        self.calls = 0

    def invoke(self, payload):
        self.calls += 1
        if self.fail:
            return json.dumps({
                "status": "error",
                "error": f"{self.name} forced failure",
            })
        if self.outputs:
            output = self.outputs[min(self.calls - 1, len(self.outputs) - 1)]
            return json.dumps(output)
        return json.dumps({"status": "success"})


def make_blast_output(identity=95.0, aln_length=None):
    aln_length = aln_length if aln_length is not None else len(TEST_SEQ)
    return {
        "status": "success",
        "species": "Apis_mellifera",
        "query_length": len(TEST_SEQ),
        "hits_count": 1,
        "risky_hits_count": 1,
        "off_target_risk": "high",
        "results": [
            {
                "hit_number": 1,
                "subject_id": "XM_026440900.1",
                "identity_pct": identity,
                "aln_length": aln_length,
                "mismatches": 0,
                "gap_opens": 0,
                "q_start": 1,
                "q_end": aln_length,
                "s_start": 602,
                "s_end": 708,
                "evalue": 1e-20,
                "bit_score": 85.0,
            }
        ],
    }


def make_fetch_output():
    return {
        "status": "success",
        "species": "Apis_mellifera",
        "results": [
            {
                "subject_id": "XM_026440900.1",
                "start": 602,
                "end": 708,
                "length": len(TEST_SEQ),
                "sequence": TEST_SEQ,
            }
        ],
    }


def make_clustal_output(longest):
    return {
        "status": "success",
        "window_size": 21,
        "analysis": {
            "max_continuous_match": longest,
            "threshold": 21,
            "off_target_risk": longest >= 21,
            "matching_regions": [{"start_alignment_pos": 0, "length": longest}] if longest >= 21 else [],
        },
        "alignment": ">dsRNA_query\nATGC\n>target\nATGC\n",
    }


def test_rule_engine_boundaries():
    hit = {"identity_pct": 81, "aln_length": 81}
    assert compute_coverage_pct(hit, 100) == 81.0
    blast = {"status": "success", "query_length": 100, "results": [hit]}
    assert len(select_fetchable_hits(blast, "A" * 100)) == 1

    assert select_fetchable_hits({"query_length": 100, "results": [{"identity_pct": 80, "aln_length": 81}]}, "A" * 100) == []
    assert select_fetchable_hits({"query_length": 100, "results": [{"identity_pct": 81, "aln_length": 80}]}, "A" * 100) == []

    assert score_species_risk(21, True) == "high"
    assert score_species_risk(20, True) == "medium"
    assert score_species_risk(19, True) == "medium"
    assert score_species_risk(18, True) == "low"
    assert score_species_risk(0, False) == "negligible"
    assert score_species_risk(21, True, has_tool_error=True) == "error"

    assert get_longest_match([{"analysis": {"max_continuous_match": 18}}, {"analysis": {"max_continuous_match": 21}}]) == 21
    assert derive_overall_risk_level([{"risk_level": "low"}, {"risk_level": "medium"}]) == "medium"
    assert derive_overall_risk_level([{"risk_level": "error"}]) == "incomplete"


def test_nto_json_retrieval():
    # Load two JSON files and verify deduplication
    records_a = read_nto_json(Path(__file__).resolve().parent.parent / "database" / "NTOs_lists" / "ricefield_NTOs.json")
    records_b = read_nto_json(Path(__file__).resolve().parent.parent / "database" / "NTOs_lists" / "EPA_NTOs.json")
    combined = {r["scientific_name"]: r for r in records_a + records_b}
    assert "Bubalus_bubalus" in combined
    assert any(r["source_file"] == "EPA_NTOs.json" for r in combined.values())


def invoke_species_subgraph(longest):
    subgraph = build_species_analysis_subgraph(
        blast_tool=FakeTool("nto_blast", [make_blast_output()]),
        fetch_tool=FakeTool("fetch_nto_seq", [make_fetch_output()]),
        align_tool=FakeTool("clustal", [make_clustal_output(longest)]),
    )
    return subgraph.invoke({
        "species": {"scientific_name": "Apis_mellifera", "category": "Pollinator"},
        "dsrna_sequence": TEST_SEQ,
        "phase": "initialized",
        "retry_counts": {},
        "species_results": [],
        "errors": [],
    })


def test_species_subgraph_high():
    result = invoke_species_subgraph(21)
    species_result = result["species_results"][-1]
    assert species_result["risk_level"] == "high"
    assert species_result["longest_match"] == 21
    assert species_result["rule_trace"]["computed_by"] == "rule_engine"


def test_species_subgraph_medium():
    result = invoke_species_subgraph(19)
    assert result["species_results"][-1]["risk_level"] == "medium"


def test_species_subgraph_negligible():
    subgraph = build_species_analysis_subgraph(
        blast_tool=FakeTool("nto_blast", [{"status": "success", "query_length": len(TEST_SEQ), "hits_count": 0, "results": []}]),
        fetch_tool=FakeTool("fetch_nto_seq", [make_fetch_output()]),
        align_tool=FakeTool("clustal", [make_clustal_output(21)]),
    )
    result = subgraph.invoke({
        "species": {"scientific_name": "Apis_mellifera"},
        "dsrna_sequence": TEST_SEQ,
        "phase": "initialized",
        "retry_counts": {},
        "species_results": [],
        "errors": [],
    })
    assert result["species_results"][-1]["risk_level"] == "negligible"


def test_species_subgraph_retry_error():
    blast_tool = FakeTool("nto_blast", fail=True)
    subgraph = build_species_analysis_subgraph(
        blast_tool=blast_tool,
        fetch_tool=FakeTool("fetch_nto_seq", [make_fetch_output()]),
        align_tool=FakeTool("clustal", [make_clustal_output(21)]),
    )
    result = subgraph.invoke({
        "species": {"scientific_name": "Apis_mellifera"},
        "dsrna_sequence": TEST_SEQ,
        "phase": "initialized",
        "retry_counts": {},
        "species_results": [],
        "errors": [],
    })
    assert blast_tool.calls == 3
    assert result["species_results"][-1]["risk_level"] == "error"


def test_send_fan_out_and_aggregate():
    """Test that parallel Send dispatches multiple species and results aggregate correctly."""
    import operator
    from typing import Annotated
    from agents.safety_inspector import SafetyState, build_species_analysis_subgraph
    from langgraph.graph import END, START, StateGraph
    from langgraph.types import Send
    from typing_extensions import TypedDict

    species_subgraph = build_species_analysis_subgraph(
        blast_tool=FakeTool("nto_blast", [make_blast_output()]),
        fetch_tool=FakeTool("fetch_nto_seq", [make_fetch_output()]),
        align_tool=FakeTool("clustal", [make_clustal_output(21)]),
    )

    class TestState(TypedDict, total=False):
        candidate_species: list[dict]
        dsrna_sequence: Annotated[str, operator.add]
        species_results: Annotated[list[dict], operator.add]
        high_risk_species: list[dict]
        medium_risk_species: list[dict]
        low_risk_species: list[dict]

    def candidate_build(state: TestState) -> dict:
        return {
            "candidate_species": [
                {"scientific_name": "Apis_mellifera", "category": "Pollinator"},
                {"scientific_name": "Bombus_terrestris", "category": "Pollinator"},
            ],
            "dsrna_sequence": TEST_SEQ,
        }

    def dispatch(state: TestState):
        dsrna = state["dsrna_sequence"]
        return [
            Send(
                "subgraph",
                {
                    "species": s,
                    "dsrna_sequence": dsrna,
                    "phase": "initialized",
                    "retry_counts": {},
                    "species_results": [],
                    "errors": [],
                },
            )
            for s in state["candidate_species"]
        ]

    def aggregate(state: TestState) -> dict:
        results: list = state.get("species_results", [])
        return {
            "high_risk_species": [r for r in results if r.get("risk_level") == "high"],
            "medium_risk_species": [r for r in results if r.get("risk_level") == "medium"],
            "low_risk_species": [r for r in results if r.get("risk_level") in ("low", "negligible")],
        }

    builder = StateGraph(TestState)
    builder.add_node("candidate_build", candidate_build)
    builder.add_node("subgraph", species_subgraph)
    builder.add_node("aggregate", aggregate)
    builder.add_edge(START, "candidate_build")
    builder.add_conditional_edges("candidate_build", dispatch, ["subgraph", "aggregate"])
    builder.add_edge("subgraph", "aggregate")
    builder.add_edge("aggregate", END)
    graph = builder.compile()

    result = graph.invoke(
        {"candidate_species": [], "dsrna_sequence": TEST_SEQ},
        config={"configurable": {"thread_id": "safety-test-send"}},
    )
    assert len(result.get("species_results", [])) == 2
    assert all(r.get("risk_level") == "high" for r in result["species_results"])


if __name__ == "__main__":
    print("=" * 60)
    print("🧬 SafetyInspectorGraph 测试")
    print("=" * 60)

    check("规则引擎边界", test_rule_engine_boundaries)
    check("NTO json 清单读取", test_nto_json_retrieval)
    check("NTOAnalysisSubgraph high", test_species_subgraph_high)
    check("NTOAnalysisSubgraph medium", test_species_subgraph_medium)
    check("NTOAnalysisSubgraph negligible", test_species_subgraph_negligible)
    check("NTOAnalysisSubgraph retry error", test_species_subgraph_retry_error)
    check("Send fan-out 聚合", test_send_fan_out_and_aggregate)

    print("\n" + "=" * 60)
    print(f"  结果: {PASS} 通过 / {FAIL} 失败 / {PASS + FAIL} 总计")
    if FAIL == 0:
        print("  🎉 全部通过！")
    else:
        print(f"  ⚠️ {FAIL} 项测试失败")
    print("=" * 60)

    if FAIL:
        sys.exit(1)
