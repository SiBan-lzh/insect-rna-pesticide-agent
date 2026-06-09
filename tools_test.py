"""
tools_test.py —— 生信工具集成测试脚本

测试所有已封装的 LangChain Tool，验证工具注册、调用、输出完整性。
每个工具打印完整 JSON 输出。

用法:
    cd /home/lizonghuan/langgraph
    source langgraph_env/bin/activate
    python tools_test.py
"""

import json
import sys
sys.path.insert(0, "/home/lizonghuan/langgraph")

from tools import ALL_TOOLS
from tools.insect_blast import insect_blast_tool
from tools.nto_blast import nto_blast_tool
from tools.primer3 import primer3_tool
from tools.oligowalk import oligowalk_tool
from tools.insect_anno import insect_anno_tool
from tools.clustal import clustal_tool
from tools.fetch_seq import fetch_nto_seq_tool, fetch_insect_cds_tool
from tools.kinship import kinship_tool
from tools.pubmed_esearch import pubmed_esearch_tool
from tools.pubmed_efetch import pubmed_efetch_tool
from tools.openalex_search import openalex_search_tool
from tools.clip_seq import clip_seq_tool

# ============================================================
# 公共测试序列（家蚕 V-ATPase 107bp）
# ============================================================
TEST_SEQ = (
    "ATGAAGCGGCAGAATGTACGAACATTGTCACTTGTGGTTTGCACTTTTACGTATCTT"
    "CTCATCGGAGCAGCGGTCTTTGATGCATTGGAGTCAGACACCGAAAGTAA"
)

PASS = 0
FAIL = 0


def run_test(name: str, tool, params: dict, expectations: list[str] = None):
    """调用工具、打印完整输出、验证预期内容。"""
    global PASS, FAIL

    print("-" * 50)
    print(f"[{name}]")
    print(f"参数: {json.dumps(params, ensure_ascii=False, indent=2)}")
    print()

    try:
        result = tool.invoke(params)
        print(result)
    except Exception as e:
        FAIL += 1
        print(f"❌ 异常: {e}")
        print()
        return

    print()

    if expectations:
        missing = [e for e in expectations if e not in result]
        if missing:
            FAIL += 1
            print(f"❌ 验证失败 — 缺失: {missing}")
        else:
            PASS += 1
            print(f"✅ 通过")
    else:
        PASS += 1
        print(f"✅ 通过")


# ============================================================
print("=" * 60)
print("🧬 生信工具集成测试")
print(f"   已注册工具: {len(ALL_TOOLS)} 个")
print("=" * 60)

# ============================================================
# 1. insect_blast
# ============================================================
print("\n[1] insect_blast — 昆虫基因组 BLAST 搜索")

run_test("正常搜索 (Bombyx_mori)", insect_blast_tool, {
    "sequence": TEST_SEQ,
    "species": "Bombyx_mori",
}, ["status", "hits"])

run_test("缺失物种 (NoSuchBug)", insect_blast_tool, {
    "sequence": "ATGCGTACGT",
    "species": "NoSuchBug",
}, ["error"])

# ============================================================
# 2. nto_blast
# ============================================================
print("\n[2] nto_blast — 非靶标生物 BLAST 搜索")

run_test("正常搜索 (Apis_mellifera)", nto_blast_tool, {
    "sequence": TEST_SEQ,
    "species": "Apis_mellifera",
}, ["status", "hits", "off_target_risk"])

# ============================================================
# 3. primer3
# ============================================================
print("\n[3] primer3 — PCR 引物设计")

run_test("默认参数 (3对引物)", primer3_tool, {
    "sequence": TEST_SEQ,
    "num_return": 3,
}, ["status", "primers", "forward_with_T7"])

run_test("带靶标区域 (target_start=10, target_len=60)", primer3_tool, {
    "sequence": TEST_SEQ,
    "target_start": 10,
    "target_len": 60,
    "num_return": 5,
    "primer_product_size_range": [80, 200],
}, ["status", "primers"])

# ============================================================
# 4. oligowalk
# ============================================================
print("\n[4] oligowalk — siRNA 热力学打分")

run_test("fast 模式 (top_n=5)", oligowalk_tool, {
    "sequence": TEST_SEQ,
    "run_type": "fast",
    "top_n": 5,
}, ["status", "top_candidates", "Overall (kcal/mol)"])

run_test("research 模式 (top_n=3, 短序列)", oligowalk_tool, {
    "sequence": TEST_SEQ[:80],
    "run_type": "research",
    "top_n": 3,
}, ["status", "top_candidates", "End Diff"])

# ============================================================
# 5. insect_anno
# ============================================================
print("\n[5] insect_anno — BLAST命中基因组注释")

# 用 insect_blast 的真实输出来测试
run_test("注释 chr25 上两个 BLAST 命中 (Bombyx_mori)", insect_anno_tool, {
    "blast_hits": [
        {
            "chromosome": "BMSK_chr25",
            "start_position": 10215315,
            "end_position": 10215421,
            "name": "chr25_hit1",
            "score": 171.0,
            "strand": "+",
        },
        {
            "chromosome": "BMSK_chr25",
            "start_position": 13437514,
            "end_position": 13437620,
            "name": "chr25_hit2",
            "score": 171.0,
            "strand": "+",
        },
    ],
    "species": "Bombyx_mori",
    "window_size": 100,
}, ["status", "features", "gene", "CDS"])

run_test("缺失物种", insect_anno_tool, {
    "blast_hits": [
        {
            "chromosome": "chr1",
            "start_position": 1,
            "end_position": 100,
            "name": "test",
            "score": 10.0,
            "strand": "+",
        },
    ],
    "species": "NoSuchBug",
}, ["error"])

# ============================================================
# 6. clustal
# ============================================================
print("\n[6] clustal — 双序列比对与脱靶评估")

run_test("siRNA vs 自身 (100% 连续匹配)", clustal_tool, {
    "sequences": [
        {"name": "siRNA", "sequence": TEST_SEQ},
        {"name": "target", "sequence": TEST_SEQ},
    ],
    "window_size": 21,
}, ["status", "off_target_risk", "alignment"])

run_test("不相关序列 (无风险)", clustal_tool, {
    "sequences": [
        {"name": "siRNA", "sequence": "ATGAAGCGGCAGAATGTACGA"},
        {"name": "gene", "sequence": "TTGCCGTATAGCTACGTGCCT"},
    ],
    "window_size": 21,
}, ["status", "off_target_risk"])

# ============================================================
# 7. fetch_seq
# ============================================================
print("\n[7] fetch_nto_seq — NTO 参考序列提取")

run_test("提取 Apis_mellifera 两个命中区段", fetch_nto_seq_tool, {
    "species": "Apis_mellifera",
    "hits": [
        {"subject_id": "XM_026440900.1", "s_start": 602, "s_end": 708},
        {"subject_id": "XM_006562189.3", "s_start": 588, "s_end": 694},
    ],
}, ["status", "sequence", "length"])

# ============================================================
# 8. fetch_insect_cds
# ============================================================
print("\n[8] fetch_insect_cds — 昆虫 CDS 序列查询")

run_test("查询 Bmor000255.1 (Bombyx_mori)", fetch_insect_cds_tool, {
    "species": "Bombyx_mori",
    "transcript_id": "Bmor000255.1",
}, ["status", "sequence", "length"])

run_test("不存在的转录本 ID", fetch_insect_cds_tool, {
    "species": "Bombyx_mori",
    "transcript_id": "NoSuchTranscript",
}, ["error"])

# ============================================================
# 9. kinship
# ============================================================
print("\n[9] kinship — 物种亲缘关系计算")

run_test("蚜虫 (Acyrthosiphon_pisum) 近缘物种 (top 5)", kinship_tool, {
    "target_species": "Acyrthosiphon_pisum",
    "target_total": 5,
}, ["status", "species_name", "query_name", "divergence_time_mya", "Congeneric"])

run_test("不存在的物种", kinship_tool, {
    "target_species": "NoSuchBug_FakeSpecies",
    "target_total": 5,
}, ["error", "not_found_in_tree"])

# ============================================================
# 10. clip_seq
# ============================================================
print("\n[10] clip_seq — 序列片段截取")

run_test("从 TEST_SEQ 截取 50bp (默认 sense 链)", clip_seq_tool, {
    "sequence": TEST_SEQ,
    "start": 10,
    "length": 50,
}, ["status", "sequence", "gc_content", "strand", "actual_length"])

run_test("返回反义互补链", clip_seq_tool, {
    "sequence": TEST_SEQ,
    "start": 1,
    "length": 60,
    "as_reverse_complement": True,
}, ["status", "sequence", "strand"])

run_test("起始位置超出序列范围", clip_seq_tool, {
    "sequence": TEST_SEQ,
    "start": 999,
    "length": 50,
}, ["error"])

# ============================================================
# 11. pubmed_esearch
# ============================================================
print("\n[11] pubmed_esearch — PubMed 文献检索")

run_test("搜索 RNAi insect pest (前3篇)", pubmed_esearch_tool, {
    "query": "RNAi insect pest control",
    "max_results": 3,
}, ["success", "pmids", "total_count"])

run_test("空查询（应报错）", pubmed_esearch_tool, {
    "query": "",
    "max_results": 3,
}, ["error"])

# ============================================================
# 12. pubmed_efetch
# ============================================================
print("\n[12] pubmed_efetch — PubMed 文献详情获取")

# 先用 ESearch 拿到真实 PMID，再传给 EFetch
print("  先搜索获取 PMID...")
try:
    search_result = json.loads(pubmed_esearch_tool.invoke({
        "query": "RNAi Bombyx mori",
        "max_results": 2,
    }))
    if search_result.get("success") and search_result.get("pmids"):
        real_pmids = ",".join(search_result["pmids"])
        run_test(f"获取文献详情 (PMIDs: {real_pmids})", pubmed_efetch_tool, {
            "pmids": real_pmids,
        }, ["success", "articles", "title", "abstract"])
    else:
        print("  ⚠️ ESearch 返回空结果，跳过 EFetch 测试")
except Exception as e:
    print(f"  ⚠️ ESearch 调用失败: {e}，跳过 EFetch 测试")

run_test("空 PMID（应报错）", pubmed_efetch_tool, {
    "pmids": "",
}, ["false"])

# ============================================================
# 13. openalex_search
# ============================================================
print("\n[13] openalex_search — OpenAlex 文献检索")

run_test("搜索 RNAi insect (前3篇, 不限领域)", openalex_search_tool, {
    "keyword": "RNA interference pest control",
    "max_results": 3,
}, ["success", "articles", "title", "openalex_id"])

run_test("搜索 insect 领域 RNAi 文献", openalex_search_tool, {
    "keyword": "RNAi",
    "max_results": 3,
    "field": "insect",
    "min_year": "2020",
    "max_year": "2024",
}, ["success", "articles", "applied_filters"])

run_test("空关键词（应报错）", openalex_search_tool, {
    "keyword": "",
    "max_results": 3,
}, ["error"])

# ============================================================
print("\n" + "=" * 60)
print(f"  结果: {PASS} 通过 / {FAIL} 失败 / {PASS + FAIL} 总计")
if FAIL == 0:
    print("  🎉 全部通过！")
else:
    print(f"  ⚠️ {FAIL} 项测试失败")
print("=" * 60)
