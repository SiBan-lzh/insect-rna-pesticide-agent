"""
kinship.py —— 物种亲缘关系计算工具

纯 Python 工具，使用 ETE4 进化树 + NCBI 分类数据库，计算目标物种与
其它昆虫物种之间的分化时间（Mya）和分类学关系。

输出仅包含双名法物种（Genus species），过滤属级/未定种等无法用于
下游 RAG 检索的名称。

典型工作流:
    kinship(A) → 近缘物种列表 → RAG 检索 RNAi 研究记录
    → 提取已验证 dsRNA → BLAST 同源比对 → 找到跨物种保守靶标

从 old_tools/kinship_calculation/executor.py 抽取核心逻辑。

调用链路：
    LLM → KinshipTool._run() → ETE4 Tree + NCBITaxa → JSON
"""

import json
import logging
import re
from typing import List, Tuple, Optional
from pydantic import BaseModel, Field
from langchain_core.tools import BaseTool

from ete4 import Tree
from ete4.ncbi_taxonomy.ncbiquery import NCBITaxa

from tool_config import INSECT_TREE_PATH, INSECT_TAXA_DB_PATH

logger = logging.getLogger("RPA_Tools.Kinship")

# ============================================================
# NCBI taxonomy database (lazy-loaded on first use)
# ============================================================
_ncbi = None

def _get_ncbi():
    """Lazy-load NCBITaxa singleton, returns None on failure."""
    global _ncbi
    if _ncbi is None and INSECT_TAXA_DB_PATH.exists():
        try:
            _ncbi = NCBITaxa(dbfile=str(INSECT_TAXA_DB_PATH))
            logger.info("NCBI Taxa DB loaded: %s", INSECT_TAXA_DB_PATH)
        except Exception as e:
            logger.error("Failed to load NCBI Taxa DB: %s", e)
    return _ncbi

# ============================================================
# 分类学关系标签（从最亲到最远）
# ============================================================
RANK_LABELS = {
    "genus": "Congeneric",
    "family": "Confamilial",
    "suborder": "Consubordinal",
    "order": "Conordinal",
    "superorder": "Consuperordinal",
}

# 代理搜索：当目标物种不在树中时，依次向上查找
SEARCH_HIERARCHY = ["genus", "family", "suborder", "order", "superorder"]

# 仅接受标准双名法格式: "Genus species"（两个单词）
_BINOMIAL_PATTERN = re.compile(r"^[A-Z][a-z\-]+ [a-z\-]+$")


# ============================================================
# Pydantic 输入 Schema
# ============================================================
class KinshipInput(BaseModel):
    """物种亲缘关系查询参数。"""

    target_species: str = Field(
        description="目标物种名称，如 'Bombyx_mori' 或 'Bombyx mori'"
    )
    target_total: int = Field(
        default=8, ge=1, le=50,
        description="返回的最相近物种数量"
    )


# ============================================================
# 核心函数（从原始 executor.py 移植，适配 ete4 API）
# ============================================================
def _is_usable_species(display_name: str) -> bool:
    """判断是否为可用的双名法物种名。

    接受:  Bombyx mandarina, Carthaea saturnioides
    拒绝:  Epia (属名), Quentalia sp. JCR-2007 (未定种), Anthela_varia (未清洗)

    下游 RAG 检索只能用标准双名法物种名，属名/未定种/复杂标记均无法搜到论文。
    """
    return bool(_BINOMIAL_PATTERN.match(display_name))


def _raw_to_display(name: str) -> str:
    """树节点原始名 → 可展示的物种名。

    'Bombyx_mori' → 'Bombyx mori'
    'Epia' → 'Epia'
    """
    return name.replace("'", "").replace("_", " ").strip()


def _clean_leaf_name(name: str) -> str:
    """清理树中叶节点的名称（去除引号，标准化为小写）。"""
    return name.replace("'", "").replace("_", " ").strip().lower()


def _extract_genus(species_name: str) -> str:
    """从双名法物种名中提取属名。"""
    return species_name.split()[0].lower()


def _load_tree() -> Tree:
    """加载物种进化树（Newick 格式）。"""
    with open(str(INSECT_TREE_PATH), "r") as f:
        newick_str = f.read()
    return Tree(newick_str, parser=1)


def _get_relation_label(target_clean: str, candidate_name: str) -> str:
    """使用 NCBI 分类法判断两个物种间的分类学关系。"""
    if not _get_ncbi():
        return "unknown"

    try:
        target_map = _get_ncbi().get_name_translator([target_clean])
        cand_map = _get_ncbi().get_name_translator([candidate_name])
        if not target_map or not cand_map:
            return "unknown"

        target_id = list(target_map.values())[0][0]
        cand_id = list(cand_map.values())[0][0]

        target_lineage = set(_get_ncbi().get_lineage(target_id))
        cand_lineage = _get_ncbi().get_lineage(cand_id)
        ranks = _get_ncbi().get_rank(list(target_lineage | set(cand_lineage)))

        for tid in reversed(cand_lineage):
            if tid in target_lineage:
                rank = ranks.get(tid, "")
                if rank in RANK_LABELS:
                    return RANK_LABELS[rank]
        return "unknown"
    except Exception:
        return "unknown"


def get_stepped_candidates(
    target_species: str,
    target_total: int,
) -> Tuple[List[dict], str, Optional[str], str]:
    """查找与目标物种最相近的、可用的双名法物种列表。

    返回: (候选列表, 搜索模式, 代理物种名或 None, 目标属名)
    """
    # 加载树
    try:
        t = _load_tree()
    except Exception as e:
        logger.error("Failed to load tree: %s", e)
        return [], "tree_load_error", None, ""

    clean_name = target_species.replace("_", " ").strip()
    formatted_target = clean_name.replace(" ", "_")
    target_genus = _extract_genus(clean_name)

    # 所有叶节点（ete4: Tree 可迭代，返回叶节点）
    all_leaves = list(t)

    # 在树中查找目标物种（用原始节点名，保留下划线）
    target_nodes = [
        n for n in all_leaves
        if formatted_target.lower() in n.name.replace("'", "").lower()
    ]

    target_node = None
    proxy_species = None
    search_mode = "tree_based"

    if target_nodes:
        target_node = target_nodes[0]
    elif _get_ncbi():
        # 代理回退：沿 NCBI 分类层级向上查找树中存在的近亲
        logger.warning(
            "Target '%s' not found in tree, searching for taxonomic proxy...",
            clean_name,
        )
        try:
            name_map = _get_ncbi().get_name_translator([clean_name])
            if name_map:
                target_id = list(name_map.values())[0][0]
                lineage = _get_ncbi().get_lineage(target_id)
                lineage_ranks = _get_ncbi().get_rank(lineage)

                tree_leaf_set = {_clean_leaf_name(l.name) for l in all_leaves}

                for rank_name in SEARCH_HIERARCHY:
                    rank_ids = [
                        tid for tid, r in lineage_ranks.items()
                        if r == rank_name
                    ]
                    if not rank_ids:
                        continue

                    descendants = _get_ncbi().get_descendant_taxa(
                        rank_ids[0], collapse_subspecies=True
                    )
                    id_to_name = _get_ncbi().get_taxid_translator(descendants)

                    group_candidates = []
                    for name in id_to_name.values():
                        norm = name.strip().lower()
                        if norm in tree_leaf_set and norm != clean_name.lower():
                            group_candidates.append(name.strip())

                    if group_candidates:
                        for leaf in all_leaves:
                            if _clean_leaf_name(leaf.name) in [
                                g.lower() for g in group_candidates
                            ]:
                                target_node = leaf
                                proxy_species = _raw_to_display(leaf.name)
                                search_mode = f"proxy_based_{rank_name}"
                                logger.info(
                                    "Found proxy '%s' at '%s' level for '%s'",
                                    proxy_species, rank_name, clean_name,
                                )
                                break
                    if target_node:
                        break
        except Exception as proxy_err:
            logger.error("Taxonomic proxy search crashed: %s", proxy_err)

    if not target_node:
        logger.error(
            "Target and relatives absent from tree: %s", clean_name
        )
        return [], "not_found_in_tree", None, ""

    # 计算目标到所有其他叶节点的分支距离
    raw_results = []
    for leaf in all_leaves:
        if leaf == target_node:
            continue
        try:
            dist = t.get_distance(target_node, leaf)
            mya = round(dist / 2, 4)
        except Exception:
            continue

        display_name = _raw_to_display(leaf.name)
        raw_results.append({
            "display_name": display_name,
            "mya": mya,
        })

    if not raw_results:
        return [], "no_relatives_in_tree", proxy_species, target_genus

    # ============================================================
    # 过滤：仅保留双名法物种（可用 RAG 检索的）
    # ============================================================
    usable = [r for r in raw_results if _is_usable_species(r["display_name"])]

    if not usable:
        logger.error("No usable species-level relatives found in tree")
        return [], "no_usable_species", proxy_species, target_genus

    # ============================================================
    # 排序：同属优先，组内按 mya 升序
    # ============================================================
    def sort_key(item: dict) -> tuple:
        same_genus = _extract_genus(item["display_name"]) == target_genus
        return (0 if same_genus else 1, item["mya"])

    usable.sort(key=sort_key)

    # 截取 top N
    top_n = usable[:target_total]

    # 标注分类学关系
    candidates = []
    for item in top_n:
        relation = _get_relation_label(clean_name, item["display_name"])
        candidates.append({
            "species_name": item["display_name"],
            "query_name": item["display_name"].lower(),
            "relation": relation,
            "mya": item["mya"],
        })

    return candidates, search_mode, proxy_species, target_genus


# ============================================================
# LangChain Tool
# ============================================================
class KinshipTool(BaseTool):
    """物种亲缘关系计算工具。

    基于昆虫物种进化树和 NCBI 分类数据库，计算目标物种与
    其他昆虫物种之间的分歧时间和分类学关系。

    仅返回双名法物种（Genus species），过滤属级节点和未定种，
    确保下游 RAG 检索可以直接使用输出的物种名搜索研究论文。

    典型用途：给定靶标昆虫 A，找到有 RNAi 研究记录的近缘物种 B、C，
    从 B/C 中提取已验证的 dsRNA 序列，反哺 A 的靶标设计。

    对应原始服务: old_tools/kinship_calculation/executor.py
    """

    name: str = "kinship"
    description: str = (
        "物种亲缘关系计算工具。输入目标昆虫物种名（如 Bombyx_mori），"
        "返回与其亲缘最近的可检索昆虫物种列表，"
        "每项含标准物种名（可直接用于文献搜索）、分歧时间（Mya）、"
        "分类学关系（Congeneric 同属 / Confamilial 同科 等）。"
        "结果已过滤属级节点和未定种，可直接用于 RAG 检索。"
    )
    args_schema: type = KinshipInput

    def _run(
        self,
        target_species: str,
        target_total: int = 8,
    ) -> str:
        """计算物种亲缘关系（同步）。"""

        try:
            candidates, search_mode, proxy, target_genus = get_stepped_candidates(
                target_species, target_total
            )

            if not candidates:
                return json.dumps(
                    {
                        "status": "error",
                        "error": f"No usable species-level relatives found for '{target_species}'",
                        "search_mode": search_mode,
                        "hint": (
                            "All close relatives in the tree are genus-level "
                            "or undescribed species, which cannot be used for "
                            "RAG literature search."
                        ),
                    },
                    ensure_ascii=False,
                    indent=2,
                )

            # 格式化输出
            results = []
            for cand in candidates:
                results.append({
                    "species_name": cand["species_name"],
                    "query_name": cand["query_name"],
                    "relation": cand["relation"],
                    "divergence_time_mya": cand["mya"],
                })

            return json.dumps(
                {
                    "status": "success",
                    "target": target_species,
                    "search_mode": search_mode,
                    "proxy_used": proxy is not None,
                    "proxy_species": proxy,
                    "genus": target_genus.capitalize() if target_genus else None,
                    "returned_count": len(results),
                    "results": results,
                },
                ensure_ascii=False,
                indent=2,
            )

        except Exception as e:
            logger.exception("Kinship calculation failed")
            return json.dumps(
                {
                    "status": "error",
                    "error": "internal error",
                    "details": str(e),
                },
                ensure_ascii=False,
                indent=2,
            )


# ============================================================
# 单例导出
# ============================================================
kinship_tool = KinshipTool()
