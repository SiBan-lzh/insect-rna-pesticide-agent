"""
kinship.py -- Species kinship calculation tool.

Pure Python tool using ETE4 phylogeny + NCBI taxonomy database.
Computes divergence time (Mya) and taxonomic relationship between
a target insect species and other insect species.

Output contains only binomial species (Genus species), filtering out
genus-level or undescribed names unusable for downstream RAG retrieval.
"""

import json
import logging
import re
from typing import List, Tuple, Optional
from pydantic import BaseModel, Field
from langchain_core.tools import BaseTool

from ete4 import Tree
from ete4.ncbi_taxonomy.ncbiquery import NCBITaxa

from .config import INSECT_TREE_PATH, INSECT_TAXA_DB_PATH

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
# Taxonomic relation labels (closest to farthest)
# ============================================================
RANK_LABELS = {
    "genus": "Congeneric",
    "family": "Confamilial",
    "suborder": "Consubordinal",
    "order": "Conordinal",
    "superorder": "Consuperordinal",
}

# Proxy search: when target not in tree, look up hierarchy
SEARCH_HIERARCHY = ["genus", "family", "suborder", "order", "superorder"]

# Accept only standard binomial format: "Genus species" (two words)
_BINOMIAL_PATTERN = re.compile(r"^[A-Z][a-z\-]+ [a-z\-]+$")


# ============================================================
# Pydantic input schema
# ============================================================
class KinshipInput(BaseModel):
    """Species kinship query parameters."""

    target_species: str = Field(
        description="Target species name, e.g. 'Bombyx_mori' or 'Bombyx mori'"
    )
    target_total: int = Field(
        default=8, ge=1, le=50,
        description="Number of closest relatives to return"
    )


# ============================================================
# Core functions
# ============================================================
def _is_usable_species(display_name: str) -> bool:
    """Check whether name is a valid binomial species name."""
    return bool(_BINOMIAL_PATTERN.match(display_name))


def _raw_to_display(name: str) -> str:
    """Convert raw tree node name to display name."""
    return name.replace("'", "").replace("_", " ").strip()


def _clean_leaf_name(name: str) -> str:
    """Clean leaf node name (strip quotes, lowercase)."""
    return name.replace("'", "").replace("_", " ").strip().lower()


def _extract_genus(species_name: str) -> str:
    """Extract genus from a binomial species name."""
    return species_name.split()[0].lower()


def _load_tree() -> Tree:
    """Load species phylogeny tree (Newick format)."""
    with open(str(INSECT_TREE_PATH), "r") as f:
        newick_str = f.read()
    return Tree(newick_str, parser=1)


def _get_relation_label(target_clean: str, candidate_name: str) -> str:
    """Determine taxonomic relation between two species via NCBI taxonomy."""
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
    """Find closest usable binomial species relatives of the target.

    Returns: (candidates list, search mode, proxy species or None, target genus)
    """
    # Load tree
    try:
        t = _load_tree()
    except Exception as e:
        logger.error("Failed to load tree: %s", e)
        return [], "tree_load_error", None, ""

    clean_name = target_species.replace("_", " ").strip()
    formatted_target = clean_name.replace(" ", "_")
    target_genus = _extract_genus(clean_name)

    # All leaf nodes (ete4: Tree is iterable, yields leaf nodes)
    all_leaves = list(t)

    # Find target in tree (using raw node name with underscores)
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
        # Proxy fallback: walk up NCBI taxonomy to find a tree-present relative
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

    # Compute branch distances from target to all other leaf nodes
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
    # Filter: keep only binomial species (usable for RAG search)
    # ============================================================
    usable = [r for r in raw_results if _is_usable_species(r["display_name"])]

    if not usable:
        logger.error("No usable species-level relatives found in tree")
        return [], "no_usable_species", proxy_species, target_genus

    # ============================================================
    # Sort: congeneric first, then by mya ascending
    # ============================================================
    def sort_key(item: dict) -> tuple:
        same_genus = _extract_genus(item["display_name"]) == target_genus
        return (0 if same_genus else 1, item["mya"])

    usable.sort(key=sort_key)

    # Take top N
    top_n = usable[:target_total]

    # Annotate taxonomic relation
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
    """Species kinship calculation tool.

    Uses an insect phylogeny tree and NCBI taxonomy database to compute
    divergence time and taxonomic relation between a target insect species
    and other insect species.

    Returns only binomial species (Genus species), filtering out genus-level
    nodes and undescribed taxa so downstream RAG retrieval can search papers
    by species name directly.
    """

    name: str = "kinship"
    description: str = (
        "Species kinship tool. Input a target insect species (e.g. Bombyx_mori), "
        "returns the closest retrievable species list, each with standard species "
        "name (usable for literature search), divergence time (Mya), "
        "and taxonomic relation (Congeneric / Confamilial, etc.). "
        "Genus-level nodes and undescribed taxa are filtered out; "
        "output is ready for RAG retrieval."
    )
    args_schema: type = KinshipInput

    def _run(
        self,
        target_species: str,
        target_total: int = 8,
    ) -> str:
        """Compute species kinship (synchronous)."""

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

            # Format output
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
# Singleton export
# ============================================================
kinship_tool = KinshipTool()
