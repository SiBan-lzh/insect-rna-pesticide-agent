# Insect RNAi Pesticide Agent

A multi-agent system for designing RNAi-based insecticides. Built with LangGraph.

## Quick Install

```bash
# 1. Python dependencies
pip install -r requirements.txt

# 2. System binaries (Ubuntu/Debian)
sudo apt install -y ncbi-blast+ bedtools clustalo samtools

# 3. Verify
python tools_test.py
```

**Prerequisites:** Python 3.13+, Linux x86-64.

OligoWalk is pre-compiled and bundled under `database/RNAstructure/` — no separate install needed.

## Project Structure

```
langgraph/
├── agents/             # Single ReAct Agent (agent.py)
│   ├── __init__.py
│   └── agent.py        # Main agent + CLI entry point
├── skills/             # Domain skill documents (.skill)
│   ├── skill_loader.py # Auto-discover skills via frontmatter
│   ├── behavior/       # Behavioral standards
│   └── rnai/           # RNAi design workflows
├── tools/              # LangChain BaseTool wrappers (20 tools)
│   ├── tool_loader.py  # Auto-discover tools from subdirectory
│   ├── insect_blast/   # BLAST against target insect genomes
│   ├── nto_blast/      # BLAST against non-target organism genomes
│   ├── primer3/        # PCR primer design (primer3-py)
│   ├── oligowalk/      # siRNA thermodynamic scoring (OligoWalk)
│   ├── clustal/        # Pairwise alignment & off-target detection
│   ├── fetch_insect_cds/ # Fetch insect CDS by transcript ID
│   ├── fetch_nto_seq/  # Sequence extraction (samtools)
│   ├── kinship/        # Species phylogenetic relationship (ETE4)
│   ├── pubmed_esearch/ # PubMed literature search
│   ├── pubmed_efetch/  # PubMed article details
│   ├── openalex_search/ # OpenAlex academic search
│   ├── ...             # Plus discovery tools (list_tools, list_skills, etc.)
├── ragbase/            # RAG knowledge base (Chroma + BGE embeddings)
├── scripts/            # Utility scripts (build/remove RAG KB)
├── database/           # Reference data (BLAST DBs, GFF3, FASTA, etc.)
├── llm_config.py       # LLM factory (OpenAI-compatible)
├── tool_config.py      # Centralized path configuration reference
└── .env                # Environment variables (API keys, paths)
```

## Tools

| Tool | Binary / Library | Purpose |
|------|-----------------|---------|
| `insect_blast` | blastn (ncbi-blast+) | Homology search against target insect genomes |
| `nto_blast` | blastn (ncbi-blast+) | Off-target risk check against NTO genomes |
| `primer3` | primer3-py | PCR primer design with T7 promoter tails |
| `oligowalk` | OligoWalk (bundled) | siRNA thermodynamic scoring (sliding window) |
| `insect_anno` | bedtools | GFF3 gene feature annotation around BLAST hits |
| `clustal` | clustalo | Pairwise alignment & continuous match detection |
| `fetch_seq` | samtools | Extract NTO reference sequence regions |
| `fetch_insect_cds` | — (pure Python) | Fetch insect CDS by transcript ID |
| `kinship` | ETE4 + NCBI taxa | Species relatedness (divergence time, taxonomy) |

## Workflows

```
Target design pipeline:
  insect_blast → insect_anno → fetch_insect_cds → primer3 → oligowalk

Safety assessment pipeline:
  nto_blast → fetch_seq → clustal
  kinship → literature search (RAG) → nto_blast
```

## Quick Test

```bash
python tools_test.py
```

Expected: `16 通过 / 0 失败  🎉 全部通过！`

Test a single tool:

```python
from tools.insect_blast import insect_blast_tool

result = insect_blast_tool.invoke({
    "sequence": "ATGAAGCGGCAGAATGTACGA...",
    "species": "Bombyx_mori",
})
print(result)
```

## Configuration

Database paths are managed in `tool_config.py`. Override via environment variables:

```bash
export DATABASE_ROOT=/path/to/your/database
export INSECT_BLAST_DB=/path/to/blast/db
```
