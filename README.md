# Insect RNAi Pesticide Agent

> **v0.0.3** — Modular tools & skills, topic-isolated memory, shell safety, skill-based workflow guides

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

## Quick Start

```bash
# Start the interactive CLI
python agents/agent.py
```

The agent will run startup checks automatically, then present an interactive prompt.

## Project Structure

```
langgraph/
├── agents/             # Single ReAct Agent (agent.py)
│   ├── __init__.py
│   └── agent.py        # Main agent + CLI entry point
├── skills/             # Domain skill documents (.skill)
│   ├── skill_loader.py # Auto-discover skills via frontmatter
│   ├── behavior/       # Behavioral standards (analysis-standards, shell, memory)
│   └── rnai/           # RNAi design workflows (dsrna_design, safety_inspection, target_find)
├── tools/              # LangChain BaseTool wrappers (24 tools, modular subdirectories)
│   ├── tool_loader.py  # Auto-discover tools from subdirectories
│   ├── shell/          # Shell execution with safety classification
│   ├── save_memory/    # Topic-isolated long-term memory (save)
│   ├── load_memory/    # Topic-isolated long-term memory (load)
│   ├── list_tools/     # Tool discovery
│   ├── list_skills/    # Skill document discovery
│   ├── list_database/  # Local genome data discovery
│   ├── list_ragbase/   # RAG knowledge base discovery
│   ├── read_skill/     # Load skill document content
│   ├── search_knowledge/ # RAG knowledge base search
│   ├── clean_seq/      # Sequence cleaning
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
│   └── ...             # Plus insect_anno, clip_seq
├── ragbase/            # RAG knowledge base (Chroma + BGE embeddings)
├── scripts/            # Utility scripts (build/remove RAG KB)
├── tests/              # Environment checks and config validation
├── benchmark/          # Workflow benchmark records
├── database/           # Reference data (BLAST DBs, GFF3, FASTA, etc.)
├── llm_config.py       # LLM factory (OpenAI-compatible)
├── .env                # Environment variables (API keys, paths)
└── workspace/          # User-facing output files
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
| `fetch_nto_seq` | samtools | Extract NTO reference sequence regions |
| `fetch_insect_cds` | — (pure Python) | Fetch insect CDS by transcript ID |
| `kinship` | ETE4 + NCBI taxa | Species relatedness (divergence time, taxonomy) |
| `shell` | — | Shell execution with read/write/blocked classification |
| `save_memory` | ChromaDB | Topic-isolated long-term memory save |
| `load_memory` | ChromaDB | Topic-isolated long-term memory load |

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
# Environment check
python tests/config_check.py
```

## Configuration

Configure via `.env` file at the project root:

```bash
# LLM configuration
MODEL_PROVIDER=openai
MODEL_NAME=gpt-4o
API_KEY=sk-...
```
