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
├── tools/              # LangChain BaseTool wrappers (9 instances)
│   ├── __init__.py     # tool registry + Harness permission whitelists
│   ├── insect_blast.py # BLAST against target insect genomes
│   ├── nto_blast.py    # BLAST against non-target organism genomes
│   ├── primer3.py      # PCR primer design (primer3-py)
│   ├── oligowalk.py    # siRNA thermodynamic scoring (OligoWalk)
│   ├── insect_anno.py  # GFF3 gene annotation (bedtools)
│   ├── clustal.py      # pairwise alignment & off-target detection (clustalo)
│   ├── fetch_seq.py    # sequence extraction (samtools + CDS search)
│   └── kinship.py      # species phylogenetic relationship (ETE4)
├── database/           # reference data (BLAST DBs, GFF3, FASTA, tree, taxa)
├── tool_config.py      # centralized path configuration (env-var overridable)
├── tools_test.py       # integration test suite (16 test cases)
├── requirements.txt    # Python package dependencies
└── old_tools/          # original Docker/FastAPI implementations (reference only)
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
