---
name: clean_seq
tag: sequence processing
description: "Normalize and validate raw nucleotide sequences. Always call this as the FIRST step when a user provides a sequence. Accepts FASTA format (with '>' header) or plain sequences. Removes whitespace/newlines, converts to single-line uppercase, validates against DNA (ATCGN) or RNA (AUCGN) alphabet, and reports invalid characters with exact bp positions. DNA mode converts U->T; RNA mode converts T->U."
---
