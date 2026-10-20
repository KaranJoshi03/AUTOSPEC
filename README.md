# Predii Automotive Vehicle Specification Extraction Pipeline

## Executive Summary
This repository contains a production-grade, zero-cost ($0.00 budget) RAG pipeline tailored for automotive workshop service manuals.

- **Dataset**: /kaggle/input/datasets/karan2528/predii-service-manual/Dataset_servicemanual.pdf (852 pages, 43.6 MB)
- **Chunks**: 1,053 semantic chunks with markdown table preservation
- **Embeddings**: BGE-large-en-v1.5 (1024d) + Nomic-embed-v1.5 (768d)
- **Vector Search**: FAISS IndexFlatIP + Lexical BM25 Okapi + RRF Fusion (k=60)
- **Reranker**: BAAI/bge-reranker-v2-m3 Cross-Encoder
- **Extraction**: Groq Qwen-2.5 27B with multi-tier parsing and fallback
- **Total Specs Extracted**: 689
- **Test Suite**: 20/20 Tests Passed (100.0%)

## Deliverables Generated
- `specs.json`: 689 extracted specifications with page citations
- `specs.csv`: Structured tabular specifications export
- `metrics_report.csv`: Complete 123-query benchmark evaluation report
- `ablation_results.csv`: 54-run architectural ablation matrix
- `proof_report.html`: Interactive dark-mode proof certificate
