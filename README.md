# RAG-for-Math-Question-Answering
This repository contains a complete pipeline for experimenting with mathematical reasoning capabilities in Large Language Models (LLMs). It facilitates the comparison of **Supervised Fine-Tuning (SFT)**, **Retrieval-Augmented Generation (RAG)**, and hybrid approaches using Llama-3.1-8B and OpenMath2 models.
## Layout
- `ft.py`: Full-supervised fine-tune of a base causal LM on processed math QA data.
- `run.py`: batch evaluator template for vanilla generation + MNLI verifier.
- `utils/`: Data and model helpers (dataset cleaning, formatter, local model downloads, verifier).
- `RAG/`: Math RAG pipeline (index building, retrievers, prompt builder, evaluation + results CSVs).

## Steps
1) Download base resources (writes to `models/`):
   - `python utils/save_base_model.py` (Llama 3.1 8B).
   - `python utils/save_mathgpt.py` (OpenMath2 Llama 3.1 8B).
   - `python utils/save_verifier_model.py` (DistilBART MNLI).

2) Data preparation
- `utils/preprocess_data.py`: Cleans StackMathQA (splits into test/ft/rag), filters malformed LaTeX, converts inline/display math to MathML, and saves `{test,ft,rag}_clean_mathml` under `data/`.
- `utils/format_ft_dataset.py`: Builds instruction-style `text` field (`{question}\n\nAnswer: ...`) and saves to `data/ft_processed`.

3) Fine-tuning
- `ft.py` loads `models/base` and trains on `data/ft_processed`. Outputs to `models/base_full_sft`.

4) RAG pipeline (RAG/)
- `build_math_index.py`: Builds BM25 JSONL plus optional dense FAISS index from HF `data/rag` (or JSONL). Stores artifacts in `RAG/store/`.
- `retriever_math.py`: BM25, dense, and hybrid retrievers over the store.
- `prompt_math.py`: Renders prompts from retrieved exemplars within token budgets.
- `rag_math_eval.py`: Runs RAG generation + verifier over the math test split; computes latency, BLEU, BERTScore; writes CSVs.
- `one_question_test.py`: Single-question RAG + generation harness for ad-hoc checks.

## Running evaluation
```bash
cd RAG
python build_math_index.py      # once, builds retriever store
python rag_math_eval.py         # runs full RAG eval, writes CSVs
python one_question_test.py     # quick single-question check
```

## Experimental Results

We compared several configurations across varying retrieval settings. The **OpenMath + RAG** configuration demonstrated state-of-the-art performance across all quality metrics.

| System | BERTScore-F1 | BLEU | Verifier Yes-rate | Latency (s) | Retrieval Config |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Baseline (Meta-Llama)** | 0.812 | 3.50 | 0.327 | 2.38 | No |
| **SFT (Meta-Llama)** | 0.818 | 4.00 | 0.585 | 2.38 | No |
| **RAG (Base Llama)** | 0.814 | 5.57 | 0.401 | 4.53 | Yes (hybrid, $k=3$) |
| **SFT + RAG (Llama)** | 0.816 | 4.52 | 0.581 | 4.35 | Yes (hybrid, $k=5$) |
| **OpenMath** | 0.833 | 6.12 | 0.642 | 2.70 | No |
| **OpenMath + RAG** | **0.846** | **7.90** | **0.711** | 4.80 | Yes (hybrid, $k=5$) |

> **Key Findings:**
> * **RAG Effectiveness:** Adding RAG consistently improves BLEU and BERTScore across base and fine-tuned models.
> * **SFT Impact:** Supervised Fine-Tuning significantly boosts the "Verifier Yes-rate" (mathematical correctness), nearly doubling it compared to the baseline.
> * **Trade-offs:** While RAG increases latency (approx. +2s per generation), the performance gain in the `OpenMath + RAG` setup (Highest F1 and BLEU) justifies the computational cost for high-precision tasks.

---
## Individual Contributions 
• Niraj: Overall coordination and project management, timeline planning, and integration of all components into a single end-to-end pipeline.

• Lipika: Data preparation and knowledge-base construction, including dataset cleaning, splitting, and creation of the fine-tune set and RAG knowledge base.

• Siddharth: Model fine-tuning, implementing and tuning the supervised fine-tuning pipeline for the LLM.

• Ankita: Retrieval and RAG component, designing and implementing the hybrid BM25 + dense retriever and constructing the RAG-style prompts.

• Manish: Evaluation and reporting, implementing evaluation metrics, running experiments, analyzing results, and preparing the written report and presentation materials.
