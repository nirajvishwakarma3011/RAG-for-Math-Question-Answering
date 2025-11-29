#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import time
import csv
import statistics as stats
from typing import List
from tqdm import tqdm

import torch
import torch.nn.functional as F
from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification

from retriever_math import load_retriever
from prompt_math import build_prompt

# ---------------- CONFIG ----------------
MODEL_DIR              = "/home/ankitam/dlnlp_rag/models/base_full_sft"
TEST_DATA_PATH         = "/home/ankitam/dlnlp_rag/data/test"
STORE_DIR              = "store"
RETRIEVER              = "hybrid"     # "bm25" | "dense" | "hybrid"
TOP_K                  = 5

# token budgets
MAX_INPUT_TOKENS       = 3072
CTX_BUDGET_TOKENS      = 1800
MAX_NEW_TOKENS         = 1024

# decoding
TEMPERATURE            = 0.9
TOP_P                  = 0.9

# batching / sampling
NUM_SAMPLES_PER_PROMPT = 3
GEN_BATCH_SIZE         = 64
SEED_BASE              = 1000

# subset for quick testing
LIMIT_N                = 1    # set None to run the full test set

# metrics toggles
COMPUTE_BERT           = True        # requires: pip install bert-score
COMPUTE_BLEU           = True        # requires: pip install sacrebleu

# Verifier (MNLI-style NLI)
VERIFIER_MODEL_NAME    = os.path.join(os.path.dirname(MODEL_DIR), "verifier")  # adjust to your path
VERIFIER_DEVICE        = "cuda:0"   # e.g., "cuda:0" or "cpu"
ENTAILMENT_THRESH      = 0.5        # require ENTAILMENT prob >= threshold to vote "Yes"

# outputs
OUT_CSV                = "math_rag_plus_ft_results.csv"
OUT_SUMMARY_CSV        = "math_rag_plus_ft_results_summary.csv"
# ---------------------------------------


# --------------- Helpers ---------------
def load_llm_and_tokenizer(model_dir: str):
    tok = AutoTokenizer.from_pretrained(model_dir, use_fast=False, padding_side="left")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    model.eval()
    return model, tok

# @torch.no_grad()
# def batched_generate(model, tokenizer, prompts: List[str], max_new_tokens=128, temperature=0.7, top_p=0.9, seed=0):
#     """
#     Batch-generate for a list of prompts (<= GEN_BATCH_SIZE).
#     Returns (decoded_texts, per_sample_latency_seconds list).
#     """
#     # seed for reproducibility per pass
#     torch.manual_seed(seed)
#     if torch.cuda.is_available():
#         torch.cuda.manual_seed_all(seed)

#     inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(model.device)
#     t0 = time.perf_counter()
#     out = model.generate(
#         **inputs,
#         do_sample=True,
#         temperature=temperature,
#         top_p=top_p,
#         max_new_tokens=max_new_tokens,
#         pad_token_id=tokenizer.pad_token_id,
#     )
#     batch_latency = time.perf_counter() - t0
#     decoded = [tokenizer.decode(o, skip_special_tokens=True) for o in out]

#     # strip prompt prefix when possible
#     results = []
#     for prompt, text in zip(prompts, decoded):
#         if text.startswith(prompt):
#             results.append(text[len(prompt):].strip())
#         else:
#             results.append(text.strip())

#     per_sample_latency = batch_latency / max(1, len(prompts))
#     latencies = [per_sample_latency] * len(prompts)
#     return results, latencies


@torch.no_grad()
def batched_generate(model, tokenizer, prompts, max_new_tokens=128, temperature=0.7, top_p=0.9, seed=0):
    inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(model.device)
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))

    t0 = time.perf_counter()
    out = model.generate(
        **inputs,
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.pad_token_id,
    )
    batch_latency = time.perf_counter() - t0

    attn = inputs["attention_mask"]
    input_lens = attn.sum(dim=1).tolist()

    import re
    pat = re.compile(r'(?i)\banswer\s*:\s*')

    results = []
    for i in range(out.size(0)):
        cont = tokenizer.decode(out[i, input_lens[i]:], skip_special_tokens=True)
        m = pat.search(cont)
        results.append(cont[m.end():].strip() if m else cont.strip())

    per_sample_latency = batch_latency / max(1, len(prompts))
    return results, [per_sample_latency] * len(prompts)




def maybe_compute_bertscore(preds: List[str], refs: List[str], model_type="roberta-large"):
    if not COMPUTE_BERT:
        return [None] * len(preds)
    from bert_score import score as bert_score
    _, _, F = bert_score(preds, refs, lang="en", model_type=model_type, verbose=False)
    return [float(f) for f in F]

def maybe_compute_bleu_pairwise(preds: List[str], refs: List[str]):
    if not COMPUTE_BLEU:
        return [None] * len(preds), [None] * len(preds)
    import sacrebleu
    bleu_100, bleu_norm = [], []
    for p, r in zip(preds, refs):
        try:
            s = sacrebleu.sentence_bleu(
                p, [r],
                smooth_method="exp",
                smooth_value=None,
                tokenize="13a"
            ).score
        except Exception:
            s = 0.0
        bleu_100.append(s)
        bleu_norm.append(s / 100.0)
    return bleu_100, bleu_norm

# --- Verifier utilities (MNLI) ---
def load_verifier(model_name: str, device: str = "cuda"):
    vtok = AutoTokenizer.from_pretrained(model_name)
    vmdl = AutoModelForSequenceClassification.from_pretrained(model_name)
    vmdl.to(device)
    vmdl.eval()
    return vmdl, vtok

@torch.no_grad()
def verifier_batch_predict(verifier, tokenizer, preds: List[str], golds: List[str], device: str = "cpu", batch_size: int = 32):
    """
    Run verifier in batches on pairs (preds[i], golds[i]).
    Returns list of (label_str, prob_dict) where label_str is top label name.
    """
    results = []
    id2label = verifier.config.id2label if hasattr(verifier.config, "id2label") else {0: "LABEL_0", 1: "LABEL_1", 2: "LABEL_2"}
    for i in range(0, len(preds), batch_size):
        p_chunk = preds[i:i+batch_size]
        g_chunk = golds[i:i+batch_size]
        inputs = tokenizer(p_chunk, g_chunk, return_tensors="pt", padding=True, truncation=True).to(device)
        logits = verifier(**inputs).logits  # (B, num_labels)
        probs = F.softmax(logits, dim=-1).detach().cpu().tolist()
        for prob in probs:
            top_idx = int(max(range(len(prob)), key=lambda j: prob[j]))
            label = id2label[top_idx]
            prob_dict = {id2label[j]: prob[j] for j in range(len(prob))}
            results.append((label, prob_dict))
    return results
# ---------------------------------------


# ----------------- Main -----------------
def main():
    # load dataset
    print("Loading test dataset from:", TEST_DATA_PATH)
    ds = load_from_disk(TEST_DATA_PATH)
    if hasattr(ds, "keys"):  # DatasetDict
        ds = ds["test"] if "test" in ds else next(iter(ds.values()))

    n_total = len(ds)
    if LIMIT_N is not None and LIMIT_N > 0 and n_total > LIMIT_N:
        ds = ds.select(range(LIMIT_N))
    n = len(ds)
    print(f"Test size (after LIMIT_N): {n} / {n_total}")

    # build Q/A lists
    Qs, As = [], []
    for i in range(n):
        row = ds[i]
        q = (row.get("Q") or row.get("question") or "").strip()
        a = (row.get("A") or row.get("answer") or "").strip()
        Qs.append(q)
        As.append(a)

    assert any(len(q.strip()) > 0 for q in Qs), "All questions are empty—column mapping is wrong."
    assert any(len(a.strip()) > 0 for a in As), "All gold answers are empty—column mapping is wrong."

    # load retriever + LLM
    print(f"Loading retriever ({RETRIEVER}) from: {STORE_DIR}")
    retriever = load_retriever(STORE_DIR, RETRIEVER)

    print("Loading LLM model and tokenizer...")
    llm_model, llm_tokenizer = load_llm_and_tokenizer(MODEL_DIR)
    device = next(llm_model.parameters()).device
    print("LLM device:", device)

    # load verifier
    print("Loading verifier:", VERIFIER_MODEL_NAME)
    verifier, verifier_tok = load_verifier(VERIFIER_MODEL_NAME, device=VERIFIER_DEVICE)

    # 1) Retrieve once for all Qs and build prompts
    print("Retrieving top-k examples for all questions...")
    all_hits = []
    prompts = []
    ctx_token_counts = []

    t0 = time.perf_counter()
    for q in tqdm(Qs, desc="Retrieve"):
        hits = retriever.search(q, TOP_K)  # similarity on QUESTION only
        all_hits.append(hits)

        pr = build_prompt(llm_tokenizer, q, hits, MAX_INPUT_TOKENS, CTX_BUDGET_TOKENS)  # includes (Q,A) exemplars
        prompts.append(pr)

        # record approximate token count of full prompt
        ctx_token_counts.append(len(llm_tokenizer.encode(pr, add_special_tokens=False)))
    retrieval_latency_avg = (time.perf_counter() - t0) / max(1, n)

    # 2) Generate NUM_SAMPLES_PER_PROMPT passes (like your vanilla harness)
    print(f"Generating {NUM_SAMPLES_PER_PROMPT} samples per prompt...")
    all_preds = [[""] * NUM_SAMPLES_PER_PROMPT for _ in range(n)]
    all_lat   = [[0.0] * NUM_SAMPLES_PER_PROMPT for _ in range(n)]

    for pass_i in range(NUM_SAMPLES_PER_PROMPT):
        print(f"Pass {pass_i+1}/{NUM_SAMPLES_PER_PROMPT}")
        for batch_start in tqdm(range(0, n, GEN_BATCH_SIZE), desc="Batched gen"):
            batch_end = min(batch_start + GEN_BATCH_SIZE, n)
            batch_prompts = prompts[batch_start:batch_end]
            seed = SEED_BASE + pass_i * 100000 + (batch_start // GEN_BATCH_SIZE)

            gen_texts, gen_lats = batched_generate(
                llm_model,
                llm_tokenizer,
                batch_prompts,
                max_new_tokens=MAX_NEW_TOKENS,
                temperature=TEMPERATURE,
                top_p=TOP_P,
                seed=seed
            )
            for j, (txt, lat) in enumerate(zip(gen_texts, gen_lats)):
                idx = batch_start + j
                all_preds[idx][pass_i] = txt
                all_lat[idx][pass_i] = lat

    # 3) Metrics (BERT/BLEU) over flattened lists
    flat_preds = [p for preds in all_preds for p in preds]
    flat_refs  = [g for _ in range(NUM_SAMPLES_PER_PROMPT) for g in As]

    bert_fs_flat = maybe_compute_bertscore(flat_preds, flat_refs)
    bleu_flat, bleu_norm_flat = maybe_compute_bleu_pairwise(flat_preds, flat_refs)

    # reshape to per-example lists
    if COMPUTE_BERT:
        bert_per_example = [bert_fs_flat[i*NUM_SAMPLES_PER_PROMPT:(i+1)*NUM_SAMPLES_PER_PROMPT] for i in range(n)]
    else:
        bert_per_example = [[None]*NUM_SAMPLES_PER_PROMPT for _ in range(n)]

    if COMPUTE_BLEU:
        bleu_per_example      = [bleu_flat[i*NUM_SAMPLES_PER_PROMPT:(i+1)*NUM_SAMPLES_PER_PROMPT] for i in range(n)]
        bleu_norm_per_example = [bleu_norm_flat[i*NUM_SAMPLES_PER_PROMPT:(i+1)*NUM_SAMPLES_PER_PROMPT] for i in range(n)]
    else:
        bleu_per_example      = [[None]*NUM_SAMPLES_PER_PROMPT for _ in range(n)]
        bleu_norm_per_example = [[None]*NUM_SAMPLES_PER_PROMPT for _ in range(n)]

    # 4) Verifier on (pred, gold), flattened then reshaped
    print("Running batched verifier (MNLI)...")
    verifier_results = verifier_batch_predict(
        verifier, verifier_tok,
        flat_preds, flat_refs,
        device=VERIFIER_DEVICE, batch_size=64
    )
    ver_labels = [verifier_results[i*NUM_SAMPLES_PER_PROMPT:(i+1)*NUM_SAMPLES_PER_PROMPT] for i in range(n)]

    # 5) Write CSV (reference-compatible + RAG columns + verifier)
    os.makedirs(os.path.dirname(OUT_CSV) or ".", exist_ok=True)
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        header = [
            "idx", "question_raw", "question_prompt", "gold",
            "pred_1", "latency_1", "bert_1", "bleu_1", "bleu_norm_1",
            "pred_2", "latency_2", "bert_2", "bleu_2", "bleu_norm_2",
            "pred_3", "latency_3", "bert_3", "bleu_3", "bleu_norm_3",
            "avg_latency", "avg_bert_f1", "avg_bleu", "avg_bleu_norm",
            "verifier_votes", "verifier_probs",
            # RAG logging
            "retriever", "top_k", "avg_retrieval_latency_sec",
            "prompt_tokens_est",
            "retrieved_doc_ids", "retrieval_scores"
        ]
        w.writerow(header)

        avg_bert_list, avg_bleu_list, avg_bleu_norm_list, avg_latency_list, yes_rate_list = [], [], [], [], []

        for i in tqdm(range(n), desc="Write rows"):
            preds = all_preds[i]
            lats  = all_lat[i]
            berts = bert_per_example[i]
            bleus = bleu_per_example[i]
            bleus_norm = bleu_norm_per_example[i]

            # averages (ignore None)
            def mean_or_none(vals):
                vals = [v for v in vals if v is not None]
                return sum(vals)/len(vals) if vals else None

            avg_latency   = sum(lats) / len(lats)
            avg_bert      = mean_or_none(berts)
            avg_bleu      = mean_or_none(bleus)
            avg_bleu_norm = mean_or_none(bleus_norm)

            if avg_bert is not None:      avg_bert_list.append(avg_bert)
            if avg_bleu is not None:      avg_bleu_list.append(avg_bleu)
            if avg_bleu_norm is not None: avg_bleu_norm_list.append(avg_bleu_norm)
            avg_latency_list.append(avg_latency)

            # verifier votes per item
            ver_chunk = ver_labels[i]  # list of tuples (label, prob_dict)
            votes, probs = [], []
            for (label, prob_dict) in ver_chunk:
                ent_prob = prob_dict.get("ENTAILMENT") or prob_dict.get("entailment") or 0.0
                if (label.upper() == "ENTAILMENT") and (ent_prob >= ENTAILMENT_THRESH):
                    votes.append("Yes")
                else:
                    votes.append("No")
                probs.append(ent_prob)
            yes_rate = sum(1 for v in votes if v == "Yes") / max(1, len(votes))
            yes_rate_list.append(yes_rate)

            hits = all_hits[i]
            doc_ids_str = " | ".join(h[0] for h in hits)
            scores_str  = " | ".join(f"{h[1]:.4f}" for h in hits)

            row = [
                i, Qs[i], prompts[i], As[i],
                preds[0] if len(preds) > 0 else "", lats[0] if len(lats) > 0 else "", (berts[0] if berts[0] is not None else ""), (bleus[0] if bleus[0] is not None else ""), (bleus_norm[0] if bleus_norm[0] is not None else ""),
                preds[1] if len(preds) > 1 else "", lats[1] if len(lats) > 1 else "", (berts[1] if len(berts) > 1 and berts[1] is not None else ""), (bleus[1] if len(bleus) > 1 and bleus[1] is not None else ""), (bleus_norm[1] if len(bleus_norm) > 1 and bleus_norm[1] is not None else ""),
                preds[2] if len(preds) > 2 else "", lats[2] if len(lats) > 2 else "", (berts[2] if len(berts) > 2 and berts[2] is not None else ""), (bleus[2] if len(bleus) > 2 and bleus[2] is not None else ""), (bleus_norm[2] if len(bleus_norm) > 2 and bleus_norm[2] is not None else ""),
                avg_latency, (avg_bert if avg_bert is not None else ""), (avg_bleu if avg_bleu is not None else ""), (avg_bleu_norm if avg_bleu_norm is not None else ""),
                " | ".join(votes),
                " | ".join(f"{p:.3f}" for p in probs),
                RETRIEVER, TOP_K, f"{retrieval_latency_avg:.4f}",
                ctx_token_counts[i],
                doc_ids_str, scores_str
            ]
            w.writerow(row)

    print("Saved:", OUT_CSV)

    # 6) Summary CSV
    def mean_std(x):
        if not x:
            return 0.0, 0.0
        return float(stats.mean(x)), (float(stats.pstdev(x)) if len(x) > 1 else 0.0)

    m_lat, s_lat = mean_std(avg_latency_list)
    m_bert, s_bert = mean_std(avg_bert_list) if avg_bert_list else (0.0, 0.0)
    m_bleu, s_bleu = mean_std(avg_bleu_list) if avg_bleu_list else (0.0, 0.0)
    m_bleu_n, s_bleu_n = mean_std(avg_bleu_norm_list) if avg_bleu_norm_list else (0.0, 0.0)
    m_yes, s_yes = mean_std(yes_rate_list)

    os.makedirs(os.path.dirname(OUT_SUMMARY_CSV) or ".", exist_ok=True)
    with open(OUT_SUMMARY_CSV, "w", newline="", encoding="utf-8") as fsum:
        w = csv.writer(fsum)
        w.writerow(["key", "value_mean", "value_std", "notes"])
        w.writerow(["num_examples", n, 0, ""])
        w.writerow(["samples_per_prompt", NUM_SAMPLES_PER_PROMPT, 0, ""])
        w.writerow(["avg_latency_sec", m_lat, s_lat, "Per-example mean of per-sample latency"])
        w.writerow(["avg_bert_f1", m_bert, s_bert, "BERTScore F1; requires COMPUTE_BERT=True"])
        w.writerow(["avg_bleu", m_bleu, s_bleu, "Sentence BLEU (0..100); requires COMPUTE_BLEU=True"])
        w.writerow(["avg_bleu_norm", m_bleu_n, s_bleu_n, "Sentence BLEU normalized (0..1)"])
        w.writerow(["verifier_yes_rate", m_yes, s_yes, f"ENTAILMENT_THRESH={ENTAILMENT_THRESH}"])
        w.writerow(["retriever", RETRIEVER, 0, ""])
        w.writerow(["top_k", TOP_K, 0, ""])
        w.writerow(["avg_retrieval_latency_sec", retrieval_latency_avg, 0, "mean per question"])
        w.writerow(["limit_n", (LIMIT_N if LIMIT_N is not None else -1), 0, ""])
        w.writerow(["max_input_tokens", MAX_INPUT_TOKENS, 0, ""])
        w.writerow(["ctx_budget_tokens", CTX_BUDGET_TOKENS, 0, ""])
        w.writerow(["max_new_tokens", MAX_NEW_TOKENS, 0, ""])
    print("Saved:", OUT_SUMMARY_CSV)
    print("Done.")

if __name__ == "__main__":
    main()
