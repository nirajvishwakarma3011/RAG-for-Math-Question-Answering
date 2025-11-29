# """
# Batched evaluation of a local LLaMA model on a saved test split.
# - Generates NUM_SAMPLES_PER_PROMPT samples per example (by running NUM_SAMPLES_PER_PROMPT passes).
# - Uses batched model.generate calls for throughput.
# - Uses a separate MNLI verifier in BATCHED mode; interprets ENTAILMENT with threshold as "Yes".

# Requirements:
#   pip install transformers datasets torch bert-score sacrebleu tqdm
# """

# import os
# import time
# import csv
# from typing import List
# from math import ceil
# from tqdm import tqdm

# import torch
# import torch.nn.functional as F
# from transformers import (
#     AutoTokenizer,
#     AutoModelForCausalLM,
#     AutoModelForSequenceClassification,
# )
# from datasets import load_from_disk

# # metrics
# from bert_score import score as bert_score
# import sacrebleu

# # ---------------- CONFIG ----------------
# PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
# # MODEL_DIR = os.path.join(PROJECT_ROOT, "models", "base"). ## for base model
# MODEL_DIR = os.path.join(PROJECT_ROOT, "models", "base_full_sft") ## for fine tuned
# VERIFIER_MODEL_NAME = os.path.join(PROJECT_ROOT, "models", "verifier")
# TEST_DATA_PATH = os.path.join(PROJECT_ROOT, "data", "test")

# OUTPUT_CSV = os.path.join(PROJECT_ROOT, "eval", "vanilla", "results_finetuned.csv")

# NUM_SAMPLES_PER_PROMPT = 3
# GEN_BATCH_SIZE = 64              # number of examples processed per generate() call
# MAX_NEW_TOKENS = 1024
# TEMPERATURE = 0.9
# TOP_P = 0.9
# SEED_BASE = 1000

# PROMPT_TEMPLATE = "{question}\n\nAnswer:"

# # Verifier (MNLI)
# VERIFIER_DEVICE = "cuda:0"         # "cuda" if you want verifier on GPU
# ENTAILMENT_THRESH = 0.4        # require ENTAILMENT prob > threshold to say "Yes"
# # ----------------------------------------

# # ------------------ Helpers ------------------
# def load_llm_and_tokenizer(model_dir: str):
#     tok = AutoTokenizer.from_pretrained(model_dir, local_files_only=True, use_fast=False, padding_side='left')
#     model = AutoModelForCausalLM.from_pretrained(model_dir, local_files_only=True, torch_dtype=torch.float16, device_map="auto")
#     if tok.pad_token is None:
#         # simplest: reuse eos token as pad
#         tok.pad_token = tok.eos_token
#     model.eval()
#     return model, tok

# def batched_generate(model, tokenizer, prompts: List[str], max_new_tokens=128, temperature=0.7, top_p=0.9, seed=0):
#     """
#     Batch generate for a list of prompts (size = GEN_BATCH_SIZE or less).
#     Returns list of generated strings and list of latencies (per-batch averaged to per-sample).
#     We create a single torch.Generator seeded with seed so results are reproducible per-call.
#     """
#     inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(model.device)
#     torch.manual_seed(seed)
#     if torch.cuda.is_available():
#         torch.cuda.manual_seed_all(seed)
#     t0 = time.perf_counter()
#     with torch.no_grad():
#         out = model.generate(
#             **inputs,
#             do_sample=True,
#             temperature=temperature,
#             top_p=top_p,
#             max_new_tokens=max_new_tokens,
#             pad_token_id=tokenizer.pad_token_id,
#         )
#     batch_latency = time.perf_counter() - t0
#     # decode each output and strip prompt prefix when possible
#     decoded = [tokenizer.decode(o, skip_special_tokens=True) for o in out]
#     results = []
#     for prompt, text in zip(prompts, decoded):
#         if text.startswith(prompt):
#             results.append(text[len(prompt):].strip())
#         else:
#             results.append(text.strip())
#     # return generated list and latency-per-sample (we'll record batch_latency / len(prompts))
#     per_sample_latency = batch_latency / max(1, len(prompts))
#     latencies = [per_sample_latency] * len(prompts)
#     return results, latencies

# def batchify(lst, batch_size):
#     for i in range(0, len(lst), batch_size):
#         yield lst[i:i + batch_size]

# # Verifier utilities (batched)
# def load_verifier(model_name: str, device: str = "cpu"):
#     verifier_tok = AutoTokenizer.from_pretrained(model_name)
#     verifier = AutoModelForSequenceClassification.from_pretrained(model_name)
#     verifier.to(device)
#     verifier.eval()
#     return verifier, verifier_tok

# def verifier_batch_predict(verifier, tokenizer, preds: List[str], golds: List[str], device: str = "cpu", batch_size: int = 32):
#     """
#     Run verifier in batches on pairs (preds[i], golds[i]).
#     Returns list of (label_str, prob_dict) where label_str is top label name.
#     """
#     results = []
#     id2label = verifier.config.id2label if hasattr(verifier.config, "id2label") else {0: "LABEL_0", 1: "LABEL_1", 2: "LABEL_2"}
#     for i in range(0, len(preds), batch_size):
#         p_chunk = preds[i:i+batch_size]
#         g_chunk = golds[i:i+batch_size]
#         inputs = tokenizer(p_chunk, g_chunk, return_tensors="pt", padding=True, truncation=True).to(device)
#         with torch.no_grad():
#             logits = verifier(**inputs).logits  # (B, num_labels)
#             probs = F.softmax(logits, dim=-1).cpu().tolist()  # list of lists
#         for prob in probs:
#             top_idx = int(max(range(len(prob)), key=lambda j: prob[j]))
#             label = id2label[top_idx]
#             prob_dict = {id2label[j]: prob[j] for j in range(len(prob))}
#             results.append((label, prob_dict))
#     return results

# # Metric computations (BERTScore / BLEU)
# def compute_bert_scores(preds: List[str], refs: List[str], model_type="roberta-large"):
#     P, R, F = bert_score(preds, refs, lang="en", model_type=model_type, verbose=False)
#     return [float(f) for f in F]

# def compute_bleu(preds: List[str], refs: List[str]):
#     try:
#         bleu = sacrebleu.corpus_bleu(preds, [refs])
#         return [bleu.score for _ in preds]
#     except Exception:
#         out = []
#         for p, r in zip(preds, refs):
#             try:
#                 score = sacrebleu.sentence_bleu(p, [r]).score
#             except Exception:
#                 score = 0.0
#             out.append(score)
#         return out

# # ------------------ Main ------------------
# def main():
#     # load LLM & tokenizer
#     print("Loading LLM model and tokenizer...")
#     llm_model, llm_tokenizer = load_llm_and_tokenizer(MODEL_DIR)
#     device = next(llm_model.parameters()).device
#     print("LLM device:", device)

#     # load verifier
#     print("Loading verifier:", VERIFIER_MODEL_NAME)
#     verifier, verifier_tok = load_verifier(VERIFIER_MODEL_NAME, device=VERIFIER_DEVICE)

#     # load dataset
#     print("Loading test dataset from:", TEST_DATA_PATH)
#     ds_test = load_from_disk(TEST_DATA_PATH)
#     if hasattr(ds_test, "keys"):
#         ds_test = ds_test["test"] if "test" in ds_test else next(iter(ds_test.values()))
#     LIMIT_N = 10  # smoke test size
#     if LIMIT_N is not None and LIMIT_N > 0 and len(ds_test) > LIMIT_N:
#         ds_test = ds_test.select(range(LIMIT_N))  # first 100
#     n = len(ds_test)
#     print(f"Test size: {n}")

#     # build prompt list (preserve order)
#     # questions = []
#     # golds = []
#     # for idx in range(n):
#     #     row = ds_test[idx]
#     #     q = row.get("question") or row.get("prompt") or ""
#     #     g = row.get("answer") or row.get("target") or ""
#     #     questions.append(PROMPT_TEMPLATE.format(question=q))
#     #     golds.append(g)

#     ##Prompt building 
#     questions = []
#     golds = []
#     for idx in range(n):
#         row = ds_test[idx]

#         # tolerant getters for multiple dataset schemas
#         q = (
#             (row.get("question") if isinstance(row, dict) else None)
#             or (row.get("prompt")   if isinstance(row, dict) else None)
#             or (row.get("Q")        if isinstance(row, dict) else None)
#             or row["Q"]  if "Q" in ds_test.features else ""
#         )

#         g = (
#             (row.get("answer") if isinstance(row, dict) else None)
#             or (row.get("target") if isinstance(row, dict) else None)
#             or (row.get("A")      if isinstance(row, dict) else None)
#             or row["A"]  if "A" in ds_test.features else ""
#         )

#         # final string sanitation
#         q = (q or "").strip()
#         g = (g or "").strip()

#         questions.append(PROMPT_TEMPLATE.format(question=q))
#         golds.append(g)

#     # (optional) quick sanity check
#     print("Sample prompt:", questions[0][:2].replace("\n", " ⏎ "))
#     print("Sample gold  :", golds[0][:2].replace("\n", " ⏎ "))

#     assert any(len(x.strip()) > 0 for x in questions), "All questions are empty—column mapping is wrong."
#     assert any(len(x.strip()) > 0 for x in golds),     "All golds are empty—column mapping is wrong."


#     # Prepare CSV writer
#     os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
#     csv_file = open(OUTPUT_CSV, "w", newline="", encoding="utf-8")
#     writer = csv.writer(csv_file)
#     header = [
#         "idx", "question", "gold",
#         "pred_1", "latency_1", "bert_1", "bleu_1",
#         "pred_2", "latency_2", "bert_2", "bleu_2",
#         "pred_3", "latency_3", "bert_3", "bleu_3",
#         "avg_latency", "avg_bert_f1", "avg_bleu",
#         "verifier_votes", "verifier_probs"
#     ]
#     writer.writerow(header)

#     # Pre-allocate storage for preds and latencies: list of lists per example
#     all_preds = [[""] * NUM_SAMPLES_PER_PROMPT for _ in range(n)]
#     all_lat = [[0.0] * NUM_SAMPLES_PER_PROMPT for _ in range(n)]

#     # For throughput: run NUM_SAMPLES_PER_PROMPT passes; each pass generates one sample per example in batches
#     for pass_i in range(NUM_SAMPLES_PER_PROMPT):
#         print(f"Generation pass {pass_i+1}/{NUM_SAMPLES_PER_PROMPT}")
#         # iterate in batches
#         for batch_start in tqdm(range(0, n, GEN_BATCH_SIZE), desc="Batched gen"):
#             batch_end = min(batch_start + GEN_BATCH_SIZE, n)
#             batch_prompts = questions[batch_start:batch_end]
#             # compute a seed per batch to vary outputs across passes
#             seed = SEED_BASE + pass_i * 100000 + (batch_start // GEN_BATCH_SIZE)
#             gen_texts, gen_lats = batched_generate(
#                 llm_model,
#                 llm_tokenizer,
#                 batch_prompts,
#                 max_new_tokens=MAX_NEW_TOKENS,
#                 temperature=TEMPERATURE,
#                 top_p=TOP_P,
#                 seed=seed
#             )
#             # store
#             for i, (txt, lat) in enumerate(zip(gen_texts, gen_lats)):
#                 idx = batch_start + i
#                 all_preds[idx][pass_i] = txt
#                 all_lat[idx][pass_i] = lat

#     # Compute BERTScore & BLEU per-prediction in batched fashion (to be efficient, compute across flattened lists)
#     flat_preds = [p for preds in all_preds for p in preds]
#     flat_refs = [g for _ in range(NUM_SAMPLES_PER_PROMPT) for g in golds]  # repeated gold per sample
#     print("Computing BERTScore on all predictions (this may take a while)...")
#     bert_fs_flat = compute_bert_scores(flat_preds, flat_refs)
#     print("Computing BLEU scores...")
#     bleu_flat = compute_bleu(flat_preds, flat_refs)

#     # reshape back to per-example lists
#     bert_per_example = [bert_fs_flat[i*NUM_SAMPLES_PER_PROMPT:(i+1)*NUM_SAMPLES_PER_PROMPT] for i in range(n)]
#     bleu_per_example = [bleu_flat[i*NUM_SAMPLES_PER_PROMPT:(i+1)*NUM_SAMPLES_PER_PROMPT] for i in range(n)]

#     # Run verifier in batches: for each pred/gold pair
#     print("Running batched verifier (MNLI)...")
#     flat_preds_for_ver = flat_preds  # same flattening
#     verifier_results = verifier_batch_predict(verifier, verifier_tok, flat_preds_for_ver, flat_refs, device=VERIFIER_DEVICE, batch_size=64)
#     # reshape to per-example
#     ver_labels = [verifier_results[i*NUM_SAMPLES_PER_PROMPT:(i+1)*NUM_SAMPLES_PER_PROMPT] for i in range(n)]

#     # Now write CSV rows with aggregated metrics
#     print("Writing CSV results...")
#     for i in tqdm(range(n), desc="Writing rows"):
#         preds = all_preds[i]
#         lats = all_lat[i]
#         bert_fs = bert_per_example[i]
#         bleu_scores = bleu_per_example[i]

#         avg_latency = sum(lats) / len(lats)
#         avg_bert = sum(bert_fs) / len(bert_fs)
#         avg_bleu = sum(bleu_scores) / len(bleu_scores)

#         # verifier: interpret labels/probs -> votes using threshold
#         ver_chunk = ver_labels[i]  # list of tuples (label, prob_dict)
#         ver_votes = []
#         ver_probs = []
#         for (label, prob_dict) in ver_chunk:
#             # if label is ENTAILMENT and probability exceeds threshold -> Yes
#             ent_prob = prob_dict.get("ENTAILMENT") or prob_dict.get("entailment") or 0.0
#             if (label.upper() == "ENTAILMENT") and (ent_prob >= ENTAILMENT_THRESH):
#                 ver_votes.append("Yes")
#             else:
#                 ver_votes.append("No")
#             ver_probs.append(ent_prob)

#         # CSV row
#         row = [
#             i, questions[i], golds[i],
#             preds[0], lats[0], bert_fs[0], bleu_scores[0],
#             preds[1] if len(preds) > 1 else "", lats[1] if len(lats) > 1 else "", bert_fs[1] if len(bert_fs) > 1 else "", bleu_scores[1] if len(bleu_scores) > 1 else "",
#             preds[2] if len(preds) > 2 else "", lats[2] if len(lats) > 2 else "", bert_fs[2] if len(bert_fs) > 2 else "", bleu_scores[2] if len(bleu_scores) > 2 else "",
#             avg_latency, avg_bert, avg_bleu,
#             " | ".join(ver_votes),
#             " | ".join([f"{p:.3f}" for p in ver_probs])
#         ]
#         writer.writerow(row)
#     csv_file.close()
#     print("Done. Results saved to:", OUTPUT_CSV)

# if __name__ == "__main__":
#     main()


############# V2 ##############
"""
Batched evaluation of a local LLaMA model on a saved test split.
- Generates NUM_SAMPLES_PER_PROMPT samples per example (by running NUM_SAMPLES_PER_PROMPT passes).
- Uses batched model.generate calls for throughput.
- Uses a separate MNLI verifier in BATCHED mode; interprets ENTAILMENT with threshold as "Yes".

Requirements:
  pip install transformers datasets torch bert-score sacrebleu tqdm
"""

import os
import time
import csv
import statistics as stats
from typing import List
from math import ceil
from tqdm import tqdm

import torch
import torch.nn.functional as F
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
)
from datasets import load_from_disk

# metrics
from bert_score import score as bert_score
import sacrebleu

# ---------------- CONFIG ----------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
MODEL_DIR = os.path.join(PROJECT_ROOT, "models", "base")  # for base model
# MODEL_DIR = os.path.join(PROJECT_ROOT, "models", "base_full_sft")  # for fine-tuned model
VERIFIER_MODEL_NAME = os.path.join(PROJECT_ROOT, "models", "verifier")
TEST_DATA_PATH = os.path.join(PROJECT_ROOT, "data", "test")

OUTPUT_CSV = os.path.join(PROJECT_ROOT, "eval", "vanilla", "results_1000.csv")
OUTPUT_SUMMARY_CSV = os.path.join(PROJECT_ROOT, "eval", "vanilla", "results_1000_summary.csv")

NUM_SAMPLES_PER_PROMPT = 3
GEN_BATCH_SIZE = 64              # number of examples processed per generate() call
MAX_NEW_TOKENS = 1024
TEMPERATURE = 0.9
TOP_P = 0.9
SEED_BASE = 1000

PROMPT_TEMPLATE = "{question}\n\nAnswer:"

# Verifier (MNLI)
VERIFIER_DEVICE = "cuda:0"       # e.g., "cuda:0" for first GPU, or "cpu"
ENTAILMENT_THRESH = 0.5          # require ENTAILMENT prob > threshold to say "Yes"
# ----------------------------------------

# ------------------ Helpers ------------------
def load_llm_and_tokenizer(model_dir: str):
    tok = AutoTokenizer.from_pretrained(model_dir, local_files_only=True, use_fast=False, padding_side='left')
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        local_files_only=True,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    if tok.pad_token is None:
        # simplest: reuse eos token as pad
        tok.pad_token = tok.eos_token
    model.eval()
    return model, tok

def batched_generate(model, tokenizer, prompts: List[str], max_new_tokens=128, temperature=0.7, top_p=0.9, seed=0):
    """
    Batch generate for a list of prompts (size = GEN_BATCH_SIZE or less).
    Returns list of generated strings and list of latencies (per-batch averaged to per-sample).
    We create a single torch.Generator seeded with seed so results are reproducible per-call.
    """
    inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(model.device)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    t0 = time.perf_counter()
    with torch.no_grad():
        out = model.generate(
            **inputs,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
        )
    batch_latency = time.perf_counter() - t0
    # decode each output and strip prompt prefix when possible
    decoded = [tokenizer.decode(o, skip_special_tokens=True) for o in out]
    results = []
    for prompt, text in zip(prompts, decoded):
        if text.startswith(prompt):
            results.append(text[len(prompt):].strip())
        else:
            results.append(text.strip())
    # return generated list and latency-per-sample (we'll record batch_latency / len(prompts))
    per_sample_latency = batch_latency / max(1, len(prompts))
    latencies = [per_sample_latency] * len(prompts)
    return results, latencies

def batchify(lst, batch_size):
    for i in range(0, len(lst), batch_size):
        yield lst[i:i + batch_size]

# Verifier utilities (batched)
def load_verifier(model_name: str, device: str = "cpu"):
    verifier_tok = AutoTokenizer.from_pretrained(model_name)
    verifier = AutoModelForSequenceClassification.from_pretrained(model_name)
    verifier.to(device)
    verifier.eval()
    return verifier, verifier_tok

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
        with torch.no_grad():
            logits = verifier(**inputs).logits  # (B, num_labels)
            probs = F.softmax(logits, dim=-1).cpu().tolist()  # list of lists
        for prob in probs:
            top_idx = int(max(range(len(prob)), key=lambda j: prob[j]))
            label = id2label[top_idx]
            prob_dict = {id2label[j]: prob[j] for j in range(len(prob))}
            results.append((label, prob_dict))
    return results

# Metric computations (BERTScore / BLEU)
def compute_bert_scores(preds: List[str], refs: List[str], model_type="roberta-large"):
    P, R, F = bert_score(preds, refs, lang="en", model_type=model_type, verbose=False)
    return [float(f) for f in F]

def compute_bleu_pairwise(preds: List[str], refs: List[str]):
    """
    Returns two parallel lists:
      - bleu_100: sentence BLEU per pair on a 0..100 scale
      - bleu_norm: normalized BLEU per pair on a 0..1 scale (bleu_100 / 100)
    Uses sacrebleu.sentence_bleu with '13a' tokenizer and exp smoothing.
    """
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

# ------------------ Main ------------------
def main():
    # load LLM & tokenizer
    print("Loading LLM model and tokenizer...")
    llm_model, llm_tokenizer = load_llm_and_tokenizer(MODEL_DIR)
    device = next(llm_model.parameters()).device
    print("LLM device:", device)

    # load verifier
    print("Loading verifier:", VERIFIER_MODEL_NAME)
    verifier, verifier_tok = load_verifier(VERIFIER_MODEL_NAME, device=VERIFIER_DEVICE)

    # load dataset
    print("Loading test dataset from:", TEST_DATA_PATH)
    ds_test = load_from_disk(TEST_DATA_PATH)
    if hasattr(ds_test, "keys"):
        ds_test = ds_test["test"] if "test" in ds_test else next(iter(ds_test.values()))
    LIMIT_N = 1000  # smoke test size
    if LIMIT_N is not None and LIMIT_N > 0 and len(ds_test) > LIMIT_N:
        ds_test = ds_test.select(range(LIMIT_N))  # first LIMIT_N
    n = len(ds_test)
    print(f"Test size: {n}")

    # build prompt list (preserve order)
    questions = []
    golds = []
    for idx in range(n):
        row = ds_test[idx]

        # tolerant getters for multiple dataset schemas
        q = (
            (row.get("question") if isinstance(row, dict) else None)
            or (row.get("prompt")   if isinstance(row, dict) else None)
            or (row.get("Q")        if isinstance(row, dict) else None)
            or (row["Q"] if hasattr(ds_test, "features") and "Q" in ds_test.features else "")
        )

        g = (
            (row.get("answer") if isinstance(row, dict) else None)
            or (row.get("target") if isinstance(row, dict) else None)
            or (row.get("A")      if isinstance(row, dict) else None)
            or (row["A"] if hasattr(ds_test, "features") and "A" in ds_test.features else "")
        )

        # final string sanitation
        q = (q or "").strip()
        g = (g or "").strip()

        questions.append(PROMPT_TEMPLATE.format(question=q))
        golds.append(g)

    # (optional) quick sanity check
    print("Sample prompt:", questions[0][:200].replace("\n", " ⏎ ") if n > 0 else "")
    print("Sample gold  :", golds[0][:200].replace("\n", " ⏎ ") if n > 0 else "")

    assert any(len(x.strip()) > 0 for x in questions), "All questions are empty—column mapping is wrong."
    assert any(len(x.strip()) > 0 for x in golds),     "All golds are empty—column mapping is wrong."

    # Prepare CSV writer
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    csv_file = open(OUTPUT_CSV, "w", newline="", encoding="utf-8")
    writer = csv.writer(csv_file)
    header = [
        "idx", "question", "gold",
        "pred_1", "latency_1", "bert_1", "bleu_1", "bleu_norm_1",
        "pred_2", "latency_2", "bert_2", "bleu_2", "bleu_norm_2",
        "pred_3", "latency_3", "bert_3", "bleu_3", "bleu_norm_3",
        "avg_latency", "avg_bert_f1", "avg_bleu", "avg_bleu_norm",
        "verifier_votes", "verifier_probs"
    ]
    writer.writerow(header)

    # Pre-allocate storage for preds and latencies: list of lists per example
    all_preds = [[""] * NUM_SAMPLES_PER_PROMPT for _ in range(n)]
    all_lat = [[0.0] * NUM_SAMPLES_PER_PROMPT for _ in range(n)]

    # For throughput: run NUM_SAMPLES_PER_PROMPT passes; each pass generates one sample per example in batches
    for pass_i in range(NUM_SAMPLES_PER_PROMPT):
        print(f"Generation pass {pass_i+1}/{NUM_SAMPLES_PER_PROMPT}")
        # iterate in batches
        for batch_start in tqdm(range(0, n, GEN_BATCH_SIZE), desc="Batched gen"):
            batch_end = min(batch_start + GEN_BATCH_SIZE, n)
            batch_prompts = questions[batch_start:batch_end]
            # compute a seed per batch to vary outputs across passes
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
            # store
            for i, (txt, lat) in enumerate(zip(gen_texts, gen_lats)):
                idx = batch_start + i
                all_preds[idx][pass_i] = txt
                all_lat[idx][pass_i] = lat

    # Compute BERTScore & BLEU per-prediction in batched fashion (compute across flattened lists)
    flat_preds = [p for preds in all_preds for p in preds]
    flat_refs = [g for _ in range(NUM_SAMPLES_PER_PROMPT) for g in golds]  # repeated gold per sample

    print("Computing BERTScore on all predictions (this may take a while)...")
    bert_fs_flat = compute_bert_scores(flat_preds, flat_refs)

    print("Computing BLEU scores (pairwise)...")
    bleu_flat, bleu_norm_flat = compute_bleu_pairwise(flat_preds, flat_refs)

    # reshape back to per-example lists
    bert_per_example       = [bert_fs_flat[i*NUM_SAMPLES_PER_PROMPT:(i+1)*NUM_SAMPLES_PER_PROMPT] for i in range(n)]
    bleu_per_example       = [bleu_flat[i*NUM_SAMPLES_PER_PROMPT:(i+1)*NUM_SAMPLES_PER_PROMPT] for i in range(n)]
    bleu_norm_per_example  = [bleu_norm_flat[i*NUM_SAMPLES_PER_PROMPT:(i+1)*NUM_SAMPLES_PER_PROMPT] for i in range(n)]

    # Run verifier in batches: for each pred/gold pair
    print("Running batched verifier (MNLI)...")
    flat_preds_for_ver = flat_preds  # same flattening
    verifier_results = verifier_batch_predict(verifier, verifier_tok, flat_preds_for_ver, flat_refs, device=VERIFIER_DEVICE, batch_size=64)
    # reshape to per-example
    ver_labels = [verifier_results[i*NUM_SAMPLES_PER_PROMPT:(i+1)*NUM_SAMPLES_PER_PROMPT] for i in range(n)]

    # --- accumulators for dataset-level summary ---
    avg_bert_list, avg_bleu_list, avg_bleu_norm_list, avg_latency_list, yes_rate_list = [], [], [], [], []

    # Now write CSV rows with aggregated metrics
    print("Writing CSV results...")
    for i in tqdm(range(n), desc="Writing rows"):
        preds = all_preds[i]
        lats = all_lat[i]
        bert_fs = bert_per_example[i]
        bleu_scores = bleu_per_example[i]
        bleu_norm_scores = bleu_norm_per_example[i]

        avg_latency   = sum(lats) / len(lats)
        avg_bert      = sum(bert_fs) / len(bert_fs)
        avg_bleu      = sum(bleu_scores) / len(bleu_scores)
        avg_bleu_norm = sum(bleu_norm_scores) / len(bleu_norm_scores)

        # verifier: interpret labels/probs -> votes using threshold
        ver_chunk = ver_labels[i]  # list of tuples (label, prob_dict)
        ver_votes = []
        ver_probs = []
        for (label, prob_dict) in ver_chunk:
            # if label is ENTAILMENT and probability exceeds threshold -> Yes
            ent_prob = prob_dict.get("ENTAILMENT") or prob_dict.get("entailment") or 0.0
            if (label.upper() == "ENTAILMENT") and (ent_prob >= ENTAILMENT_THRESH):
                ver_votes.append("Yes")
            else:
                ver_votes.append("No")
            ver_probs.append(ent_prob)

        # --- accumulate for summary ---
        yes_rate = sum(1 for v in ver_votes if v == "Yes") / max(1, len(ver_votes))
        avg_bert_list.append(avg_bert)
        avg_bleu_list.append(avg_bleu)
        avg_bleu_norm_list.append(avg_bleu_norm)
        avg_latency_list.append(avg_latency)
        yes_rate_list.append(yes_rate)

        # CSV row
        row = [
            i, questions[i], golds[i],
            preds[0], lats[0], bert_fs[0], bleu_scores[0], bleu_norm_scores[0],
            preds[1] if len(preds) > 1 else "", lats[1] if len(lats) > 1 else "", bert_fs[1] if len(bert_fs) > 1 else "", bleu_scores[1] if len(bleu_scores) > 1 else "", (bleu_norm_scores[1] if len(bleu_norm_scores) > 1 else ""),
            preds[2] if len(preds) > 2 else "", lats[2] if len(lats) > 2 else "", bert_fs[2] if len(bert_fs) > 2 else "", bleu_scores[2] if len(bleu_scores) > 2 else "", (bleu_norm_scores[2] if len(bleu_norm_scores) > 2 else ""),
            avg_latency, avg_bert, avg_bleu, avg_bleu_norm,
            " | ".join(ver_votes),
            " | ".join([f"{p:.3f}" for p in ver_probs])
        ]
        writer.writerow(row)
    csv_file.close()
    print("Per-example results saved to:", OUTPUT_CSV)

    # ---- dataset-level summary ----
    def mean_std(x):
        return (float(stats.mean(x)) if x else 0.0,
                float(stats.pstdev(x)) if len(x) > 1 else 0.0)

    m_bert, s_bert           = mean_std(avg_bert_list)
    m_bleu, s_bleu           = mean_std(avg_bleu_list)
    m_bleu_norm, s_bleu_norm = mean_std(avg_bleu_norm_list)
    m_lat,  s_lat            = mean_std(avg_latency_list)
    m_yes,  s_yes            = mean_std(yes_rate_list)

    os.makedirs(os.path.dirname(OUTPUT_SUMMARY_CSV), exist_ok=True)
    with open(OUTPUT_SUMMARY_CSV, "w", newline="", encoding="utf-8") as fsum:
        w = csv.writer(fsum)
        w.writerow(["key", "value_mean", "value_std", "notes"])
        w.writerow(["num_examples", n, 0, ""])
        w.writerow(["samples_per_prompt", NUM_SAMPLES_PER_PROMPT, 0, ""])
        w.writerow(["avg_bert_f1", m_bert, s_bert, "Per-example mean of BERTScore F1 (then averaged)"])
        w.writerow(["avg_bleu", m_bleu, s_bleu, "Sentence BLEU (0..100) averaged over examples"])
        w.writerow(["avg_bleu_norm", m_bleu_norm, s_bleu_norm, "Sentence BLEU normalized to 0..1"])
        w.writerow(["avg_latency_sec", m_lat, s_lat, "Per-example mean of per-sample latency"])
        w.writerow(["verifier_yes_rate", m_yes, s_yes, f"ENTAILMENT_THRESH={ENTAILMENT_THRESH}"])

    print("Summary saved to:", OUTPUT_SUMMARY_CSV)
    print("Done.")

if __name__ == "__main__":
    main()


