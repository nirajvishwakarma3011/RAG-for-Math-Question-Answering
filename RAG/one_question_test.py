#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Run RAG+LLM generation for a single math question (no dataset file).
Requires your existing modules:
 - retriever_math.load_retriever(STORE_DIR, mode)
 - prompt_math.build_prompt(tokenizer, question, hits, MAX_INPUT_TOKENS, CTX_BUDGET_TOKENS)
Adjust MODEL_DIR, STORE_DIR, VERIFIER_MODEL if needed.
"""

import os
import time
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification
from retriever_math import load_retriever
from prompt_math import build_prompt

# ---------------- CONFIG (adjust paths as needed) ----------------
MODEL_DIR         = "/home/ankitam/dlnlp_rag/models/openmath2_llama"
STORE_DIR         = "store"            # where retriever index lives
RETRIEVER         = "hybrid"           # "bm25" | "dense" | "hybrid"
TOP_K             = 1

MAX_INPUT_TOKENS  = 10000
CTX_BUDGET_TOKENS = 7000
MAX_NEW_TOKENS    = 2042

TEMPERATURE       = 0.9
TOP_P             = 0.9

NUM_SAMPLES_PER_PROMPT = 3
GEN_BATCH_SIZE         = 8
SEED_BASE              = 1000

VERIFIER_MODEL_NAME = os.path.join(os.path.dirname(MODEL_DIR), "verifier")  # optional
VERIFIER_DEVICE     = "cuda:0"
ENTAILMENT_THRESH   = 0.5

# The single question to test (exactly as you gave it)
# QUESTION =  (
#     "Let a, b, c > 0 such that a^3 + b^3 + c^3 = 3. "
#     "Prove that (a^3)/(a + b) + (b^3)/(b + c) + (c^3)/(c + a) >= 3/2. "
#     # "Show step-by-step reasoning using inequalities (Cauchy–Schwarz, Young)."
# )

QUESTION = (
    "Proof that the sum of two Gaussian variables is Gaussian."
)
# ----------------------------------------------------------------

def load_llm_and_tokenizer(model_dir: str):
    tok = AutoTokenizer.from_pretrained(model_dir, use_fast=False, padding_side="left")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto"   # will place on available devices
    )
    model.eval()
    return model, tok

@torch.no_grad()
def batched_generate(model, tokenizer, prompts, max_new_tokens=128, temperature=0.7, top_p=0.9, seed=0):
    # prompts: list of strings (len <= GEN_BATCH_SIZE)
    inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(model.device)
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))

    t0 = time.perf_counter()
    out = model.generate(
        **inputs,
        do_sample=False,
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

def load_verifier_if_exists(model_name: str, device: str = "cpu"):
    if not os.path.exists(model_name):
        return None, None
    vtok = AutoTokenizer.from_pretrained(model_name)
    vmdl = AutoModelForSequenceClassification.from_pretrained(model_name)
    vmdl.to(device)
    vmdl.eval()
    return vmdl, vtok

def verifier_predict_once(verifier, tokenizer, pred: str, gold: str, device="cpu"):
    # returns (label, prob_dict)
    if verifier is None:
        return None
    inputs = tokenizer([pred], [gold], return_tensors="pt", padding=True, truncation=True).to(device)
    logits = verifier(**inputs).logits
    probs = F.softmax(logits, dim=-1).detach().cpu().tolist()[0]
    id2label = verifier.config.id2label if hasattr(verifier.config, "id2label") else {0: "LABEL_0", 1: "LABEL_1", 2: "LABEL_2"}
    top_idx = int(max(range(len(probs)), key=lambda j: probs[j]))
    label = id2label[top_idx]
    prob_dict = {id2label[j]: probs[j] for j in range(len(probs))}
    return (label, prob_dict)

def main():
    print("Loading retriever from:", STORE_DIR, "mode:", RETRIEVER)
    retriever = load_retriever(STORE_DIR, RETRIEVER)

    print("Loading LLM & tokenizer from:", MODEL_DIR)
    llm_model, llm_tokenizer = load_llm_and_tokenizer(MODEL_DIR)
    device = next(llm_model.parameters()).device
    print("LLM device:", device)

    # optional verifier
    verifier, verifier_tok = load_verifier_if_exists(VERIFIER_MODEL_NAME, device=VERIFIER_DEVICE)
    if verifier is not None:
        print("Loaded verifier at:", VERIFIER_MODEL_NAME)
    else:
        print("No verifier found at:", VERIFIER_MODEL_NAME, "(skipping verifier)")

    # Retrieve top-k
    print("\n=== Retrieving top-k documents ===")
    hits = retriever.search(QUESTION, TOP_K)   # expected list of (doc_id, score, maybe text)
    # handle both tuple shapes
    normalized_hits = []
    for h in hits:
        if isinstance(h, (list, tuple)) and len(h) >= 2:
            docid = h[0]
            score = float(h[1])
            normalized_hits.append((docid, score))
        else:
            normalized_hits.append((str(h), 0.0))
    for rank, (docid, score) in enumerate(normalized_hits, start=1):
        print(f"[{rank}] id={docid}   score={score:.6f}")

    # Build prompt
    print("\n=== Building prompt ===")
    prompt = build_prompt(llm_tokenizer, QUESTION, hits, MAX_INPUT_TOKENS, CTX_BUDGET_TOKENS)
    tokenized = llm_tokenizer.encode(prompt, add_special_tokens=False)
    print(f"Prompt length (tokens): {len(tokenized)}")
    print("---- prompt preview (first 1000 chars) ----")
    print(prompt)
    print("-------------------------------------------")

    # Generate NUM_SAMPLES_PER_PROMPT answers (in batches if needed)
    print(f"\n=== Generating {NUM_SAMPLES_PER_PROMPT} samples ===")
    all_preds = []
    all_lats = []
    for pass_i in range(NUM_SAMPLES_PER_PROMPT):
        seed = SEED_BASE + pass_i * 100000
        gen_texts, gen_lats = batched_generate(
            llm_model,
            llm_tokenizer,
            [prompt],
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            seed=seed
        )
        all_preds.append(gen_texts[0])
        all_lats.append(gen_lats[0])
        print(f"\n--- Sample {pass_i+1} (latency {gen_lats[0]:.4f}s) ---")
        print(gen_texts[0])
        # optionally run verifier if present (using the question as 'gold' is not ideal; skip unless you have real gold)
        # If you do have a gold answer string, you can call verifier_predict_once(verifier, verifier_tok, gen_texts[0], GOLD_ANSWER, device=VERIFIER_DEVICE)

    # Summary
    avg_lat = sum(all_lats) / len(all_lats) if all_lats else 0.0
    print("\n=== Summary ===")
    for i, (p, l) in enumerate(zip(all_preds, all_lats), start=1):
        print(f"sample_{i}: latency={l:.4f}s  chars={len(p)} tokens(approx)={len(llm_tokenizer.encode(p, add_special_tokens=False))}")
    print(f"avg_latency = {avg_lat:.4f}s")

if __name__ == "__main__":
    main()
