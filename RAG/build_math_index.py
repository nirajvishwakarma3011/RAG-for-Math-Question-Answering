#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Builds a math-aware retrieval store from a StackExchange-style HF dataset.

Supported inputs at DATA_DIR:
  1) HF DatasetDict saved to disk (with splits like 'train')
  2) HF Dataset (single split directory)
  3) A .jsonl file (fallback) containing objects with Q, A, meta.url (optional)

Outputs under STORE_DIR:
  - bm25.jsonl   : JSONL with {doc_id, Q, A, url, text=(Q+" "+A)} for BM25
  - faiss.index  : (optional) dense FAISS index over text (Q + A)
  - meta.json    : {doc_ids, Q, A, url}
"""

import os
import re
import json
from pathlib import Path
from typing import Iterable, List, Dict, Any

from tqdm import tqdm

# ---------- CONFIG ----------
STORE_DIR      = "store"
DATA_DIR       = "/home/ankitam/dlnlp_rag/data/rag"   # path to HF 'save_to_disk' folder or a .jsonl file
USE_DENSE      = True
DENSE_ENCODER  = "sentence-transformers/all-MiniLM-L6-v2"
PREVIEW_N      = 2               # show a tiny preview of collected rows
# ----------------------------


def normalize_math(s: str) -> str:
    s = (s or "")
    s = s.replace("×", "*").replace("÷", "/").replace("–", "-").replace("−", "-")
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _iter_hf_dataset(ds) -> Iterable[Dict[str, Any]]:
    """
    Accepts either a datasets.DatasetDict or a datasets.Dataset and yields rows.
    Preference order if DatasetDict: 'train' > any other split.
    """
    from datasets import Dataset, DatasetDict

    if isinstance(ds, DatasetDict):
        if "train" in ds:
            yield from ds["train"]
        else:
            # take the first available split
            first_split = next(iter(ds.keys()))
            yield from ds[first_split]
    elif isinstance(ds, Dataset):
        # single split already; just iterate
        yield from ds
    else:
        raise TypeError(f"Unsupported HF object type: {type(ds)}")


def load_corpus_rows(data_dir: str) -> List[Dict[str, Any]]:
    """
    Loads corpus rows with columns Q, A, and optional meta.url
    from either:
      - HF dataset saved to disk (Dataset or DatasetDict)
      - or a .jsonl file
    Returns a list of dicts with keys: Q, A, url
    """
    p = Path(data_dir)

    # Case A: a jsonl file present
    jsonl_files = list(p.glob("*.jsonl")) if p.exists() and p.is_dir() else []
    if p.is_file() and p.suffix == ".jsonl":
        jsonl_files = [p]
    if jsonl_files:
        rows = []
        for jl in jsonl_files:
            with open(jl, "r", encoding="utf-8") as f:
                for line in f:
                    obj = json.loads(line)
                    q = normalize_math(obj.get("Q", ""))
                    a = normalize_math(obj.get("A", ""))
                    meta = obj.get("meta") or {}
                    url = meta.get("url") or obj.get("url") or ""
                    rows.append({"Q": q, "A": a, "url": url})
        return rows

    # Case B: HF dataset saved with datasets.save_to_disk
    from datasets import load_from_disk
    try:
        ds = load_from_disk(str(p))
    except Exception as e:
        raise RuntimeError(
            f"Could not load corpus from '{data_dir}'. "
            f"Provide a HF dataset folder or a .jsonl file. Original error: {e}"
        )

    rows = []
    for r in _iter_hf_dataset(ds):
        q = normalize_math(r.get("Q", ""))
        a = normalize_math(r.get("A", ""))
        meta = r.get("meta") or {}
        url = meta.get("url") or ""
        rows.append({"Q": q, "A": a, "url": url})
    return rows


def main():
    os.makedirs(STORE_DIR, exist_ok=True)

    rows = load_corpus_rows(DATA_DIR)

    # Filter & validate
    kept = []
    for r in rows:
        q, a = r.get("Q", ""), r.get("A", "")
        if q and a:
            kept.append(r)

    if not kept:
        raise ValueError(
            f"No valid rows found in '{DATA_DIR}'. "
            f"Expected at least columns 'Q' and 'A'."
        )

    # Preview a couple of rows
    print(f"Loaded {len(kept)} rows. Preview:")
    for i, r in enumerate(kept[:PREVIEW_N], 1):
        print(f"  {i}. Q: {r['Q'][:80]}{'...' if len(r['Q'])>80 else ''}")
        print(f"     A: {r['A'][:80]}{'...' if len(r['A'])>80 else ''}")
        if r.get("url"):
            print(f"     url: {r['url']}")

    # Assemble fields for storage
    doc_ids, questions, answers, urls, text_for_index = [], [], [], [], []
    for i, r in enumerate(tqdm(kept, desc="Collect")):
        did = str(i)
        q, a, url = r["Q"], r["A"], r.get("url", "")
        doc_ids.append(did)
        questions.append(q)
        answers.append(a)
        urls.append(url)
        text_for_index.append((q + " " + a).strip())

    # Save BM25 surface
    bm25_path = Path(STORE_DIR) / "bm25.jsonl"
    with bm25_path.open("w", encoding="utf-8") as f:
        for did, q, a, url, t in zip(doc_ids, questions, answers, urls, text_for_index):
            f.write(json.dumps(
                {"doc_id": did, "Q": q, "A": a, "url": url, "text": t},
                ensure_ascii=False
            ) + "\n")
    print(f"Wrote {bm25_path}")

    # Optional dense index
    if USE_DENSE:
        import numpy as np
        import faiss
        from sentence_transformers import SentenceTransformer

        print("Building dense embeddings (this can take a while)...")
        enc = SentenceTransformer(DENSE_ENCODER)
        embs = enc.encode(
            text_for_index,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=True,
        )
        index = faiss.IndexFlatIP(embs.shape[1])
        index.add(embs.astype(np.float32))
        faiss_path = Path(STORE_DIR) / "faiss.index"
        faiss.write_index(index, str(faiss_path))
        print(f"Wrote {faiss_path}")

    # Meta for prompting
    meta_path = Path(STORE_DIR) / "meta.json"
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(
            {"doc_ids": doc_ids, "Q": questions, "A": answers, "url": urls},
            f,
            ensure_ascii=False,
        )
    print(f"Wrote {meta_path}")
    print("Done ✅")


if __name__ == "__main__":
    main()
