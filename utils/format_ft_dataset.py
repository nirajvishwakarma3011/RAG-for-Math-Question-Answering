# utils/format_ft_dataset.py
from datasets import load_from_disk
from transformers import AutoTokenizer
import os

BASE_DIR = os.path.join(os.path.dirname(__file__), "..")
MODEL_DIR = os.path.join(BASE_DIR, "models", "base")
FT_PATH   = os.path.join(BASE_DIR, "data", "ft")
FT_PATH_Processed   = os.path.join(BASE_DIR, "data", "ft_processed")


PROMPT_TEMPLATE = "{question}\n\nAnswer:"
EOS = "</s>"  # will be replaced by tokenizer.eos_token

def build_text(example):
    q = example.get("Q") or example.get("prompt") or ""
    a = example.get("A") or example.get("target") or ""
    return PROMPT_TEMPLATE.format(question=q) + " " + a

def main():
    tok = AutoTokenizer.from_pretrained(MODEL_DIR, use_fast=True)
    ds  = load_from_disk(FT_PATH)
    if hasattr(ds, "keys"):
        ds = ds["train"] if "train" in ds else next(iter(ds.values()))

    ds = ds.map(lambda ex: {"text": build_text(ex)}, remove_columns=[c for c in ds.column_names if c != "text"])
    ds.save_to_disk(FT_PATH_Processed)  # overwrite with a 'text' column
    print("Re-saved ft dataset with 'text' column.")

if __name__ == "__main__":
    main()
