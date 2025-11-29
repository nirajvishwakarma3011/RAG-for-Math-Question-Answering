# train/full_sft.py
import os, torch
os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # force GPU 1

from datasets import load_from_disk
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    TrainingArguments, DataCollatorForLanguageModeling, Trainer
)

# BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
# print(BASE)
MODEL_DIR = os.path.join("/home/ankitam/dlnlp_rag", "models", "base")
FT_PATH   = os.path.join("/home/ankitam/dlnlp_rag", "data", "ft_processed")
OUT_DIR   = os.path.join("/home/ankitam/dlnlp_rag", "models", "base_full_sft")

def tokenize_with_labels(batch, tok, max_len=2048):
    out = tok(batch["text"], padding=False, truncation=True, max_length=max_len)
    out["labels"] = out["input_ids"].copy()  # causal LM
    return out

def main():
    # Good matmul perf on Ampere/ADA/Hopper
    torch.set_float32_matmul_precision("high")

    tok = AutoTokenizer.from_pretrained(MODEL_DIR, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # Load full model (no quantization)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR,
        torch_dtype=torch.bfloat16,   # BF16 for weights/acts
        device_map="auto"             # the single visible GPU 1
    )
    # Save VRAM on activations
    model.gradient_checkpointing_enable()

    ds = load_from_disk(FT_PATH)
    if hasattr(ds, "keys"):
        ds = ds["train"] if "train" in ds else next(iter(ds.values()))
    ds_tok = ds.map(lambda b: tokenize_with_labels(b, tok), batched=True, remove_columns=ds.column_names)

    collator = DataCollatorForLanguageModeling(tok, mlm=False)

    args = TrainingArguments(
        output_dir=OUT_DIR,
        per_device_train_batch_size=1,     # start safe; raise to 2 if plenty of headroom
        gradient_accumulation_steps=32,    # effective batch 32
        num_train_epochs=1,                # warm start; increase later
        learning_rate=1e-5,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        logging_steps=50,
        save_steps=1000,
        save_total_limit=2,
        bf16=True,                         # use BF16
        fp16=False,
        gradient_checkpointing=True,
        optim="adamw_torch",               # classic AdamW; swap to adamw_bnb_8bit if you want optimizer sharded
        report_to="none"
    )

    trainer = Trainer(model=model, args=args, train_dataset=ds_tok, data_collator=collator)
    trainer.train()
    trainer.save_model(OUT_DIR)
    tok.save_pretrained(OUT_DIR)
    print("Saved full SFT model to:", OUT_DIR)

if __name__ == "__main__":
    main()

