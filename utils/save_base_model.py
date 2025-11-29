#HF_TOKEN = "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"
from transformers import AutoModelForCausalLM, AutoTokenizer
import os

HF_ID = "meta-llama/Llama-3.1-8B"
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "models", "base")

# Load once (this may download once)
tokenizer = AutoTokenizer.from_pretrained(HF_ID, use_fast=True)
model = AutoModelForCausalLM.from_pretrained(HF_ID, trust_remote_code=True)

# Save locally for fast reuse
tokenizer.save_pretrained(OUT_DIR)
model.save_pretrained(OUT_DIR, safe_serialization=True)   # attempts safetensors where supported
