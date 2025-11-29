from transformers import AutoModelForSequenceClassification, AutoTokenizer
import os

model_name = "valhalla/distilbart-mnli-12-1"
cache_dir = os.path.join(os.path.dirname(__file__), "..", "models", "verifier")

# This downloads once, saves in cache_dir
tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
model = AutoModelForSequenceClassification.from_pretrained(model_name, trust_remote_code=True)

# Save locally for fast reuse
tok.save_pretrained(cache_dir)
model.save_pretrained(cache_dir, safe_serialization=True)   # attempts safetensors where supported
