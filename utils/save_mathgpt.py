# Optional: HF Token if the model is gated/private
# HF_TOKEN = "hf_XXXXXXXXXXXXXXXXXXXXXXXXXXX"

from transformers import AutoModelForCausalLM, AutoTokenizer
import os

# Hugging Face Model ID
HF_ID = "nvidia/OpenMath2-Llama3.1-8B"

# Local save directory
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "models", "openmath2_llama")

# Create directory if not exists
os.makedirs(OUT_DIR, exist_ok=True)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    HF_ID,
    use_fast=True,
    trust_remote_code=True
)

# Load model
model = AutoModelForCausalLM.from_pretrained(
    HF_ID,
    trust_remote_code=True,
    torch_dtype="auto"   # lets HF choose best dtype (fp16/bf16)
)

# Save locally
tokenizer.save_pretrained(OUT_DIR)
model.save_pretrained(OUT_DIR, safe_serialization=True)

print("Model saved at:", OUT_DIR)
