import os
import torch
from transformers import T5Tokenizer, T5EncoderModel

# Force local-only loading (optional)
# os.environ["HF_HUB_OFFLINE"] = "1"

# Select device: GPU if available, else CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Path to the local T5 model directory
# Assumes this file is in project root and there is an 'encoders/t5' folder
T5_PATH = os.path.join(os.path.dirname(__file__), "encoders", "t5")
print(f"🔠 Loading T5 tokenizer and encoder from local path: {T5_PATH}")

# Load tokenizer and encoder from local files only
tokenizer = T5Tokenizer.from_pretrained(
    T5_PATH,
    local_files_only=True
)
encoder = T5EncoderModel.from_pretrained(
    T5_PATH,
    local_files_only=True
).to(device)

# Optional test function
def test_encoding(prompt: str = "Test prompt for embeddings"):
    print(f"🧠 Encoding prompt: {prompt}")
    inputs = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    outputs = encoder(input_ids=inputs)
    emb = outputs.last_hidden_state
    print(f"✅ Embeddings shape: {emb.shape}")

if __name__ == "__main__":
    test_encoding()

