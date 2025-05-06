import torch
from transformers import T5Tokenizer, T5EncoderModel

# Local path to T5 encoder
T5_PATH = "encoders/t5"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load T5 tokenizer and encoder from local files
print("🔠 Loading tokenizer and encoder from:", T5_PATH)
tokenizer = T5Tokenizer.from_pretrained(T5_PATH)
encoder = T5EncoderModel.from_pretrained(T5_PATH).to(DEVICE)

# Test with a prompt
prompt = "A gentle ocean wave crashing"
print("🧠 Encoding prompt:", prompt)
tokenized = tokenizer(prompt, return_tensors="pt").input_ids.to(DEVICE)
embeddings = encoder(input_ids=tokenized).last_hidden_state
print("✅ T5 embeddings shape:", embeddings.shape)

