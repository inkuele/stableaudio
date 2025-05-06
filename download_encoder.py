import os
from huggingface_hub import snapshot_download

# Disable offline mode to allow downloading
os.environ.pop("HF_HUB_OFFLINE", None)
# Alternatively, explicitly enable online mode:
# os.environ["HF_HUB_OFFLINE"] = "0"

# Correct model repo on Hugging Face
REPO_ID = "google-t5/t5-small"
# Local target directory for downloaded files
TARGET_DIR = os.path.join("encoders", "t5")

print(f"Downloading '{REPO_ID}' to '{TARGET_DIR}'...")
# Download all files in the repo to TARGET_DIR
snapshot_download(
    repo_id=REPO_ID,
    local_dir=TARGET_DIR,
    local_dir_use_symlinks=False,
    # You can restrict to common model file patterns
    allow_patterns=["*.bin", "*.safetensors", "*.json", "*.model", "*.txt"],
    force_download=False
)
print("Download complete. Files are ready in the local_models/t5 folder.")

