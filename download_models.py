from huggingface_hub import snapshot_download
import os

# Directory to store all models locally
MODEL_DIR = "./models"

# List of models to download
models = [
    "stabilityai/stable-audio-open-1.0",
    "PsiPi/audio",
    "RoyalCities/RC_Infinite_Pianos",
    "RoyalCities/Vocal_Textures_Main",
    "Nekochu/stable-audio-open-1.0-Music",
    "santifiorino/SAO-Instrumental-Finetune",
    "adlb/Audialab_EDM_Elements",
    "bleepybloops/sao_vae_tuned_100k"
]

os.makedirs(MODEL_DIR, exist_ok=True)

# Download each model into ./models/<repo_name>
for model_id in models:
    local_dir = os.path.join(MODEL_DIR, model_id.replace("/", "__"))
    print(f"⬇️ Downloading {model_id} to {local_dir}...")
    snapshot_download(repo_id=model_id, local_dir=local_dir, local_dir_use_symlinks=False)
    print(f"✅ Done: {model_id}")
