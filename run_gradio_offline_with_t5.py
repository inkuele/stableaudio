import os
import json
import torch
from einops import rearrange
import torchaudio
import tempfile
import gradio as gr
import time
import zipfile

# === Load local T5 encoder ===
from encoders import tokenizer, encoder
# =============================

from stable_audio_tools.models.factory import create_model_from_config
from stable_audio_tools.models.utils import load_ckpt_state_dict
from stable_audio_tools.inference.generation import generate_diffusion_cond

# --- Device selection ---
def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")
device = get_device()
print(f"Using device: {device}")

# --- Checkpoint management ---
MODEL_DIR = "models"

def list_checkpoints():
    """
    Scan each model folder and list all .safetensors and .ckpt files.
    Returns list of strings 'folder/filename'.
    """
    options = []
    if os.path.isdir(MODEL_DIR):
        for folder in sorted(os.listdir(MODEL_DIR)):
            folder_path = os.path.join(MODEL_DIR, folder)
            if not os.path.isdir(folder_path):
                continue
            for fname in sorted(os.listdir(folder_path)):
                if fname.endswith(".safetensors") or fname.endswith(".ckpt"):
                    options.append(f"{folder}/{fname}")
    return options

ckpt_options = list_checkpoints()
if not ckpt_options:
    raise RuntimeError(f"No checkpoints found under '{MODEL_DIR}'")

def parse_ckpt_selection(selection: str):
    """Split 'folder/filename' into folder and filename"""
    folder, fname = selection.split('/', 1)
    return folder, fname

# Globals for current model
model = None
model_config = None
default_sample_rate = 44100

# --- Load model by checkpoint selection ---
def load_model(selection: str):
    global model, model_config, default_sample_rate
    folder, fname = parse_ckpt_selection(selection)
    root = os.path.join(MODEL_DIR, folder)
    # Read config
    with open(os.path.join(root, "model_config.json"), 'r') as f:
        model_config = json.load(f)
    # Load checkpoint file explicitly
    ckpt_path = os.path.join(root, fname)
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Checkpoint file not found: {ckpt_path}")
    # Instantiate
    m = create_model_from_config(model_config)
    state = load_ckpt_state_dict(ckpt_path)
    m.load_state_dict(state)
    model = m.to(device)
    default_sample_rate = model_config.get("sample_rate", 44100)
    print(f"Loaded '{selection}' at {default_sample_rate}Hz on {device}")
    return f"Loaded: {selection}"  

# Initialize default
current_ckpt = ckpt_options[0]
model_status = load_model(current_ckpt)

# --- Presets ---
PRESETS = {
    "Ambient": "ambient drone, reverb tails, 60 BPM",
    "Tech House Loop": "128 BPM tech house drum loop, dry mix",
    "Lofi Beat": "lofi hip hop beat with vinyl crackle, 75 BPM",
    "Glitchy IDM": "glitchy IDM with bitcrush textures, 100 BPM",
    "Modular Synth": "modular synth arpeggio in stereo, 90 BPM"
}

# --- Audio generation with timing info ---
def generate_audio(
    prompts, neg_prompt, start_sec, duration_sec,
    steps, cfg, sampler, sigma_min, sigma_max,
    sr, upload, mix,
    progress=gr.Progress(track_tqdm=True)
):
    global model
    lines = [l.strip() for l in prompts.splitlines() if l.strip()]
    # Encode prompts
    embs = []
    enc_dev = next(encoder.parameters()).device
    for txt in lines:
        ids = tokenizer(txt, return_tensors="pt").input_ids.to(enc_dev)
        emb = encoder(input_ids=ids).last_hidden_state.to(device)
        embs.append(emb)
    neg_emb = None
    if neg_prompt:
        ids = tokenizer(neg_prompt, return_tensors="pt").input_ids.to(enc_dev)
        neg_emb = encoder(input_ids=ids).last_hidden_state.to(device)
    # Generate
    previews, files, statuses = [], [], []
    total = len(embs)
    times = []
    for i, emb in enumerate(embs):
        t0 = time.time()
        frac = i/total if total else 0
        progress(frac, desc=f"Gen {i+1}/{total}")
        cond = [{"prompt": lines[i], "text_embeddings": emb,
                 "seconds_start": start_sec, "seconds_total": duration_sec, "weight": 1.0}]
        if neg_emb:
            cond.append({"prompt": neg_prompt, "text_embeddings": neg_emb,
                         "seconds_start": start_sec, "seconds_total": duration_sec, "weight": -cfg})
        if upload and mix>0:
            cond.append({"audio_filepath": upload,
                         "seconds_start": start_sec, "seconds_total": duration_sec, "weight": mix})
        with torch.no_grad():
            out = generate_diffusion_cond(
                model=model, steps=steps, cfg_scale=cfg,
                conditioning=cond, sample_size=sr*duration_sec,
                sigma_min=sigma_min, sigma_max=sigma_max,
                sampler_type=sampler, device=device
            )
        audio = rearrange(out, "b d n -> d (b n)").float().cpu().numpy()
        audio /= (abs(audio).max()+1e-5)
        path = os.path.join(tempfile.gettempdir(), f"preview_{i}.wav")
        torchaudio.save(path, torch.tensor(audio), sr)
        files.append(path)
        elapsed = time.time() - t0
        times.append(elapsed)
        avg = sum(times)/len(times)
        rem = avg*(total-(i+1))
        statuses.append(f"{lines[i]} ⏱{elapsed:.1f}s (avg {avg:.1f}s, rem ~{rem:.1f}s)")
    progress(1.0, desc="Done")
    zipf = os.path.join(tempfile.gettempdir(), f"audio_{int(time.time())}.zip")
    with zipfile.ZipFile(zipf, "w") as zf:
        for p in files: zf.write(p, os.path.basename(p))
    while len(files)<3: files.append(None)
    return (*files[:3], zipf, "\n".join(statuses))

# --- UI ---
with gr.Blocks(title="Stable Audio Offline") as ui:
    gr.Markdown("## 🎵 Stable Audio (Offline)")
    device_tb = gr.Textbox(label="Compute Device", value=str(device), interactive=False)
    with gr.Row():
        with gr.Column():
            ckpt_dd = gr.Dropdown(label="Checkpoint", choices=ckpt_options, value=current_ckpt)
            status_tb = gr.Textbox(label="Status", value=model_status, interactive=False)
            ckpt_dd.change(
                fn=lambda sel: (load_model(sel), str(next(model.parameters()).device)),
                inputs=[ckpt_dd], outputs=[status_tb, device_tb]
            )
            preset_dd = gr.Dropdown(label="Preset", choices=[""]+list(PRESETS.keys()))
            prompt_tb = gr.Textbox(label="Prompts (one per line)", lines=6)
            preset_dd.change(lambda p: PRESETS.get(p, ""), preset_dd, prompt_tb)
            neg_tb = gr.Textbox(label="Negative Prompt", lines=2)
            start_sl = gr.Slider(0,60,value=0,label="Start (s)")
            dur_sl = gr.Slider(1,240,value=10,label="Duration (s)")
            gen_btn = gr.Button("Generate")
            stop_btn = gr.Button("Stop", variant="stop")
            steps_sl = gr.Slider(20,250,value=100,label="Steps")
            cfg_sl = gr.Slider(1,12,value=7,label="CFG Scale")
            samp_dd = gr.Dropdown(label="Sampler",choices=["dpmpp-3m-sde","dpmpp-2m","euler","heun","lms"],value="dpmpp-3m-sde")
            smin_sl = gr.Slider(0.0,1.0,value=0.3,label="Sigma Min")
            smax_sl = gr.Slider(0.0,1000.0,value=500.0,label="Sigma Max")
            sr_dd = gr.Dropdown(label="Sample Rate",choices=[16000,22050,32000,44100,48000],value=default_sample_rate)
            audio_up = gr.Audio(label="Upload Audio", type="filepath")
            mix_sl = gr.Slider(0.0,1.0,value=0.5,label="Audio Mix")

        with gr.Column():
            aud1 = gr.Audio(label="Preview 1", type="filepath")
            aud2 = gr.Audio(label="Preview 2", type="filepath")
            aud3 = gr.Audio(label="Preview 3", type="filepath")
            zip_dl = gr.File(label="Download ZIP")
            hist = gr.Textbox(label="History", lines=10)
    gen_btn.click(
        generate_audio,
        inputs=[prompt_tb, neg_tb, start_sl, dur_sl, steps_sl, cfg_sl, samp_dd, smin_sl, smax_sl, sr_dd, audio_up, mix_sl],
        outputs=[aud1, aud2, aud3, zip_dl, hist]
    )
    stop_btn.click(lambda: None, [], [])

if __name__ == "__main__":
    ui.queue().launch(share=False, server_name="0.0.0.0", server_port=7860)
