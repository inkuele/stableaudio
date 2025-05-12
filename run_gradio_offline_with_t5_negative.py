import os
import re
import json
import time
import tempfile
import zipfile

import torch
import torchaudio
from einops import rearrange
import gradio as gr

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
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

device = get_device()
print(f"Using device: {device}")

# --- Checkpoint management ---
MODEL_DIR = "models"

def list_checkpoints():
    opts = []
    for folder in sorted(os.listdir(MODEL_DIR)):
        path = os.path.join(MODEL_DIR, folder)
        if os.path.isdir(path):
            for fname in sorted(os.listdir(path)):
                if fname.endswith('.ckpt'):
                    opts.append(f"{folder}/{fname}")
    return opts

ckpt_options = list_checkpoints()
if not ckpt_options:
    raise RuntimeError("No checkpoints found in 'models/' folder.")

def parse_ckpt(sel: str):
    return sel.split('/', 1)

# Globals
model = None
model_config = None
default_sample_rate = 44100

# --- Load model callback ---
def load_model(selection: str):
    global model, model_config, default_sample_rate
    folder, fname = parse_ckpt(selection)
    root = os.path.join(MODEL_DIR, folder)
    with open(os.path.join(root, 'model_config.json'), 'r') as f:
        model_config = json.load(f)
    ckpt_path = os.path.join(root, fname)
    m = create_model_from_config(model_config)
    state = load_ckpt_state_dict(ckpt_path)
    m.load_state_dict(state)
    model = m.to(device)
    default_sample_rate = model_config.get('sample_rate', 44100)
    msg = f"Loaded: {selection} @ {default_sample_rate}Hz on {device}"
    print(msg)
    return msg

# Initialize default
preferred = [c for c in ckpt_options if c.startswith("stabilityai__stable-audio-open-1.0/")]
current_ckpt = preferred[0] if preferred else ckpt_options[0]
model_status = load_model(current_ckpt)

# --- Presets ---
PRESETS = {
    "Ambient": "ambient drone, reverb tails, 60 BPM",
    "Tech House Loop": "128 BPM tech house drum loop, dry mix",
    "Lofi Beat": "lofi hip hop beat with vinyl crackle, 75 BPM",
    "Glitchy IDM": "glitchy IDM with bitcrush textures, 100 BPM",
    "Modular Synth": "modular synth arpeggio in stereo, 90 BPM"
}

# --- Audio generation ---
# Similar to `generate_audio_batch` from the HF example: loop per prompt for stochastic diversity
stop_requested = False

def stop_generation():
    global stop_requested
    stop_requested = True
    return gr.update(value="🛑 Stop requested...")

prompt_history = []

def generate_audio(
    prompts, neg_prompt, start_sec, duration_sec,
    steps, cfg, sampler, sigma_min, sigma_max,
    sr, _batch_mode, upload, mix
):
    global stop_requested, prompt_history
    stop_requested = False
    lines = [p.strip() for p in prompts.splitlines() if p.strip()]
    if not lines:
        return None, None, None, None, "No prompts provided."

    previews, files, statuses = [], [], []
    start_total = time.time()

    for i, prompt in enumerate(lines):
        if stop_requested:
            statuses.append("🛑 Generation stopped by user.")
            break

        # reseed for diversity
        torch.manual_seed(int(time.time() * 1e6) % (2**32))

        # build positive conditioning list
        cond_pos = [{
            "prompt": prompt,
            "seconds_start": start_sec,
            "seconds_total": duration_sec,
            "weight": 1.0 - mix
        }]
        if upload and mix > 0:
            cond_pos.append({
                "audio_filepath": upload,
                "seconds_start": start_sec,
                "seconds_total": duration_sec,
                "weight": mix
            })

        # build negative conditioning list
        cond_neg = []
        if neg_prompt and neg_prompt.strip():
            cond_neg = [{
                "prompt": neg_prompt,
                "seconds_start": start_sec,
                "seconds_total": duration_sec,
                "weight": 1.0
            }]

                # combine positive + negative entries into single conditioning list
        cond = cond_pos.copy()
        if neg_prompt and neg_prompt.strip():
            cond.append({
                "prompt": neg_prompt,
                "seconds_start": start_sec,
                "seconds_total": duration_sec,
                "weight": -cfg
            })

                # generate
        try:
            out = generate_diffusion_cond(
                model=model,
                steps=steps,
                cfg_scale=cfg,
                conditioning=cond,
                sample_size=int(duration_sec * sr),
                sigma_min=sigma_min,
                sigma_max=sigma_max,
                sampler_type=sampler,
                device=device
            )
        except RuntimeError as e:
            # fallback for models that do not support negative conditioning (shape mismatch)
            err = str(e)
            if 'size of tensor a' in err or 'Conditioner key' in err:
                statuses.append('⚠️ Negative prompt unsupported, retrying without it.')
                # retry without negative weights
                cond = cond_pos.copy()
                out = generate_diffusion_cond(
                    model=model,
                    steps=steps,
                    cfg_scale=cfg,
                    conditioning=cond,
                    sample_size=int(duration_sec * sr),
                    sigma_min=sigma_min,
                    sigma_max=sigma_max,
                    sampler_type=sampler,
                    device=device
                )
            else:
                raise

        # post-process audio (single sample)
        audio = rearrange(out, "b d n -> d (b n)").float().cpu()
        audio = audio / (audio.abs().max() + 1e-5)
        audio_int16 = audio.clamp(-1, 1).mul(32767).to(torch.int16)

        elapsed = time.time() - start_total
        safe = "_".join(prompt.lower().split())[:25]
        fname = f"{int(elapsed)}s_{safe}.wav"
        path = os.path.join(tempfile.gettempdir(), fname)
        torchaudio.save(path, audio_int16, sr)

        previews.append(path)
        files.append(path)
        entry = f"{prompt}  ⏱ {elapsed:.1f}s"
        prompt_history.append(entry)
        statuses.append(entry)

    # pad preview list to 3
    while len(previews) < 3:
        previews.append(None)

    # package into ZIP
    zip_path = os.path.join(tempfile.gettempdir(), f"audio_{int(time.time())}.zip")
    with zipfile.ZipFile(zip_path, "w") as zf:
        for f in files:
            zf.write(f, os.path.basename(f))

    return previews[0], previews[1], previews[2], zip_path, "".join(statuses)

# --- UI layout ---
with gr.Blocks(title="Stable Audio Offline") as ui:
    gr.Markdown("## 🎵 Stable Audio (Offline)")
    with gr.Row():
        gen_btn = gr.Button("Generate")
        stop_btn = gr.Button("🛑 Stop", variant="stop")
        device_md = gr.Markdown(f"**Device:** `{device}`")
        ckpt_dd = gr.Dropdown(label="Checkpoint", choices=ckpt_options, value=current_ckpt)
        status_tb = gr.Textbox(label="Status", value=model_status, interactive=False)
    with gr.Row():
        with gr.Column(scale=3):
            preset_dd = gr.Dropdown(label="Preset", choices=[""] + list(PRESETS.keys()))
            prompt_tb = gr.Textbox(label="Prompts (one per line)", lines=4)
            preset_dd.change(lambda p: PRESETS.get(p, ""), preset_dd, prompt_tb)
            neg_tb = gr.Textbox(label="Negative Prompt", lines=2)
            start_sl = gr.Slider(0, 60, value=0, label="Start (s)")
            dur_sl = gr.Slider(1, 240, value=10, label="Duration (s)")
            steps_sl = gr.Slider(20, 250, value=100, label="Steps")
            cfg_sl = gr.Slider(1, 12, value=7, label="CFG Scale")
            samp_dd = gr.Dropdown(label="Sampler", choices=["dpmpp-3m-sde","dpmpp-2m","euler","heun","lms"], value="dpmpp-3m-sde")
            smin_sl = gr.Slider(0.0, 1.0, value=0.3, label="Sigma Min")
            smax_sl = gr.Slider(0.0, 1000.0, value=500.0, label="Sigma Max")
            sr_dd = gr.Dropdown(label="Sample Rate", choices=[16000,22050,32000,44100,48000], value=default_sample_rate)
            batch_cb = gr.Checkbox(label="Batch mode", value=False)
            audio_up = gr.Audio(label="Upload Audio", type="filepath")
            mix_sl = gr.Slider(0.0, 1.0, value=0.5, label="Audio Mix")
        with gr.Column():
            aud1 = gr.Audio(label="Preview 1", type="filepath")
            aud2 = gr.Audio(label="Preview 2", type="filepath")
            aud3 = gr.Audio(label="Preview 3", type="filepath")
            zip_dl = gr.File(label="Download ZIP")
            hist = gr.Textbox(label="History", lines=10)
        ckpt_dd.change(load_model, inputs=[ckpt_dd], outputs=[status_tb])
    gen_btn.click(
        generate_audio,
        inputs=[
            prompt_tb,
            neg_tb,
            start_sl,
            dur_sl,
            steps_sl,
            cfg_sl,
            samp_dd,
            smin_sl,
            smax_sl,
            sr_dd,
            batch_cb,
            audio_up,
            mix_sl
        ],
        outputs=[aud1, aud2, aud3, zip_dl, hist]
    )
    stop_btn.click(stop_generation, inputs=[], outputs=[hist])

if __name__ == '__main__':
    ui.queue().launch(share=False, server_name="0.0.0.0", server_port=7860)
