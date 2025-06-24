import os
import re
import json
import random
import socket
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
# default sample rate; will be overwritten by model_config
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
    m.eval()
    # skipping torch.compile to preserve model attributes
    try:
        m = torch.compile(m)
    except Exception as e:
        print("Warning: torch.compile() failed, running without compilation.", e)
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
stop_requested = False
prompt_history = []

def stop_generation():
    global stop_requested
    stop_requested = True
    return gr.update(value="🛑 Stop requested...")


def generate_audio(
    prompts, neg_prompt, start_sec, duration_sec,
    steps, cfg, sampler, sigma_min, sigma_max,
    sr, upload, mix
):
    global stop_requested, prompt_history
    stop_requested = False
    lines = [p.strip() for p in prompts.splitlines() if p.strip()]
    if not lines:
        return None, None, None, None, "No prompts provided."

    previews, files, statuses = [], [], []
    start_total = time.time()

    for prompt in lines:
        if stop_requested:
            statuses.append("🛑 Generation stopped by user.")
            break

        # reseed for diversity
        seed = random.randint(0, 2**31 - 1)
        print(f"🔀 reseeded with random seed {seed}")

        # positive conditioning
        cond_pos = [{
            "prompt": prompt,
            "seconds_start": start_sec,
            "seconds_total": duration_sec,
            "weight": 1.0
        }]

        # prepare init audio if mixing
        init_audio_tuple = None
        # init_noise_level: fraction of original noise (1=full original, 0=none)
        init_noise_level = mix
        if upload and mix > 0:
            wav, wav_sr = torchaudio.load(upload)
            if wav_sr != sr:
                wav = torchaudio.functional.resample(wav, orig_freq=wav_sr, new_freq=sr)
            wav = wav.mean(dim=0, keepdim=True)
            desired_len = int(duration_sec * sr)
            if wav.shape[1] < desired_len:
                pad = desired_len - wav.shape[1]
                wav = torch.nn.functional.pad(wav, (0, pad))
            else:
                wav = wav[:, :desired_len]
            init_audio_tuple = (sr, wav.to(device))

        # negative conditioning
        cond_neg = None
        if neg_prompt and neg_prompt.strip():
            cond_neg = [{
                "prompt": neg_prompt,
                "seconds_start": start_sec,
                "seconds_total": duration_sec
            }]

        # generate audio
        try:
            with torch.cuda.amp.autocast():
                out = generate_diffusion_cond(
                    model=model,
                    steps=steps,
                    cfg_scale=cfg,
                    conditioning=cond_pos,
                    negative_conditioning=cond_neg,
                    sample_size=int(duration_sec * sr),
                    sigma_min=sigma_min,
                    sigma_max=sigma_max,
                    sampler_type=sampler,
                    device=device,
                    seed=seed,
                    init_audio=init_audio_tuple,
                    init_noise_level=init_noise_level
                )
        except RuntimeError as e:
            err = str(e)
            if 'size of tensor a' in err or 'Conditioner key' in err:
                statuses.append('⚠️ Negative prompt unsupported, retrying without it.')
                with torch.cuda.amp.autocast():
                    out = generate_diffusion_cond(
                        model=model,
                        steps=steps,
                        cfg_scale=cfg,
                        conditioning=cond_pos,
                        sample_size=int(duration_sec * sr),
                        sigma_min=sigma_min,
                        sigma_max=sigma_max,
                        sampler_type=sampler,
                        device=device,
                        seed=seed,
                        init_audio=init_audio_tuple,
                        init_noise_level=init_noise_level
                    )
            else:
                raise

        # post-process
        audio = rearrange(out, "b d n -> d (b n)").float().cpu()
        audio = audio / (audio.abs().max() + 1e-5)
        audio_int16 = audio.clamp(-1, 1).mul(32767).to(torch.int16)

        elapsed = time.time() - start_total
        safe = re.sub(r'[^\w]+', '_', prompt.lower()).strip('_')
        fname = f"{int(elapsed)}s_{safe}.wav"
        path = os.path.join(tempfile.gettempdir(), fname)
        torchaudio.save(path, audio_int16, sr)

        previews.append(path)
        files.append(path)
        entry = f"{prompt}  ⏱ {elapsed:.1f}s"
        prompt_history.append(entry)
        statuses.append(entry)

    while len(previews) < 3:
        previews.append(None)

    # package zip
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
            dur_sl = gr.Slider(1, 60, value=30, label="Duration (s)")
            steps_sl = gr.Slider(20, 250, value=100, label="Steps")
            cfg_sl = gr.Slider(1, 12, value=7, label="CFG Scale")
            samp_dd = gr.Dropdown(label="Sampler", choices=["dpmpp-3m-sde","dpmpp-2m","euler","heun","lms"], value="dpmpp-3m-sde")
            smin_sl = gr.Slider(0.0, 1.0, value=0.3, label="Sigma Min")
            smax_sl = gr.Slider(0.0, 1000.0, value=500.0, label="Sigma Max")
            sr_dd = gr.Dropdown(label="Sample Rate", choices=[16000,22050,32000,44100], value=default_sample_rate)
            audio_up = gr.Audio(label="Upload Audio", type="filepath")
            mix_sl = gr.Slider(0.1, 100.0, value=50.0, step=0.1, label="Audio Mix")
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
            audio_up,
            mix_sl
        ],
        outputs=[aud1, aud2, aud3, zip_dl, hist]
    )
    stop_btn.click(stop_generation, inputs=[], outputs=[hist])

def get_local_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("10.255.255.255", 1))
        IP = s.getsockname()[0]
    except Exception:
        IP = "127.0.0.1"
    finally:
        s.close()
    return IP

if __name__ == '__main__':
    local_ip = get_local_ip()
    print(f"🌐 Serving Gradio interface on: http://{local_ip}:7880")
    ui.queue().launch(share=False, server_name=local_ip, server_port=7880)

