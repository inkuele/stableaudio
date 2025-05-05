import os
import json
import torch
from einops import rearrange
import torchaudio
import tempfile
import gradio as gr
import time
import zipfile
from stable_audio_tools.models.factory import create_model_from_config
from stable_audio_tools.models.utils import load_ckpt_state_dict
from stable_audio_tools.inference.generation import generate_diffusion_cond

# Global interrupt flag
stop_requested = False

# Device selection: MPS -> CUDA -> CPU
def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available() and torch.version.cuda is not None:
        return torch.device("cuda")
    else:
        return torch.device("cpu")

device = get_device()
print(f"Using device: {device}")

# Helper to find the model directory
def find_model_subdir(base_dir: str) -> str:
    if os.path.isfile(os.path.join(base_dir, "model_config.json")):
        return base_dir
    for entry in os.listdir(base_dir):
        subdir = os.path.join(base_dir, entry)
        if os.path.isdir(subdir) and os.path.isfile(os.path.join(subdir, "model_config.json")):
            return subdir
    raise FileNotFoundError(f"No Stable Audio model found under '{base_dir}'")

# Model loading from local folder
def load_model():
    global model, model_config, sample_rate
    model_root = find_model_subdir("models")
    config_path = os.path.join(model_root, "model_config.json")
    ckpt_path = os.path.join(model_root, "model.safetensors")
    # Load config JSON
    with open(config_path, "r") as f:
        model_config = json.load(f)
    # Instantiate and load checkpoint
    model = create_model_from_config(model_config)
    state = load_ckpt_state_dict(ckpt_path)
    model.load_state_dict(state)
    model = model.to(device)
    sample_rate = model_config.get("sample_rate", 44100)

# Initialize the single offline model
load_model()

# Prompt style presets
PRESETS = {
    "Ambient": "ambient drone, reverb tails, 60 BPM",
    "Tech House Loop": "128 BPM tech house drum loop, dry mix",
    "Lofi Beat": "lofi hip hop beat with vinyl crackle, 75 BPM",
    "Glitchy IDM": "glitchy IDM with bitcrush textures, 100 BPM",
    "Modular Synth": "modular synth arpeggio in stereo, 90 BPM"
}

# History of prompts
prompt_history = []

# Audio generation function
def generate_audio_batch(
    prompts_text, duration_sec, steps, cfg_scale, sampler_type,
    uploaded_audio, audio_mix, progress=gr.Progress(track_tqdm=True)
):
    global stop_requested
    stop_requested = False
    previews = []
    statuses = []
    start_total = time.time()

    prompts = [p.strip() for p in prompts_text.splitlines() if p.strip()]
    for idx, prompt in enumerate(prompts):
        if stop_requested:
            statuses.append("🛑 Stopped by user.")
            break
        progress((idx+1)/len(prompts), desc=f"Generating: {prompt[:30]}…")
        start = time.time()
        sample_size = int(duration_sec * sample_rate)
        conditioning = []
        if prompt and audio_mix < 1.0:
            conditioning.append({
                "prompt": prompt,
                "seconds_start": 0,
                "seconds_total": duration_sec,
                "weight": 1.0 - audio_mix
            })
        if uploaded_audio and audio_mix > 0.0:
            conditioning.append({
                "audio_filepath": uploaded_audio,
                "seconds_start": 0,
                "seconds_total": duration_sec,
                "weight": audio_mix
            })
                                # AMP context: use CUDA autocast on CUDA, else no grad (MPS autocast unsupported)
        if device.type == "cuda":
            amp_ctx = torch.cuda.amp.autocast()
        else:
            amp_ctx = torch.no_grad()
        with amp_ctx:
            output = generate_diffusion_cond(
                model=model,
                steps=steps,
                cfg_scale=cfg_scale,
                conditioning=conditioning,
                sample_size=sample_size,
                sigma_min=model_config.get("sigma_min", 0.3),
                sigma_max=model_config.get("sigma_max", 500),
                sampler_type=sampler_type,
                device=device,
            )
        # Convert to PCM16
        audio = rearrange(output, "b d n -> d (b n)").float().cpu()
        norm = audio.abs().max().clamp(min=1e-5)
        audio = (audio / norm * 32767).to(torch.int16)
        # Save each as WAV
        tmpf = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        torchaudio.save(tmpf.name, audio, sample_rate)
        previews.append(tmpf.name)
        elapsed = time.time() - start
        total_elapsed = time.time() - start_total
        status = f"{prompt}  ⏱ {elapsed:.1f}s | total: {total_elapsed:.1f}s"
        prompt_history.append(status)
        statuses.append(status)
    # Bundle all outputs as ZIP
    zip_path = tempfile.NamedTemporaryFile(delete=False, suffix=".zip").name
    with zipfile.ZipFile(zip_path, "w") as zf:
        for p in previews:
            zf.write(p, arcname=os.path.basename(p))
    # Provide up to 3 previews
    while len(previews) < 3:
        previews.append(None)
    return (*previews[:3], zip_path, "\n".join(statuses))

# Stop callback
def stop_generation():
    global stop_requested
    stop_requested = True
    return gr.update(value="🛑 Stop requested…")

# Preset setter
def set_preset(preset):
    return PRESETS.get(preset, "")

# Build Gradio GUI
with gr.Blocks(title="🎵 Stable Audio Generator") as demo:
    gr.Markdown("## 🎵 Stable Audio (Offline)")
    with gr.Row():
        with gr.Column():
            preset_dropdown = gr.Dropdown(label="Style Preset", choices=[""] + list(PRESETS.keys()), interactive=True)
            prompt_input = gr.Textbox(label="Prompts (one per line)", lines=6)
            duration_slider = gr.Slider(1, 240, value=10, label="Duration (seconds)")
            steps_slider = gr.Slider(20, 250, value=100, label="Sampling Steps")
            cfg_slider = gr.Slider(1, 12, value=7, label="CFG Scale")
            sampler_dropdown = gr.Dropdown(
                label="Sampler",
                choices=["dpmpp-3m-sde", "dpmpp-2m", "euler", "heun", "lms"],
                value="dpmpp-3m-sde"
            )
            audio_upload = gr.Audio(label="Upload Audio (optional)", type="filepath")
            audio_mix_slider = gr.Slider(0.0, 1.0, value=0.5, label="Prompt / Audio Mix")
            generate_btn = gr.Button("Generate")
            stop_btn = gr.Button("🛑 Stop", variant="stop")
        with gr.Column():
            audio1 = gr.Audio(label="Preview Clip 1", type="filepath")
            audio2 = gr.Audio(label="Preview Clip 2", type="filepath")
            audio3 = gr.Audio(label="Preview Clip 3", type="filepath")
            zip_download = gr.File(label="Download All as ZIP")
            history_box = gr.Textbox(label="Prompt History", lines=10)

    preset_dropdown.change(set_preset, inputs=[preset_dropdown], outputs=[prompt_input])
    generate_btn.click(
        generate_audio_batch,
        inputs=[
            prompt_input,
            duration_slider,
            steps_slider,
            cfg_slider,
            sampler_dropdown,
            audio_upload,
            audio_mix_slider
        ],
        outputs=[audio1, audio2, audio3, zip_download, history_box]
    )
    stop_btn.click(stop_generation, outputs=[history_box])

if __name__ == "__main__":
    demo.queue().launch(share=False, server_name="0.0.0.0", server_port=7860)

