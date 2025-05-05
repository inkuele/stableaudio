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
    global model, model_config, default_sample_rate
    model_root = find_model_subdir("models")
    config_path = os.path.join(model_root, "model_config.json")
    ckpt_path = os.path.join(model_root, "model.safetensors")
    with open(config_path, "r") as f:
        model_config = json.load(f)
    model = create_model_from_config(model_config)
    state = load_ckpt_state_dict(ckpt_path)
    model.load_state_dict(state)
    model = model.to(device)
    default_sample_rate = model_config.get("sample_rate", 44100)

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
    prompts_text,
    negative_prompt,
    start_sec,
    duration_sec,
    steps,
    cfg_scale,
    sampler_type,
    sigma_min,
    sigma_max,
    out_sample_rate,
    uploaded_audio,
    audio_mix,
    progress=gr.Progress(track_tqdm=True)
):
    global stop_requested
    stop_requested = False
    previews = []
    statuses = []
    t_total_start = time.time()

    prompts = [p.strip() for p in prompts_text.splitlines() if p.strip()]
    for idx, prompt in enumerate(prompts):
        if stop_requested:
            statuses.append("🛑 Stopped by user.")
            break
        progress((idx+1)/len(prompts), desc=f"Generating: {prompt[:30]}…")
        t_start = time.time()
        sample_size = int(duration_sec * default_sample_rate)
        conditioning = []
        # Positive prompt
        if prompt and audio_mix < 1.0:
            conditioning.append({
                "prompt": prompt,
                "seconds_start": start_sec,
                "seconds_total": duration_sec,
                "weight": 1.0 - audio_mix
            })
        # Negative prompt
        if negative_prompt:
            conditioning.append({
                "prompt": negative_prompt,
                "seconds_start": start_sec,
                "seconds_total": duration_sec,
                "weight": -cfg_scale
            })
        # Initial audio
        if uploaded_audio and audio_mix > 0.0:
            conditioning.append({
                "audio_filepath": uploaded_audio,
                "seconds_start": start_sec,
                "seconds_total": duration_sec,
                "weight": audio_mix
            })
        # Inference
        with torch.no_grad():
            output = generate_diffusion_cond(
                model=model,
                steps=steps,
                cfg_scale=cfg_scale,
                conditioning=conditioning,
                sample_size=sample_size,
                sigma_min=sigma_min,
                sigma_max=sigma_max,
                sampler_type=sampler_type,
                device=device,
            )
        # Convert output [b, d, n] -> [d, b*n]
        audio_tensor = rearrange(output, "b d n -> d (b n)").float().cpu().numpy()
        # Normalize
        audio_tensor = audio_tensor / (abs(audio_tensor).max() + 1e-5)
        previews.append(audio_tensor)
        # Status
        elapsed = time.time() - t_start
        total_elapsed = time.time() - t_total_start
        status = f"{prompt}  ⏱ {elapsed:.1f}s | total: {total_elapsed:.1f}s"
        prompt_history.append(status)
        statuses.append(status)
    # Write files and zip
    file_paths = []
    for audio in previews:
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        torchaudio.save(tmp.name, torch.tensor(audio), out_sample_rate)
        file_paths.append(tmp.name)
    zip_path = tempfile.NamedTemporaryFile(delete=False, suffix=".zip").name
    with zipfile.ZipFile(zip_path, "w") as zf:
        for p in file_paths:
            zf.write(p, arcname=os.path.basename(p))
    # Ensure 3 previews slots
    while len(file_paths) < 3:
        file_paths.append(None)
    return (*file_paths[:3], zip_path, "\n".join(statuses))

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
            # Preset and prompts
            preset_dropdown = gr.Dropdown(
                label="Style Preset",
                choices=[""] + list(PRESETS.keys()),
                interactive=True
            )
            prompt_input = gr.Textbox(label="Prompts (one per line)", lines=6)
            preset_dropdown.change(set_preset, inputs=[preset_dropdown], outputs=[prompt_input])
            # Additional GUI options from original
            negative_input = gr.Textbox(label="Negative Prompt (optional)", lines=2)
            start_slider = gr.Slider(0.0, 60.0, value=0.0, step=0.5, label="Seconds Start")
            duration_slider = gr.Slider(1, 240, value=10, label="Duration (seconds)")
            steps_slider = gr.Slider(20, 250, value=100, label="Sampling Steps")
            cfg_slider = gr.Slider(1.0, 12.0, value=7.0, label="CFG Scale")
            sampler_dropdown = gr.Dropdown(
                label="Sampler",
                choices=["dpmpp-3m-sde", "dpmpp-2m", "euler", "heun", "lms"],
                value="dpmpp-3m-sde"
            )
            sigma_min_slider = gr.Slider(0.0, 1.0, value=model_config.get("sigma_min", 0.3), label="Sigma Min")
            sigma_max_slider = gr.Slider(0.0, 1000.0, value=model_config.get("sigma_max", 500.0), label="Sigma Max")
            sample_rate_dropdown = gr.Dropdown(
                label="Output Sample Rate",
                choices=[16000, 22050, 32000, 44100, 48000],
                value=default_sample_rate
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

    generate_btn.click(
        generate_audio_batch,
        inputs=[
            prompt_input,
            negative_input,
            start_slider,
            duration_slider,
            steps_slider,
            cfg_slider,
            sampler_dropdown,
            sigma_min_slider,
            sigma_max_slider,
            sample_rate_dropdown,
            audio_upload,
            audio_mix_slider
        ],
        outputs=[audio1, audio2, audio3, zip_download, history_box]
    )
    stop_btn.click(stop_generation, outputs=[history_box])

if __name__ == "__main__":
    demo.queue().launch(share=False, server_name="0.0.0.0", server_port=7860)

