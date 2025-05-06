import os
import json
import torch
from einops import rearrange
import torchaudio
import tempfile
import gradio as gr
import time
import zipfile

# === New: load local T5 encoder ===
from encoders import tokenizer, encoder
# ====================================

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

# Load and initialize the Stable Audio model
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
    file_paths = []
    t_total_start = time.time()

    # Generate embeddings with T5 encoder
    raw_prompts = [p.strip() for p in prompts_text.splitlines() if p.strip()]
    prompt_embs = []
    enc_device = next(encoder.parameters()).device
    for p in raw_prompts:
        toks = tokenizer(p, return_tensors="pt").input_ids.to(enc_device)
        emb = encoder(input_ids=toks).last_hidden_state.to(device)
        prompt_embs.append(emb)
    neg_emb = None
    if negative_prompt:
        neg_toks = tokenizer(negative_prompt, return_tensors="pt").input_ids.to(enc_device)
        neg_emb = encoder(input_ids=neg_toks).last_hidden_state.to(device)

    for idx, emb in enumerate(prompt_embs):
        if stop_requested:
            statuses.append("🛑 Stopped by user.")
            break

        progress((idx+1)/len(prompt_embs), desc=f"Generating prompt {idx+1}/{len(prompt_embs)}…")
        conditioning = []

        conditioning.append({
            "prompt": raw_prompts[idx],
            "text_embeddings": emb,
            "seconds_start": start_sec,
            "seconds_total": duration_sec,
            "weight": 1.0
        })
        if neg_emb is not None:
            conditioning.append({
                "prompt": negative_prompt,
                "text_embeddings": neg_emb,
                "seconds_start": start_sec,
                "seconds_total": duration_sec,
                "weight": -cfg_scale
            })
        if uploaded_audio and audio_mix > 0.0:
            conditioning.append({
                "audio_filepath": uploaded_audio,
                "seconds_start": start_sec,
                "seconds_total": duration_sec,
                "weight": audio_mix
            })

        with torch.no_grad():
            output = generate_diffusion_cond(
                model=model,
                steps=steps,
                cfg_scale=cfg_scale,
                conditioning=conditioning,
                sample_size=out_sample_rate * duration_sec,
                sigma_min=sigma_min,
                sigma_max=sigma_max,
                sampler_type=sampler_type,
                device=device,
            )

        # Post-process
        audio_tensor = rearrange(output, "b d n -> d (b n)").float().cpu().numpy()
        audio_tensor /= (abs(audio_tensor).max() + 1e-5)
        previews.append(audio_tensor)

        # Save preview WAV (2D tensor: channels x samples)
        out_path = os.path.join(tempfile.gettempdir(), f"preview_{idx}.wav")
        torchaudio.save(out_path, torch.tensor(audio_tensor), out_sample_rate)
        file_paths.append(out_path)

        statuses.append(f"{raw_prompts[idx]}  ⏱ {time.time() - t_total_start:.1f}s")

    # Zip outputs
    zip_path = os.path.join(tempfile.gettempdir(), f"stablediff_audio_{int(time.time())}.zip")
    with zipfile.ZipFile(zip_path, "w") as zf:
        for p in file_paths:
            zf.write(p, arcname=os.path.basename(p))

    while len(file_paths) < 3:
        file_paths.append(None)

    return (*file_paths[:3], zip_path, "\n".join(statuses))

# Stop callback
def stop_generation():
    global stop_requested
    stop_requested = True
    return gr.update(value="🛑 Stop requested…")

# Build Gradio UI
with gr.Blocks(title="🎵 Stable Audio Generator") as demo:
    gr.Markdown("## 🎵 Stable Audio (Offline)")
    with gr.Row():
        with gr.Column():
            preset_dropdown = gr.Dropdown(label="Style Preset", choices=[""] + list(PRESETS.keys()))
            prompt_input = gr.Textbox(label="Prompts (one per line)", lines=6)
            preset_dropdown.change(lambda p: PRESETS.get(p, ""), preset_dropdown, prompt_input)
            negative_input = gr.Textbox(label="Negative Prompt (optional)", lines=2)
            start_slider = gr.Slider(0.0, 60.0, value=0.0, step=0.5, label="Seconds Start")
            duration_slider = gr.Slider(1, 240, value=10, label="Duration (seconds)")
            steps_slider = gr.Slider(20, 250, value=100, label="Sampling Steps")
            cfg_slider = gr.Slider(1.0, 12.0, value=7.0, label="CFG Scale")
            sampler_dropdown = gr.Dropdown(label="Sampler", choices=["dpmpp-3m-sde","dpmpp-2m","euler","heun","lms"], value="dpmpp-3m-sde")
            sigma_min_slider = gr.Slider(0.0,1.0,value=model_config.get("sigma_min",0.3),label="Sigma Min")
            sigma_max_slider = gr.Slider(0.0,1000.0,value=model_config.get("sigma_max",500.0),label="Sigma Max")
            sample_rate_dropdown = gr.Dropdown(label="Output Sample Rate",choices=[16000,22050,32000,44100,48000],value=default_sample_rate)
            audio_upload = gr.Audio(label="Upload Audio (optional)",type="filepath")
            audio_mix_slider = gr.Slider(0.0,1.0,value=0.5,label="Prompt / Audio Mix")
            generate_btn = gr.Button("Generate")
            stop_btn = gr.Button("🛑 Stop", variant="stop")
        with gr.Column():
            audio1 = gr.Audio(label="Preview Clip 1",type="filepath")
            audio2 = gr.Audio(label="Preview Clip 2",type="filepath")
            audio3 = gr.Audio(label="Preview Clip 3",type="filepath")
            zip_download = gr.File(label="Download All as ZIP")
            history_box = gr.Textbox(label="Prompt History",lines=10)

    generate_btn.click(generate_audio_batch, inputs=[prompt_input,negative_input,start_slider,duration_slider,steps_slider,cfg_slider,sampler_dropdown,sigma_min_slider,sigma_max_slider,sample_rate_dropdown,audio_upload,audio_mix_slider], outputs=[audio1,audio2,audio3,zip_download,history_box])
    stop_btn.click(stop_generation, outputs=[history_box])

if __name__ == "__main__":
    demo.queue().launch(share=False, server_name="0.0.0.0", server_port=7860)
