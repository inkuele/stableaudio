import torch
from einops import rearrange
import torchaudio
import tempfile
import gradio as gr
import time
import os
import zipfile
from stable_audio_tools import get_pretrained_model
from stable_audio_tools.inference.generation import generate_diffusion_cond

# Global interrupt flag
stop_requested = False

# Device selection
if torch.backends.mps.is_available():
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

# Load model
model, model_config = get_pretrained_model("stabilityai/stable-audio-open-1.0")
model = model.to(device)
sample_rate = model_config["sample_rate"]

# Presets for convenience
PRESETS = {
    "Ambient": "ambient drone, reverb tails, 60 BPM",
    "Tech House Loop": "128 BPM tech house drum loop, dry mix",
    "Lofi Beat": "lofi hip hop beat with vinyl crackle, 75 BPM",
    "Glitchy IDM": "glitchy IDM with bitcrush textures, 100 BPM",
    "Modular Synth": "modular synth arpeggio in stereo, 90 BPM"
}

prompt_history = []

def generate_audio_batch(prompts_text, duration_sec, steps, cfg_scale, sampler_type, uploaded_audio, audio_mix, progress=gr.Progress(track_tqdm=True)):
    global stop_requested
    stop_requested = False

    prompts = [p.strip() for p in prompts_text.strip().split("\n") if p.strip()]
    audio_paths = []
    all_prompt_statuses = []
    total_start_time = time.time()

    for i, prompt in enumerate(prompts):
        if stop_requested:
            all_prompt_statuses.append("🛑 Stopped by user.")
            break

        progress((i + 1) / len(prompts), desc=f"Generating: {prompt[:30]}...")
        start_time = time.time()
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

        output = generate_diffusion_cond(
            model=model,
            steps=steps,
            cfg_scale=cfg_scale,
            conditioning=conditioning,
            sample_size=sample_size,
            sigma_min=0.3,
            sigma_max=500,
            sampler_type=sampler_type,
            device=device
        )

        output = rearrange(output, "b d n -> d (b n)")
        output = output.to(torch.float32).div(torch.max(torch.abs(output))).clamp(-1, 1).mul(32767).to(torch.int16).cpu()

        elapsed_time = time.time() - start_time
        safe_name = "_".join(prompt.lower().split())[:25]
        filename = f"{int(elapsed_time)}s_{safe_name}.wav"

        out_path = os.path.join(tempfile.gettempdir(), filename)
        torchaudio.save(out_path, output, sample_rate)
        audio_paths.append(out_path)

        total_so_far = time.time() - total_start_time
        prompt_entry = f"{prompt}  ⏱ {elapsed_time:.1f}s | total: {total_so_far:.1f}s"
        prompt_history.append(prompt_entry)
        all_prompt_statuses.append(prompt_entry)

    zip_path = tempfile.NamedTemporaryFile(delete=False, suffix=".zip").name
    with zipfile.ZipFile(zip_path, "w") as zipf:
        for path in audio_paths:
            zipf.write(path, arcname=os.path.basename(path))

    preview_paths = audio_paths[-3:] if len(audio_paths) >= 3 else audio_paths + [None] * (3 - len(audio_paths))
    return (*preview_paths, zip_path, "\n".join(all_prompt_statuses))

def set_preset(preset):
    return PRESETS.get(preset, "")

def stop_generation():
    global stop_requested
    stop_requested = True
    return gr.update(value="🛑 Stop requested...")

with gr.Blocks(title="🎵 Stable Audio Generator (with Stop)") as demo:
    gr.Markdown("## 🎵 Stable Audio Generator")
    gr.Markdown("Generate audio with optional prompts. You can stop generation between clips.")

    with gr.Row():
        with gr.Column():
            preset_dropdown = gr.Dropdown(label="Style Preset", choices=[""] + list(PRESETS.keys()), interactive=True)
            prompt_input = gr.Textbox(label="Prompts (one per line)", lines=8)
            duration_slider = gr.Slider(1, 240, value=10, label="Duration (seconds)")
            steps_slider = gr.Slider(20, 250, value=100, label="Sampling Steps")
            cfg_slider = gr.Slider(1, 12, value=7, label="CFG Scale")
            sampler_dropdown = gr.Dropdown(label="Sampler", choices=["dpmpp-3m-sde", "dpmpp-2m", "euler", "heun", "lms"], value="dpmpp-3m-sde")
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

    preset_dropdown.change(set_preset, inputs=preset_dropdown, outputs=prompt_input)

    generate_event = generate_btn.click(
        fn=generate_audio_batch,
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

    stop_btn.click(fn=stop_generation, inputs=[], outputs=[history_box])

if __name__ == "__main__":
    demo.queue().launch()
