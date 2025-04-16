import torch
import tempfile
import gradio as gr
import time
import soundfile as sf
import numpy as np
from diffusers import StableAudioPipeline

# Device selection
if torch.backends.mps.is_available():
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

# Load model
pipe = StableAudioPipeline.from_pretrained(
    "stabilityai/stable-audio-open-1.0",
    torch_dtype=torch.float16 if device == "cuda" else torch.float32
).to(device)

sample_rate = pipe.vae.sample_rate

# Style presets
PRESETS = {
    "Ambient": "ambient drone, reverb tails, 60 BPM",
    "Tech House Loop": "128 BPM tech house drum loop, dry mix",
    "Lofi Beat": "lofi hip hop beat with vinyl crackle, 75 BPM",
    "Glitchy IDM": "glitchy IDM with bitcrush textures, 100 BPM",
    "Modular Synth": "modular synth arpeggio in stereo, 90 BPM"
}

prompt_history = []

def generate_audio(prompt, duration_sec, seed):
    start_time = time.time()

    generator = torch.Generator(device=device).manual_seed(seed)

    result = pipe(
        prompt=prompt,
        negative_prompt="low quality",
        num_inference_steps=200,
        audio_end_in_s=duration_sec,
        num_waveforms_per_prompt=1,
        generator=generator,
    )

    audio = result.audios[0]
    output = audio.T.float().cpu().numpy()

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as fp:
        sf.write(fp.name, output, sample_rate)
        file_path = fp.name

    elapsed_time = time.time() - start_time
    prompt_history.append(f"{prompt} (⏱ {elapsed_time:.1f}s)")
    return file_path, "\n".join(reversed(prompt_history[-10:])), f"✅ Done in {elapsed_time:.1f} seconds"

def set_preset(preset):
    return PRESETS.get(preset, "")

# Gradio UI
with gr.Blocks(title="🎵 Stable Audio Generator (Diffusers)") as demo:
    gr.Markdown("## 🎵 Stable Audio Generator (Diffusers)")
    gr.Markdown("Generate audio using Stability AI's open diffusion model via the Diffusers pipeline.")

    with gr.Row():
        with gr.Column():
            preset_dropdown = gr.Dropdown(label="Style Preset (optional)", choices=[""] + list(PRESETS.keys()), interactive=True)
            prompt_input = gr.Textbox(label="Prompt", placeholder="e.g. Ambient drone at 110 BPM")
            duration_slider = gr.Slider(1, 20, value=10, label="Duration (seconds)")
            seed_input = gr.Number(value=0, label="Seed")
            generate_btn = gr.Button("Generate")

        with gr.Column():
            audio_output = gr.Audio(label="Generated Audio", type="filepath")
            download_btn = gr.File(label="Download WAV")
            history_box = gr.Textbox(label="Prompt History (last 10)", lines=10, interactive=False)
            timing_info = gr.Textbox(label="Status", interactive=False)

    preset_dropdown.change(set_preset, inputs=preset_dropdown, outputs=prompt_input)
    generate_btn.click(
        fn=generate_audio,
        inputs=[prompt_input, duration_slider, seed_input],
        outputs=[audio_output, history_box, timing_info]
    )
    audio_output.change(lambda x: x, inputs=audio_output, outputs=download_btn)

if __name__ == "__main__":
    demo.launch()

