import torch
from einops import rearrange
import torchaudio
import tempfile
import gradio as gr
from stable_audio_tools import get_pretrained_model
from stable_audio_tools.inference.generation import generate_diffusion_cond

if torch.backends.mps.is_available():
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"


# Load model once at startup
model, model_config = get_pretrained_model("stabilityai/stable-audio-open-1.0")
model = model.to(device)
sample_rate = model_config["sample_rate"]

def generate_audio(prompt, duration_sec, steps, cfg_scale):
    sample_size = int(duration_sec * sample_rate)

    conditioning = [{
        "prompt": prompt,
        "seconds_start": 0,
        "seconds_total": duration_sec
    }]

    output = generate_diffusion_cond(
        model=model,
        steps=steps,
        cfg_scale=cfg_scale,
        conditioning=conditioning,
        sample_size=sample_size,
        sigma_min=0.3,
        sigma_max=500,
        sampler_type="dpmpp-3m-sde",
        device=device
    )

    # Rearrange & normalize
    output = rearrange(output, "b d n -> d (b n)")
    output = output.to(torch.float32).div(torch.max(torch.abs(output))).clamp(-1, 1).mul(32767).to(torch.int16).cpu()

    # Save to temp WAV file
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as fp:
        torchaudio.save(fp.name, output, sample_rate)
        return fp.name

# Launch Gradio interface
gr.Interface(
    fn=generate_audio,
    inputs=[
        gr.Textbox(label="Prompt", placeholder="e.g. ambient drone, 110 BPM"),
        gr.Slider(1, 30, value=10, label="Duration (seconds)"),
        gr.Slider(20, 250, value=100, label="Steps"),
        gr.Slider(1, 12, value=7, label="CFG Scale"),
    ],
    outputs=gr.Audio(label="Generated Audio"),
    title="🎵 Stable Audio Generator",
    description="Generate music with Stability AI's open diffusion model"
).launch()

