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

# Load model
model, model_config = get_pretrained_model("stabilityai/stable-audio-open-1.0")
model = model.to(device)
sample_rate = model_config["sample_rate"]

# Style presets for easy access
PRESETS = {
    "Ambient": "ambient drone, reverb tails, 60 BPM",
    "Tech House Loop": "128 BPM tech house drum loop, dry mix",
    "Lofi Beat": "lofi hip hop beat with vinyl crackle, 75 BPM",
    "Glitchy IDM": "glitchy IDM with bitcrush textures, 100 BPM",
    "Modular Synth": "modular synth arpeggio in stereo, 90 BPM"
}

prompt_history = []

def generate_audio(prompt, duration_sec, steps, cfg_scale, sampler_type):
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
        sampler_type=sampler_type,
        device=device
    )

    # Process output
    output = rearrange(output, "b d n -> d (b n)")
    output = output.to(torch.float32).div(torch.max(torch.abs(output))).clamp(-1, 1).mul(32767).to(torch.int16).cpu()

    # Save to temp WAV file
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as fp:
        torchaudio.save(fp.name, output, sample_rate)
        file_path = fp.name

    # Save prompt history
    prompt_history.append(prompt)
    return file_path, "\n".join(reversed(prompt_history[-10:]))

def set_preset(preset):
    return PRESETS[preset]

with gr.Blocks(title="🎵 Stable Audio Generator") as demo:
    gr.Markdown("## 🎵 Stable Audio Generator")
    gr.Markdown("Generate audio using Stability AI's open diffusion model")

    with gr.Row():
        with gr.Column():
            preset_dropdown = gr.Dropdown(label="Style Preset", choices=list(PRESETS.keys()), interactive=True)
            prompt_input = gr.Textbox(label="Prompt", placeholder="e.g. Ambient drone at 110 BPM")
            duration_slider = gr.Slider(1, 30, value=10, label="Duration (seconds)")
            steps_slider = gr.Slider(20, 250, value=100, label="Sampling Steps")
            cfg_slider = gr.Slider(1, 12, value=7, label="CFG Scale")
            sampler_dropdown = gr.Dropdown(label="Sampler Type", choices=[
                "dpmpp-3m-sde", "dpmpp-2m", "euler", "heun", "lms"
            ], value="dpmpp-3m-sde")
            generate_btn = gr.Button("Generate")

        with gr.Column():
            audio_output = gr.Audio(label="Generated Audio", type="filepath")
            download_btn = gr.File(label="Download WAV")
            history_box = gr.Textbox(label="Prompt History (last 10)", lines=10, interactive=False)

    # Bind interactions
    preset_dropdown.change(set_preset, inputs=preset_dropdown, outputs=prompt_input)
    generate_btn.click(
        fn=generate_audio,
        inputs=[prompt_input, duration_slider, steps_slider, cfg_slider, sampler_dropdown],
        outputs=[audio_output, history_box]
    )
    audio_output.change(lambda x: x, inputs=audio_output, outputs=download_btn)

if __name__ == "__main__":
    demo.launch()

