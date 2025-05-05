import os
import json
tempfile = __import__('tempfile')
import torch
import gradio as gr
import soundfile as sf
from einops import rearrange
from stable_audio_tools.models.factory import create_model_from_config
from stable_audio_tools.inference.generation import generate_diffusion_cond

# Device detection to favor MPS on Apple Silicon

def get_device():
    if torch.backends.mps.is_available():
        return "mps"
    elif torch.cuda.is_available() and torch.version.cuda is not None:
        return "cuda"
    else:
        return "cpu"

device = get_device()
print(f"Using device: {device}")


def find_model_subdir(base_dir: str) -> str:
    """
    Locate the first subdirectory under base_dir that contains a model_config.json.
    If base_dir itself contains model_config.json, returns base_dir.
    """
    if os.path.isfile(os.path.join(base_dir, "model_config.json")):
        return base_dir
    for entry in os.listdir(base_dir):
        subdir = os.path.join(base_dir, entry)
        if os.path.isdir(subdir) and os.path.isfile(os.path.join(subdir, "model_config.json")):
            return subdir
    raise FileNotFoundError(
        f"Could not find a subdirectory with model_config.json under {base_dir}"
    )


def load_audio_model(audio_model_dir: str = "models"):
    """
    Load Stable Audio model and its config dict from a local folder.
    Automatically picks up nested model directories.
    """
    model_dir = find_model_subdir(audio_model_dir)
    config_path = os.path.join(model_dir, "model_config.json")
    with open(config_path, "r") as f:
        model_config = json.load(f)
    model = create_model_from_config(model_config).to(device)
    return model, model_config


def generate_audio(
    prompt: str,
    model,
    model_config: dict,
    num_steps: int = 100,
    cfg_scale: float = 7.0,
):
    """
    Generate audio from a text prompt and write to a temporary WAV file.
    Returns the file path for Gradio to play.
    """
    # Prepare conditioning
    duration_s = model_config["sample_size"] / model_config["sample_rate"]
    conditioning = [{
        "prompt": prompt,
        "seconds_start": 0,
        "seconds_total": duration_s,
    }]

    # Amp context for MPS vs CUDA
    amp_context = torch.autocast(device_type=device) if device == "mps" else torch.cuda.amp.autocast()
    with amp_context:
        output = generate_diffusion_cond(
            model,
            steps=num_steps,
            cfg_scale=cfg_scale,
            conditioning=conditioning,
            sample_size=model_config["sample_size"],
            sigma_min=model_config.get("sigma_min", 0.3),
            sigma_max=model_config.get("sigma_max", 500),
            sampler_type=model_config.get("sampler", "dpmpp-3m-sde"),
            device=device,
        )
    # Convert to numpy
    audio = rearrange(output, "b d n -> d (b n)").float().cpu().numpy()
    # Write to temp WAV
    sample_rate = model_config["sample_rate"]
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    sf.write(tmp.name, audio.T, sample_rate)
    return tmp.name


def main():
    model, model_config = load_audio_model("models")

    with gr.Blocks() as demo:
        gr.Markdown("# Stable Audio (Offline)")
        with gr.Row():
            with gr.Column():
                prompt_input = gr.Textbox(
                    label="Text Prompt",
                    placeholder="Enter description...",
                    lines=2,
                )
                steps = gr.Slider(
                    minimum=1,
                    maximum=200,
                    step=1,
                    value=100,
                    label="Inference Steps"
                )
                cfg = gr.Slider(
                    minimum=1.0,
                    maximum=20.0,
                    step=0.5,
                    value=7.0,
                    label="CFG Scale"
                )
                gen_btn = gr.Button("Generate")
            with gr.Column():
                out_audio = gr.Audio(type="file", label="Generated Audio")

        gen_btn.click(
            fn=lambda p, s, c: generate_audio(p, model, model_config, s, c),
            inputs=[prompt_input, steps, cfg],
            outputs=[out_audio]
        )

    demo.launch(share=False, server_name="0.0.0.0", server_port=7860)


if __name__ == "__main__":
    main()
