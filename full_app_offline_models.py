import os
import tempfile

import gradio as gr
import torch
import torchaudio
from diffusers import StableAudioPipeline

# device selection
def get_device():
    if torch.backends.mps.is_available():
        return "mps"
    elif torch.cuda.is_available() and torch.version.cuda is not None:
        return "cuda"
    else:
        return "cpu"

device = get_device()

# path to the specific StabilityAI Diffusers model
def get_model_path():
    return os.path.join("models", "stabilityai__stable-audio-open-1.0")

# load and prepare pipeline once at startup
MODEL_PATH = get_model_path()
pipe = StableAudioPipeline.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.float16 if device.startswith("cuda") else torch.float32,
    local_files_only=True,
)
pipe = pipe.to(device)

# inference function
def infer(prompt, negative_prompt, steps, cfg_scale, seed, length_s):
    """
    Run the StabilityAI StableAudioPipeline loaded from MODEL_PATH.
    """
    generator = torch.Generator(device=device).manual_seed(int(seed))
    output = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=int(steps),
        guidance_scale=float(cfg_scale),
        generator=generator,
        audio_length_s=int(length_s),
    )
    audios = output.audios  # List[np.ndarray]
    sample_rate = output.sample_rate

    # save results to temporary files
    out_paths = []
    for idx, np_audio in enumerate(audios):
        td = tempfile.mkdtemp()
        fn = os.path.join(td, f"output_{idx}.wav")
        tensor = torch.from_numpy(np_audio)
        if tensor.ndim == 1:
            tensor = tensor.unsqueeze(0)
        torchaudio.save(fn, tensor.cpu(), sample_rate=sample_rate)
        out_paths.append(fn)
    return out_paths

# build Gradio interface

def interface():
    with gr.Blocks() as demo:
        gr.Markdown("## Stable Audio – Offline Local Model")

        prompt = gr.Textbox(label="Prompt", placeholder="e.g. a calm beach at sunset")
        negative = gr.Textbox(label="Negative Prompt", placeholder="e.g. noise, distortion")
        steps = gr.Slider(label="Steps", minimum=1, maximum=150, value=50, step=1)
        cfg = gr.Slider(label="CFG Scale", minimum=0.1, maximum=30, value=7.5, step=0.1)
        seed = gr.Number(label="Seed", value=42, precision=0)
        length = gr.Slider(label="Audio Length (s)", minimum=1, maximum=60, value=10, step=1)

        generate_btn = gr.Button("Generate")
        output_audio = gr.Audio(label="Generated Audio", type="filepath")

        generate_btn.click(
            fn=infer,
            inputs=[prompt, negative, steps, cfg, seed, length],
            outputs=[output_audio]
        )

    return demo

if __name__ == "__main__":
    interface().queue().launch()
