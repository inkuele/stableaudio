# Stable Audio Offline Interface – User Guide

This interface provides an **offline audio generation tool** using Stability AI's open-source diffusion model. Prompts and/or audio files can be used to condition the generation, with full parameter control and preview/download functionality.

---

## 🧩 Features Overview

### ✅ Model Management

- **Checkpoint Selector (`Checkpoint`)**
  - Choose from available model checkpoints in the `models/` folder.
  - Each entry must include a `.ckpt` file and a corresponding `model_config.json`.
  - Example: `stabilityai__stable-audio-open-1.0/stable-audio.ckpt`

- **Status Display (`Status`)**
  - Shows model name, sample rate, and active hardware device (CPU, CUDA, MPS).

---

### ✏️ Prompt Controls

- **Prompt Input (`Prompts`)**
  - Multi-line textbox, one prompt per line.
  - These serve as **positive conditioning text**.
  - ✅ *Recommended:* Describe content, style, and tempo  
    _Example:_ `glitchy IDM with bitcrush textures, 100 BPM`

- **Preset Selector (`Preset`)**
  - Automatically fills the prompt box with one of the following:
    - Ambient
    - Tech House Loop
    - Lofi Beat
    - Glitchy IDM
    - Modular Synth

- **Negative Prompt (`Negative Prompt`)**
  - Optional textbox to **suppress unwanted content** (e.g. `no vocals`, `no drums`).
  - ⚠️ If not supported by the model, generation will retry without negative prompts.

---

## 🎛 Generation Parameters

| Parameter | Description | Range | ✅ Recommended |
|----------|-------------|-------|----------------|
| **Start (s)** | Start time (in seconds) when using uploaded audio | `0–59` | `0` |
| **Duration (s)** | Length of generated audio | `1–60` | `<= 60` (longer can cause instability) |
| **Steps** | Number of denoising diffusion steps | `20–250` | `80–120` for quality/performance balance |
| **CFG Scale** | Guidance strength for prompt conditioning | `1–12` | `6–9` |
| **Sampler** | Sampling algorithm used in diffusion | `dpmpp-3m-sde`, `dpmpp-2m`, `euler`, `heun`, `lms` | `dpmpp-3m-sde` (default, stable) |
| **Sigma Min** | Minimum sigma (noise level) | `0.0–1.0` | `0.3` |
| **Sigma Max** | Maximum sigma | `0.0–1000.0` | `400–600` |
| **Sample Rate** | Output audio rate (Hz) | `16000–44100` | `44100` |

---

### 🎧 Audio Conditioning

- **Upload Audio (`Upload Audio`)**
  - Use a `.wav` file (mono or stereo) for conditioning.
  - Combines with prompt if `Audio Mix > 0`.

- **Audio Mix (`Audio Mix`)**
  - Ratio between prompt and audio conditioning.
  - `0.0` = only prompt, `1.0` = only uploaded audio.
  - ✅ *Recommended:* `0.3–0.6`

---

### ⚙️ Execution & Output

- **Generate Button**
  - Starts generation for all entered prompts (processed sequentially).

- **Audio Previews**
  - Up to 3 preview players for the latest outputs.

- **ZIP Download**
  - All generated `.wav` files packed in one ZIP archive.

- **Prompt History**
  - Running log of past prompts and durations.

---

## 📁 Folder Structure Requirements

- Place model checkpoints in `models/{model_name}/`, including:
  - `model_config.json`
  - `{model_name}.ckpt`

- The local T5 encoder must be located in the `encoders/` directory with:
  - `tokenizer` and `encoder` modules loaded via `from encoders import tokenizer, encoder`

---

## ✅ Best Practices

- Keep prompt descriptions specific and structured (e.g., *"style + content + tempo"*).
- Use audio durations under 60 seconds to ensure stability.
- Start with default settings and gradually experiment with advanced parameters (sigma, sampler).
- Use high sample rates (`44100`) for final output; use lower ones for rapid previews.

---

