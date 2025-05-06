# Stable Audio Offline GUI

This repository provides a simple-to-use interface for the Stable Audio open model and other fine-tuned variants, running fully offline with local models and encoders.

## A. Overview

* **Purpose**: Create a standalone web GUI for generating audio using the Stable Audio open model (and any fine-tuned checkpoints) without external API calls.
* **Features**:

  * Local T5-based text encoding
  * Diffusion-based audio generation via `stable-audio-tools`
  * Gradio-powered interface for prompt input and preview

## B. Installation

### 1. Install Python 3.8

#### Linux

```bash
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt update
sudo apt install python3.8-full
```

#### macOS

```bash
brew install pyenv
pyenv install 3.8.10
cd ~/my-project
pyenv local 3.8.10
```

### 2. Create and Activate Virtual Environment

```bash
python3.8 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip3.8 install https://github.com/buscon/stable-audio-tools/archive/refs/heads/main.zip
```

### 4. Run the Test Script

```bash
python3.8 stableaudio_test.py
```

### 5. (macOS Only) Fix `soundfile` Errors

If you receive errors related to the `soundfile` library:

```bash
brew install libsndfile
pip uninstall soundfile
pip install soundfile
```

## C. Models & Encoders

Copy the `models` and `encoders` folders from your workshop leader, or download them yourself:

```bash
python3.8 download_encoder.py
python3.8 download_models.py
```

> **Note:** The model checkpoints are large and may take time to download.

## D. Launch the Web Interface

Once models and encoders are in place, run:

```bash
python3.8 run_gradio_offline_with_t5.py
```

Open your browser at the address printed in the console to start generating audio!

## E. Packaging as a Standalone Installer

You can create a standalone executable or installer using **PyInstaller**, which bundles Python, your scripts, and dependencies into a single file or folder.

1. **Install PyInstaller**

   ```bash
   pip install pyinstaller
   ```

2. **Package your script**

   ```bash
   pyinstaller \
     --name stableaudio_gui \
     --onefile \
     --add-data "models:models" \
     --add-data "encoders:encoders" \
     --hidden-import transformers.tokenization_t5 \
     --hidden-import transformers.models.t5 \
     run_gradio_offline_with_t5.py
   ```

   * `--onefile` bundles everything into a single executable.
   * `--add-data` includes your `models` and `encoders` directories.
   * `--hidden-import` ensures T5 tokenizer/encoder modules are included.

3. **Run the Bundled App**

   * On **Linux/macOS**:

     ```bash
     ./dist/stableaudio_gui
     ```
   * On **Windows**:

     ```powershell
     .\dist\stableaudio_gui.exe
     ```

4. **Distribute**

   Share the single executable in `dist/` with users—no Python or external dependencies needed.

> **Tip:** For platform-specific installers (e.g., `.msi` on Windows or `.dmg` on macOS), wrap the PyInstaller output using tools like [WiX](https://wixtoolset.org/) or macOS's `hdiutil`.

---

*Last updated: May 6, 2025*

