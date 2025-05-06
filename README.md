# Stable Audio Offline GUI

This repository provides a simple-to-use interface for the Stable Audio open model and other fine-tuned variants, running fully offline with local models and encoders.

This repository and workshop preparation have been created for the [Inküle](https://www.inkuele.de).

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

## E. Packaging as a Standalone Executable (Script Only)

To distribute only the Python script (assuming users will provide their own local `models/` and `encoders/` folders), use PyInstaller:

1. **Install PyInstaller**

   ```bash
   pip install pyinstaller
   ```

2. **Package your script**

   ```bash
    pyinstaller \
      --name stableaudio_gui \
      --onefile \
      --collect-data gradio_client \
      --collect-data gradio \
      run_gradio_offline_with_t5.py
   ```

   * `--onefile` bundles Python, your script, and all required libraries into a single executable.
   * You **do not** include `models/` or `encoders/`—those directories must exist beside the executable at runtime.

3. **Prepare your distribution folder**

   After building, ship the executable alongside the `models/` and `encoders/` directories:

   ```
   dist/
   ├── stableaudio_gui      # your bundled executable
   models/                  # user-provided models folder
   encoders/                # user-provided encoders folder
   ```

4. **Run the Executable**

   * On **Linux/macOS**:

     ```bash
     ./dist/stableaudio_gui
     ```
   * On **Windows**:

     ```powershell
     .\dist\stableaudio_gui.exe
     ```

Users will need to place their own `models/` and `encoders/` folders in the same location as the executable. This keeps the distribution lightweight and assumes local assets are managed separately.

---

*Last updated: May 6, 2025*

