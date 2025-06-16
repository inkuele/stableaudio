
This repository and workshop preparation have been created for the [Inküle](https://www.inkuele.de).

# Stable Audio Offline GUI

This repository provides a simple-to-use interface for the Stable Audio open model and other fine-tuned variants, running fully offline with local models and encoders.

## A. Overview

* **Purpose**: Create a standalone web GUI for generating audio using the Stable Audio open model (and any fine-tuned checkpoints) without external API calls.
* **Features**:

  * Local T5-based text encoding
  * Diffusion-based audio generation via `stable-audio-tools`
  * Gradio-powered interface for prompt input and preview

## B. Installation

### 1. Install Python 3.10

#### Linux

```bash
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt update
sudo apt install python3.10-full
```

#### macOS

```bash
brew install pyenv
pyenv install 3.10
cd ~/my-project
pyenv local 3.10

```
### Windows
```bash
https://www.python.org/ftp/python/3.10.0/python-3.10.0-amd64.exe
```

### 2. Create and Activate Virtual Environment

```bash
python3.10 -m venv venv
source venv/bin/activate
```

#### Windows
```bash
py -3.10 -m venv venv
.\venv\Scripts\Activate.ps1


if you get this error on windows:
 .\venv\Scripts\Activate.ps1
.\venv\Scripts\Activate.ps1 : Die Datei "C:\Users\anast\OneDrive\Dokumente\git\stableaudio\venv\Scripts\Activate.ps1"
kann nicht geladen werden, da die Ausführung von Skripts auf diesem System deaktiviert ist. Weitere Informationen
finden Sie unter "about_Execution_Policies" (https:/go.microsoft.com/fwlink/?LinkID=135170).
In Zeile:1 Zeichen:1
+ .\venv\Scripts\Activate.ps1
+ ~~~~~~~~~~~~~~~~~~~~~~~~~~~
    + CategoryInfo          : Sicherheitsfehler: (:) [], PSSecurityException
    + FullyQualifiedErrorId : UnauthorizedAccess

either temporary allow scripts:
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process

or permanently allow scripts:
Set-ExecutionPolicy RemoteSigned -Scope CurrentUser
```


### 3. Install Dependencies

You must install CUDA for your graphic card.

### Windows

For Windows you have to specify the installation of pytorch libraries for your CUDA version
In our case, we installed CUDA 12.9 and we installed pytorth 12.1 version:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```
Then

### for every system:

```bash
pip3.10 install stable-audio-tools
```

### 4. Run the gradio script

```bash
python3.10 run_gradio_offline.py
```

### Windows
```bash
python .\run_gradio_offline.py
```

### 5. (macOS Only) Fix `soundfile` Errors

If you receive errors related to the `soundfile` library:

```bash
brew install libsndfile
pip uninstall soundfile
pip install soundfile
```

## C. Models & Encoders

Copy the `models` folders from your workshop leader, or download them yourself:

```bash
python3.10 download_models.py
```

> **Note:** The model checkpoints are large and may take time to download.

## D. Launch the Web Interface

Once models and encoders are in place, run:

```bash
python3.8 run_gradio_offline_with_t5.py
```

Open your browser at the address printed in the console to start generating audio!

## E. Packaging as a Standalone Executable (Script Only) - NOT WORKING YET, only for reference

To distribute only the Python script (assuming users will provide their own local `models/` and `encoders/` folders), use PyInstaller. You have two approaches:

### E.1 Automatic Collection (Convenient)

1. **Install PyInstaller**

   ```bash
   pip install pyinstaller
   ```
2. **Build including Gradio data**

   ```bash
   pyinstaller \
     --name stableaudio_gui \
     --onefile \
     --collect-data gradio_client \
     run_gradio_offline_with_t5.py
   ```

   * `--collect-data gradio_client` ensures files like `version.txt` are bundled.

### E.2 Manual Data Inclusion (Safer)

1. **Locate the missing file**

   ```bash
    python -c "import gradio_client, os; print(os.path.join(os.path.dirname(gradio_client.__file__),'version.txt'))"
    python -c "import gradio_client, os; print(os.path.join(os.path.dirname(gradio_client.__file__),'types.json'))"
    python -c "import gradio, os; print(os.path.join(os.path.dirname(gradio.__file__),'version.txt'))"
   ```
2. **Build with explicit `add-data`**

   ```bash
    pyinstaller \
      --name stableaudio_gui \
      --onefile \
      --add-data "/path/to/gradio_client/version.txt:gradio_client" \
      --add-data "/path/to/gradio_client/types.json:gradio_client" \
      --add-data "/path/to/gradio/version.txt:gradio" \
      run_gradio_offline_with_t5.py
   ```

    * Replace each /path/to/... with the paths printed.
    * This ensures all required metadata files are included.

---

*Last updated: May 6, 2025*

