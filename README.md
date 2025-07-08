
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
In our case, we installed CUDA 12.9 
```
https://developer.nvidia.com/cuda-downloads?target_os=Windows&target_arch=x86_64&target_version=11&target_type=exe_local
```
and then we installed pytorth 12.1 version:

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

Copy the `models` and `encoders` folders from your workshop leader, or download them yourself:

```bash
python3.10 download_models.py
python3.10 download_encoders.py
```

> **Note:** The model checkpoints are large and may take time to download.

## D. Launch the Web Interface

Once models and encoders are in place, run:

```bash
python3.8 run_gradio_offline.py
```

Open your browser at the address printed in the console to start generating audio!


## E. Docker version


### Create the Docker container

```bash
docker build -t my-stableaudio-app .
```

### Run your app 

```bash
docker run --gpus all -p 7880:7880 -v $(pwd):/app my-stableaudio-app
```

- --gpus all: enables GPU access
- -p 7880:7880: exposes the Gradio or web port (if relevant)
- -v $(pwd):/app: mounts the current directory into the container


### Install the NVIDIA Container Toolkit

```
https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html
```

and restart docker:

```bash
sudo systemctl restart docker
```

### Test GPU access inside Docker

```bash
docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi
```

### Now run your Stable Audio app

```bash
docker run --gpus all -p 7880:7880 -v $(pwd)/models:/app/models my-stableaudio-app
```

### Pushing Docker Images to GitHub Registry(ghcr.io)

Login with a github account with write access:
```bash
docker login --username inkuele --password your_personal_access_token ghcr.io
```

Use the docker build command to build the Docker image and tag the image with the GHCR repository URL and version
```bash
docker build . -t ghcr.io/inkuele/actions-runner-controller-ghcr:latest              
```

Push the image to GHCR using the docker push command.
```bash
docker push ghcr.io/inkuele/actions-runner-controller-ghcr:latest
```



