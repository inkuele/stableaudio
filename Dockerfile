# Base image with PyTorch + CUDA 11.8
FROM pytorch/pytorch:2.2.0-cuda11.8-cudnn8-runtime

# Optional: set non-root user
# RUN useradd -m appuser && chown -R appuser /app
# USER appuser

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git ffmpeg libsndfile1-dev \
    && apt-get clean

# Set workdir
WORKDIR /app

# Copy your Python code and other necessary files
COPY . /app

# Install Python dependencies
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# If you're using local encoders or models, ensure they are copied above
# and your script points to the correct paths

# Default command to run your script (edit as needed)
CMD ["python", "run_gradio_offline.py"]

