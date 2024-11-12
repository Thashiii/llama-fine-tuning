# Use a base image with Python and CUDA support for GPU usage (modify as needed)
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

# Set environment variables for Python
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Install system dependencies
RUN apt-get update && \
    apt-get install -y git wget && \
    rm -rf /var/lib/apt/lists/*

# Install Python dependencies, including Unsloth
RUN pip install --upgrade pip && \
    pip install unsloth torch transformers datasets accelerate && \
    pip uninstall unsloth -y && pip install --upgrade --no-cache-dir "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git" && \
    pip install -U bitsandbytes

# Copy the fine-tuning script and any other files (e.g., config files)
COPY . /workspace/.
# Adjust path if needed
#COPY data /workspace/data

# Set the working directory
WORKDIR /workspace

# Default command to run the Unsloth fine-tuning script
CMD ["python", "finetuning.py"]