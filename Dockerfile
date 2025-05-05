# Use Ubuntu 22.04 as base image
FROM ubuntu:22.04

# Set timezone to avoid interactive prompt
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && \
    apt-get install -y tzdata && \
    ln -fs /usr/share/zoneinfo/Asia/Ho_Chi_Minh /etc/localtime && \
    dpkg-reconfigure -f noninteractive tzdata

# Update package list and install required dependencies
RUN apt-get update && \
    apt-get install -y \
    software-properties-common \
    curl \
    build-essential \
    zlib1g-dev \
    libncurses5-dev \
    libgdbm-dev \
    libnss3-dev \
    libssl-dev \
    libreadline-dev \
    libffi-dev \
    libbz2-dev \
    python3-dev \
    python3-pip

RUN pip install -U pip

# Install Rust
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

WORKDIR /app

COPY ./underthesea_core /app/underthesea_core
COPY ./underthesea /app/underthesea
COPY ./TTS /app/TTS

# Install underthesea_core
RUN pip install maturin && \
    cd underthesea_core && \
    maturin build --release && \
    cp target/wheels/*.whl /app/ && \
    pip install /app/*.whl

# Install underthesea
RUN cd underthesea && \
    python3 setup.py install

# Copy requirements and install project dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install TTS
RUN cd TTS && \
    pip install --no-cache-dir --ignore-installed -e .

# Download Japanese/Chinese tokenizer (necessary for unidic)
RUN python3 -m unidic download

# Copy the rest of the application
COPY . .

# Create necessary directories
RUN mkdir -p model output assets

# Expose port for FastAPI
EXPOSE 8000

# Command to run the application
CMD ["uvicorn", "vixtts_api:app", "--host", "0.0.0.0", "--port", "8000"]