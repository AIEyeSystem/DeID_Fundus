FROM nvcr.io/nvidia/pytorch:21.08-py3

COPY deid /deid

WORKDIR /deid

RUN apt update && \
    apt install -y build-essential gcc g++ && \
    apt clean && rm -rf /var/lib/apt/lists/*

RUN pip install \
    torch==1.11.0+cu113 \
    torchvision==0.12.0+cu113 \
    torchaudio==0.11.0 \
    --extra-index-url https://download.pytorch.org/whl/cu113 && \
    pip install -r requirements.txt


CMD sh scripts/start.sh

