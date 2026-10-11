# Образ для serverless-движка «Видеоцех» на RunPod.
FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

# ffmpeg для чтения/записи видео
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg && rm -rf /var/lib/apt/lists/*

# библиотеки
RUN pip install --no-cache-dir \
    runpod \
    transformers \
    timm \
    einops \
    kornia \
    accelerate \
    opencv-python-headless

# заранее качаем веса модели в образ, чтобы не качать на каждом запуске
RUN python -c "from transformers import AutoModelForImageSegmentation; AutoModelForImageSegmentation.from_pretrained('ZhengPeng7/BiRefNet_dynamic', trust_remote_code=True)"

COPY handler.py /handler.py
CMD ["python", "-u", "/handler.py"]
