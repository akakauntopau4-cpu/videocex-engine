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

# Модель НЕ качаем во время сборки (в сборочной среде нет доступа к HuggingFace).
# Она скачается при первом запуске воркера — там сеть есть.

COPY handler.py /handler.py
CMD ["python", "-u", "/handler.py"]
