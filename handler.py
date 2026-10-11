# handler.py — serverless-движок «Видеоцех» для RunPod.
# Принимает видео (base64), убирает фон (BiRefNet), ставит зелёный фон, 30 FPS,
# возвращает готовый MP4 (base64).
import base64, time, os, json, subprocess, tempfile
import numpy as np, cv2, torch
import torch.nn.functional as F
from transformers import AutoModelForImageSegmentation
import runpod

MODEL_ID = "ZhengPeng7/BiRefNet_dynamic"
DEV = "cuda"
FPS, SIZE, TS = 30, 1024, 0.15

print("Загружаю BiRefNet...")
model = AutoModelForImageSegmentation.from_pretrained(MODEL_ID, trust_remote_code=True).to(DEV).eval().half()
MEAN = torch.tensor([0.485, 0.456, 0.406], device=DEV).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225], device=DEV).view(1, 3, 1, 1)
print("Модель готова.")


def _pred(o):
    if hasattr(o, "logits"):
        o = o.logits
    while isinstance(o, (list, tuple)):
        o = o[-1]
    return o


def _process(inp, silent, outp):
    pr = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0",
                         "-show_entries", "stream=width,height", "-of", "json", inp],
                        capture_output=True, text=True)
    st = json.loads(pr.stdout)["streams"][0]
    W, H = int(st["width"]) // 2 * 2, int(st["height"]) // 2 * 2

    def frames():
        p = subprocess.Popen(["ffmpeg", "-v", "error", "-i", inp, "-vf", f"fps={FPS}",
                              "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                             stdout=subprocess.PIPE, bufsize=10 ** 8)
        n = W * H * 3
        while True:
            b = p.stdout.read(n)
            if len(b) < n:
                break
            yield np.frombuffer(b, np.uint8).reshape(H, W, 3)
        p.wait()

    wr = subprocess.Popen(["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                           "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-", "-an",
                           "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
                           "-pix_fmt", "yuv420p", silent], stdin=subprocess.PIPE)
    green = np.array([0, 255, 0], np.float32)
    prev = None
    nf = 0
    with torch.inference_mode():
        for f in frames():
            x = torch.from_numpy(f).to(DEV).permute(2, 0, 1).float().div(255).unsqueeze(0)
            x = F.interpolate(x, (SIZE, SIZE), mode="bilinear", align_corners=False)
            p = _pred(model(((x - MEAN) / STD).half())).sigmoid().float()
            p = F.interpolate(p, (H, W), mode="bilinear", align_corners=False)[0, 0].cpu().numpy()
            m = cv2.GaussianBlur(np.clip(p, 0, 1), (0, 0), 0.6)
            if prev is not None:
                m = m * (1 - TS) + prev * TS
            prev = m
            a = m[..., None]
            res = (f.astype(np.float32) * a + green * (1 - a)).clip(0, 255).astype(np.uint8)
            wr.stdin.write(res.tobytes())
            nf += 1
    wr.stdin.close()
    wr.wait()

    # вернуть оригинальный звук
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", silent, "-i", inp,
                    "-map", "0:v", "-map", "1:a?", "-c:v", "copy", "-c:a", "aac",
                    "-b:a", "192k", "-shortest", outp], check=True)
    return nf


def handler(job):
    inp = job.get("input", {})
    b64 = inp.get("video_base64")
    if not b64:
        return {"error": "Нет поля video_base64"}
    try:
        data = base64.b64decode(b64)
    except Exception as e:
        return {"error": f"Не удалось декодировать видео: {e}"}

    t0 = time.time()
    with tempfile.TemporaryDirectory() as d:
        src = os.path.join(d, "in.mp4")
        silent = os.path.join(d, "silent.mp4")
        dst = os.path.join(d, "out.mp4")
        with open(src, "wb") as fh:
            fh.write(data)
        try:
            nf = _process(src, silent, dst)
            out = open(dst, "rb").read()
        except Exception as e:
            return {"error": f"Ошибка обработки: {e}"}

    dt = round(time.time() - t0, 1)
    return {
        "video_base64": base64.b64encode(out).decode(),
        "frames": nf,
        "seconds": dt,
    }


runpod.serverless.start({"handler": handler})
