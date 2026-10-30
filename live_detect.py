"""live_detect.py: Canlı nesne tespiti. Kullanım: python live_detect.py [effdet|ssd] → http://<PI_IP>:5000"""
import sys
import time
import threading
import numpy as np
import cv2
from flask import Flask, Response
from picamera2 import Picamera2
from ai_edge_litert.interpreter import Interpreter

MODELS = {
    "ssd":    "models/ssd-mobilenet-v1-tflite-default-v1.tflite",
    "effdet": "models/lite-model_efficientdet_lite0_detection_metadata_1.tflite",
}
MODEL = sys.argv[1] if len(sys.argv) > 1 else "effdet"   # varsayılan: EfficientDet
THRESHOLD = 0.5

# ── Model ve etiketler: bir kez yükle ────────────────────────────
labels = [line.strip() for line in open("models/coco_labels.txt")]
itp = Interpreter(model_path=MODELS[MODEL], num_threads=4)
itp.allocate_tensors()
inp = itp.get_input_details()[0]
IN_H, IN_W = inp['shape'][1], inp['shape'][2]


def read_outputs():
    """Çıkışları SIRAYA değil İÇERİĞE göre tanı (iki modelde de çalışır)."""
    boxes = classes = scores = count = None
    for o in itp.get_output_details():
        t = itp.get_tensor(o['index'])[0] if len(o['shape']) > 1 else itp.get_tensor(o['index'])
        if t.ndim == 2 and t.shape[-1] == 4:
            boxes = t
        elif t.size == 1:
            count = int(t.ravel()[0])
        elif np.allclose(t, np.round(t)):
            classes = t
        else:
            scores = t
    return boxes, classes, scores, count


def color_for(class_id):
    """Her sınıfa sabit ve farklı bir renk (aynı sınıf hep aynı renk)."""
    rng = np.random.default_rng(class_id)
    return tuple(int(c) for c in rng.integers(80, 256, 3))


# ── Kamera: tam görüş alanı, 640x480 (çizimler net görünsün) ─────
picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(
    main={"size": (640, 480), "format": "BGR888"},
    sensor={"output_size": (1640, 1232)}))
picam2.start()
time.sleep(2)

latest_jpeg = None
lock = threading.Lock()


def loop():
    global latest_jpeg
    fps, t_prev = 0.0, time.perf_counter()
    while True:
        frame = picam2.capture_array()                              # (480, 640, 3) RGB
        H, W = frame.shape[:2]

        x = cv2.resize(frame, (IN_W, IN_H))[None].astype(inp['dtype'])
        itp.set_tensor(inp['index'], x)
        t0 = time.perf_counter()
        itp.invoke()
        ms = (time.perf_counter() - t0) * 1000
        boxes, classes, scores, count = read_outputs()

        bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        n = 0
        for i in range(count):
            if scores[i] < THRESHOLD:
                continue
            n += 1
            ymin, xmin, ymax, xmax = np.clip(boxes[i], 0, 1)
            x1, y1, x2, y2 = int(xmin * W), int(ymin * H), int(xmax * W), int(ymax * H)   # normalize → piksel
            cid = int(classes[i])
            color = color_for(cid)
            text = f"{labels[cid]} {scores[i]*100:.0f}%"
            cv2.rectangle(bgr, (x1, y1), (x2, y2), color, 2)
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
            cv2.rectangle(bgr, (x1, max(y1 - th - 8, 0)), (x1 + tw + 6, max(y1, th + 8)), color, -1)
            cv2.putText(bgr, text, (x1 + 3, max(y1 - 5, th + 3)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2)

        now = time.perf_counter()
        fps = 0.9 * fps + 0.1 / (now - t_prev)
        t_prev = now
        cv2.rectangle(bgr, (0, 0), (W, 28), (0, 0, 0), -1)
        cv2.putText(bgr, f"{MODEL} | {n} nesne | cikarim {ms:.1f} ms | {fps:.0f} FPS", (8, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        ok, buf = cv2.imencode(".jpg", bgr)
        if ok:
            with lock:
                latest_jpeg = buf.tobytes()


app = Flask(__name__)


@app.route("/")
def index():
    return f'<h1>🎯 Canlı Nesne Tespiti ({MODEL})</h1><img src="/stream" width="640" height="480">'


@app.route("/stream")
def stream():
    def gen():
        while True:
            with lock:
                jpg = latest_jpeg
            if jpg:
                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n"
            time.sleep(0.03)
    return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=frame")


if __name__ == "__main__":
    threading.Thread(target=loop, daemon=True).start()
    try:
        app.run(host="0.0.0.0", port=5000, threaded=True)
    finally:
        picam2.stop()
        picam2.close()
