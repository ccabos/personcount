"""
YOLO Person Counter - Hugging Face Spaces Edition

Supports two inference modes:
  - Standard YOLO (fast)
  - SAHI sliced inference (slower, much better for large images with many
    small persons, e.g. wide orchestra or audience photos)
"""

import gradio as gr
import numpy as np
import cv2
from ultralytics import YOLO

# Model caches
yolo_models = {}
sahi_models = {}


def get_yolo(model_name):
    """Load and cache a plain YOLO model."""
    if model_name not in yolo_models:
        print(f"Loading YOLO model: {model_name}")
        yolo_models[model_name] = YOLO(model_name)
        print(f"YOLO model {model_name} loaded!")
    return yolo_models[model_name]


def get_sahi(model_name, confidence):
    """Load and cache a SAHI detector wrapping the YOLO weights."""
    from sahi import AutoDetectionModel

    key = (model_name, round(confidence, 3))
    if key in sahi_models:
        return sahi_models[key]

    last_error = None
    for model_type in ("ultralytics", "yolov8"):
        try:
            print(f"Loading SAHI model ({model_type}): {model_name}")
            detector = AutoDetectionModel.from_pretrained(
                model_type=model_type,
                model_path=model_name,
                confidence_threshold=confidence,
                device="cpu",
            )
            sahi_models[key] = detector
            print("SAHI model loaded!")
            return detector
        except Exception as exc:
            last_error = exc
            continue
    raise RuntimeError(f"Could not initialize SAHI detector: {last_error}")


def _draw(image, bboxes, scores):
    annotated = image.copy()
    color = (0, 255, 0)
    for bbox, score in zip(bboxes, scores):
        x1, y1, x2, y2 = (int(v) for v in bbox)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        label = f"{score:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(annotated, (x1, max(0, y1 - th - 4)),
                      (x1 + tw + 4, y1), color, -1)
        cv2.putText(annotated, label, (x1 + 2, max(th, y1 - 2)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    return annotated


# Pre-load default model
get_yolo("yolov8n.pt")


def count_persons(image, model_name, confidence, use_sahi, slice_size, slice_overlap):
    """Detect and count persons in an image."""
    if image is None:
        return None, "Bitte laden Sie ein Bild hoch."

    try:
        if use_sahi:
            from sahi.predict import get_sliced_prediction

            detector = get_sahi(model_name, confidence)
            prediction = get_sliced_prediction(
                image,
                detector,
                slice_height=int(slice_size),
                slice_width=int(slice_size),
                overlap_height_ratio=float(slice_overlap),
                overlap_width_ratio=float(slice_overlap),
                verbose=0,
            )
            bboxes, scores = [], []
            for obj in prediction.object_prediction_list:
                if obj.category.id != 0:  # COCO person
                    continue
                bboxes.append(obj.bbox.to_xyxy())
                scores.append(float(obj.score.value))

            person_count = len(bboxes)
            # Gradio passes RGB; OpenCV draw also expects 3-channel arrays
            annotated = _draw(image, bboxes, scores)
            mode = f"SAHI ({int(slice_size)}px, {float(slice_overlap):.0%} overlap)"
        else:
            model = get_yolo(model_name)
            results = model(image, conf=confidence, classes=[0], verbose=False)
            result = results[0]
            boxes = result.boxes
            person_count = len(boxes)
            annotated = result.plot()
            scores = boxes.conf.cpu().numpy().tolist() if person_count else []
            mode = "Standard YOLO"

        # Add count overlay
        h, w = annotated.shape[:2]
        text = f"{person_count} Personen"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = max(0.8, min(w, h) / 500)
        thickness = max(2, int(font_scale * 2))
        (text_w, text_h), _ = cv2.getTextSize(text, font, font_scale, thickness)

        cv2.rectangle(annotated, (10, 10), (text_w + 30, text_h + 30), (0, 0, 0), -1)
        cv2.rectangle(annotated, (10, 10), (text_w + 30, text_h + 30), (0, 255, 0), 2)
        cv2.putText(annotated, text, (20, text_h + 20), font, font_scale,
                    (0, 255, 0), thickness)

        if person_count == 0:
            result_text = f"Keine Personen erkannt.\nModus: {mode}"
        else:
            avg_conf = (sum(scores) / len(scores)) * 100 if scores else 0.0
            result_text = (
                f"{person_count} Personen erkannt\n"
                f"Modell: {model_name}\n"
                f"Modus: {mode}\n"
                f"Konfidenz: {avg_conf:.1f}%"
            )

        return annotated, result_text

    except Exception as e:
        return None, f"Fehler: {str(e)}"


demo = gr.Interface(
    fn=count_persons,
    inputs=[
        gr.Image(label="Bild hochladen", type="numpy"),
        gr.Dropdown(
            choices=["yolov8n.pt", "yolov8s.pt", "yolov8m.pt"],
            value="yolov8n.pt",
            label="Modell (n=schnell, m=genau)"
        ),
        gr.Slider(minimum=0.1, maximum=0.9, value=0.25, step=0.05, label="Konfidenz"),
        gr.Checkbox(value=False, label="SAHI aktivieren (für große Bilder mit vielen kleinen Personen)"),
        gr.Slider(minimum=256, maximum=1280, value=640, step=64, label="SAHI Kachelgröße (px)"),
        gr.Slider(minimum=0.0, maximum=0.5, value=0.2, step=0.05, label="SAHI Überlappung"),
    ],
    outputs=[
        gr.Image(label="Ergebnis"),
        gr.Textbox(label="Anzahl")
    ],
    title="🎻 Orchester Personenzähler",
    description=(
        "Laden Sie ein Foto hoch, um Personen zu zählen. Verwendet YOLOv8.\n"
        "Für hochauflösende Bilder mit vielen kleinen Personen "
        "(z.B. ganzes Orchester oder Zuschauerraum) **SAHI aktivieren**."
    ),
    allow_flagging="never"
)

if __name__ == "__main__":
    demo.launch()
