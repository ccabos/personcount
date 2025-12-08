"""
YOLO Person Counter - Hugging Face Spaces Edition
"""

import gradio as gr
import numpy as np
import cv2
from ultralytics import YOLO

# Model cache
models = {}

def get_model(model_name):
    """Load and cache YOLO model."""
    if model_name not in models:
        print(f"Loading model: {model_name}")
        models[model_name] = YOLO(model_name)
        print(f"Model {model_name} loaded!")
    return models[model_name]

# Pre-load default model
get_model("yolov8n.pt")


def count_persons(image, model_name, confidence):
    """Detect and count persons in an image."""
    if image is None:
        return None, "Bitte laden Sie ein Bild hoch."

    try:
        # Load model
        model = get_model(model_name)

        # Run inference
        results = model(image, conf=confidence, classes=[0], verbose=False)
        result = results[0]

        # Get detections
        boxes = result.boxes
        person_count = len(boxes)

        # Draw annotations
        annotated = result.plot()

        # Add count overlay
        h, w = annotated.shape[:2]
        text = f"{person_count} Personen"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = max(0.8, min(w, h) / 500)
        thickness = max(2, int(font_scale * 2))

        (text_w, text_h), _ = cv2.getTextSize(text, font, font_scale, thickness)

        # Draw background
        cv2.rectangle(annotated, (10, 10), (text_w + 30, text_h + 30), (0, 0, 0), -1)
        cv2.rectangle(annotated, (10, 10), (text_w + 30, text_h + 30), (0, 255, 0), 2)
        cv2.putText(annotated, text, (20, text_h + 20), font, font_scale, (0, 255, 0), thickness)

        # Result text
        if person_count == 0:
            result_text = "Keine Personen erkannt."
        else:
            avg_conf = np.mean(boxes.conf.cpu().numpy()) * 100
            result_text = f"{person_count} Personen erkannt\nModell: {model_name}\nKonfidenz: {avg_conf:.1f}%"

        return annotated, result_text

    except Exception as e:
        return None, f"Fehler: {str(e)}"


# Interface with model choice (Gradio 3.x syntax)
demo = gr.Interface(
    fn=count_persons,
    inputs=[
        gr.Image(label="Bild hochladen", type="numpy"),
        gr.Dropdown(
            choices=["yolov8n.pt", "yolov8s.pt", "yolov8m.pt"],
            value="yolov8n.pt",
            label="Modell (n=schnell, m=genau)"
        ),
        gr.Slider(minimum=0.1, maximum=0.9, value=0.25, step=0.05, label="Konfidenz")
    ],
    outputs=[
        gr.Image(label="Ergebnis"),
        gr.Textbox(label="Anzahl")
    ],
    title="🎻 Orchester Personenzähler",
    description="Laden Sie ein Foto hoch, um Personen zu zählen. Verwendet YOLOv8.",
    allow_flagging="never"
)

if __name__ == "__main__":
    demo.launch()
