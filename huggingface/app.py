"""
YOLO Person Counter - Hugging Face Spaces Edition
Gradio-based web interface for person detection in orchestra/audience photos.
"""

import gradio as gr
import numpy as np
import cv2
from PIL import Image
from ultralytics import YOLO
import traceback

# Global model cache
models = {}


def get_model(model_name: str) -> YOLO:
    """Load and cache YOLO model."""
    if model_name not in models:
        print(f"Loading model: {model_name}")
        models[model_name] = YOLO(model_name)
        print(f"Model {model_name} loaded successfully")
    return models[model_name]


def count_persons(image, model_name="yolov8n.pt", confidence=0.25):
    """
    Detect and count persons in an image.

    Args:
        image: Input image as numpy array
        model_name: YOLO model to use
        confidence: Minimum confidence threshold

    Returns:
        Tuple of (annotated image, result text)
    """
    try:
        if image is None:
            return None, "Bitte laden Sie ein Bild hoch."

        print(f"Processing image with model {model_name}, confidence {confidence}")

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
        annotated = draw_count_overlay(annotated, person_count)

        # Build result text
        if person_count == 0:
            result_text = "Keine Personen erkannt."
        else:
            confidences = boxes.conf.cpu().numpy()
            avg_conf = np.mean(confidences) * 100
            result_text = f"""## Ergebnis

**{person_count} Person{"en" if person_count != 1 else ""} erkannt**

- Modell: {model_name.replace('.pt', '').upper()}
- Konfidenz-Schwelle: {confidence:.0%}
- Durchschnittliche Konfidenz: {avg_conf:.1f}%
"""

        return annotated, result_text

    except Exception as e:
        error_msg = f"Fehler: {str(e)}\n{traceback.format_exc()}"
        print(error_msg)
        return None, f"**Fehler bei der Verarbeitung:**\n```\n{str(e)}\n```"


def draw_count_overlay(image: np.ndarray, count: int) -> np.ndarray:
    """Draw person count overlay on image."""
    h, w = image.shape[:2]

    text = f"{count} Person{'en' if count != 1 else ''}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(0.8, min(w, h) / 500)
    thickness = max(2, int(font_scale * 2))

    (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)

    # Background box
    padding = 15
    box_x = (w - text_w) // 2 - padding
    box_y = 10
    box_w = text_w + padding * 2
    box_h = text_h + padding * 2 + baseline

    # Draw semi-transparent background
    overlay = image.copy()
    cv2.rectangle(overlay, (box_x, box_y), (box_x + box_w, box_y + box_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, image, 0.3, 0, image)

    # Draw border
    color = (0, 255, 0) if count > 0 else (150, 150, 150)
    cv2.rectangle(image, (box_x, box_y), (box_x + box_w, box_y + box_h), color, 2)

    # Draw text
    text_x = (w - text_w) // 2
    text_y = box_y + padding + text_h
    cv2.putText(image, text, (text_x, text_y), font, font_scale, color, thickness)

    return image


# Create Gradio interface
with gr.Blocks(title="Orchester Personenzähler", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🎻 Orchester & Zuschauer Personenzähler

    Laden Sie ein Foto eines Orchesters oder Publikums hoch, um die Anzahl der Personen zu zählen.
    Verwendet YOLOv8 für die Erkennung.
    """)

    with gr.Row():
        with gr.Column(scale=1):
            input_image = gr.Image(
                label="Bild hochladen",
                type="numpy",
                sources=["upload", "clipboard"]
            )

            model_dropdown = gr.Dropdown(
                choices=["yolov8n.pt", "yolov8s.pt", "yolov8m.pt"],
                value="yolov8n.pt",
                label="Modell"
            )

            confidence_slider = gr.Slider(
                minimum=0.1,
                maximum=0.9,
                value=0.25,
                step=0.05,
                label="Mindest-Konfidenz"
            )

            detect_btn = gr.Button("Personen erkennen", variant="primary")

        with gr.Column(scale=1):
            output_image = gr.Image(label="Ergebnis")
            result_text = gr.Markdown()

    # Event handlers
    detect_btn.click(
        fn=count_persons,
        inputs=[input_image, model_dropdown, confidence_slider],
        outputs=[output_image, result_text],
        api_name="detect"
    )

# Launch
if __name__ == "__main__":
    demo.launch()
