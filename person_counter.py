#!/usr/bin/env python3
"""
Orchester & Zuschauer Personenzähler
====================================
Ein Tool zur Zählung von Personen auf Fotos von Orchestern oder Zuschauerräumen
unter Verwendung von YOLO (You Only Look Once) für die Objekterkennung.

Optional: SAHI (Slicing Aided Hyper Inference) für die zuverlässige Erkennung
vieler kleiner Personen auf hochauflösenden Bildern.
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO


class PersonCounter:
    """Zählt Personen auf Bildern mit YOLO-Objekterkennung."""

    # COCO-Klassen-ID für "person"
    PERSON_CLASS_ID = 0

    def __init__(
        self,
        model_name: str = "yolov8n.pt",
        confidence: float = 0.25,
        use_sahi: bool = False,
        slice_height: int = 640,
        slice_width: int = 640,
        overlap_height_ratio: float = 0.2,
        overlap_width_ratio: float = 0.2,
        device: Optional[str] = None,
    ):
        """
        Initialisiert den PersonCounter.

        Args:
            model_name: Name des YOLO-Modells (z.B. yolov8n.pt, yolov8s.pt, yolov8m.pt)
            confidence: Mindest-Konfidenz für die Erkennung (0.0 - 1.0)
            use_sahi: Wenn True, wird SAHI (sliced inference) für die Erkennung
                kleiner Personen auf großen Bildern verwendet.
            slice_height: Höhe der SAHI-Kacheln in Pixeln (nur bei use_sahi=True).
            slice_width: Breite der SAHI-Kacheln in Pixeln (nur bei use_sahi=True).
            overlap_height_ratio: Überlappung der Kacheln in Höhe (0.0-1.0).
            overlap_width_ratio: Überlappung der Kacheln in Breite (0.0-1.0).
            device: Gerät für die Inferenz ("cpu" / "cuda" / "mps"). None = auto.
        """
        self.model_name = model_name
        self.confidence = confidence
        self.use_sahi = use_sahi
        self.slice_height = slice_height
        self.slice_width = slice_width
        self.overlap_height_ratio = overlap_height_ratio
        self.overlap_width_ratio = overlap_width_ratio
        self.device = device

        print(f"Lade YOLO-Modell: {model_name}...")
        self.model = YOLO(model_name)
        print("Modell erfolgreich geladen.")

        self.sahi_model = None
        if self.use_sahi:
            self.sahi_model = self._load_sahi_model()

    def _load_sahi_model(self):
        """Lädt das SAHI-Detektor-Wrapper für das YOLO-Modell."""
        try:
            from sahi import AutoDetectionModel
        except ImportError as exc:
            raise ImportError(
                "SAHI ist nicht installiert. Bitte installieren mit: "
                "pip install sahi"
            ) from exc

        # SAHI unterstützt sowohl 'ultralytics' (neuere Versionen) als auch
        # 'yolov8' (ältere Versionen) als Modelltyp.
        model_type_candidates = ("ultralytics", "yolov8")
        last_error: Optional[Exception] = None
        for model_type in model_type_candidates:
            try:
                print(f"Lade SAHI-Modell ({model_type}): {self.model_name}...")
                detection_model = AutoDetectionModel.from_pretrained(
                    model_type=model_type,
                    model_path=self.model_name,
                    confidence_threshold=self.confidence,
                    device=self.device,
                )
                print("SAHI-Modell erfolgreich geladen.")
                return detection_model
            except Exception as exc:  # pragma: no cover - depends on sahi version
                last_error = exc
                continue

        raise RuntimeError(
            f"Konnte SAHI-Detektor nicht initialisieren: {last_error}"
        )

    def _predict_sahi(self, image_path: str):
        """
        Führt SAHI-basierte sliced inference durch und liefert (confidences, bboxes).
        """
        from sahi.predict import get_sliced_prediction

        prediction = get_sliced_prediction(
            image_path,
            self.sahi_model,
            slice_height=self.slice_height,
            slice_width=self.slice_width,
            overlap_height_ratio=self.overlap_height_ratio,
            overlap_width_ratio=self.overlap_width_ratio,
            verbose=0,
        )

        confidences: list = []
        bboxes: list = []
        for obj in prediction.object_prediction_list:
            if obj.category.id != self.PERSON_CLASS_ID:
                continue
            bbox = obj.bbox.to_xyxy()
            confidences.append(float(obj.score.value))
            bboxes.append([float(c) for c in bbox])

        return confidences, bboxes

    def _draw_detections(
        self,
        image: np.ndarray,
        bboxes: list,
        confidences: list,
    ) -> np.ndarray:
        """Zeichnet Bounding-Boxes mit Konfidenzwerten auf ein Bild."""
        annotated = image.copy()
        color = (0, 255, 0)
        for bbox, conf in zip(bboxes, confidences):
            x1, y1, x2, y2 = (int(v) for v in bbox)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            label = f"person {conf:.2f}"
            (tw, th), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
            )
            cv2.rectangle(
                annotated,
                (x1, max(0, y1 - th - 4)),
                (x1 + tw + 4, y1),
                color,
                -1,
            )
            cv2.putText(
                annotated,
                label,
                (x1 + 2, max(th, y1 - 2)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0),
                1,
                cv2.LINE_AA,
            )
        return annotated

    def count_persons(self, image_path: str, save_output: bool = False) -> dict:
        """
        Zählt die Personen auf einem Bild.

        Args:
            image_path: Pfad zum Eingabebild
            save_output: Ob das annotierte Bild gespeichert werden soll

        Returns:
            Dictionary mit Zählergebnis und Details
        """
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Bild nicht gefunden: {image_path}")

        if self.use_sahi:
            confidences, bboxes = self._predict_sahi(str(image_path))
            person_count = len(bboxes)

            output_path = None
            if save_output:
                image = cv2.imread(str(image_path))
                annotated_frame = self._draw_detections(
                    image, bboxes, confidences
                )
                output_path = (
                    image_path.parent
                    / f"{image_path.stem}_counted{image_path.suffix}"
                )
                cv2.imwrite(str(output_path), annotated_frame)
        else:
            # YOLO-Inferenz durchführen
            results = self.model(
                str(image_path),
                conf=self.confidence,
                classes=[self.PERSON_CLASS_ID],  # Nur Personen erkennen
                verbose=False
            )

            result = results[0]
            boxes = result.boxes

            # Personen zählen
            person_count = len(boxes)

            # Konfidenzwerte und Bounding-Boxen extrahieren
            confidences = boxes.conf.cpu().numpy().tolist() if len(boxes) > 0 else []
            bboxes = boxes.xyxy.cpu().numpy().tolist() if len(boxes) > 0 else []

            output_path = None
            if save_output:
                output_path = image_path.parent / f"{image_path.stem}_counted{image_path.suffix}"
                annotated_frame = result.plot()
                cv2.imwrite(str(output_path), annotated_frame)

        return {
            "image_path": str(image_path),
            "person_count": person_count,
            "confidences": confidences,
            "bounding_boxes": bboxes,
            "output_path": str(output_path) if output_path else None,
            "used_sahi": self.use_sahi,
        }

    def count_persons_in_directory(
        self,
        directory: str,
        save_output: bool = False,
        extensions: tuple = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
    ) -> list:
        """
        Zählt Personen in allen Bildern eines Verzeichnisses.

        Args:
            directory: Pfad zum Verzeichnis
            save_output: Ob annotierte Bilder gespeichert werden sollen
            extensions: Erlaubte Dateierweiterungen

        Returns:
            Liste mit Ergebnissen für jedes Bild
        """
        directory = Path(directory)
        if not directory.is_dir():
            raise NotADirectoryError(f"Verzeichnis nicht gefunden: {directory}")

        results = []
        image_files = [
            f for f in directory.iterdir()
            if f.suffix.lower() in extensions
        ]

        print(f"Gefundene Bilder: {len(image_files)}")

        for i, image_file in enumerate(sorted(image_files), 1):
            print(f"Verarbeite [{i}/{len(image_files)}]: {image_file.name}...")
            try:
                result = self.count_persons(str(image_file), save_output)
                results.append(result)
                print(f"  -> {result['person_count']} Person(en) erkannt")
            except Exception as e:
                print(f"  -> Fehler: {e}")
                results.append({
                    "image_path": str(image_file),
                    "error": str(e)
                })

        return results

    def visualize_result(
        self,
        image_path: str,
        output_path: Optional[str] = None,
        show_count: bool = True
    ) -> np.ndarray:
        """
        Visualisiert die Erkennungsergebnisse mit Bounding-Boxes.

        Args:
            image_path: Pfad zum Eingabebild
            output_path: Optionaler Pfad zum Speichern des Ergebnisses
            show_count: Ob die Gesamtzahl angezeigt werden soll

        Returns:
            Annotiertes Bild als numpy-Array
        """
        if self.use_sahi:
            confidences, bboxes = self._predict_sahi(image_path)
            image = cv2.imread(image_path)
            annotated_frame = self._draw_detections(image, bboxes, confidences)
            person_count = len(bboxes)
        else:
            results = self.model(
                image_path,
                conf=self.confidence,
                classes=[self.PERSON_CLASS_ID],
                verbose=False
            )

            result = results[0]
            annotated_frame = result.plot()
            person_count = len(result.boxes)

        if show_count:
            # Zähler-Text hinzufügen
            text = f"Personen: {person_count}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1.5
            thickness = 3
            color = (0, 255, 0)  # Grün

            # Text-Hintergrund
            (text_width, text_height), baseline = cv2.getTextSize(
                text, font, font_scale, thickness
            )
            cv2.rectangle(
                annotated_frame,
                (10, 10),
                (20 + text_width, 20 + text_height + baseline),
                (0, 0, 0),
                -1
            )
            cv2.putText(
                annotated_frame,
                text,
                (15, 15 + text_height),
                font,
                font_scale,
                color,
                thickness
            )

        if output_path:
            cv2.imwrite(output_path, annotated_frame)

        return annotated_frame


def print_summary(results: list) -> None:
    """Gibt eine Zusammenfassung der Ergebnisse aus."""
    print("\n" + "=" * 60)
    print("ZUSAMMENFASSUNG")
    print("=" * 60)

    total_persons = 0
    successful = 0
    failed = 0

    for r in results:
        if "error" in r:
            failed += 1
        else:
            successful += 1
            total_persons += r["person_count"]

    print(f"Verarbeitete Bilder: {len(results)}")
    print(f"  - Erfolgreich: {successful}")
    print(f"  - Fehlgeschlagen: {failed}")
    print(f"Gesamtzahl erkannter Personen: {total_persons}")

    if successful > 0:
        avg = total_persons / successful
        print(f"Durchschnitt pro Bild: {avg:.1f}")

    print("=" * 60)


def main():
    """Hauptfunktion für die Kommandozeilennutzung."""
    parser = argparse.ArgumentParser(
        description="Zählt Personen auf Orchesterfotos oder Zuschauerbildern mit YOLO",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Beispiele:
  python person_counter.py foto.jpg
  python person_counter.py konzert.png --save
  python person_counter.py ./bilder/ --model yolov8m.pt --confidence 0.3
  python person_counter.py orchester.jpg --save --output ergebnis.jpg
  python person_counter.py grosses_orchester.jpg --sahi --save
        """
    )

    parser.add_argument(
        "input",
        help="Pfad zu einem Bild oder Verzeichnis mit Bildern"
    )
    parser.add_argument(
        "--model", "-m",
        default="yolov8n.pt",
        help="YOLO-Modell (yolov8n.pt=schnell, yolov8m.pt=ausgewogen, yolov8x.pt=genau)"
    )
    parser.add_argument(
        "--confidence", "-c",
        type=float,
        default=0.25,
        help="Mindest-Konfidenz für Erkennung (0.0-1.0, Standard: 0.25)"
    )
    parser.add_argument(
        "--save", "-s",
        action="store_true",
        help="Speichere annotiertes Bild mit Bounding-Boxes"
    )
    parser.add_argument(
        "--output", "-o",
        help="Ausgabepfad für annotiertes Bild (nur bei einzelnem Bild)"
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Weniger Ausgaben"
    )
    parser.add_argument(
        "--sahi",
        action="store_true",
        help="SAHI (sliced inference) aktivieren für bessere Erkennung "
             "kleiner Personen auf großen Bildern"
    )
    parser.add_argument(
        "--slice-size",
        type=int,
        default=640,
        help="Kachelgröße für SAHI in Pixeln (Standard: 640)"
    )
    parser.add_argument(
        "--slice-overlap",
        type=float,
        default=0.2,
        help="Überlappung der SAHI-Kacheln (0.0-1.0, Standard: 0.2)"
    )

    args = parser.parse_args()

    # PersonCounter initialisieren
    counter = PersonCounter(
        model_name=args.model,
        confidence=args.confidence,
        use_sahi=args.sahi,
        slice_height=args.slice_size,
        slice_width=args.slice_size,
        overlap_height_ratio=args.slice_overlap,
        overlap_width_ratio=args.slice_overlap,
    )

    input_path = Path(args.input)

    if input_path.is_file():
        # Einzelnes Bild verarbeiten
        result = counter.count_persons(str(input_path), save_output=args.save)

        if args.output:
            counter.visualize_result(str(input_path), args.output)
            result["output_path"] = args.output

        print(f"\n{'=' * 40}")
        print(f"Bild: {result['image_path']}")
        print(f"Erkannte Personen: {result['person_count']}")
        if result.get('used_sahi'):
            print("Modus: SAHI (sliced inference)")

        if result['confidences'] and not args.quiet:
            avg_conf = sum(result['confidences']) / len(result['confidences'])
            print(f"Durchschnittliche Konfidenz: {avg_conf:.2%}")

        if result['output_path']:
            print(f"Annotiertes Bild: {result['output_path']}")

        print(f"{'=' * 40}\n")

    elif input_path.is_dir():
        # Verzeichnis verarbeiten
        results = counter.count_persons_in_directory(
            str(input_path),
            save_output=args.save
        )
        print_summary(results)

    else:
        print(f"Fehler: '{args.input}' ist weder eine Datei noch ein Verzeichnis.")
        sys.exit(1)


if __name__ == "__main__":
    main()
