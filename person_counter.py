#!/usr/bin/env python3
"""
Orchester & Zuschauer Personenzähler
====================================
Ein Tool zur Zählung von Personen auf Fotos von Orchestern oder Zuschauerräumen
unter Verwendung von YOLO (You Only Look Once) für die Objekterkennung.
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

    def __init__(self, model_name: str = "yolov8n.pt", confidence: float = 0.25):
        """
        Initialisiert den PersonCounter.

        Args:
            model_name: Name des YOLO-Modells (z.B. yolov8n.pt, yolov8s.pt, yolov8m.pt)
            confidence: Mindest-Konfidenz für die Erkennung (0.0 - 1.0)
        """
        self.confidence = confidence
        print(f"Lade YOLO-Modell: {model_name}...")
        self.model = YOLO(model_name)
        print("Modell erfolgreich geladen.")

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

        # Konfidenzwerte extrahieren
        confidences = boxes.conf.cpu().numpy().tolist() if len(boxes) > 0 else []

        # Bounding-Box-Koordinaten extrahieren
        bboxes = boxes.xyxy.cpu().numpy().tolist() if len(boxes) > 0 else []

        output_path = None
        if save_output:
            # Annotiertes Bild speichern
            output_path = image_path.parent / f"{image_path.stem}_counted{image_path.suffix}"
            annotated_frame = result.plot()
            cv2.imwrite(str(output_path), annotated_frame)

        return {
            "image_path": str(image_path),
            "person_count": person_count,
            "confidences": confidences,
            "bounding_boxes": bboxes,
            "output_path": str(output_path) if output_path else None
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
        results = self.model(
            image_path,
            conf=self.confidence,
            classes=[self.PERSON_CLASS_ID],
            verbose=False
        )

        result = results[0]
        annotated_frame = result.plot()

        if show_count:
            person_count = len(result.boxes)
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

    args = parser.parse_args()

    # PersonCounter initialisieren
    counter = PersonCounter(model_name=args.model, confidence=args.confidence)

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
