# Orchester & Zuschauer Personenzähler

Ein Tool zur automatischen Zählung von Personen auf Fotos von Orchestern oder Zuschauerräumen unter Verwendung von KI-Objekterkennung.

## Web-Version

**[Jetzt im Browser testen](https://ccabos.github.io/personcount/)**

Die Web-Version läuft komplett im Browser mit TensorFlow.js - keine Installation erforderlich!

## Features

### Web-Version
- Läuft komplett im Browser (keine Installation)
- Drag & Drop Bildupload
- Echtzeit-Personenerkennung mit TensorFlow.js
- Demo-Bilder zum Testen

### Python-Version
- Zählung von Personen auf einzelnen Bildern oder in ganzen Verzeichnissen
- Unterstützung verschiedener YOLO-Modelle (schnell bis hochpräzise)
- Visualisierung mit Bounding-Boxes
- Einstellbare Erkennungs-Konfidenz
- Export annotierter Bilder
- Batch-Verarbeitung ganzer Ordner

## Installation

```bash
# Repository klonen
git clone <repository-url>
cd personcount

# Virtuelle Umgebung erstellen (empfohlen)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# oder: venv\Scripts\activate  # Windows

# Abhängigkeiten installieren
pip install -r requirements.txt
```

## Verwendung

### Kommandozeile

```bash
# Einzelnes Bild analysieren
python person_counter.py orchester.jpg

# Mit Ausgabe-Bild speichern
python person_counter.py konzert.png --save

# Ganzes Verzeichnis verarbeiten
python person_counter.py ./konzertbilder/

# Mit größerem Modell für bessere Genauigkeit
python person_counter.py publikum.jpg --model yolov8m.pt

# Mit angepasster Konfidenz
python person_counter.py foto.jpg --confidence 0.4 --save
```

### Als Python-Modul

```python
from person_counter import PersonCounter

# Counter initialisieren
counter = PersonCounter(model_name="yolov8n.pt", confidence=0.25)

# Einzelnes Bild zählen
result = counter.count_persons("orchester.jpg", save_output=True)
print(f"Erkannte Personen: {result['person_count']}")

# Verzeichnis verarbeiten
results = counter.count_persons_in_directory("./bilder/", save_output=True)

# Visualisierung erstellen
counter.visualize_result("konzert.jpg", output_path="ergebnis.jpg")
```

## YOLO-Modelle

| Modell | Geschwindigkeit | Genauigkeit | Empfehlung |
|--------|-----------------|-------------|------------|
| `yolov8n.pt` | Sehr schnell | Basis | Schnelle Vorschau |
| `yolov8s.pt` | Schnell | Gut | Alltagsgebrauch |
| `yolov8m.pt` | Mittel | Sehr gut | **Empfohlen für Orchester** |
| `yolov8l.pt` | Langsam | Excellent | Hochauflösende Bilder |
| `yolov8x.pt` | Sehr langsam | Maximal | Maximale Präzision |

## Parameter

| Parameter | Kurz | Standard | Beschreibung |
|-----------|------|----------|--------------|
| `--model` | `-m` | yolov8n.pt | YOLO-Modellname |
| `--confidence` | `-c` | 0.25 | Mindest-Konfidenz (0.0-1.0) |
| `--save` | `-s` | False | Annotiertes Bild speichern |
| `--output` | `-o` | - | Ausgabepfad für Bild |
| `--quiet` | `-q` | False | Weniger Ausgaben |

## Tipps für Orchesterfotos

1. **Hohe Auflösung**: Verwende hochauflösende Bilder für bessere Erkennung
2. **Gute Beleuchtung**: Gut beleuchtete Szenen werden besser erkannt
3. **Modellwahl**: Für große Orchester mit vielen Personen empfiehlt sich `yolov8m.pt` oder größer
4. **Konfidenz anpassen**: Bei Übererkennung Konfidenz erhöhen (z.B. 0.4), bei Untererkennung senken (z.B. 0.2)

## Beispiel-Ausgabe

```
$ python person_counter.py konzert.jpg --save

Lade YOLO-Modell: yolov8n.pt...
Modell erfolgreich geladen.

========================================
Bild: konzert.jpg
Erkannte Personen: 47
Durchschnittliche Konfidenz: 78.34%
Annotiertes Bild: konzert_counted.jpg
========================================
```

## Lizenz

MIT License
