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
- **SAHI (Slicing Aided Hyper Inference)** für die zuverlässige Erkennung
  vieler kleiner Personen auf hochauflösenden Orchester- oder Publikumsfotos
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

# SAHI für große Bilder mit vielen kleinen Personen (z.B. ganzes Orchester)
python person_counter.py grosses_orchester.jpg --sahi --save

# SAHI mit angepasster Kachelgröße und Überlappung
python person_counter.py konzertsaal.jpg --sahi --slice-size 512 --slice-overlap 0.3
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

# SAHI für große Bilder mit vielen kleinen Personen
sahi_counter = PersonCounter(
    model_name="yolov8m.pt",
    confidence=0.25,
    use_sahi=True,
    slice_height=640,
    slice_width=640,
    overlap_height_ratio=0.2,
    overlap_width_ratio=0.2,
)
result = sahi_counter.count_persons("grosses_orchester.jpg", save_output=True)
print(f"Erkannte Personen (SAHI): {result['person_count']}")
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
| `--sahi` | - | False | SAHI sliced inference aktivieren |
| `--slice-size` | - | 640 | SAHI-Kachelgröße in Pixeln |
| `--slice-overlap` | - | 0.2 | SAHI-Kachelüberlappung (0.0-1.0) |

## Tipps für Orchesterfotos

1. **Hohe Auflösung**: Verwende hochauflösende Bilder für bessere Erkennung
2. **Gute Beleuchtung**: Gut beleuchtete Szenen werden besser erkannt
3. **Modellwahl**: Für große Orchester mit vielen Personen empfiehlt sich `yolov8m.pt` oder größer
4. **Konfidenz anpassen**: Bei Übererkennung Konfidenz erhöhen (z.B. 0.4), bei Untererkennung senken (z.B. 0.2)
5. **SAHI aktivieren**: Bei großen Bildern (>2000px) mit vielen kleinen Personen
   liefert `--sahi` deutlich bessere Ergebnisse, da das Bild in überlappende
   Kacheln zerlegt und einzeln analysiert wird. Standardmäßig 640×640 Kacheln
   mit 20% Überlappung. Die Verarbeitung ist langsamer, aber wesentlich
   genauer für entfernte Musiker oder Zuschauer.

### Wann SAHI verwenden?

SAHI (Slicing Aided Hyper Inference) ist besonders sinnvoll, wenn:
- Das Bild groß ist (z.B. 3000×2000 Pixel oder mehr)
- Viele Personen klein im Bild erscheinen (Weitwinkelaufnahmen von Bühnen,
  Konzertsälen, Zuschauerrängen)
- Standard-YOLO offensichtlich Personen übersieht

Beispiel-Aufruf:
```bash
python person_counter.py orchester_4k.jpg --sahi --model yolov8m.pt --save
```

## SAHI auf Hugging Face Spaces

Der Space unter `huggingface/` enthält ebenfalls einen **SAHI-Schalter** in
der Gradio-Oberfläche. Auf einem öffentlichen Hugging Face Space deployen:

1. **Space anlegen** auf <https://huggingface.co/new-space>
   - SDK: **Gradio**
   - Hardware: CPU reicht aus, GPU beschleunigt SAHI deutlich
2. **Dateien hochladen** aus dem `huggingface/`-Ordner dieses Repos:
   - `app.py` (enthält den SAHI-Schalter samt Kachelgrößen-Slidern)
   - `requirements.txt` (enthält bereits `sahi>=0.11.18`)
   - `README.md` (Space-Metadaten)
3. **Build abwarten** – der erste Build dauert länger, weil `sahi` und
   `ultralytics` nachgeladen werden.
4. **Im Space** das Bild hochladen, das gewünschte YOLO-Modell wählen, dann
   **"SAHI aktivieren"** anhaken. Mit den Slidern lassen sich Kachelgröße
   (Standard 640 px) und Überlappung (Standard 20%) anpassen.

Empfehlungen für Hugging Face:

- **Hardware**: SAHI ist auf CPU rechenintensiv. Bei großen Bildern oder
  vielen Aufrufen auf eine GPU-Instanz upgraden (`T4 small` reicht meist).
- **Modellwahl**: Auf der gratis CPU-Stufe `yolov8n.pt` mit SAHI verwenden;
  mit GPU eher `yolov8m.pt` oder `yolov8l.pt`.
- **Kachelgröße**: 640 px ist der Standard. Bei sehr großen Bildern (z.B.
  6000 px breit) erhöht 768–1024 px die Geschwindigkeit ohne große
  Genauigkeitsverluste.
- **Überlappung**: 20% ist ein guter Kompromiss. Wenn Personen an
  Kachelrändern doppelt gezählt werden, Überlappung leicht reduzieren;
  wenn sie an Kachelrändern verloren gehen, erhöhen (max. ~0.4).
- **Timeout**: Sehr große Bilder mit SAHI können die Gradio-Default-Timeouts
  überschreiten. Notfalls Bild vorher auf ~3000 px Breite verkleinern.

### Updates zum Space pushen

Sobald der Space existiert, gibt es drei Wege, neue Versionen von
`huggingface/app.py`, `huggingface/requirements.txt` und
`huggingface/README.md` hochzuladen.

**Vorbereitung (einmalig):**

1. Auf <https://huggingface.co/settings/tokens> einen **Write**-Token erstellen.
2. `pip install -U huggingface_hub` lokal installieren.

**Option A – Python-API (empfohlen, ein Befehl):**

```bash
export HF_TOKEN=hf_xxx   # frisch erstelltes Write-Token

python - <<'PY'
import os
from huggingface_hub import HfApi

api = HfApi(token=os.environ["HF_TOKEN"])
print("whoami:", api.whoami()["name"])

api.upload_folder(
    repo_id="<user>/<space-name>",          # z.B. "Ccab/orchcount"
    repo_type="space",
    folder_path="huggingface",              # Ordner in diesem Repo
    path_in_repo=".",                       # ins Space-Root hochladen
    commit_message="Update SAHI pipeline",
    allow_patterns=["app.py", "requirements.txt", "README.md"],
)
print("Done.")
PY
```

**Option B – Hugging Face CLI:**

```bash
huggingface-cli login                       # Token einmalig speichern
huggingface-cli upload <user>/<space-name> huggingface . \
    --repo-type space \
    --include "app.py" "requirements.txt" "README.md" \
    --commit-message "Update SAHI pipeline"
```

**Option C – Git (gut für viele Iterationen):**

```bash
git clone https://huggingface.co/spaces/<user>/<space-name> hf-space
cd hf-space
cp ../personcount/huggingface/{app.py,requirements.txt,README.md} .
git add app.py requirements.txt README.md
git commit -m "Update SAHI pipeline"
git push                                    # Token als Passwort eingeben
```

Nach dem Push baut der Space automatisch neu (~2–5 min, weil `sahi` und
`ultralytics` nachgezogen werden). Den Fortschritt unter dem **Logs**-Tab
des Space verfolgen. Falls der Build fehlschlägt, dort steht der Grund.

> **Sicherheit**: Tokens nicht in das Repo committen. Nach versehentlicher
> Veröffentlichung sofort unter
> <https://huggingface.co/settings/tokens> widerrufen.

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
