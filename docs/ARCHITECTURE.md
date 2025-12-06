# Architecture Documentation

## Overview

The Person Counter is a Python-based tool that uses YOLO (You Only Look Once) deep learning models to detect and count people in images, specifically designed for orchestra and audience photography.

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         User Interface                          │
│  ┌─────────────────────┐      ┌─────────────────────────────┐  │
│  │   CLI (argparse)    │      │    Python API (import)      │  │
│  └──────────┬──────────┘      └──────────────┬──────────────┘  │
└─────────────┼────────────────────────────────┼──────────────────┘
              │                                │
              ▼                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                      PersonCounter Class                         │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  • count_persons()           - Single image counting      │   │
│  │  • count_persons_in_directory() - Batch processing        │   │
│  │  • visualize_result()        - Generate annotated images  │   │
│  └──────────────────────────────────────────────────────────┘   │
└─────────────────────────────┬───────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    YOLO Model Layer (Ultralytics)                │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  • Model Loading (yolov8n/s/m/l/x.pt)                     │   │
│  │  • Inference Engine                                       │   │
│  │  • Bounding Box Detection                                 │   │
│  │  • Confidence Scoring                                     │   │
│  └──────────────────────────────────────────────────────────┘   │
└─────────────────────────────┬───────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Image Processing Layer                      │
│  ┌────────────────┐  ┌────────────────┐  ┌────────────────┐    │
│  │    OpenCV      │  │     PIL        │  │    NumPy       │    │
│  │  (cv2)         │  │   (Pillow)     │  │                │    │
│  │  - Image I/O   │  │  - Format      │  │  - Array ops   │    │
│  │  - Annotation  │  │    handling    │  │  - Data manip  │    │
│  └────────────────┘  └────────────────┘  └────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
```

## Component Details

### 1. PersonCounter Class (`person_counter.py:22-142`)

The main class that orchestrates person detection.

#### Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `model` | `YOLO` | Loaded YOLO model instance |
| `confidence` | `float` | Minimum confidence threshold (0.0-1.0) |
| `PERSON_CLASS_ID` | `int` | COCO class ID for "person" (0) |

#### Methods

```python
class PersonCounter:
    def __init__(self, model_name: str, confidence: float)
    def count_persons(self, image_path: str, save_output: bool) -> dict
    def count_persons_in_directory(self, directory: str, ...) -> list
    def visualize_result(self, image_path: str, ...) -> np.ndarray
```

### 2. YOLO Integration

The system uses Ultralytics YOLOv8, which provides:

- **Pre-trained Models**: Trained on COCO dataset (80 classes)
- **Person Detection**: Class ID 0 in COCO corresponds to "person"
- **Multiple Model Sizes**: Trade-off between speed and accuracy

```
Model Performance Comparison:
┌──────────┬───────────┬──────────┬─────────────┐
│ Model    │ Size (MB) │ Speed    │ mAP@50      │
├──────────┼───────────┼──────────┼─────────────┤
│ yolov8n  │ 6.3       │ Fastest  │ 37.3        │
│ yolov8s  │ 21.5      │ Fast     │ 44.9        │
│ yolov8m  │ 49.7      │ Medium   │ 50.2        │
│ yolov8l  │ 83.7      │ Slow     │ 52.9        │
│ yolov8x  │ 130.5     │ Slowest  │ 53.9        │
└──────────┴───────────┴──────────┴─────────────┘
```

### 3. Data Flow

```
Input Image
    │
    ▼
┌─────────────────┐
│ Load & Validate │
│  (Path check)   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ YOLO Inference  │
│ • Preprocess    │
│ • Forward pass  │
│ • NMS filtering │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Filter Results  │
│ • class == 0    │
│ • conf >= thresh│
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Extract Data    │
│ • Bounding boxes│
│ • Confidences   │
│ • Count         │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Output          │
│ • JSON result   │
│ • Annotated img │
└─────────────────┘
```

### 4. Output Format

#### Single Image Result

```python
{
    "image_path": "/path/to/image.jpg",
    "person_count": 42,
    "confidences": [0.95, 0.87, 0.92, ...],
    "bounding_boxes": [[x1, y1, x2, y2], ...],
    "output_path": "/path/to/image_counted.jpg"  # if save_output=True
}
```

#### Batch Processing Result

```python
[
    {"image_path": "img1.jpg", "person_count": 15, ...},
    {"image_path": "img2.jpg", "person_count": 23, ...},
    {"image_path": "img3.jpg", "error": "File not found"}
]
```

## Design Decisions

### 1. Why YOLOv8?

- **Accuracy**: State-of-the-art object detection
- **Speed**: Real-time capable inference
- **Ease of Use**: Simple Python API via Ultralytics
- **Flexibility**: Multiple model sizes for different use cases

### 2. Why Filter to Person Class Only?

```python
results = self.model(image, classes=[self.PERSON_CLASS_ID])
```

- Reduces false positives from other objects
- Improves processing speed
- Focuses on the specific use case

### 3. Confidence Threshold

Default: `0.25` (25%)

- Lower threshold: More detections, more false positives
- Higher threshold: Fewer detections, fewer false positives
- Recommended for crowded scenes: `0.3-0.4`

## Error Handling

```python
# File validation
if not image_path.exists():
    raise FileNotFoundError(f"Image not found: {image_path}")

# Directory validation
if not directory.is_dir():
    raise NotADirectoryError(f"Directory not found: {directory}")

# Graceful batch processing
try:
    result = self.count_persons(image_file)
except Exception as e:
    results.append({"image_path": str(image_file), "error": str(e)})
```

## Performance Considerations

### Memory Usage

- Model loading: 50-500 MB depending on model size
- Image processing: Proportional to image resolution
- Batch processing: Sequential to avoid memory spikes

### Optimization Tips

1. **Use smaller models** for preview/testing
2. **Resize large images** before processing
3. **Process in batches** rather than loading all images

## File Structure

```
personcount/
├── person_counter.py      # Main module
├── create_test_images.py  # Test image generator
├── requirements.txt       # Dependencies
├── README.md              # User documentation
├── docs/
│   └── ARCHITECTURE.md    # This file
├── tests/
│   ├── __init__.py
│   ├── test_person_counter.py
│   └── conftest.py        # Pytest fixtures
└── test_images/           # Generated test images
```

## Dependencies

| Package | Purpose | Version |
|---------|---------|---------|
| ultralytics | YOLO implementation | >= 8.0.0 |
| opencv-python | Image I/O & annotation | >= 4.8.0 |
| Pillow | Image format handling | >= 10.0.0 |
| numpy | Array operations | >= 1.24.0 |
| pytest | Unit testing | >= 7.0.0 |

## Future Improvements

1. **GPU Acceleration**: Automatic CUDA detection
2. **Video Support**: Frame-by-frame processing
3. **Pose Estimation**: Distinguish seated vs standing
4. **Crowd Density**: Heat map visualization
5. **API Server**: REST API for web integration
