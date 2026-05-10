#!/usr/bin/env python3
"""
Flask API Backend for YOLO Person Counter
Provides REST API endpoints for person detection in images.
"""

import base64
import io
import os
import tempfile
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, jsonify, request, send_file
from flask_cors import CORS
from PIL import Image
from werkzeug.utils import secure_filename

from person_counter import PersonCounter

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Configuration
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'webp', 'bmp'}

# Global model cache
_model_cache = {}


def get_model(
    model_name: str = "yolov8n.pt",
    confidence: float = 0.25,
    use_sahi: bool = False,
    slice_size: int = 640,
    slice_overlap: float = 0.2,
) -> PersonCounter:
    """Get or create a cached PersonCounter instance."""
    cache_key = (
        f"{model_name}_{confidence}_sahi={use_sahi}"
        f"_slice={slice_size}_overlap={slice_overlap}"
    )
    if cache_key not in _model_cache:
        _model_cache[cache_key] = PersonCounter(
            model_name=model_name,
            confidence=confidence,
            use_sahi=use_sahi,
            slice_height=slice_size,
            slice_width=slice_size,
            overlap_height_ratio=slice_overlap,
            overlap_width_ratio=slice_overlap,
        )
    return _model_cache[cache_key]


def allowed_file(filename: str) -> bool:
    """Check if file extension is allowed."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def image_to_base64(image: np.ndarray) -> str:
    """Convert OpenCV image to base64 string."""
    _, buffer = cv2.imencode('.jpg', image)
    return base64.b64encode(buffer).decode('utf-8')


def base64_to_image(base64_string: str) -> np.ndarray:
    """Convert base64 string to OpenCV image."""
    # Remove data URL prefix if present
    if ',' in base64_string:
        base64_string = base64_string.split(',')[1]

    image_data = base64.b64decode(base64_string)
    nparr = np.frombuffer(image_data, np.uint8)
    return cv2.imdecode(nparr, cv2.IMREAD_COLOR)


@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'models_loaded': list(_model_cache.keys())
    })


@app.route('/api/models', methods=['GET'])
def list_models():
    """List available YOLO models."""
    return jsonify({
        'models': [
            {'id': 'yolov8n.pt', 'name': 'YOLOv8 Nano', 'description': 'Schnellstes Modell, geringste Genauigkeit'},
            {'id': 'yolov8s.pt', 'name': 'YOLOv8 Small', 'description': 'Schnell mit guter Genauigkeit'},
            {'id': 'yolov8m.pt', 'name': 'YOLOv8 Medium', 'description': 'Ausgewogen (empfohlen)'},
            {'id': 'yolov8l.pt', 'name': 'YOLOv8 Large', 'description': 'Hohe Genauigkeit, langsamer'},
            {'id': 'yolov8x.pt', 'name': 'YOLOv8 XLarge', 'description': 'Höchste Genauigkeit, am langsamsten'}
        ]
    })


@app.route('/api/detect', methods=['POST'])
def detect_persons():
    """
    Detect persons in an uploaded image.

    Accepts:
        - multipart/form-data with 'image' file
        - JSON with 'image' as base64 string

    Query params:
        - model: YOLO model name (default: yolov8n.pt)
        - confidence: minimum confidence threshold (default: 0.25)
        - annotate: whether to return annotated image (default: true)
        - sahi: enable SAHI sliced inference for large images (default: false)
        - slice_size: tile size for SAHI in pixels (default: 640)
        - slice_overlap: tile overlap ratio for SAHI (default: 0.2)
    """
    # Get parameters
    model_name = request.args.get('model', 'yolov8n.pt')
    confidence = float(request.args.get('confidence', 0.25))
    annotate = request.args.get('annotate', 'true').lower() == 'true'
    use_sahi = request.args.get('sahi', 'false').lower() == 'true'
    slice_size = int(request.args.get('slice_size', 640))
    slice_overlap = float(request.args.get('slice_overlap', 0.2))

    # Validate model name
    valid_models = ['yolov8n.pt', 'yolov8s.pt', 'yolov8m.pt', 'yolov8l.pt', 'yolov8x.pt']
    if model_name not in valid_models:
        return jsonify({'error': f'Invalid model. Choose from: {valid_models}'}), 400

    # Get image from request
    temp_path = None
    try:
        if request.content_type and 'multipart/form-data' in request.content_type:
            # Handle file upload
            if 'image' not in request.files:
                return jsonify({'error': 'No image file provided'}), 400

            file = request.files['image']
            if file.filename == '':
                return jsonify({'error': 'No file selected'}), 400

            if not allowed_file(file.filename):
                return jsonify({'error': f'Invalid file type. Allowed: {ALLOWED_EXTENSIONS}'}), 400

            # Save to temp file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp:
                file.save(tmp.name)
                temp_path = tmp.name

        elif request.is_json:
            # Handle base64 image
            data = request.get_json()
            if 'image' not in data:
                return jsonify({'error': 'No image data provided'}), 400

            # Convert base64 to image and save
            img = base64_to_image(data['image'])
            with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp:
                cv2.imwrite(tmp.name, img)
                temp_path = tmp.name
        else:
            return jsonify({'error': 'Invalid content type. Use multipart/form-data or application/json'}), 400

        # Get model and run detection
        counter = get_model(
            model_name, confidence, use_sahi, slice_size, slice_overlap
        )
        result = counter.count_persons(temp_path, save_output=False)

        # Build response
        response = {
            'success': True,
            'person_count': result['person_count'],
            'detections': [
                {
                    'id': i + 1,
                    'confidence': conf,
                    'bbox': bbox
                }
                for i, (conf, bbox) in enumerate(zip(result['confidences'], result['bounding_boxes']))
            ],
            'model': model_name,
            'confidence_threshold': confidence,
            'sahi': {
                'enabled': use_sahi,
                'slice_size': slice_size,
                'slice_overlap': slice_overlap,
            } if use_sahi else {'enabled': False},
        }

        # Add annotated image if requested
        if annotate:
            annotated = counter.visualize_result(temp_path, show_count=True)
            response['annotated_image'] = 'data:image/jpeg;base64,' + image_to_base64(annotated)

        return jsonify(response)

    except Exception as e:
        return jsonify({'error': str(e)}), 500

    finally:
        # Cleanup temp file
        if temp_path and os.path.exists(temp_path):
            os.unlink(temp_path)


@app.route('/api/detect/url', methods=['POST'])
def detect_from_url():
    """
    Detect persons in an image from URL.

    JSON body:
        - url: URL of the image
        - model: YOLO model name (optional)
        - confidence: minimum confidence (optional)
    """
    import urllib.request

    data = request.get_json()
    if not data or 'url' not in data:
        return jsonify({'error': 'No URL provided'}), 400

    model_name = data.get('model', 'yolov8n.pt')
    confidence = float(data.get('confidence', 0.25))
    use_sahi = bool(data.get('sahi', False))
    slice_size = int(data.get('slice_size', 640))
    slice_overlap = float(data.get('slice_overlap', 0.2))

    temp_path = None
    try:
        # Download image
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp:
            urllib.request.urlretrieve(data['url'], tmp.name)
            temp_path = tmp.name

        # Run detection
        counter = get_model(
            model_name, confidence, use_sahi, slice_size, slice_overlap
        )
        result = counter.count_persons(temp_path, save_output=False)

        # Get annotated image
        annotated = counter.visualize_result(temp_path, show_count=True)

        return jsonify({
            'success': True,
            'person_count': result['person_count'],
            'detections': [
                {
                    'id': i + 1,
                    'confidence': conf,
                    'bbox': bbox
                }
                for i, (conf, bbox) in enumerate(zip(result['confidences'], result['bounding_boxes']))
            ],
            'model': model_name,
            'sahi': {
                'enabled': use_sahi,
                'slice_size': slice_size,
                'slice_overlap': slice_overlap,
            } if use_sahi else {'enabled': False},
            'annotated_image': 'data:image/jpeg;base64,' + image_to_base64(annotated)
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

    finally:
        if temp_path and os.path.exists(temp_path):
            os.unlink(temp_path)


@app.route('/')
def index():
    """Redirect to API documentation."""
    return jsonify({
        'name': 'YOLO Person Counter API',
        'version': '1.0.0',
        'endpoints': {
            'GET /api/health': 'Health check',
            'GET /api/models': 'List available models',
            'POST /api/detect': 'Detect persons in uploaded image',
            'POST /api/detect/url': 'Detect persons in image from URL'
        }
    })


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='YOLO Person Counter API Server')
    parser.add_argument('--host', default='0.0.0.0', help='Host to bind to')
    parser.add_argument('--port', type=int, default=5000, help='Port to listen on')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    parser.add_argument('--preload', help='Preload a model (e.g., yolov8n.pt)')

    args = parser.parse_args()

    # Preload model if specified
    if args.preload:
        print(f"Preloading model: {args.preload}")
        get_model(args.preload)

    print(f"Starting server on {args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=args.debug)
