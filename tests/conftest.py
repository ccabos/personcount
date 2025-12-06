"""
Pytest fixtures for Person Counter tests.
"""

import os
import tempfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image


@pytest.fixture(scope="session")
def test_images_dir():
    """Create a temporary directory with test images."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create simple test images
        create_test_image(
            os.path.join(tmpdir, "simple.jpg"),
            width=200,
            height=200,
            num_shapes=3
        )
        create_test_image(
            os.path.join(tmpdir, "empty.jpg"),
            width=200,
            height=200,
            num_shapes=0
        )
        create_test_image(
            os.path.join(tmpdir, "crowded.jpg"),
            width=400,
            height=400,
            num_shapes=10
        )

        yield tmpdir


@pytest.fixture(scope="session")
def single_test_image():
    """Create a single temporary test image."""
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
        create_test_image(f.name, 300, 300, 5)
        yield f.name
        os.unlink(f.name)


@pytest.fixture(scope="session")
def empty_test_image():
    """Create an empty test image (no person-like shapes)."""
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
        img = Image.new('RGB', (200, 200), color=(100, 150, 200))
        img.save(f.name)
        yield f.name
        os.unlink(f.name)


@pytest.fixture
def temp_output_dir():
    """Create a temporary directory for output files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


def create_test_image(path: str, width: int, height: int, num_shapes: int):
    """
    Create a test image with person-like shapes.

    Creates simple head+body shapes that may be detected as persons
    by YOLO (though not guaranteed for synthetic images).
    """
    img = Image.new('RGB', (width, height), color=(100, 150, 200))

    if num_shapes > 0:
        from PIL import ImageDraw
        draw = ImageDraw.Draw(img)

        spacing = width // (num_shapes + 1)
        for i in range(num_shapes):
            x = spacing * (i + 1)
            y = height // 2

            # Draw head (circle)
            head_radius = 15
            draw.ellipse(
                [x - head_radius, y - 40 - head_radius,
                 x + head_radius, y - 40 + head_radius],
                fill=(210, 180, 160)
            )

            # Draw body (rectangle)
            draw.rectangle(
                [x - 20, y - 25, x + 20, y + 40],
                fill=(50, 50, 100)
            )

    img.save(path)


@pytest.fixture(scope="module")
def person_counter():
    """
    Create a PersonCounter instance.

    Uses the smallest model for faster testing.
    Skips if ultralytics is not installed.
    """
    pytest.importorskip("ultralytics")

    from person_counter import PersonCounter
    return PersonCounter(model_name="yolov8n.pt", confidence=0.25)
