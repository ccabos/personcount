"""
Unit tests for Person Counter using mocking.
These tests work in offline environments where YOLO models cannot be downloaded.
"""

import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest
from PIL import Image


def create_mock_result():
    """Create a mock YOLO result object."""
    mock_result = MagicMock()

    # Mock boxes with 3 detected persons
    mock_boxes = MagicMock()
    mock_boxes.__len__ = Mock(return_value=3)
    mock_boxes.conf = MagicMock()
    mock_boxes.conf.cpu.return_value.numpy.return_value.tolist.return_value = [0.95, 0.87, 0.78]
    mock_boxes.xyxy = MagicMock()
    mock_boxes.xyxy.cpu.return_value.numpy.return_value.tolist.return_value = [
        [10.0, 20.0, 50.0, 100.0],
        [60.0, 25.0, 100.0, 110.0],
        [110.0, 30.0, 150.0, 105.0]
    ]

    mock_result.boxes = mock_boxes
    mock_result.plot.return_value = np.zeros((480, 640, 3), dtype=np.uint8)

    return mock_result


@pytest.fixture
def mock_yolo():
    """Create a mock YOLO model."""
    with patch('person_counter.YOLO') as mock_yolo_class:
        mock_model = MagicMock()
        mock_result = create_mock_result()
        mock_model.return_value = [mock_result]
        mock_yolo_class.return_value = mock_model
        yield mock_yolo_class


@pytest.fixture
def temp_image():
    """Create a temporary test image."""
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
        img = Image.new('RGB', (640, 480), color=(100, 150, 200))
        img.save(f.name)
        yield f.name
        os.unlink(f.name)


@pytest.fixture
def temp_directory():
    """Create a temporary directory with test images."""
    with tempfile.TemporaryDirectory() as tmpdir:
        for i in range(3):
            img = Image.new('RGB', (200, 200), color=(100 + i * 30, 100, 100))
            img.save(os.path.join(tmpdir, f"test_{i}.jpg"))
        yield tmpdir


class TestPersonCounterWithMock:
    """Tests using mocked YOLO model."""

    def test_init_loads_model(self, mock_yolo):
        """Test that PersonCounter loads YOLO model on init."""
        from person_counter import PersonCounter

        counter = PersonCounter(model_name="yolov8n.pt", confidence=0.3)

        mock_yolo.assert_called_once_with("yolov8n.pt")
        assert counter.confidence == 0.3
        assert counter.PERSON_CLASS_ID == 0

    def test_count_persons_returns_correct_structure(self, mock_yolo, temp_image):
        """Test that count_persons returns dict with all expected keys."""
        from person_counter import PersonCounter

        counter = PersonCounter()
        result = counter.count_persons(temp_image)

        assert isinstance(result, dict)
        assert "image_path" in result
        assert "person_count" in result
        assert "confidences" in result
        assert "bounding_boxes" in result
        assert "output_path" in result

    def test_count_persons_returns_correct_count(self, mock_yolo, temp_image):
        """Test that person count matches mock data."""
        from person_counter import PersonCounter

        counter = PersonCounter()
        result = counter.count_persons(temp_image)

        assert result["person_count"] == 3  # Mock returns 3 persons

    def test_count_persons_returns_confidences(self, mock_yolo, temp_image):
        """Test that confidences are returned correctly."""
        from person_counter import PersonCounter

        counter = PersonCounter()
        result = counter.count_persons(temp_image)

        assert len(result["confidences"]) == 3
        assert result["confidences"] == [0.95, 0.87, 0.78]

    def test_count_persons_returns_bounding_boxes(self, mock_yolo, temp_image):
        """Test that bounding boxes are returned correctly."""
        from person_counter import PersonCounter

        counter = PersonCounter()
        result = counter.count_persons(temp_image)

        assert len(result["bounding_boxes"]) == 3
        assert result["bounding_boxes"][0] == [10.0, 20.0, 50.0, 100.0]

    def test_count_persons_file_not_found(self, mock_yolo):
        """Test FileNotFoundError for missing image."""
        from person_counter import PersonCounter

        counter = PersonCounter()

        with pytest.raises(FileNotFoundError):
            counter.count_persons("/nonexistent/path/image.jpg")

    def test_count_persons_save_output(self, mock_yolo, temp_image):
        """Test that save_output creates annotated image."""
        from person_counter import PersonCounter

        counter = PersonCounter()

        with tempfile.TemporaryDirectory() as tmpdir:
            # Copy test image to temp dir
            import shutil
            test_path = os.path.join(tmpdir, "test.jpg")
            shutil.copy(temp_image, test_path)

            result = counter.count_persons(test_path, save_output=True)

            assert result["output_path"] is not None
            assert "_counted" in result["output_path"]

    def test_count_persons_no_save(self, mock_yolo, temp_image):
        """Test that output_path is None when not saving."""
        from person_counter import PersonCounter

        counter = PersonCounter()
        result = counter.count_persons(temp_image, save_output=False)

        assert result["output_path"] is None

    def test_model_called_with_correct_params(self, mock_yolo, temp_image):
        """Test that YOLO model is called with correct parameters."""
        from person_counter import PersonCounter

        counter = PersonCounter(confidence=0.4)
        counter.count_persons(temp_image)

        # Check model was called with image path, confidence, and person class
        counter.model.assert_called_once()
        call_kwargs = counter.model.call_args[1]
        assert call_kwargs['conf'] == 0.4
        assert call_kwargs['classes'] == [0]  # Person class
        assert call_kwargs['verbose'] is False


class TestDirectoryProcessingWithMock:
    """Tests for directory processing with mocked YOLO."""

    def test_directory_returns_list(self, mock_yolo, temp_directory):
        """Test that directory processing returns list of results."""
        from person_counter import PersonCounter

        counter = PersonCounter()
        results = counter.count_persons_in_directory(temp_directory)

        assert isinstance(results, list)
        assert len(results) == 3  # 3 images created in fixture

    def test_directory_not_found(self, mock_yolo):
        """Test NotADirectoryError for missing directory."""
        from person_counter import PersonCounter

        counter = PersonCounter()

        with pytest.raises(NotADirectoryError):
            counter.count_persons_in_directory("/nonexistent/dir")

    def test_directory_empty(self, mock_yolo):
        """Test empty directory returns empty list."""
        from person_counter import PersonCounter

        counter = PersonCounter()

        with tempfile.TemporaryDirectory() as tmpdir:
            results = counter.count_persons_in_directory(tmpdir)
            assert results == []

    def test_directory_filters_non_images(self, mock_yolo):
        """Test that non-image files are filtered out."""
        from person_counter import PersonCounter

        counter = PersonCounter()

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create one image and one text file
            Image.new('RGB', (100, 100)).save(os.path.join(tmpdir, "img.jpg"))
            with open(os.path.join(tmpdir, "text.txt"), 'w') as f:
                f.write("not an image")

            results = counter.count_persons_in_directory(tmpdir)
            assert len(results) == 1


class TestVisualizationWithMock:
    """Tests for visualization with mocked YOLO."""

    def test_visualize_returns_array(self, mock_yolo, temp_image):
        """Test that visualize_result returns numpy array."""
        from person_counter import PersonCounter

        counter = PersonCounter()
        result = counter.visualize_result(temp_image)

        assert isinstance(result, np.ndarray)

    def test_visualize_correct_shape(self, mock_yolo, temp_image):
        """Test that returned array has 3 channels."""
        from person_counter import PersonCounter

        counter = PersonCounter()
        result = counter.visualize_result(temp_image)

        assert len(result.shape) == 3
        assert result.shape[2] == 3  # RGB

    @patch('cv2.imwrite')
    def test_visualize_saves_file(self, mock_imwrite, mock_yolo, temp_image):
        """Test that output file is written when path specified."""
        from person_counter import PersonCounter

        counter = PersonCounter()
        counter.visualize_result(temp_image, output_path="/tmp/output.jpg")

        mock_imwrite.assert_called_once()


class TestEmptyResults:
    """Tests for handling images with no detected persons."""

    def test_no_persons_detected(self, temp_image):
        """Test handling when no persons are detected."""
        with patch('person_counter.YOLO') as mock_yolo_class:
            mock_model = MagicMock()

            # Create result with no detections
            mock_result = MagicMock()
            mock_boxes = MagicMock()
            mock_boxes.__len__ = Mock(return_value=0)
            mock_boxes.conf = MagicMock()
            mock_boxes.conf.cpu.return_value.numpy.return_value.tolist.return_value = []
            mock_boxes.xyxy = MagicMock()
            mock_boxes.xyxy.cpu.return_value.numpy.return_value.tolist.return_value = []
            mock_result.boxes = mock_boxes

            mock_model.return_value = [mock_result]
            mock_yolo_class.return_value = mock_model

            from person_counter import PersonCounter

            counter = PersonCounter()
            result = counter.count_persons(temp_image)

            assert result["person_count"] == 0
            assert result["confidences"] == []
            assert result["bounding_boxes"] == []


class TestPrintSummary:
    """Tests for print_summary function (no mocking needed)."""

    def test_summary_with_successful_results(self, capsys):
        """Test summary output with successful results."""
        from person_counter import print_summary

        results = [
            {"image_path": "a.jpg", "person_count": 10, "confidences": []},
            {"image_path": "b.jpg", "person_count": 20, "confidences": []},
        ]

        print_summary(results)
        output = capsys.readouterr().out

        assert "ZUSAMMENFASSUNG" in output
        assert "30" in output  # Total
        assert "Erfolgreich: 2" in output

    def test_summary_with_errors(self, capsys):
        """Test summary output with some errors."""
        from person_counter import print_summary

        results = [
            {"image_path": "a.jpg", "person_count": 10, "confidences": []},
            {"image_path": "b.jpg", "error": "Failed to process"},
        ]

        print_summary(results)
        output = capsys.readouterr().out

        assert "Erfolgreich: 1" in output
        assert "Fehlgeschlagen: 1" in output

    def test_summary_empty(self, capsys):
        """Test summary with empty results."""
        from person_counter import print_summary

        print_summary([])
        output = capsys.readouterr().out

        assert "Verarbeitete Bilder: 0" in output


class TestConfidenceThreshold:
    """Tests for confidence threshold behavior."""

    def test_different_confidence_values(self, temp_image):
        """Test that different confidence thresholds are passed to model."""
        with patch('person_counter.YOLO') as mock_yolo_class:
            mock_model = MagicMock()
            mock_result = create_mock_result()
            mock_model.return_value = [mock_result]
            mock_yolo_class.return_value = mock_model

            from person_counter import PersonCounter

            for conf in [0.1, 0.25, 0.5, 0.9]:
                counter = PersonCounter(confidence=conf)
                counter.count_persons(temp_image)

                call_kwargs = mock_model.call_args[1]
                assert call_kwargs['conf'] == conf
