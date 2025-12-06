"""
Unit tests for the Person Counter module.
"""

import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PIL import Image


class TestPersonCounterInit:
    """Tests for PersonCounter initialization."""

    def test_init_default_model(self, person_counter):
        """Test that PersonCounter initializes with default settings."""
        assert person_counter is not None
        assert person_counter.confidence == 0.25
        assert person_counter.PERSON_CLASS_ID == 0

    def test_init_custom_confidence(self):
        """Test PersonCounter with custom confidence threshold."""
        pytest.importorskip("ultralytics")
        from person_counter import PersonCounter

        counter = PersonCounter(confidence=0.5)
        assert counter.confidence == 0.5

    def test_init_invalid_model(self):
        """Test that invalid model raises appropriate error."""
        pytest.importorskip("ultralytics")
        from person_counter import PersonCounter

        with pytest.raises(Exception):
            PersonCounter(model_name="nonexistent_model.pt")


class TestCountPersons:
    """Tests for the count_persons method."""

    def test_count_persons_returns_dict(self, person_counter, single_test_image):
        """Test that count_persons returns a dictionary with expected keys."""
        result = person_counter.count_persons(single_test_image)

        assert isinstance(result, dict)
        assert "image_path" in result
        assert "person_count" in result
        assert "confidences" in result
        assert "bounding_boxes" in result
        assert "output_path" in result

    def test_count_persons_path_in_result(self, person_counter, single_test_image):
        """Test that the image path is correctly returned."""
        result = person_counter.count_persons(single_test_image)
        assert result["image_path"] == single_test_image

    def test_count_persons_non_negative(self, person_counter, single_test_image):
        """Test that person count is non-negative."""
        result = person_counter.count_persons(single_test_image)
        assert result["person_count"] >= 0

    def test_count_persons_confidences_match_count(self, person_counter, single_test_image):
        """Test that confidence list length matches person count."""
        result = person_counter.count_persons(single_test_image)
        assert len(result["confidences"]) == result["person_count"]

    def test_count_persons_bboxes_match_count(self, person_counter, single_test_image):
        """Test that bounding box list length matches person count."""
        result = person_counter.count_persons(single_test_image)
        assert len(result["bounding_boxes"]) == result["person_count"]

    def test_count_persons_file_not_found(self, person_counter):
        """Test that FileNotFoundError is raised for missing files."""
        with pytest.raises(FileNotFoundError):
            person_counter.count_persons("/nonexistent/path/image.jpg")

    def test_count_persons_save_output(self, person_counter, single_test_image, temp_output_dir):
        """Test that save_output creates an annotated image."""
        # Copy test image to temp dir
        import shutil
        test_img_path = os.path.join(temp_output_dir, "test.jpg")
        shutil.copy(single_test_image, test_img_path)

        result = person_counter.count_persons(test_img_path, save_output=True)

        assert result["output_path"] is not None
        assert os.path.exists(result["output_path"])
        assert "_counted" in result["output_path"]

    def test_count_persons_no_save_output(self, person_counter, single_test_image):
        """Test that output_path is None when save_output is False."""
        result = person_counter.count_persons(single_test_image, save_output=False)
        assert result["output_path"] is None

    def test_confidences_in_valid_range(self, person_counter, single_test_image):
        """Test that all confidences are between 0 and 1."""
        result = person_counter.count_persons(single_test_image)
        for conf in result["confidences"]:
            assert 0.0 <= conf <= 1.0

    def test_bounding_boxes_format(self, person_counter, single_test_image):
        """Test that bounding boxes have correct format [x1, y1, x2, y2]."""
        result = person_counter.count_persons(single_test_image)
        for bbox in result["bounding_boxes"]:
            assert len(bbox) == 4
            x1, y1, x2, y2 = bbox
            assert x1 <= x2  # x1 should be left of x2
            assert y1 <= y2  # y1 should be above y2


class TestCountPersonsInDirectory:
    """Tests for the count_persons_in_directory method."""

    def test_directory_returns_list(self, person_counter, test_images_dir):
        """Test that directory processing returns a list."""
        results = person_counter.count_persons_in_directory(test_images_dir)
        assert isinstance(results, list)

    def test_directory_processes_all_images(self, person_counter, test_images_dir):
        """Test that all images in directory are processed."""
        results = person_counter.count_persons_in_directory(test_images_dir)
        # Should find at least the images we created
        assert len(results) >= 3

    def test_directory_not_found(self, person_counter):
        """Test that NotADirectoryError is raised for missing directory."""
        with pytest.raises(NotADirectoryError):
            person_counter.count_persons_in_directory("/nonexistent/directory")

    def test_directory_empty(self, person_counter):
        """Test processing an empty directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            results = person_counter.count_persons_in_directory(tmpdir)
            assert results == []

    def test_directory_filters_extensions(self, person_counter):
        """Test that only specified extensions are processed."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create image file
            img_path = os.path.join(tmpdir, "test.jpg")
            Image.new('RGB', (100, 100)).save(img_path)

            # Create non-image file
            txt_path = os.path.join(tmpdir, "test.txt")
            with open(txt_path, 'w') as f:
                f.write("not an image")

            results = person_counter.count_persons_in_directory(tmpdir)
            assert len(results) == 1


class TestVisualizeResult:
    """Tests for the visualize_result method."""

    def test_visualize_returns_array(self, person_counter, single_test_image):
        """Test that visualize_result returns a numpy array."""
        result = person_counter.visualize_result(single_test_image)
        assert isinstance(result, np.ndarray)

    def test_visualize_array_shape(self, person_counter, single_test_image):
        """Test that returned array has correct shape (H, W, 3)."""
        result = person_counter.visualize_result(single_test_image)
        assert len(result.shape) == 3
        assert result.shape[2] == 3  # RGB channels

    def test_visualize_saves_output(self, person_counter, single_test_image, temp_output_dir):
        """Test that output file is created when path is specified."""
        output_path = os.path.join(temp_output_dir, "output.jpg")
        person_counter.visualize_result(single_test_image, output_path=output_path)
        assert os.path.exists(output_path)

    def test_visualize_with_count_overlay(self, person_counter, single_test_image):
        """Test visualization with count overlay enabled."""
        result = person_counter.visualize_result(single_test_image, show_count=True)
        assert result is not None


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_very_small_image(self, person_counter):
        """Test processing a very small image."""
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
            Image.new('RGB', (10, 10)).save(f.name)
            try:
                result = person_counter.count_persons(f.name)
                assert result["person_count"] >= 0
            finally:
                os.unlink(f.name)

    def test_large_image(self, person_counter):
        """Test processing a larger image."""
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
            Image.new('RGB', (1920, 1080), color=(100, 100, 100)).save(f.name)
            try:
                result = person_counter.count_persons(f.name)
                assert result["person_count"] >= 0
            finally:
                os.unlink(f.name)

    def test_different_image_formats(self, person_counter):
        """Test processing different image formats."""
        formats = [(".jpg", "JPEG"), (".png", "PNG"), (".bmp", "BMP")]

        for ext, fmt in formats:
            with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as f:
                img = Image.new('RGB', (100, 100))
                img.save(f.name, format=fmt if fmt != "JPEG" else None)
                try:
                    result = person_counter.count_persons(f.name)
                    assert "person_count" in result
                finally:
                    os.unlink(f.name)


class TestPrintSummary:
    """Tests for the print_summary function."""

    def test_print_summary_with_results(self, capsys):
        """Test print_summary with successful results."""
        from person_counter import print_summary

        results = [
            {"image_path": "img1.jpg", "person_count": 10, "confidences": []},
            {"image_path": "img2.jpg", "person_count": 5, "confidences": []},
        ]
        print_summary(results)

        captured = capsys.readouterr()
        assert "ZUSAMMENFASSUNG" in captured.out
        assert "15" in captured.out  # Total count
        assert "2" in captured.out   # Number of images

    def test_print_summary_with_errors(self, capsys):
        """Test print_summary with some failed results."""
        from person_counter import print_summary

        results = [
            {"image_path": "img1.jpg", "person_count": 10, "confidences": []},
            {"image_path": "img2.jpg", "error": "File not found"},
        ]
        print_summary(results)

        captured = capsys.readouterr()
        assert "Erfolgreich: 1" in captured.out
        assert "Fehlgeschlagen: 1" in captured.out

    def test_print_summary_empty(self, capsys):
        """Test print_summary with empty results."""
        from person_counter import print_summary

        print_summary([])

        captured = capsys.readouterr()
        assert "Verarbeitete Bilder: 0" in captured.out


class TestCLI:
    """Tests for command-line interface."""

    def test_main_with_single_image(self, single_test_image, capsys):
        """Test main function with a single image."""
        import sys
        from person_counter import main

        original_argv = sys.argv
        sys.argv = ["person_counter.py", single_test_image]

        try:
            main()
            captured = capsys.readouterr()
            assert "Erkannte Personen:" in captured.out
        finally:
            sys.argv = original_argv

    def test_main_with_directory(self, test_images_dir, capsys):
        """Test main function with a directory."""
        import sys
        from person_counter import main

        original_argv = sys.argv
        sys.argv = ["person_counter.py", test_images_dir]

        try:
            main()
            captured = capsys.readouterr()
            assert "ZUSAMMENFASSUNG" in captured.out
        finally:
            sys.argv = original_argv

    def test_main_invalid_path(self, capsys):
        """Test main function with invalid path."""
        import sys
        from person_counter import main

        original_argv = sys.argv
        sys.argv = ["person_counter.py", "/nonexistent/path"]

        try:
            with pytest.raises(SystemExit):
                main()
        finally:
            sys.argv = original_argv
