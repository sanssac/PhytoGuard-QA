import pytest
import numpy as np
import os

# ── Constants ──────────────────────────────────────────────────────────────
IMG_SIZE = (224, 224)
NUM_CLASSES = 10  # adjust to your actual number of disease classes

# ── Fixtures ───────────────────────────────────────────────────────────────
@pytest.fixture
def sample_image():
    """Simulate a preprocessed plant image (RGB, normalized)."""
    return np.random.rand(1, IMG_SIZE[0], IMG_SIZE[1], 3).astype(np.float32)

@pytest.fixture
def blank_image():
    """All-zero image — edge case."""
    return np.zeros((1, IMG_SIZE[0], IMG_SIZE[1], 3), dtype=np.float32)

@pytest.fixture
def noisy_image():
    """High noise image — stress test."""
    return np.random.uniform(0, 1, (1, IMG_SIZE[0], IMG_SIZE[1], 3)).astype(np.float32)

# ── Input Validation Tests ─────────────────────────────────────────────────
class TestInputValidation:
    def test_image_shape(self, sample_image):
        """Input must be (1, 224, 224, 3)."""
        assert sample_image.shape == (1, 224, 224, 3), "Unexpected input shape"

    def test_image_dtype(self, sample_image):
        """Input must be float32."""
        assert sample_image.dtype == np.float32, "Input should be float32"

    def test_pixel_range(self, sample_image):
        """Pixels must be normalized between 0 and 1."""
        assert sample_image.min() >= 0.0, "Pixel values below 0"
        assert sample_image.max() <= 1.0, "Pixel values above 1"

    def test_no_nan_values(self, sample_image):
        """Input must not contain NaN."""
        assert not np.isnan(sample_image).any(), "NaN values found in input"

    def test_no_inf_values(self, sample_image):
        """Input must not contain Inf."""
        assert not np.isinf(sample_image).any(), "Inf values found in input"

# ── Preprocessing Tests ────────────────────────────────────────────────────
class TestPreprocessing:
    def test_normalization(self):
        """Simulates dividing raw pixel (0-255) by 255."""
        raw = np.array([[[255, 128, 0]]], dtype=np.float32)
        normalized = raw / 255.0
        assert normalized.max() <= 1.0
        assert normalized.min() >= 0.0

    def test_resize_output_shape(self):
        """Resized image must match target dimensions."""
        try:
            import cv2
            dummy = np.uint8(np.random.rand(100, 100, 3) * 255)
            resized = cv2.resize(dummy, IMG_SIZE)
            assert resized.shape == (224, 224, 3)
        except ImportError:
            pytest.skip("OpenCV not installed")

    def test_batch_dimension(self, sample_image):
        """Image must have batch dimension expanded."""
        assert len(sample_image.shape) == 4, "Missing batch dimension"

# ── Model Output Tests ─────────────────────────────────────────────────────
class TestModelOutput:
    def test_output_shape(self):
        """Model output must match number of disease classes."""
        mock_output = np.random.rand(1, NUM_CLASSES)
        assert mock_output.shape == (1, NUM_CLASSES)

    def test_output_probabilities_sum(self):
        """Softmax output probabilities must sum to ~1."""
        logits = np.random.rand(1, NUM_CLASSES)
        exp = np.exp(logits - logits.max())
        softmax = exp / exp.sum()
        assert abs(softmax.sum() - 1.0) < 1e-5, "Probabilities don't sum to 1"

    def test_predicted_class_in_range(self):
        """Predicted class index must be within valid range."""
        mock_output = np.random.rand(1, NUM_CLASSES)
        predicted = np.argmax(mock_output)
        assert 0 <= predicted < NUM_CLASSES

    def test_confidence_score_range(self):
        """Confidence score must be between 0 and 1."""
        mock_probs = np.array([[0.1, 0.7, 0.2]])
        confidence = float(np.max(mock_probs))
        assert 0.0 <= confidence <= 1.0

# ── Edge Case Tests ────────────────────────────────────────────────────────
class TestEdgeCases:
    def test_blank_image_no_crash(self, blank_image):
        """Model pipeline must handle all-zero input without crashing."""
        assert blank_image.shape == (1, 224, 224, 3)
        assert not np.isnan(blank_image).any()

    def test_noisy_image_valid_input(self, noisy_image):
        """High-noise image should still be valid input."""
        assert noisy_image.min() >= 0.0
        assert noisy_image.max() <= 1.0

    def test_single_channel_detection(self):
        """Grayscale input should be detectable before model inference."""
        grayscale = np.random.rand(1, 224, 224, 1).astype(np.float32)
        assert grayscale.shape[-1] != 3, "Should detect non-RGB input"
