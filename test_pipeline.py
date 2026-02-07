"""Tests for the GPU-accelerated ASCII art restoration pipeline."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from pipeline import (
    RestoreConfig,
    apply_clahe,
    apply_fft_notch_filter,
    build_notch_mask,
    detect_cell_size,
    detect_fundamental_frequencies,
    gpu_guided_filter,
    gpu_unsharp_mask,
    gpu_upscale,
    grid_blur,
    restore_ascii_art,
)


@pytest.fixture
def device() -> torch.device:
    """Return available torch device."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def config() -> RestoreConfig:
    """Return default config."""
    return RestoreConfig()


@pytest.fixture
def blank_image() -> np.ndarray:
    """128x128 mid-gray image."""
    return np.full((128, 128), 128, dtype=np.uint8)


@pytest.fixture
def gradient_image() -> np.ndarray:
    """128x128 horizontal gradient image."""
    row = np.linspace(0, 255, 128, dtype=np.uint8)
    return np.tile(row, (128, 1))


@pytest.fixture
def grid_image() -> np.ndarray:
    """256x256 image with a periodic text-like grid pattern."""
    img = np.full((256, 256), 128, dtype=np.uint8)
    img[::8, :] = 255
    img[:, ::10] = 255
    return img


# ------------------------------------------------------------------
# Stage 1: Grid Blur
# ------------------------------------------------------------------
class TestDetectCellSize:
    def test_returns_floats(
        self, blank_image: np.ndarray, device: torch.device
    ) -> None:
        img_t = torch.from_numpy(blank_image.astype(np.float32) / 255.0).to(device)
        cell_h, cell_w = detect_cell_size(img_t)
        assert isinstance(cell_h, float)
        assert isinstance(cell_w, float)

    def test_detects_periodic_grid(
        self, grid_image: np.ndarray, device: torch.device
    ) -> None:
        img_t = torch.from_numpy(grid_image.astype(np.float32) / 255.0).to(device)
        cell_h, cell_w = detect_cell_size(img_t, prominence=0.05)
        assert cell_h > 0 or cell_w > 0

    def test_blank_image_no_detection(
        self, blank_image: np.ndarray, device: torch.device
    ) -> None:
        img_t = torch.from_numpy(blank_image.astype(np.float32) / 255.0).to(device)
        cell_h, cell_w = detect_cell_size(img_t)
        assert cell_h == 0.0
        assert cell_w == 0.0


class TestGridBlur:
    def test_output_shape_and_dtype(
        self, grid_image: np.ndarray, device: torch.device
    ) -> None:
        config = RestoreConfig(cell_height=8.0, cell_width=10.0)
        result = grid_blur(grid_image, config, device)
        assert result.shape == grid_image.shape
        assert result.dtype == np.uint8

    def test_reduces_grid_contrast(
        self, grid_image: np.ndarray, device: torch.device
    ) -> None:
        config = RestoreConfig(cell_height=8.0, cell_width=10.0)
        result = grid_blur(grid_image, config, device)
        # Blur should reduce the sharp grid lines
        assert result.std() < grid_image.std()

    def test_skips_when_no_grid(
        self, blank_image: np.ndarray, config: RestoreConfig, device: torch.device
    ) -> None:
        result = grid_blur(blank_image, config, device)
        assert np.array_equal(result, blank_image)


# ------------------------------------------------------------------
# Stage 2: Bicubic Upscale
# ------------------------------------------------------------------
class TestGpuUpscale:
    def test_upscales_to_target(
        self, blank_image: np.ndarray, device: torch.device
    ) -> None:
        result = gpu_upscale(blank_image, 256, 256, device)
        assert result.shape == (256, 256)
        assert result.dtype == np.uint8

    def test_no_change_when_same_size(
        self, blank_image: np.ndarray, device: torch.device
    ) -> None:
        result = gpu_upscale(blank_image, 128, 128, device)
        assert np.array_equal(result, blank_image)


# ------------------------------------------------------------------
# FFT Notch Filter (utility)
# ------------------------------------------------------------------
class TestDetectFundamentalFrequencies:
    def test_returns_ints(
        self, blank_image: np.ndarray, device: torch.device
    ) -> None:
        img_t = torch.from_numpy(blank_image.astype(np.float32) / 255.0).to(device)
        row_f, col_f = detect_fundamental_frequencies(img_t)
        assert isinstance(row_f, int)
        assert isinstance(col_f, int)

    def test_blank_image_no_peaks(
        self, blank_image: np.ndarray, device: torch.device
    ) -> None:
        img_t = torch.from_numpy(blank_image.astype(np.float32) / 255.0).to(device)
        row_f, col_f = detect_fundamental_frequencies(img_t)
        assert row_f == 0
        assert col_f == 0


class TestBuildNotchMask:
    def test_shape_and_range(self, device: torch.device) -> None:
        mask = build_notch_mask(64, 64, 8, 10, device=device)
        assert mask.shape == (64, 64)
        assert mask.min() >= 0.0
        assert mask.max() <= 1.0

    def test_no_frequencies_returns_ones(self, device: torch.device) -> None:
        mask = build_notch_mask(64, 64, 0, 0, device=device)
        assert torch.allclose(mask, torch.ones_like(mask))

    def test_notch_suppresses_target(self, device: torch.device) -> None:
        mask = build_notch_mask(128, 128, 16, 0, sigma=3.0, device=device)
        assert mask[16, 0].item() < 0.5


class TestApplyFftNotchFilter:
    def test_output_shape_and_dtype(
        self, grid_image: np.ndarray, config: RestoreConfig, device: torch.device
    ) -> None:
        result = apply_fft_notch_filter(grid_image, config, device)
        assert result.shape == grid_image.shape
        assert result.dtype == np.uint8

    def test_blank_image_unchanged(
        self, blank_image: np.ndarray, config: RestoreConfig, device: torch.device
    ) -> None:
        result = apply_fft_notch_filter(blank_image, config, device)
        assert np.allclose(result, blank_image, atol=2)


# ------------------------------------------------------------------
# Guided Filter (utility, not in main pipeline)
# ------------------------------------------------------------------
class TestGpuGuidedFilter:
    def test_output_shape_and_dtype(
        self, gradient_image: np.ndarray, config: RestoreConfig, device: torch.device
    ) -> None:
        result = gpu_guided_filter(gradient_image, config, device)
        assert result.shape == gradient_image.shape
        assert result.dtype == np.uint8

    def test_preserves_smooth_image(
        self, gradient_image: np.ndarray, config: RestoreConfig, device: torch.device
    ) -> None:
        result = gpu_guided_filter(gradient_image, config, device)
        diff = np.abs(result.astype(float) - gradient_image.astype(float))
        assert diff.mean() < 15

    def test_smooths_noisy_image(
        self, config: RestoreConfig, device: torch.device
    ) -> None:
        rng = np.random.default_rng(42)
        noisy = np.clip(128 + rng.normal(0, 30, (128, 128)), 0, 255).astype(np.uint8)
        result = gpu_guided_filter(noisy, config, device)
        assert result.std() < noisy.std()


# ------------------------------------------------------------------
# Stage 4: CLAHE
# ------------------------------------------------------------------
class TestApplyClahe:
    def test_output_shape_and_dtype(
        self, gradient_image: np.ndarray, config: RestoreConfig
    ) -> None:
        result = apply_clahe(gradient_image, config)
        assert result.shape == gradient_image.shape
        assert result.dtype == np.uint8

    def test_enhances_low_contrast(self, config: RestoreConfig) -> None:
        low_contrast = np.full((128, 128), 128, dtype=np.uint8)
        low_contrast[32:96, 32:96] = 130
        result = apply_clahe(low_contrast, config)
        assert result.std() >= low_contrast.std()


# ------------------------------------------------------------------
# Stage 5: Unsharp Mask
# ------------------------------------------------------------------
class TestGpuUnsharpMask:
    def test_output_shape_and_dtype(
        self, gradient_image: np.ndarray, config: RestoreConfig, device: torch.device
    ) -> None:
        result = gpu_unsharp_mask(gradient_image, config, device)
        assert result.shape == gradient_image.shape
        assert result.dtype == np.uint8

    def test_increases_edge_contrast(
        self, config: RestoreConfig, device: torch.device
    ) -> None:
        edge_img = np.zeros((128, 128), dtype=np.uint8)
        edge_img[:, 64:] = 200
        result = gpu_unsharp_mask(edge_img, config, device)
        diff = result.astype(float) - edge_img.astype(float)
        assert diff.max() > 0


# ------------------------------------------------------------------
# Integration: Full Pipeline
# ------------------------------------------------------------------
class TestRestoreAsciiArt:
    def test_full_pipeline_with_grid(self, grid_image: np.ndarray) -> None:
        config = RestoreConfig(cell_height=8.0, cell_width=10.0, upscale_factor=2)
        result = restore_ascii_art(grid_image, config)
        expected_h = grid_image.shape[0] * 2
        expected_w = grid_image.shape[1] * 2
        assert result.shape == (expected_h, expected_w)
        assert result.dtype == np.uint8

    def test_default_config_blank(self, blank_image: np.ndarray) -> None:
        config = RestoreConfig(upscale_factor=1)
        result = restore_ascii_art(blank_image, config)
        assert result.dtype == np.uint8

    def test_cpu_fallback(self, blank_image: np.ndarray) -> None:
        config = RestoreConfig(device="cpu", upscale_factor=1)
        result = restore_ascii_art(blank_image, config)
        assert result.dtype == np.uint8
