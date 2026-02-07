"""GPU-accelerated pipeline for ASCII art image restoration.

Replaces brute-force iterative averaging with anisotropic Gaussian blur
and frequency-domain post-processing.

Pipeline stages:
    1. Grid Blur — light anisotropic Gaussian matched to character cell size
    2. Bicubic Upscale — smooth interpolation to target resolution
    3. FFT Notch Filter — residual grid frequency suppression
    4. CLAHE — adaptive local contrast enhancement
    5. Unsharp Mask — GPU-accelerated sharpening

Also provides standalone guided filter utilities for edge-preserving smoothing.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from scipy.signal import find_peaks

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------
@dataclass(frozen=True)
class RestoreConfig:
    """All pipeline parameters in one place."""

    # Stage 1: Grid Blur (0 = auto-detect cell size from FFT)
    cell_height: float = 0.0
    cell_width: float = 0.0
    blur_sigma_scale: float = 0.7  # sigma = cell_size * scale (light blur)

    # Stage 2: Upscale (relative to original image)
    upscale_factor: int = 2

    # Grid detection parameters
    notch_sigma: float = 4.0
    peak_prominence: float = 0.15
    peak_min_distance: int = 5

    # Guided Filter (utility, not in main pipeline)
    guided_radius: int = 8
    guided_eps: float = 0.01

    # Stage 4: CLAHE
    clahe_clip_limit: float = 3.0
    clahe_tile_grid: tuple[int, int] = (8, 8)

    # Stage 5: Unsharp Mask
    sharpen_sigma: float = 2.0
    sharpen_amount: float = 1.5

    # General
    device: str = "cuda"


# ------------------------------------------------------------------
# Stage 1: Grid Blur (anisotropic Gaussian matched to cell size)
# ------------------------------------------------------------------
def detect_cell_size(
    img_t: torch.Tensor,
    *,
    prominence: float = 0.15,
    min_distance: int = 5,
) -> tuple[float, float]:
    """Detect the character cell size in pixels from FFT fundamental frequency.

    Args:
        img_t: Float32 tensor on device, values in [0, 1], shape (H, W).
        prominence: Minimum peak prominence (fraction of max).
        min_distance: Minimum distance between peaks in frequency bins.

    Returns:
        Tuple of (cell_height, cell_width) in pixels. Either may be 0.0
        if no periodic pattern was detected.
    """
    spectrum = torch.abs(torch.fft.fft2(img_t))
    h, w = spectrum.shape

    row_profile = spectrum[: h // 2, :].mean(dim=1).cpu().numpy()
    col_profile = spectrum[:, : w // 2].mean(dim=0).cpu().numpy()

    row_profile = row_profile / row_profile.max()
    col_profile = col_profile / col_profile.max()

    cell_h = 0.0
    row_peaks, row_props = find_peaks(
        row_profile[1:], prominence=prominence, distance=min_distance
    )
    if len(row_peaks) > 0:
        best = int(np.argmax(row_props["prominences"]))
        row_fund = row_peaks[best] + 1
        cell_h = h / row_fund

    cell_w = 0.0
    col_peaks, col_props = find_peaks(
        col_profile[1:], prominence=prominence, distance=min_distance
    )
    if len(col_peaks) > 0:
        best = int(np.argmax(col_props["prominences"]))
        col_fund = col_peaks[best] + 1
        cell_w = w / col_fund

    logger.info("Detected cell size: %.1f x %.1f px", cell_h, cell_w)
    return cell_h, cell_w


def _make_gaussian_kernel_1d(sigma: float, device: torch.device) -> torch.Tensor:
    """Create a 1-D Gaussian kernel on device."""
    ksize = int(6 * sigma) | 1
    half = ksize // 2
    x = torch.arange(-half, half + 1, device=device, dtype=torch.float32)
    k = torch.exp(-x * x / (2 * sigma * sigma))
    return k / k.sum()


def grid_blur(
    img: np.ndarray,
    config: RestoreConfig,
    device: torch.device,
) -> np.ndarray:
    """Remove character grid via anisotropic Gaussian blur matched to cell size.

    Uses separate sigma for each axis, automatically derived from the
    detected cell dimensions. sigma = cell_size * blur_sigma_scale ensures
    the blur washes out the grid pattern without excessive blurring.

    Args:
        img: Grayscale uint8 image.
        config: Pipeline configuration.
        device: Torch device.

    Returns:
        Grid-removed grayscale uint8 image (same dimensions as input).
    """
    h, w = img.shape
    img_t = torch.from_numpy(img.astype(np.float32) / 255.0).to(device)

    cell_h = config.cell_height
    cell_w = config.cell_width
    if cell_h == 0.0 or cell_w == 0.0:
        detected_h, detected_w = detect_cell_size(
            img_t,
            prominence=config.peak_prominence,
            min_distance=config.peak_min_distance,
        )
        if cell_h == 0.0:
            cell_h = detected_h
        if cell_w == 0.0:
            cell_w = detected_w

    if cell_h < 2.0 and cell_w < 2.0:
        logger.warning("Could not detect grid — skipping blur")
        return img

    scale = config.blur_sigma_scale
    sigma_y = cell_h * scale if cell_h >= 2.0 else 1.0
    sigma_x = cell_w * scale if cell_w >= 2.0 else 1.0

    logger.info(
        "Grid blur: sigma_y=%.1f, sigma_x=%.1f (cell %.1fx%.1f, scale %.1f)",
        sigma_y, sigma_x, cell_h, cell_w, scale,
    )

    img_t = img_t.unsqueeze(0).unsqueeze(0)

    # Separable anisotropic Gaussian blur
    kx = _make_gaussian_kernel_1d(sigma_x, device)
    ky = _make_gaussian_kernel_1d(sigma_y, device)

    blurred = F.conv2d(img_t, kx.view(1, 1, 1, -1), padding=(0, len(kx) // 2))
    blurred = F.conv2d(blurred, ky.view(1, 1, -1, 1), padding=(len(ky) // 2, 0))
    blurred = blurred.squeeze().clamp(0.0, 1.0)

    return (blurred.cpu().numpy() * 255.0).astype(np.uint8)


# ------------------------------------------------------------------
# Stage 2: Bicubic Upscale
# ------------------------------------------------------------------
def gpu_upscale(
    img: np.ndarray,
    target_h: int,
    target_w: int,
    device: torch.device,
) -> np.ndarray:
    """Bicubic upscale on GPU to specific target dimensions.

    Args:
        img: Grayscale uint8 image.
        target_h: Target height.
        target_w: Target width.
        device: Torch device.

    Returns:
        Upscaled grayscale uint8 image.
    """
    if img.shape[0] == target_h and img.shape[1] == target_w:
        return img

    t = torch.from_numpy(img.astype(np.float32) / 255.0).to(device)
    t = t.unsqueeze(0).unsqueeze(0)

    t = F.interpolate(t, size=(target_h, target_w), mode="bicubic", align_corners=False)
    t = t.squeeze().clamp(0.0, 1.0)

    return (t.cpu().numpy() * 255.0).astype(np.uint8)


# ------------------------------------------------------------------
# Stage 3: FFT Notch Filter (residual grid frequency suppression)
# ------------------------------------------------------------------
def detect_fundamental_frequencies(
    img_t: torch.Tensor,
    *,
    prominence: float = 0.15,
    min_distance: int = 5,
) -> tuple[int, int]:
    """Detect the fundamental row and column frequencies of any residual grid.

    Args:
        img_t: Float32 tensor on device, values in [0, 1], shape (H, W).
        prominence: Minimum peak prominence (fraction of max).
        min_distance: Minimum distance between peaks in frequency bins.

    Returns:
        Tuple of (row_fundamental, col_fundamental). Either may be 0.
    """
    spectrum = torch.abs(torch.fft.fft2(img_t))
    h, w = spectrum.shape

    row_profile = spectrum[: h // 2, :].mean(dim=1).cpu().numpy()
    col_profile = spectrum[:, : w // 2].mean(dim=0).cpu().numpy()

    row_profile = row_profile / row_profile.max()
    col_profile = col_profile / col_profile.max()

    row_fund = 0
    row_peaks, row_props = find_peaks(
        row_profile[1:], prominence=prominence, distance=min_distance
    )
    if len(row_peaks) > 0:
        best = int(np.argmax(row_props["prominences"]))
        row_fund = int(row_peaks[best] + 1)

    col_fund = 0
    col_peaks, col_props = find_peaks(
        col_profile[1:], prominence=prominence, distance=min_distance
    )
    if len(col_peaks) > 0:
        best = int(np.argmax(col_props["prominences"]))
        col_fund = int(col_peaks[best] + 1)

    return row_fund, col_fund


def build_notch_mask(
    h: int,
    w: int,
    row_fund: int,
    col_fund: int,
    *,
    sigma: float = 3.0,
    device: torch.device,
) -> torch.Tensor:
    """Build a 2-D notch mask for grid frequency suppression.

    Places Gaussian notches at all lattice points (kr*fr, kc*fc)
    and their mirrors.

    Args:
        h: Image height.
        w: Image width.
        row_fund: Fundamental row frequency (0 = none).
        col_fund: Fundamental column frequency (0 = none).
        sigma: Gaussian notch width in frequency-domain pixels.
        device: Torch device.

    Returns:
        Float32 tensor of shape (h, w) with values in [0, 1].
    """
    mask = torch.ones(h, w, device=device, dtype=torch.float32)

    if row_fund == 0 and col_fund == 0:
        return mask

    inv_2sigma2 = 1.0 / (2.0 * sigma * sigma)

    max_kr = (h // 2) // row_fund if row_fund > 0 else 0
    max_kc = (w // 2) // col_fund if col_fund > 0 else 0

    gy = torch.arange(h, device=device, dtype=torch.float32)
    gx = torch.arange(w, device=device, dtype=torch.float32)

    points: list[tuple[int, int]] = []
    for kr in range(0, max_kr + 1):
        for kc in range(0, max_kc + 1):
            if kr == 0 and kc == 0:
                continue
            fy = kr * row_fund
            fx = kc * col_fund
            for sy in ([fy, h - fy] if fy > 0 else [0]):
                for sx in ([fx, w - fx] if fx > 0 else [0]):
                    points.append((sy, sx))

    if not points:
        return mask

    chunk_size = 256
    for i in range(0, len(points), chunk_size):
        chunk = points[i : i + chunk_size]
        targets_y = torch.tensor(
            [p[0] for p in chunk], device=device, dtype=torch.float32
        )
        targets_x = torch.tensor(
            [p[1] for p in chunk], device=device, dtype=torch.float32
        )
        gauss_y = torch.exp(
            -((gy.unsqueeze(1) - targets_y.unsqueeze(0)) ** 2) * inv_2sigma2
        )
        gauss_x = torch.exp(
            -((gx.unsqueeze(1) - targets_x.unsqueeze(0)) ** 2) * inv_2sigma2
        )
        notch_2d = torch.einsum("hn,wn->hw", gauss_y, gauss_x)
        mask -= notch_2d

    mask.clamp_(0.0, 1.0)
    return mask


def apply_fft_notch_filter(
    img: np.ndarray,
    config: RestoreConfig,
    device: torch.device,
) -> np.ndarray:
    """Remove residual grid pattern via 2-D FFT notch filtering on GPU.

    Args:
        img: Grayscale uint8 image.
        config: Pipeline configuration.
        device: Torch device.

    Returns:
        Filtered grayscale uint8 image (same shape as input).
    """
    h, w = img.shape
    img_t = torch.from_numpy(img.astype(np.float32) / 255.0).to(device)

    row_fund, col_fund = detect_fundamental_frequencies(
        img_t,
        prominence=config.peak_prominence,
        min_distance=config.peak_min_distance,
    )

    if row_fund == 0 and col_fund == 0:
        logger.info("No residual grid detected — skipping notch filter")
        return img

    logger.info("Residual grid detected (row=%d, col=%d) — applying notch", row_fund, col_fund)

    freq = torch.fft.fft2(img_t)
    mask = build_notch_mask(
        h, w, row_fund, col_fund,
        sigma=config.notch_sigma, device=device,
    )
    result = torch.fft.ifft2(freq * mask).real.clamp(0.0, 1.0)

    return (result.cpu().numpy() * 255.0).astype(np.uint8)


# ------------------------------------------------------------------
# Guided Filter (utility, not in main pipeline)
# ------------------------------------------------------------------
def _box_filter_2d(x: torch.Tensor, radius: int) -> torch.Tensor:
    """Box filter via conv2d.

    Args:
        x: Tensor of shape (1, 1, H, W).
        radius: Filter radius.

    Returns:
        Filtered tensor of same shape.
    """
    ksize = 2 * radius + 1
    kernel = torch.ones(1, 1, ksize, ksize, device=x.device, dtype=x.dtype) / (
        ksize * ksize
    )
    return F.conv2d(x, kernel, padding=radius)


def gpu_guided_filter(
    img: np.ndarray,
    config: RestoreConfig,
    device: torch.device,
) -> np.ndarray:
    """Edge-preserving smoothing via self-guided filter on GPU.

    Args:
        img: Grayscale uint8 image.
        config: Pipeline configuration.
        device: Torch device.

    Returns:
        Smoothed grayscale uint8 image.
    """
    r = config.guided_radius
    eps = config.guided_eps

    p = torch.from_numpy(img.astype(np.float32) / 255.0).to(device)
    p = p.unsqueeze(0).unsqueeze(0)

    mean_i = _box_filter_2d(p, r)
    mean_ii = _box_filter_2d(p * p, r)

    var_i = mean_ii - mean_i * mean_i
    a = var_i / (var_i + eps)
    b = mean_i - a * mean_i

    mean_a = _box_filter_2d(a, r)
    mean_b = _box_filter_2d(b, r)

    q = (mean_a * p + mean_b).squeeze().clamp(0.0, 1.0)
    return (q.cpu().numpy() * 255.0).astype(np.uint8)


# ------------------------------------------------------------------
# Stage 4: CLAHE
# ------------------------------------------------------------------
def apply_clahe(img: np.ndarray, config: RestoreConfig) -> np.ndarray:
    """Adaptive local contrast enhancement via CLAHE.

    Args:
        img: Grayscale uint8 image.
        config: Pipeline configuration.

    Returns:
        Contrast-enhanced grayscale uint8 image.
    """
    clahe = cv2.createCLAHE(
        clipLimit=config.clahe_clip_limit,
        tileGridSize=config.clahe_tile_grid,
    )
    return clahe.apply(img)


# ------------------------------------------------------------------
# Stage 5: GPU Unsharp Mask
# ------------------------------------------------------------------
def gpu_unsharp_mask(
    img: np.ndarray,
    config: RestoreConfig,
    device: torch.device,
) -> np.ndarray:
    """Unsharp masking with separable 1-D Gaussian blur on GPU.

    Args:
        img: Grayscale uint8 image.
        config: Pipeline configuration.
        device: Torch device.

    Returns:
        Sharpened grayscale uint8 image.
    """
    sigma = config.sharpen_sigma
    amount = config.sharpen_amount

    ksize = int(6 * sigma) | 1
    half = ksize // 2
    x = torch.arange(-half, half + 1, device=device, dtype=torch.float32)
    kernel_1d = torch.exp(-x * x / (2 * sigma * sigma))
    kernel_1d = kernel_1d / kernel_1d.sum()

    t = torch.from_numpy(img.astype(np.float32) / 255.0).to(device)
    t = t.unsqueeze(0).unsqueeze(0)

    k_h = kernel_1d.view(1, 1, 1, -1)
    k_v = kernel_1d.view(1, 1, -1, 1)
    blurred = F.conv2d(t, k_h, padding=(0, half))
    blurred = F.conv2d(blurred, k_v, padding=(half, 0))

    sharpened = (t + amount * (t - blurred)).squeeze().clamp(0.0, 1.0)
    return (sharpened.cpu().numpy() * 255.0).astype(np.uint8)


# ------------------------------------------------------------------
# Full Pipeline
# ------------------------------------------------------------------
def restore_ascii_art(
    img: np.ndarray,
    config: RestoreConfig | None = None,
) -> np.ndarray:
    """Run the full restoration pipeline.

    Args:
        img: Grayscale uint8 input image (ASCII art).
        config: Pipeline configuration. Uses defaults if None.

    Returns:
        Restored grayscale uint8 image.
    """
    if config is None:
        config = RestoreConfig()

    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    orig_h, orig_w = img.shape
    target_h = orig_h * config.upscale_factor
    target_w = orig_w * config.upscale_factor
    timings: dict[str, float] = {}

    # Stage 1: Grid Blur
    t0 = time.perf_counter()
    img = grid_blur(img, config, device)
    timings["1_grid_blur"] = time.perf_counter() - t0
    logger.info("Stage 1 (Grid Blur): %.1f ms", timings["1_grid_blur"] * 1000)

    # Stage 2: Bicubic Upscale
    t0 = time.perf_counter()
    img = gpu_upscale(img, target_h, target_w, device)
    timings["2_upscale"] = time.perf_counter() - t0
    logger.info("Stage 2 (Bicubic Upscale): %.1f ms", timings["2_upscale"] * 1000)

    # Stage 3: FFT Notch Filter
    t0 = time.perf_counter()
    img = apply_fft_notch_filter(img, config, device)
    timings["3_fft_notch"] = time.perf_counter() - t0
    logger.info("Stage 3 (FFT Notch Filter): %.1f ms", timings["3_fft_notch"] * 1000)

    # Stage 4: CLAHE
    t0 = time.perf_counter()
    img = apply_clahe(img, config)
    timings["4_clahe"] = time.perf_counter() - t0
    logger.info("Stage 4 (CLAHE): %.1f ms", timings["4_clahe"] * 1000)

    # Stage 5: Unsharp Mask
    t0 = time.perf_counter()
    img = gpu_unsharp_mask(img, config, device)
    timings["5_unsharp"] = time.perf_counter() - t0
    logger.info("Stage 5 (Unsharp Mask): %.1f ms", timings["5_unsharp"] * 1000)

    total = sum(timings.values())
    logger.info("Total pipeline: %.1f ms", total * 1000)

    return img
