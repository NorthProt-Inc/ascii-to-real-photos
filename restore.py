"""ASCII Art Image Restorer v2 — GPU-accelerated pipeline.

Usage:
    python restore.py [--input FILE] [--output FILE]
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import cv2

from pipeline import RestoreConfig, restore_ascii_art

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

PROJECT_DIR = Path(__file__).parent
DEFAULT_INPUT = PROJECT_DIR / "input_image.png"
DEFAULT_OUTPUT = PROJECT_DIR / "restored_result.png"


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Restore ASCII art images (GPU)")
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="Input image path (default: %(default)s)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output image path (default: %(default)s)",
    )
    return parser.parse_args()


def main() -> None:
    """Load image, run pipeline, save result."""
    args = parse_args()

    if not args.input.exists():
        logger.error("Input file not found: %s", args.input)
        return

    logger.info("Loading image: %s", args.input)
    img = cv2.imread(str(args.input), cv2.IMREAD_GRAYSCALE)
    if img is None:
        logger.error("Failed to read image")
        return

    logger.info("Image size: %dx%d", img.shape[1], img.shape[0])

    config = RestoreConfig()
    t0 = time.perf_counter()
    result = restore_ascii_art(img, config)
    elapsed = time.perf_counter() - t0

    cv2.imwrite(str(args.output), result)
    logger.info("Saved: %s (%.1f ms total)", args.output, elapsed * 1000)


if __name__ == "__main__":
    main()
