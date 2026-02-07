import cv2
import numpy as np
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# ==========================================
# [설정 영역]
# ==========================================
PROJECT_DIR = Path(__file__).parent
INPUT_FILE = PROJECT_DIR / "input_image.png"
OUTPUT_FILE = PROJECT_DIR / "restored_human-made_result.png"

UPSCALE_FACTOR = 4
CONTRAST_STRENGTH = 1.2

LOOP_START_K = 15
LOOP_END_K = 3

APPLY_SHARPEN = True
SHARPEN_SIGMA = 2.5
SHARPEN_AMOUNT = 1.5


# ==========================================
# [내부 함수]
# ==========================================

def apply_neighbor_average(img: np.ndarray, k_size: int) -> np.ndarray:
    """Compute average of neighboring pixels excluding the center."""
    if k_size < 3:
        return img
    img_float = img.astype(float)
    kernel = np.ones((k_size, k_size), dtype=float)
    center = k_size // 2
    kernel[center, center] = 0

    neighbor_sum = cv2.filter2D(img_float, -1, kernel, borderType=cv2.BORDER_CONSTANT)
    ones = np.ones_like(img_float)
    neighbor_count = cv2.filter2D(ones, -1, kernel, borderType=cv2.BORDER_CONSTANT)
    neighbor_count[neighbor_count == 0] = 1

    avg_img = neighbor_sum / neighbor_count
    return np.clip(avg_img, 0, 255).astype(np.uint8)


def apply_unsharp_mask(img: np.ndarray, sigma: float = 2.0, amount: float = 1.5) -> np.ndarray:
    """Apply unsharp masking to enhance sharpness."""
    blurred = cv2.GaussianBlur(img, (0, 0), sigma)
    sharpened = float(amount + 1) * img - float(amount) * blurred
    return np.clip(sharpened, 0, 255).astype(np.uint8)


def main() -> None:
    if not INPUT_FILE.exists():
        logger.error("파일을 찾을 수 없습니다: %s", INPUT_FILE)
        return

    logger.info("이미지 로드 중: %s", INPUT_FILE)
    img = cv2.imread(str(INPUT_FILE), cv2.IMREAD_GRAYSCALE)
    if img is None:
        logger.error("이미지를 읽을 수 없습니다.")
        return

    # 1. 1차 스무딩 (문자 제거)
    logger.info("1단계: 문자 제거 스무딩 (%dx%d)...", LOOP_START_K, LOOP_START_K)
    img = apply_neighbor_average(img, k_size=LOOP_START_K)

    # 2. 해상도 확대
    logger.info("2단계: 해상도 %d배 확대...", UPSCALE_FACTOR)
    img = cv2.resize(img, None, fx=UPSCALE_FACTOR, fy=UPSCALE_FACTOR, interpolation=cv2.INTER_NEAREST)

    # 3. 명암 및 대비 보정
    logger.info("3단계: 명암 정규화 및 대비 %d%% 증가...", int((CONTRAST_STRENGTH - 1) * 100))
    img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
    img_f = img.astype(float)
    img_contrast = (img_f - 127.0) * CONTRAST_STRENGTH + 127.0
    img = np.clip(img_contrast, 0, 255).astype(np.uint8)

    # 4. 점진적 반복 정제
    logger.info("4단계: 정밀 반복 복원 (커널 %d -> %d)", LOOP_START_K, LOOP_END_K)
    loop_range = range(LOOP_START_K, LOOP_END_K - 1, -1)
    total_steps = len(loop_range)

    for i, k in enumerate(loop_range):
        if i % 5 == 0 or i == total_steps - 1:
            logger.info("   [진행률 %d%%] 스무딩 크기: %d", int((i + 1) / total_steps * 100), k)
        img = apply_neighbor_average(img, k_size=k)
        img = apply_neighbor_average(img, k_size=3)

    # 5. 선명화
    if APPLY_SHARPEN:
        logger.info("5단계: 선명화 적용...")
        img = apply_unsharp_mask(img, sigma=SHARPEN_SIGMA, amount=SHARPEN_AMOUNT)

    cv2.imwrite(str(OUTPUT_FILE), img)
    logger.info("완료! 저장됨: %s", OUTPUT_FILE)


if __name__ == "__main__":
    main()