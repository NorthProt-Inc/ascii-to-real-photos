이 설명서에는 **설치 방법, 전체 코드, 그리고 내 입맛대로 설정을 바꾸는 방법**이 모두 포함되어 있습니다. 복사해서 바로 사용해보세요!

---

# 🎨 ASCII Art Image Restorer (아스키 이미지 복원기)

이 프로그램은 아스키 문자로 이루어진 이미지를 분석하여 문자를 지우고, 해상도를 높이며, 부드러운 명암을 가진 고화질 이미지로 변환해줍니다.

### 🛠️ 1. 준비 사항 (설치)

이 코드를 실행하려면 **Python**이 설치되어 있어야 하며, 이미지 처리를 위한 라이브러리 2개가 필요합니다. 터미널에서 아래 명령어를 입력해 설치하세요.

```bash
pip install opencv-python numpy
```

---

### 📝 2. 전체 코드

아래 코드를 복사하여 메모장에 붙여넣고, 파일 이름을 `restore.py`로 저장하세요.

```python
import cv2
import numpy as np
import os

# ==========================================
# [⚙️ 설정 영역] 여기서 파일명과 옵션을 수정하세요
# ==========================================
INPUT_FILE = "input_image.png"       # 1. 변환할 원본 이미지 파일명 (확장자 포함)
OUTPUT_FILE = "restored_result.png"  # 2. 저장될 결과 이미지 파일명

# 화질 및 크기 설정
UPSCALE_FACTOR = 2          # 확대 배율 (2 = 2배 확대, 1 = 원본 크기 유지)
CONTRAST_STRENGTH = 1.2     # 대비 강도 (1.0 = 원본, 1.2 = 대비 20% 증가)

# 스무딩 루프 설정 (숫자가 클수록 더 많이 뭉개집니다)
LOOP_START_K = 30           # 반복 시작 커널 크기 (글자가 크면 이 값을 키우세요)
LOOP_END_K = 9              # 반복 종료 커널 크기

# 선명화(Sharpen) 설정 (블러가 너무 심하면 True로 켜세요)
APPLY_SHARPEN = True        # True: 선명하게함, False: 부드러운 상태 유지
SHARPEN_SIGMA = 3.0         # 윤곽선을 잡는 범위 (클수록 굵은 선 강조)
SHARPEN_AMOUNT = 1.5        # 선명도 강도 (1.0 ~ 2.0 추천)

# ==========================================
# [🚫 내부 함수] 이 아래는 건드리지 않으셔도 됩니다
# ==========================================

def apply_neighbor_average(img, k_size):
    """자신을 제외한 주변 픽셀 평균 구하기"""
    if k_size < 3: return img
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

def apply_unsharp_mask(img, sigma=2.0, amount=1.5):
    """언샤프 마스킹으로 선명도 높이기"""
    blurred = cv2.GaussianBlur(img, (0, 0), sigma)
    sharpened = float(amount + 1) * img - float(amount) * blurred
    return np.clip(sharpened, 0, 255).astype(np.uint8)

def main():
    if not os.path.exists(INPUT_FILE):
        print(f"❌ 오류: '{INPUT_FILE}' 파일을 찾을 수 없습니다.")
        return

    print(f"📂 이미지 로드 중: {INPUT_FILE}")
    img = cv2.imread(INPUT_FILE, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print("❌ 오류: 이미지를 읽을 수 없습니다.")
        return

    # 1. 1차 스무딩 (문자 제거)
    print(f"▶️ 1단계: 문자 제거 스무딩 ({LOOP_START_K}x{LOOP_START_K})...")
    img = apply_neighbor_average(img, k_size=LOOP_START_K)
    
    # 2. 해상도 확대
    print(f"▶️ 2단계: 해상도 {UPSCALE_FACTOR}배 확대...")
    img = cv2.resize(img, None, fx=UPSCALE_FACTOR, fy=UPSCALE_FACTOR, interpolation=cv2.INTER_NEAREST)
    
    # 3. 명암 및 대비 보정
    print(f"▶️ 3단계: 명암 정규화 및 대비 {int((CONTRAST_STRENGTH-1)*100)}% 증가...")
    img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
    img_f = img.astype(float)
    img_contrast = (img_f - 127.0) * CONTRAST_STRENGTH + 127.0
    img = np.clip(img_contrast, 0, 255).astype(np.uint8)
    
    # 4. 점진적 반복 정제 (Heavy Loop)
    print(f"▶️ 4단계: 정밀 반복 복원 시작 (커널 {LOOP_START_K} -> {LOOP_END_K})")
    loop_range = range(LOOP_START_K, LOOP_END_K - 1, -1)
    total_steps = len(loop_range)
    
    for i, k in enumerate(loop_range):
        if i % 5 == 0 or i == total_steps - 1: # 로그 너무 많이 뜨지 않게 조절
            print(f"   [진행률 {int((i+1)/total_steps*100)}%] 스무딩 크기: {k}")
        img = apply_neighbor_average(img, k_size=k) # 덩어리 깨기
        img = apply_neighbor_average(img, k_size=3) # 입자 다듬기
        
    # 5. 선명화 단계
    if APPLY_SHARPEN:
        print("▶️ 5단계: 안경 씌우기 (선명화 적용)...")
        img = apply_unsharp_mask(img, sigma=SHARPEN_SIGMA, amount=SHARPEN_AMOUNT)

    cv2.imwrite(OUTPUT_FILE, img)
    print(f"\n✅ 완료! 파일이 저장되었습니다: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
```

---

### 🚀 3. 사용 방법

1.  작업할 폴더에 위 코드를 `restore.py`로 저장합니다.
2.  변환하고 싶은 이미지를 같은 폴더에 넣습니다 (예: `input_image.