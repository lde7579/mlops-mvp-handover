"""이미지 시뮬레이터: 카메라가 수직 위에서 내려다본 볼트 6개(2행x3열)의 체결 유무, 128x128 흑백.

링 조명(카메라 둘레 원형 조명)을 가정해 그림자를 한쪽으로 두지 않고 대칭으로 그린다.
- 배경: 가공된 금속판 (약한 가로 결 + 큰 얼룩)
- 체결(정상): 플랜지 + 육각 머리(방향은 볼트마다 무작위) + 둥근 밝은 윗면
- 미체결(불량): 모따기 고리 + 안쪽으로 어두워지는 나사산 동심원 + 거의 검은 바닥
4배 크기로 그린 뒤 128x128로 줄인다. 드리프트 3종(dark, noise, shift)은 같은 그림에 변화를 더한다.
"""
import math

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from common.schemas import IMAGE_SIZE

SIZE = IMAGE_SIZE
SCALE = 4  # 4배로 그린 뒤 줄인다
CENTERS = [(24, 36), (64, 36), (104, 36), (24, 92), (64, 92), (104, 92)]  # (x, y), 볼트 1~6번
BOLT_COUNT = len(CENTERS)
GAIN = [1.00, 0.96, 1.04, 0.98, 1.02, 0.95]  # 볼트마다 밝기가 조금씩 다름
PLATE_BRIGHT = 95.0
THREAD_RINGS = [(8.0, 70), (7.4, 40), (6.8, 62), (6.2, 34), (5.6, 52), (5.0, 26), (4.4, 40), (3.8, 18)]  # (반지름, 밝기)
BASE_NOISE = 6.0  # 모든 이미지에 들어가는 약한 노이즈
JITTER = 1.5  # 미세한 위치 흔들림(항상)
DRIFT_TYPES = ("dark", "noise", "shift")


def _plate(rng: np.random.Generator) -> Image.Image:
    """금속판: 약한 가로 결 + 큰 얼룩."""
    big = SIZE * SCALE
    base = np.full((big, big), PLATE_BRIGHT)
    grain = rng.normal(0, 3, (big, 1))  # 줄마다 밝기가 조금씩 달라 가로 결처럼 보임
    blot = Image.fromarray(rng.normal(128, 40, (SIZE, SIZE)).clip(0, 255).astype(np.uint8))
    blot = blot.resize((big, big), Image.Resampling.BICUBIC).filter(ImageFilter.GaussianBlur(12))
    base = base + grain + (np.asarray(blot, dtype=float) - 128) * 0.10
    return Image.fromarray(base.clip(0, 255).astype(np.uint8))


def _circle(d, c, r, fill=None, outline=None, width=1.0):
    x, y = c[0] * SCALE, c[1] * SCALE
    d.ellipse([x - r * SCALE, y - r * SCALE, x + r * SCALE, y + r * SCALE],
              fill=fill, outline=outline, width=max(1, int(width * SCALE)))


def _hexagon(c, r, rot):
    return [((c[0] + r * math.cos(math.radians(60 * i + rot))) * SCALE,
             (c[1] + r * math.sin(math.radians(60 * i + rot))) * SCALE) for i in range(6)]


def _draw_bolt(d, c, g, rot):
    """체결: 플랜지 + 육각 머리 + 둥근 윗면 (위에서 본 모습)."""
    _circle(d, c, 17.5, fill=int(70 * g))   # 플랜지와 판 사이의 어두운 틈
    _circle(d, c, 16.8, fill=int(150 * g))  # 플랜지 윗면
    _circle(d, c, 14.5, fill=int(120 * g))  # 머리 쪽으로 꺾이는 경사
    # 육각 머리: 옆면은 위에서 거의 안 보이고, 꼭짓점 쪽 모따기면이 비스듬히 반사돼 어둡다
    d.polygon(_hexagon(c, 13.5, rot), fill=int(95 * g), outline=int(40 * g), width=SCALE)
    _circle(d, c, 10.6, fill=int(185 * g))                   # 윗면 모따기
    _circle(d, c, 10.6, outline=int(215 * g), width=0.7)     # 밝은 반사 고리
    _circle(d, c, 9.4, fill=int(178 * g))                    # 평평한 윗면: 링 조명이면 고르게 밝다
    for k in range(3):  # 머리 각인 흉내: 아주 옅은 점 3개
        a = math.radians(rot + 120 * k)
        _circle(d, (c[0] + 4.5 * math.cos(a), c[1] + 4.5 * math.sin(a)), 0.9, fill=int(150 * g))


def _draw_hole(d, c, g):
    """미체결: 모따기 고리 + 나사산(안쪽으로 어두워짐) + 거의 검은 바닥."""
    _circle(d, c, 9.5, fill=int(130 * g))
    _circle(d, c, 9.5, outline=int(60 * g), width=0.6)
    for r, v in THREAD_RINGS:
        _circle(d, c, r, fill=int(v * g))
    _circle(d, c, 3.0, fill=5)


def make_image(rng: np.random.Generator, unfastened, drift_type=None, strength=0.0) -> np.ndarray:
    """이미지 1장(uint8, 128x128). unfastened: 미체결 볼트 번호 목록(1~6)."""
    gx, gy = 0.0, 0.0  # 전체 이동
    bright = rng.uniform(0.85, 1.15)  # 밝기 변화(항상)
    sigma = BASE_NOISE
    bolt_shift = JITTER

    if drift_type == "dark":
        bright *= 1.0 - 0.9 * strength
    elif drift_type == "noise":
        sigma = BASE_NOISE + 100.0 * strength
    elif drift_type == "shift":
        bolt_shift = 2.0 + 25.0 * strength
        gx, gy = rng.uniform(-1, 1, 2) * bolt_shift
    elif drift_type is not None:
        raise ValueError(f"알 수 없는 drift_type: {drift_type}")

    img = _plate(rng)
    d = ImageDraw.Draw(img)
    for i, (cx, cy) in enumerate(CENTERS):
        dx, dy = rng.uniform(-bolt_shift, bolt_shift, 2)
        c = (cx + gx + dx, cy + gy + dy)
        rot = rng.uniform(0, 60)  # 체결 각도는 볼트마다 다름 (미체결이어도 같은 순서로 뽑는다)
        if (i + 1) in unfastened:
            _draw_hole(d, c, GAIN[i])
        else:
            _draw_bolt(d, c, GAIN[i], rot)

    img = img.resize((SIZE, SIZE), Image.Resampling.LANCZOS)
    arr = np.asarray(img, dtype=np.float32) * bright
    arr += rng.normal(0.0, sigma, arr.shape)
    return np.clip(arr, 0, 255).astype(np.uint8)
