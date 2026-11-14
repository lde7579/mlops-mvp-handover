"""파트 1(수집) 계약 테스트 — docs/data-contract.md 2장, 9장."""
from datetime import datetime, timedelta, timezone

import numpy as np
from PIL import Image

from common.schemas import Inspection, Label, validate_image_array, validate_pil_image
from parts.collector import Collector

FIXED_TIME = datetime(2026, 10, 18, 12, 0, 0, tzinfo=timezone(timedelta(hours=9)))
REQUIRED_FIELDS = {"inspection_id", "captured_at", "image_path", "true_label", "missing_parts", "scenario", "data_split"}
SPLITS = ["train", "pool", "test_normal", "test_drift"]


def make_collector(root, **kw):
    return Collector(root=root, clock=lambda: FIXED_TIME, **kw)


def test_collector_inspection_contract(tmp_path):
    collector = make_collector(tmp_path)
    samples = [
        collector.next_sample(scenario="drift" if i % 3 == 0 else "normal", data_split=SPLITS[i % 4])
        for i in range(60)
    ]

    labels = {s.inspection.true_label for s in samples}
    assert labels == {Label.OK, Label.NG}, "검사에 OK 와 NG 가 모두 나와야 계약을 확인할 수 있다"

    for s in samples:
        # 이미지 128x128 흑백: 메모리 배열과 저장된 PNG 모두
        validate_image_array(s.image)
        with Image.open(tmp_path / s.inspection.image_path) as png:
            png.load()
            validate_pil_image(png)
            assert np.array_equal(np.asarray(png), s.image)

        # 필수 필드가 모두 있고, 문서 스키마로 다시 읽어도 같다
        d = s.inspection.to_dict()
        assert REQUIRED_FIELDS <= d.keys()
        assert Inspection.from_dict(d) == s.inspection

        # 정상(OK)이면 missing_parts 가 비어 있다
        if s.inspection.true_label is Label.OK:
            assert s.inspection.missing_parts == []


def test_collector_same_seed_same_result(tmp_path):
    a = make_collector(tmp_path / "a")
    b = make_collector(tmp_path / "b")
    sa = [a.next_sample(scenario="drift" if i % 2 else "normal") for i in range(20)]
    sb = [b.next_sample(scenario="drift" if i % 2 else "normal") for i in range(20)]

    for x, y in zip(sa, sb):
        assert np.array_equal(x.image, y.image)
        assert x.inspection.to_dict() == y.inspection.to_dict()
        assert (tmp_path / "a" / x.inspection.image_path).read_bytes() == (tmp_path / "b" / y.inspection.image_path).read_bytes()

    other = make_collector(tmp_path / "c", seed=43)
    sc = [other.next_sample(scenario="drift" if i % 2 else "normal") for i in range(20)]
    assert any(not np.array_equal(x.image, z.image) for x, z in zip(sa, sc))


def test_collector_defect_rate_matches_config(tmp_path):
    collector = make_collector(tmp_path)
    # 300건이면 표준편차 약 0.023 이라 허용 오차 0.07 은 약 3 시그마. 시드가 고정이라 결과는 매번 같다.
    n = 300
    ng = sum(collector.next_sample().inspection.true_label is Label.NG for _ in range(n))
    assert abs(ng / n - collector.config.defect_rate) <= 0.07
