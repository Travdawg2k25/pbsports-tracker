"""Court geometry: high school floors and cameras that only see one basket."""

from __future__ import annotations

import json

import pytest

from hoopvision.court import (
    COURT_LENGTH_FT,
    HS_COURT_LENGTH_FT,
    Calibration,
    landmarks_ft,
)

# A right-end half court seen straight on, 20 px per foot, so the maths is checkable.
PX_PER_FT = 20.0


def _image_points(calib: Calibration, names: list[str]) -> dict[str, tuple[float, float]]:
    marks = calib.landmarks
    return {n: (marks[n][0] * PX_PER_FT, marks[n][1] * PX_PER_FT) for n in names}


def test_high_school_court_moves_the_far_hoop_in():
    hs = Calibration(court_length_ft=HS_COURT_LENGTH_FT)
    assert hs.hoops["left"] == pytest.approx((5.25, 25.0))
    assert hs.hoops["right"] == pytest.approx((HS_COURT_LENGTH_FT - 5.25, 25.0))
    assert Calibration().hoops["right"] == pytest.approx((COURT_LENGTH_FT - 5.25, 25.0))


def test_lane_and_free_throw_line_sit_where_the_rules_put_them():
    marks = landmarks_ft(length=HS_COURT_LENGTH_FT)
    assert marks["right_lane_baseline_near"] == (HS_COURT_LENGTH_FT, 19.0)
    assert marks["right_lane_baseline_far"] == (HS_COURT_LENGTH_FT, 31.0)
    assert marks["right_ft_line_near"] == (HS_COURT_LENGTH_FT - 19.0, 19.0)
    assert marks["left_ft_line_far"] == (19.0, 31.0)


def test_one_basket_is_enough_to_calibrate():
    """Lane and free-throw line alone — no baseline corners, no halfcourt."""
    calib = Calibration(court_length_ft=HS_COURT_LENGTH_FT)
    names = [
        "right_lane_baseline_near",
        "right_lane_baseline_far",
        "right_ft_line_near",
        "right_ft_line_far",
    ]
    calib.image_points = _image_points(calib, names)

    for name in names:
        want = calib.landmarks[name]
        got = calib.to_court(*calib.image_points[name])
        assert got == pytest.approx(want, abs=0.1)

    rim = calib.hoops["right"]
    assert calib.hoop_distance_ft(rim, "right") == pytest.approx(0.0, abs=0.01)


def test_custom_geometry_survives_a_save(tmp_path):
    calib = Calibration(
        image_points={"right_ft_line_near": (10.0, 20.0)},
        court_length_ft=HS_COURT_LENGTH_FT,
        lane_width_ft=12.0,
        ft_line_ft=19.0,
    )
    calib.save(tmp_path / "calib.json")
    back = Calibration.load(tmp_path / "calib.json")
    assert back.court_length_ft == HS_COURT_LENGTH_FT
    assert back.hoops == calib.hoops


def test_a_calibration_from_before_lane_marking_still_loads(tmp_path):
    path = tmp_path / "old.json"
    path.write_text(
        json.dumps(
            {
                "image_points": {"left_baseline_near_sideline": [0, 0]},
                "rim_boxes": {},
                "court_length_ft": 94.0,
                "court_width_ft": 50.0,
            }
        )
    )
    calib = Calibration.load(path)
    assert calib.lane_width_ft == 12.0
    assert calib.ft_line_ft == 19.0
