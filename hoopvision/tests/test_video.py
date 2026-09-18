"""Frame access."""

from __future__ import annotations

import numpy as np

from hoopvision import synthetic
from hoopvision.video import FrameSeeker, crop, iter_frames, probe


def test_probe_reports_the_clip_geometry(tmp_path):
    synthetic.generate(tmp_path, write_tracks=False)
    meta = probe(tmp_path / "synthetic.mp4")
    assert (meta.width, meta.height) == (synthetic.W, synthetic.H)
    assert meta.frame_count > 0
    assert meta.duration_s > 0


def test_iter_frames_honours_stride_and_end(tmp_path):
    synthetic.generate(tmp_path, write_tracks=False)
    idxs = [i for i, _ in iter_frames(tmp_path / "synthetic.mp4", stride=5, end=49)]
    assert idxs == list(range(0, 50, 5))


def test_seeker_repeats_backtracks_and_skips(tmp_path):
    """Several tracks ask for the same frame, then the next one, then jump back."""
    synthetic.generate(tmp_path, write_tracks=False)
    with FrameSeeker(tmp_path / "synthetic.mp4") as seeker:
        first = seeker.get(30)
        assert first is not None
        assert np.array_equal(seeker.get(30), first)  # repeat
        assert seeker.get(31) is not None  # forward
        assert np.array_equal(seeker.get(30), first)  # backwards seek
        assert seeker.get(200) is not None  # long jump
        assert seeker.get(10**7) is None  # past the end


def test_crop_clamps_to_the_frame():
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    assert crop(img, (-20, -20, 40, 40)).shape == (40, 40, 3)
    assert crop(img, (50, 50, 10, 10)).size == 0
