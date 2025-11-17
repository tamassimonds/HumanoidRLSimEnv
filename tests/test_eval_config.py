from __future__ import annotations

import pytest

from robottraining.rl.config import EvaluationConfig


def test_evaluation_config_sets_render_mode_for_video(tmp_path):
    cfg = EvaluationConfig(policy_path="model.zip", video_path=tmp_path / "vid.mp4")
    assert cfg.render_mode == "rgb_array"
    assert cfg.video_path.suffix == ".mp4"


def test_evaluation_config_requires_policy_path():
    with pytest.raises(ValueError):
        EvaluationConfig(policy_path="")
