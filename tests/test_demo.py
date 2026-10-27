"""Roadmap 1.4: the step callbacks and the scripted demo."""

from pathlib import Path

import pytest

from sim import skills
from sim.demo import TASKS, VIDEO_FPS, VIDEO_HEIGHT, VIDEO_WIDTH, run_tasks, start_video
from sim.scene_state import get_scene_state, load_scene
import imageio

SO101 = Path(__file__).resolve().parent.parent / "third_party" / "mujoco_menagerie" / "robotstudio_so101" / "so101.xml"
pytestmark = pytest.mark.skipif(not SO101.exists(), reason="SO-101 model not downloaded (see README)")


def test_callback_sees_every_step(monkeypatch):
    monkeypatch.setattr(skills, "STEP_CALLBACKS", [])  # a fresh list, restored after the test
    times = []
    skills.STEP_CALLBACKS.append(lambda model, data: times.append(data.time))
    model, data = load_scene()

    skills.run_for(model, data, 0.5)

    assert len(times) == 100  # 0.5 s / 0.005 s per step
    assert times[-1] == pytest.approx(0.5)


def test_move_to_calls_the_callbacks(monkeypatch):
    monkeypatch.setattr(skills, "STEP_CALLBACKS", [])
    calls = []
    skills.STEP_CALLBACKS.append(lambda model, data: calls.append(1))
    model, data = load_scene()

    assert skills.move_to(model, data, 0.25, 0.0, 0.08) is None
    assert len(calls) == 300 + 60  # 1.5 s of motion + 0.3 s settling, 5 ms per step


def test_demo_sorts_the_cubes():
    model, data = load_scene()
    results = run_tasks(model, data)

    assert [result for _, _, result in results] == [None] * len(TASKS)
    state = get_scene_state(data)
    for cube, bin_name in TASKS:
        assert state[cube]["in_bin"] == bin_name


def test_video_is_recorded(tmp_path, monkeypatch):
    monkeypatch.setattr(skills, "STEP_CALLBACKS", [])
    model, data = load_scene()
    path = tmp_path / "clip.mp4"

    writer = start_video(model, path)
    skills.run_for(model, data, 1.0)
    writer.close()

    frames = imageio.v3.imread(path)  # all frames as one array: (frame, height, width, RGB)
    assert frames.shape == (VIDEO_FPS, VIDEO_HEIGHT, VIDEO_WIDTH, 3)  # 1 s of video
    assert frames[0].std() > 10  # a real picture, not a blank image