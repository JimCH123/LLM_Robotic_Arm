"""Roadmap 1.4: run a scripted pick-and-place demo, and optionally watch it in the MuJoCo viewer.

    python -m sim.demo            # no window: just print the results
    python -m sim.demo --viewer   # open the viewer and play the demo in real time
    python -m sim.demo --video outputs/demo.mp4   # record the demo to a video file
"""

import argparse
import time
from pathlib import Path

import imageio
import mujoco
import mujoco.viewer

from sim import skills
from sim.scene_state import get_scene_state, load_scene

# (cube, bin) pairs to sort. One cube per bin: a second cube dropped at the bin's center would land on the first.
TASKS = [("red_cube", "left_bin"), ("blue_cube", "right_bin")]

VIDEO_FPS = 25  # one frame every 8 physics steps of 5 ms
VIDEO_WIDTH, VIDEO_HEIGHT = 640, 480


def run_tasks(model, data, tasks=TASKS):
    """Pick each cube and place it in its bin. Returns a list of (cube, bin, result)."""
    results = []
    for cube, bin_name in tasks:
        result = skills.pick(model, data, cube)
        if result is None:
            result = skills.place(model, data, bin_name)
        results.append((cube, bin_name, result))
    return results


def watch_in_viewer(model, data):
    """Open the viewer and show every physics step in it, at real-time speed. Returns the viewer."""
    viewer = mujoco.viewer.launch_passive(model, data)
    time.sleep(1.0)  # give the window a moment to open before the arm starts moving
    wall_start = time.perf_counter()
    sim_start = data.time

    def show(model, data):
        if not viewer.is_running():
            return  # the window was closed: keep simulating without drawing
        viewer.sync()
        # The simulation runs faster than real time; wait so it plays at normal speed.
        ahead = (data.time - sim_start) - (time.perf_counter() - wall_start)
        if ahead > 0:
            time.sleep(ahead)

    skills.STEP_CALLBACKS.append(show)
    return viewer


def start_video(model, path):
    """Record every physics step from now on into an mp4 file. Returns the writer: call .close() at the end."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    renderer = mujoco.Renderer(model, height=VIDEO_HEIGHT, width=VIDEO_WIDTH)  # draws images without a window
    camera = mujoco.MjvCamera()  # a fixed camera looking at the cubes and both bins
    camera.lookat[:] = [0.17, 0.0, 0.03]
    camera.distance = 0.75
    camera.azimuth = 200  # degrees around the vertical axis
    camera.elevation = -35  # degrees, negative = looking down
    writer = imageio.get_writer(path, fps=VIDEO_FPS)
    steps_per_frame = round(1 / (VIDEO_FPS * model.opt.timestep))

    def capture(model, data):
        step_number = round(data.time / model.opt.timestep)
        if step_number % steps_per_frame == 0:
            renderer.update_scene(data, camera)
            writer.append_data(renderer.render())

    skills.STEP_CALLBACKS.append(capture)
    return writer


def main():
    parser = argparse.ArgumentParser(description="Scripted pick-and-place demo.")
    parser.add_argument("--viewer", action="store_true", help="watch the demo in the MuJoCo viewer")
    parser.add_argument("--video", metavar="FILE", help="record the demo to an mp4 file, e.g. outputs/demo.mp4")
    args = parser.parse_args()

    model, data = load_scene()
    viewer = watch_in_viewer(model, data) if args.viewer else None
    writer = start_video(model, args.video) if args.video else None

    for cube, bin_name, result in run_tasks(model, data):
        print(f"{cube} -> {bin_name}: {result or 'ok'}")
    for cube, info in get_scene_state(data).items():
        print(f"{cube}: position {info['position']}, in {info['in_bin'] or 'no bin'}")

    if writer is not None:
        skills.run_for(model, data, 1.0)  # one more second so the video does not stop abruptly
        writer.close()
        print(f"Video saved to {args.video}")

    if viewer is not None:
        print("Done. Close the viewer window to exit.")
        while viewer.is_running():
            time.sleep(0.1)


if __name__ == "__main__":
    main()