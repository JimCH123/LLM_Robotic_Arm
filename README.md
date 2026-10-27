# LLM Robotic Arm

An LLM agent that will sort objects with a simulated SO-101 robot arm in MuJoCo from natural-language instructions (English and Chinese), plan from SOP documents with citations (RAG), remember what it did, and pass every action through a rule-based safety gate.

This repository contains the first stage: the simulation and the robot skills that the agent will call as tools.

<!-- DEMO VIDEO: on github.com, edit this file and drag outputs/demo.mp4 onto this line. -->

*The arm picks the red and blue cubes and places each in its bin. Every step is checked against the simulator's ground truth.*

## What works now

- **Simulated tabletop:** SO-101 arm ([MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie/tree/main/robotstudio_so101)), three 3 cm cubes and two bins (`sim/scene.xml`).
- **Ground-truth scene state:** `get_scene_state()` reports where each cube is and which bin it is in. This is what the agent will see, and what decides success.
- **Analytic inverse kinematics:** a closed-form 5-joint IK, first written in MATLAB and adapted to the SO-101 geometry. Link lengths and offsets are measured from the model itself, and the solution is verified by a forward-kinematics round trip (`sim/ik.py`, `sim/arm_geometry.py`).
- **Robot skills:** `move_to(x, y, z)`, `open_gripper()`, `close_gripper()`, `is_holding()`, `pick(cube)`, `place(bin)` (`sim/skills.py`).
- **Demo:** watch it live in the MuJoCo viewer or record it to mp4 (`sim/demo.py`).
- **57 tests** (pytest), including failure cases.

## Design choices

- **Skills report why they failed.** Each skill returns `None` on success, or a short reason the agent can act on. For example:
  - `"unreachable: no joint angles for (0.55, 0.0, 0.015)"`
  - `"missed: the gripper closed on nothing at (...)"`
  - `"missed the bin: blue_cube ended at [...], in no bin"`
  - `"unknown cube: purple_cube (known cubes: red_cube, green_cube, blue_cube)"`

  This is the interface an LLM needs to replan after a failure (ideas from [Code as Policies](https://arxiv.org/abs/2209.07753) and [Inner Monologue](https://arxiv.org/abs/2207.05608)).
- **Success comes from the simulator, not from the robot's own report.** `move_to` measures where the gripper really ended up. `pick` checks that the cube actually left the table, and `place` checks that it is inside the bin.
- **Failures are tested on purpose.** The tests cover targets out of reach, unknown objects, picking while already holding, and a simulated perception error. In the last case the reported cube position is wrong, and the skill must answer "missed" instead of claiming success.
- **Skills do not know who is watching.** Every physics step runs through one `step()` function with a list of callbacks. The live viewer and the video recorder are just callbacks, so the skills stay unchanged.

## Quick start (Ubuntu, Python 3.12)

Requires [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/JimCH123/LLM_Robotic_Arm.git
cd LLM_Robotic_Arm
uv sync

# SO-101 model: only this folder of MuJoCo Menagerie, into the git-ignored third_party/
git clone --filter=blob:none --sparse https://github.com/google-deepmind/mujoco_menagerie.git third_party/mujoco_menagerie
git -C third_party/mujoco_menagerie sparse-checkout set robotstudio_so101

uv run pytest                                          # all tests; the LLM test is skipped without an API key
uv run python -m sim.demo --viewer                     # watch the demo live (about 25 s)
uv run python -m sim.demo --video outputs/demo.mp4     # or record it
```

- **Viewer crashes on GNOME with Wayland:** run `export PYGLFW_LIBRARY_VARIANT=x11` first.
- **OpenGL error while recording video:** prefix the command with `MUJOCO_GL=egl`.
- **LLM test:** to run it, copy `.env.example` to `.env` and add an OpenAI API key.

## Project layout

| Path | Contents |
| --- | --- |
| `sim/scene.xml` | MuJoCo scene: SO-101 arm, three cubes, two bins |
| `sim/scene_state.py` | `load_scene()`, `get_scene_state()`: true cube positions and which bin each is in |
| `sim/arm_geometry.py` | `measure_arm_geometry()`: link lengths and zero-pose angles read from the model |
| `sim/ik.py` | `solve_ik()`: analytic inverse kinematics |
| `sim/skills.py` | Robot skills; each returns `None` or a failure reason |
| `sim/demo.py` | Scripted pick-and-place demo with `--viewer` and `--video` |
| `tests/` | pytest tests |
| `docs/PROJECT.md` | Goal, architecture, design decisions and full roadmap |

## Roadmap

- [x] **Phase 0 – Setup:** Python environment, MuJoCo, LLM API access
- [ ] **Phase 1 – Simulation and skills:** scene, ground-truth state, IK, `move_to` / `pick` / `place`, demo video (done); 10-run success rate (next)
- [ ] **Phase 2 – Agent:** skills as LangChain tools; ReAct baseline; LangGraph plan → act → check → replan loop; Chinese instruction test set
- [ ] **Phase 3 – RAG and memory:** SOP retrieval with a citation for every step; episodic memory of past actions
- [ ] **Phase 4 – Safety:** rule-based safety gate (never an LLM judgment); ask a human when unsure; poisoned-SOP and jailbreak tests
- [ ] **Phase 5 – Evaluation:** success rate, pass^k consistency, cost and latency; ablations; skills as an MCP server

Details are in [`docs/PROJECT.md`](docs/PROJECT.md).

## Tech stack

Python 3.12 · MuJoCo 3 · NumPy · pytest · uv · LangChain / LangGraph and OpenAI API (agent phases)

## Credits

- **Robot model:** [The Robot Studio SO-101](https://github.com/TheRobotStudio/SO-ARM100), from [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie) (Apache-2.0). It is downloaded during setup and not included in this repository.

## License

[MIT](LICENSE)
