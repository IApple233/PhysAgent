# PhysAgent: Reflective Agentic Physics Control for Physically Plausible Video Generation

## About

PhysAgent is a reflective agent framework for physics-grounded video generation.
It targets prompts with fine-grained object dynamics, force interactions,
complex trajectories, and temporally structured events, where one-shot
vision-language model prediction is often too brittle to configure coupled
simulation parameters correctly.

Instead of treating a physical program as a fixed prediction, PhysAgent treats it
as an executable hypothesis. The agent closes the loop between vision-language
configuration, physics simulation, reflective evaluation, and iterative
parameter refinement, progressively improving prompt alignment, physical
plausibility, and event-level control.

A single reflective agent alternates among four roles:

- **Generator** - Produces simulation configs and high-level physics-control
  programs from an input image and prompt.
- **Simulator** - Executes reconstruction and physics simulation to render the
  proposed motion outcome.
- **Evaluator** - Checks masks, reconstruction artifacts, and simulated videos
  against the prompt and expected physical behavior.
- **Reflector** - Diagnoses failures and revises the physical program for the
  next iteration.

This repository contains a cleaned single-case PhysAgent pipeline. It keeps the
multi-agent generation loop and simulation code while excluding benchmark
outputs, historical result folders, and heavy model files.

## What is included

- `run_single.py`: one entry point for both generation and simulation.
- `cases/multiagent_loop.py`: multi-agent config/handler generation and refinement.
- `cases/caption_force.py`: shared DashScope/VLM generation utilities.
- `cases/*_rule.md`: generation, evaluation, and reflection prompts.
- `case_simulation.py` and `simulation/`: reconstruction and simulation runtime.
- `vidgen/` and `wan/`: lightweight video-generation runtime code.

Large external dependencies such as SAM2, SAM3D Objects, Genesis, checkpoints,
and model weights are intentionally not copied into this folder. Install or link
them following the original project environment before running the full pipeline.

## Action API Generation Flow

PhysAgent no longer asks the VLM to write raw Genesis control code for every
case. The simulation control layer exposes reusable high-level actions under:

- `simulation/case_simulation/action_handler.py`
- `simulation/case_simulation/physics_actions.py`

Generated handlers inherit from `ActionCaseHandler` and only define object names
plus a `build_actions()` schedule. The available actions are:

- `SetVelocity(name, velocity, frame)`
- `SetAngularVelocity(name, angular_velocity, frame)`
- `ApplyForce(name, force, point=None, duration=1, frame=0)`
- `ApplyTorque(name, torque, duration=1, frame=0)`
- `ApplyAngledForce(name, magnitude, angle, duration=1, frame=0)`
- `ApplyDisturbance(name, model, amplitude, duration=1, frame=0)`
- `SetPosition(name, position, frame=0)`
- `SetOrientation(name, rotation, frame=0)`
- `FixObject(name, duration, frame=0)`

Example generated handler:

```python
from simulation.case_simulation.case_handler import register_case
from simulation.case_simulation.action_handler import ActionCaseHandler
from simulation.case_simulation.physics_actions import ApplyForce, SetVelocity


@register_case("scooter_trash_can")
class ScooterTrashCan(ActionCaseHandler):
    object_names = ["scooter", "trash_can"]

    def build_actions(self):
        return [
            SetVelocity("scooter", self.direction_to("trash_can", magnitude=3.0, horizontal=True), frame=0),
            ApplyForce("scooter", self.direction_to("trash_can", magnitude=8.0, horizontal=True), duration=10, frame=0),
        ]
```

## Installation

The environment follows the original RealWonder repository. The tested setup is
CUDA 12.1 with the `realwonder` conda environment from `default.yml`.

### 1. Create Environment

```bash
conda env create -f default.yml
conda activate realwonder
```

### 2. Prepare External Submodules

This release folder keeps only lightweight project runtime code. Clone or
symlink the external reconstruction and simulation repositories into these
exact paths:

```bash
mkdir -p submodules
git clone https://github.com/facebookresearch/sam-3d-objects.git submodules/sam_3d_objects
git clone https://github.com/facebookresearch/sam2.git submodules/sam2
git clone https://github.com/Genesis-Embodied-AI/Genesis.git submodules/Genesis
```

If you are working from the full original RealWonder checkout, symlinks are also
fine:

```bash
ln -s /path/to/RealWonder/submodules/sam_3d_objects submodules/sam_3d_objects
ln -s /path/to/RealWonder/submodules/sam2 submodules/sam2
ln -s /path/to/RealWonder/submodules/Genesis submodules/Genesis
```

### 3. Install SAM 3D Objects

```bash
cd submodules/sam_3d_objects
export PIP_EXTRA_INDEX_URL="https://pypi.ngc.nvidia.com https://download.pytorch.org/whl/cu121"
pip install -e '.[dev]'
pip install -e '.[p3d]'
export PIP_FIND_LINKS="https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.5.1_cu121.html"
pip install -e '.[inference]'
./patching/hydra
cd ../..
```

Download SAM 3D Objects checkpoints:

```bash
pip install 'huggingface-hub[cli]<1.0'
TAG=hf
hf download --repo-type model --local-dir checkpoints/${TAG}-download --max-workers 1 facebook/sam-3d-objects
mv checkpoints/${TAG}-download/checkpoints checkpoints/${TAG}
rm -rf checkpoints/${TAG}-download
```

### 4. Install SAM 2

```bash
cd submodules/sam2
pip install -e .
cd checkpoints && ./download_ckpts.sh && cd ..
cd ../..
```

### 5. Install Genesis

```bash
cd submodules/Genesis
git checkout 3aa206cd84729bc7cc14fb4007aeb95a0bead7aa
pip install -e .
cd ../..
```

### 6. Install Other Dependencies

```bash
pip install -r requirements.txt
```

### 7. Download RealWonder Video Model Checkpoints

These are needed for the final video generation path. The single-case agent and
short simulation can still be useful before running final video synthesis.

```bash
hf download ziyc/realwonder --include "Realwonder-Distilled-AR-I2V-Flow/*" --local-dir ckpts/
hf download alibaba-pai/Wan2.1-Fun-V1.1-1.3B-InP --local-dir wan_models/Wan2.1-Fun-V1.1-1.3B-InP
```

## Generate From Image And Prompt

```bash
python run_single.py \
  --image_path /path/to/input.png \
  --prompt "a ball rolls down the ramp and hits the block" \
  --api_key YOUR_DASHSCOPE_API_KEY
```

By default this creates:

- `cases/<case_name>/input.png`
- `cases/<case_name>/config.yaml`
- `simulation/case_simulation/<case_name>.py`
- `runs/<case_name>/<timestamp>/...`

Use a custom case name when needed:

```bash
python run_single.py \
  --image_path /path/to/input.png \
  --prompt "three oranges move on a conveyor belt" \
  --case_name three_oranges_demo \
  --api_key YOUR_DASHSCOPE_API_KEY
```

## Simulate From Existing Config

If you already have a config, the agent is skipped:

```bash
python run_single.py \
  --mode simulate \
  --image_path /path/to/input.png \
  --config_path /path/to/config.yaml \
  --handler_path /path/to/handler.py
```

`--handler_path` is optional if the matching handler already exists under
`simulation/case_simulation/<example_name>.py`.

Short simulation is the default. For full simulation:

```bash
python run_single.py \
  --mode simulate \
  --image_path /path/to/input.png \
  --config_path /path/to/config.yaml \
  --simulation_mode full
```

## Auto Mode

`run_single.py` defaults to `--mode auto`:

- if `--config_path` is provided, it simulates directly;
- else if `cases/<case_name>/config.yaml` already exists, it simulates directly;
- otherwise it runs the multi-agent generation loop and requires `--api_key`.

## Packaged Examples

This release includes six lightweight examples under `examples/`:

- `toy_car_acrylic_ramp_event1`
- `pumpkin_fall_pumpkin_splats_stool_intact`
- `nailong_ring_half_hangs_on_body`
- `three_oranges_conveyor_event1`
- `scooter_collide`
- `foam_block_acrylic_ramp`

Run them with the packaged configs:

```bash
python scripts/run_examples.py \
  --simulation-mode short_sim
```

Because these examples include `config.yaml`, the script skips agent generation
and runs simulation directly. To ignore the packaged configs and regenerate them
with the agent:

```bash
python scripts/run_examples.py \
  --regenerate-configs \
  --api-key YOUR_DASHSCOPE_API_KEY
```

## API Key

The API key is not stored in code. Pass it explicitly with `--api_key` whenever
the agent is used.
