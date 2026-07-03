#!/usr/bin/env python3
import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parent


def time_run_id() -> str:
    return time.strftime("%m-%d-%H-%M", time.localtime(time.time()))


def sanitize_case_name(value: str) -> str:
    cleaned = []
    for char in value.strip():
        if char.isalnum() or char in {"-", "_"}:
            cleaned.append(char)
        elif char in {" ", ".", "/"}:
            cleaned.append("_")
    case_name = "".join(cleaned).strip("_")
    return case_name or "realwonder_case"


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run one RealWonder case. With no config, generate the config/handler "
            "from an image and prompt. With an existing config, skip the agent and "
            "run simulation directly."
        )
    )
    parser.add_argument("--image_path", required=True, help="Input image path.")
    parser.add_argument(
        "--prompt",
        default="",
        help="Physics prompt. Required when generating via the multi-agent pipeline.",
    )
    parser.add_argument(
        "--mode",
        choices=["auto", "agent", "simulate"],
        default="auto",
        help="auto simulates when a config is available, otherwise runs the agent.",
    )
    parser.add_argument("--case_name", help="Case name. Defaults to the image filename stem.")
    parser.add_argument(
        "--config_path",
        help="Existing YAML config. If provided in auto mode, the agent is skipped.",
    )
    parser.add_argument(
        "--handler_path",
        help="Optional existing Python handler to copy into simulation/case_simulation.",
    )
    parser.add_argument(
        "--config_name",
        default="config.yaml",
        help="Config filename under cases/<case_name>/ when the agent generates one.",
    )
    parser.add_argument(
        "--api_key",
        help="DashScope API key. Required only when the agent is used.",
    )
    parser.add_argument(
        "--base_url",
        default="https://dashscope.aliyuncs.com/api/v1",
        help="DashScope API base URL.",
    )
    parser.add_argument("--model", default="qwen3.6-plus", help="DashScope model name.")
    parser.add_argument(
        "--output_root",
        default=str(REPO_ROOT / "runs"),
        help="Root directory for generated artifacts and simulation outputs.",
    )
    parser.add_argument(
        "--simulation_mode",
        choices=["short_sim", "full"],
        default="short_sim",
        help="Simulation mode used when running an existing config.",
    )
    parser.add_argument("--max_frames", type=int, default=81, help="Frames for short simulation.")
    parser.add_argument(
        "--run_full_after_agent",
        action="store_true",
        help="After agent generation, also run full simulation from the final config.",
    )
    parser.add_argument("--max_mask_rounds", type=int, default=5)
    parser.add_argument("--max_reconstruction_rounds", type=int, default=1)
    parser.add_argument("--max_motion_rounds", type=int, default=1)
    parser.add_argument("--mask_accept_score", type=float, default=0.80)
    parser.add_argument("--continue_on_mask_failure", action="store_true")
    parser.add_argument("--enable_sequential_object_inpainting", action="store_true")
    return parser.parse_args()


def load_yaml(path: Path):
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    return loaded if isinstance(loaded, dict) else {}


def copy_input_image(image_path: Path, case_name: str) -> Path:
    case_dir = REPO_ROOT / "cases" / case_name
    case_dir.mkdir(parents=True, exist_ok=True)
    runtime_image = case_dir / "input.png"
    shutil.copy2(image_path, runtime_image)
    return runtime_image


def maybe_install_handler(handler_path: str, config_path: Path):
    if not handler_path:
        return
    config = load_yaml(config_path)
    case_name = config.get("example_name") or config_path.parent.name
    target = REPO_ROOT / "simulation" / "case_simulation" / f"{case_name}.py"
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(Path(handler_path).expanduser().resolve(), target)
    print(f"Installed handler: {target}")


def sync_image_to_config_data_path(image_path: Path, config_path: Path):
    config = load_yaml(config_path)
    data_path = config.get("data_path")
    if not data_path:
        return
    target_dir = Path(data_path)
    if not target_dir.is_absolute():
        target_dir = REPO_ROOT / target_dir
    target_dir.mkdir(parents=True, exist_ok=True)
    target_image = target_dir / "input.png"
    if target_image.resolve() != image_path.resolve():
        shutil.copy2(image_path, target_image)
        print(f"Synced input image to config data_path: {target_image}")


def run(cmd):
    print("Running:", " ".join(str(part) for part in cmd), flush=True)
    subprocess.run([str(part) for part in cmd], cwd=REPO_ROOT, check=True)


def run_agent(args, runtime_image: Path, case_name: str, run_root: Path) -> Path:
    if not args.prompt.strip():
        raise ValueError("--prompt is required when running the agent.")
    if not args.api_key:
        raise ValueError("--api_key is required when running the agent.")

    cmd = [
        sys.executable,
        "cases/multiagent_loop.py",
        "--image_path",
        runtime_image,
        "--case_name",
        case_name,
        "--prompt",
        args.prompt,
        "--config_output",
        args.config_name,
        "--artifacts_dir",
        run_root,
        "--short_sim_frames",
        args.max_frames,
        "--max_mask_rounds",
        args.max_mask_rounds,
        "--max_reconstruction_rounds",
        args.max_reconstruction_rounds,
        "--max_motion_rounds",
        args.max_motion_rounds,
        "--mask_accept_score",
        args.mask_accept_score,
        "--model",
        args.model,
        "--api_key",
        args.api_key,
        "--base_url",
        args.base_url,
    ]
    if args.continue_on_mask_failure:
        cmd.append("--continue_on_mask_failure")
    if args.enable_sequential_object_inpainting:
        cmd.append("--enable_sequential_object_inpainting")
    run(cmd)
    return REPO_ROOT / "cases" / case_name / args.config_name


def run_simulation(config_path: Path, run_root: Path, simulation_mode: str, max_frames: int):
    sim_dir = run_root / "simulation_outputs"
    manifest_path = sim_dir / "artifact_manifest.yaml"
    cmd = [
        sys.executable,
        "case_simulation.py",
        "--config_path",
        config_path,
        "--output_folder",
        sim_dir,
        "--run_mode",
        simulation_mode,
        "--artifact_manifest",
        manifest_path,
    ]
    if simulation_mode == "short_sim":
        cmd.extend(["--max_frames", max_frames])
    run(cmd)


def resolve_mode(args, default_config: Path) -> str:
    if args.mode != "auto":
        return args.mode
    if args.config_path or default_config.exists():
        return "simulate"
    return "agent"


def main():
    args = parse_args()
    image_path = Path(args.image_path).expanduser().resolve()
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    case_name = sanitize_case_name(args.case_name or image_path.stem)
    runtime_image = copy_input_image(image_path, case_name)
    run_root = Path(args.output_root).expanduser().resolve() / case_name / time_run_id()
    run_root.mkdir(parents=True, exist_ok=True)

    default_config = REPO_ROOT / "cases" / case_name / args.config_name
    mode = resolve_mode(args, default_config)

    if mode == "agent":
        config_path = run_agent(args, runtime_image, case_name, run_root)
        if args.run_full_after_agent:
            run_simulation(config_path, run_root / "full_after_agent", "full", args.max_frames)
        print(f"Agent artifacts: {run_root}")
        print(f"Generated config: {config_path}")
        return

    config_path = Path(args.config_path).expanduser().resolve() if args.config_path else default_config
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    maybe_install_handler(args.handler_path, config_path)
    sync_image_to_config_data_path(runtime_image, config_path)
    run_simulation(config_path, run_root, args.simulation_mode, args.max_frames)
    print(f"Simulation artifacts: {run_root}")


if __name__ == "__main__":
    main()
