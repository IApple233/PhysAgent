#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class BenchmarkCase:
    case_id: str
    case_name: str
    image_path: Path
    prompt: str
    packaged_config_path: Path | None


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def time_run_id() -> str:
    return time.strftime("%m-%d-%H-%M", time.localtime(time.time()))


def parse_args() -> argparse.Namespace:
    root = repo_root()
    default_case_json = root / "examples" / "cases.json"
    default_output_root = root / "runs" / "examples" / time_run_id()

    parser = argparse.ArgumentParser(
        description=(
            "Run the packaged PhysAgent examples. Cases with a "
            "packaged config are simulated directly; otherwise run_single.py "
            "falls back to agent generation."
        )
    )
    parser.add_argument("--case-json", type=Path, default=default_case_json)
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument("--output-root", type=Path, default=default_output_root)
    parser.add_argument("--simulation-mode", choices=["short_sim", "full"], default="short_sim")
    parser.add_argument("--max-frames", type=int, default=81)
    parser.add_argument(
        "--regenerate-configs",
        action="store_true",
        help="Ignore packaged configs and regenerate each case through the agent.",
    )
    parser.add_argument("--api-key", default=os.getenv("DASHSCOPE_API_KEY") or os.getenv("API_KEY"))
    parser.add_argument("--base-url", default=os.getenv("DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/api/v1"))
    parser.add_argument("--model", default="qwen3.6-plus")
    parser.add_argument("--stop-on-failure", action="store_true")
    return parser.parse_args()


def resolve_path(value: str, base_dir: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else base_dir / path


def load_cases(case_json: Path) -> list[BenchmarkCase]:
    case_json = case_json.expanduser().resolve()
    payload = json.loads(case_json.read_text(encoding="utf-8"))
    items = payload["items"] if isinstance(payload, dict) else payload

    cases = []
    for item in items:
        image_path = resolve_path(item["image"], case_json.parent)
        packaged_config = item.get("config")
        packaged_config_path = resolve_path(packaged_config, case_json.parent) if packaged_config else None
        prompt = item.get("prompt_en") or item.get("prompt_cn") or item.get("prompt") or ""
        cases.append(
            BenchmarkCase(
                case_id=item["case_id"],
                case_name=item["case_name"],
                image_path=image_path,
                prompt=prompt.replace("\n", " "),
                packaged_config_path=packaged_config_path,
            )
        )
    return cases


def build_command(args: argparse.Namespace, case: BenchmarkCase) -> list[str]:
    use_packaged_config = (
        not args.regenerate_configs
        and case.packaged_config_path is not None
        and case.packaged_config_path.exists()
    )

    command = [
        args.python_bin,
        "run_single.py",
        "--image_path",
        str(case.image_path),
        "--case_name",
        case.case_name,
        "--prompt",
        case.prompt,
        "--output_root",
        str(args.output_root),
        "--simulation_mode",
        args.simulation_mode,
        "--max_frames",
        str(args.max_frames),
        "--model",
        args.model,
        "--base_url",
        args.base_url,
    ]

    if use_packaged_config:
        command.extend(["--mode", "simulate", "--config_path", str(case.packaged_config_path)])
        return command

    if not args.api_key:
        raise ValueError(
            f"{case.case_name} needs agent generation, but no API key was provided. "
            "Pass --api-key or set DASHSCOPE_API_KEY."
        )
    command.extend(["--mode", "agent", "--api_key", args.api_key])
    return command


def run_case(args: argparse.Namespace, case: BenchmarkCase) -> bool:
    if not case.image_path.exists():
        raise FileNotFoundError(f"Missing image: {case.image_path}")

    command = build_command(args, case)
    print()
    print(f"{case.case_id}: {case.case_name}")
    print(f"Image: {case.image_path}")
    if "--config_path" in command:
        print(f"Packaged config: {case.packaged_config_path}")
    else:
        print("Packaged config: not used")
    print(f"Prompt: {case.prompt}")
    print("Running:", " ".join(command), flush=True)
    return subprocess.run(command, cwd=repo_root()).returncode == 0


def main() -> int:
    args = parse_args()
    args.case_json = args.case_json.expanduser().resolve()
    args.output_root = args.output_root.expanduser().resolve()
    args.output_root.mkdir(parents=True, exist_ok=True)

    cases = load_cases(args.case_json)
    print(f"Case json: {args.case_json}")
    print(f"Output root: {args.output_root}")
    print(f"Simulation mode: {args.simulation_mode}")
    print(f"Regenerate configs: {args.regenerate_configs}")
    print(f"Cases: {len(cases)}")

    failed = []
    for index, case in enumerate(cases, start=1):
        print(f"[{index}/{len(cases)}]", end=" ")
        try:
            ok = run_case(args, case)
        except Exception as exc:
            print(f"Failed before execution: {exc}")
            ok = False

        if not ok:
            failed.append(case.case_name)
            if args.stop_on_failure:
                break

    if failed:
        failed_path = args.output_root / "failed_cases.txt"
        failed_path.write_text("\n".join(failed) + "\n", encoding="utf-8")
        print()
        print("Failed cases:")
        for case_name in failed:
            print(f"  {case_name}")
        return 1

    print()
    print("All cases finished.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
