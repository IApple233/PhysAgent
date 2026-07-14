import argparse
import os
import random
import shutil
import sys
import time
from pathlib import Path

os.environ.setdefault("SETUPTOOLS_USE_DISTUTILS", "stdlib")

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from omegaconf import DictConfig, ListConfig, OmegaConf
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent
SUBMODULE_PATHS = [
    REPO_ROOT,
]
for path in SUBMODULE_PATHS:
    if path.exists() and str(path) not in sys.path:
        sys.path.insert(0, str(path))


def time_run_id():
    return f"{time.time():.6f}".replace(".", "_")


def set_seed(seed: int, deterministic: bool = False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.use_deterministic_algorithms(True)


def process_simulated_results(input_image, raw_video_frames, points_masks, mesh_masks, crop_start=176):
    from simulation.utils import resize_and_crop_pil

    input_image = resize_and_crop_pil(input_image, crop_start)
    raw_video_frames = [resize_and_crop_pil(frame, crop_start) for frame in raw_video_frames]
    points_masks = preprocess_masks_downsample(points_masks)
    mesh_masks = preprocess_masks_downsample(mesh_masks)

    return input_image, raw_video_frames, points_masks, mesh_masks


def preprocess_masks_downsample(masks):
    num_masks = len(masks)
    masks = torch.stack(masks, dim=0).squeeze(-1)
    resized_masks = F.interpolate(masks.unsqueeze(1).float(), size=(832, 832), mode='bilinear', align_corners=False)
    crop_height = 480
    crop_width = 832
    start_y = (832 - crop_height) // 2
    cropped_masks = resized_masks[:, :, start_y:start_y + crop_height, :]
    masks_downsampled = F.interpolate(cropped_masks.float(), size=(60, 104), mode='bilinear', align_corners=False).squeeze(1)
    time_averaged_masks = []
    for i in range(0, num_masks, 4):
        time_averaged_masks.append(masks_downsampled[i: i + 4, :, :].mean(dim=0, keepdim=True))
    masks_downsampled = torch.cat(time_averaged_masks, dim=0)
    masks_downsampled = masks_downsampled > 0.5
    return masks_downsampled


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, required=True, help="Path to the config file")
    parser.add_argument(
        "--run_mode",
        choices=["full", "short_sim"],
        default="full",
        help="Use short_sim for multi-agent motion debugging without final noise/video packaging.",
    )
    parser.add_argument(
        "--max_frames",
        type=int,
        default=40,
        help="Number of rendered frames for short_sim mode. Defaults to 40.",
    )
    parser.add_argument(
        "--skip_noise_warp",
        action="store_true",
        help="Skip NoiseWarper processing in full mode.",
    )
    parser.add_argument(
        "--artifact_manifest",
        type=str,
        help="Optional path to write a YAML artifact manifest.",
    )
    parser.add_argument(
        "--output_folder",
        type=str,
        help="Optional exact output folder. If omitted, config['output_folder']/<time.time id> is used.",
    )
    return parser.parse_args()


def find_reconstruction_outputs(config_path):
    config_path = Path(config_path).expanduser().resolve()
    candidates = []
    if config_path.parent.name == "reconstruction_outputs":
        candidates.append(config_path.parent)
    candidates.append(config_path.parent / "reconstruction_outputs")
    if config_path.parent.name == "configs":
        candidates.append(config_path.parent.parent)

    for candidate in candidates:
        if (candidate / "inpainted_image.png").exists():
            return candidate
    return None


def attach_precomputed_reconstruction_artifacts(config, config_path):
    """Reuse multi-agent reconstruction outputs when simulating from an iter config."""
    if config.get("precomputed_inpainted_image_path"):
        return

    reconstruction_outputs = find_reconstruction_outputs(config_path)
    if reconstruction_outputs is None:
        return

    visual_path = reconstruction_outputs / "inpainted_image.png"
    collision_path = reconstruction_outputs / "background_collision_inpainted_image.png"
    if visual_path.exists():
        config["precomputed_inpainted_image_path"] = str(visual_path)
        config["precomputed_inpainted_image_source"] = "multiagent_reconstruction_outputs"
        print(f"Reusing precomputed inpainted image: {visual_path}")
    if collision_path.exists():
        config["precomputed_background_collision_inpainted_image_path"] = str(collision_path)
        print(f"Reusing precomputed background collision inpainted image: {collision_path}")


def prepare_output(config, config_path, requested_output_folder=None):
    if requested_output_folder:
        output_folder = str(Path(requested_output_folder).expanduser().resolve())
    else:
        timestamp = time_run_id()
        output_folder = os.path.join(config['output_folder'], timestamp)
    os.makedirs(output_folder, exist_ok=True)
    config['output_folder'] = output_folder
    debug = config.get('debug', False)

    if debug:
        debug_config_save_path = os.path.join(config['output_folder'], "config.yaml")
        OmegaConf.save(config, debug_config_save_path)

    configs_folder = os.path.join(config['output_folder'], "configs")
    os.makedirs(configs_folder, exist_ok=True)
    OmegaConf.save(config, os.path.join(configs_folder, "config.yaml"))
    shutil.copy2(config_path, os.path.join(configs_folder, Path(config_path).name))

    handler_path = Path("simulation") / "case_simulation" / f"{config['example_name']}.py"
    if handler_path.exists():
        shutil.copy2(handler_path, os.path.join(configs_folder, handler_path.name))
    else:
        print(f"Warning: case handler file not found, skipped saving to configs: {handler_path}")

    return output_folder, debug


def write_manifest(manifest_path, payload):
    if not manifest_path:
        return
    manifest_path = Path(manifest_path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(yaml.safe_dump(to_yaml_safe(payload), sort_keys=False, allow_unicode=True), encoding="utf-8")


def to_yaml_safe(value):
    if isinstance(value, (DictConfig, ListConfig)):
        return to_yaml_safe(OmegaConf.to_container(value, resolve=True))
    if isinstance(value, dict):
        return {str(key): to_yaml_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_yaml_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return to_yaml_safe(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    return value


def run_short_sim(args, config, input_image):
    from simulation.genesis_simulator import DiffSim

    genesis_simulator = DiffSim(config)
    OmegaConf.save(config, Path(config['output_folder']) / "config.yaml")
    raw_video_frames, points_masks, mesh_masks = genesis_simulator.simulation_pc_render(
        max_render_frames=args.max_frames,
    )

    short_sim_folder = Path(config['output_folder']) / "short_sim"
    short_sim_folder.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(config, short_sim_folder / "config.yaml")

    input_image, video_frames, points_masks_downsampled, mesh_masks_downsampled = process_simulated_results(
        input_image,
        raw_video_frames,
        points_masks,
        mesh_masks,
        crop_start=config['crop_start'],
    )

    frame_folder = short_sim_folder / "frames"
    frame_folder.mkdir(parents=True, exist_ok=True)
    for i, frame in enumerate(video_frames):
        frame.save(frame_folder / f"frame_{i:04d}.png")

    video_path = short_sim_folder / "simulation.mp4"
    from simulation.utils import save_video_from_pil

    save_video_from_pil(video_frames, video_path, fps=10)
    if not video_path.exists() or video_path.stat().st_size == 0:
        raise RuntimeError(f"short_sim simulation.mp4 was not generated: {video_path}")
    input_image.save(short_sim_folder / "resized_input_image.png")
    torch.save(points_masks_downsampled, short_sim_folder / "points_masks_downsampled.pt")
    torch.save(mesh_masks_downsampled, short_sim_folder / "mesh_masks_downsampled.pt")

    output_folder = Path(config['output_folder'])
    background_collision_mesh_paths = [
        output_folder / "background_collision_mesh_pt3d.obj",
        output_folder / "background_collision_mesh_gs.obj",
    ]
    reconstruction_artifacts = {
        "masks": sorted(str(path) for path in output_folder.glob("mask_*.png"))
        + sorted(str(path) for path in output_folder.glob("refined_mask_*.png")),
        "background_collision_mesh": [
            str(path) for path in background_collision_mesh_paths if path.exists()
        ],
        "background_collision": {
            key: config.get(key)
            for key in [
                "background_collision_mode",
                "background_collision_mesh_path_gs",
                "background_collision_mesh_bounds_gs",
                "background_collision_mesh_source",
                "background_plane_normal_pt3d",
                "background_plane_point_pt3d",
                "background_plane_normal_gs",
                "background_plane_point_gs",
                "gravity_direction",
            ]
            if key in config
        },
        "static_collision_objects": config.get("static_collision_objects", []),
    }

    manifest = {
        "run_mode": "short_sim",
        "max_frames": args.max_frames,
        "output_folder": str(config['output_folder']),
        "short_sim_folder": str(short_sim_folder),
        "all_object_masks_idx": config.get("all_object_masks_idx", []),
        "all_object_masks_idx_semantics": (
            "Each all_object_masks_idx[i] is a local proposal index within SAM2's "
            "multimask outputs for object i. Repeated values across objects are "
            "valid and do not imply shared masks."
        ),
        "artifacts": {
            "video": str(video_path),
            "frames": str(frame_folder),
            "runtime_config": str(short_sim_folder / "config.yaml"),
            "initial_scene_layout_obj": str(Path(config['output_folder']) / "initial_scene_layout.obj"),
            "resized_input_image": str(short_sim_folder / "resized_input_image.png"),
            "points_masks_downsampled": str(short_sim_folder / "points_masks_downsampled.pt"),
            "mesh_masks_downsampled": str(short_sim_folder / "mesh_masks_downsampled.pt"),
            "genesis_render_video": str(Path(config['output_folder']) / "simulation" / "render_gs.mp4"),
            "genesis_frames": str(Path(config['output_folder']) / "simulation" / "gs_frames"),
            "render_frames": str(Path(config['output_folder']) / "simulation" / "render" / "frames"),
            "reconstruction": reconstruction_artifacts,
        },
    }
    write_manifest(args.artifact_manifest, manifest)
    print(f"Short simulation saved to: {short_sim_folder}")


def run_full(args, config, input_image, debug):
    from simulation.genesis_simulator import DiffSim

    genesis_simulator = DiffSim(config)
    OmegaConf.save(config, Path(config['output_folder']) / "config.yaml")
    raw_video_frames, points_masks, mesh_masks = genesis_simulator.simulation_pc_render()

    input_image, video_frames, points_masks_downsampled, mesh_masks_downsampled = process_simulated_results(
        input_image,
        raw_video_frames,
        points_masks,
        mesh_masks,
        crop_start=config['crop_start'],
    )

    final_sim_folder = os.path.join(config['output_folder'], "final_sim")
    os.makedirs(final_sim_folder, exist_ok=True)

    config_save_path = os.path.join(final_sim_folder, "config.yaml")
    OmegaConf.save(config, config_save_path)

    optical_flows = np.array(genesis_simulator.svr.optical_flow)[..., :2]
    optical_flows = np.transpose(optical_flows, (0, 3, 1, 2))

    if debug:
        np.save(os.path.join(final_sim_folder, "flows.npy"), optical_flows)

    frame_folder = os.path.join(final_sim_folder, "frames")
    os.makedirs(frame_folder, exist_ok=True)
    for i, frame in enumerate(video_frames):
        frame_path = os.path.join(frame_folder, f"frame_{i:04d}.png")
        frame.save(frame_path)

    if debug:
        from simulation.utils import visualize_optical_flow_advanced

        visualize_optical_flow_advanced(
            frame_folder,
            os.path.join(final_sim_folder, "flows.npy"),
            os.path.join(final_sim_folder, "optical_flow_viz"),
            arrow_density=30,
        )

    if not args.skip_noise_warp:
        from simulation.image23D.noise_warp.make_warped_noise import NoiseWarper

        noise_warper = NoiseWarper()
        noise_warper.process(video_frames, final_sim_folder, crop_start=config['crop_start'], input_flow=False, debug=debug)

    torch.save(points_masks_downsampled, os.path.join(final_sim_folder, "points_masks_downsampled.pt"))
    torch.save(mesh_masks_downsampled, os.path.join(final_sim_folder, "mesh_masks_downsampled.pt"))

    video_path = os.path.join(final_sim_folder, "simulation.mp4")
    from simulation.utils import save_video_from_pil

    save_video_from_pil(video_frames, video_path, fps=10)

    input_image_path = os.path.join(final_sim_folder, "resized_input_image.png")
    input_image.save(input_image_path)

    prompt_txt_path = os.path.join(final_sim_folder, "prompt.txt")
    with open(prompt_txt_path, "w") as f:
        f.write(config['vgen_prompt'])

    manifest = {
        "run_mode": "full",
        "output_folder": str(config['output_folder']),
        "final_sim_folder": str(final_sim_folder),
        "artifacts": {
            "video": video_path,
            "frames": frame_folder,
            "initial_scene_layout_obj": str(Path(config['output_folder']) / "initial_scene_layout.obj"),
            "resized_input_image": input_image_path,
            "points_masks_downsampled": os.path.join(final_sim_folder, "points_masks_downsampled.pt"),
            "mesh_masks_downsampled": os.path.join(final_sim_folder, "mesh_masks_downsampled.pt"),
            "flows": os.path.join(final_sim_folder, "flows.npy"),
            "genesis_render_video": str(Path(config['output_folder']) / "simulation" / "render_gs.mp4"),
            "background_collision_mesh": [
                str(path)
                for path in [
                    Path(config['output_folder']) / "background_collision_mesh_pt3d.obj",
                    Path(config['output_folder']) / "background_collision_mesh_gs.obj",
                ]
                if path.exists()
            ],
            "static_collision_objects": config.get("static_collision_objects", []),
        },
    }
    write_manifest(args.artifact_manifest, manifest)


def main():
    args = parse_args()
    config = OmegaConf.load(args.config_path)
    attach_precomputed_reconstruction_artifacts(config, args.config_path)
    output_folder, debug = prepare_output(config, args.config_path, args.output_folder)

    device = torch.device("cuda")
    set_seed(config['seed'])
    torch.set_grad_enabled(False)
    input_image = Image.open(os.path.join(config['data_path'], 'input.png')).convert('RGB')

    if args.run_mode == "short_sim":
        config['debug_short_sim_frames'] = args.max_frames
        run_short_sim(args, config, input_image)
    else:
        run_full(args, config, input_image, debug)


if __name__ == "__main__":
    main()
