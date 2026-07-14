import argparse
import copy
import importlib.util
import os
import re
import shutil
import sys
import tempfile
import time
import uuid
from pathlib import Path

import dashscope
import yaml
from PIL import Image

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
FLUX_INPAINTING_PATH = REPO_ROOT / "submodules" / "flux_controlnet_inpainting"
for path in [REPO_ROOT, SCRIPT_DIR, FLUX_INPAINTING_PATH]:
    if path.exists() and str(path) not in sys.path:
        sys.path.insert(0, str(path))
DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com/api/v1"
TARGET_AREA = 1000 * 1000
MAX_FEEDBACK_IMAGES = 14
VALID_MATERIAL_TYPES = {
    "rigid",
    "pbd_cloth",
    "pbd_liquid",
    "pbd_elastic",
    "pbd_particle", 
    "mpm_sand",
    "mpm_snow",
    "mpm_elastic",
    "mpm_liquid",
    "mpm_elastic2plastic",
}

POINT_FIELDS_TO_SCALE = (
    "all_object_points",
    "support_object_points",
    "static_object_points",
    "fixed_object_points",
)


def time_run_id():
    return f"{time.time():.6f}".replace(".", "_")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate simulation configs and case handlers, then optionally refine them once from simulation feedback."
    )
    parser.add_argument("--image_path", required=True, help="Path to the input image.")
    parser.add_argument("--prompt", required=True, help="Physics action prompt.")
    parser.add_argument("--case_name", help="Override the case name derived from the image folder.")
    parser.add_argument(
        "--config_output",
        default="config2.yaml",
        help="Config filename to write inside the case folder.",
    )
    parser.add_argument(
        "--force_output",
        help="Handler filename to write inside simulation/case_simulation. Defaults to <case_name>.py",
    )
    parser.add_argument("--model", default="qwen3.5-plus", help="DashScope model name.")
    parser.add_argument(
        "--feedback_mode",
        choices=["none", "vlm", "manual"],
        default="none",
        help="Choose how the second-round revision gets feedback.",
    )
    parser.add_argument(
        "--simulation_video",
        help="Path to simulation.mp4 used for VLM review. If omitted in VLM mode, the script will ask for it after round 1.",
    )
    parser.add_argument(
        "--max_feedback_images",
        type=int,
        default=MAX_FEEDBACK_IMAGES,
        help="Maximum number of debug images to attach during VLM feedback refinement.",
    )
    parser.add_argument(
        "--manual_feedback",
        help="Manual text feedback describing what is wrong with the simulation video.",
    )
    parser.add_argument(
        "--manual_feedback_file",
        help="Path to a text file containing manual feedback.",
    )
    parser.add_argument(
        "--artifacts_dir",
        help="Directory to save per-iteration artifacts. Defaults to cases/<case>/caption_force_runs/<timestamp>",
    )
    parser.add_argument(
        "--point_scale_mode",
        choices=["relative_1000", "auto", "area", "none"],
        default="relative_1000",
        help=(
            "How to map model-point coordinates back to the original image. "
            "Default relative_1000 treats VLM points as x/1000 and y/1000."
        ),
    )
    parser.add_argument(
        "--target_area",
        type=int,
        default=TARGET_AREA,
        help="Deprecated reference area used by old point_scale_mode=area runs.",
    )
    parser.add_argument(
        "--api_key",
        required=True,
        help="DashScope API key.",
    )
    parser.add_argument(
        "--base_url",
        default=os.getenv("DASHSCOPE_BASE_URL", DEFAULT_BASE_URL),
        help="DashScope base URL.",
    )
    return parser.parse_args()


def read_system_prompt(file_path: Path) -> str:
    if not file_path.exists():
        raise FileNotFoundError(f"Missing rule file: {file_path}")
    return file_path.read_text(encoding="utf-8")


def extract_yaml(text: str) -> str:
    patterns = [r"```yaml\n(.*?)\n```", r"```yml\n(.*?)\n```", r"```\n(.*?)\n```"]
    for pattern in patterns:
        match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()
    return text.strip()


def extract_python(text: str) -> str:
    match = re.search(r"```python\n(.*?)\n```", text, re.DOTALL | re.IGNORECASE)
    return match.group(1).strip() if match else ""


def scale_points_to_original(parsed_yaml, image_path: Path, mode: str, target_area: int):
    """Convert VLM point output once before writing config YAML."""
    scaling_info = {
        "mode": mode,
        "target_area": target_area,
        "image_path": str(image_path),
        "raw_points": {
            key: copy.deepcopy(parsed_yaml.get(key))
            for key in POINT_FIELDS_TO_SCALE
            if key in parsed_yaml
        },
    }

    if mode == "none":
        scaling_info["corrected_points"] = {
            key: copy.deepcopy(parsed_yaml.get(key))
            for key in POINT_FIELDS_TO_SCALE
            if key in parsed_yaml
        }
        return parsed_yaml, scaling_info

    with Image.open(image_path) as img:
        orig_w, orig_h = img.size

    scaling_info.update(
        {
            "original_width": orig_w,
            "original_height": orig_h,
            "coordinate_convention": "relative_1000",
        }
    )

    def is_point(value):
        if not isinstance(value, (list, tuple)) or len(value) < 2:
            return False
        return all(isinstance(coord, (int, float)) for coord in value[:2])

    def normalize_point_objects(value, field_name):
        if value is None:
            return value
        if is_point(value):
            return [[list(value)]]
        if not isinstance(value, list):
            raise ValueError(f"{field_name} must be a list of objects, got {type(value).__name__}.")
        if all(is_point(item) for item in value):
            return [[list(item)] for item in value]

        normalized = []
        for obj_idx, obj_points in enumerate(value):
            if is_point(obj_points):
                normalized.append([list(obj_points)])
                continue
            if not isinstance(obj_points, list):
                raise ValueError(
                    f"{field_name}[{obj_idx}] must be a point or list of points, got {type(obj_points).__name__}."
                )
            normalized.append(obj_points)
        return normalized

    corrected_by_field = {}
    for point_field in POINT_FIELDS_TO_SCALE:
        if point_field not in parsed_yaml:
            continue
        parsed_yaml[point_field] = normalize_point_objects(parsed_yaml[point_field], point_field)
        corrected_points = []
        for obj_points in parsed_yaml[point_field]:
            new_obj = []
            for pt in obj_points:
                if len(pt) >= 2:
                    x_orig = int(round(orig_w * float(pt[0]) / 1000.0))
                    y_orig = int(round(orig_h * float(pt[1]) / 1000.0))
                    x_orig = max(0, min(x_orig, orig_w - 1))
                    y_orig = max(0, min(y_orig, orig_h - 1))
                    new_obj.append([x_orig, y_orig] + pt[2:])
                else:
                    new_obj.append(pt)
            corrected_points.append(new_obj)
        parsed_yaml[point_field] = corrected_points
        corrected_by_field[point_field] = copy.deepcopy(corrected_points)

    scaling_info["corrected_points"] = corrected_by_field
    return parsed_yaml, scaling_info


def validate_and_normalize_config(config_dict, case_name: str):
    if not isinstance(config_dict, dict):
        raise ValueError("Generated YAML is not a mapping.")

    required_keys = ["segmenter", "all_object_points", "all_object_masks_idx", "material_type"]
    missing_keys = [key for key in required_keys if key not in config_dict]
    if missing_keys:
        raise ValueError(f"Generated config is missing required keys: {missing_keys}")

    all_object_points = config_dict["all_object_points"]
    all_object_masks_idx = config_dict["all_object_masks_idx"]
    material_type = config_dict["material_type"]

    if not all_object_points:
        raise ValueError("all_object_points cannot be empty.")
    if not isinstance(all_object_masks_idx, list):
        all_object_masks_idx = [all_object_masks_idx]
    if len(all_object_masks_idx) == 1 and len(all_object_points) > 1:
        all_object_masks_idx = all_object_masks_idx * len(all_object_points)
    elif len(all_object_masks_idx) < len(all_object_points):
        all_object_masks_idx = all_object_masks_idx + [0] * (len(all_object_points) - len(all_object_masks_idx))
    elif len(all_object_masks_idx) > len(all_object_points):
        all_object_masks_idx = all_object_masks_idx[: len(all_object_points)]
    config_dict["all_object_masks_idx"] = all_object_masks_idx

    if not isinstance(material_type, list):
        material_type = [material_type]
    if len(material_type) == 1 and len(all_object_points) > 1:
        material_type = material_type * len(all_object_points)
    elif len(material_type) < len(all_object_points):
        material_type = material_type + [material_type[-1] if material_type else "rigid"] * (
            len(all_object_points) - len(material_type)
        )
    elif len(material_type) > len(all_object_points):
        material_type = material_type[: len(all_object_points)]
    config_dict["material_type"] = material_type

    invalid_materials = [item for item in material_type if item not in VALID_MATERIAL_TYPES]
    if invalid_materials:
        raise ValueError(f"Unsupported material types: {invalid_materials}")

    object_names = config_dict.get("object_names") or config_dict.get("all_object_names") or []
    if isinstance(object_names, str):
        object_names = [object_names]
    if not isinstance(object_names, list):
        object_names = []
    object_names = [_normalize_object_name(name) for name in object_names]
    if len(object_names) < len(all_object_points):
        object_names = object_names + [f"object_{idx}" for idx in range(len(object_names), len(all_object_points))]
    elif len(object_names) > len(all_object_points):
        object_names = object_names[: len(all_object_points)]
    config_dict["object_names"] = object_names

    for obj_idx, obj_points in enumerate(all_object_points):
        if not obj_points:
            raise ValueError(f"Object {obj_idx} has no prompt points.")
        for point_idx, point in enumerate(obj_points):
            if len(point) != 3:
                raise ValueError(
                    f"Object {obj_idx} point {point_idx} must contain [x, y, label], got {point}"
                )

    support_points = None
    support_key = None
    for candidate_key in ("support_object_points", "static_object_points", "fixed_object_points"):
        if candidate_key in config_dict and config_dict[candidate_key]:
            support_key = candidate_key
            support_points = config_dict[candidate_key]
            break
    if support_points:
        masks_key = {
            "support_object_points": "support_object_masks_idx",
            "static_object_points": "static_object_masks_idx",
            "fixed_object_points": "fixed_object_masks_idx",
        }[support_key]
        support_masks_idx = config_dict.get(masks_key, config_dict.get("support_object_masks_idx", [0] * len(support_points)))
        if not isinstance(support_masks_idx, list):
            support_masks_idx = [support_masks_idx]
        if len(support_masks_idx) == 1 and len(support_points) > 1:
            support_masks_idx = support_masks_idx * len(support_points)
        elif len(support_masks_idx) < len(support_points):
            support_masks_idx = support_masks_idx + [0] * (len(support_points) - len(support_masks_idx))
        elif len(support_masks_idx) > len(support_points):
            support_masks_idx = support_masks_idx[: len(support_points)]
        config_dict["support_object_points"] = support_points
        config_dict["support_object_masks_idx"] = support_masks_idx

        for obj_idx, obj_points in enumerate(support_points):
            if not obj_points:
                raise ValueError(f"Support object {obj_idx} has no prompt points.")
            for point_idx, point in enumerate(obj_points):
                if len(point) != 3:
                    raise ValueError(
                        f"Support object {obj_idx} point {point_idx} must contain [x, y, label], got {point}"
                    )

        names = config_dict.get("support_object_names", [])
        if isinstance(names, str):
            names = [names]
        if names and len(names) < len(support_points):
            names = names + [f"support_object_{idx:02d}" for idx in range(len(names), len(support_points))]
        if names:
            config_dict["support_object_names"] = names[: len(support_points)]

        # Static support should replace expensive/low-value reconstructed background
        # collision. Keep the visual background, but do not build a separate
        # background collision mesh when a dedicated support mesh exists.
        config_dict["static_support_replaces_background_collision"] = True
        config_dict["background_collision_mode"] = "static_support"
        config_dict["use_reconstructed_background_collision"] = False
        config_dict["remove_support_from_background_inpainting"] = False
        config_dict["force_regenerate_inpainted_with_support"] = False
        config_dict.setdefault("static_support_clearance", 0.03)
        config_dict.setdefault("static_support_patch_margin", 0.06)
        config_dict.setdefault("static_support_resolution_passes", 3)

    # Hard constraints for simulation/video length regardless of VLM output.
    config_dict["simulated_frames_num"] = 81
    config_dict["num_output_frames"] = 21

    config_dict["example_name"] = case_name
    config_dict["output_folder"] = f"result/{case_name}"
    config_dict["data_path"] = f"cases/{case_name}/"
    return config_dict


def _normalize_object_name(name):
    name = str(name).strip().lower()
    name = re.sub(r"[^a-z0-9]+", "_", name)
    return name.strip("_") or "object"


def normalize_handler_code(handler_code: str, case_name: str) -> str:
    if not handler_code.strip():
        raise ValueError("No python handler code was found in the model response.")

    register_pattern = r'@register_case\((["\'])(.+?)\1\)'
    if not re.search(register_pattern, handler_code):
        raise ValueError("Generated handler is missing a @register_case(...) decorator.")

    normalized_code = re.sub(
        register_pattern,
        f'@register_case("{case_name}")',
        handler_code,
        count=1,
    )
    if "ActionCaseHandler" not in normalized_code:
        raise ValueError("Generated handler must inherit from ActionCaseHandler.")
    if "def build_actions" not in normalized_code:
        raise ValueError("Generated handler must implement build_actions(self).")
    forbidden_patterns = {
        "def custom_simulation": "Do not override custom_simulation; return high-level actions from build_actions().",
        "apply_links_external_force": "Use ApplyForce instead of raw Genesis force APIs.",
        "apply_links_external_torque": "Use ApplyTorque instead of raw Genesis torque APIs.",
        "set_dofs_velocity": "Use SetVelocity or SetAngularVelocity instead of raw DOF velocity APIs.",
        "self.all_objs": "Use object names and packaged action APIs instead of direct self.all_objs indexing.",
    }
    for pattern, message in forbidden_patterns.items():
        if pattern in normalized_code:
            raise ValueError(message)
    compile(normalized_code, f"{case_name}.py", "exec")
    return normalized_code


def validate_handler_import(handler_path: Path, case_name: str):
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))

    module_name = f"_generated_case_validation_{case_name}_{handler_path.stat().st_mtime_ns}"
    spec = importlib.util.spec_from_file_location(module_name, handler_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to build module spec for {handler_path}")

    try:
        from simulation.case_simulation.case_handler import CASE_REGISTRY
    except ModuleNotFoundError as exc:
        print(
            f"Warning: skipped runtime handler import validation because dependency '{exc.name}' is unavailable in the current environment."
        )
        return

    previous_entry = CASE_REGISTRY.pop(case_name, None)
    try:
        module = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(module)
        except ModuleNotFoundError as exc:
            print(
                f"Warning: skipped runtime handler import validation because dependency '{exc.name}' is unavailable in the current environment."
            )
            return
        registered = CASE_REGISTRY.pop(case_name, None)
        if registered is None:
            raise ValueError(f"Handler imported successfully but did not register case '{case_name}'.")
    finally:
        if previous_entry is not None:
            CASE_REGISTRY[case_name] = previous_entry


def build_initial_messages(system_prompt: str, image_path: Path, case_name: str, user_prompt: str):
    return [
        {"role": "system", "content": [{"text": system_prompt}]},
        {
            "role": "user",
            "content": [
                {"image": f"file://{image_path}"},
                {
                    "text": (
                        "Please analyze the image and generate both outputs in order:\n"
                        f"1. A YAML config for the case named {case_name}.\n"
                        "2. A Python case handler implementation.\n\n"
                        f"Physics action description: {user_prompt}"
                    )
                },
            ],
        },
    ]


def build_vlm_feedback_message(image_path: Path, video_path: Path, user_prompt: str, max_feedback_images: int):
    debug_artifacts = collect_feedback_artifacts(video_path, max_images=max(0, max_feedback_images))
    debug_content = []
    if debug_artifacts:
        debug_content.append(
            {
                "text": (
                    "Additional debug artifacts are attached below. They are chosen because they reflect config-controlled "
                    "intermediate results. Use them to decide whether to revise all_object_points, all_object_masks_idx, "
                    "obj_kp_matching, obj_kp, mesh_resize_factor, remap_depth, alpha_threshold, "
                    "fg_points_render_radius, material_type, physics parameters, or the Python action handler."
                )
            }
        )
        for artifact in debug_artifacts:
            debug_content.extend(
                [
                    {"text": f"Debug artifact - {artifact['label']}: {artifact['path'].name}"},
                    {"image": f"file://{artifact['path']}"},
                ]
            )

    return {
        "role": "user",
        "content": [
            {"image": f"file://{image_path}"},
            {"video": f"file://{video_path}"},
            *debug_content,
            {
                "text": (
                    "The attached simulation video was produced from your previous YAML config and Python action handler. "
                    "Compare the actual motion and the attached debug artifacts against the original prompt, diagnose what is wrong, and then revise both outputs. "
                    "Focus especially on whether segmentation points are inside the intended moving objects, whether the chosen SAM mask is correct, "
                    "whether object keypoint matching aligns the reconstructed mesh to the image, "
                    "and whether the rendered masks reveal bad alpha thresholds, scale, material choice, force direction, or motion timing. "
                    "Keep the same case name and output only the revised YAML followed by the revised Python code.\n\n"
                    f"Original prompt: {user_prompt}"
                )
            },
        ],
    }


def unique_existing(paths):
    seen = set()
    existing = []
    for path in paths:
        if path is None:
            continue
        path = Path(path)
        if not path.exists() or not path.is_file():
            continue
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        existing.append(resolved)
    return existing


def unique_dirs(paths):
    seen = set()
    existing = []
    for path in paths:
        path = Path(path)
        if not path.exists() or not path.is_dir():
            continue
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        existing.append(resolved)
    return existing


def sample_paths(paths, max_count: int):
    paths = sorted(unique_existing(paths), key=lambda path: path.name)
    if len(paths) <= max_count:
        return paths
    if max_count <= 1:
        return paths[:max_count]
    indices = [round(i * (len(paths) - 1) / (max_count - 1)) for i in range(max_count)]
    return [paths[idx] for idx in indices]


def load_yaml_if_exists(path: Path):
    if not path.exists():
        return {}
    try:
        loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return loaded if isinstance(loaded, dict) else {}


def collect_sam2_debug_artifacts(run_dir: Path, config: dict, remaining_slots: int):
    if remaining_slots <= 0:
        return []

    debug_dirs = unique_dirs(
        [
            Path.cwd() / "debug" / "sam2",
            REPO_ROOT / "debug" / "sam2",
            run_dir / "debug" / "sam2",
        ]
    )

    video_mtime = None
    simulation_video = run_dir / "final_sim" / "simulation.mp4"
    if simulation_video.exists():
        video_mtime = simulation_video.stat().st_mtime

    selected = []
    object_count = len(config.get("all_object_points", []) or [])
    mask_indices = config.get("all_object_masks_idx", []) or []
    support_points = config.get("support_object_points", []) or []
    support_mask_indices = config.get("support_object_masks_idx", []) or []

    for debug_dir in debug_dirs:
        candidates = []
        for object_idx in range(object_count):
            candidates.append((f"SAM2 input points object {object_idx}", debug_dir / f"input_points_{object_idx:02d}.png"))
            if object_idx < len(mask_indices):
                try:
                    mask_idx = int(mask_indices[object_idx])
                except (TypeError, ValueError):
                    mask_idx = 0
                candidates.append(
                    (
                        f"SAM2 selected mask object {object_idx}",
                        debug_dir / f"object_{object_idx:02d}_masks_{mask_idx:02d}.png",
                    )
                )

        for support_idx in range(len(support_points)):
            candidates.append((f"SAM2 input points support {support_idx}", debug_dir / f"support_input_points_{support_idx:02d}.png"))
            try:
                mask_idx = int(support_mask_indices[support_idx])
            except (IndexError, TypeError, ValueError):
                mask_idx = 0
            candidates.append(
                (
                    f"SAM2 selected mask support {support_idx}",
                    debug_dir / f"support_object_{support_idx:02d}_masks_{mask_idx:02d}.png",
                )
            )

        if not candidates:
            candidates = [("SAM2 input points", path) for path in sorted(debug_dir.glob("input_points_*.png"))]
            candidates.extend(("SAM2 mask candidate", path) for path in sorted(debug_dir.glob("object_*_masks_*.png")))

        for label, path in candidates:
            if not path.exists() or not path.is_file():
                continue
            if video_mtime is not None and path.stat().st_mtime < video_mtime - 6 * 60 * 60:
                continue
            selected.append({"label": label, "path": path.resolve()})
            if len(selected) >= remaining_slots:
                return selected

    return selected[:remaining_slots]


def collect_feedback_artifacts(video_path: Path, max_images: int = MAX_FEEDBACK_IMAGES):
    video_path = Path(video_path).expanduser().resolve()
    final_sim_dir = video_path.parent
    run_dir = final_sim_dir.parent
    render_dir = run_dir / "render"
    config = load_yaml_if_exists(final_sim_dir / "config.yaml")

    artifacts = []

    sam_budget = min(max(4, max_images // 3), max_images // 2, 8)
    sam2_artifacts = collect_sam2_debug_artifacts(run_dir, config, remaining_slots=sam_budget)
    artifacts.extend(sam2_artifacts)

    priority_groups = [
        (
            "selected object mask",
            sample_paths(sorted(run_dir.glob("refined_mask_*.png")) or sorted(run_dir.glob("mask_*.png")), max_count=3),
        ),
        (
            "selected fixed support mask",
            sample_paths(sorted(run_dir.glob("support_mask_*.png")), max_count=2),
        ),
        (
            "object keypoint matching",
            sample_paths(sorted(render_dir.glob("gt_kps_*.png")), max_count=3)
            + sample_paths(sorted(render_dir.glob("mesh_kps_*.png")), max_count=3),
        ),
        (
            "mesh proxy render before keypoint match",
            sample_paths(sorted(render_dir.glob("mesh_init_render_proxy_color_*.png")), max_count=3),
        ),
        (
            "rendered foreground mask",
            sample_paths((render_dir / "masks").glob("points_mask_*.png"), max_count=2),
        ),
        (
            "resized input after crop",
            [final_sim_dir / "resized_input_image.png"],
        ),
    ]

    for label, paths in priority_groups:
        if len(artifacts) >= max_images:
            break
        for path in unique_existing(paths):
            artifacts.append({"label": label, "path": path})
            if len(artifacts) >= max_images:
                break

    deduped = []
    seen = set()
    for artifact in artifacts:
        path = artifact["path"].resolve()
        if path in seen:
            continue
        seen.add(path)
        deduped.append({"label": artifact["label"], "path": path})
        if len(deduped) >= max_images:
            break

    return deduped


def build_manual_feedback_message(image_path: Path, user_prompt: str, manual_feedback: str):
    return {
        "role": "user",
        "content": [
            {"image": f"file://{image_path}"},
            {
                "text": (
                    "A human reviewed the simulation result from your previous YAML config and Python action handler and reported these issues:\n"
                    f"{manual_feedback}\n\n"
                    "Revise both outputs so the next simulation better matches the prompt. "
                    "If the problem may come from incorrect object prompt points after image resizing, you may also adjust all_object_points. "
                    "Keep the same case name and output only the revised YAML followed by the revised Python code.\n\n"
                    f"Original prompt: {user_prompt}"
                )
            },
        ],
    }


def call_model(model: str, messages):
    staged_messages = copy.deepcopy(messages)
    staged_dir = Path(tempfile.mkdtemp(prefix="caption_force_media_"))

    try:
        for message in staged_messages:
            for content_item in message.get("content", []):
                for media_key in ("image", "video"):
                    media_url = content_item.get(media_key)
                    if not isinstance(media_url, str) or not media_url.startswith("file://"):
                        continue

                    src_path = Path(media_url[7:]).expanduser().resolve()
                    if not src_path.exists():
                        raise FileNotFoundError(f"Media file not found: {src_path}")

                    suffix = src_path.suffix or ".bin"
                    staged_path = staged_dir / f"{src_path.stem}_{uuid.uuid4().hex}{suffix}"
                    shutil.copy2(src_path, staged_path)
                    content_item[media_key] = f"file://{staged_path}"

        response = dashscope.MultiModalConversation.call(model=model, messages=staged_messages)
        if response.status_code != 200:
            raise RuntimeError(f"DashScope call failed: {response.code} {response.message}")
        return response.output.choices[0].message.content[0]["text"]
    finally:
        shutil.rmtree(staged_dir, ignore_errors=True)


def build_iteration_paths(artifacts_dir: Path, iteration_idx: int):
    iteration_dir = artifacts_dir / f"iter_{iteration_idx}"
    iteration_dir.mkdir(parents=True, exist_ok=True)
    return {
        "dir": iteration_dir,
        "raw_response": iteration_dir / "raw_response.txt",
        "raw_yaml": iteration_dir / "raw_extracted_yaml.txt",
        "raw_python": iteration_dir / "raw_extracted_python.py",
        "config": iteration_dir / "config.yaml",
        "handler": iteration_dir / "handler.py",
        "scaling": iteration_dir / "point_scaling.yaml",
        "feedback": iteration_dir / "feedback.txt",
        "meta": iteration_dir / "meta.yaml",
    }


def save_iteration_outputs(
    iteration_idx: int,
    artifacts_dir: Path,
    raw_response: str,
    raw_yaml_text: str,
    raw_python_text: str,
    config_dict,
    handler_code: str,
    scaling_info,
    feedback_mode: str,
    feedback_text: str,
    standard_config_path: Path,
    standard_handler_path: Path,
):
    paths = build_iteration_paths(artifacts_dir, iteration_idx)
    paths["raw_response"].write_text(raw_response, encoding="utf-8")
    paths["raw_yaml"].write_text(raw_yaml_text.strip() + "\n", encoding="utf-8")
    paths["raw_python"].write_text(raw_python_text.strip() + "\n", encoding="utf-8")
    paths["config"].write_text(
        yaml.safe_dump(config_dict, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    paths["handler"].write_text(handler_code + "\n", encoding="utf-8")
    paths["scaling"].write_text(
        yaml.safe_dump(scaling_info, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    paths["meta"].write_text(
        yaml.safe_dump(
            {
                "iteration": iteration_idx,
                "feedback_mode": feedback_mode,
                "standard_config_path": str(standard_config_path),
                "standard_handler_path": str(standard_handler_path),
            },
            sort_keys=False,
            allow_unicode=True,
        ),
        encoding="utf-8",
    )
    if feedback_text:
        paths["feedback"].write_text(feedback_text.strip() + "\n", encoding="utf-8")

    standard_config_path.write_text(
        yaml.safe_dump(config_dict, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    standard_handler_path.write_text(handler_code + "\n", encoding="utf-8")
    return paths


def prompt_user(message: str, default: str = "") -> str:
    prompt = message if not default else f"{message} [{default}] "
    value = input(prompt).strip()
    return value or default


def guess_latest_simulation_video(case_name: str):
    result_root = REPO_ROOT / "result" / case_name
    if not result_root.exists():
        return None
    candidates = sorted(
        result_root.glob("*/final_sim/simulation.mp4"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


def resolve_simulation_video(case_name: str, provided_path: str):
    if provided_path:
        path = Path(provided_path).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"Simulation video not found: {path}")
        return path

    guessed = guess_latest_simulation_video(case_name)
    if not sys.stdin.isatty():
        raise FileNotFoundError(
            "feedback_mode=vlm requires --simulation_video when no interactive terminal is available."
        )
    print("\nRound 1 outputs are saved. Run the simulation now if needed, then provide the simulation.mp4 path for VLM review.")
    entered = prompt_user("Simulation video path:", str(guessed) if guessed else "")
    path = Path(entered).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Simulation video not found: {path}")
    return path


def collect_manual_feedback(args):
    if args.manual_feedback_file:
        feedback = Path(args.manual_feedback_file).expanduser().read_text(encoding="utf-8").strip()
        if feedback:
            return feedback
    if args.manual_feedback:
        return args.manual_feedback.strip()
    if not sys.stdin.isatty():
        piped = sys.stdin.read().strip()
        if piped:
            return piped
        raise ValueError("feedback_mode=manual requires --manual_feedback, --manual_feedback_file, or interactive input.")

    print("\nRound 1 outputs are saved. Inspect the simulation video yourself, then paste feedback below.")
    print("End your input with a single line containing END")
    lines = []
    while True:
        line = input()
        if line.strip() == "END":
            break
        lines.append(line)
    feedback = "\n".join(lines).strip()
    if not feedback:
        raise ValueError("Manual feedback cannot be empty.")
    return feedback


def run_generation_round(
    messages,
    model: str,
    case_name: str,
    image_path: Path,
    point_scale_mode: str,
    target_area: int,
):
    raw_response = call_model(model, messages)
    raw_yaml_text = extract_yaml(raw_response)
    raw_python_text = extract_python(raw_response)

    config_dict = yaml.safe_load(raw_yaml_text)
    config_dict, scaling_info = scale_points_to_original(
        config_dict,
        image_path,
        mode=point_scale_mode,
        target_area=target_area,
    )
    config_dict = validate_and_normalize_config(config_dict, case_name)
    handler_code = normalize_handler_code(raw_python_text, case_name)
    return raw_response, raw_yaml_text, raw_python_text, config_dict, handler_code, scaling_info


def ensure_api(args):
    if not args.api_key:
        raise EnvironmentError("DashScope API key is required. Pass --api_key.")
    dashscope.base_http_api_url = args.base_url
    dashscope.api_key = args.api_key


def main():
    args = parse_args()
    ensure_api(args)

    image_path = Path(args.image_path).expanduser().resolve()
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    case_name = args.case_name or image_path.parent.name
    output_dir = image_path.parent
    standard_config_path = output_dir / args.config_output
    force_dir = REPO_ROOT / "simulation" / "case_simulation"
    force_dir.mkdir(parents=True, exist_ok=True)
    standard_handler_path = force_dir / (args.force_output or f"{case_name}.py")

    timestamp = time_run_id()
    artifacts_dir = (
        Path(args.artifacts_dir).expanduser().resolve()
        if args.artifacts_dir
        else output_dir / "caption_force_runs" / timestamp
    )
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    system_prompt = "\n\n".join(
        [
            read_system_prompt(SCRIPT_DIR / "caption_rule.md"),
            read_system_prompt(SCRIPT_DIR / "force_rule.md"),
        ]
    )

    messages = build_initial_messages(system_prompt, image_path, case_name, args.prompt)

    print("Calling DashScope to generate round 0 config and handler...")
    round0 = run_generation_round(
        messages,
        model=args.model,
        case_name=case_name,
        image_path=image_path,
        point_scale_mode=args.point_scale_mode,
        target_area=args.target_area,
    )
    (
        raw_response,
        raw_yaml_text,
        raw_python_text,
        config_dict,
        handler_code,
        scaling_info,
    ) = round0

    save_iteration_outputs(
        iteration_idx=0,
        artifacts_dir=artifacts_dir,
        raw_response=raw_response,
        raw_yaml_text=raw_yaml_text,
        raw_python_text=raw_python_text,
        config_dict=config_dict,
        handler_code=handler_code,
        scaling_info=scaling_info,
        feedback_mode="initial",
        feedback_text="",
        standard_config_path=standard_config_path,
        standard_handler_path=standard_handler_path,
    )
    validate_handler_import(standard_handler_path, case_name)

    print(f"Round 0 config saved to: {standard_config_path}")
    print(f"Round 0 handler saved to: {standard_handler_path}")
    print(f"Per-iteration artifacts saved under: {artifacts_dir}")

    if args.feedback_mode == "none":
        print("No feedback round requested. Done.")
        return

    messages = messages + [{"role": "assistant", "content": [{"text": raw_response}]}]

    if args.feedback_mode == "vlm":
        video_path = resolve_simulation_video(case_name, args.simulation_video)
        feedback_text = f"VLM reviewed video: {video_path}"
        feedback_message = build_vlm_feedback_message(
            image_path,
            video_path,
            args.prompt,
            max_feedback_images=args.max_feedback_images,
        )
    else:
        manual_feedback = collect_manual_feedback(args)
        feedback_text = manual_feedback
        feedback_message = build_manual_feedback_message(image_path, args.prompt, manual_feedback)

    messages = messages + [feedback_message]

    print(f"Calling DashScope to generate round 1 revision via feedback_mode={args.feedback_mode}...")
    round1 = run_generation_round(
        messages,
        model=args.model,
        case_name=case_name,
        image_path=image_path,
        point_scale_mode=args.point_scale_mode,
        target_area=args.target_area,
    )
    (
        raw_response_1,
        raw_yaml_text_1,
        raw_python_text_1,
        config_dict_1,
        handler_code_1,
        scaling_info_1,
    ) = round1

    save_iteration_outputs(
        iteration_idx=1,
        artifacts_dir=artifacts_dir,
        raw_response=raw_response_1,
        raw_yaml_text=raw_yaml_text_1,
        raw_python_text=raw_python_text_1,
        config_dict=config_dict_1,
        handler_code=handler_code_1,
        scaling_info=scaling_info_1,
        feedback_mode=args.feedback_mode,
        feedback_text=feedback_text,
        standard_config_path=standard_config_path,
        standard_handler_path=standard_handler_path,
    )
    validate_handler_import(standard_handler_path, case_name)

    print(f"Round 1 revised config saved to: {standard_config_path}")
    print(f"Round 1 revised handler saved to: {standard_handler_path}")
    print(f"All artifacts are under: {artifacts_dir}")


if __name__ == "__main__":
    main()
