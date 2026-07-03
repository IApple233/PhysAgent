import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
FLUX_INPAINTING_PATH = REPO_ROOT / "submodules" / "flux_controlnet_inpainting"
for path in [REPO_ROOT, SCRIPT_DIR, FLUX_INPAINTING_PATH]:
    if path.exists() and str(path) not in sys.path:
        sys.path.insert(0, str(path))

from caption_force import (
    build_initial_messages,
    call_model,
    ensure_api,
    extract_python,
    extract_yaml,
    read_system_prompt,
    run_generation_round,
    save_iteration_outputs,
    validate_handler_import,
)


def time_run_id():
    return time.strftime("%m-%d-%H-%M", time.localtime(time.time()))


def iter_dir(artifacts_dir: Path, iteration_idx: int) -> Path:
    return artifacts_dir / f"iter_{iteration_idx}"


def parse_args():
    parser = argparse.ArgumentParser(description="Run the RealWonder three-agent generation/evaluation/reflection loop.")
    parser.add_argument("--image_path", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--case_name")
    parser.add_argument("--config_output", default="config2.yaml")
    parser.add_argument("--force_output")
    parser.add_argument("--model", default="qwen3.6-plus")
    parser.add_argument("--artifacts_dir")
    parser.add_argument("--point_scale_mode", default="relative_1000")
    parser.add_argument("--target_area", type=int, default=1000 * 1000)
    parser.add_argument("--max_mask_rounds", type=int, default=5)
    parser.add_argument("--max_reconstruction_rounds", type=int, default=1)
    parser.add_argument("--max_motion_rounds", type=int, default=1)
    parser.add_argument("--short_sim_frames", type=int, default=81)
    parser.add_argument("--mask_accept_score", type=float, default=0.80)
    parser.add_argument("--reconstruction_accept_score", type=float, default=0.75)
    parser.add_argument("--short_sim_accept_score", type=float, default=0.70)
    parser.add_argument(
        "--continue_on_mask_failure",
        action="store_true",
        help="Continue to SAM3D reconstruction even if the mask-only gate is rejected.",
    )
    parser.add_argument(
        "--enable_sequential_object_inpainting",
        action="store_true",
        help=(
            "Allow the legacy occlusion-aware flow that segments object 0, "
            "inpaints it away, then segments later objects on the updated image. "
            "By default multi-agent runs use one-shot segmentation plus union inpainting."
        ),
    )
    parser.add_argument(
        "--forbid_static_support",
        action="store_true",
        help=(
            "Force dynamic-object-only reconstruction: remove support/static/fixed "
            "support points and use a single infinite plane for collision."
        ),
    )
    parser.add_argument(
        "--force_top_down_camera",
        action="store_true",
        help="Force a manual fully top-down Genesis/SVR camera pose.",
    )
    parser.add_argument("--api_key", required=True, help="DashScope API key.")
    parser.add_argument("--base_url", default=os.getenv("DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/api/v1"))
    return parser.parse_args()


def load_yaml(path):
    path = Path(path)
    if not path.exists():
        return {}
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    return loaded if isinstance(loaded, dict) else {}


def extract_json(text):
    match = re.search(r"```json\s*(.*?)\s*```", text, re.DOTALL | re.IGNORECASE)
    candidate = match.group(1) if match else text
    start = candidate.find("{")
    end = candidate.rfind("}")
    if start >= 0 and end > start:
        candidate = candidate[start:end + 1]
    return json.loads(candidate)


def media_item(path):
    path = Path(path)
    if not path.exists() or not path.is_file():
        return None
    key = "video" if path.suffix.lower() in {".mp4", ".mov", ".avi"} else "image"
    return {key: f"file://{path.resolve()}"}


def add_media_items(content, paths, max_items=12):
    count = 0
    for path in paths:
        item = media_item(path)
        if item is None:
            continue
        content.append({"text": f"Artifact: {Path(path).name}"})
        content.append(item)
        count += 1
        if count >= max_items:
            break


def extend_unique(paths, new_paths):
    seen = {str(path) for path in paths}
    for path in new_paths or []:
        path_key = str(path)
        if path_key in seen:
            continue
        paths.append(path)
        seen.add(path_key)


def reconstruction_media_from_manifest(manifest):
    artifacts = manifest.get("artifacts", {})
    paths = []

    for object_artifact in artifacts.get("object_artifacts", []) or []:
        extend_unique(paths, object_artifact.get("sam2_input_points_debug", []))
        extend_unique(paths, object_artifact.get("sam2_all_mask_candidates_debug", []))
        extend_unique(paths, object_artifact.get("sam2_selected_mask_debug", []))
        extend_unique(paths, object_artifact.get("saved_object_mask", []))
        extend_unique(paths, object_artifact.get("keypoint_debug", []))

    for support_artifact in artifacts.get("support_object_artifacts", []) or []:
        extend_unique(paths, support_artifact.get("sam2_input_points_debug", []))
        extend_unique(paths, support_artifact.get("sam2_all_mask_candidates_debug", []))
        extend_unique(paths, support_artifact.get("sam2_selected_mask_debug", []))
        extend_unique(paths, support_artifact.get("saved_object_mask", []))
        extend_unique(paths, support_artifact.get("keypoint_debug", []))

    for key in [
        "all_object_points_overlay",
        "union_inpainting_mask",
        "support_inpainting_mask",
        "masks",
        "support_masks",
        "inpainting",
        "depth",
        "keypoints_by_object",
        "mesh_proxy_by_object",
    ]:
        extend_unique(paths, artifacts.get(key, []))
    return paths


def short_sim_media_from_manifest(manifest):
    artifacts = manifest.get("artifacts", {})
    # Evaluate the final SVR-rendered short simulation. The Genesis render is a
    # debug artifact and can be tiny/incomplete enough for VLM upload to reject.
    paths = [
        artifacts.get("video"),
        artifacts.get("resized_input_image"),
    ]
    frames_dir = artifacts.get("frames")
    if frames_dir and Path(frames_dir).exists():
        frames = sorted(Path(frames_dir).glob("*.png"))
        if frames:
            paths.extend([frames[0], frames[len(frames) // 2], frames[-1]])
    reconstruction = artifacts.get("reconstruction", {}) or {}
    paths.extend((reconstruction.get("masks", []) or [])[:3])
    return [path for path in paths if path]


def call_evaluator(stage, args, config_path, handler_path, manifest_path, output_path):
    system_prompt = read_system_prompt(SCRIPT_DIR / "eval_rule.md")
    manifest = load_yaml(manifest_path)
    config = load_yaml(config_path)
    if stage == "mask":
        stage_guidance = (
            "This is the pre-SAM3D mask gate. Evaluate only SAM2 prompt points, "
            "candidate masks, selected masks, and saved masks. Do not penalize "
            "missing mesh, depth, inpainting, or keypoint artifacts at this stage. "
            "Primary editable targets are all_object_points first, then all_object_masks_idx."
        )
    elif stage == "reconstruction":
        stage_guidance = (
            "This stage runs after the mask gate. SAM3D internals are fixed; if masks "
            "are acceptable but alignment is poor, focus on obj_kp_matching/obj_kp "
            "and gt_kps vs mesh_kps evidence."
        )
    else:
        stage_guidance = ""

    content = [
        media_item(args.image_path),
        {
            "text": (
                f"Stage: {stage}\n"
                f"User prompt: {args.prompt}\n"
                f"Config path: {config_path}\n"
                f"Handler path: {handler_path}\n"
                f"Evaluation thresholds: mask={args.mask_accept_score}, "
                f"reconstruction={args.reconstruction_accept_score}, "
                f"short_simulation={args.short_sim_accept_score}\n"
                f"{stage_guidance}\n"
                "Important config semantics: all_object_masks_idx[i] is a local "
                "SAM2 proposal index for object i. Repeated values are valid and "
                "must not be treated as shared masks without per-object mask evidence.\n"
                "Important scene semantics: support_object_points, when present, "
                "describe fixed collision-only scene support (for example a table). "
                "They are not dynamic objects and do not need material_type entries. "
                "Do not add or require support_object_points for ordinary object-object "
                "motion on a broad flat ground/floor/lane/table surface when an infinite "
                "plane under the reconstructed dynamic objects is sufficient. Reserve "
                "static support reconstruction for ramps, raised table edges, shelves, "
                "walls, fences, ledges, trays, bowls, or other finite/localized colliders "
                "that clearly affect the motion. "
                "When support objects are present, static_support_replaces_background_collision "
                "should normally be true and background_collision_mode should use static_support, "
                "so the pipeline relies on the reconstructed support mesh. "
                "For books, plates, paper, cards, and other slab-like objects, do not "
                "reject reconstruction solely because the visible mesh is thin; use "
                "object_quality_summary, mask coverage, mesh extents, and keypoint "
                "alignment before deciding that geometry is unusable.\n"
                "Config YAML:\n"
                f"{yaml.safe_dump(config, sort_keys=False, allow_unicode=True)}\n"
                "Artifact manifest:\n"
                f"{yaml.safe_dump(manifest, sort_keys=False, allow_unicode=True)}"
            )
        },
    ]
    content = [item for item in content if item is not None]
    if stage == "mask":
        add_media_items(content, reconstruction_media_from_manifest(manifest), max_items=24)
    elif stage == "reconstruction":
        add_media_items(content, reconstruction_media_from_manifest(manifest), max_items=18)
    else:
        add_media_items(content, short_sim_media_from_manifest(manifest), max_items=8)

    messages = [
        {"role": "system", "content": [{"text": system_prompt}]},
        {"role": "user", "content": content},
    ]
    raw = call_model(args.model, messages)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.with_suffix(".raw.txt").write_text(raw, encoding="utf-8")
    verdict = extract_json(raw)
    output_path.write_text(json.dumps(verdict, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return verdict


def call_reflection(stage, args, config_path, handler_path, manifest_path, evaluator_path, output_path):
    system_prompt = read_system_prompt(SCRIPT_DIR / "reflection_rule.md")
    manifest = load_yaml(manifest_path)
    config = load_yaml(config_path)
    evaluator = json.loads(Path(evaluator_path).read_text(encoding="utf-8"))
    handler_code = Path(handler_path).read_text(encoding="utf-8") if Path(handler_path).exists() else ""
    if stage == "mask":
        stage_guidance = (
            "This rejection happened before SAM3D. Diagnose segmentation only. "
            "Return the previous coordinates and observed mask effect for each "
            "object. Do not prescribe exact replacement coordinates; the generator "
            "will choose new coordinates from the original image and point overlay. "
            "Prefer moving or adding all_object_points deeper inside the intended "
            "object body; change all_object_masks_idx only when the saved candidate "
            "evidence shows a better local proposal. In patch_intent, describe "
            "questions/problems to check and the effect needed next; do not give "
            "explicit edit commands or exact numeric replacements."
        )
    elif stage == "reconstruction":
        stage_guidance = (
            "This rejection happened after masks were accepted. Do not propose "
            "changing SAM3D internals. If gt_kps points are near an edge or narrow "
            "visible strip, revise obj_kp/obj_kp_matching toward stable interior regions. "
            "In patch_intent, describe questions/problems to check and the effect "
            "needed next; do not give explicit edit commands or exact numeric replacements."
        )
    else:
        stage_guidance = (
            "In patch_intent, describe questions/problems to check and the effect "
            "needed next; do not give explicit edit commands or exact numeric replacements."
        )

    content = [
        media_item(args.image_path),
        {
            "text": (
                f"Stage: {stage}\n"
                f"User prompt: {args.prompt}\n"
                f"{stage_guidance}\n"
                "Evaluator verdict:\n"
                f"{json.dumps(evaluator, indent=2, ensure_ascii=False)}\n"
                "Important reflection semantics: support_object_points are a valid "
                "repair target when the visible support is a table, shelf, chair seat, "
                "counter, or other non-floor/non-wall collider whose depth is unreliable. "
                "Do not diagnose physically thin book-like meshes as failed solely from "
                "thin proxy appearance if mask, extents, and keypoints are coherent.\n"
                "Config YAML:\n"
                f"{yaml.safe_dump(config, sort_keys=False, allow_unicode=True)}\n"
                "Handler Python:\n"
                f"```python\n{handler_code}\n```\n"
                "Artifact manifest:\n"
                f"{yaml.safe_dump(manifest, sort_keys=False, allow_unicode=True)}"
            )
        },
    ]
    content = [item for item in content if item is not None]
    if stage == "mask":
        add_media_items(content, reconstruction_media_from_manifest(manifest), max_items=18)
    elif stage == "reconstruction":
        add_media_items(content, reconstruction_media_from_manifest(manifest), max_items=12)
    else:
        add_media_items(content, short_sim_media_from_manifest(manifest), max_items=8)

    messages = [
        {"role": "system", "content": [{"text": system_prompt}]},
        {"role": "user", "content": content},
    ]
    raw = call_model(args.model, messages)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.with_suffix(".raw.txt").write_text(raw, encoding="utf-8")
    reflection = extract_json(raw)
    output_path.write_text(json.dumps(reflection, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return reflection


def run_subprocess(cmd):
    print("Running:", " ".join(str(part) for part in cmd), flush=True)
    env = os.environ.copy()
    env.setdefault("SETUPTOOLS_USE_DISTUTILS", "stdlib")
    pythonpath_parts = [
        str(REPO_ROOT),
        str(SCRIPT_DIR),
        str(FLUX_INPAINTING_PATH),
    ]
    if env.get("PYTHONPATH"):
        pythonpath_parts.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(dict.fromkeys(pythonpath_parts))
    subprocess.run([str(part) for part in cmd], cwd=REPO_ROOT, check=True, env=env)


FRICTION_KEYS = (
    "rigid_friction",
    "plane_friction",
    "rigid_coup_friction",
    "friction",
    "support_friction",
    "static_support_friction",
)


def clamp_genesis_friction_values(config, min_friction=1e-2, max_friction=5.0):
    if not isinstance(config, dict):
        return config
    for key in FRICTION_KEYS:
        if key not in config:
            continue
        try:
            value = float(config[key])
        except (TypeError, ValueError):
            continue
        config[key] = float(min(max(value, min_friction), max_friction))

    for spec in config.get("static_collision_objects", []) or []:
        if not isinstance(spec, dict):
            continue
        for key in FRICTION_KEYS:
            if key not in spec:
                continue
            try:
                value = float(spec[key])
            except (TypeError, ValueError):
                continue
            spec[key] = float(min(max(value, min_friction), max_friction))
    return config


MOTION_REFLECTION_CONFIG_ALLOWLIST = {
    "vgen_prompt",
    "dt",
    "substeps",
    "simulated_frames_num",
    "frame_steps",
    "rigid_friction",
    "plane_friction",
    "rigid_coup_friction",
    "rigid_coup_softness",
    "static_support_clearance",
    "static_support_patch_margin",
    "static_support_resolution_passes",
    "static_support_local_height_quantile",
    "static_support_overlap_margin",
}


def preserve_accepted_config_for_motion(previous_config, generated_config):
    """Keep accepted mask/reconstruction/camera config stable during motion retries."""
    if not isinstance(previous_config, dict) or not isinstance(generated_config, dict):
        return previous_config
    merged = dict(previous_config)
    for key in MOTION_REFLECTION_CONFIG_ALLOWLIST:
        if key in generated_config:
            merged[key] = generated_config[key]
    merged["motion_reflection_config_preserved"] = True
    return merged


def apply_pipeline_config_policy(args, config):
    if not isinstance(config, dict):
        return config
    if not args.enable_sequential_object_inpainting:
        config["sequential_object_inpainting"] = False
        config["object_occlusion_aware_inpainting"] = False

    if getattr(args, "forbid_static_support", False):
        for key in (
            "support_object_points",
            "support_object_masks_idx",
            "support_object_names",
            "static_object_points",
            "static_object_masks_idx",
            "static_object_names",
            "fixed_object_points",
            "fixed_object_masks_idx",
            "fixed_object_names",
            "static_collision_objects",
        ):
            config.pop(key, None)
        config["static_support_replaces_background_collision"] = False
        config["remove_support_from_background_inpainting"] = False
        config["force_regenerate_inpainted_with_support"] = False

    has_support = bool(config.get("support_object_points") or [])
    config["background_collision_mode"] = "static_support" if has_support else "plane"
    config.setdefault("background_plane_position_mode", "object_support")
    config.setdefault("background_plane_offset", 0.0)

    if has_support:
        config["static_support_replaces_background_collision"] = True
        config["remove_support_from_background_inpainting"] = False
        config["force_regenerate_inpainted_with_support"] = False

    for key in list(config):
        if key.startswith("background_collision_mesh"):
            config.pop(key, None)
    for key in (
        "use_reconstructed_background_collision",
        "background_collision_roi",
        "background_collision_mesh_roi",
        "background_collision_margin_px",
        "background_collision_sets_gravity",
        "background_collision_uses_support_removed_inpainting",
        "skip_background_collision_reconstruction",
    ):
        config.pop(key, None)
    if getattr(args, "force_top_down_camera", False):
        config["genesis_camera_mode"] = "manual"
        config["sync_svr_render_camera_to_genesis"] = True
        config["svr_render_camera_mode"] = "genesis"
        config["align_reconstruction_to_ground"] = True
        config["preserve_input_camera_after_world_alignment"] = True
        config["camera_pose_space"] = "direct_gs"
        config["manual_camera_space"] = "direct_gs"
        config["sim_camera_pos_gs"] = [0.0, 0.0, 3.0]
        config["sim_camera_lookat_gs"] = [0.0, 0.0, 0.0]
        config["sim_camera_up_gs"] = [0.0, 1.0, 0.0]
        config["sim_camera_fov_y_degrees"] = 35.0
    clamp_genesis_friction_values(config)
    return config


def enforce_standard_config_policy(args, config_path):
    config = load_yaml(config_path)
    if not config:
        return config
    updated = apply_pipeline_config_policy(args, config)
    Path(config_path).write_text(
        yaml.safe_dump(updated, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    return updated


def build_point_effect_summary(config, manifest, evaluator, reflection):
    artifacts = manifest.get("artifacts", {}) if isinstance(manifest, dict) else {}
    object_artifacts = artifacts.get("object_artifacts", []) or []
    errors = evaluator.get("reconstruction_errors", []) or []
    evidence = evaluator.get("evidence", []) or []
    reflected_effects = reflection.get("per_object_point_effects", []) or []
    reflected_by_idx = {
        effect.get("object_index"): effect
        for effect in reflected_effects
        if isinstance(effect, dict) and "object_index" in effect
    }

    rows = []
    object_points = config.get("all_object_points", []) or []
    mask_indices = config.get("all_object_masks_idx", []) or []
    for object_idx, points in enumerate(object_points):
        artifact = next(
            (item for item in object_artifacts if item.get("object_index") == object_idx),
            {},
        )
        related_text = []
        pattern = re.compile(rf"\b(?:object|obj)\s*{object_idx}\b", re.IGNORECASE)
        for text in [*errors, *evidence]:
            if isinstance(text, str) and pattern.search(text):
                related_text.append(text)
        reflected = reflected_by_idx.get(object_idx, {})
        if reflected.get("observed_effect"):
            related_text.insert(0, reflected["observed_effect"])
        status = reflected.get("status")
        if not status:
            status = "bad" if related_text else "not_explicitly_diagnosed"
        rows.append(
            {
                "object_index": object_idx,
                "previous_points_original_pixels": points,
                "selected_sam_local_mask_index": mask_indices[object_idx] if object_idx < len(mask_indices) else None,
                "status": status,
                "observed_effect": related_text or [
                    "No per-object text diagnosis was returned; inspect point overlay, SAM2 candidates, selected mask, and saved mask."
                ],
                "sam2_input_points_debug": artifact.get("sam2_input_points_debug", []),
                "sam2_selected_mask_debug": artifact.get("sam2_selected_mask_debug", []),
                "saved_object_mask": artifact.get("saved_object_mask", []),
            }
        )
    return rows


def regenerate_from_reflection(
    messages,
    reflection,
    args,
    case_name,
    image_path,
    artifacts_dir,
    iteration_idx,
    config_path,
    handler_path,
    stage=None,
    evaluator_path=None,
    manifest_path=None,
):
    current_config = load_yaml(config_path)
    evaluator = {}
    if evaluator_path and Path(evaluator_path).exists():
        evaluator = json.loads(Path(evaluator_path).read_text(encoding="utf-8"))
    manifest = load_yaml(manifest_path) if manifest_path else {}

    if stage == "mask":
        stage_instruction = (
            "This feedback is from the pre-SAM3D mask gate. Revise segmentation keys "
            "first: all_object_points, then all_object_masks_idx. Preserve unrelated "
            "physics, background, and handler logic unless the reflection explicitly "
            "requires a change. You must treat the current all_object_points below as "
            "the failed coordinates and output a new complete YAML with corrected "
            "points. The failed coordinates below come from the saved runtime config "
            "and are in original-image pixels; your new YAML must still output "
            "all_object_points on the required 0-1000 relative grid. Do not blindly "
            "copy numeric replacement coordinates from the reflection; use the "
            "original image, colored point overlay, and per-object mask effects to "
            "choose corrected interior points. Keep retrying conceptually until the "
            "mask evaluator would pass."
        )
    elif stage == "reconstruction":
        stage_instruction = (
            "This feedback is from the SAM3D reconstruction gate. SAM3D itself is not "
            "editable here; focus on config-controlled alignment such as obj_kp, "
            "obj_kp_matching, mesh_resize_factor, and related reconstruction parameters."
        )
    else:
        stage_instruction = (
            "This feedback is from the simulation gate. Treat the accepted YAML mask, "
            "reconstruction, support, camera, and gravity parameters in the previous "
            "config as the base version. Revise the Python handler first for force, "
            "velocity, placement, and contact timing. "
            "Only minor scalar physics/contact YAML values may change; do not change "
            "segmentation points, mask indices, support points, camera mode, camera pose, "
            "or reconstruction settings."
        )
    if stage == "mask":
        point_effect_summary = build_point_effect_summary(current_config, manifest, evaluator, reflection)
        stage_context = (
            "\n\nCurrent failed segmentation config:\n"
            f"{yaml.safe_dump({key: current_config.get(key) for key in ['all_object_points', 'all_object_masks_idx', 'support_object_points', 'support_object_masks_idx'] if key in current_config}, sort_keys=False, allow_unicode=True)}"
            "\nPrevious coordinates and observed mask effects by object:\n"
            f"{yaml.safe_dump(point_effect_summary, sort_keys=False, allow_unicode=True)}"
            "\nMask evaluator error JSON:\n"
            f"{json.dumps(evaluator, indent=2, ensure_ascii=False)}"
            "\n\nMask artifact manifest:\n"
            f"{yaml.safe_dump(manifest, sort_keys=False, allow_unicode=True)}"
        )
    else:
        stage_context = (
            "\n\nCurrent config YAML:\n"
            f"{yaml.safe_dump(current_config, sort_keys=False, allow_unicode=True)}"
            "\nEvaluator error JSON:\n"
            f"{json.dumps(evaluator, indent=2, ensure_ascii=False)}"
        )
    feedback = (
        "The reflection agent diagnosed the previous result. Revise the outputs using this diagnosis. "
        f"{stage_instruction} "
        "Output the full revised YAML followed by the full revised Python handler. "
        "For simulation feedback, the YAML should be a minimal local patch over the "
        "current config shown below.\n\n"
        f"Reflection JSON:\n{json.dumps(reflection, indent=2, ensure_ascii=False)}"
        f"{stage_context}"
    )
    feedback_content = []
    if stage == "mask":
        feedback_content.append(media_item(image_path))
    feedback_content.append({"text": feedback})
    if stage == "mask" and manifest:
        add_media_items(feedback_content, reconstruction_media_from_manifest(manifest), max_items=18)
    feedback_content = [item for item in feedback_content if item is not None]
    round_messages = messages + [{"role": "user", "content": feedback_content}]
    raw_response, raw_yaml_text, raw_python_text, config_dict, handler_code, scaling_info = run_generation_round(
        round_messages,
        model=args.model,
        case_name=case_name,
        image_path=image_path,
        point_scale_mode=args.point_scale_mode,
        target_area=args.target_area,
    )
    if stage == "short_simulation":
        config_dict = preserve_accepted_config_for_motion(current_config, config_dict)
    apply_pipeline_config_policy(args, config_dict)
    save_iteration_outputs(
        iteration_idx=iteration_idx,
        artifacts_dir=artifacts_dir,
        raw_response=raw_response,
        raw_yaml_text=raw_yaml_text,
        raw_python_text=raw_python_text,
        config_dict=config_dict,
        handler_code=handler_code,
        scaling_info=scaling_info,
        feedback_mode="reflection",
        feedback_text=feedback,
        standard_config_path=config_path,
        standard_handler_path=handler_path,
    )
    validate_handler_import(handler_path, case_name)
    return raw_response


def write_run_state(args, artifacts_dir, case_name, prompt, timestamp, current_iter, config_path, handler_path, final_stage, stopped_reason=None, last_verdict=None):
    run_state = {
        "case_name": case_name,
        "prompt": prompt,
        "run_timestamp": timestamp,
        "current_iter": current_iter,
        "artifacts_dir": str(artifacts_dir),
        "run_root": str(artifacts_dir),
        "config_path": str(config_path),
        "handler_path": str(handler_path),
        "mask_accept_score": args.mask_accept_score,
        "reconstruction_accept_score": args.reconstruction_accept_score,
        "short_sim_accept_score": args.short_sim_accept_score,
        "max_mask_rounds": args.max_mask_rounds,
        "max_reconstruction_rounds": args.max_reconstruction_rounds,
        "max_motion_rounds": args.max_motion_rounds,
        "short_sim_frames": args.short_sim_frames,
        "enable_sequential_object_inpainting": args.enable_sequential_object_inpainting,
        "final_stage": final_stage,
        "stopped_reason": stopped_reason,
        "last_verdict": last_verdict,
        "iteration_layout": "iter_<n>/{config.yaml,handler.py,mask/,mask_outputs/,reconstruction/,reconstruction_outputs/,short_sim/,simulation_outputs/}",
    }
    (artifacts_dir / "run_state.yaml").write_text(
        yaml.safe_dump(run_state, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    return run_state


def main():
    args = parse_args()
    ensure_api(args)
    image_path = Path(args.image_path).expanduser().resolve()
    case_name = args.case_name or image_path.parent.name
    config_path = image_path.parent / args.config_output
    handler_path = REPO_ROOT / "simulation" / "case_simulation" / (args.force_output or f"{case_name}.py")
    timestamp = time_run_id()
    artifacts_dir = (
        Path(args.artifacts_dir).expanduser().resolve()
        if args.artifacts_dir
        else REPO_ROOT / "result" / case_name / timestamp
    )
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    system_prompt = "\n\n".join(
        [
            read_system_prompt(SCRIPT_DIR / "caption_rule.md"),
            read_system_prompt(SCRIPT_DIR / "force_rule.md"),
        ]
    )
    messages = build_initial_messages(system_prompt, image_path, case_name, args.prompt)

    print("Agent 1: generating initial config and handler...", flush=True)
    raw_response, raw_yaml_text, raw_python_text, config_dict, handler_code, scaling_info = run_generation_round(
        messages,
        model=args.model,
        case_name=case_name,
        image_path=image_path,
        point_scale_mode=args.point_scale_mode,
        target_area=args.target_area,
    )
    apply_pipeline_config_policy(args, config_dict)
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
        standard_config_path=config_path,
        standard_handler_path=handler_path,
    )
    validate_handler_import(handler_path, case_name)

    current_iter = 0
    mask_accepted = False
    last_mask_verdict = None
    for mask_round in range(args.max_mask_rounds + 1):
        enforce_standard_config_policy(args, config_path)
        current_iter_dir = iter_dir(artifacts_dir, current_iter)
        stage_dir = current_iter_dir / "mask"
        mask_output_dir = current_iter_dir / "mask_outputs"
        manifest_path = stage_dir / "artifact_manifest.yaml"
        run_subprocess(
            [
                sys.executable,
                "cases/run_reconstruction_debug.py",
                "--config_path",
                config_path,
                "--output_folder",
                mask_output_dir,
                "--artifact_manifest",
                manifest_path,
                "--mask_only",
            ]
        )
        evaluator_path = stage_dir / "evaluator.json"
        verdict = call_evaluator("mask", args, config_path, handler_path, manifest_path, evaluator_path)
        last_mask_verdict = verdict
        print(f"Mask evaluator: pass={verdict.get('pass')} score={verdict.get('score')}", flush=True)
        if verdict.get("pass") and float(verdict.get("score", 0.0)) >= args.mask_accept_score:
            mask_accepted = True
            break
        if mask_round >= args.max_mask_rounds:
            break
        reflection_path = stage_dir / "reflection.json"
        reflection = call_reflection("mask", args, config_path, handler_path, manifest_path, evaluator_path, reflection_path)
        current_iter += 1
        regenerate_from_reflection(
            messages,
            reflection,
            args,
            case_name,
            image_path,
            artifacts_dir,
            current_iter,
            config_path,
            handler_path,
            stage="mask",
            evaluator_path=evaluator_path,
            manifest_path=manifest_path,
        )

    if not mask_accepted and not args.continue_on_mask_failure:
        write_run_state(
            args,
            artifacts_dir,
            case_name,
            args.prompt,
            timestamp,
            current_iter,
            config_path,
            handler_path,
            final_stage="mask",
            stopped_reason="mask_rejected_before_sam3d",
            last_verdict=last_mask_verdict,
        )
        print(
            "Stopping before SAM3D because the mask gate was rejected. "
            f"Use --continue_on_mask_failure to force reconstruction. Artifacts: {artifacts_dir}",
            flush=True,
        )
        return

    enforce_standard_config_policy(args, config_path)
    current_iter_dir = iter_dir(artifacts_dir, current_iter)
    stage_dir = current_iter_dir / "reconstruction"
    reconstruction_output_dir = current_iter_dir / "reconstruction_outputs"
    manifest_path = stage_dir / "artifact_manifest.yaml"
    run_subprocess(
        [
            sys.executable,
            "cases/run_reconstruction_debug.py",
            "--config_path",
            config_path,
            "--output_folder",
            reconstruction_output_dir,
            "--artifact_manifest",
            manifest_path,
        ]
    )
    print("Reconstruction completed; skipping reconstruction evaluator/reflection.", flush=True)

    for motion_round in range(args.max_motion_rounds + 1):
        enforce_standard_config_policy(args, config_path)
        current_iter_dir = iter_dir(artifacts_dir, current_iter)
        stage_dir = current_iter_dir / "short_sim"
        short_sim_output_dir = current_iter_dir / "simulation_outputs"
        manifest_path = stage_dir / "artifact_manifest.yaml"
        run_subprocess(
            [
                sys.executable,
                "case_simulation.py",
                "--config_path",
                config_path,
                "--output_folder",
                short_sim_output_dir,
                "--run_mode",
                "short_sim",
                "--max_frames",
                args.short_sim_frames,
                "--artifact_manifest",
                manifest_path,
            ]
        )
        evaluator_path = stage_dir / "evaluator.json"
        verdict = call_evaluator("short_simulation", args, config_path, handler_path, manifest_path, evaluator_path)
        print(f"Short simulation evaluator: pass={verdict.get('pass')} score={verdict.get('score')}", flush=True)
        if verdict.get("pass") and float(verdict.get("score", 0.0)) >= args.short_sim_accept_score:
            break
        if motion_round >= args.max_motion_rounds:
            break
        reflection_path = stage_dir / "reflection.json"
        reflection = call_reflection("short_simulation", args, config_path, handler_path, manifest_path, evaluator_path, reflection_path)
        current_iter += 1
        regenerate_from_reflection(
            messages,
            reflection,
            args,
            case_name,
            image_path,
            artifacts_dir,
            current_iter,
            config_path,
            handler_path,
            stage="short_simulation",
            evaluator_path=evaluator_path,
            manifest_path=manifest_path,
        )

    write_run_state(
        args,
        artifacts_dir,
        case_name,
        args.prompt,
        timestamp,
        current_iter,
        config_path,
        handler_path,
        final_stage="short_simulation",
    )
    print(f"Multi-agent run artifacts saved under: {artifacts_dir}", flush=True)


if __name__ == "__main__":
    main()
