import argparse
import gc
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np
if "bool" not in np.__dict__:
    np.bool = bool
import torch
import yaml
from omegaconf import DictConfig, ListConfig, OmegaConf
from PIL import Image, ImageDraw

try:
    import trimesh
except Exception:
    trimesh = None

REPO_ROOT = Path(__file__).resolve().parents[1]
SUBMODULE_PATHS = [
    REPO_ROOT,
]
for path in SUBMODULE_PATHS:
    if path.exists() and str(path) not in sys.path:
        sys.path.insert(0, str(path))


def time_run_id():
    return f"{time.time():.6f}".replace(".", "_")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run only the single-view reconstruction stage and save debug artifacts for multi-agent evaluation."
    )
    parser.add_argument("--config_path", required=True, help="Path to the YAML config.")
    parser.add_argument(
        "--output_folder",
        help="Optional output folder. Defaults to result/<case>/<timestamp>.",
    )
    parser.add_argument(
        "--artifact_manifest",
        help="Optional path to write a YAML artifact manifest.",
    )
    parser.add_argument(
        "--mask_only",
        action="store_true",
        help="Run only SAM/RepViT segmentation and write mask debug artifacts; skip inpainting, MoGe, and SAM3D.",
    )
    return parser.parse_args()


def list_existing(paths):
    return [str(path) for path in paths if Path(path).exists()]


def sorted_existing(pattern):
    return sorted(str(path) for path in pattern if Path(path).exists())


def object_artifacts(output_folder, config):
    output_folder = Path(output_folder)
    debug_dir = REPO_ROOT / "debug" / "sam2"
    points = config.get("all_object_points", []) or []
    mask_indices = config.get("all_object_masks_idx", []) or []
    material_type = config.get("material_type", []) or []
    objects = []

    for object_idx in range(len(points)):
        try:
            mask_idx = int(mask_indices[object_idx])
        except (IndexError, TypeError, ValueError):
            mask_idx = 0
        candidate_paths = sorted_existing(debug_dir.glob(f"object_{object_idx:02d}_masks_*.png"))
        if candidate_paths:
            selected_idx = max(0, min(mask_idx, len(candidate_paths) - 1))
            selected_mask_debug = [candidate_paths[selected_idx]]
        else:
            selected_mask_debug = list_existing(
                [debug_dir / f"object_{object_idx:02d}_masks_{mask_idx:02d}.png"]
            )
        objects.append(
            {
                "object_index": object_idx,
                "prompt_points": points[object_idx],
                "selected_sam_proposal_index": mask_idx,
                "material_type": material_type[object_idx] if object_idx < len(material_type) else None,
                "sam2_input_points_debug": list_existing(
                    [debug_dir / f"input_points_{object_idx:02d}.png"]
                ),
                "sam2_selected_mask_debug": selected_mask_debug,
                "sam2_all_mask_candidates_debug": candidate_paths,
                "saved_object_mask": list_existing(
                    [
                        output_folder / f"mask_{object_idx:02d}.png",
                        output_folder / f"refined_mask_{object_idx:02d}.png",
                    ]
                ),
                "mesh": list_existing(
                    [
                        output_folder / f"sam3d_mesh_{object_idx:02d}.obj",
                        output_folder / f"sam3d_mesh_{object_idx:02d}_simplified.obj",
                    ]
                ),
                "point_cloud": list_existing(
                    [output_folder / f"merged_per_points_{object_idx:02d}.ply"]
                ),
                "keypoint_debug": list_existing(
                    [
                        output_folder / "render" / f"gt_kps_{object_idx:02d}.png",
                        output_folder / "render" / f"mesh_kps_{object_idx:02d}.png",
                        output_folder / "render" / f"mesh_init_render_proxy_color_{object_idx:02d}.png",
                    ]
                ),
            }
        )

    return objects


def support_object_artifacts(output_folder, config):
    output_folder = Path(output_folder)
    debug_dir = REPO_ROOT / "debug" / "sam2"
    points = first_present(config, ["support_object_points", "static_object_points", "fixed_object_points"], []) or []
    mask_indices = first_present(config, ["support_object_masks_idx", "static_object_masks_idx", "fixed_object_masks_idx"], []) or []
    names = config.get("support_object_names", []) or config.get("static_object_names", []) or []
    objects = []

    for object_idx in range(len(points)):
        try:
            mask_idx = int(mask_indices[object_idx])
        except (IndexError, TypeError, ValueError):
            mask_idx = 0
        candidate_paths = sorted_existing(debug_dir.glob(f"support_object_{object_idx:02d}_masks_*.png"))
        if candidate_paths:
            selected_idx = max(0, min(mask_idx, len(candidate_paths) - 1))
            selected_mask_debug = [candidate_paths[selected_idx]]
        else:
            selected_mask_debug = list_existing(
                [debug_dir / f"support_object_{object_idx:02d}_masks_{mask_idx:02d}.png"]
            )
        objects.append(
            {
                "object_index": object_idx,
                "name": names[object_idx] if object_idx < len(names) else f"support_object_{object_idx:02d}",
                "prompt_points": points[object_idx],
                "selected_sam_proposal_index": mask_idx,
                "sam2_input_points_debug": list_existing(
                    [debug_dir / f"support_input_points_{object_idx:02d}.png"]
                ),
                "sam2_selected_mask_debug": selected_mask_debug,
                "sam2_all_mask_candidates_debug": candidate_paths,
                "saved_object_mask": list_existing(
                    [output_folder / f"support_mask_{object_idx:02d}.png"]
                ),
                "mesh": list_existing(
                    [
                        output_folder / f"support_object_mesh_{object_idx:02d}.obj",
                        output_folder / f"support_object_mesh_{object_idx:02d}_simplified.obj",
                        output_folder / f"static_support_mesh_{object_idx:02d}_pt3d.obj",
                        output_folder / f"static_support_mesh_{object_idx:02d}_gs.obj",
                    ]
                ),
                "keypoint_debug": list_existing(
                    [
                        output_folder / "render" / f"gt_kps_support_{object_idx:02d}.png",
                        output_folder / "render" / f"mesh_kps_support_{object_idx:02d}.png",
                        output_folder / "render" / f"mesh_init_render_proxy_color_support_{object_idx:02d}.png",
                    ]
                ),
            }
        )

    return objects


def first_present(config, keys, default=None):
    for key in keys:
        if key in config and config.get(key):
            return config.get(key)
    return default


def mask_summary(path):
    path = Path(path)
    if not path.exists():
        return {}
    try:
        with Image.open(path) as image:
            arr = np.asarray(image.convert("L"))
    except Exception as exc:
        return {"path": str(path), "error": str(exc)}

    mask = arr > 0
    height, width = mask.shape
    area = int(mask.sum())
    summary = {
        "path": str(path),
        "image_size": [int(width), int(height)],
        "area_px": area,
        "area_ratio": float(area / max(1, width * height)),
    }
    if area > 0:
        ys, xs = np.where(mask)
        bbox = [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]
        bbox_w = max(1, bbox[2] - bbox[0])
        bbox_h = max(1, bbox[3] - bbox[1])
        summary.update(
            {
                "bbox_xyxy": bbox,
                "bbox_size": [int(bbox_w), int(bbox_h)],
                "bbox_aspect_w_over_h": float(bbox_w / bbox_h),
                "bbox_fill_ratio": float(area / max(1, bbox_w * bbox_h)),
            }
        )
    return summary


def mesh_summary(path):
    path = Path(path)
    if trimesh is None or not path.exists():
        return {}
    try:
        mesh = trimesh.load(path, process=False)
    except Exception as exc:
        return {"path": str(path), "error": str(exc)}
    if hasattr(mesh, "geometry"):
        meshes = [geom for geom in mesh.geometry.values() if hasattr(geom, "vertices")]
        if not meshes:
            return {"path": str(path), "error": "no mesh geometry"}
        mesh = trimesh.util.concatenate(meshes)
    vertices = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.faces)
    if vertices.size == 0:
        return {"path": str(path), "vertex_count": 0, "face_count": int(len(faces))}
    bounds_min = vertices.min(axis=0)
    bounds_max = vertices.max(axis=0)
    extents = bounds_max - bounds_min
    sorted_extents = np.sort(extents)
    return {
        "path": str(path),
        "vertex_count": int(vertices.shape[0]),
        "face_count": int(faces.shape[0]),
        "bounds_min": bounds_min.tolist(),
        "bounds_max": bounds_max.tolist(),
        "extents": extents.tolist(),
        "thinness_ratio_min_over_max": float(sorted_extents[0] / max(sorted_extents[-1], 1e-8)),
    }


def object_quality_summary(output_folder, config):
    summaries = []
    object_count = len(config.get("all_object_points", []) or [])
    for object_idx in range(object_count):
        mask_paths = list_existing(
            [
                Path(output_folder) / f"refined_mask_{object_idx:02d}.png",
                Path(output_folder) / f"mask_{object_idx:02d}.png",
            ]
        )
        mesh_path = Path(output_folder) / f"sam3d_mesh_{object_idx:02d}_simplified.obj"
        summaries.append(
            {
                "object_index": object_idx,
                "material_type": (config.get("material_type", []) or [None])[object_idx]
                if object_idx < len(config.get("material_type", []) or [])
                else None,
                "mask": mask_summary(mask_paths[0]) if mask_paths else {},
                "simplified_mesh": mesh_summary(mesh_path),
            }
        )
    return summaries


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


def write_manifest(path, payload):
    if not path:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(to_yaml_safe(payload), sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )


def normalize_mask_indices(mask_indices, count):
    if mask_indices is None:
        normalized = [0] * count
    elif isinstance(mask_indices, (list, tuple)):
        normalized = list(mask_indices)
    elif hasattr(mask_indices, "__iter__") and not isinstance(mask_indices, (str, bytes)):
        normalized = list(mask_indices)
    else:
        normalized = [mask_indices]
    if len(normalized) == 1 and count > 1:
        normalized = normalized * count
    elif len(normalized) < count:
        normalized = normalized + [0] * (count - len(normalized))
    elif len(normalized) > count:
        normalized = normalized[:count]
    return normalized


def save_mask_image(mask, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.asarray(mask).astype(bool).astype(np.uint8) * 255
    Image.fromarray(arr).save(path)


def draw_points_overlay(config, output_folder):
    output_folder = Path(output_folder)
    image_path = REPO_ROOT / config["data_path"] / "input.png"
    if not image_path.exists():
        return None

    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    colors = [
        (230, 25, 75),    # red
        (60, 180, 75),    # green
        (0, 130, 200),    # blue
        (245, 130, 48),   # orange
        (145, 30, 180),   # purple
        (70, 240, 240),   # cyan
        (240, 50, 230),   # magenta
        (210, 245, 60),   # lime
    ]
    radius = max(5, int(round(min(image.size) * 0.012)))

    def draw_point(x, y, color, label, shape="circle"):
        x = float(x)
        y = float(y)
        bbox = [x - radius, y - radius, x + radius, y + radius]
        if shape == "square":
            draw.rectangle(bbox, fill=color, outline=(255, 255, 255), width=2)
        else:
            draw.ellipse(bbox, fill=color, outline=(255, 255, 255), width=2)
        text_pos = (x + radius + 3, y - radius - 3)
        draw.text((text_pos[0] + 1, text_pos[1] + 1), label, fill=(0, 0, 0))
        draw.text(text_pos, label, fill=color)

    for object_idx, object_points in enumerate(config.get("all_object_points", []) or []):
        color = colors[object_idx % len(colors)]
        for point_idx, point in enumerate(object_points or []):
            if len(point) < 2:
                continue
            suffix = f".{point_idx}" if len(object_points) > 1 else ""
            draw_point(point[0], point[1], color, f"obj{object_idx}{suffix}")

    support_points = first_present(
        config,
        ["support_object_points", "static_object_points", "fixed_object_points"],
        [],
    ) or []
    for support_idx, object_points in enumerate(support_points):
        for point_idx, point in enumerate(object_points or []):
            if len(point) < 2:
                continue
            suffix = f".{point_idx}" if len(object_points) > 1 else ""
            draw_point(point[0], point[1], (255, 225, 25), f"support{support_idx}{suffix}", shape="square")

    output_path = output_folder / "all_object_points_overlay.png"
    image.save(output_path)
    return str(output_path)


def run_mask_only(config, output_folder):
    from simulation.image23D.segmenter import RepViTSegmenter, SegmentAnythingSegmenter
    from simulation.utils import remove_isolated_areas

    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)
    image_path = REPO_ROOT / config["data_path"] / "input.png"
    input_image_pil = Image.open(image_path).convert("RGB")
    device = config["device"]
    segmenter_name = config.get("segmenter", "repvit")

    if segmenter_name == "repvit":
        segmenter = RepViTSegmenter(device)
        target_masks = segmenter(
            input_image_pil,
            target_class=config["object_id"],
            merge_mask=bool(config.get("merge_mask", False)),
        )
    elif segmenter_name == "sam2":
        segmenter = SegmentAnythingSegmenter(config, device)
        object_points = config.get("all_object_points", []) or []
        config["all_object_masks_idx"] = normalize_mask_indices(
            config.get("all_object_masks_idx", [0]),
            len(object_points),
        )
        if bool(config.get("sequential_object_inpainting", config.get("object_occlusion_aware_inpainting", False))) and len(object_points) > 1:
            from simulation.image23D.inpainter import FluxInpainter
            from torchvision.transforms import ToTensor

            target_masks = []
            current_image_pil = input_image_pil
            inpainter = None
            for object_idx, object_points_i in enumerate(object_points):
                per_mask = segmenter.predict_masks(
                    current_image_pil,
                    [object_points_i],
                    [config["all_object_masks_idx"][object_idx]],
                    input_prefix="input_points",
                    mask_prefix="object",
                    debug_index_offset=object_idx,
                )[0]
                target_masks.append(per_mask)
                if inpainter is None:
                    inpainter = FluxInpainter(device=device)
                current_image_tensor = ToTensor()(current_image_pil).to(device)
                current_mask_tensor = torch.from_numpy(per_mask).to(device)
                current_image_pil = inpainter(
                    current_image_tensor,
                    current_mask_tensor,
                    size=input_image_pil.size,
                    prompt=config["inpainting_prompt"],
                    negative_prompt=config["inpainting_negative_prompt"],
                )
                if object_idx < len(object_points) - 1:
                    current_image_pil.save(output_folder / f"object_alignment_image_{object_idx + 1:02d}.png")
                del current_image_tensor, current_mask_tensor
                torch.cuda.empty_cache()
            current_image_pil.save(output_folder / "inpainted_image.png")
            if inpainter is not None:
                del inpainter
                gc.collect()
                torch.cuda.empty_cache()
            config["sequential_object_inpainting"] = True
        else:
            target_masks = segmenter(input_image_pil)
            config["sequential_object_inpainting"] = False
    else:
        raise ValueError(f"Invalid segmenter: {segmenter_name}")

    saved_masks = []
    refine_mask = bool(config.get("refine_mask", False))
    min_size = int(config.get("min_size", 100))
    for idx, mask in enumerate(target_masks):
        mask_bool = np.asarray(mask).astype(bool)
        if refine_mask:
            mask_bool = remove_isolated_areas(mask_bool, min_size=min_size)
            mask_path = output_folder / f"refined_mask_{idx:02d}.png"
        else:
            mask_path = output_folder / f"mask_{idx:02d}.png"
        save_mask_image(mask_bool, mask_path)
        saved_masks.append(mask_bool)

    support_points = first_present(
        config,
        ["support_object_points", "static_object_points", "fixed_object_points"],
        [],
    ) or []
    support_saved_masks = []
    if support_points and isinstance(segmenter, SegmentAnythingSegmenter):
        support_masks_idx = normalize_mask_indices(
            first_present(
                config,
                ["support_object_masks_idx", "static_object_masks_idx", "fixed_object_masks_idx"],
                [0],
            ),
            len(support_points),
        )
        config["support_object_masks_idx"] = support_masks_idx
        support_masks = segmenter.predict_masks(
            input_image_pil,
            support_points,
            support_masks_idx,
            input_prefix="support_input_points",
            mask_prefix="support_object",
        )
        for support_idx, support_mask in enumerate(support_masks):
            save_mask_image(support_mask, output_folder / f"support_mask_{support_idx:02d}.png")
            support_saved_masks.append(np.asarray(support_mask).astype(bool))

    if "segmenter" in locals():
        del segmenter
        gc.collect()
        torch.cuda.empty_cache()

    remove_support_from_background = bool(
        config.get("remove_support_from_background_inpainting", bool(support_saved_masks))
    )
    config["remove_support_from_background_inpainting"] = remove_support_from_background
    config["background_collision_uses_support_removed_inpainting"] = bool(
        remove_support_from_background and support_saved_masks
    )
    if saved_masks:
        visual_union_mask = np.zeros_like(saved_masks[0], dtype=bool)
        for mask in saved_masks:
            visual_union_mask = visual_union_mask | mask
        save_mask_image(visual_union_mask, output_folder / "inpainter_masks.png")
        if remove_support_from_background and support_saved_masks:
            support_union = np.zeros_like(saved_masks[0], dtype=bool)
            for support_mask in support_saved_masks:
                support_union = support_union | support_mask
            background_collision_union = visual_union_mask | support_union
            save_mask_image(support_union, output_folder / "support_inpainter_masks.png")
            save_mask_image(background_collision_union, output_folder / "background_collision_inpainter_masks.png")


def main():
    args = parse_args()
    config_path = Path(args.config_path).expanduser().resolve()
    config = OmegaConf.load(config_path)

    if args.output_folder:
        output_folder = Path(args.output_folder).expanduser().resolve()
    else:
        timestamp = time_run_id()
        output_folder = REPO_ROOT / config["output_folder"] / timestamp
    output_folder.mkdir(parents=True, exist_ok=True)
    config["output_folder"] = str(output_folder)
    config["debug"] = True

    OmegaConf.save(config, output_folder / "config.yaml")
    configs_folder = output_folder / "configs"
    configs_folder.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(config, configs_folder / "config.yaml")
    shutil.copy2(config_path, configs_folder / config_path.name)

    handler_path = REPO_ROOT / "simulation" / "case_simulation" / f"{config['example_name']}.py"
    if handler_path.exists():
        shutil.copy2(handler_path, configs_folder / handler_path.name)

    if args.mask_only:
        run_mask_only(config, output_folder)
    else:
        torch.set_grad_enabled(False)
        from simulation.image23D.single_view_reconstructor import SingleViewReconstructor

        reconstructor = SingleViewReconstructor(config)
        reconstructor.reconstruct()
    draw_points_overlay(config, output_folder)
    OmegaConf.save(config, output_folder / "config.yaml")
    OmegaConf.save(config, configs_folder / "config.yaml")

    render_dir = output_folder / "render"
    object_count = len(config.get("all_object_points", []) or [])
    support_points = first_present(config, ["support_object_points", "static_object_points", "fixed_object_points"], []) or []
    artifacts = {
        "input_image": str(REPO_ROOT / config["data_path"] / "input.png"),
        "sam2_debug_dir": str(REPO_ROOT / "debug" / "sam2"),
        "config": str(output_folder / "config.yaml"),
        "configs_dir": str(configs_folder),
        "object_count": object_count,
        "scene_context": {
            "dynamic_object_count": object_count,
            "material_type": config.get("material_type", []),
            "dynamic_object_names": config.get("object_names", []),
            "support_object_count": len(support_points),
            "support_object_names": config.get("support_object_names", []) or config.get("static_object_names", []),
            "static_support_strategy": (
                "support_object_points, when present, are reconstructed as fixed collision-only "
                "objects. They do not increase material_type length and should not be counted "
                "as moving foreground objects."
            ),
            "remove_support_from_background_inpainting": config.get("remove_support_from_background_inpainting", False),
            "force_regenerate_inpainted_with_support": config.get("force_regenerate_inpainted_with_support", False),
            "background_collision_uses_support_removed_inpainting": config.get("background_collision_uses_support_removed_inpainting", False),
            "static_support_replaces_background_collision": config.get("static_support_replaces_background_collision", False),
            "support_background_removal_strategy": (
                "Final visual compositing uses inpainter_masks.png, which removes dynamic objects only and "
                "keeps visible support objects such as tables. If static_support_replaces_background_collision "
                "is true, the pipeline uses the reconstructed support mesh directly and does not need a "
                "separate background collision reconstruction. Otherwise, when "
                "remove_support_from_background_inpainting is true, "
                "background_collision_inpainter_masks.png also includes support_object_masks so only "
                "the reconstructed background collision mesh is built after removing fixed support geometry."
            ),
            "thin_object_evaluation_note": (
                "Books, plates, paper, cards, and other slab-like objects are physically thin. "
                "Do not reject their reconstruction only because a proxy render appears thin; "
                "use mask coverage, mesh extents, and keypoint alignment before diagnosing a mesh failure."
            ),
        },
        "all_object_points": config.get("all_object_points", []),
        "all_object_points_overlay": list_existing([output_folder / "all_object_points_overlay.png"]),
        "all_object_masks_idx": config.get("all_object_masks_idx", []),
        "all_object_masks_idx_semantics": (
            "Each all_object_masks_idx[i] is a local proposal index within SAM2's "
            "multimask outputs for object i. Repeated values across objects are "
            "valid and do not imply shared masks."
        ),
        "object_artifacts": object_artifacts(output_folder, config),
        "object_quality_summary": object_quality_summary(output_folder, config),
        "support_object_artifacts": support_object_artifacts(output_folder, config),
        "static_collision_objects": config.get("static_collision_objects", []),
        "masks": sorted_existing(output_folder.glob("mask_*.png"))
        + sorted_existing(output_folder.glob("refined_mask_*.png")),
        "support_masks": sorted_existing(output_folder.glob("support_mask_*.png")),
        "union_inpainting_mask": list_existing([output_folder / "inpainter_masks.png"]),
        "support_inpainting_mask": list_existing([output_folder / "support_inpainter_masks.png"]),
        "background_collision_inpainting_mask": list_existing([output_folder / "background_collision_inpainter_masks.png"]),
        "inpainting": list_existing(
            [
                output_folder / "inpainted_image.png",
                output_folder / "stitched_inpainted_image.png",
            ]
        )
        + sorted_existing(output_folder.glob("object_alignment_image_*.png")),
        "background_collision_inpainting": list_existing(
            [
                output_folder / "background_collision_inpainted_image.png",
                output_folder / "stitched_background_collision_inpainted_image.png",
            ]
        ),
        "depth": sorted_existing(output_folder.glob("depth_input_*.png"))
        + sorted_existing(output_folder.glob("depth_inpainted_*.png"))
        + sorted_existing(output_folder.glob("depth_background_collision_*.png"))
        + sorted_existing(output_folder.glob("depth_object_alignment_*.png")),
        "keypoints_by_object": sorted_existing(render_dir.glob("gt_kps_*.png"))
        + sorted_existing(render_dir.glob("mesh_kps_*.png")),
        "mesh_proxy_by_object": sorted_existing(render_dir.glob("mesh_init_render_proxy_color_*.png")),
        "meshes": sorted(str(path) for path in output_folder.glob("*.obj")),
        "point_clouds": sorted(str(path) for path in output_folder.glob("*.ply")),
        "background_collision_mesh": list_existing(
            [
                output_folder / "background_collision_mesh_pt3d.obj",
                output_folder / "background_collision_mesh_gs.obj",
            ]
        ),
        "background_collision_mesh_summary": {
            "pt3d": mesh_summary(output_folder / "background_collision_mesh_pt3d.obj"),
            "gs": mesh_summary(output_folder / "background_collision_mesh_gs.obj"),
            "effective_stride": config.get("background_collision_mesh_effective_stride", config.get("background_collision_mesh_stride")),
            "max_faces": config.get("background_collision_mesh_max_faces", None),
        },
        "background_plane": {
            key: config.get(key)
            for key in [
                "background_plane_raw_normal_pt3d",
                "background_plane_normal_pt3d",
                "background_plane_point_pt3d",
                "background_plane_snap_angle_degrees",
            ]
            if key in config
        },
        "artifact_roles": {
            "object_artifacts": "Per dynamic object evidence. Use this first for multi-object segmentation, mesh, point cloud, and keypoint debugging.",
            "object_quality_summary": "Numeric mask and mesh summaries. Use these before calling a physically thin object bad.",
            "support_object_artifacts": "Optional fixed support/collision objects, such as a table reconstructed by SAM3D. These are not dynamic foreground objects.",
            "static_collision_objects": "Runtime fixed Genesis collision meshes exported from support_object_points.",
            "sam2_input_points_debug": "Shows the prompt point(s) sent to SAM2 for that object.",
            "all_object_points_overlay": "Original input image annotated with all current object prompt points. Dynamic objects use colored circles labeled obj0, obj1, etc.; support points use yellow squares.",
            "sam2_selected_mask_debug": "Shows the selected SAM2 proposal before reconstruction. Compare with all_object_masks_idx.",
            "sam2_all_mask_candidates_debug": "Shows every SAM2 multimask candidate saved for that object. Use this to decide whether all_object_masks_idx or all_object_points should change.",
            "saved_object_mask": "Mask actually used for each object reconstruction. For multi-object scenes, expect one entry per object.",
            "union_inpainting_mask": "Union mask used for final visual background inpainting. It includes dynamic-object masks only, so visible support objects remain in the composed video background.",
            "support_inpainting_mask": "Union of support-object masks, saved when support objects are reconstructed separately.",
            "background_collision_inpainting_mask": "Union mask used only for reconstructed background collision. It includes dynamic-object masks plus support-object masks when remove_support_from_background_inpainting is true.",
            "inpainting": "Final visual background after removing dynamic foreground objects. Support objects remain visible here.",
            "background_collision_inpainting": "Collision-only background image after removing dynamic objects and separately reconstructed support objects.",
            "depth": "MoGe depth maps for the input, visual inpainted background, optional support-removed background collision image, and optional sequential object-alignment images when sequential_object_inpainting is enabled.",
            "keypoints_by_object": "Per-object numbered keypoint alignment images, e.g. gt_kps_00.png and mesh_kps_00.png.",
            "mesh_proxy_by_object": "Per-object numbered pre-alignment proxy renders, e.g. mesh_init_render_proxy_color_00.png.",
            "meshes": "OBJ files for every reconstructed foreground object. Count simplified/original pairs carefully.",
            "point_clouds": "Foreground point clouds per object plus projected_bg_points.ply for the final visual background and projected_background_collision_points.ply for the collision-only background when present.",
            "background_collision_mesh": "Fixed collision-only mesh reconstructed from the support-removed background collision depth when support objects are reconstructed separately; otherwise from the visual inpainted background depth. The *_pt3d mesh is camera/PyTorch3D coordinates; the *_gs mesh is already converted to Genesis coordinates and preserves the same camera-derived scale as the foreground objects.",
            "background_plane": "Estimated support plane normal/point after snapping. Use for support/gravity diagnosis.",
        },
    }

    manifest = {
        "run_mode": "mask" if args.mask_only else "reconstruction",
        "output_folder": str(output_folder),
        "artifacts": artifacts,
    }
    write_manifest(args.artifact_manifest, manifest)
    run_label = "Mask" if args.mask_only else "Reconstruction"
    print(f"{run_label} debug artifacts saved to: {output_folder}")


if __name__ == "__main__":
    main()
