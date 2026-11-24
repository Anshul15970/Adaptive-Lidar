"""
Compare KITTI 3D object point retention for uniform voxel downsampling and
the self-adaptive-grid sampler.

Labels are used ONLY for evaluation. They do not influence sampling.

Run from the project root:
    python class_evaluation_self_adaptive.py

Expected project structure:
    data/object_detection/training/velodyne/*.bin
    data/object_detection/training/label_2/*.txt
    data/object_detection/training/calib/*.txt
"""

from pathlib import Path
import csv
import time

import numpy as np
import open3d as o3d
from tqdm import tqdm

from adaptive_sampling_self_adaptive_grid import adaptive_sample_self_adaptive_grid


ROOT = Path(__file__).resolve().parent
VELODYNE_DIR = ROOT / "data" / "object_detection" / "training" / "velodyne"
LABEL_DIR = ROOT / "data" / "object_detection" / "training" / "label_2"
CALIB_DIR = ROOT / "data" / "object_detection" / "training" / "calib"
OUT_CSV = ROOT / "results" / "class_retention_self_adaptive.csv"

# Start with the same first 20 frames as the quality benchmark.
FRAME_LIMIT = 20
VOXEL_SIZE = 0.8
CLASSES = ("Car", "Pedestrian", "Cyclist")


def read_calibration(path):
    values = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if ":" not in line:
                continue
            key, value = line.split(":", 1)
            try:
                values[key] = np.fromstring(value, sep=" ")
            except ValueError:
                continue

    if "Tr_velo_to_cam" not in values or "R0_rect" not in values:
        raise ValueError(f"Missing Tr_velo_to_cam or R0_rect in {path}")

    tr = values["Tr_velo_to_cam"].reshape(3, 4)
    rect = values["R0_rect"].reshape(3, 3)
    return tr, rect


def lidar_to_camera(points_xyz, tr_velo_to_cam, r0_rect):
    points_h = np.column_stack(
        (np.asarray(points_xyz, dtype=np.float64), np.ones(len(points_xyz)))
    )
    cam = points_h @ tr_velo_to_cam.T
    return cam @ r0_rect.T


def read_labels(path):
    objects = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 15:
                continue

            cls = parts[0]
            if cls not in CLASSES:
                continue

            # KITTI label fields: h, w, l, location x/y/z, rotation_y.
            h, w, length = map(float, parts[8:11])
            loc = np.array(list(map(float, parts[11:14])), dtype=np.float64)
            rotation_y = float(parts[14])

            if h <= 0 or w <= 0 or length <= 0:
                continue

            objects.append({
                "class": cls,
                "h": h,
                "w": w,
                "l": length,
                "loc": loc,
                "rotation_y": rotation_y,
            })
    return objects


def points_inside_kitti_box(points_cam, obj):
    """Return mask for a KITTI camera-coordinate 3D box.

    KITTI location is the bottom-center of the box. In object-local coordinates
    the vertical range is [-h, 0], with x +/- l/2 and z +/- w/2.
    """
    if len(points_cam) == 0:
        return np.zeros(0, dtype=bool)

    angle = obj["rotation_y"]
    c, s = np.cos(angle), np.sin(angle)
    rot_y = np.array([
        [c, 0.0, s],
        [0.0, 1.0, 0.0],
        [-s, 0.0, c],
    ])

    # Inverse object pose, using row-vector points.
    local = (points_cam - obj["loc"]) @ rot_y
    return (
        (np.abs(local[:, 0]) <= obj["l"] / 2.0)
        & (local[:, 1] >= -obj["h"])
        & (local[:, 1] <= 0.0)
        & (np.abs(local[:, 2]) <= obj["w"] / 2.0)
    )


def load_lidar(path):
    raw = np.fromfile(path, dtype=np.float32)
    if raw.size % 4 != 0:
        raise ValueError(f"Unexpected LiDAR binary length: {path}")
    return raw.reshape(-1, 4)[:, :3]


def retention_counts(points_lidar, objects, tr, rect):
    points_cam = lidar_to_camera(points_lidar, tr, rect)
    per_class = {
        cls: {"objects": 0, "objects_with_points": 0,
              "original_box_points": 0, "retained_box_points": 0,
              "object_retention_sum": 0.0, "objects_with_original_points": 0}
        for cls in CLASSES
    }

    for obj in objects:
        cls = obj["class"]
        mask = points_inside_kitti_box(points_cam, obj)
        count = int(mask.sum())
        stats = per_class[cls]
        stats["objects"] += 1
        if count > 0:
            stats["objects_with_original_points"] += 1
            stats["original_box_points"] += count
            stats["object_retention_sum"] += 0.0  # filled by caller for reduced points

    return points_cam, per_class


def evaluate_reduced(original_lidar, reduced_lidar, objects, tr, rect):
    original_cam = lidar_to_camera(original_lidar, tr, rect)
    reduced_cam = lidar_to_camera(reduced_lidar, tr, rect)

    stats = {
        cls: {
            "objects": 0,
            "objects_with_original_points": 0,
            "objects_with_retained_points": 0,
            "original_box_points": 0,
            "retained_box_points": 0,
            "object_retention_sum": 0.0,
            "object_retention_count": 0,
        }
        for cls in CLASSES
    }

    for obj in objects:
        cls = obj["class"]
        original_mask = points_inside_kitti_box(original_cam, obj)
        reduced_mask = points_inside_kitti_box(reduced_cam, obj)
        original_count = int(original_mask.sum())
        retained_count = int(reduced_mask.sum())

        st = stats[cls]
        st["objects"] += 1
        if original_count > 0:
            st["objects_with_original_points"] += 1
            st["original_box_points"] += original_count
            st["retained_box_points"] += retained_count
            st["object_retention_sum"] += retained_count / original_count
            st["object_retention_count"] += 1
            if retained_count > 0:
                st["objects_with_retained_points"] += 1

    rows = []
    for cls in CLASSES:
        st = stats[cls]
        orig = st["original_box_points"]
        rows.append({
            "class": cls,
            "labeled_objects": st["objects"],
            "objects_with_original_lidar_points": st["objects_with_original_points"],
            "objects_retaining_at_least_one_point": st["objects_with_retained_points"],
            "object_coverage_pct": (
                100.0 * st["objects_with_retained_points"]
                / st["objects_with_original_points"]
                if st["objects_with_original_points"] else float("nan")
            ),
            "original_points_inside_boxes": orig,
            "retained_points_inside_boxes": st["retained_box_points"],
            "pooled_point_retention_pct": (
                100.0 * st["retained_box_points"] / orig if orig else float("nan")
            ),
            "mean_per_object_point_retention_pct": (
                100.0 * st["object_retention_sum"] / st["object_retention_count"]
                if st["object_retention_count"] else float("nan")
            ),
        })
    return rows


def main():
    if not VELODYNE_DIR.exists():
        raise FileNotFoundError(f"LiDAR directory not found: {VELODYNE_DIR}")
    if not LABEL_DIR.exists():
        raise FileNotFoundError(f"Label directory not found: {LABEL_DIR}")
    if not CALIB_DIR.exists():
        raise FileNotFoundError(f"Calibration directory not found: {CALIB_DIR}")

    frames = sorted(VELODYNE_DIR.glob("*.bin"))[:FRAME_LIMIT]
    if not frames:
        raise FileNotFoundError(f"No .bin frames found in {VELODYNE_DIR}")

    all_rows = []
    start_all = time.perf_counter()

    for lidar_path in tqdm(frames, desc="Class retention"):
        stem = lidar_path.stem
        label_path = LABEL_DIR / f"{stem}.txt"
        calib_path = CALIB_DIR / f"{stem}.txt"
        if not label_path.exists() or not calib_path.exists():
            print(f"Skipping {stem}: matching label or calibration file missing")
            continue

        original = load_lidar(lidar_path)
        objects = read_labels(label_path)
        tr, rect = read_calibration(calib_path)

        # Uniform baseline; target budget is exactly its output count.
        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(original)
        uniform = np.asarray(
            cloud.voxel_down_sample(VOXEL_SIZE).points, dtype=np.float32
        )

        t0 = time.perf_counter()
        adaptive = adaptive_sample_self_adaptive_grid(
            original,
            target_points=len(uniform),
            voxel_size=VOXEL_SIZE,
            ransac_iterations=15,
        )
        adaptive_runtime = time.perf_counter() - t0

        for method, reduced in (("uniform", uniform), ("self_adaptive_grid", adaptive)):
            rows = evaluate_reduced(original, reduced, objects, tr, rect)
            for row in rows:
                row["frame"] = stem
                row["method"] = method
                row["original_points"] = len(original)
                row["reduced_points"] = len(reduced)
                row["reduction_pct"] = 100.0 * (1.0 - len(reduced) / len(original))
                row["adaptive_runtime_s"] = adaptive_runtime if method == "self_adaptive_grid" else ""
                all_rows.append(row)

    if not all_rows:
        raise RuntimeError("No frames were evaluated. Check the label/calibration paths.")

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(all_rows[0].keys())
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"\nFrames evaluated: {len(set(r['frame'] for r in all_rows))}")
    print(f"Saved per-frame results: {OUT_CSV}")
    print("\nMean metrics across evaluated frames (object-weighted retention is averaged per object within each frame):")

    for method in ("uniform", "self_adaptive_grid"):
        print(f"\n{method}")
        for cls in CLASSES:
            subset = [r for r in all_rows if r["method"] == method and r["class"] == cls]
            if not subset:
                continue
            def mean(key):
                vals = [float(r[key]) for r in subset if np.isfinite(float(r[key]))]
                return float(np.mean(vals)) if vals else float("nan")

            print(
                f"  {cls:10s} | pooled point retention: "
                f"{mean('pooled_point_retention_pct'):.2f}% | "
                f"mean per-object retention: {mean('mean_per_object_point_retention_pct'):.2f}% | "
                f"object coverage: {mean('object_coverage_pct'):.2f}%"
            )

    print(f"\nTotal runtime: {time.perf_counter() - start_all:.1f} s")


if __name__ == "__main__":
    main()
