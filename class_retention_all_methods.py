
from pathlib import Path
from time import perf_counter
import csv

import numpy as np
import open3d as o3d
from tqdm import tqdm

# from adaptive_sampling import adaptive_sample
# from adaptive_sampling_v2 import adaptive_sample_v2
# from adaptive_sampling_v3 import adaptive_sample_v3
# from adaptive_sampling_v4 import adaptive_sample_v4
# from adaptive_sampling_self_adaptive_grid import (
#     adaptive_sample_self_adaptive_grid,
# )
# from hybrid_sampling_v2 import hybrid_sample_v2
# from hybrid_sampling_v3 import hybrid_sample_v3
from hybrid_sampling_v4 import hybrid_sample_v4


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "object_detection" / "training"
VELODYNE = DATA / "velodyne"
LABELS = DATA / "label_2"
CALIB = DATA / "calib"

OUT_DIR = ROOT / "results"
OBJECT_OUT = OUT_DIR / "class_retention_all_methods.csv"
SUMMARY_OUT = OUT_DIR / "class_retention_summary.csv"
RUNTIME_OUT = OUT_DIR / "sampling_runtime_all_methods.csv"

FRAME_LIMIT = 20
VOXEL_SIZE = 0.8
V4_BASE_VOXEL_SIZE = 0.4

CLASSES = {"Car", "Pedestrian", "Cyclist"}


# ============================================================
# Sampling methods
# ============================================================

METHODS = {
    # "adaptive_v1": lambda xyz, target: adaptive_sample(
    #     xyz, target_points=target
    # ),

    # "adaptive_v2": lambda xyz, target: adaptive_sample_v2(
    #     xyz, target_points=target
    # ),

    # "adaptive_v3": lambda xyz, target: adaptive_sample_v3(
    #     xyz, target_points=target
    # ),

    # "adaptive_v4": lambda xyz, target: adaptive_sample_v4(
    #     xyz,
    #     target_points=target,
    #     base_voxel_size=V4_BASE_VOXEL_SIZE,
    # ),

    # "self_adaptive_grid": lambda xyz, target: (
    #     adaptive_sample_self_adaptive_grid(
    #         xyz,
    #         target_points=target,
    #         voxel_size=VOXEL_SIZE,
    #     )
    # ),

    # "hybrid_v2": lambda xyz, target: hybrid_sample_v2(
    #     xyz,
    #     target_points=target,
    #     voxel_size=VOXEL_SIZE,
    # ),

    # "hybrid_v3": lambda xyz, target: hybrid_sample_v3(
    #     xyz,
    #     target_points=target,
    #     voxel_size=VOXEL_SIZE,
    # ),

    "hybrid_v4": lambda xyz, target: hybrid_sample_v4(
        xyz,
        target_points=target,
        voxel_size=VOXEL_SIZE,
    ),
}


# ============================================================
# KITTI calibration
# ============================================================

def read_calibration(path):
    values = {}

    with open(path, encoding="utf-8") as f:
        for line in f:
            if ":" not in line:
                continue

            key, raw = line.split(":", 1)
            arr = np.fromstring(raw, sep=" ")

            if arr.size:
                values[key.strip()] = arr

    if "Tr_velo_to_cam" not in values:
        raise ValueError(f"Missing Tr_velo_to_cam in {path}")

    rect_key = (
        "R0_rect"
        if "R0_rect" in values
        else "R_rect"
        if "R_rect" in values
        else None
    )

    if rect_key is None:
        raise ValueError(f"Missing rectification matrix in {path}")

    Tr = values["Tr_velo_to_cam"].reshape(3, 4)
    R0 = values[rect_key].reshape(3, 3)

    rotation = R0 @ Tr[:, :3]
    translation = R0 @ Tr[:, 3]

    return rotation, translation


def lidar_to_camera(xyz, rotation, translation):
    return xyz @ rotation.T + translation


# ============================================================
# KITTI object labels and 3D bounding boxes
# ============================================================

def read_boxes(path):
    boxes = []

    with open(path, encoding="utf-8") as f:
        for line in f:
            p = line.split()

            if not p or p[0] not in CLASSES:
                continue

            boxes.append({
                "class": p[0],
                "h": float(p[8]),
                "w": float(p[9]),
                "l": float(p[10]),
                "location": np.array(
                    [
                        float(p[11]),
                        float(p[12]),
                        float(p[13]),
                    ],
                    dtype=np.float32,
                ),
                "rotation_y": float(p[14]),
            })

    return boxes


def points_in_box(points_cam, box):
    # KITTI location is the bottom center of the 3D box.
    center = box["location"].copy()
    center[1] -= box["h"] / 2.0

    delta = points_cam - center

    c = np.cos(box["rotation_y"])
    s = np.sin(box["rotation_y"])

    local_x = c * delta[:, 0] - s * delta[:, 2]
    local_y = delta[:, 1]
    local_z = s * delta[:, 0] + c * delta[:, 2]

    return (
        (np.abs(local_x) <= box["l"] / 2.0)
        & (np.abs(local_y) <= box["h"] / 2.0)
        & (np.abs(local_z) <= box["w"] / 2.0)
    )


# ============================================================
# Point-cloud preprocessing and uniform baseline
# ============================================================

def clean_points(points):
    points = np.asarray(points, dtype=np.float32)

    if points.ndim != 2 or points.shape[1] < 3:
        raise ValueError(
            f"Unexpected point-cloud shape: {points.shape}"
        )

    points = points[:, :3]

    return points[np.isfinite(points).all(axis=1)]


def uniform_downsample(xyz):
    cloud = o3d.geometry.PointCloud()

    cloud.points = o3d.utility.Vector3dVector(
        xyz.astype(np.float64)
    )

    return np.asarray(
        cloud.voxel_down_sample(VOXEL_SIZE).points,
        dtype=np.float32,
    ).reshape(-1, 3)


# ============================================================
# Main retention evaluation
# ============================================================

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    frames = sorted(VELODYNE.glob("*.bin"))[:FRAME_LIMIT]

    if not frames:
        raise FileNotFoundError(
            f"No .bin frames found in {VELODYNE}"
        )

    object_rows = []
    runtime_rows = []

    for frame in tqdm(frames, desc="Evaluating frames"):
        stem = frame.stem

        label_path = LABELS / f"{stem}.txt"
        calib_path = CALIB / f"{stem}.txt"

        if not label_path.exists() or not calib_path.exists():
            print(f"Skipping {stem}: missing label or calibration")
            continue

        # Load original LiDAR points: x, y, z, reflectance.
        raw = np.fromfile(frame, dtype=np.float32)

        if raw.size % 4 != 0:
            print(f"Skipping {stem}: invalid LiDAR binary size")
            continue

        raw = raw.reshape(-1, 4)
        xyz = clean_points(raw[:, :3])

        if len(xyz) == 0:
            print(f"Skipping {stem}: no valid points")
            continue

        # Uniform baseline determines the common point budget.
        t0 = perf_counter()
        uniform = uniform_downsample(xyz)
        uniform_time = perf_counter() - t0

        target = len(uniform)

        # Transform original points to KITTI camera coordinates.
        rotation, translation = read_calibration(calib_path)

        original_cam = lidar_to_camera(
            xyz, rotation, translation
        )

        boxes = read_boxes(label_path)

        outputs = {"uniform": uniform}
        runtimes = {"uniform": uniform_time}

        # Run every adaptive/hybrid method, including V4.
        for method_name, sampler in METHODS.items():
            t0 = perf_counter()

            sampled = sampler(xyz, target)

            runtimes[method_name] = perf_counter() - t0
            outputs[method_name] = clean_points(sampled)

        # Record point counts, reduction and runtime.
        for method_name, sampled in outputs.items():
            runtime_rows.append({
                "frame": stem,
                "method": method_name,
                "original_points": len(xyz),
                "target_points": target,
                "output_points": len(sampled),
                "runtime_s": runtimes[method_name],
                "reduction_pct": (
                    100.0 * (1.0 - len(sampled) / len(xyz))
                    if len(xyz)
                    else 0.0
                ),
            })

        # Transform every sampled cloud into camera coordinates.
        camera_outputs = {
            name: lidar_to_camera(
                points, rotation, translation
            )
            for name, points in outputs.items()
        }

        # Calculate retention separately for every labeled object.
        for object_id, box in enumerate(boxes):
            original_count = int(
                points_in_box(original_cam, box).sum()
            )

            # Objects with no original LiDAR points cannot have
            # a meaningful retention percentage.
            if original_count == 0:
                continue

            for method_name, points_cam in camera_outputs.items():
                retained = int(
                    points_in_box(points_cam, box).sum()
                )

                retention_pct = (
                    100.0 * retained / original_count
                )

                object_rows.append({
                    "frame": stem,
                    "object_id": object_id,
                    "class": box["class"],
                    "method": method_name,
                    "original_box_points": original_count,
                    "retained_box_points": retained,
                    "retention_pct": retention_pct,
                    "target_points": target,
                    "output_points": len(outputs[method_name]),
                    "runtime_s": runtimes[method_name],
                })

    if not object_rows:
        raise RuntimeError(
            "No evaluable objects found. Check dataset paths, "
            "KITTI labels, calibration files and bounding boxes."
        )

    # ========================================================
    # Write per-object retention CSV
    # ========================================================

    with open(
        OBJECT_OUT, "w", newline="", encoding="utf-8"
    ) as f:
        writer = csv.DictWriter(
            f, fieldnames=list(object_rows[0].keys())
        )
        writer.writeheader()
        writer.writerows(object_rows)

    # ========================================================
    # Aggregate class-level retention statistics
    # ========================================================

    summary_rows = []

    all_methods = ["uniform", *METHODS.keys()]

    for cls in sorted(CLASSES):
        class_objects = [
            row for row in object_rows
            if row["class"] == cls
        ]

        for method in all_methods:
            values = np.array([
                row["retention_pct"]
                for row in class_objects
                if row["method"] == method
            ], dtype=np.float64)

            if len(values) == 0:
                continue

            summary_rows.append({
                "class": cls,
                "method": method,
                "object_count": len(values),
                "mean_retention_pct": float(np.mean(values)),
                "median_retention_pct": float(np.median(values)),
                "std_retention_pct": float(np.std(values)),
            })

    with open(
        SUMMARY_OUT, "w", newline="", encoding="utf-8"
    ) as f:
        writer = csv.DictWriter(
            f, fieldnames=list(summary_rows[0].keys())
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    # ========================================================
    # Write runtime and point-reduction CSV
    # ========================================================

    with open(
        RUNTIME_OUT, "w", newline="", encoding="utf-8"
    ) as f:
        writer = csv.DictWriter(
            f, fieldnames=list(runtime_rows[0].keys())
        )
        writer.writeheader()
        writer.writerows(runtime_rows)

    # ========================================================
    # Print retention summary
    # ========================================================

    print("\nCLASS RETENTION SUMMARY")
    print("=" * 90)

    print(
        f"{'Class':12} {'Method':24} {'Objects':>7} "
        f"{'Mean %':>10} {'Median %':>10} {'Std %':>10}"
    )

    for row in summary_rows:
        print(
            f"{row['class']:12} "
            f"{row['method']:24} "
            f"{row['object_count']:7d} "
            f"{row['mean_retention_pct']:10.2f} "
            f"{row['median_retention_pct']:10.2f} "
            f"{row['std_retention_pct']:10.2f}"
        )

    # ========================================================
    # Print runtime summary
    # ========================================================

    print("\nRUNTIME AND POINT REDUCTION SUMMARY")
    print("=" * 90)

    for method in all_methods:
        rows = [
            row for row in runtime_rows
            if row["method"] == method
        ]

        if not rows:
            continue

        times = np.array([
            row["runtime_s"] for row in rows
        ])

        reductions = np.array([
            row["reduction_pct"] for row in rows
        ])

        print(
            f"{method:24} "
            f"mean_time={np.mean(times):.3f}s  "
            f"median_time={np.median(times):.3f}s  "
            f"mean_reduction={np.mean(reductions):.2f}%"
        )

    print("\nSaved results:")
    print(f"Per-object retention: {OBJECT_OUT}")
    print(f"Class summary:        {SUMMARY_OUT}")
    print(f"Runtime summary:      {RUNTIME_OUT}")


if __name__ == "__main__":
    main()
