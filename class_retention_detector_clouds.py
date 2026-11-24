from pathlib import Path
import csv

import numpy as np
from tqdm import tqdm

ROOT = Path("/mnt/e/Projects/adaptive-lidar")

ORIGINAL = ROOT / "data/object_detection/training/velodyne"
LABELS = ROOT / "data/object_detection/training/label_2"
CALIB = ROOT / "data/object_detection/training/calib"

UNIFORM = ROOT / "data/detector_sampling/uniform/velodyne"
HYBRID_V4 = ROOT / "data/detector_sampling/hybrid_v4/velodyne"
HYBRID_V5 = ROOT / "data/detector_sampling/hybrid_v5/velodyne"

VAL_IDS = (
    Path.home() / "adaptive-lidar-detector/OpenPCDet/data/kitti/ImageSets/val.txt"
)

OUT_DIR = ROOT / "results"

# Separate files to preserve previous results
OBJECT_OUT = OUT_DIR / "class_retention_detector_clouds_v5_comparison.csv"
SUMMARY_OUT = OUT_DIR / "class_retention_detector_summary_v5_comparison.csv"

CLASSES = {"Car", "Pedestrian", "Cyclist"}

METHODS = {
    "uniform_random": UNIFORM,
    "hybrid_v4": HYBRID_V4,
    "hybrid_v5": HYBRID_V5,
}


def load_xyz(path):
    raw = np.fromfile(path, dtype=np.float32)

    if raw.size % 4 != 0:
        raise ValueError(f"Invalid XYZI binary file: {path}")

    points = raw.reshape(-1, 4)[:, :3]
    return points[np.isfinite(points).all(axis=1)]


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
        raise ValueError(f"Missing Tr_velo_to_cam: {path}")

    rect_key = "R0_rect" if "R0_rect" in values else "R_rect"
    if rect_key not in values:
        raise ValueError(f"Missing rectification matrix: {path}")

    Tr = values["Tr_velo_to_cam"].reshape(3, 4)
    R0 = values[rect_key].reshape(3, 3)

    rotation = R0 @ Tr[:, :3]
    translation = R0 @ Tr[:, 3]
    return rotation, translation


def lidar_to_camera(xyz, rotation, translation):
    return xyz @ rotation.T + translation


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
                    [float(p[11]), float(p[12]), float(p[13])],
                    dtype=np.float32,
                ),
                "rotation_y": float(p[14]),
            })

    return boxes


def points_in_box(points_cam, box):
    # KITTI 3D box location is its bottom center.
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


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    ids = VAL_IDS.read_text(encoding="utf-8").split()
    if len(ids) != 3769:
        print(f"Warning: expected 3769 validation IDs, found {len(ids)}")

    # Check all required files before doing the full analysis.
    required_dirs = [ORIGINAL, LABELS, CALIB, *METHODS.values()]
    missing_dirs = [str(p) for p in required_dirs if not p.is_dir()]
    if missing_dirs:
        raise FileNotFoundError(
            "Required directories are missing:\n" + "\n".join(missing_dirs)
        )

    missing = []
    for frame_id in ids:
        paths = [
            ORIGINAL / f"{frame_id}.bin",
            LABELS / f"{frame_id}.txt",
            CALIB / f"{frame_id}.txt",
            *(folder / f"{frame_id}.bin" for folder in METHODS.values()),
        ]
        for path in paths:
            if not path.is_file():
                missing.append(str(path))

    if missing:
        print("Missing files (first 20):")
        for path in missing[:20]:
            print("  ", path)
        raise FileNotFoundError(
            f"{len(missing)} required files are missing. Fix these paths first."
        )

    object_rows = []
    total_frames = 0

    for frame_id in tqdm(ids, desc="Analysing validation frames"):
        original_xyz = load_xyz(ORIGINAL / f"{frame_id}.bin")
        rotation, translation = read_calibration(
            CALIB / f"{frame_id}.txt"
        )
        original_cam = lidar_to_camera(
            original_xyz, rotation, translation
        )
        boxes = read_boxes(LABELS / f"{frame_id}.txt")

        sampled_cam = {}
        sampled_counts = {}

        for method, folder in METHODS.items():
            xyz = load_xyz(folder / f"{frame_id}.bin")
            sampled_counts[method] = len(xyz)
            sampled_cam[method] = lidar_to_camera(
                xyz, rotation, translation
            )

        total_frames += 1

        for object_id, box in enumerate(boxes):
            original_mask = points_in_box(original_cam, box)
            original_count = int(original_mask.sum())

            if original_count == 0:
                continue

            for method in METHODS:
                retained = int(
                    points_in_box(sampled_cam[method], box).sum()
                )

                object_rows.append({
                    "frame": frame_id,
                    "object_id": object_id,
                    "class": box["class"],
                    "method": method,
                    "original_box_points": original_count,
                    "retained_box_points": retained,
                    "retention_pct": 100.0 * retained / original_count,
                    "original_frame_points": len(original_xyz),
                    "sampled_frame_points": sampled_counts[method],
                    "frame_retention_pct": (
                        100.0 * sampled_counts[method] / len(original_xyz)
                        if len(original_xyz) else 0.0
                    ),
                })

    if not object_rows:
        raise RuntimeError("No labelled objects with original LiDAR points found.")

    with open(OBJECT_OUT, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(object_rows[0].keys()))
        writer.writeheader()
        writer.writerows(object_rows)

    summary_rows = []

    for cls in sorted(CLASSES):
        for method in METHODS:
            rows = [
                r for r in object_rows
                if r["class"] == cls and r["method"] == method
            ]
            if not rows:
                continue

            values = np.array(
                [r["retention_pct"] for r in rows], dtype=np.float64
            )
            original_sum = sum(r["original_box_points"] for r in rows)
            retained_sum = sum(r["retained_box_points"] for r in rows)

            summary_rows.append({
                "class": cls,
                "method": method,
                "object_count": len(rows),
                "mean_object_retention_pct": float(values.mean()),
                "median_object_retention_pct": float(np.median(values)),
                "weighted_point_retention_pct": (
                    100.0 * retained_sum / original_sum
                    if original_sum else 0.0
                ),
                "original_box_points_sum": original_sum,
                "retained_box_points_sum": retained_sum,
            })

    with open(SUMMARY_OUT, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"\nFrames analysed: {total_frames}")
    print("\nCLASS-WISE OBJECT POINT RETENTION")
    print("-" * 100)
    for row in summary_rows:
        print(
            f"{row['class']:12} {row['method']:16} "
            f"objects={row['object_count']:5d} "
            f"mean={row['mean_object_retention_pct']:7.2f}% "
            f"median={row['median_object_retention_pct']:7.2f}% "
            f"weighted={row['weighted_point_retention_pct']:7.2f}%"
        )

    print("\nSaved files:")
    print(OBJECT_OUT)
    print(SUMMARY_OUT)


if __name__ == "__main__":
    main()
