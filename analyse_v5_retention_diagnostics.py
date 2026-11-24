from pathlib import Path
import csv

import numpy as np
from tqdm import tqdm

ROOT = Path("/mnt/e/Projects/adaptive-lidar")

ORIGINAL = ROOT / "data/object_detection/training/velodyne"
LABELS = ROOT / "data/object_detection/training/label_2"
CALIB = ROOT / "data/object_detection/training/calib"

METHODS = {
    "uniform_random": ROOT / "data/detector_sampling/uniform/velodyne",
    "hybrid_v4": ROOT / "data/detector_sampling/hybrid_v4/velodyne",
    "hybrid_v5": ROOT / "data/detector_sampling/hybrid_v5/velodyne",
}

VAL_IDS = (
    Path.home() / "adaptive-lidar-detector/OpenPCDet/data/kitti/ImageSets/val.txt"
)

OUT = ROOT / "results"
CLASSES = {"Car", "Pedestrian", "Cyclist"}

DISTANCE_BINS = {
    "near_0_20m": (0.0, 20.0),
    "medium_20_40m": (20.0, 40.0),
    "far_over_40m": (40.0, float("inf")),
}

THRESHOLDS = [10, 20, 40, 60]


def load_xyz(path):
    raw = np.fromfile(path, dtype=np.float32)

    if raw.size % 4 != 0:
        raise ValueError(f"Invalid XYZI file: {path}")

    xyz = raw.reshape(-1, 4)[:, :3]
    return xyz[np.isfinite(xyz).all(axis=1)]


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


def distance_group(z):
    for name, (lower, upper) in DISTANCE_BINS.items():
        if lower <= z < upper:
            return name
    return "outside_bins"


def write_csv(path, rows):
    if not rows:
        print(f"No rows to write: {path}")
        return

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    if not VAL_IDS.is_file():
        raise FileNotFoundError(f"Validation ID file not found: {VAL_IDS}")

    ids = VAL_IDS.read_text(encoding="utf-8").split()

    print("Validation frames:", len(ids))

    required_dirs = [ORIGINAL, LABELS, CALIB, *METHODS.values()]
    missing_dirs = [str(p) for p in required_dirs if not p.is_dir()]

    if missing_dirs:
        raise FileNotFoundError(
            "Missing directories:\n" + "\n".join(missing_dirs)
        )

    # Check every required file before beginning the analysis.
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
            print(" ", path)

        raise FileNotFoundError(f"{len(missing)} files are missing.")

    frame_rows = []
    object_rows = []

    # ---------------------------------------------------------
    # Main analysis
    # ---------------------------------------------------------

    for frame_id in tqdm(ids, desc="Analysing frames"):

        original_xyz = load_xyz(ORIGINAL / f"{frame_id}.bin")
        original_count = len(original_xyz)

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
            sampled_xyz = load_xyz(folder / f"{frame_id}.bin")

            sampled_counts[method] = len(sampled_xyz)

            sampled_cam[method] = lidar_to_camera(
                sampled_xyz, rotation, translation
            )

        # Global point retention for this frame.
        for method in METHODS:
            sampled_count = sampled_counts[method]

            frame_rows.append({
                "frame": frame_id,
                "method": method,
                "original_points": original_count,
                "sampled_points": sampled_count,
                "retention_pct": (
                    100.0 * sampled_count / original_count
                    if original_count else 0.0
                ),
                "difference_from_20pct": (
                    100.0 * sampled_count / original_count - 20.0
                    if original_count else 0.0
                ),
            })

        # Object-level retention.
        for object_id, box in enumerate(boxes):

            original_mask = points_in_box(original_cam, box)
            original_box_count = int(original_mask.sum())

            if original_box_count == 0:
                continue

            z = float(box["location"][2])
            group = distance_group(z)

            for method in METHODS:

                retained_count = int(
                    points_in_box(sampled_cam[method], box).sum()
                )

                retention = (
                    100.0 * retained_count / original_box_count
                )

                object_rows.append({
                    "frame": frame_id,
                    "object_id": object_id,
                    "class": box["class"],
                    "distance_z_m": z,
                    "distance_group": group,
                    "method": method,
                    "original_box_points": original_box_count,
                    "retained_box_points": retained_count,
                    "retention_pct": retention,
                })

    # ---------------------------------------------------------
    # 1. Global retention summary
    # ---------------------------------------------------------

    global_summary = []

    for method in METHODS:

        rows = [r for r in frame_rows if r["method"] == method]

        original_total = sum(r["original_points"] for r in rows)
        sampled_total = sum(r["sampled_points"] for r in rows)

        frame_values = np.array(
            [r["retention_pct"] for r in rows],
            dtype=np.float64,
        )

        global_summary.append({
            "method": method,
            "frames": len(rows),
            "original_points_total": original_total,
            "sampled_points_total": sampled_total,
            "overall_weighted_retention_pct": (
                100.0 * sampled_total / original_total
                if original_total else 0.0
            ),
            "mean_frame_retention_pct": float(frame_values.mean()),
            "median_frame_retention_pct": float(np.median(frame_values)),
            "minimum_frame_retention_pct": float(frame_values.min()),
            "maximum_frame_retention_pct": float(frame_values.max()),
            "frames_exactly_20pct": int(
                np.isclose(frame_values, 20.0, atol=1e-6).sum()
            ),
        })

    # ---------------------------------------------------------
    # 2. Distance-based object retention
    # ---------------------------------------------------------

    distance_summary = []

    for cls in sorted(CLASSES):
        for group in DISTANCE_BINS:
            for method in METHODS:

                rows = [
                    r for r in object_rows
                    if r["class"] == cls
                    and r["distance_group"] == group
                    and r["method"] == method
                ]

                if not rows:
                    continue

                values = np.array(
                    [r["retention_pct"] for r in rows],
                    dtype=np.float64,
                )

                original_total = sum(
                    r["original_box_points"] for r in rows
                )

                retained_total = sum(
                    r["retained_box_points"] for r in rows
                )

                distance_summary.append({
                    "class": cls,
                    "distance_group": group,
                    "method": method,
                    "object_count": len(rows),
                    "mean_retention_pct": float(values.mean()),
                    "median_retention_pct": float(np.median(values)),
                    "weighted_retention_pct": (
                        100.0 * retained_total / original_total
                        if original_total else 0.0
                    ),
                    "original_box_points": original_total,
                    "retained_box_points": retained_total,
                })

    # ---------------------------------------------------------
    # 3. Object-level retention distribution
    # ---------------------------------------------------------

    distribution_summary = []

    for cls in sorted(CLASSES):
        for method in METHODS:

            rows = [
                r for r in object_rows
                if r["class"] == cls and r["method"] == method
            ]

            if not rows:
                continue

            values = np.array(
                [r["retention_pct"] for r in rows],
                dtype=np.float64,
            )

            result = {
                "class": cls,
                "method": method,
                "object_count": len(rows),
            }

            for threshold in THRESHOLDS:
                below = int((values < threshold).sum())

                result[f"objects_below_{threshold}pct"] = below
                result[f"percentage_below_{threshold}pct"] = (
                    100.0 * below / len(values)
                )

            distribution_summary.append(result)

    # ---------------------------------------------------------
    # Save all outputs
    # ---------------------------------------------------------

    outputs = {
        "global_frame_retention_v5_diagnostics.csv": frame_rows,
        "global_retention_summary_v5_diagnostics.csv": global_summary,
        "distance_retention_v5_diagnostics.csv": distance_summary,
        "object_retention_distribution_v5_diagnostics.csv":
            distribution_summary,
        "object_retention_details_v5_diagnostics.csv": object_rows,
    }

    for filename, rows in outputs.items():
        path = OUT / filename
        write_csv(path, rows)
        print("Saved:", path)

    # ---------------------------------------------------------
    # Print summaries
    # ---------------------------------------------------------

    print("\n" + "=" * 90)
    print("1. GLOBAL POINT RETENTION")
    print("=" * 90)

    for r in global_summary:
        print(
            f"{r['method']:16} "
            f"overall={r['overall_weighted_retention_pct']:7.3f}% "
            f"mean_frame={r['mean_frame_retention_pct']:7.3f}% "
            f"median_frame={r['median_frame_retention_pct']:7.3f}% "
            f"min={r['minimum_frame_retention_pct']:7.3f}% "
            f"max={r['maximum_frame_retention_pct']:7.3f}% "
            f"exact_20%={r['frames_exactly_20pct']}/{r['frames']}"
        )

    print("\n" + "=" * 90)
    print("2. DISTANCE-BASED OBJECT RETENTION")
    print("=" * 90)

    for r in distance_summary:
        print(
            f"{r['class']:12} {r['distance_group']:16} "
            f"{r['method']:16} "
            f"objects={r['object_count']:5d} "
            f"mean={r['mean_retention_pct']:7.2f}% "
            f"median={r['median_retention_pct']:7.2f}% "
            f"weighted={r['weighted_retention_pct']:7.2f}%"
        )

    print("\n" + "=" * 90)
    print("3. OBJECT-LEVEL RETENTION DISTRIBUTION")
    print("=" * 90)

    for r in distribution_summary:
        print(
            f"{r['class']:12} {r['method']:16} "
            f"objects={r['object_count']:5d} "
            f"<10%={r['percentage_below_10pct']:6.2f}% "
            f"<20%={r['percentage_below_20pct']:6.2f}% "
            f"<40%={r['percentage_below_40pct']:6.2f}% "
            f"<60%={r['percentage_below_60pct']:6.2f}%"
        )

    print("\nAnalysis completed.")


if __name__ == "__main__":
    main()
