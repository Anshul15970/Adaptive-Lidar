from pathlib import Path
import numpy as np
from tqdm import tqdm

from class_retention_detector_clouds import (
    load_xyz,
    read_calibration,
    lidar_to_camera,
    read_boxes,
    points_in_box,
)

ROOT = Path("/mnt/e/Projects/adaptive-lidar")

ORIGINAL = ROOT / "data/object_detection/training/velodyne"
LABELS = ROOT / "data/object_detection/training/label_2"
CALIB = ROOT / "data/object_detection/training/calib"

METHODS = {
    "uniform": ROOT / "data/detector_sampling/uniform/velodyne",
    "v5": ROOT / "data/detector_sampling/hybrid_v5/velodyne",
}

SPLIT = (
    Path.home() /
    "adaptive-lidar-detector/OpenPCDet/data/kitti/ImageSets/val.txt"
)

ids = SPLIT.read_text().split()[:50]

results = []

for frame_id in tqdm(ids, desc="Comparing Uniform and V5"):
    original = load_xyz(ORIGINAL / f"{frame_id}.bin")

    rotation, translation = read_calibration(
        CALIB / f"{frame_id}.txt"
    )

    original_cam = lidar_to_camera(
        original, rotation, translation
    )

    boxes = read_boxes(LABELS / f"{frame_id}.txt")

    sampled = {}
    for method, folder in METHODS.items():
        xyz = load_xyz(folder / f"{frame_id}.bin")
        sampled[method] = lidar_to_camera(
            xyz, rotation, translation
        )

    for box in boxes:
        original_mask = points_in_box(original_cam, box)
        original_count = int(original_mask.sum())

        if original_count == 0:
            continue

        z = float(box["location"][2])
        depth = (
            "near" if z < 20 else
            "medium" if z < 40 else
            "far"
        )

        row = {
            "class": box["class"],
            "depth": depth,
            "original": original_count,
        }

        for method in METHODS:
            retained = int(
                points_in_box(sampled[method], box).sum()
            )
            row[method] = retained

        results.append(row)

print("\n========== MATCHED 50-FRAME COMPARISON ==========")

for cls in ["Car", "Pedestrian", "Cyclist"]:
    for depth in ["near", "medium", "far"]:
        group = [
            r for r in results
            if r["class"] == cls and r["depth"] == depth
        ]

        if not group:
            continue

        original_sum = sum(r["original"] for r in group)

        print(
            f"\n{cls:12} {depth:8} "
            f"objects={len(group):4} "
            f"original_points={original_sum}"
        )

        for method in METHODS:
            retained_sum = sum(r[method] for r in group)
            weighted = 100 * retained_sum / original_sum

            object_values = np.array([
                100 * r[method] / r["original"]
                for r in group
            ])

            print(
                f"  {method:10} "
                f"weighted={weighted:7.2f}% "
                f"mean={object_values.mean():7.2f}% "
                f"median={np.median(object_values):7.2f}%"
            )
