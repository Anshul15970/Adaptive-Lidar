from pathlib import Path
from collections import defaultdict
import numpy as np
from tqdm import tqdm

from trace_v5_selection import trace_selection
from class_retention_detector_clouds import (
    read_calibration, lidar_to_camera, read_boxes, points_in_box
)

ROOT = Path("/mnt/e/Projects/adaptive-lidar")
SOURCE = ROOT / "data/object_detection/training/velodyne"
LABELS = ROOT / "data/object_detection/training/label_2"
CALIB = ROOT / "data/object_detection/training/calib"
SPLIT = Path.home() / "adaptive-lidar-detector/OpenPCDet/data/kitti/ImageSets/val.txt"

FRAME_LIMIT = 50
VOXEL_SIZE = 0.8

BINS = [
    ("1-5", 1, 5),
    ("6-10", 6, 10),
    ("11-20", 11, 20),
    ("21-50", 21, 50),
    ("51+", 51, float("inf")),
]

CLASSES = ["Car", "Pedestrian", "Cyclist"]
DEPTHS = ["near", "medium", "far"]


def get_depth(z):
    if z < 20:
        return "near"
    if z < 40:
        return "medium"
    return "far"


def density_bin(n):
    for name, low, high in BINS:
        if low <= n <= high:
            return name
    return None


def main():
    ids = SPLIT.read_text().splitlines()[:FRAME_LIMIT]

    # category, depth, density -> counts
    totals = defaultdict(lambda: np.zeros(6, dtype=np.int64))

    for frame_id in tqdm(ids, desc="Voxel density analysis"):
        raw = np.fromfile(
            SOURCE / f"{frame_id}.bin", dtype=np.float32
        ).reshape(-1, 4)

        xyz = raw[:, :3]
        xyz = xyz[np.isfinite(xyz).all(axis=1)]

        rotation, translation = read_calibration(
            CALIB / f"{frame_id}.txt"
        )
        camera_xyz = lidar_to_camera(xyz, rotation, translation)
        boxes = read_boxes(LABELS / f"{frame_id}.txt")

        target = len(raw) // 5
        indices, stages = trace_selection(
            xyz, target, VOXEL_SIZE, random_state=0
        )

        selected_stage = np.zeros(len(xyz), dtype=np.int8)
        selected_stage[indices] = stages

        # Create independent category masks.
        masks = {}

        for cls in CLASSES:
            for depth in DEPTHS:
                masks[(cls, depth)] = np.zeros(len(xyz), dtype=bool)

        all_objects = np.zeros(len(xyz), dtype=bool)

        for box in boxes:
            cls = box["class"]
            if cls not in CLASSES:
                continue

            mask = points_in_box(camera_xyz, box)
            depth = get_depth(float(box["location"][2]))

            masks[(cls, depth)] |= mask
            all_objects |= mask

        masks[("Background", "all")] = ~all_objects

        # Build voxel membership once per frame.
        keys = np.floor(xyz / VOXEL_SIZE).astype(np.int32)
        unique_keys, inverse = np.unique(
            keys, axis=0, return_inverse=True
        )

        voxel_indices = [
            np.flatnonzero(inverse == i)
            for i in range(len(unique_keys))
        ]

        for category, mask in masks.items():
            cls, depth = category

            if not mask.any():
                continue

            for point_ids in voxel_indices:
                category_ids = point_ids[mask[point_ids]]

                count = len(category_ids)
                if count == 0:
                    continue

                bin_name = density_bin(count)
                if bin_name is None:
                    continue

                selected = selected_stage[category_ids]

                key = (cls, depth, bin_name)
                totals[key][0] += count
                totals[key][1] += np.sum(selected == 1)
                totals[key][2] += np.sum(selected == 2)
                totals[key][3] += np.sum(selected == 3)
                totals[key][4] += np.sum(selected == 4)
                totals[key][5] += 1

    print("\n========== RETENTION BY VOXEL DENSITY ==========")

    categories = [
        (cls, depth)
        for cls in CLASSES
        for depth in DEPTHS
    ] + [("Background", "all")]

    for cls, depth in categories:
        print(f"\n{'='*65}")
        print(f"{cls} | {depth}")
        print(
            f"{'Density':<10} {'Voxels':>8} {'Points':>10} "
            f"{'Rep%':>8} {'Sec%':>8} {'Fill%':>8} {'Final%':>9}"
        )

        for bin_name, _, _ in BINS:
            values = totals[(cls, depth, bin_name)]
            points, rep, sec, fill, fallback, voxels = values

            if points == 0:
                continue

            final = rep + sec + fill + fallback

            print(
                f"{bin_name:<10} {voxels:>8,} {points:>10,} "
                f"{100*rep/points:>7.2f}% "
                f"{100*sec/points:>7.2f}% "
                f"{100*fill/points:>7.2f}% "
                f"{100*final/points:>8.2f}%"
            )

    print("\nAnalysis complete.")


if __name__ == "__main__":
    main()
