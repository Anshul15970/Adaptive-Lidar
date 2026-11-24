from pathlib import Path
import csv

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
    "v5": ROOT / "data/detector_sampling/hybrid_v5/velodyne",
    "v6_a00": ROOT / "data/detector_sampling/v6_alpha_0.00/velodyne",
    "v6_a025": ROOT / "data/detector_sampling/v6_alpha_0.25/velodyne",
    "v6_a050": ROOT / "data/detector_sampling/v6_alpha_0.50/velodyne",
    "v6_a075": ROOT / "data/detector_sampling/v6_alpha_0.75/velodyne",
    "v7_l025": ROOT / "data/detector_sampling/v7_lambda_0.25/velodyne",
    "v7_l050": ROOT / "data/detector_sampling/v7_lambda_0.50/velodyne",
    "v7_l075": ROOT / "data/detector_sampling/v7_lambda_0.75/velodyne",
}

SPLIT = (
    Path.home()
    / "adaptive-lidar-detector/OpenPCDet/data/kitti/ImageSets/val.txt"
)

CLASSES = ["Car", "Pedestrian", "Cyclist"]
DEPTHS = ["near", "medium", "far"]

OUTPUT = ROOT / "results/v5_v6_v7_retention_50.csv"


def main():

    frame_ids = SPLIT.read_text().split()[:50]

    # [method][class][depth] = [original, selected]
    totals = {
        method: {
            cls: {
                depth: [0, 0]
                for depth in DEPTHS
            }
            for cls in CLASSES
        }
        for method in METHODS
    }

    object_records = []

    for frame_id in tqdm(frame_ids, desc="Retention comparison"):

        raw = np.fromfile(
            ORIGINAL / f"{frame_id}.bin",
            dtype=np.float32
        ).reshape(-1, 4)

        xyz = raw[:, :3]
        finite = np.isfinite(xyz).all(axis=1)

        xyz = xyz[finite]

        rotation, translation = read_calibration(
            CALIB / f"{frame_id}.txt"
        )

        camera = lidar_to_camera(
            xyz, rotation, translation
        )

        boxes = read_boxes(
            LABELS / f"{frame_id}.txt"
        )

        original_masks = {
            cls: np.zeros(len(xyz), dtype=bool)
            for cls in CLASSES
        }

        for box in boxes:

            cls = box["class"]

            if cls not in CLASSES:
                continue

            mask = points_in_box(camera, box)

            original_masks[cls] |= mask

            original_count = int(mask.sum())

            if original_count == 0:
                continue

            distance = box["location"][2]

            if distance < 20:
                depth = "near"
            elif distance < 40:
                depth = "medium"
            else:
                depth = "far"

            record = {
                "frame_id": frame_id,
                "class": cls,
                "depth": depth,
                "original_points": original_count,
            }

            for method in METHODS:

                sampled = load_xyz(
                    METHODS[method] / f"{frame_id}.bin"
                )

                sampled_camera = lidar_to_camera(
                    sampled, rotation, translation
                )

                retained = int(
                    points_in_box(
                        sampled_camera, box
                    ).sum()
                )

                totals[method][cls][depth][0] += original_count
                totals[method][cls][depth][1] += retained

                record[method] = retained

            object_records.append(record)

    # Save object-level records.
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    object_output = (
        ROOT / "results/v5_v6_v7_object_retention_50.csv"
    )

    if object_records:

        with open(object_output, "w", newline="") as file:

            writer = csv.DictWriter(
                file,
                fieldnames=list(object_records[0].keys())
            )

            writer.writeheader()
            writer.writerows(object_records)

    # Print weighted point retention.
    print("\n========== WEIGHTED OBJECT POINT RETENTION ==========")

    for cls in CLASSES:

        print(f"\n{cls.upper()}")

        for depth in DEPTHS:

            print(f"\n  {depth.upper()}")

            for method in METHODS:

                original, selected = (
                    totals[method][cls][depth]
                )

                retention = (
                    100.0 * selected / original
                    if original else 0.0
                )

                print(
                    f"    {method:10} "
                    f"{retention:8.2f}% "
                    f"({selected:,}/{original:,})"
                )

    print("\nSaved object records:", object_output)


if __name__ == "__main__":
    main()
