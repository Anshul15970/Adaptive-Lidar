from pathlib import Path
import csv
import numpy as np
from tqdm import tqdm

from trace_v5_selection import trace_selection
from class_retention_detector_clouds import (
    load_xyz,
    read_calibration,
    lidar_to_camera,
    read_boxes,
    points_in_box,
)

ROOT = Path("/mnt/e/Projects/adaptive-lidar")

SOURCE = ROOT / "data/object_detection/training/velodyne"
LABELS = ROOT / "data/object_detection/training/label_2"
CALIB = ROOT / "data/object_detection/training/calib"

SPLIT = (
    Path.home()
    / "adaptive-lidar-detector/OpenPCDet/data/kitti/ImageSets/val.txt"
)

OUTPUT = ROOT / "results/v5_voxel_allocation_50frames.csv"

FRAME_LIMIT = 50
VOXEL_SIZE = 0.8

CLASSES = ["Car", "Pedestrian", "Cyclist"]
DEPTHS = ["near", "medium", "far"]

STAGES = {
    1: "representative",
    2: "secondary",
    3: "score_fill",
    4: "fallback",
}


def depth_group(z):
    if z < 20:
        return "near"
    elif z < 40:
        return "medium"
    return "far"


def main():
    frame_ids = [
        x.strip()
        for x in SPLIT.read_text().splitlines()
        if x.strip()
    ][:FRAME_LIMIT]

    rows = []

    for frame_id in tqdm(frame_ids, desc="Analysing V5 allocation"):

        original_path = SOURCE / f"{frame_id}.bin"
        label_path = LABELS / f"{frame_id}.txt"
        calib_path = CALIB / f"{frame_id}.txt"

        if not all(p.exists() for p in [
            original_path, label_path, calib_path
        ]):
            continue

        raw = np.fromfile(original_path, dtype=np.float32).reshape(-1, 4)
        xyz = raw[:, :3]

        finite = np.isfinite(xyz).all(axis=1)
        xyz = xyz[finite]

        target = len(raw) // 5

        indices, stages = trace_selection(
            xyz,
            target,
            voxel_size=VOXEL_SIZE,
            random_state=0
        )

        # Stage masks refer to the finite XYZ array.
        selected_stage = np.zeros(len(xyz), dtype=np.int8)
        selected_stage[indices] = stages

        rotation, translation = read_calibration(calib_path)

        original_cam = lidar_to_camera(xyz, rotation, translation)

        boxes = read_boxes(label_path)

        # Assign points to GT object categories for diagnostic purposes.
        # Overlapping boxes are assigned to the first matching object.
        object_category = np.full(len(xyz), "background", dtype=object)

        for box in boxes:
            cls = box["class"]
            mask = points_in_box(original_cam, box)

            if cls not in CLASSES:
                continue

            # Keep first assignment if GT boxes overlap.
            available = mask & (object_category == "background")
            object_category[available] = cls

        # Build groups for each class and distance.
        groups = {}

        for cls in CLASSES:
            for depth in DEPTHS:
                groups[(cls, depth)] = np.zeros(len(xyz), dtype=bool)

        for box in boxes:
            cls = box["class"]

            if cls not in CLASSES:
                continue

            depth = depth_group(float(box["location"][2]))
            mask = points_in_box(original_cam, box)

            groups[(cls, depth)] |= mask

        # Background = points not inside any evaluated GT object box.
        all_object_mask = np.zeros(len(xyz), dtype=bool)

        for mask in groups.values():
            all_object_mask |= mask

        groups[("Background", "all")] = ~all_object_mask

        for (cls, depth), mask in groups.items():

            original_count = int(mask.sum())

            if original_count == 0:
                continue

            selected = selected_stage[mask]

            stage_counts = {
                stage: int(np.sum(selected == stage))
                for stage in STAGES
            }

            retained = int(np.sum(selected > 0))

            # Count unique occupied voxels containing category points.
            voxel_keys = np.floor(
                xyz[mask] / VOXEL_SIZE
            ).astype(np.int32)

            occupied_voxels = len(np.unique(voxel_keys, axis=0))

            # Count occupied voxels containing at least one selected point.
            selected_points = xyz[mask][selected > 0]

            if len(selected_points):
                selected_keys = np.floor(
                    selected_points / VOXEL_SIZE
                ).astype(np.int32)

                selected_voxels = len(
                    np.unique(selected_keys, axis=0)
                )
            else:
                selected_voxels = 0

            rows.append({
                "frame_id": frame_id,
                "category": cls,
                "depth": depth,
                "original_points": original_count,
                "occupied_voxels": occupied_voxels,
                "selected_voxels": selected_voxels,
                "representative": stage_counts[1],
                "secondary": stage_counts[2],
                "score_fill": stage_counts[3],
                "fallback": stage_counts[4],
                "retained_points": retained,
            })

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        raise RuntimeError("No diagnostic rows were generated.")

    with OUTPUT.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print("\n========== V5 VOXEL ALLOCATION ==========")

    categories = [
        ("Car", DEPTHS),
        ("Pedestrian", DEPTHS),
        ("Cyclist", DEPTHS),
        ("Background", ["all"]),
    ]

    for cls, depths in categories:
        for depth in depths:

            group = [
                row for row in rows
                if row["category"] == cls
                and row["depth"] == depth
            ]

            if not group:
                continue

            original = sum(r["original_points"] for r in group)
            voxels = sum(r["occupied_voxels"] for r in group)
            selected_voxels = sum(r["selected_voxels"] for r in group)

            reps = sum(r["representative"] for r in group)
            secondary = sum(r["secondary"] for r in group)
            fill = sum(r["score_fill"] for r in group)
            fallback = sum(r["fallback"] for r in group)
            retained = sum(r["retained_points"] for r in group)

            print(f"\n{cls} | {depth}")
            print(f"Frames/objects with points: {len(group)}")
            print(f"Original points: {original:,}")
            print(f"Occupied voxels: {voxels:,}")
            print(f"Selected occupied voxels: {selected_voxels:,}")
            print(f"Average points per occupied voxel: {original / max(voxels, 1):.2f}")

            print(f"Representatives: {reps:,} ({100*reps/original:.2f}%)")
            print(f"Secondary:      {secondary:,} ({100*secondary/original:.2f}%)")
            print(f"Score fill:     {fill:,} ({100*fill/original:.2f}%)")
            print(f"Fallback:       {fallback:,} ({100*fallback/original:.2f}%)")
            print(f"Final retained: {retained:,} ({100*retained/original:.2f}%)")

    print("\nSaved:", OUTPUT)


if __name__ == "__main__":
    main()
