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

ORIGINAL = ROOT / "data/object_detection/training/velodyne"
LABELS = ROOT / "data/object_detection/training/label_2"
CALIB = ROOT / "data/object_detection/training/calib"

VAL_IDS = (
    Path.home() / "adaptive-lidar-detector/OpenPCDet/data/kitti/ImageSets/val.txt"
)

OUT_DIR = ROOT / "results"
DETAIL_OUT = OUT_DIR / "v5_stage_retention_50frames.csv"
SUMMARY_OUT = OUT_DIR / "v5_stage_retention_summary_50frames.csv"

FRAME_LIMIT = 50
VOXEL_SIZE = 0.8

STAGES = {
    1: "representatives",
    2: "secondary",
    3: "score_fill",
    4: "fallback",
}


def depth_bin(z):
    if z < 20:
        return "near"
    elif z < 40:
        return "medium"
    return "far"


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    ids = [
        x.strip()
        for x in VAL_IDS.read_text(encoding="utf-8").splitlines()
        if x.strip()
    ][:FRAME_LIMIT]

    rows = []

    for frame_id in tqdm(ids, desc="Analysing V5 stages"):
        original_path = ORIGINAL / f"{frame_id}.bin"

        original_xyz = load_xyz(original_path)

        rotation, translation = read_calibration(
            CALIB / f"{frame_id}.txt"
        )

        original_cam = lidar_to_camera(
            original_xyz, rotation, translation
        )

        boxes = read_boxes(LABELS / f"{frame_id}.txt")

        # Same target calculation as export_hybrid_v5.py.
        target = len(np.fromfile(
            original_path, dtype=np.float32
        )) // 20

        indices, stage_ids = trace_selection(
            original_xyz,
            target_points=target,
            voxel_size=VOXEL_SIZE,
            random_state=0,
        )

        if len(indices) != target:
            raise RuntimeError(
                f"{frame_id}: expected {target}, traced {len(indices)}"
            )

        # Build one mask per exclusive selection stage.
        stage_masks = {}

        for stage_id, stage_name in STAGES.items():
            mask = np.zeros(len(original_xyz), dtype=bool)
            mask[indices[stage_ids == stage_id]] = True
            stage_masks[stage_id] = mask

        cumulative_masks = {}
        cumulative = np.zeros(len(original_xyz), dtype=bool)

        for stage_id in STAGES:
            cumulative |= stage_masks[stage_id]
            cumulative_masks[stage_id] = cumulative.copy()

        for object_id, box in enumerate(boxes):
            original_mask = points_in_box(original_cam, box)
            original_count = int(original_mask.sum())

            if original_count == 0:
                continue

            stage_counts = {
                stage_id: int(
                    (original_mask & stage_masks[stage_id]).sum()
                )
                for stage_id in STAGES
            }

            cumulative_counts = {
                stage_id: int(
                    (original_mask & cumulative_masks[stage_id]).sum()
                )
                for stage_id in STAGES
            }

            row = {
                "frame": frame_id,
                "object_id": object_id,
                "class": box["class"],
                "depth_m": float(box["location"][2]),
                "depth_bin": depth_bin(float(box["location"][2])),
                "original_points": original_count,
            }

            for stage_id, stage_name in STAGES.items():
                exclusive = stage_counts[stage_id]
                cumulative_count = cumulative_counts[stage_id]

                row[f"{stage_name}_points"] = exclusive
                row[f"{stage_name}_cumulative_points"] = cumulative_count
                row[f"{stage_name}_cumulative_pct"] = (
                    100.0 * cumulative_count / original_count
                )

            rows.append(row)

    if not rows:
        raise RuntimeError("No labelled objects with LiDAR points found.")

    with open(DETAIL_OUT, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    # Aggregate by class and depth.
    summary = []

    groups = sorted({
        (r["class"], r["depth_bin"])
        for r in rows
    })

    for cls, depth in groups:
        group = [
            r for r in rows
            if r["class"] == cls and r["depth_bin"] == depth
        ]

        original_sum = sum(r["original_points"] for r in group)

        result = {
            "class": cls,
            "depth_bin": depth,
            "object_count": len(group),
            "original_box_points_sum": original_sum,
        }

        for stage_id, stage_name in STAGES.items():
            exclusive_sum = sum(
                r[f"{stage_name}_points"] for r in group
            )
            cumulative_sum = sum(
                r[f"{stage_name}_cumulative_points"] for r in group
            )

            result[f"{stage_name}_exclusive_pct"] = (
                100.0 * exclusive_sum / original_sum
                if original_sum else 0.0
            )

            result[f"{stage_name}_cumulative_pct"] = (
                100.0 * cumulative_sum / original_sum
                if original_sum else 0.0
            )

        final_values = np.array(
            [r["fallback_cumulative_pct"] for r in group],
            dtype=np.float64,
        )

        result["final_mean_object_retention_pct"] = float(
            final_values.mean()
        )
        result["final_median_object_retention_pct"] = float(
            np.median(final_values)
        )

        summary.append(result)

    with open(SUMMARY_OUT, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=list(summary[0].keys())
        )
        writer.writeheader()
        writer.writerows(summary)

    print("\n========== V5 STAGE RETENTION ==========")
    print("Frames analysed:", len(ids))
    print("Objects analysed:", len(rows))
    print()

    for row in summary:
        print(
            f"{row['class']:12} {row['depth_bin']:8} "
            f"objects={row['object_count']:4}"
        )

        for stage_id, stage_name in STAGES.items():
            print(
                f"  {stage_name:16} "
                f"exclusive={row[f'{stage_name}_exclusive_pct']:7.2f}% "
                f"cumulative={row[f'{stage_name}_cumulative_pct']:7.2f}%"
            )

        print(
            f"  FINAL OBJECT RETENTION "
            f"mean={row['final_mean_object_retention_pct']:.2f}% "
            f"median={row['final_median_object_retention_pct']:.2f}%"
        )
        print()

    print("Saved:")
    print(DETAIL_OUT)
    print(SUMMARY_OUT)


if __name__ == "__main__":
    main()
