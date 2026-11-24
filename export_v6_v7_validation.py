from pathlib import Path
import numpy as np

from v6_density_sampler import sample_v6_indices
from v7_score_density_sampler import sample_v7_indices


PROJECT_ROOT = Path("/mnt/e/Projects/adaptive-lidar")

SOURCE_DIR = (
    PROJECT_ROOT / "data/object_detection/training/velodyne"
)

VAL_FILE = (
    Path.home()
    / "adaptive-lidar-detector/OpenPCDet/data/kitti/ImageSets/val.txt"
)

OUTPUT_ROOT = PROJECT_ROOT / "data/detector_sampling"

METHODS = {
    "v6_alpha_0.25": "v6",
    "v7_lambda_0.25": "v7",
}

VOXEL_SIZE = 0.8
ALPHA = 0.25
LAMBDA = 0.25


def export_frame(frame_id, method, sampler):
    source_path = SOURCE_DIR / f"{frame_id}.bin"
    output_dir = OUTPUT_ROOT / method / "velodyne"
    output_path = output_dir / f"{frame_id}.bin"

    if not source_path.exists():
        raise FileNotFoundError(f"Missing source frame: {source_path}")

    raw = np.fromfile(source_path, dtype=np.float32)

    if raw.size % 4 != 0:
        raise ValueError(f"Invalid XYZI file: {source_path}")

    raw = raw.reshape(-1, 4)
    target = len(raw) // 5

    # Keep an existing export only if its point budget is correct.
    if output_path.exists():
        existing = np.fromfile(output_path, dtype=np.float32)

        if existing.size % 4 == 0 and existing.size // 4 == target:
            return "already_valid"

    xyz = raw[:, :3]

    if sampler == "v6":
        selected_indices = sample_v6_indices(
            xyz=xyz,
            target_points=target,
            alpha=ALPHA,
            voxel_size=VOXEL_SIZE,
        )

    elif sampler == "v7":
        selected_indices = sample_v7_indices(
            xyz=xyz,
            target_points=target,
            density_weight=LAMBDA,
            voxel_size=VOXEL_SIZE,
        )

    else:
        raise ValueError(f"Unknown sampler: {sampler}")

    # Recover the exact original XYZI rows using original indices.
    sampled_xyzi = raw[selected_indices]

    if len(sampled_xyzi) != target:
        raise RuntimeError(
            f"{frame_id}: expected {target}, got {len(sampled_xyzi)}"
        )

    if len(np.unique(selected_indices)) != target:
        raise RuntimeError(f"{frame_id}: duplicate indices detected")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Write atomically so interrupted exports cannot leave partial files.
    temporary_path = output_path.with_suffix(".bin.tmp")
    sampled_xyzi.astype(np.float32).tofile(temporary_path)
    temporary_path.replace(output_path)

    return "exported"


def main():
    frame_ids = [
        line.strip()
        for line in VAL_FILE.read_text().splitlines()
        if line.strip()
    ]

    if len(frame_ids) != len(set(frame_ids)):
        raise ValueError("Duplicate frame IDs in validation split")

    print(f"Validation frames: {len(frame_ids)}")
    print(f"Source directory: {SOURCE_DIR}")
    print(f"Voxel size: {VOXEL_SIZE}")
    print("Point budget: 20% (floor(original_points / 5))")

    for method, sampler in METHODS.items():
        print(f"\n{'=' * 65}")
        print(f"Exporting {method}")
        print(f"{'=' * 65}")

        exported = 0
        skipped = 0

        for number, frame_id in enumerate(frame_ids, start=1):
            status = export_frame(frame_id, method, sampler)

            if status == "exported":
                exported += 1
            else:
                skipped += 1

            if number % 25 == 0 or number == len(frame_ids):
                print(
                    f"[{number}/{len(frame_ids)}] "
                    f"Exported this run: {exported} | "
                    f"Already valid: {skipped} | "
                    f"Latest: {frame_id}"
                )

        print(f"\n{method} complete.")
        print(f"Newly exported: {exported}")
        print(f"Already valid: {skipped}")


if __name__ == "__main__":
    main()
