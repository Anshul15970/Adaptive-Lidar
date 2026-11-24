
from pathlib import Path
from tqdm import TqdmKeyError, tqdm
import numpy as np
import open3d as o3d
import csv
from time import perf_counter
from adaptive_sampling import adaptive_sample

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "object_detection" / "training"
OUT = ROOT / "results" / "class_evaluation.csv"

VOXEL_SIZE = 0.8


def read_calibration(path):
    calib = {}
    with open(path, "r") as f:
        for line in f:
            if ":" not in line:
                continue
            key, value = line.split(":", 1)
            try:
                calib[key] = np.array(
                    [float(x) for x in value.split()]
                )
            except ValueError:
                pass

    Tr = calib["Tr_velo_to_cam"].reshape(3, 4)
    R0 = np.eye(4)
    R0[:3, :3] = calib["R0_rect"].reshape(3, 3)

    transform = np.eye(4)
    transform[:3, :] = Tr
    return R0 @ transform


def points_inside_box(points_cam, label):
    # KITTI label: type, ..., h, w, l, x, y, z, rotation_y
    fields = label.split()
    obj_class = fields[0]

    if obj_class not in {"Car", "Pedestrian", "Cyclist"}:
        return None

    h, w, length = map(float, fields[8:11])
    x, y, z = map(float, fields[11:14])
    rotation = float(fields[14])

    # Convert points into the object's local coordinates.
    dx = points_cam[:, 0] - x
    dy = points_cam[:, 1] - y
    dz = points_cam[:, 2] - z

    local_x = np.cos(rotation) * dx - np.sin(rotation) * dz
    local_z = np.sin(rotation) * dx + np.cos(rotation) * dz

    inside = (
        (np.abs(local_x) <= length / 2) &
        (np.abs(dy + h / 2) <= h / 2) &
        (np.abs(local_z) <= w / 2)
    )

    return obj_class, inside


def to_camera(points, transform):
    homogeneous = np.column_stack(
        [points[:, :3], np.ones(len(points))]
    )
    return (homogeneous @ transform.T)[:, :3]


def main():
    velodyne = DATA / "velodyne"
    labels_dir = DATA / "label_2"
    calib_dir = DATA / "calib"

    rows = []

    bin_files = sorted(velodyne.glob("*.bin"))[:100]
    
    for bin_path in tqdm(bin_files, desc="Evaluating frames", unit="frame"):
        frame = bin_path.stem
        label_path = labels_dir / f"{frame}.txt"
        calib_path = calib_dir / f"{frame}.txt"

        if not label_path.exists() or not calib_path.exists():
            continue

        points = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)
        xyz = points[:, :3]

        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(xyz)

        uniform = np.asarray(
            cloud.voxel_down_sample(VOXEL_SIZE).points
        )
        
        # Change this call if your adaptive_downsample signature differs.
        adaptive = adaptive_sample(xyz, len(uniform))
        adaptive = np.asarray(adaptive)[:, :3]

        transform = read_calibration(calib_path)
        original_cam = to_camera(xyz, transform)
        uniform_cam = to_camera(uniform, transform)
        adaptive_cam = to_camera(adaptive, transform)

        with open(label_path, "r") as f:
            for label in f:
                result = points_inside_box(original_cam, label)
                if result is None:
                    continue

                cls, original_mask = result
                original_count = int(original_mask.sum())

                if original_count == 0:
                    continue

                # Count reduced points that remain inside the same box.
                uniform_result = points_inside_box(uniform_cam, label)
                adaptive_result = points_inside_box(adaptive_cam, label)

                uniform_count = int(uniform_result[1].sum())
                adaptive_count = int(adaptive_result[1].sum())

                rows.append({
                    "frame": frame,
                    "class": cls,
                    "original_points": original_count,
                    "uniform_points": uniform_count,
                    "adaptive_points": adaptive_count,
                    "uniform_retention_pct":
                        100 * uniform_count / original_count,
                    "adaptive_retention_pct":
                        100 * adaptive_count / original_count,
                })

    OUT.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        print("No labeled objects with LiDAR points found.")
        return

    with open(OUT, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print(f"Objects evaluated: {len(rows)}")
    print(f"Results saved to: {OUT}")

    for cls in ("Car", "Pedestrian", "Cyclist"):
        selected = [r for r in rows if r["class"] == cls]
        if selected:
            print(
                f"{cls}: uniform retention "
                f"{np.mean([r['uniform_retention_pct'] for r in selected]):.2f}% | "
                f"adaptive retention "
                f"{np.mean([r['adaptive_retention_pct'] for r in selected]):.2f}%"
            )


if __name__ == "__main__":
    main()
