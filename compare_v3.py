
from pathlib import Path
import csv
import time

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree
from tqdm import tqdm

from adaptive_sampling import adaptive_sample
from adaptive_sampling_v2 import adaptive_sample_v2
from adaptive_sampling_v3 import adaptive_sample_v3

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "object_detection" / "training"
VOXEL_SIZE = 0.8
FRAME_LIMIT = 100

def read_calibration(path):
    values = {}
    with open(path) as f:
        for line in f:
            if ":" in line:
                key, val = line.split(":", 1)
                try:
                    values[key] = np.fromstring(val, sep=" ")
                except ValueError:
                    pass

    transform = np.eye(4)
    transform[:3, :] = values["Tr_velo_to_cam"].reshape(3, 4)

    rect = np.eye(4)
    rect[:3, :3] = values["R0_rect"].reshape(3, 3)
    return rect @ transform

def to_camera(xyz, transform):
    homogeneous = np.column_stack((xyz, np.ones(len(xyz))))
    return (homogeneous @ transform.T)[:, :3]

def object_mask(points_cam, fields):
    cls = fields[0]
    if cls not in {"Car", "Pedestrian", "Cyclist"}:
        return None

    h, w, length = map(float, fields[8:11])
    x, y, z = map(float, fields[11:14])
    rot = float(fields[14])

    dx = points_cam[:, 0] - x
    dy = points_cam[:, 1] - y
    dz = points_cam[:, 2] - z

    lx = np.cos(rot) * dx - np.sin(rot) * dz
    lz = np.sin(rot) * dx + np.cos(rot) * dz

    mask = (
        (np.abs(lx) <= length / 2) &
        (dy <= 0) & (dy >= -h) &
        (np.abs(lz) <= w / 2)
    )
    return cls, mask

def main():
    velodyne = DATA / "velodyne"
    labels_dir = DATA / "label_2"
    calib_dir = DATA / "calib"

    files = sorted(velodyne.glob("*.bin"))[:FRAME_LIMIT]
    records = []

    for path in tqdm(files, desc="Comparing methods", unit="frame"):
        frame = path.stem
        labels_path = labels_dir / f"{frame}.txt"
        calib_path = calib_dir / f"{frame}.txt"

        if not labels_path.exists() or not calib_path.exists():
            continue

        xyz = np.fromfile(path, np.float32).reshape(-1, 4)[:, :3]

        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(xyz)

        start = time.perf_counter()
        uniform = np.asarray(
            cloud.voxel_down_sample(VOXEL_SIZE).points
        )
        uniform_time = time.perf_counter() - start

        budget = len(uniform)

        start = time.perf_counter()
        original_adaptive = adaptive_sample(xyz, budget)
        original_time = time.perf_counter() - start

        start = time.perf_counter()
        v2 = adaptive_sample_v2(xyz, budget)
        v2_time = time.perf_counter() - start

        start = time.perf_counter()
        v3 = adaptive_sample_v3(xyz, budget)
        v3_time = time.perf_counter() - start

        transform = read_calibration(calib_path)
        original_cam = to_camera(xyz, transform)

        methods = {
            "uniform": (uniform, uniform_time),
            "adaptive_v1": (original_adaptive, original_time),
            "adaptive_v2": (v2, v2_time),
            "adaptive_v3": (v3, v3_time),  
        }
        camera_points = {
            name: to_camera(points, transform)
            for name, (points, _) in methods.items()
        }

        # Reconstruction error: original points to nearest reduced point.
        for name, (reduced, elapsed) in methods.items():
            distances, _ = cKDTree(reduced).query(xyz, k=1)
            reconstruction_mean = float(distances.mean())

            with open(labels_path) as f:
                for line in f:
                    fields = line.split()
                    result = object_mask(original_cam, fields)
                    if result is None:
                        continue

                    cls, orig_mask = result
                    original_count = int(orig_mask.sum())
                    if original_count == 0:
                        continue

                    reduced_result = object_mask(
                        camera_points[name], fields
                    )
                    reduced_count = int(reduced_result[1].sum())

                    records.append({
                        "frame": frame,
                        "method": name,
                        "class": cls,
                        "original_object_points": original_count,
                        "retained_points": reduced_count,
                        "retention_pct":
                            100 * reduced_count / original_count,
                        "reconstruction_mean_m": reconstruction_mean,
                        "runtime_s": elapsed,
                    })

    output = ROOT / "results" / "compare_v2.csv"
    output.parent.mkdir(exist_ok=True)

    if not records:
        print("No labeled objects with LiDAR points found.")
        return

    with open(output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=records[0].keys())
        writer.writeheader()
        writer.writerows(records)

    print("\nAverage object retention by method:")
    for method in ("uniform", "adaptive_v1", "adaptive_v2", "adaptive_v3"):
        for cls in ("Car", "Pedestrian", "Cyclist"):
            rows = [
                r for r in records
                if r["method"] == method and r["class"] == cls
            ]
            if rows:
                avg = np.mean([r["retention_pct"] for r in rows])
                print(f"{method:12s} {cls:10s}: {avg:.2f}%")

    print(f"\nDetailed results saved to: {output}")

if __name__ == "__main__":
    main()
