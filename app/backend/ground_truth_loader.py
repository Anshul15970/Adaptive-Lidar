from pathlib import Path
from typing import Dict, Any, List, Optional, Set, Tuple, Union

import numpy as np

from .config import LABEL_DIR, SUPPORTED_CLASSES, DISTANCE_BINS
from .calib_service import CalibService


class GroundTruthLoader:
    """
    Parses KITTI ground-truth 3D bounding boxes from label_2,
    transforms them into the LiDAR coordinate frame via CalibService,
    and supports computing point-in-box containment for evaluation.
    """

    def __init__(self, label_dir: Path = LABEL_DIR, calib_service: Optional[CalibService] = None):
        self.label_dir = label_dir
        self.calib_service = calib_service or CalibService()
        self._labels_cache: Dict[str, List[Dict[str, Any]]] = {}

    def get_label_path(self, frame_id: Union[str, int]) -> Path:
        fid = f"{int(str(frame_id).split('.')[0]):06d}"
        return self.label_dir / f"{fid}.txt"

    @staticmethod
    def get_distance_bin(distance_z_m: float) -> str:
        """
        Classify depth distance into standard research bins:
        near_0_20m, medium_20_40m, far_over_40m
        """
        for bin_name, (low, high) in DISTANCE_BINS.items():
            if low <= distance_z_m < high:
                return bin_name
        return "outside_bins"

    @staticmethod
    def points_in_box_mask(
        points_lidar: np.ndarray,
        center: np.ndarray,
        dimensions: np.ndarray,
        heading: float
    ) -> np.ndarray:
        """
        Vectorized test of whether points in LiDAR frame lie inside an oriented 3D box.
        points_lidar: (N, 3) or (N, 4)
        center: [x, y, z] centroid
        dimensions: [dx, dy, dz] (length, width, height)
        heading: yaw angle in radians
        Returns: boolean mask (N,)
        """
        pts = points_lidar[:, :3]
        dx, dy, dz = dimensions
        delta = pts - center

        # Rotate points to box-local coordinate frame
        c = np.cos(-heading)
        s = np.sin(-heading)

        local_x = c * delta[:, 0] - s * delta[:, 1]
        local_y = s * delta[:, 0] + c * delta[:, 1]
        local_z = delta[:, 2]

        return (
            (np.abs(local_x) <= dx / 2.0)
            & (np.abs(local_y) <= dy / 2.0)
            & (np.abs(local_z) <= dz / 2.0)
        )

    def load_ground_truth(
        self,
        frame_id: Union[str, int],
        filter_classes: Optional[List[str]] = None,
        points_lidar: Optional[np.ndarray] = None
    ) -> List[Dict[str, Any]]:
        """
        Load ground truth annotations for a frame, converted to LiDAR space.
        filter_classes: list of class names (defaults to ['Car', 'Pedestrian', 'Cyclist'])
        points_lidar: optional point array to compute points-in-box counts
        """
        fid = f"{int(str(frame_id).split('.')[0]):06d}"
        path = self.get_label_path(fid)

        if not path.is_file():
            raise FileNotFoundError(f"Label file not found for frame {fid}: {path}")

        target_classes: Set[str] = set(filter_classes or SUPPORTED_CLASSES)

        raw_objects = []
        with open(path, encoding="utf-8") as f:
            for idx, line in enumerate(f):
                parts = line.strip().split()
                if not parts:
                    continue

                obj_type = parts[0]
                if target_classes and obj_type not in target_classes:
                    continue

                # Camera frame parameters
                truncated = float(parts[1])
                occluded = int(parts[2])
                alpha = float(parts[3])
                bbox_2d = [float(parts[4]), float(parts[5]), float(parts[6]), float(parts[7])]
                h, w, l = float(parts[8]), float(parts[9]), float(parts[10])
                loc_cam = [float(parts[11]), float(parts[12]), float(parts[13])]
                rot_y = float(parts[14])

                # Depth distance in camera frame (z-axis forward)
                depth_z = loc_cam[2]
                dist_bin = self.get_distance_bin(depth_z)

                box_cam = {
                    "h": h, "w": w, "l": l,
                    "location": loc_cam,
                    "rotation_y": rot_y
                }

                # Transform to LiDAR coordinates
                box_lidar = self.calib_service.transform_ground_truth_box(box_cam, fid)

                # Optional point-in-box count
                point_count = None
                if points_lidar is not None and len(points_lidar) > 0:
                    mask = self.points_in_box_mask(
                        points_lidar,
                        np.array(box_lidar["center"]),
                        np.array(box_lidar["dimensions"]),
                        box_lidar["heading"]
                    )
                    point_count = int(mask.sum())

                raw_objects.append({
                    "object_id": idx,
                    "class": obj_type,
                    "distance_z_m": round(depth_z, 2),
                    "distance_group": dist_bin,
                    "truncated": truncated,
                    "occluded": occluded,
                    "alpha": round(alpha, 3),
                    "bbox_2d": bbox_2d,
                    "box_3d": box_lidar,
                    "contained_points": point_count,
                })

        return raw_objects

