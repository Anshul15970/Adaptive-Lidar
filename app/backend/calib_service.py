from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union

import numpy as np

from .config import CALIB_DIR


class CalibService:
    """
    Parses KITTI calibration files and converts 3D coordinates between
    rectified camera space (camera 2) and LiDAR velodyne coordinate space.
    """

    def __init__(self, calib_dir: Path = CALIB_DIR):
        self.calib_dir = calib_dir
        self._calib_cache: Dict[str, Dict[str, np.ndarray]] = {}

    def get_calib_path(self, frame_id: Union[str, int]) -> Path:
        fid = f"{int(str(frame_id).split('.')[0]):06d}"
        return self.calib_dir / f"{fid}.txt"

    def read_calibration(self, frame_id: Union[str, int]) -> Dict[str, np.ndarray]:
        """
        Parse calibration matrices from file. Cached per frame.
        """
        fid = f"{int(str(frame_id).split('.')[0]):06d}"
        if fid in self._calib_cache:
            return self._calib_cache[fid]

        path = self.get_calib_path(fid)
        if not path.is_file():
            raise FileNotFoundError(f"Calibration file not found for frame {fid}: {path}")

        raw_values = {}
        with open(path, encoding="utf-8") as f:
            for line in f:
                if ":" not in line:
                    continue
                key, val = line.split(":", 1)
                arr = np.fromstring(val, sep=" ")
                if arr.size > 0:
                    raw_values[key.strip()] = arr

        if "Tr_velo_to_cam" not in raw_values:
            raise ValueError(f"Missing Tr_velo_to_cam in calibration: {path}")

        rect_key = "R0_rect" if "R0_rect" in raw_values else "R_rect"
        if rect_key not in raw_values:
            raise ValueError(f"Missing rectification matrix (R0_rect) in calibration: {path}")

        Tr_velo_to_cam = raw_values["Tr_velo_to_cam"].reshape(3, 4)
        R0_rect = raw_values[rect_key].reshape(3, 3)

        # Transformation: P_cam = R0_rect @ (Tr[:, :3] @ P_velo + Tr[:, 3])
        R_velo_to_cam = R0_rect @ Tr_velo_to_cam[:, :3]
        t_velo_to_cam = R0_rect @ Tr_velo_to_cam[:, 3]

        # Inverse transformation: P_velo = R_cam_to_velo @ (P_cam - t_velo_to_cam)
        R_cam_to_velo = np.linalg.inv(R_velo_to_cam)
        t_cam_to_velo = -R_cam_to_velo @ t_velo_to_cam

        calib_data = {
            "Tr_velo_to_cam": Tr_velo_to_cam,
            "R0_rect": R0_rect,
            "R_velo_to_cam": R_velo_to_cam,
            "t_velo_to_cam": t_velo_to_cam,
            "R_cam_to_velo": R_cam_to_velo,
            "t_cam_to_velo": t_cam_to_velo,
        }

        # Optional camera projections P2
        if "P2" in raw_values:
            calib_data["P2"] = raw_values["P2"].reshape(3, 4)

        self._calib_cache[fid] = calib_data
        return calib_data

    def cam_to_lidar(self, points_cam: np.ndarray, frame_id: Union[str, int]) -> np.ndarray:
        """
        Transform 3D points from rectified camera coordinates to LiDAR coordinates.
        points_cam: (N, 3) or (3,)
        """
        calib = self.read_calibration(frame_id)
        pts = np.atleast_2d(points_cam)
        # P_velo = (P_cam - t_velo_to_cam) @ R_cam_to_velo.T
        pts_velo = (pts - calib["t_velo_to_cam"]) @ calib["R_cam_to_velo"].T
        return pts_velo if points_cam.ndim == 2 else pts_velo[0]

    def lidar_to_cam(self, points_lidar: np.ndarray, frame_id: Union[str, int]) -> np.ndarray:
        """
        Transform 3D points from LiDAR coordinates to rectified camera coordinates.
        points_lidar: (N, 3) or (3,)
        """
        calib = self.read_calibration(frame_id)
        pts = np.atleast_2d(points_lidar)
        pts_cam = pts @ calib["R_velo_to_cam"].T + calib["t_velo_to_cam"]
        return pts_cam if points_lidar.ndim == 2 else pts_cam[0]

    @staticmethod
    def compute_box_3d_corners(
        center: np.ndarray,
        dimensions: np.ndarray,
        heading: float
    ) -> np.ndarray:
        """
        Compute the 8 corner vertices of an oriented 3D bounding box in LiDAR coordinates.
        center: [x, y, z] (box geometric centroid)
        dimensions: [dx (length), dy (width), dz (height)]
        heading: yaw angle around Z-axis in radians
        Returns: (8, 3) array of corner coordinates.
        """
        dx, dy, dz = dimensions
        # Canonical unit box corners around origin (bottom 4, top 4)
        x_corners = np.array([dx, dx, -dx, -dx, dx, dx, -dx, -dx]) / 2.0
        y_corners = np.array([dy, -dy, -dy, dy, dy, -dy, -dy, dy]) / 2.0
        z_corners = np.array([-dz, -dz, -dz, -dz, dz, dz, dz, dz]) / 2.0

        corners = np.vstack([x_corners, y_corners, z_corners])  # (3, 8)

        # 2D rotation matrix around Z-axis
        c = np.cos(heading)
        s = np.sin(heading)
        R = np.array([
            [c, -s, 0.0],
            [s,  c, 0.0],
            [0.0, 0.0, 1.0]
        ])

        rotated_corners = R @ corners  # (3, 8)
        corners_lidar = (rotated_corners.T + center)  # (8, 3)
        return corners_lidar

    @staticmethod
    def get_box_wireframe_lines(corners: np.ndarray) -> List[List[List[float]]]:
        """
        Generate 12 wireframe edge segments + 1 front orientation diagonal for rendering.
        corners: (8, 3) array
        Returns list of [[x1, y1, z1], [x2, y2, z2]] pairs.
        """
        edges = [
            # Bottom face
            (0, 1), (1, 2), (2, 3), (3, 0),
            # Top face
            (4, 5), (5, 6), (6, 7), (7, 4),
            # Vertical edges
            (0, 4), (1, 5), (2, 6), (3, 7),
            # Front face cross (between corners 0 and 5, 1 and 4) to indicate heading
            (4, 1), (5, 0),
        ]
        return [[corners[i].tolist(), corners[j].tolist()] for i, j in edges]

    def transform_ground_truth_box(
        self,
        box_cam: Dict[str, Any],
        frame_id: Union[str, int]
    ) -> Dict[str, Any]:
        """
        Convert a KITTI camera-frame 3D ground truth box to LiDAR coordinates.
        Input box_cam contains:
          - h, w, l: height, width, length
          - location: [x, y, z] (bottom center in camera frame)
          - rotation_y: yaw angle in camera frame
        Output:
          - center_lidar: [x, y, z] centroid in LiDAR space
          - dimensions_lidar: [dx, dy, dz] (l, w, h)
          - heading_lidar: yaw angle around Z-axis in radians
          - corners_lidar: (8, 3) list
          - lines_lidar: list of edge coordinate pairs
        """
        h = float(box_cam["h"])
        w = float(box_cam["w"])
        l = float(box_cam["l"])

        loc_cam_bottom = np.array(box_cam["location"], dtype=np.float64)
        # Shift bottom center to geometric center (in camera frame, y is downwards)
        loc_cam_center = loc_cam_bottom.copy()
        loc_cam_center[1] -= h / 2.0

        # Transform to LiDAR coordinates
        center_lidar = self.cam_to_lidar(loc_cam_center, frame_id)

        # In LiDAR: dx is length (l), dy is width (w), dz is height (h)
        dims_lidar = np.array([l, w, h], dtype=np.float64)

        # Convert rotation_y to LiDAR heading: theta_lidar = -rotation_y - pi/2
        rot_y = float(box_cam["rotation_y"])
        heading_lidar = -rot_y - (np.pi / 2.0)
        # Normalize to [-pi, pi]
        heading_lidar = float((heading_lidar + np.pi) % (2.0 * np.pi) - np.pi)

        # Compute 8 corners in LiDAR coordinates
        corners = self.compute_box_3d_corners(center_lidar, dims_lidar, heading_lidar)
        lines = self.get_box_wireframe_lines(corners)

        # Calculate distance from LiDAR sensor (Euclidean norm to origin)
        distance_m = float(np.linalg.norm(center_lidar[:2]))

        return {
            "center": [round(float(c), 3) for c in center_lidar],
            "dimensions": [round(float(d), 3) for d in dims_lidar],
            "heading": round(heading_lidar, 4),
            "distance_m": round(distance_m, 2),
            "corners": [[round(float(coord), 3) for coord in pt] for pt in corners],
            "lines": lines,
        }

