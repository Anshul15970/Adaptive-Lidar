import re
from collections import OrderedDict
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union

import numpy as np

from .config import (
    METHOD_DIRECTORIES,
    METHOD_ALIASES,
    METHOD_METADATA,
    VAL_FILE_CANDIDATES,
    LRU_CACHE_CAPACITY,
    DEFAULT_MAX_RENDER_POINTS,
)


class KittiLoader:
    """
    Robust point-cloud loader for KITTI binary XYZI velodyne frames.
    Handles original and sampled frames across all research methods.
    """

    def __init__(self, cache_capacity: int = LRU_CACHE_CAPACITY):
        self._cache_capacity = cache_capacity
        self._cache: OrderedDict[Tuple[str, str], np.ndarray] = OrderedDict()
        self._val_frame_ids: Optional[List[str]] = None

    @staticmethod
    def normalize_frame_id(frame_id: Union[str, int]) -> str:
        """
        Normalize frame ID to 6-digit zero-padded string (e.g. 1 -> '000001').
        """
        if isinstance(frame_id, int):
            return f"{frame_id:06d}"
        
        s = str(frame_id).strip()
        if s.isdigit():
            return f"{int(s):06d}"
        
        # If it contains extension like '000001.bin'
        stem = Path(s).stem
        if stem.isdigit():
            return f"{int(stem):06d}"
            
        raise ValueError(f"Invalid frame ID: '{frame_id}'. Must be a 6-digit number or integer.")

    @staticmethod
    def normalize_method(method: str) -> str:
        """
        Normalize method identifier and resolve aliases.
        """
        m = method.strip().lower()
        if m in METHOD_ALIASES:
            m = METHOD_ALIASES[m]
        
        if m not in METHOD_DIRECTORIES:
            valid = list(METHOD_DIRECTORIES.keys()) + list(METHOD_ALIASES.keys())
            raise ValueError(f"Unknown sampling method: '{method}'. Valid options: {valid}")
        
        return m

    def get_validation_frame_ids(self) -> List[str]:
        """
        Retrieve the exact 3,769 validation split frame IDs.
        Checks candidate validation split paths first, falling back to 
        the v6/v7 directory contents if val.txt is not accessible.
        """
        if self._val_frame_ids is not None:
            return self._val_frame_ids

        # 1. Try reading val.txt candidate paths
        for candidate in VAL_FILE_CANDIDATES:
            try:
                if candidate.is_file():
                    content = candidate.read_text(encoding="utf-8").strip()
                    ids = [self.normalize_frame_id(line) for line in content.split() if line.strip()]
                    if len(ids) == 3769:
                        self._val_frame_ids = ids
                        return self._val_frame_ids
            except Exception:
                continue

        # 2. Fallback: inspect v6 directory which has exact 3,769 validation frames
        v6_dir = METHOD_DIRECTORIES.get("v6_alpha_0.25")
        if v6_dir and v6_dir.is_dir():
            ids = []
            for path in sorted(v6_dir.glob("*.bin")):
                if path.stem.isdigit() and len(path.stem) == 6:
                    ids.append(path.stem)
            if len(ids) == 3769:
                self._val_frame_ids = ids
                return self._val_frame_ids

        raise FileNotFoundError(
            "Could not locate the 3,769 KITTI validation frame IDs. "
            "Please ensure val.txt or data/detector_sampling/v6_alpha_0.25/velodyne is accessible."
        )

    def get_frame_path(self, frame_id: Union[str, int], method: str) -> Path:
        """
        Resolve the file path for a given frame ID and method.
        """
        fid = self.normalize_frame_id(frame_id)
        m = self.normalize_method(method)
        folder = METHOD_DIRECTORIES[m]
        return folder / f"{fid}.bin"

    def is_frame_available(self, frame_id: Union[str, int], method: str) -> bool:
        """
        Check if a frame file exists on disk and is non-empty.
        """
        try:
            path = self.get_frame_path(frame_id, method)
            return path.is_file() and path.stat().st_size > 0
        except Exception:
            return False

    def load_raw_points(self, frame_id: Union[str, int], method: str) -> np.ndarray:
        """
        Load point cloud as (N, 4) float32 numpy array [x, y, z, intensity].
        Validates file existence, byte size divisibility, and data finiteness.
        Maintains an in-memory LRU cache of recent frames.
        """
        fid = self.normalize_frame_id(frame_id)
        m = self.normalize_method(method)
        cache_key = (fid, m)

        # Check LRU cache
        if cache_key in self._cache:
            self._cache.move_to_end(cache_key)
            return self._cache[cache_key]

        path = self.get_frame_path(fid, m)
        if not path.is_file():
            raise FileNotFoundError(f"Point cloud file not found: {path}")

        file_size = path.stat().st_size
        if file_size == 0:
            raise ValueError(f"Point cloud file is empty: {path}")

        # KITTI format: float32 quadruplets (x, y, z, intensity) = 16 bytes per point
        if file_size % 16 != 0:
            raise ValueError(
                f"Invalid KITTI binary file dimensions in {path}: "
                f"size {file_size} is not divisible by 16 bytes (4 x float32)."
            )

        raw = np.fromfile(path, dtype=np.float32)
        points = raw.reshape(-1, 4)

        if not np.isfinite(points).all():
            # Filter non-finite points if any
            finite_mask = np.isfinite(points).all(axis=1)
            points = points[finite_mask]

        # Enforce LRU cache eviction
        if len(self._cache) >= self._cache_capacity:
            self._cache.popitem(last=False)
        self._cache[cache_key] = points

        return points

    @staticmethod
    def downsample_for_rendering(
        points: np.ndarray,
        max_points: Optional[int] = DEFAULT_MAX_RENDER_POINTS
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Deterministically downsample points if exceeding max_points for browser WebGL rendering.
        Always preserves and reports the true original/sampled point counts in the metadata.
        """
        total_points = len(points)
        if max_points is not None and max_points > 0 and total_points > max_points:
            # Deterministic, spatially uniform striding
            indices = np.linspace(0, total_points - 1, max_points, dtype=np.int64)
            rendered = points[indices]
            meta = {
                "total_points": total_points,
                "rendered_points": len(rendered),
                "is_downsampled": True,
                "sampling_ratio": round(len(rendered) / total_points, 4),
                "render_cap": max_points,
            }
            return rendered, meta
        else:
            meta = {
                "total_points": total_points,
                "rendered_points": total_points,
                "is_downsampled": False,
                "sampling_ratio": 1.0,
                "render_cap": max_points,
            }
            return points, meta

    def get_frame_points(
        self,
        frame_id: Union[str, int],
        method: str,
        max_points: Optional[int] = DEFAULT_MAX_RENDER_POINTS
    ) -> Dict[str, Any]:
        """
        Get structured point cloud data including coordinates, intensity,
        bounding box bounds, and rendering downsample metadata.
        """
        fid = self.normalize_frame_id(frame_id)
        m = self.normalize_method(method)
        raw_pts = self.load_raw_points(fid, m)

        rendered_pts, render_meta = self.downsample_for_rendering(raw_pts, max_points)

        # Compute bounding limits
        if len(rendered_pts) > 0:
            bounds = {
                "x_min": float(np.min(rendered_pts[:, 0])),
                "x_max": float(np.max(rendered_pts[:, 0])),
                "y_min": float(np.min(rendered_pts[:, 1])),
                "y_max": float(np.max(rendered_pts[:, 1])),
                "z_min": float(np.min(rendered_pts[:, 2])),
                "z_max": float(np.max(rendered_pts[:, 2])),
                "intensity_min": float(np.min(rendered_pts[:, 3])),
                "intensity_max": float(np.max(rendered_pts[:, 3])),
            }
        else:
            bounds = {
                "x_min": 0.0, "x_max": 0.0, "y_min": 0.0, "y_max": 0.0,
                "z_min": 0.0, "z_max": 0.0, "intensity_min": 0.0, "intensity_max": 0.0
            }

        return {
            "frame_id": fid,
            "method": m,
            "method_metadata": METHOD_METADATA.get(m, {}),
            "render_metadata": render_meta,
            "bounds": bounds,
            "points": rendered_pts.tolist(),
        }

    def get_frame_points_binary(
        self,
        frame_id: Union[str, int],
        method: str,
        max_points: Optional[int] = DEFAULT_MAX_RENDER_POINTS
    ) -> Tuple[bytes, Dict[str, Any]]:
        """
        Fast binary point cloud streamer returning raw float32 bytes for WebGL GPU buffers.
        Returns (raw_bytes, metadata_dict).
        """
        fid = self.normalize_frame_id(frame_id)
        m = self.normalize_method(method)
        raw_pts = self.load_raw_points(fid, m)

        rendered_pts, render_meta = self.downsample_for_rendering(raw_pts, max_points)
        contiguous_pts = np.ascontiguousarray(rendered_pts, dtype=np.float32)

        return contiguous_pts.tobytes(), {
            "frame_id": fid,
            "method": m,
            **render_meta,
        }

    def get_multi_method_comparison(
        self,
        frame_id: Union[str, int],
        methods: Optional[List[str]] = None,
        max_points: Optional[int] = DEFAULT_MAX_RENDER_POINTS
    ) -> Dict[str, Any]:
        """
        Load points and retention comparison for multiple algorithms on a single frame.
        """
        fid = self.normalize_frame_id(frame_id)
        if methods is None or len(methods) == 0:
            methods = ["original", "uniform", "hybrid_v5", "v6_alpha_0.25", "v7_lambda_0.25"]
        
        normalized_methods = [self.normalize_method(m) for m in methods]

        # Always load original point count for retention baseline
        orig_pts = self.load_raw_points(fid, "original")
        orig_count = len(orig_pts)

        clouds: Dict[str, Any] = {}
        retention_stats: Dict[str, Any] = {}

        for m in normalized_methods:
            if not self.is_frame_available(fid, m):
                continue

            pts = self.load_raw_points(fid, m)
            count = len(pts)
            retention_pct = (count / orig_count * 100.0) if orig_count > 0 else 0.0

            rendered, render_meta = self.downsample_for_rendering(pts, max_points)

            retention_stats[m] = {
                "method_id": m,
                "method_name": METHOD_METADATA.get(m, {}).get("name", m),
                "total_points": count,
                "retention_pct": round(retention_pct, 4),
                "difference_from_20pct": round(retention_pct - 20.0, 4) if m != "original" else 0.0,
            }

            clouds[m] = {
                "points": rendered.tolist(),
                "render_metadata": render_meta,
            }

        return {
            "frame_id": fid,
            "original_points": orig_count,
            "retention_stats": retention_stats,
            "clouds": clouds,
        }

