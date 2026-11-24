import pickle
from pathlib import Path
from typing import Dict, Any, List, Optional, Union

import numpy as np

from .config import DETECTOR_EVAL_DIRS
from .calib_service import CalibService


class DetectorLoader:
    """
    Loads pretrained PointPillars 3D detection predictions from result.pkl files,
    indexing predictions by (method, frame_id) for visualization alongside ground truth.
    """

    METHOD_MAPPING = {
        "uniform": "pointpillar_uniform20/fixed_ckpt_uniform20/eval/epoch_40/val/val/result.pkl",
        "hybrid_v5": "pointpillar_v5/fixed_ckpt_v5/eval/epoch_40/val/val/result.pkl",
        "v6_alpha_0.25": "pointpillar_v6_alpha_0.25/fixed_ckpt_v6_alpha_0.25/eval/epoch_40/val/val/result.pkl",
        "v7_lambda_0.25": "pointpillar_v7_lambda_0.25/fixed_ckpt_v7_lambda_0.25/eval/epoch_40/val/val/result.pkl",
        "original": "pointpillar_full/fixed_ckpt_full/eval/epoch_40/val/val/result.pkl",
    }

    def __init__(self, calib_service: Optional[CalibService] = None):
        self.calib_service = calib_service or CalibService()
        self._predictions_cache: Dict[str, Dict[str, Dict[str, Any]]] = {}
        self._eval_base_dir: Optional[Path] = self._locate_eval_dir()

    def _locate_eval_dir(self) -> Optional[Path]:
        for candidate in DETECTOR_EVAL_DIRS:
            if candidate.is_dir():
                return candidate
        return None

    def _load_method_predictions(self, method: str) -> Dict[str, Dict[str, Any]]:
        if method in self._predictions_cache:
            return self._predictions_cache[method]

        if not self._eval_base_dir:
            return {}

        rel_path = self.METHOD_MAPPING.get(method)
        if not rel_path:
            return {}

        pkl_path = self._eval_base_dir / rel_path
        if not pkl_path.is_file():
            return {}

        try:
            with open(pkl_path, "rb") as f:
                records = pickle.load(f)

            indexed: Dict[str, Dict[str, Any]] = {}
            for rec in records:
                fid = str(rec.get("frame_id", "")).strip()
                if fid:
                    fid_str = f"{int(fid):06d}"
                    indexed[fid_str] = rec

            self._predictions_cache[method] = indexed
            return indexed
        except Exception:
            return {}

    def get_frame_predictions(
        self,
        frame_id: Union[str, int],
        method: str,
        score_threshold: float = 0.25
    ) -> List[Dict[str, Any]]:
        """
        Retrieve 3D predicted boxes for a specific frame and method.
        Filters by confidence score threshold.
        Returns oriented boxes in LiDAR coordinate space with corners and lines.
        """
        fid = f"{int(str(frame_id).split('.')[0]):06d}"
        indexed = self._load_method_predictions(method)
        if fid not in indexed:
            return []

        rec = indexed[fid]
        names = rec.get("name", [])
        scores = rec.get("score", [])
        boxes_lidar = rec.get("boxes_lidar", [])  # (M, 7): [x, y, z, dx, dy, dz, heading]

        predicted_objects = []
        for i in range(len(names)):
            score = float(scores[i])
            if score < score_threshold:
                continue

            cls_name = str(names[i])
            box = boxes_lidar[i]
            center = box[:3]
            dims = box[3:6]
            heading = float(box[6])

            corners = self.calib_service.compute_box_3d_corners(center, dims, heading)
            lines = self.calib_service.get_box_wireframe_lines(corners)
            distance_m = float(np.linalg.norm(center[:2]))

            predicted_objects.append({
                "class": cls_name,
                "score": round(score, 3),
                "distance_m": round(distance_m, 2),
                "box_3d": {
                    "center": [round(float(c), 3) for c in center],
                    "dimensions": [round(float(d), 3) for d in dims],
                    "heading": round(heading, 4),
                    "corners": [[round(float(coord), 3) for coord in pt] for pt in corners],
                    "lines": lines,
                }
            })

        return predicted_objects

