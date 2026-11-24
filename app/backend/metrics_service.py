import re
from pathlib import Path
from typing import Dict, Any, List, Optional, Union

import pandas as pd
import numpy as np

from .config import CSV_FILES, RESULTS_DIR, METHOD_METADATA


class MetricsService:
    """
    In-memory query and aggregation service for pre-computed research CSV outputs
    and detector evaluation logs.
    """

    def __init__(self):
        self._global_summary_df: Optional[pd.DataFrame] = None
        self._frame_retention_df: Optional[pd.DataFrame] = None
        self._distance_retention_df: Optional[pd.DataFrame] = None
        self._distribution_df: Optional[pd.DataFrame] = None
        self._object_details_df: Optional[pd.DataFrame] = None
        self._detector_benchmark: Optional[Dict[str, Any]] = None

        self._load_csv_data()

    def _load_csv_data(self):
        """
        Load all 5 comparison CSV files into memory.
        """
        # 1. Global retention summary
        p_sum = CSV_FILES["global_retention_summary"]
        if p_sum.is_file():
            self._global_summary_df = pd.read_csv(p_sum)

        # 2. Per-frame retention
        p_frame = CSV_FILES["global_frame_retention"]
        if p_frame.is_file():
            self._frame_retention_df = pd.read_csv(p_frame)

        # 3. Distance retention
        p_dist = CSV_FILES["distance_retention"]
        if p_dist.is_file():
            self._distance_retention_df = pd.read_csv(p_dist)

        # 4. Object retention distribution
        p_distrib = CSV_FILES["object_retention_distribution"]
        if p_distrib.is_file():
            self._distribution_df = pd.read_csv(p_distrib)

        # 5. Object retention details
        p_details = CSV_FILES["object_retention_details"]
        if p_details.is_file():
            self._object_details_df = pd.read_csv(p_details)

    def get_global_retention_summary(self) -> List[Dict[str, Any]]:
        """
        Retrieve macro global retention metrics across all 3,769 validation frames.
        """
        if self._global_summary_df is None:
            return []

        records = []
        for _, row in self._global_summary_df.iterrows():
            m = row["method"]
            meta = METHOD_METADATA.get(m, {})
            records.append({
                "method_id": m,
                "method_name": meta.get("name", m),
                "frames_count": int(row["frames"]),
                "original_points_total": int(row["original_points_total"]),
                "sampled_points_total": int(row["sampled_points_total"]),
                "weighted_retention_pct": round(float(row["overall_weighted_retention_pct"]), 5),
                "mean_frame_retention_pct": round(float(row["mean_frame_retention_pct"]), 5),
                "median_frame_retention_pct": round(float(row["median_frame_retention_pct"]), 5),
                "min_frame_retention_pct": round(float(row["minimum_frame_retention_pct"]), 5),
                "max_frame_retention_pct": round(float(row["maximum_frame_retention_pct"]), 5),
                "frames_exactly_20pct": int(row["frames_exactly_20pct"]),
            })
        return records

    def get_frame_retention(self, frame_id: Union[str, int]) -> Dict[str, Any]:
        """
        Retrieve per-frame retention for a specific frame ID across all methods.
        """
        fid_int = int(str(frame_id).split(".")[0])
        fid_str = f"{fid_int:06d}"

        if self._frame_retention_df is None:
            return {"frame_id": fid_str, "methods": {}}

        frame_data = self._frame_retention_df[self._frame_retention_df["frame"] == fid_int]
        if frame_data.empty:
            return {"frame_id": fid_str, "methods": {}}

        methods_dict = {}
        original_points = int(frame_data.iloc[0]["original_points"])

        for _, row in frame_data.iterrows():
            m = row["method"]
            meta = METHOD_METADATA.get(m, {})
            methods_dict[m] = {
                "method_id": m,
                "method_name": meta.get("name", m),
                "sampled_points": int(row["sampled_points"]),
                "retention_pct": round(float(row["retention_pct"]), 4),
                "difference_from_20pct": round(float(row["difference_from_20pct"]), 4),
            }

        return {
            "frame_id": fid_str,
            "original_points": original_points,
            "methods": methods_dict,
        }

    def get_distance_retention(
        self,
        class_name: Optional[str] = None,
        method: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve retention broken down by distance bins (near, medium, far)
        and object classes (Car, Pedestrian, Cyclist).
        """
        if self._distance_retention_df is None:
            return []

        df = self._distance_retention_df.copy()
        if class_name:
            df = df[df["class"].str.lower() == class_name.lower()]
        if method:
            df = df[df["method"] == method]

        records = []
        for _, row in df.iterrows():
            m = row["method"]
            meta = METHOD_METADATA.get(m, {})
            records.append({
                "class": row["class"],
                "distance_group": row["distance_group"],
                "method_id": m,
                "method_name": meta.get("name", m),
                "object_count": int(row["object_count"]),
                "mean_retention_pct": round(float(row["mean_retention_pct"]), 2),
                "median_retention_pct": round(float(row["median_retention_pct"]), 2),
                "weighted_retention_pct": round(float(row["weighted_retention_pct"]), 2),
                "original_box_points": int(row["original_box_points"]),
                "retained_box_points": int(row["retained_box_points"]),
            })
        return records

    def get_object_retention_distribution(
        self,
        class_name: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve starvation distribution percentages for classes across methods.
        Demonstrates the percentage of objects retaining <10%, <20%, <40%, <60% points.
        """
        if self._distribution_df is None:
            return []

        df = self._distribution_df.copy()
        if class_name:
            df = df[df["class"].str.lower() == class_name.lower()]

        records = []
        for _, row in df.iterrows():
            m = row["method"]
            meta = METHOD_METADATA.get(m, {})
            records.append({
                "class": row["class"],
                "method_id": m,
                "method_name": meta.get("name", m),
                "object_count": int(row["object_count"]),
                "objects_below_10pct": int(row["objects_below_10pct"]),
                "pct_below_10pct": round(float(row["percentage_below_10pct"]), 2),
                "objects_below_20pct": int(row["objects_below_20pct"]),
                "pct_below_20pct": round(float(row["percentage_below_20pct"]), 2),
                "objects_below_40pct": int(row["objects_below_40pct"]),
                "pct_below_40pct": round(float(row["percentage_below_40pct"]), 2),
                "objects_below_60pct": int(row["objects_below_60pct"]),
                "pct_below_60pct": round(float(row["percentage_below_60pct"]), 2),
            })
        return records

    def get_frame_objects_retention(self, frame_id: Union[str, int]) -> List[Dict[str, Any]]:
        """
        Retrieve object retention details for all annotated objects in a specific frame.
        """
        fid_int = int(str(frame_id).split(".")[0])
        if self._object_details_df is None:
            return []

        df_frame = self._object_details_df[self._object_details_df["frame"] == fid_int]
        if df_frame.empty:
            return []

        # Group by object_id
        objects_dict: Dict[int, Dict[str, Any]] = {}
        for _, row in df_frame.iterrows():
            obj_id = int(row["object_id"])
            if obj_id not in objects_dict:
                objects_dict[obj_id] = {
                    "object_id": obj_id,
                    "class": row["class"],
                    "distance_z_m": round(float(row["distance_z_m"]), 2),
                    "distance_group": row["distance_group"],
                    "original_box_points": int(row["original_box_points"]),
                    "method_retention": {},
                }

            m = row["method"]
            objects_dict[obj_id]["method_retention"][m] = {
                "retained_points": int(row["retained_box_points"]),
                "retention_pct": round(float(row["retention_pct"]), 2),
            }

        return list(objects_dict.values())

    def get_detector_benchmark_metrics(self) -> Dict[str, Any]:
        """
        Parse detector evaluation logs to extract official KITTI 3D AP and BEV AP metrics.
        """
        if self._detector_benchmark is not None:
            return self._detector_benchmark

        logs_dir = RESULTS_DIR / "detector_evaluation" / "logs"
        log_files = {
            "uniform": logs_dir / "uniform20.log",
            "hybrid_v5": logs_dir / "v5.log",
            "v6_alpha_0.25": logs_dir / "v6_alpha_0.25.log",
            "v7_lambda_0.25": logs_dir / "v7_lambda_0.25.log",
        }

        benchmarks: Dict[str, Any] = {}

        for method_key, log_path in log_files.items():
            if not log_path.is_file():
                continue

            text = log_path.read_text(encoding="utf-8", errors="ignore")
            method_results: Dict[str, Any] = {"method_id": method_key, "classes": {}}

            for cls in ["Car", "Pedestrian", "Cyclist"]:
                # Match 3d AP_R40
                pattern_3d = rf"{cls} AP_R40@[0-9., ]+:\s*\n[^\n]+\n[^\n]+\n\s*3d\s+AP:([0-9., ]+)"
                m_3d = re.search(pattern_3d, text)

                # Match bev AP_R40
                pattern_bev = rf"{cls} AP_R40@[0-9., ]+:\s*\n[^\n]+\n\s*bev\s+AP:([0-9., ]+)"
                m_bev = re.search(pattern_bev, text)

                cls_metrics = {}
                if m_3d:
                    parts = [float(v.strip()) for v in m_3d.group(1).split(",")]
                    if len(parts) >= 3:
                        cls_metrics["3d_ap_r40"] = {
                            "easy": parts[0],
                            "moderate": parts[1],
                            "hard": parts[2],
                        }

                if m_bev:
                    parts = [float(v.strip()) for v in m_bev.group(1).split(",")]
                    if len(parts) >= 3:
                        cls_metrics["bev_ap_r40"] = {
                            "easy": parts[0],
                            "moderate": parts[1],
                            "hard": parts[2],
                        }

                method_results["classes"][cls] = cls_metrics

            benchmarks[method_key] = method_results

        # Add Full PointPillars reference benchmark if log exists
        full_log = Path(r"\\wsl.localhost\Ubuntu-22.04\home\anshul_gurjar\adaptive-lidar-detector\OpenPCDet\output\kitti_models\evaluation\pointpillar_full\fixed_ckpt_full\eval\epoch_40\val\val\log_eval_20261101-193917.txt")
        if full_log.is_file():
            text = full_log.read_text(encoding="utf-8", errors="ignore")
            full_results: Dict[str, Any] = {"method_id": "full", "classes": {}}
            for cls in ["Car", "Pedestrian", "Cyclist"]:
                m_3d = re.search(rf"{cls} AP_R40@[0-9., ]+:\s*\n[^\n]+\n[^\n]+\n\s*3d\s+AP:([0-9., ]+)", text)
                if m_3d:
                    parts = [float(v.strip()) for v in m_3d.group(1).split(",")]
                    if len(parts) >= 3:
                        full_results["classes"][cls] = {
                            "3d_ap_r40": {"easy": parts[0], "moderate": parts[1], "hard": parts[2]}
                        }
            benchmarks["full"] = full_results

        self._detector_benchmark = benchmarks
        return benchmarks

