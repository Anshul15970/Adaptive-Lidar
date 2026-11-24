from pathlib import Path
from typing import Dict, Any, List

# Locate root directory
BACKEND_DIR = Path(__file__).resolve().parent
APP_DIR = BACKEND_DIR.parent
PROJECT_ROOT = APP_DIR.parent

DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"

ORIGINAL_VELODYNE_DIR = DATA_DIR / "object_detection" / "training" / "velodyne"
LABEL_DIR = DATA_DIR / "object_detection" / "training" / "label_2"
CALIB_DIR = DATA_DIR / "object_detection" / "training" / "calib"

METHOD_DIRECTORIES: Dict[str, Path] = {
    "original": ORIGINAL_VELODYNE_DIR,
    "uniform": DATA_DIR / "detector_sampling" / "uniform" / "velodyne",
    "hybrid_v5": DATA_DIR / "detector_sampling" / "hybrid_v5" / "velodyne",
    "v6_alpha_0.25": DATA_DIR / "detector_sampling" / "v6_alpha_0.25" / "velodyne",
    "v7_lambda_0.25": DATA_DIR / "detector_sampling" / "v7_lambda_0.25" / "velodyne",
}

METHOD_ALIASES: Dict[str, str] = {
    "orig": "original",
    "full": "original",
    "raw": "original",
    "uniform20": "uniform",
    "v5": "hybrid_v5",
    "v6": "v6_alpha_0.25",
    "v7": "v7_lambda_0.25",
}

METHOD_METADATA: Dict[str, Dict[str, Any]] = {
    "original": {
        "id": "original",
        "name": "Original",
        "description": "Full uncompressed raw LiDAR point cloud (~120k points)",
        "voxel_size_m": None,
        "target_retention_pct": 100.0,
        "color": "#9ca3af", # Gray
        "is_baseline": False,
    },
    "uniform": {
        "id": "uniform",
        "name": "Uniform 20%",
        "description": "Uniform spatial voxel grid downsampling baseline (0.8m voxel size)",
        "voxel_size_m": 0.8,
        "target_retention_pct": 20.0,
        "color": "#ef4444", # Red
        "is_baseline": True,
    },
    "hybrid_v5": {
        "id": "hybrid_v5",
        "name": "Hybrid V5",
        "description": "Geometry complexity & voxel sparsity-based sampling (0.8m voxel size)",
        "voxel_size_m": 0.8,
        "target_retention_pct": 20.0,
        "color": "#f59e0b", # Amber
        "is_baseline": False,
    },
    "v6_alpha_0.25": {
        "id": "v6_alpha_0.25",
        "name": "V6 α=0.25",
        "description": "Density-aware adaptive sampling (α=0.25, 0.8m voxel size)",
        "voxel_size_m": 0.8,
        "target_retention_pct": 20.0,
        "color": "#10b981", # Emerald
        "is_baseline": False,
    },
    "v7_lambda_0.25": {
        "id": "v7_lambda_0.25",
        "name": "V7 λ=0.25",
        "description": "Hybrid score-density sampling (λ=0.25, 0.8m voxel size)",
        "voxel_size_m": 0.8,
        "target_retention_pct": 20.0,
        "color": "#3b82f6", # Blue
        "is_baseline": False,
    },
}

# Supported object classes
SUPPORTED_CLASSES = ["Car", "Pedestrian", "Cyclist"]

# Distance bands
DISTANCE_BINS = {
    "near_0_20m": (0.0, 20.0),
    "medium_20_40m": (20.0, 40.0),
    "far_over_40m": (40.0, float("inf")),
}

# CSV Results paths
CSV_FILES = {
    "global_retention_summary": RESULTS_DIR / "global_retention_summary_v5_v6_v7_comparison.csv",
    "global_frame_retention": RESULTS_DIR / "global_frame_retention_v5_v6_v7_comparison.csv",
    "distance_retention": RESULTS_DIR / "distance_retention_v5_v6_v7_comparison.csv",
    "object_retention_distribution": RESULTS_DIR / "object_retention_distribution_v5_v6_v7_comparison.csv",
    "object_retention_details": RESULTS_DIR / "object_retention_details_v5_v6_v7_comparison.csv",
}

# Candidate validation split file paths
VAL_FILE_CANDIDATES: List[Path] = [
    Path.home() / "adaptive-lidar-detector" / "OpenPCDet" / "data" / "kitti" / "ImageSets" / "val.txt",
    Path(r"\\wsl.localhost\Ubuntu-22.04\home\anshul_gurjar\adaptive-lidar-detector\OpenPCDet\data\kitti\ImageSets\val.txt"),
    PROJECT_ROOT / "val.txt",
]

# OpenPCDet evaluation output candidates for 3D predicted boxes
DETECTOR_EVAL_DIRS: List[Path] = [
    Path.home() / "adaptive-lidar-detector" / "OpenPCDet" / "output" / "kitti_models" / "evaluation",
    Path(r"\\wsl.localhost\Ubuntu-22.04\home\anshul_gurjar\adaptive-lidar-detector\OpenPCDet\output\kitti_models\evaluation"),
    RESULTS_DIR / "detector_evaluation",
]

# Performance / Rendering limits
DEFAULT_MAX_RENDER_POINTS = 60000  # Default cap for browser WebGL rendering
LRU_CACHE_CAPACITY = 64            # In-memory point-cloud frame cache size

