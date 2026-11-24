import logging
from typing import Dict, Any, List, Optional
from fastapi import FastAPI, HTTPException, Query, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from fastapi.staticfiles import StaticFiles
from .config import (
    METHOD_METADATA,
    METHOD_DIRECTORIES,
    SUPPORTED_CLASSES,
    DEFAULT_MAX_RENDER_POINTS,
    APP_DIR,
)
from .kitti_loader import KittiLoader
from .calib_service import CalibService
from .ground_truth_loader import GroundTruthLoader
from .metrics_service import MetricsService
from .detector_loader import DetectorLoader

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("adaptive_lidar_lab")

app = FastAPI(
    title="Adaptive LiDAR Lab API",
    description="Backend & Data-Access Layer for Adaptive LiDAR Point Cloud Downsampling & Detection Research",
    version="1.0.0",
)

# Enable CORS for local frontend development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=[
        "X-Frame-ID",
        "X-Method",
        "X-Total-Points",
        "X-Rendered-Points",
        "X-Is-Downsampled",
        "X-Sampling-Ratio",
        "X-Render-Cap",
    ],
)

# Initialize services
kitti_loader = KittiLoader()
calib_service = CalibService()
gt_loader = GroundTruthLoader(calib_service=calib_service)
metrics_service = MetricsService()
detector_loader = DetectorLoader(calib_service=calib_service)


@app.get("/api/health")
def health_check() -> Dict[str, Any]:
    """Health check and data availability status."""
    try:
        val_ids = kitti_loader.get_validation_frame_ids()
        val_count = len(val_ids)
    except Exception as e:
        val_count = 0

    return {
        "status": "healthy",
        "validation_frames_count": val_count,
        "available_methods": list(METHOD_METADATA.keys()),
        "classes": SUPPORTED_CLASSES,
    }


@app.get("/api/algorithms")
def get_algorithms() -> List[Dict[str, Any]]:
    """Return all supported point-cloud downsampling algorithms and their metadata."""
    return list(METHOD_METADATA.values())


@app.get("/api/frames")
def get_validation_frames(
    limit: Optional[int] = Query(None, description="Optional pagination limit"),
    offset: int = Query(0, description="Optional pagination offset"),
    search: Optional[str] = Query(None, description="Optional search string for frame ID")
) -> Dict[str, Any]:
    """
    Expose the 3,769 KITTI validation frame IDs without artificial generation.
    """
    try:
        all_ids = kitti_loader.get_validation_frame_ids()
    except Exception as e:
        logger.error(f"Failed to load validation frame IDs: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    filtered_ids = all_ids
    if search:
        search_str = search.strip()
        filtered_ids = [fid for fid in all_ids if search_str in fid]

    total = len(filtered_ids)
    if limit is not None and limit > 0:
        paged_ids = filtered_ids[offset : offset + limit]
    else:
        paged_ids = filtered_ids[offset:]

    return {
        "total_count": total,
        "validation_total": len(all_ids),
        "offset": offset,
        "limit": limit or total,
        "frame_ids": paged_ids,
        "default_frame": all_ids[0] if all_ids else "000001",
    }


@app.get("/api/frames/{frame_id}/points")
def get_frame_points(
    frame_id: str,
    method: str = Query("uniform", description="Algorithm method ID or alias"),
    max_points: Optional[int] = Query(
        DEFAULT_MAX_RENDER_POINTS,
        description="Point cap for browser rendering (0 or null for full cloud)"
    )
) -> Dict[str, Any]:
    """
    Get point-cloud data (XYZI) and metadata for a specific frame and algorithm.
    """
    try:
        cap = None if (max_points is not None and max_points <= 0) else max_points
        data = kitti_loader.get_frame_points(frame_id, method, max_points=cap)
        return data
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception("Error loading frame points")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/frames/{frame_id}/points/binary")
def get_frame_points_binary(
    frame_id: str,
    method: str = Query("uniform", description="Algorithm method ID or alias"),
    max_points: Optional[int] = Query(
        DEFAULT_MAX_RENDER_POINTS,
        description="Point cap for browser rendering"
    )
):
    """
    Fast binary point-cloud streamer returning raw float32 bytes for WebGL GPU buffers.
    Metadata is passed in HTTP response headers.
    """
    try:
        cap = None if (max_points is not None and max_points <= 0) else max_points
        raw_bytes, meta = kitti_loader.get_frame_points_binary(frame_id, method, max_points=cap)

        headers = {
            "X-Frame-ID": str(meta["frame_id"]),
            "X-Method": str(meta["method"]),
            "X-Total-Points": str(meta["total_points"]),
            "X-Rendered-Points": str(meta["rendered_points"]),
            "X-Is-Downsampled": "true" if meta["is_downsampled"] else "false",
            "X-Sampling-Ratio": str(meta["sampling_ratio"]),
            "X-Render-Cap": str(meta["render_cap"] or ""),
        }

        return Response(content=raw_bytes, media_type="application/octet-stream", headers=headers)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception("Error streaming binary points")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/frames/{frame_id}/compare")
def compare_frame_algorithms(
    frame_id: str,
    methods: Optional[str] = Query(
        None,
        description="Comma-separated list of methods (defaults to all five methods)"
    ),
    max_points: Optional[int] = Query(
        30000,
        description="Point cap per cloud for comparative multi-view rendering"
    )
) -> Dict[str, Any]:
    """
    Retrieve multiple algorithms for a single frame simultaneously,
    including exact retention percentages and downsampled clouds.
    """
    try:
        method_list = [m.strip() for m in methods.split(",")] if methods else None
        cap = None if (max_points is not None and max_points <= 0) else max_points
        data = kitti_loader.get_multi_method_comparison(frame_id, method_list, max_points=cap)
        return data
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception("Error in multi-algorithm comparison")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/frames/{frame_id}/metrics")
def get_frame_metrics(frame_id: str) -> Dict[str, Any]:
    """
    Retrieve frame-specific retention metrics and per-object point retention.
    """
    try:
        frame_retention = metrics_service.get_frame_retention(frame_id)
        object_retention = metrics_service.get_frame_objects_retention(frame_id)
        return {
            "retention": frame_retention,
            "objects": object_retention,
        }
    except Exception as e:
        logger.exception("Error loading frame metrics")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/frames/{frame_id}/ground-truth")
def get_frame_ground_truth(
    frame_id: str,
    classes: Optional[str] = Query(
        None,
        description="Comma-separated list of classes (default: Car,Pedestrian,Cyclist)"
    ),
    count_points_method: Optional[str] = Query(
        None,
        description="Optional method to compute points-in-box counts on-the-fly"
    )
) -> Dict[str, Any]:
    """
    Retrieve KITTI 3D ground truth bounding boxes transformed into the LiDAR coordinate frame.
    Optionally computes the point containment for a specific sampling method.
    """
    try:
        filter_classes = [c.strip() for c in classes.split(",")] if classes else None

        points_lidar = None
        if count_points_method:
            points_lidar = kitti_loader.load_raw_points(frame_id, count_points_method)

        objects = gt_loader.load_ground_truth(
            frame_id=frame_id,
            filter_classes=filter_classes,
            points_lidar=points_lidar
        )

        return {
            "frame_id": kitti_loader.normalize_frame_id(frame_id),
            "object_count": len(objects),
            "counted_for_method": count_points_method,
            "objects": objects,
        }
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception("Error loading ground truth")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/frames/{frame_id}/predictions")
def get_frame_predictions(
    frame_id: str,
    method: str = Query("v6_alpha_0.25", description="Model evaluated method"),
    score_threshold: float = Query(0.25, description="Confidence threshold")
) -> Dict[str, Any]:
    """
    Retrieve PointPillars predicted 3D bounding boxes in LiDAR coordinates.
    """
    try:
        norm_method = kitti_loader.normalize_method(method)
        predictions = detector_loader.get_frame_predictions(
            frame_id=frame_id,
            method=norm_method,
            score_threshold=score_threshold
        )
        return {
            "frame_id": kitti_loader.normalize_frame_id(frame_id),
            "method": norm_method,
            "score_threshold": score_threshold,
            "prediction_count": len(predictions),
            "predictions": predictions,
        }
    except Exception as e:
        logger.exception("Error loading detector predictions")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/metrics/global")
def get_global_metrics() -> List[Dict[str, Any]]:
    """Retrieve global point retention summary across all 3,769 validation frames."""
    return metrics_service.get_global_retention_summary()


@app.get("/api/metrics/distance")
def get_distance_metrics(
    class_name: Optional[str] = Query(None, description="Car, Pedestrian, or Cyclist"),
    method: Optional[str] = Query(None, description="Optional method filter")
) -> List[Dict[str, Any]]:
    """Retrieve distance retention breakdown for near, medium, and far ranges."""
    return metrics_service.get_distance_retention(class_name=class_name, method=method)


@app.get("/api/metrics/distribution")
def get_distribution_metrics(
    class_name: Optional[str] = Query(None, description="Car, Pedestrian, or Cyclist")
) -> List[Dict[str, Any]]:
    """Retrieve object retention distribution and starvation percentages."""
    return metrics_service.get_object_retention_distribution(class_name=class_name)


@app.get("/api/metrics/detector")
def get_detector_metrics() -> Dict[str, Any]:
    """Retrieve official PointPillars 3D and BEV AP benchmark metrics from logs."""
    return metrics_service.get_detector_benchmark_metrics()


# Serve compiled frontend assets if available
FRONTEND_DIST = APP_DIR / "frontend" / "dist"
if FRONTEND_DIST.is_dir():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIST), html=True), name="frontend")

