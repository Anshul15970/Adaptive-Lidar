import sys
from pathlib import Path
import numpy as np
import pytest
from fastapi.testclient import TestClient

# Ensure project root is on sys.path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.backend.server import app
from app.backend.kitti_loader import KittiLoader
from app.backend.calib_service import CalibService
from app.backend.ground_truth_loader import GroundTruthLoader
from app.backend.metrics_service import MetricsService


@pytest.fixture(scope="module")
def client():
    return TestClient(app)


def test_health_check(client):
    res = client.get("/api/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "healthy"
    assert data["validation_frames_count"] == 3769
    assert "original" in data["available_methods"]
    assert "uniform" in data["available_methods"]
    assert "hybrid_v5" in data["available_methods"]
    assert "v6_alpha_0.25" in data["available_methods"]
    assert "v7_lambda_0.25" in data["available_methods"]


def test_algorithms_metadata(client):
    res = client.get("/api/algorithms")
    assert res.status_code == 200
    algos = res.json()
    assert len(algos) == 5
    ids = {a["id"] for a in algos}
    assert ids == {"original", "uniform", "hybrid_v5", "v6_alpha_0.25", "v7_lambda_0.25"}
    
    # Check uniform metadata
    uniform = next(a for a in algos if a["id"] == "uniform")
    assert uniform["name"] == "Uniform 20%"
    assert uniform["voxel_size_m"] == 0.8
    assert uniform["target_retention_pct"] == 20.0


def test_validation_frames_list(client):
    res = client.get("/api/frames")
    assert res.status_code == 200
    data = res.json()
    assert data["total_count"] == 3769
    assert data["validation_total"] == 3769
    assert data["default_frame"] == "000001"
    assert data["frame_ids"][0] == "000001"
    assert data["frame_ids"][1] == "000002"
    assert data["frame_ids"][2] == "000004"
    assert data["frame_ids"][-1] == "007480"

    # Test pagination
    res_page = client.get("/api/frames?limit=10&offset=5")
    assert res_page.status_code == 200
    page_data = res_page.json()
    assert len(page_data["frame_ids"]) == 10
    assert page_data["frame_ids"][0] == data["frame_ids"][5]

    # Test search
    res_search = client.get("/api/frames?search=007480")
    assert res_search.status_code == 200
    assert "007480" in res_search.json()["frame_ids"]


def test_load_point_cloud_all_methods(client):
    frame_id = "000001"
    methods = ["original", "uniform", "hybrid_v5", "v6_alpha_0.25", "v7_lambda_0.25"]

    point_counts = {}
    for m in methods:
        res = client.get(f"/api/frames/{frame_id}/points?method={m}&max_points=0")
        assert res.status_code == 200, f"Method {m} failed: {res.text}"
        data = res.json()
        assert data["frame_id"] == frame_id
        assert data["method"] == m
        assert not data["render_metadata"]["is_downsampled"]
        
        pts = data["points"]
        total = data["render_metadata"]["total_points"]
        assert len(pts) == total
        assert len(pts) > 0
        point_counts[m] = total

        # Check bounds
        bounds = data["bounds"]
        assert bounds["x_min"] < bounds["x_max"]
        assert bounds["y_min"] < bounds["y_max"]
        assert bounds["z_min"] < bounds["z_max"]

    # Verify original has ~120k points and sampled methods retain ~20%
    orig_count = point_counts["original"]
    assert orig_count == 120268
    for m in ["uniform", "hybrid_v5", "v6_alpha_0.25", "v7_lambda_0.25"]:
        retention = point_counts[m] / orig_count * 100.0
        assert round(retention, 2) == 20.00


def test_render_downsampling_cap(client):
    frame_id = "000001"
    cap = 5000
    res = client.get(f"/api/frames/{frame_id}/points?method=original&max_points={cap}")
    assert res.status_code == 200
    data = res.json()

    # The actual original points count must be strictly preserved
    assert data["render_metadata"]["total_points"] == 120268
    assert data["render_metadata"]["rendered_points"] == cap
    assert data["render_metadata"]["is_downsampled"] is True
    assert data["render_metadata"]["render_cap"] == cap
    assert len(data["points"]) == cap


def test_binary_point_streaming(client):
    frame_id = "000001"
    method = "v6_alpha_0.25"
    cap = 10000

    res = client.get(f"/api/frames/{frame_id}/points/binary?method={method}&max_points={cap}")
    assert res.status_code == 200
    assert res.headers["content-type"] == "application/octet-stream"
    assert res.headers["X-Frame-ID"] == frame_id
    assert res.headers["X-Method"] == method
    assert res.headers["X-Total-Points"] == "24053"
    assert res.headers["X-Rendered-Points"] == str(cap)
    assert res.headers["X-Is-Downsampled"] == "true"

    # Verify binary byte array format
    raw_bytes = res.content
    assert len(raw_bytes) == cap * 16  # 4 floats * 4 bytes
    arr = np.frombuffer(raw_bytes, dtype=np.float32).reshape(-1, 4)
    assert arr.shape == (cap, 4)
    assert np.isfinite(arr).all()


def test_multi_method_comparison(client):
    frame_id = "000001"
    res = client.get(f"/api/frames/{frame_id}/compare?max_points=5000")
    assert res.status_code == 200
    data = res.json()
    assert data["frame_id"] == frame_id
    assert data["original_points"] == 120268

    stats = data["retention_stats"]
    assert "uniform" in stats
    assert "hybrid_v5" in stats
    assert "v6_alpha_0.25" in stats
    assert "v7_lambda_0.25" in stats

    for m in ["uniform", "hybrid_v5", "v6_alpha_0.25", "v7_lambda_0.25"]:
        assert stats[m]["total_points"] == 24053
        assert round(stats[m]["retention_pct"], 2) == 20.00


def test_calib_and_ground_truth(client):
    frame_id = "000001"
    # Ground truth without point counts
    res = client.get(f"/api/frames/{frame_id}/ground-truth")
    assert res.status_code == 200
    data = res.json()
    assert data["frame_id"] == frame_id
    assert data["object_count"] == 2  # 1 Car, 1 Cyclist (excluding Truck & DontCare)
    
    classes = [o["class"] for o in data["objects"]]
    assert "Car" in classes
    assert "Cyclist" in classes

    car_obj = next(o for o in data["objects"] if o["class"] == "Car")
    assert car_obj["box_3d"]["dimensions"] == [3.69, 1.87, 1.67] # l, w, h
    assert len(car_obj["box_3d"]["corners"]) == 8
    assert len(car_obj["box_3d"]["lines"]) == 14

    # Ground truth with on-the-fly point counting for original vs uniform vs v6
    res_orig = client.get(f"/api/frames/{frame_id}/ground-truth?count_points_method=original")
    car_orig = next(o for o in res_orig.json()["objects"] if o["class"] == "Car")
    cyc_orig = next(o for o in res_orig.json()["objects"] if o["class"] == "Cyclist")
    assert car_orig["contained_points"] == 9
    assert cyc_orig["contained_points"] == 18

    res_uni = client.get(f"/api/frames/{frame_id}/ground-truth?count_points_method=uniform")
    car_uni = next(o for o in res_uni.json()["objects"] if o["class"] == "Car")
    cyc_uni = next(o for o in res_uni.json()["objects"] if o["class"] == "Cyclist")
    assert car_uni["contained_points"] == 1
    assert cyc_uni["contained_points"] == 0  # Uniform completely starves the cyclist!

    res_v6 = client.get(f"/api/frames/{frame_id}/ground-truth?count_points_method=v6_alpha_0.25")
    car_v6 = next(o for o in res_v6.json()["objects"] if o["class"] == "Car")
    cyc_v6 = next(o for o in res_v6.json()["objects"] if o["class"] == "Cyclist")
    assert car_v6["contained_points"] == 6
    assert cyc_v6["contained_points"] == 12  # V6 preserves 66.7% of points on both!


def test_detector_predictions(client):
    frame_id = "000001"
    res = client.get(f"/api/frames/{frame_id}/predictions?method=v6_alpha_0.25&score_threshold=0.2")
    assert res.status_code == 200
    data = res.json()
    assert data["frame_id"] == frame_id
    assert data["method"] == "v6_alpha_0.25"
    assert "predictions" in data


def test_metrics_endpoints(client):
    # 1. Global retention summary
    res_global = client.get("/api/metrics/global")
    assert res_global.status_code == 200
    global_data = res_global.json()
    assert len(global_data) == 4
    for m in global_data:
        assert m["frames_count"] == 3769
        assert round(m["weighted_retention_pct"], 2) == 20.00

    # 2. Distance retention
    res_dist = client.get("/api/metrics/distance")
    assert res_dist.status_code == 200
    dist_data = res_dist.json()
    assert len(dist_data) == 36

    # Test filtering distance metrics by class
    res_car_dist = client.get("/api/metrics/distance?class_name=Car")
    assert res_car_dist.status_code == 200
    assert all(d["class"] == "Car" for d in res_car_dist.json())

    # 3. Object retention distribution
    res_distrib = client.get("/api/metrics/distribution")
    assert res_distrib.status_code == 200
    distrib_data = res_distrib.json()
    assert len(distrib_data) == 12

    # 4. Per-frame metrics
    res_frame_m = client.get("/api/frames/000001/metrics")
    assert res_frame_m.status_code == 200
    fm_data = res_frame_m.json()
    assert fm_data["retention"]["frame_id"] == "000001"
    assert fm_data["retention"]["original_points"] == 120268
    assert len(fm_data["objects"]) == 2

    # 5. Detector evaluation benchmark
    res_det = client.get("/api/metrics/detector")
    assert res_det.status_code == 200
    det_data = res_det.json()
    assert "uniform" in det_data
    assert "hybrid_v5" in det_data
    assert "v6_alpha_0.25" in det_data
    assert "v7_lambda_0.25" in det_data
    
    # Verify Car 3D AP values match evaluation logs
    v6_car_3d = det_data["v6_alpha_0.25"]["classes"]["Car"]["3d_ap_r40"]
    assert round(v6_car_3d["easy"], 2) == 78.56
    assert round(v6_car_3d["moderate"], 2) == 66.24
    assert round(v6_car_3d["hard"], 2) == 62.81


def test_diverse_validation_frames(client):
    for fid in ["000002", "000004", "000005", "007480"]:
        res = client.get(f"/api/frames/{fid}/metrics")
        assert res.status_code == 200
        data = res.json()
        assert data["retention"]["frame_id"] == fid
        assert "v6_alpha_0.25" in data["retention"]["methods"]
        
        # Verify ground truth loads for each frame
        res_gt = client.get(f"/api/frames/{fid}/ground-truth")
        assert res_gt.status_code == 200
        gt_data = res_gt.json()
        assert gt_data["frame_id"] == fid


def test_frontend_serving(client):
    res = client.get("/")
    assert res.status_code == 200
    assert "Adaptive LiDAR Lab" in res.text


def test_invalid_requests(client):
    # Non-existent frame
    res_404 = client.get("/api/frames/999999/points?method=uniform")
    assert res_404.status_code == 404

    # Invalid method
    res_bad_method = client.get("/api/frames/000001/points?method=invalid_sampler")
    assert res_bad_method.status_code == 400
