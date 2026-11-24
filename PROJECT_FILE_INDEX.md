# Adaptive LiDAR Lab: Complete Project File Index
## Detailed Directory and Source Code Inventory

**Project Root:** `/mnt/e/Projects/adaptive-lidar` (`E:\Projects\adaptive-lidar`)  
**Scope:** Complete inventory of source code, scripts, configurations, datasets, diagnostics, and application assets.  

---

### Table of Contents
1. [Core Algorithm Implementations](#1-core-algorithm-implementations)
2. [Dataset Export & Generation Scripts](#2-dataset-export--generation-scripts)
3. [Diagnostic & Retention Analysis Scripts](#3-diagnostic--retention-analysis-scripts)
4. [Comparative Benchmark & Evaluation Scripts](#4-comparative-benchmark--evaluation-scripts)
5. [Backend Application Architecture (FastAPI)](#5-backend-application-architecture-fastapi)
6. [Frontend Application Architecture (React + Three.js)](#6-frontend-application-architecture-react--threejs)
7. [Experimental Results & Evaluation Artifacts](#7-experimental-results--evaluation-artifacts)
8. [Automated Test Suite](#8-automated-test-suite)
9. [Configuration & Environment Files](#9-configuration--environment-files)

---

### 1. Core Algorithm Implementations

#### 1.1 `hybrid_sampling_v5.py`
* **Path:** `e:\Projects\adaptive-lidar\hybrid_sampling_v5.py`
* **Role:** Primary implementation of the **Hybrid V5** downsampling algorithm.
* **Key Functions:**
  * `hybrid_sample_v5(xyz, target_points=9416, voxel_size=0.8, random_state=0, verbose=False)`: Main entrypoint. Performs $0.8\,\text{m}$ cubic voxelization, centroid extraction, representative point selection, dispersion complexity $C_v$, and inverse square-root sparsity $S_v$.
  * `_select_spatially_spread(centers, scores, count)`: Spatial spreading fallback solver used when target budget $K \le M$ (voxel count).
* **Inputs:** Raw $(N, 3)$ coordinate array (float32).
* **Outputs:** Downsampled $(K, 3)$ coordinate array (float32).
* **Dependencies:** `numpy`, `time.perf_counter`.
* **Relationships:** Ingested by `export_hybrid_v5.py` and `app/backend/kitti_loader.py`. Reused conceptually by `v7_score_density_sampler.py`.

#### 1.2 `v6_density_sampler.py`
* **Path:** `e:\Projects\adaptive-lidar\v6_density_sampler.py`
* **Role:** Primary implementation of the **V6 Density-Aware Adaptive** downsampler and Hamilton-Webster integer quota allocator.
* **Key Functions:**
  * `allocate_quotas(capacities, weights, total)`: Deterministic largest-remainder integer quota solver subject to voxel capacity constraints $k_v \le c_v$.
  * `sample_v6_indices(xyz, target_points, alpha, voxel_size=0.8)`: Computes sub-linear power-law voxel weights $w_v = n_v^\alpha$, selects centroid-nearest representatives, orders remaining points by descending centroid distance, solves integer quotas, and returns indices into the valid input array.
  * `main()`: Command-line interface evaluating $\alpha \in \{0.00, 0.25, 0.50, 0.75\}$ across validation frames.
* **Inputs:** Raw $(N, 3)$ coordinate array, target point count $K$, exponent $\alpha$.
* **Outputs:** Array of $K$ integer indices into the input array.
* **Dependencies:** `numpy`, `argparse`, `pathlib`, `csv`.
* **Relationships:** Ingested by `export_v6_v7_validation.py`. Function `allocate_quotas` is directly imported and reused by `v7_score_density_sampler.py`.

#### 1.3 `v7_score_density_sampler.py`
* **Path:** `e:\Projects\adaptive-lidar\v7_score_density_sampler.py`
* **Role:** Primary implementation of the **V7 Hybrid Score-Density** downsampler.
* **Key Functions:**
  * `calculate_v5_scores(xyz, inverse, counts)`: Vectorized re-implementation of V5 geometric complexity and sparsity scores across voxels.
  * `sample_v7_indices(xyz, target_points, density_weight, voxel_size=0.8)`: Normalizes V5 score to mean 1, normalizes density to mean 1, blends them via $w_v = (1 - \lambda) Q_v^{\text{norm}} + \lambda D_v^{\text{norm}}$, and solves quotas via `allocate_quotas`.
* **Inputs:** Raw $(N, 3)$ coordinate array, target point count $K$, blending parameter $\lambda$ (`density_weight`).
* **Outputs:** Array of $K$ integer indices into the input array.
* **Dependencies:** `numpy`, `v6_density_sampler.allocate_quotas`, `argparse`, `pathlib`, `csv`.
* **Relationships:** Ingested by `export_v6_v7_validation.py`.

---

### 2. Dataset Export & Generation Scripts

#### 2.1 `export_hybrid_v5.py`
* **Path:** `e:\Projects\adaptive-lidar\export_hybrid_v5.py`
* **Role:** Batch generation script processing original KITTI validation frames and exporting binary `.bin` files for Hybrid V5.
* **Processing Logic:** Loads raw binary float32 XYZI points from `data/object_detection/training/velodyne/`, computes target $K = \lfloor N / 5 \rfloor$, calls `hybrid_sample_v5`, maps points back to XYZI via coordinate tuple lookup, and writes output to `data/detector_sampling/hybrid_v5/velodyne/`.
* **Outputs:** 3,769 `.bin` files formatted as packed float32 $(x, y, z, I)$.

#### 2.2 `export_v6_v7_validation.py`
* **Path:** `e:\Projects\adaptive-lidar\export_v6_v7_validation.py`
* **Role:** High-throughput batch exporter generating full validation sets for V6 ($\alpha=0.25$) and V7 ($\lambda=0.25$).
* **Processing Logic:** Reads official frame IDs from `ImageSets/val.txt`, invokes `sample_v6_indices` and `sample_v7_indices`, directly indexes raw arrays via `raw[selected_indices]`, and writes `.bin` files atomically using temporary files (`.bin.tmp`).
* **Outputs:** 3,769 `.bin` files in `data/detector_sampling/v6_alpha_0.25/velodyne/` and 3,769 `.bin` files in `data/detector_sampling/v7_lambda_0.25/velodyne/`.

#### 2.3 `export_detector_clouds.py`
* **Path:** `e:\Projects\adaptive-lidar\export_detector_clouds.py`
* **Role:** Legacy export script utilized during earlier prototype iterations (Uniform, V2, V3, V4).

---

### 3. Diagnostic & Retention Analysis Scripts

#### 3.1 `analyse_v5_v6_v7_retention_diagnostics.py`
* **Path:** `e:\Projects\adaptive-lidar\analyse_v5_v6_v7_retention_diagnostics.py`
* **Role:** Core post-hoc diagnostic script evaluating point retention across the full 3,769 validation split.
* **Key Functions:**
  * `read_calibration(path)`: Parses `Tr_velo_to_cam` and `R0_rect` from KITTI calibration files.
  * `lidar_to_camera(xyz, rotation, translation)`: Projects 3D points from LiDAR to rectified camera space.
  * `read_boxes(path)`: Ingests ground-truth bounding boxes for Car, Pedestrian, and Cyclist.
  * `points_in_box(points_cam, box)`: Geometric point-in-oriented-bounding-box containment test.
  * `distance_group(z)`: Bins objects by depth into Near ($0-20\,\text{m}$), Medium ($20-40\,\text{m}$), and Far ($>40\,\text{m}$).
* **Outputs:** Writes `results/distance_retention_v5_v6_v7_comparison.csv`, `results/object_retention_distribution_v5_v6_v7_comparison.csv`, `results/object_retention_details_v5_v6_v7_comparison.csv`, and `results/global_retention_summary_v5_v6_v7_comparison.csv`.

#### 3.2 `diagnose_v5_voxel_allocation.py` & `diagnose_v5_density_retention.py`
* **Path:** `e:\Projects\adaptive-lidar\diagnose_v5_voxel_allocation.py`, `e:\Projects\adaptive-lidar\diagnose_v5_density_retention.py`
* **Role:** Diagnostic scripts analyzing voxel occupancy distributions, identifying why Hybrid V5 caused severe pedestrian starvation.

#### 3.3 `trace_v5_selection.py`
* **Path:** `e:\Projects\adaptive-lidar\trace_v5_selection.py`
* **Role:** Micro-level inspection script logging individual point selection decisions (representatives vs. secondary vs. score fill) across specific test frames.

---

### 4. Comparative Benchmark & Evaluation Scripts

#### 4.1 `compare_v5_v6_v7_retention_50.py`
* **Path:** `e:\Projects\adaptive-lidar\compare_v5_v6_v7_retention_50.py`
* **Role:** Compares class-wise retention across parameter sweeps: V5, V6 ($\alpha \in \{0.0, 0.25, 0.5, 0.75\}$), and V7 ($\lambda \in \{0.25, 0.5, 0.75\}$) on the first 50 validation frames.

#### 4.2 `compare_v5_v6_voxel_allocation.py`
* **Path:** `e:\Projects\adaptive-lidar\compare_v5_v6_voxel_allocation.py`
* **Role:** Measures quota distribution across occupancy bins ($1-5, 6-10, 11-20, 21-50, 51+$ points) comparing V5 and V6.

#### 4.3 `class_retention_detector_clouds.py` & `class_retention_all_methods.py`
* **Path:** `e:\Projects\adaptive-lidar\class_retention_detector_clouds.py`
* **Role:** Baseline comparison scripts evaluating object point retention across earlier algorithm generations (Uniform, Hybrid V2, V3, V4, V5).

---

### 5. Backend Application Architecture (FastAPI)

#### 5.1 `run_backend.py`
* **Path:** `e:\Projects\adaptive-lidar\run_backend.py`
* **Role:** Production application launch entrypoint. Configures Uvicorn server, handles host/port arguments, and mounts FastAPI.

#### 5.2 `app/backend/server.py`
* **Path:** `e:\Projects\adaptive-lidar\app\backend\server.py`
* **Role:** Core FastAPI REST API application.
* **Key Endpoints:**
  * `GET /api/health`: System status, loaded validation frame counts, cache states.
  * `GET /api/frames`: Full list of 3,769 validation frame IDs.
  * `GET /api/algorithms`: Metadata, target budgets, and mathematical formulas for all algorithms.
  * `GET /api/point-cloud/{frame_id}/{method}`: Point coordinates and true counts in JSON format.
  * `GET /api/point-cloud-bin/{frame_id}/{method}`: Packed IEEE-754 binary streaming (`Float32Array`).
  * `GET /api/point-cloud-multi/{frame_id}`: Multi-method payload for split comparison.
  * `GET /api/ground-truth/{frame_id}`: Calibrated 3D wireframe bounding boxes.
  * `GET /api/metrics/*`: Global retention, distance-wise retention, starvation distribution queries.

#### 5.3 `app/backend/kitti_loader.py`
* **Path:** `e:\Projects\adaptive-lidar\app\backend\kitti_loader.py`
* **Role:** High-speed binary point-cloud reader equipped with memory-mapping and an in-memory LRU cache. Enforces client-side rendering caps while preserving exact true point counts in headers.

#### 5.4 `app/backend/calib_service.py`
* **Path:** `e:\Projects\adaptive-lidar\app\backend\calib_service.py`
* **Role:** Inverts camera-to-LiDAR coordinate frames. Generates 8-corner 3D bounding box vertices for direct WebGL rendering.

#### 5.5 `app/backend/ground_truth_loader.py`
* **Path:** `e:\Projects\adaptive-lidar\app\backend\ground_truth_loader.py`
* **Role:** Parses `data/object_detection/training/label_2/*.txt` ground-truth annotations, filters target classes, and computes inside-box point counts.

#### 5.6 `app/backend/metrics_service.py`
* **Path:** `e:\Projects\adaptive-lidar\app\backend\metrics_service.py`
* **Role:** Query engine reading pre-computed experimental CSV files in `results/`, serving aggregated metrics to the frontend charts.

#### 5.7 `app/backend/config.py`
* **Path:** `e:\Projects\adaptive-lidar\app\backend\config.py`
* **Role:** Configuration file defining filesystem paths, algorithm metadata, distance bin definitions, and server defaults.

---

### 6. Frontend Application Architecture (React + Three.js)

#### 6.1 `app/frontend/src/App.jsx`
* **Path:** `e:\Projects\adaptive-lidar\app\frontend\src\App.jsx`
* **Role:** Root component. Manages application state: active tab, selected frame, selected algorithm, sequential playback clock ticker, global keyboard shortcuts (<kbd>Space</kbd>, <kbd>Arrows</kbd>, <kbd>P</kbd>), and presentation mode rendering.

#### 6.2 `app/frontend/src/api.js`
* **Path:** `e:\Projects\adaptive-lidar\app\frontend\src\api.js`
* **Role:** Client-side HTTP client. Implements in-memory LRU caches (`pointCache`, `gtCache`, `metricsCache`) to make frame scrubbing and looping instant (<0.1ms).

#### 6.3 `app/frontend/src/components/ThreePointCloudViewer.jsx`
* **Path:** `e:\Projects\adaptive-lidar\app\frontend\src\components\ThreePointCloudViewer.jsx`
* **Role:** WebGL 3D point cloud and wireframe renderer using Three.js.
* **Capabilities:** OrbitControls, camera synchronization across multi-viewports, color modes (Elevation, Intensity, Radial Distance, Solid Color), camera view presets (3D Perspective, Bird's Eye, Front, Side), and 3D bounding box wireframe rendering.

#### 6.4 `app/frontend/src/components/PlaybackToolbar.jsx`
* **Path:** `e:\Projects\adaptive-lidar\app\frontend\src\components\PlaybackToolbar.jsx`
* **Role:** Media transport bar providing Play, Pause, Step Back, Step Forward, Restart, Speed Multipliers (0.5x to 8x), Loop Toggle, and continuous scrubbing slider.

#### 6.5 `app/frontend/src/components/Navigation.jsx`
* **Path:** `e:\Projects\adaptive-lidar\app\frontend\src\components\Navigation.jsx`
* **Role:** Top navigation ribbon. Houses brand identity, view navigation tabs, frame stepper, random frame selector, and presentation mode toggle.

#### 6.6 Views
* `OverviewView.jsx`: High-level research summary, quantitative overview cards, and end-to-end workflow diagram.
* `ExplorerView.jsx`: Dominant 3D point-cloud exploration workstation with live telemetry and collapsible controls.
* `ComparisonView.jsx`: Synchronized multi-viewport comparison console supporting Dual-Split and Quad-Grid views.
* `ResearchAnalyticsView.jsx`: Chart.js interactive analytics dashboard for global, distance-wise, and distribution metrics.
* `DetectorPerformanceView.jsx`: Downstream PointPillars benchmark tables, radar comparison plots, and relative performance indicators.
* `MethodologyView.jsx`: Comprehensive mathematical documentation, calibration derivations, and quota solver explanations.
* `PresentationView.jsx`: Immersive fullscreen presentation cockpit designed for large-screen conference demonstrations.

---

### 7. Experimental Results & Evaluation Artifacts

#### 7.1 OpenPCDet Evaluation Logs (`results/detector_evaluation/logs/`)
* `uniform20.log`: Official PointPillars evaluation log on Uniform 20% validation set (Car Mod AP: 51.14%, Ped: 23.31%, Cyc: 1.18%).
* `v5.log`: Evaluation log on Hybrid V5 validation set (Car Mod AP: 61.38%, Ped: 7.86%, Cyc: 14.54%).
* `v6_alpha_0.25.log`: Evaluation log on V6 $\alpha=0.25$ validation set (Car Mod AP: 66.24%, Ped: 19.42%, Cyc: 14.62%).
* `v7_lambda_0.25.log`: Evaluation log on V7 $\lambda=0.25$ validation set (Car Mod AP: 65.33%, Ped: 16.13%, Cyc: 12.18%).

#### 7.2 Quantitative Comparison CSVs (`results/`)
* `global_retention_summary_v5_v6_v7_comparison.csv`: Summary of 447,411,677 original points and 89,480,814 sampled points across 3,769 frames.
* `distance_retention_v5_v6_v7_comparison.csv`: Distance-binned retention for Near ($0-20\,\text{m}$), Medium ($20-40\,\text{m}$), and Far ($>40\,\text{m}$).
* `object_retention_distribution_v5_v6_v7_comparison.csv`: Severe starvation percentages ($<10\%, <20\%, <40\%, <60\%$) for Car, Pedestrian, and Cyclist.
* `object_retention_details_v5_v6_v7_comparison.csv`: Fine-grained retention statistics for all 17,474 individual annotated 3D objects.
* `logs/v5_v6_v7_full_diagnostic.log`: Full terminal output log verifying diagnostic execution.

---

### 8. Automated Test Suite

#### 8.1 `tests/test_backend.py`
* **Path:** `e:\Projects\adaptive-lidar\tests\test_backend.py`
* **Role:** Automated integration and unit test suite executed via `pytest`.
* **Coverage (13 Tests):**
  1. `test_health_check`: Verifies server status, frame count (3,769), and algorithm registration.
  2. `test_algorithms_metadata`: Validates target retention (20.0%) and voxel size (0.8m).
  3. `test_validation_frames_list`: Tests validation split list and default frame (`000001`).
  4. `test_load_point_cloud_all_methods`: Verifies point loading across Original, Uniform, V5, V6, and V7.
  5. `test_render_downsampling_cap`: Tests client-side display capping.
  6. `test_binary_point_streaming`: Validates IEEE-754 float32 byte streaming and metadata headers.
  7. `test_multi_method_comparison`: Tests multi-cloud comparison payloads.
  8. `test_calib_and_ground_truth`: Validates calibration inversion and 8-corner bounding box vertices.
  9. `test_detector_predictions`: Tests OpenPCDet prediction loading.
  10. `test_metrics_endpoints`: Tests global, distance, and distribution metrics queries.
  11. `test_diverse_validation_frames`: Tests frame loading across diverse frames (`000008, 000084, 000120, 001500`).
  12. `test_frontend_serving`: Verifies HTML single-page app bundle delivery.
  13. `test_invalid_requests`: Verifies 404/422 error handling on nonexistent frames or invalid algorithms.

---

### 9. Configuration & Environment Files

* `requirements.txt`: Python package requirements (`fastapi`, `uvicorn`, `numpy`, `pandas`, `pydantic`, `pytest`, `httpx`).
* `README.md`: High-level project summary and quickstart guide.
* `app/frontend/package.json`: Node.js manifest specifying dependencies (`react`, `three`, `chart.js`, `lucide-react`, `tailwindcss`, `vite`).
* `app/frontend/vite.config.js`: Vite build configuration.
* `app/frontend/tailwind.config.js`: Tailwind CSS theme configuration defining the dark graphite research palette.
* `PROJECT_TECHNICAL_REFERENCE.md`: Primary comprehensive scientific reference document.
* `PROJECT_FILE_INDEX.md`: This comprehensive file catalog.
* `DOCUMENTATION_AUDIT.md`: Verification audit report of codebase inspection and findings.
