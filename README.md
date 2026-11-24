# Adaptive LiDAR Lab: Interactive Research Visualization & Analytics Platform

**Adaptive LiDAR Lab** is an interactive, scientific visualization and quantitative evaluation platform designed for autonomous driving and 3D computer vision researchers. The application interactively explores, compares, and evaluates point-cloud downsampling algorithms on the **KITTI 3D Object Detection Benchmark** under a strict **20% point retention budget**, paired with a fixed pretrained **PointPillars** 3D detection network.

---

## 1. Research Background & Mathematical Formulation

### The Autonomous Driving Bottleneck
Modern mechanical and solid-state LiDAR sensors generate 100,000+ points per frame at 10–20 Hz, imposing heavy transmission and processing loads on embedded compute platforms. While uniform random downsampling is simple, it preserves point density proportional to range squared ($1/r^2$), systematically starving distant or thin objects (cyclists, pedestrians) while preserving redundant near-field asphalt.

### Core Principle: Unsupervised Downsampling vs. Post-Hoc Evaluation
* **Unsupervised Sampling**: All downsampling algorithms operate strictly without labels, without bounding boxes, and without detector feedback. Input is raw $(x, y, z, \text{intensity})$ points discretized onto a 3D voxel grid ($0.8\,\text{m} \times 0.8\,\text{m} \times 0.8\,\text{m}$).
* **Post-Hoc Evaluation**: Pretrained PointPillars (trained on full point clouds) is evaluated on downsampled clouds across the official KITTI validation split (**3,769 frames**) to measure downstream Moderate $\text{AP}_{\text{R40}}$ without model retraining.

### Evaluated Sampling Algorithms

| Algorithm | Strategy | Weight / Scoring Function | Budget Quota Solver | Moderate mAP |
| :--- | :--- | :--- | :--- | :--- |
| **Full Point Cloud** | Unsampled Raw | None (100% points) | None | **50.03%** |
| **Uniform 20%** | Random Baseline | $p_i \sim \mathcal{U}(0, 1) \le 0.20$ | Bernoulli selection | **25.21%** |
| **Hybrid V5** | Complexity & Sparsity | $S_v = \frac{\sigma_{\text{dist}}(P_v)}{\sqrt{N_v}}$ | 3-stage greedy fill | **27.93%** |
| **V6 $\alpha=0.25$** | Density-Aware Adaptive | $W_v = (N_v)^{0.25}$ | Hamilton-Webster integer allocation | **33.43%** |
| **V7 $\lambda=0.25$** | Hybrid Score-Density | $W_v = \lambda \hat{S}_v + (1 - \lambda) \hat{N}_v^{0.25}$ | Hamilton-Webster integer allocation | **31.21%** |

#### Mathematical Details:
1. **Uniform 20%**: Points sampled independently at random. Demonstrates catastrophic failure on Cyclists ($1.18\%$ AP vs. $33.39\%$ full cloud).
2. **Hybrid V5**: Divides cloud into 0.8m voxels. Stage 1 guarantees 1 representative point per voxel; Stage 2 grants a second point for sparse voxels ($N_v \le 4$); Stage 3 greedily allocates remaining quota by descending complexity score $S_v$. Strongly preserves Cars ($61.38\%$ AP) but suffers from quota starvation on Pedestrians ($7.86\%$ AP).
3. **Density-Aware V6 ($\alpha=0.25$)**: Sub-linear power damping suppresses foreground ground clusters while preserving small clusters. Integer quotas are solved via the **Hamilton-Webster largest remainders method**:
   $$q_v = K \cdot \frac{W_v}{\sum_u W_u}, \quad k_v = \min(N_v, \lfloor q_v \rfloor)$$
   Remaining deficit $R = K - \sum k_v$ is awarded to voxels with highest fractional remainders $r_v = q_v - \lfloor q_v \rfloor$. Achieves highest Car AP (**66.24%**) and top mAP (**33.43%**).
4. **Hybrid Score-Density V7 ($\lambda=0.25$)**: Convex fusion balancing 25% geometric boundary curvature and 75% sub-linear density allocation. Retains $65.33\%$ Car AP and $16.13\%$ Pedestrian AP.

---

## 2. Quantitative Benchmark Results

All metrics are derived directly from the pre-computed evaluation logs and comparison CSVs on the 3,769 validation frames:

### Downstream Moderate $\text{AP}_{\text{R40}}$ Detection Results
*Evaluated with fixed pretrained PointPillars on KITTI validation split:*

| Method | Car AP (%) | Pedestrian AP (%) | Cyclist AP (%) | Mean AP (%) | Relative to Full (%) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Full Point Cloud** | 72.9095 | 43.8000 | 33.3889 | 50.0328 | 100.0% |
| **Uniform 20%** | 51.1415 | 23.3063 | 1.1832 | 25.2103 | 50.4% |
| **Hybrid V5** | 61.3816 | 7.8629 | 14.5416 | 27.9287 | 55.8% |
| **V6 $\alpha=0.25$** | **66.2424** | 19.4154 | **14.6179** | **33.4252** | **66.8%** |
| **V7 $\lambda=0.25$** | 65.3325 | 16.1286 | 12.1795 | 31.2135 | 62.4% |

### Distance-Wise Object Point Retention (Cyclist Class)
*Retention percentage across distance bins:*
* **Near (0–20m)**: Uniform 20.08% | Hybrid V5 27.18% | V6 28.73% | V7 28.53%
* **Medium (20–40m)**: Uniform 20.18% | Hybrid V5 24.23% | V6 26.54% | V7 26.13%
* **Far (>40m)**: Uniform 20.21% | Hybrid V5 26.04% | V6 27.70% | V7 27.42%

### Critical Severe Starvation Rates (<10% Points Retained)
*Percentage of objects retaining less than 10% of their original points:*
* **Car**: Uniform 0.00% | Hybrid V5 0.35% | V6 0.05% | V7 0.08%
* **Pedestrian**: Uniform 0.00% | Hybrid V5 **53.30%** (severe starvation) | V6 1.83% | V7 4.70%
* **Cyclist**: Uniform 0.00% | Hybrid V5 2.15% | V6 0.00% | V7 0.00%

---

## 3. Sensor Calibration & 3D Bounding Box Math

KITTI annotations in `label_2` are defined in Camera 2 rectified coordinates ($y$ downward, bottom-center origin). The backend transforms these into Velodyne LiDAR space ($x$ forward, $y$ left, $z$ up, geometric centroid) using camera calibration parameters:

1. **Camera Centroid Shift**:
   $$\mathbf{p}_{\text{cam}} = \begin{bmatrix} x \\ y - h / 2 \\ z \end{bmatrix}$$
2. **LiDAR Centroid Inversion**:
   $$\mathbf{p}_{\text{lidar}} = (\mathbf{p}_{\text{cam}} - \mathbf{t}_{\text{velo}\to\text{cam}}) \cdot (\mathbf{R}_0^{\text{rect}} \cdot \mathbf{R}_{\text{velo}\to\text{cam}})^{-\top}$$
3. **LiDAR Heading Yaw**:
   $$\theta_{\text{lidar}} = -\text{rotation\_y} - \frac{\pi}{2}$$
4. **8-Corner Vertex Wireframe**:
   $$\mathbf{V}_{\text{local}} = \left[ \pm \frac{l}{2}, \pm \frac{w}{2}, \pm \frac{h}{2} \right], \quad \mathbf{V}_{\text{lidar}} = \mathbf{V}_{\text{local}} \cdot \mathbf{R}_z(\theta_{\text{lidar}})^\top + \mathbf{p}_{\text{lidar}}$$

---

## 4. System Architecture

```
adaptive-lidar/
├── app/
│   ├── backend/                      # High-performance FastAPI server
│   │   ├── config.py                 # Paths, method metadata, distance bins, cache
│   │   ├── kitti_loader.py           # Binary float32 XYZI loader, LRU cache, render caps
│   │   ├── calib_service.py          # Matrix inversion, 8-corner wireframe generation
│   │   ├── ground_truth_loader.py    # label_2 parser, class filtering, point containment
│   │   ├── detector_loader.py        # PointPillars result.pkl parser
│   │   ├── metrics_service.py        # CSV query engine (global, frame, distance, distribution)
│   │   └── server.py                 # REST & binary streaming endpoints, static bundle mount
│   └── frontend/                     # React 18 + Vite 6 + Tailwind 3 + Three.js
│       ├── src/
│       │   ├── components/
│       │   │   ├── ThreePointCloudViewer.jsx   # WebGL GPU point cloud & 3D wireframe renderer
│       │   │   └── Navigation.jsx              # Frame selector, stepper, search, random jump
│       │   ├── views/
│       │   │   ├── OverviewView.jsx            # Research hero, stats, and workflow pipeline
│       │   │   ├── ExplorerView.jsx            # Primary 3D viewer with telemetry & object drawer
│       │   │   ├── ComparisonView.jsx          # Dual/Quad multi-view with synced OrbitControls
│       │   │   ├── SamplingAnalysisView.jsx    # Distance retention, distribution, global stats
│       │   │   ├── DetectorPerformanceView.jsx # Moderate AP_R40 charts, mAP radar, table
│       │   │   └── MethodologyView.jsx         # Theoretical formulations & calibration math
│       │   ├── App.jsx                         # Main shell, view routing, telemetry sync
│       │   └── index.css                       # Dark theme scientific styling & custom scrollbars
│       ├── dist/                               # Production compiled frontend bundle
│       └── package.json
├── data/
│   ├── object_detection/training/    # Original velodyne/, label_2/, calib/
│   └── detector_sampling/            # Downsampled velodyne/ (uniform, hybrid_v5, v6, v7)
├── results/                          # Pre-computed evaluation comparison CSVs
├── tests/
│   └── test_backend.py               # 13 automated unit & integration tests
├── run_backend.py                    # Server launch script
└── requirements.txt                  # Python dependencies
```

---

## 5. Installation & Setup

### Prerequisites
* **Python**: 3.10+ (tested on Python 3.10.11)
* **Node.js**: v18+ (tested on Node.js v24.20.0 / npm 11.19.0)
* **Hardware**: Modern GPU with WebGL 2.0 support (NVIDIA / AMD / Apple Silicon / Intel Iris)

### 1. Python Environment Setup
```powershell
# Create and activate virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install required Python packages
pip install -r requirements.txt
```

### 2. Frontend Build (Optional if using pre-compiled `dist/`)
```powershell
cd app/frontend
npm install
npm run build
cd ../..
```

---

## 6. How to Run

### Production Single-Server Mode (Recommended)
The backend server automatically serves the compiled frontend bundle at `http://localhost:8000`:
```powershell
.\.venv\Scripts\python.exe run_backend.py --host 127.0.0.1 --port 8000
```
Open **`http://localhost:8000`** in Google Chrome, Microsoft Edge, or Mozilla Firefox.

### Development Mode with Live Hot-Reloading
To develop frontend and backend concurrently:
```powershell
# Terminal 1 - FastAPI Backend:
.\.venv\Scripts\python.exe run_backend.py --host 127.0.0.1 --port 8000 --reload

# Terminal 2 - Vite Dev Server:
cd app/frontend
npm run dev
```
Access the Vite dev server at `http://localhost:5173`.

---

## 7. API Reference

### Health & Metadata
* `GET /api/health`: Server status, loaded validation frame count, and cache status.
* `GET /api/frames`: List of all 3,769 validation frame IDs (`["000001", ...]`).
* `GET /api/algorithms`: Supported algorithms with metadata, mathematical formula, and descriptions.

### Point Clouds & 3D Geometry
* `GET /api/point-cloud/{frame_id}/{method}?max_points=50000`: JSON point array, intensity, true counts, and retention percentage.
* `GET /api/point-cloud-bin/{frame_id}/{method}?max_points=50000`: High-speed binary streaming (`Float32Array` of $[x, y, z, \text{intensity}]$) via custom headers `X-Original-Points` and `X-Sampled-Points`.
* `GET /api/point-cloud-multi/{frame_id}?methods=original,v6_alpha_0.25`: Simultaneous multi-method payload for split comparison.
* `GET /api/ground-truth/{frame_id}?classes=Car,Pedestrian,Cyclist`: Calibrated 3D bounding boxes, LiDAR centroid, dimensions, yaw, and inside-box point counts.
* `GET /api/detector-predictions/{frame_id}/{method}`: PointPillars predicted 3D bounding boxes and detection confidence scores.

### Research Analytics & CSV Queries
* `GET /api/metrics/global`: Global point retention summary across all 3,769 frames.
* `GET /api/metrics/frame/{frame_id}`: Frame-specific point retention comparison across all algorithms.
* `GET /api/metrics/distance-retention`: Distance-binned retention for Near ($0-20\,\text{m}$), Medium ($20-40\,\text{m}$), and Far ($>40\,\text{m}$).
* `GET /api/metrics/distribution-retention`: Object starvation percentages under $<10\%$, $<20\%$, $<40\%$, $<60\%$ point retention.
* `GET /api/metrics/detector-ap`: Moderate $\text{AP}_{\text{R40}}$ benchmark results, class APs, and relative retention ratios.

---

## 8. Test Suite Verification

Run the automated backend test suite:
```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_backend.py -v
```

### Verification Results
```
tests/test_backend.py::test_health_check PASSED                          [  7%]
tests/test_backend.py::test_algorithms_metadata PASSED                   [ 15%]
tests/test_backend.py::test_validation_frames_list PASSED                [ 23%]
tests/test_backend.py::test_load_point_cloud_all_methods PASSED          [ 30%]
tests/test_backend.py::test_render_downsampling_cap PASSED               [ 38%]
tests/test_backend.py::test_binary_point_streaming PASSED                [ 46%]
tests/test_backend.py::test_multi_method_comparison PASSED               [ 53%]
tests/test_backend.py::test_calib_and_ground_truth PASSED                [ 61%]
tests/test_backend.py::test_detector_predictions PASSED                  [ 69%]
tests/test_backend.py::test_metrics_endpoints PASSED                     [ 76%]
tests/test_backend.py::test_diverse_validation_frames PASSED             [ 84%]
tests/test_backend.py::test_frontend_serving PASSED                      [ 92%]
tests/test_backend.py::test_invalid_requests PASSED                      [100%]

======================= 13 passed in 2.74s =======================
```

---

## 9. Known Limitations & Research Insights

1. **Validation Split Scope**:
   * Point clouds for **V6 $\alpha=0.25$** and **V7 $\lambda=0.25$** were generated specifically for the official 3,769 validation frames specified in `ImageSets/val.txt`.
   * For training-only frames outside the validation split, the API safely falls back or flags availability while keeping the 3,769 evaluation set complete.
2. **Client-Side WebGL Rendering Caps**:
   * To maintain 60 FPS in interactive browser rendering, the viewer defaults to a display cap of 50,000 points.
   * Users can adjust the rendering cap to 20,000, 50,000, 100,000, or **Full (uncapped)** directly from the viewer controls.
   * Telemetry headers always display the **exact, unadulterated true point counts** (e.g., 121,438 original points vs. 24,287 sampled points).
3. **Detector Scope**:
   * Pretrained model evaluations represent PointPillars architecture. Retraining PointPillars directly on downsampled sets or evaluating with voxel-based detectors (e.g. SECOND, Part-$A^2$, PV-RCNN) represents valuable future work.

---

## 10. Research Presentation Guide

When presenting this research project to a faculty advisor, committee, or conference audience:
1. **Start with Overview**: Highlight the 20% budget constraint and the trade-off between edge bandwidth and detection accuracy.
2. **Demonstrate Failure of Uniform Sampling (Explorer / Comparison)**:
   - Load frame `000008` or `000084` with a distant cyclist.
   - Switch between **Original** and **Uniform 20%** — observe how the cyclist's points vanish into single-digit noise.
   - Switch to **V6 $\alpha=0.25$** — observe how the cyclist's structural silhouette is preserved while near-field road points are cleanly thinned.
3. **Walk Through Dual/Quad View Synchronization**:
   - Navigate to **Algorithm Comparison**.
   - Select Quad View (Original, Uniform, V5, V6). Rotate the camera and demonstrate synchronized WebGL viewports.
4. **Present the Moderate $\text{AP}_{\text{R40}}$ Quantitative Table**:
   - Navigate to **Detector Performance**.
   - Show how V6 recovers **66.24% Car AP** (+15.1% over Uniform) and boosts **mAP to 33.43%** (+8.2% over Uniform).
   - Explain why V5 failed on pedestrians ($7.86\%$) due to greedy complexity starvation.
5. **Conclude with Methodology**:
   - Navigate to **Research Methodology** to demonstrate the Hamilton-Webster integer quota solver and mathematical rigor.

