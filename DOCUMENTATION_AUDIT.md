# Adaptive LiDAR Lab: Technical Audit & Documentation Verification Report

**Audit Date:** November 2026  
**Auditor Role:** Senior Research Engineer, Technical Documentation Specialist, & CV Researcher  
**Target Repository:** `/mnt/e/Projects/adaptive-lidar` (`E:\Projects\adaptive-lidar`)  
**Associated Artifacts Created:**
1. `PROJECT_TECHNICAL_REFERENCE.md` (Primary Scientific Reference Document)
2. `PROJECT_FILE_INDEX.md` (Complete Source & Asset Directory Catalog)
3. `DOCUMENTATION_AUDIT.md` (This Audit & Verification Record)

---

## 1. Scope of Codebase Inspection

The codebase was audited using targeted file-by-file inspection, structural AST analysis, and numerical verification scripts.

### 1.1 Inspected Directories
* **Project Root (`.` / `/mnt/e/Projects/adaptive-lidar/`)**: Inspected all implementation files, exporters, diagnostic scripts, tests, dependencies, and top-level documentation.
* **Backend Layer (`app/backend/`)**: Audited `server.py`, `config.py`, `kitti_loader.py`, `calib_service.py`, `ground_truth_loader.py`, `detector_loader.py`, and `metrics_service.py`.
* **Frontend Layer (`app/frontend/`)**: Audited React architecture, `App.jsx`, `api.js`, all UI components (`Navigation.jsx`, `PlaybackToolbar.jsx`, `ThreePointCloudViewer.jsx`), and all application views (`OverviewView.jsx`, `ExplorerView.jsx`, `ComparisonView.jsx`, `ResearchAnalyticsView.jsx`, `DetectorPerformanceView.jsx`, `MethodologyView.jsx`, `PresentationView.jsx`).
* **Experimental Artifacts (`results/`)**: Inspected all CSV files (`global_retention_summary_*.csv`, `distance_retention_*.csv`, `object_retention_*.csv`, `v5_stage_retention_*.csv`), terminal logs (`logs/v5_v6_v7_full_diagnostic.log`), and OpenPCDet evaluation logs (`results/detector_evaluation/logs/*.log`).
* **Data Organization (`data/`)**: Verified directory structure and sample frames in `data/object_detection/training/` (`velodyne/`, `calib/`, `label_2/`) and downsampled sets in `data/detector_sampling/` (`uniform/`, `hybrid_v5/`, `v6_alpha_0.25/`, `v7_lambda_0.25/`).

### 1.2 Excluded from Inspection (Per Guidelines)
* Virtual environments: `.venv/`, `venv/`.
* Node modules: `app/frontend/node_modules/`.
* Git metadata: `.git/`.
* Compiled caches: `__pycache__/`, `.pytest_cache/`.
* Full binary point-cloud datasets (recursive scanning was avoided; directory existence, frame counts, and binary headers were sampled and validated via targeted scripts).

---

## 2. Verification of Mathematical Formulas and Implementation Logic

| Mathematical Component | Implementation File & Line | Claimed Formula | Verified Code Reality | Audit Status |
| :--- | :--- | :--- | :--- | :--- |
| **Voxel Discretization** | `hybrid_sampling_v5.py` L69, `v6_density_sampler.py` L93 | $g_i = \lfloor p_i / v \rfloor, v = 0.8$ | `np.floor(xyz / voxel_size).astype(np.int64)` | **VERIFIED** |
| **Voxel Centroid** | `hybrid_sampling_v5.py` L93, `v6_density_sampler.py` L129 | $c_v = \frac{1}{n_v} \sum p_i$ | `points.mean(axis=0)` | **VERIFIED** |
| **Representative Selection** | `hybrid_sampling_v5.py` L98-99, `v6_density_sampler.py` L136-137 | $p_v = \arg\min \|p - c_v\|^2$ | `np.argmin(d2)` | **VERIFIED** |
| **Geometric Complexity ($C_v$)** | `hybrid_sampling_v5.py` L104-105, `v7_score_density_sampler.py` L72-74 | $C_v = \sqrt{\frac{1}{n_v} \sum \|p - c_v\|^2}$ | `np.sqrt(np.maximum(mean_distances_sq, 0.0))` | **VERIFIED** |
| **Voxel Sparsity ($S_v$)** | `hybrid_sampling_v5.py` L109, `v7_score_density_sampler.py` L76-78 | $S_v = 1 / \sqrt{n_v}$ | `1.0 / np.sqrt(counts.astype(np.float64))` | **VERIFIED** *(See Inconsistency 1)* |
| **V5 Combined Score ($Q_v$)** | `hybrid_sampling_v5.py` L117, `v7_score_density_sampler.py` L90 | $Q_v = 0.65 \hat{S}_v + 0.35 \hat{C}_v$ | `0.65 * sparsity + 0.35 * complexity` | **VERIFIED** |
| **V6 Density Weight ($w_v$)** | `v6_density_sampler.py` L156 | $w_v = n_v^\alpha, \alpha=0.25$ | `np.power(counts.astype(np.float64), alpha)` | **VERIFIED** |
| **V7 Score-Density Blend** | `v7_score_density_sampler.py` L160-174 | $w_v = (1 - \lambda) Q_v^{\text{norm}} + \lambda D_v^{\text{norm}}$ | `(1.0 - lambda) * score_weights + lambda * density_weights` | **VERIFIED** *(See Inconsistency 2)* |
| **Hamilton-Webster Allocator** | `v6_density_sampler.py` L21-72 | Capacity-constrained largest remainders | Solves floor quotas, tracks capacity $c_v = n_v - 1$, and allocates fractional remainders via `np.lexsort` | **VERIFIED** |
| **Target Budget ($K$)** | `export_hybrid_v5.py` L28, `export_v6_v7_validation.py` L45 | $K = \lfloor N / 5 \rfloor$ | `len(raw) // 5` | **VERIFIED** |
| **Calibration Inversion** | `app/backend/calib_service.py` L75-92 | LiDAR to camera coordinate transform | $\mathbf{p}_{\text{cam}} = \mathbf{p}_{\text{velo}} (\mathbf{R}_0^{\text{rect}} \mathbf{R}_{\text{velo}\to\text{cam}})^\top + \mathbf{R}_0^{\text{rect}} \mathbf{t}_{\text{velo}\to\text{cam}}$ | **VERIFIED** |
| **Point-in-Box Test** | `analyse_v5_v6_v7_retention_diagnostics.py` L106-123 | Local coordinate bounding box test | Computes heading rotation yaw, translates by bottom-center $h/2$ offset, and tests $|x| \le l/2, |y| \le h/2, |z| \le w/2$ | **VERIFIED** |

---

## 3. Verification of Experimental Results and Numerical Integrity

### 3.1 Global Point Retention Metrics
*Source File:* `results/global_retention_summary_v5_v6_v7_comparison.csv`

| Metric Field | Documented / Expected Value | Verified Value in Primary CSV | Audit Verification |
| :--- | :---: | :---: | :---: |
| **Total Validation Frames** | 3,769 | 3,769 | **CONFIRMED** |
| **Total Original Points** | 447,411,677 | 447,411,677 | **CONFIRMED** |
| **Total Sampled Points (Uniform)** | 89,480,814 | 89,480,814 | **CONFIRMED** |
| **Total Sampled Points (V5)** | 89,480,814 | 89,480,814 | **CONFIRMED** |
| **Total Sampled Points (V6 $\alpha=0.25$)** | 89,480,814 | 89,480,814 | **CONFIRMED** |
| **Total Sampled Points (V7 $\lambda=0.25$)** | 89,480,814 | 89,480,814 | **CONFIRMED** |
| **Overall Weighted Retention %** | 19.999659955% | 19.999659955231788% | **CONFIRMED** |
| **Mean Frame Retention %** | 19.999659428% | 19.999659427924374% | **CONFIRMED** |
| **Median Frame Retention %** | 19.999662467% | 19.999662467196032% | **CONFIRMED** |
| **Minimum Frame Retention %** | 19.999063276% | 19.99906327572479% | **CONFIRMED** |
| **Maximum Frame Retention %** | 20.000000000% | 20.0% | **CONFIRMED** |
| **Frames Exactly 20.000%** | 1,465 | 1,465 | **CONFIRMED** |

### 3.2 PointPillars Detector Evaluation Metrics ($\text{AP}_{\text{R40}}$)
*Source Files:* `results/detector_evaluation/logs/uniform20.log`, `v5.log`, `v6_alpha_0.25.log`, `v7_lambda_0.25.log`

| Method | Target Class | Claimed Moderate 3D AP | Verified Value in Log File | Audit Verification |
| :--- | :--- | :---: | :---: | :---: |
| **Uniform 20%** | Car | 51.1415% | 51.1415% (3D AP_R40 line in `uniform20.log`) | **CONFIRMED** |
| | Pedestrian | 23.3063% | 23.3063% (3D AP_R40 line in `uniform20.log`) | **CONFIRMED** |
| | Cyclist | 1.1832% | 1.1832% (3D AP_R40 line in `uniform20.log`) | **CONFIRMED** |
| **Hybrid V5** | Car | 61.3816% | 61.3816% (3D AP_R40 line in `v5.log`) | **CONFIRMED** |
| | Pedestrian | 7.8629% | 7.8629% (3D AP_R40 line in `v5.log`) | **CONFIRMED** |
| | Cyclist | 14.5416% | 14.5416% (3D AP_R40 line in `v5.log`) | **CONFIRMED** |
| **V6 $\alpha=0.25$** | Car | 66.2424% | 66.2424% (3D AP_R40 line in `v6_alpha_0.25.log`) | **CONFIRMED** |
| | Pedestrian | 19.4154% | 19.4154% (3D AP_R40 line in `v6_alpha_0.25.log`) | **CONFIRMED** |
| | Cyclist | 14.6179% | 14.6179% (3D AP_R40 line in `v6_alpha_0.25.log`) | **CONFIRMED** |
| **V7 $\lambda=0.25$** | Car | 65.3325% | 65.3325% (3D AP_R40 line in `v7_lambda_0.25.log`) | **CONFIRMED** |
| | Pedestrian | 16.1286% | 16.1286% (3D AP_R40 line in `v7_lambda_0.25.log`) | **CONFIRMED** |
| | Cyclist | 12.1795% | 12.1795% (3D AP_R40 line in `v7_lambda_0.25.log`) | **CONFIRMED** |

---

## 4. Discrepancies and Inconsistencies Discovered

During this audit, three technical discrepancies between source code, comments, and previous informal documentation summaries were identified and resolved in the technical reference:

### Inconsistency 1: Code Comment vs. Implementation of Sparsity Scaling in V5
* **Location:** [`hybrid_sampling_v5.py`](file:///e:/Projects/adaptive-lidar/hybrid_sampling_v5.py#L107-L109).
* **Comment Text:** *"Sparse voxels receive higher scores. Log scaling limits the effect of isolated outliers and extremely dense voxels."*
* **Actual Executed Code:**
  ```python
  sparsity = 1.0 / np.sqrt(voxel_sizes.astype(np.float64))
  ```
* **Resolution:** The implementation does **not** execute logarithmic scaling ($\log n_v$). It strictly computes **inverse square-root scaling** ($1/\sqrt{n_v}$). Section 4.5 and Section 3.7 of `PROJECT_TECHNICAL_REFERENCE.md` document the true inverse square-root formula and note the comment discrepancy.

### Inconsistency 2: Summary Documentation vs. Actual Weighting Equation in V7
* **Location:** [`README.md`](file:///e:/Projects/adaptive-lidar/README.md#L24) vs. [`v7_score_density_sampler.py`](file:///e:/Projects/adaptive-lidar/v7_score_density_sampler.py#L160-L174).
* **README Summary Equation:**
  $$W_v = \lambda \hat{S}_v + (1 - \lambda) \hat{N}_v^{0.25}$$
* **Actual Executed Code in V7 Sampler:**
  ```python
  score_weights = scores / (scores.mean() + 1e-12)
  density_weights = counts.astype(np.float64) / (counts.mean() + 1e-12)
  weights = (1.0 - density_weight) * score_weights + density_weight * density_weights
  ```
* **Resolution:** In the actual implementation:
  1. Density is normalized linear density ($n_v / \bar{n}$), **not** power-law density ($n_v^{0.25}$).
  2. The blending parameter $\lambda$ (`density_weight`) weights the density term, while $(1 - \lambda)$ weights the V5 score term:
     $$w_v = (1 - \lambda) Q_v^{\text{norm}} + \lambda D_v^{\text{norm}}$$
     At $\lambda = 0.25$, geometry receives $75\%$ weight and density receives $25\%$ weight. This has been precisely documented in Section 6.2 and 6.3 of `PROJECT_TECHNICAL_REFERENCE.md`.

### Inconsistency 3: Informal Severe Starvation Statistics in Previous Notes
* **Location:** `README.md` lines 57-61 vs. `results/object_retention_distribution_v5_v6_v7_comparison.csv`.
* **Discrepancy:** The previous summary notes in `README.md` stated pedestrian severe starvation (<10% points retained) was 53.30% for Hybrid V5, 1.83% for V6, and 4.70% for V7.
* **Verified Full Validation CSV Reality:**
  In `results/object_retention_distribution_v5_v6_v7_comparison.csv` across all 2,279 annotated pedestrians in the 3,769 validation frames:
  * Hybrid V5: **35.02%** (798 / 2,279 objects).
  * V6 $\alpha=0.25$: **14.79%** (337 / 2,279 objects).
  * V7 $\lambda=0.25$: **14.30%** (326 / 2,279 objects).
* **Explanation:** The earlier informal summary was computed during an early preliminary 50-frame slice (`results/uniform_v5_v6_object_retention_50.csv`), whereas the full 3,769-frame evaluation yielded the true, verified figures reported in Section 11.3 of `PROJECT_TECHNICAL_REFERENCE.md`.

---

## 5. Unverified Claims and Data Limitations

In accordance with Critical Documentation Rule 2 ("Never fabricate; if unavailable, state explicitly"), the following parameters and background files could not be directly verified from the available repository files:

1. **Full-Cloud Checkpoint Training Logs**:
   * While the evaluation logs for all four downsampled methods (`uniform20.log`, `v5.log`, `v6_alpha_0.25.log`, `v7_lambda_0.25.log`) are present and verified in `results/detector_evaluation/logs/`, the original training log for the 40-epoch baseline checkpoint (`checkpoint_epoch_40.pth`) is stored in an external directory (`../output/kitti_models/pointpillar/baseline/`) outside the workspace root.
   * The Full Point Cloud baseline AP metrics ($72.9095\%$ Car, $43.8000\%$ Pedestrian, $33.3889\%$ Cyclist) are cited from project records and frontend constants, but the raw text evaluation log was not available inside `/mnt/e/Projects/adaptive-lidar/results/`.
2. **GPU Inference Hardware for Original OpenPCDet Runs**:
   * The OpenPCDet logs record `CUDA_VISIBLE_DEVICES=ALL` and batch size 4, but the exact physical GPU model string (e.g., RTX 3090, A100, V100) was truncated in the log headers. System memory and hardware requirements in Section 14.1 are therefore specified as recommended operational minimums rather than exact benchmark machine specifications.

---

## 6. Audit Conclusion & Deliverable Validation

All three required deliverables are completed and validated in the project root:
1. [PROJECT_TECHNICAL_REFERENCE.md](file:///e:/Projects/adaptive-lidar/PROJECT_TECHNICAL_REFERENCE.md): 16 comprehensive sections covering research motivation, mathematics, exact code traces of V5, V6, and V7, systematic comparison tables, detector evaluation, verified results, post-hoc diagnostics, WebGL application architecture, reproducibility, limitations, and future work.
2. [PROJECT_FILE_INDEX.md](file:///e:/Projects/adaptive-lidar/PROJECT_FILE_INDEX.md): Complete catalog of files, paths, inputs/outputs, and dependencies.
3. [DOCUMENTATION_AUDIT.md](file:///e:/Projects/adaptive-lidar/DOCUMENTATION_AUDIT.md): This verification report detailing inspected components, confirmed figures, and resolved discrepancies.

The existing codebase, virtual environment, and experimental results have been preserved with zero modifications to source files or evaluation CSVs. The technical reference document is verified as ready for ingestion as a primary technical literature source into Google NotebookLM.
