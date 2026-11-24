import React from 'react';
import {
  BookOpen,
  Cpu,
  Layers,
  GitBranch,
  Binary,
  Compass,
  CheckCircle2,
  Scale,
  Sparkles,
  Target,
} from 'lucide-react';

export default function MethodologyView() {
  return (
    <div className="flex-1 overflow-y-auto p-8 space-y-8 bg-lab-bg text-slate-200">
      {/* Title Header */}
      <div className="border-b border-lab-border pb-6">
        <div className="max-w-3xl space-y-2">
          <div className="flex items-center space-x-2 text-xs font-mono text-sky-400">
            <BookOpen className="w-4 h-4 text-sky-400" />
            <span className="uppercase">Theoretical Foundation & Algorithms</span>
          </div>
          <h1 className="text-xl sm:text-2xl font-semibold tracking-tight text-white">
            Research Methodology & Mathematical Formulation
          </h1>
          <p className="text-xs text-lab-textMuted leading-relaxed">
            Detailed mathematical specification of adaptive point-cloud downsampling algorithms, Hamilton-Webster discrete integer quota allocation,
            and sensor coordinate frame transformation.
          </p>
        </div>
      </div>

      {/* Crucial Research Distinction: Unsupervised Sampling vs Post-Hoc Evaluation */}
      <div className="rounded border border-lab-border bg-lab-surface p-5 space-y-3">
        <div className="flex items-center space-x-2 text-xs font-semibold text-white uppercase tracking-wider">
          <Scale className="w-4 h-4 text-sky-400" />
          <span>Core Architectural Principle: Unsupervised Sampling vs. Post-Hoc Benchmark</span>
        </div>
        <p className="text-xs text-lab-textMuted leading-relaxed">
          A fundamental requirement of this research is that <strong className="text-slate-200">all point downsampling methods operate strictly unsupervised</strong>.
          The algorithms process raw LiDAR range coordinates <code className="text-sky-300 font-mono text-[11px]">(x, y, z, intensity)</code> without access to ground-truth 3D bounding boxes, semantic annotations, or detector feedback.
          Ground truth annotations and the pretrained PointPillars network are employed solely for <strong className="text-slate-200">post-hoc evaluation</strong> on the KITTI validation split (3,769 frames) to quantify downstream object preservation and Moderate AP<sub>R40</sub> detection fidelity.
        </p>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 pt-1">
          <div className="p-3 bg-[#0a0c10] rounded border border-lab-border">
            <div className="text-xs font-semibold text-slate-200 mb-1 flex items-center space-x-1.5">
              <CheckCircle2 className="w-3.5 h-3.5 text-sky-400" />
              <span>Unsupervised Sampling Stage</span>
            </div>
            <p className="text-[11px] text-lab-textMuted leading-normal">
              Input: Raw point clouds from <code className="text-slate-300">data/object_detection/training/velodyne/</code>. Fixed 0.8m voxel discretization. Total budget = 20% of original points. No label leakage or supervised bias.
            </p>
          </div>
          <div className="p-3 bg-[#0a0c10] rounded border border-lab-border">
            <div className="text-xs font-semibold text-slate-200 mb-1 flex items-center space-x-1.5">
              <Target className="w-3.5 h-3.5 text-emerald-400" />
              <span>Downstream Evaluation Stage</span>
            </div>
            <p className="text-[11px] text-lab-textMuted leading-normal">
              Fixed pretrained OpenPCDet PointPillars checkpoint evaluated on downsampled clouds. Measures 3D AP<sub>R40</sub> at moderate difficulty across Car, Pedestrian, and Cyclist classes without retraining.
            </p>
          </div>
        </div>
      </div>

      {/* The 4 Downsampling Formulations */}
      <div className="space-y-4">
        <h2 className="text-sm font-semibold text-white tracking-tight flex items-center space-x-2">
          <Binary className="w-4 h-4 text-sky-400" />
          <span>Point Sampling Formulations & Quota Mechanics</span>
        </h2>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-5">
          {/* Baseline: Uniform */}
          <div className="rounded border border-lab-border bg-lab-surface p-5 space-y-3 flex flex-col justify-between">
            <div className="space-y-3">
              <div className="flex items-center justify-between">
                <span className="text-white text-xs font-semibold flex items-center space-x-2">
                  <span className="w-2 h-2 rounded-full bg-rose-400 inline-block" />
                  <span>1. Uniform 20% Random Baseline</span>
                </span>
                <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-[#161a22] text-lab-textMuted border border-lab-border">
                  Baseline
                </span>
              </div>
              <p className="text-xs text-lab-textMuted leading-relaxed">
                Each point <code className="font-mono text-slate-200 text-[11px]">p_i</code> in the input cloud is sampled independently with probability <code className="font-mono text-slate-200 text-[11px]">p = 0.20</code> via uniform random indexing:
              </p>
              <div className="p-2.5 bg-[#0a0c10] rounded border border-lab-border font-mono text-xs text-slate-300 overflow-x-auto">
                P_sampled = &#123; p_i &isin; P | u_i &le; 0.20, u_i ~ U(0, 1) &#125;
              </div>
              <ul className="text-xs text-lab-textMuted space-y-1.5 list-disc list-inside">
                <li><strong className="text-slate-300">Spatial bias:</strong> Preserves original density proportions, disproportionately retaining dense foreground ground.</li>
                <li><strong className="text-slate-300">Sparse object starvation:</strong> Distant or low-reflectivity objects (cyclists, pedestrians) lose crucial structural points below detector threshold.</li>
                <li><strong className="text-slate-300">Detector outcome:</strong> Cyclist Moderate AP falls to <strong className="text-rose-400 font-mono">1.18%</strong> (vs 33.39% full cloud).</li>
              </ul>
            </div>
            <div className="pt-3 border-t border-lab-border flex items-center justify-between text-xs font-mono text-lab-textDim">
              <span>Budget: 20.00%</span>
              <span>mAP: 25.21%</span>
            </div>
          </div>

          {/* Method 2: Hybrid V5 */}
          <div className="rounded border border-lab-border bg-lab-surface p-5 space-y-3 flex flex-col justify-between">
            <div className="space-y-3">
              <div className="flex items-center justify-between">
                <span className="text-white text-xs font-semibold flex items-center space-x-2">
                  <Layers className="w-4 h-4 text-amber-400" />
                  <span>2. Hybrid V5 (Geometry Complexity & Sparsity)</span>
                </span>
                <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-[#161a22] text-amber-400 border border-lab-border">
                  Geometry-Driven
                </span>
              </div>
              <p className="text-xs text-lab-textMuted leading-relaxed">
                Partitions the point cloud into 0.8m voxels. Computes geometric dispersion relative to voxel centroid, dampened by square-root density:
              </p>
              <div className="p-2.5 bg-[#0a0c10] rounded border border-lab-border font-mono text-xs text-amber-300 overflow-x-auto">
                S_v = &sigma;_dist(P_v) / &radic;(N_v), &nbsp; where &sigma;_dist = std(||p - &mu;_v||)
              </div>
              <ul className="text-xs text-lab-textMuted space-y-1.5 list-disc list-inside">
                <li><strong className="text-slate-300">Stage 1 (Coverage):</strong> Selects 1 representative point closest to centroid for every occupied voxel.</li>
                <li><strong className="text-slate-300">Stage 2 (Sparsity Boost):</strong> Selects a secondary point for sparse voxels with <code className="text-amber-300 font-mono">N_v &le; 4</code>.</li>
                <li><strong className="text-slate-300">Stage 3 (Complexity Fill):</strong> Fills remaining 20% quota by descending geometric score <code className="text-amber-300 font-mono">S_v</code>.</li>
                <li><strong className="text-slate-300">Diagnostic finding:</strong> Excellent Car AP (<strong className="text-amber-400 font-mono">61.38%</strong>) and Cyclist AP (<strong className="text-amber-400 font-mono">14.54%</strong>), but severe Pedestrian degradation (<strong className="text-rose-400 font-mono">7.86%</strong>) due to greedy quota exhaustion.</li>
              </ul>
            </div>
            <div className="pt-3 border-t border-lab-border flex items-center justify-between text-xs font-mono text-lab-textDim">
              <span>Budget: 20.00%</span>
              <span>mAP: 27.93%</span>
            </div>
          </div>

          {/* Method 3: V6 alpha=0.25 */}
          <div className="rounded border border-lab-border bg-lab-surface p-5 space-y-3 flex flex-col justify-between">
            <div className="space-y-3">
              <div className="flex items-center justify-between">
                <span className="text-white text-xs font-semibold flex items-center space-x-2">
                  <Cpu className="w-4 h-4 text-emerald-400" />
                  <span>3. Density-Aware Adaptive V6 (&alpha; = 0.25)</span>
                </span>
                <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-[#161a22] text-emerald-400 border border-lab-border">
                  Power-Law Damping
                </span>
              </div>
              <p className="text-xs text-lab-textMuted leading-relaxed">
                Controls the non-linear relationship between raw voxel point count and allocation quota via an exponential sub-linear damping factor:
              </p>
              <div className="p-2.5 bg-[#0a0c10] rounded border border-lab-border font-mono text-xs text-emerald-400 overflow-x-auto">
                W_v = (N_v)^&alpha; = (N_v)^0.25
              </div>
              <ul className="text-xs text-lab-textMuted space-y-1.5 list-disc list-inside">
                <li><strong className="text-slate-300">Sub-linear weighting:</strong> Dampens near-field ground planes (e.g. 500 points &rarr; 4.73 weight) while protecting distant clusters (e.g. 5 points &rarr; 1.50 weight).</li>
                <li><strong className="text-slate-300">Hamilton-Webster solver:</strong> Integer quotas distributed via largest remainders, guaranteeing exact 20.00% total frame retention without truncation drift.</li>
                <li><strong className="text-slate-300">Detector outcome:</strong> Top overall Car Moderate AP (<strong className="text-emerald-400 font-mono">66.24%</strong>), restored Pedestrian AP (<strong className="text-emerald-400 font-mono">19.42%</strong>), and highest overall mAP (<strong className="text-emerald-400 font-mono">33.43%</strong>).</li>
              </ul>
            </div>
            <div className="pt-3 border-t border-lab-border flex items-center justify-between text-xs font-mono text-lab-textDim">
              <span>Budget: 20.00%</span>
              <span className="text-emerald-400 font-semibold">mAP: 33.43% (Rank 1)</span>
            </div>
          </div>

          {/* Method 4: V7 lambda=0.25 */}
          <div className="rounded border border-lab-border bg-lab-surface p-5 space-y-3 flex flex-col justify-between">
            <div className="space-y-3">
              <div className="flex items-center justify-between">
                <span className="text-white text-xs font-semibold flex items-center space-x-2">
                  <GitBranch className="w-4 h-4 text-sky-400" />
                  <span>4. Hybrid Score-Density V7 (&lambda; = 0.25)</span>
                </span>
                <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-[#161a22] text-sky-400 border border-lab-border">
                  Convex Fusion
                </span>
              </div>
              <p className="text-xs text-lab-textMuted leading-relaxed">
                Formulates a convex combination fusing normalized geometric boundary score (V5) with the non-linear density allocation (V6):
              </p>
              <div className="p-2.5 bg-[#0a0c10] rounded border border-lab-border font-mono text-xs text-sky-400 overflow-x-auto">
                W_v = &lambda; &bull; Norm(S_v) + (1 - &lambda;) &bull; Norm((N_v)^0.25), &nbsp; where &lambda; = 0.25
              </div>
              <ul className="text-xs text-lab-textMuted space-y-1.5 list-disc list-inside">
                <li><strong className="text-slate-300">Convex balance:</strong> 25% weight on surface normal curvature/complexity + 75% weight on stable density damping.</li>
                <li><strong className="text-slate-300">Voxel quota mapping:</strong> Scaled to frame quota via Hamilton-Webster integer allocation.</li>
                <li><strong className="text-slate-300">Detector outcome:</strong> High Car AP (<strong className="text-sky-400 font-mono">65.33%</strong>), 16.13% Pedestrian AP, and 12.18% Cyclist AP. Strong alternative with boundary awareness.</li>
              </ul>
            </div>
            <div className="pt-3 border-t border-lab-border flex items-center justify-between text-xs font-mono text-lab-textDim">
              <span>Budget: 20.00%</span>
              <span>mAP: 31.21%</span>
            </div>
          </div>
        </div>
      </div>

      {/* Hamilton-Webster Integer Allocation Mechanics */}
      <div className="rounded border border-lab-border bg-lab-surface p-5 space-y-4">
        <h3 className="text-sm font-semibold text-white tracking-tight flex items-center space-x-2">
          <Binary className="w-4 h-4 text-sky-400" />
          <span>Hamilton-Webster Largest Remainder Integer Quota Allocation</span>
        </h3>

        <p className="text-xs text-lab-textMuted leading-relaxed">
          In both V6 and V7, continuous weights <code className="text-sky-300 font-mono text-[11px]">W_v</code> produce fractional point quotas that cannot be simply rounded without violating the exact 20% budget constraint or over-allocating points beyond voxel capacity <code className="text-sky-300 font-mono text-[11px]">N_v</code>.
          The exact discrete allocation operates as follows:
        </p>

        <div className="grid grid-cols-1 md:grid-cols-4 gap-4 font-mono text-xs">
          <div className="p-3 bg-[#0a0c10] rounded border border-lab-border">
            <div className="text-sky-400 font-semibold mb-1">Step 1: Continuous Quota</div>
            <div className="text-slate-300 text-[11px]">
              q_v = K &bull; (W_v / &sum;_u W_u)
            </div>
            <div className="text-[10px] text-lab-textDim mt-2">
              Where K = &lfloor;0.20 &bull; N_total&rfloor; is the exact integer budget.
            </div>
          </div>

          <div className="p-3 bg-[#0a0c10] rounded border border-lab-border">
            <div className="text-sky-400 font-semibold mb-1">Step 2: Floor Allocation</div>
            <div className="text-slate-300 text-[11px]">
              k_v = min(N_v, &lfloor;q_v&rfloor;)
            </div>
            <div className="text-[10px] text-lab-textDim mt-2">
              Each voxel receives its integer floor, bounded by its true point count N_v.
            </div>
          </div>

          <div className="p-3 bg-[#0a0c10] rounded border border-lab-border">
            <div className="text-sky-400 font-semibold mb-1">Step 3: Deficit Calculation</div>
            <div className="text-slate-300 text-[11px]">
              R = K - &sum;_v k_v
            </div>
            <div className="text-[10px] text-lab-textDim mt-2">
              Computes the remaining unallocated points (typically 100–500 points).
            </div>
          </div>

          <div className="p-3 bg-[#0a0c10] rounded border border-lab-border">
            <div className="text-sky-400 font-semibold mb-1">Step 4: Remainder Priority</div>
            <div className="text-slate-300 text-[11px]">
              Rank by r_v = q_v - &lfloor;q_v&rfloor;
            </div>
            <div className="text-[10px] text-lab-textDim mt-2">
              Top-R voxels with capacity (k_v &lt; N_v) receive +1 point. Sum exactly equals K.
            </div>
          </div>
        </div>
      </div>

      {/* Sensor Calibration Inversion Math */}
      <div className="rounded border border-lab-border bg-lab-surface p-5 space-y-4">
        <h3 className="text-sm font-semibold text-white tracking-tight flex items-center space-x-2">
          <Compass className="w-4 h-4 text-sky-400" />
          <span>KITTI Sensor Calibration & 3D Box Transformation Math</span>
        </h3>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-6 text-xs text-lab-textMuted leading-relaxed">
          <div className="space-y-2">
            <h4 className="font-semibold text-white">Camera Coordinate Frame (label_2)</h4>
            <p>
              In the raw KITTI dataset, ground-truth annotations are recorded in rectified Camera 2 coordinates:
            </p>
            <ul className="space-y-1 font-mono text-[11px] text-lab-textMuted list-disc list-inside">
              <li><span className="text-slate-200">X:</span> Lateral axis pointing to the right</li>
              <li><span className="text-slate-200">Y:</span> Vertical axis pointing downwards toward the road</li>
              <li><span className="text-slate-200">Z:</span> Longitudinal axis pointing forward along the optical axis</li>
              <li><span className="text-slate-200">Position:</span> (x, y, z) defines the bottom-center of the 3D box</li>
              <li><span className="text-slate-200">Orientation:</span> rotation_y defines heading yaw around camera Y axis</li>
            </ul>
          </div>

          <div className="space-y-2">
            <h4 className="font-semibold text-white">Velodyne LiDAR Frame Transformation</h4>
            <p>
              Velodyne LiDAR points are expressed in sensor frame: X forward, Y left, Z up.
              To transform camera bottom-center annotations into LiDAR centroids and wireframe corners:
            </p>
            <div className="p-3 bg-[#0a0c10] rounded border border-lab-border font-mono text-[11px] text-slate-300 space-y-1">
              <div className="text-lab-textDim">// 1. Shift camera bottom-center to geometric centroid:</div>
              <div>p_cam = [x, y - height / 2, z]</div>
              <div className="pt-1 text-lab-textDim">// 2. Transform through inverse calibration matrices:</div>
              <div className="text-sky-300">p_lidar = (p_cam - t_velo_to_cam) &bull; (R0_rect &bull; R_velo_to_cam)^(-T)</div>
              <div className="pt-1 text-lab-textDim">// 3. Map camera yaw to LiDAR yaw:</div>
              <div className="text-amber-300">&theta;_lidar = -rotation_y - &pi; / 2</div>
            </div>
            <p className="text-[11px] text-lab-textDim">
              The 8 bounding box corners are computed by rotating local bounding box extents <code className="text-slate-300">[&plusmn;l/2, &plusmn;w/2, &plusmn;h/2]</code> by <code className="text-slate-300">&theta;_lidar</code> and translating by <code className="text-slate-300">p_lidar</code>.
            </p>
          </div>
        </div>
      </div>

      {/* Summary Table: Parameter Specifications */}
      <div className="space-y-3">
        <h3 className="text-sm font-semibold text-white tracking-tight flex items-center space-x-2">
          <Sparkles className="w-4 h-4 text-sky-400" />
          <span>Experimental Parameter Specifications</span>
        </h3>

        <div className="rounded border border-lab-border bg-lab-surface overflow-hidden">
          <table className="w-full text-left text-xs font-mono border-collapse">
            <thead>
              <tr className="border-b border-lab-border bg-[#0e1116] text-lab-textDim text-[11px] font-semibold font-sans">
                <th className="py-2.5 px-4">Parameter</th>
                <th className="py-2.5 px-3">Configured Value</th>
                <th className="py-2.5 px-3">Applies To</th>
                <th className="py-2.5 px-4 font-sans">Scientific Rationale</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-lab-border text-slate-300 text-[11px]">
              <tr className="hover:bg-[#141820] transition-colors">
                <td className="py-2.5 px-4 font-bold text-sky-400">Voxel Grid Size</td>
                <td className="py-2.5 px-3 text-white">0.8 m &times; 0.8 m &times; 0.8 m</td>
                <td className="py-2.5 px-3 text-lab-textMuted">V5, V6, V7</td>
                <td className="py-2.5 px-4 text-lab-textMuted font-sans">Balances local surface resolution with compute efficiency across full LiDAR range.</td>
              </tr>
              <tr className="hover:bg-[#141820] transition-colors">
                <td className="py-2.5 px-4 font-bold text-sky-400">Downsampling Budget</td>
                <td className="py-2.5 px-3 text-white">~20.00% of original points</td>
                <td className="py-2.5 px-3 text-lab-textMuted">All 4 algorithms</td>
                <td className="py-2.5 px-4 text-lab-textMuted font-sans">Simulates high-bandwidth reduction for embedded vehicle edge computing.</td>
              </tr>
              <tr className="hover:bg-[#141820] transition-colors">
                <td className="py-2.5 px-4 font-bold text-emerald-400">V6 Damping Exponent (&alpha;)</td>
                <td className="py-2.5 px-3 text-white">0.25</td>
                <td className="py-2.5 px-3 text-lab-textMuted">V6, V7</td>
                <td className="py-2.5 px-4 text-lab-textMuted font-sans">Sub-linear exponent dampens large ground clusters while preserving small distant clusters.</td>
              </tr>
              <tr className="hover:bg-[#141820] transition-colors">
                <td className="py-2.5 px-4 font-bold text-sky-400">V7 Convex Factor (&lambda;)</td>
                <td className="py-2.5 px-3 text-white">0.25</td>
                <td className="py-2.5 px-3 text-lab-textMuted">V7</td>
                <td className="py-2.5 px-4 text-lab-textMuted font-sans">Blends 25% geometric surface complexity with 75% non-linear density allocation.</td>
              </tr>
              <tr className="hover:bg-[#141820] transition-colors">
                <td className="py-2.5 px-4 font-bold text-amber-400">Target Detector</td>
                <td className="py-2.5 px-3 text-white">PointPillars (Pretrained)</td>
                <td className="py-2.5 px-3 text-lab-textMuted">Downstream benchmark</td>
                <td className="py-2.5 px-4 text-lab-textMuted font-sans">Evaluates performance on real-time pillar-based feature encoding without retraining.</td>
              </tr>
              <tr className="hover:bg-[#141820] transition-colors">
                <td className="py-2.5 px-4 font-bold text-slate-300">Validation Split Size</td>
                <td className="py-2.5 px-3 text-white">3,769 frames</td>
                <td className="py-2.5 px-3 text-lab-textMuted">KITTI ImageSets/val.txt</td>
                <td className="py-2.5 px-4 text-lab-textMuted font-sans">Standard KITTI 3D object detection benchmark validation set.</td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
