import React from 'react';
import {
  Layers,
  Box,
  SplitSquareVertical,
  Target,
  ArrowRight,
  BookOpen,
  BarChart3,
  CheckCircle2,
} from 'lucide-react';

export default function OverviewView({ onSelectTab, currentFrameId }) {
  return (
    <div className="flex-1 overflow-y-auto p-8 space-y-8 bg-lab-bg text-slate-200">
      {/* Editorial Research Summary Header */}
      <div className="border-b border-lab-border pb-8">
        <div className="max-w-4xl space-y-4">
          <div className="inline-flex items-center space-x-2 text-xs font-mono text-sky-400">
            <span className="w-1.5 h-1.5 rounded-full bg-sky-400" />
            <span className="tracking-wide uppercase">KITTI Vision Benchmark &bull; PointPillars 3D Perception</span>
          </div>

          <h1 className="text-2xl sm:text-3xl font-semibold tracking-tight text-white leading-tight">
            Adaptive LiDAR Point Cloud Downsampling for 3D Object Detection
          </h1>

          <p className="text-sm text-lab-textMuted leading-relaxed max-w-3xl">
            Autonomous vehicle LiDAR sensors capture hundreds of thousands of spatial measurements per frame at 10–20 Hz.
            While uniform downsampling causes critical distant and low-reflectivity objects (cyclists, pedestrians) to drop below
            downstream detection thresholds, this research evaluates adaptive geometry and density-aware downsampling formulations
            operating under a strict 20% point retention budget ($0.8\,\text{m}$ voxel discretization).
          </p>

          <div className="pt-2 flex flex-wrap gap-2.5">
            <button
              onClick={() => onSelectTab('explorer')}
              className="flex items-center space-x-2 px-4 py-2 rounded bg-sky-600 hover:bg-sky-500 text-white font-medium text-xs transition-colors shadow-subtle"
            >
              <Box className="w-3.5 h-3.5" />
              <span>3D Point Cloud Explorer</span>
              <ArrowRight className="w-3.5 h-3.5" />
            </button>

            <button
              onClick={() => onSelectTab('comparison')}
              className="flex items-center space-x-2 px-4 py-2 rounded bg-lab-surface hover:bg-[#1a202c] border border-lab-border text-slate-200 font-medium text-xs transition-colors"
            >
              <SplitSquareVertical className="w-3.5 h-3.5 text-lab-textMuted" />
              <span>Synchronized Multi-View</span>
            </button>

            <button
              onClick={() => onSelectTab('detector')}
              className="flex items-center space-x-2 px-4 py-2 rounded bg-lab-surface hover:bg-[#1a202c] border border-lab-border text-slate-200 font-medium text-xs transition-colors"
            >
              <Target className="w-3.5 h-3.5 text-lab-textMuted" />
              <span>Detector Benchmark (AP)</span>
            </button>

            <button
              onClick={() => onSelectTab('methodology')}
              className="flex items-center space-x-2 px-4 py-2 rounded bg-lab-surface hover:bg-[#1a202c] border border-lab-border text-slate-200 font-medium text-xs transition-colors"
            >
              <BookOpen className="w-3.5 h-3.5 text-lab-textMuted" />
              <span>Research Methodology</span>
            </button>
          </div>
        </div>
      </div>

      {/* Unified Telemetry Strip - Replaces Card Soup */}
      <div className="rounded border border-lab-border bg-lab-surface p-4">
        <div className="text-[11px] font-semibold text-lab-textDim uppercase tracking-wider mb-3">
          Experimental Dataset & Evaluation Benchmark
        </div>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-6 divide-y md:divide-y-0 md:divide-x divide-lab-border">
          <div className="space-y-0.5">
            <span className="text-[11px] text-lab-textMuted uppercase block">Validation Frames</span>
            <span className="text-xl font-semibold font-mono text-white block tabular-nums">3,769</span>
            <span className="text-[11px] text-lab-textDim block">KITTI val.txt split</span>
          </div>

          <div className="pt-4 md:pt-0 md:pl-6 space-y-0.5">
            <span className="text-[11px] text-lab-textMuted uppercase block">Total Raw Points</span>
            <span className="text-xl font-semibold font-mono text-white block tabular-nums">447,411,677</span>
            <span className="text-[11px] text-lab-textDim block">Sampled: 89,480,814</span>
          </div>

          <div className="pt-4 md:pt-0 md:pl-6 space-y-0.5">
            <span className="text-[11px] text-lab-textMuted uppercase block">Retention Budget</span>
            <span className="text-xl font-semibold font-mono text-emerald-400 block tabular-nums">19.9997%</span>
            <span className="text-[11px] text-lab-textDim block">Budget target: 20.00%</span>
          </div>

          <div className="pt-4 md:pt-0 md:pl-6 space-y-0.5">
            <span className="text-[11px] text-lab-textMuted uppercase block">Car Moderate AP</span>
            <span className="text-xl font-semibold font-mono text-sky-400 block tabular-nums">+15.10%</span>
            <span className="text-[11px] text-lab-textDim block">V6 α=0.25 vs Uniform</span>
          </div>
        </div>
      </div>

      {/* Evaluated Sampling Algorithms - Technical Comparison Matrix */}
      <div className="space-y-3">
        <div className="flex items-center justify-between">
          <h2 className="text-sm font-semibold text-white tracking-tight flex items-center space-x-2">
            <Layers className="w-4 h-4 text-sky-400" />
            <span>Evaluated Sampling Algorithms (0.8m Voxel Grid)</span>
          </h2>
          <span className="text-xs text-lab-textDim font-mono">
            Unsupervised &bull; No Ground-Truth Leakage
          </span>
        </div>

        <div className="rounded border border-lab-border bg-lab-surface overflow-hidden">
          <table className="w-full text-left text-xs border-collapse">
            <thead>
              <tr className="border-b border-lab-border bg-[#0e1116] text-lab-textDim text-[11px] font-semibold">
                <th className="py-2.5 px-4">Algorithm</th>
                <th className="py-2.5 px-4">Core Mechanism</th>
                <th className="py-2.5 px-4">Quota Solver</th>
                <th className="py-2.5 px-3 font-mono text-right">Car AP</th>
                <th className="py-2.5 px-3 font-mono text-right">Pedestrian AP</th>
                <th className="py-2.5 px-3 font-mono text-right">Cyclist AP</th>
                <th className="py-2.5 px-4 font-mono text-right">Moderate mAP</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-lab-border text-slate-300">
              <tr className="hover:bg-[#141820] transition-colors">
                <td className="py-3 px-4 font-medium text-white">
                  <div className="flex items-center space-x-2">
                    <span className="w-2 h-2 rounded-full bg-slate-400" />
                    <span>Full Point Cloud</span>
                  </div>
                </td>
                <td className="py-3 px-4 text-lab-textMuted">Unsampled raw sensor data (100% points)</td>
                <td className="py-3 px-4 text-lab-textDim font-mono">None (Upper bound)</td>
                <td className="py-3 px-3 font-mono text-right tabular-nums text-slate-300">72.91%</td>
                <td className="py-3 px-3 font-mono text-right tabular-nums text-slate-300">43.80%</td>
                <td className="py-3 px-3 font-mono text-right tabular-nums text-slate-300">33.39%</td>
                <td className="py-3 px-4 font-mono text-right tabular-nums font-semibold text-white">50.03%</td>
              </tr>

              <tr className="hover:bg-[#141820] transition-colors">
                <td className="py-3 px-4 font-medium text-rose-300">
                  <div className="flex items-center space-x-2">
                    <span className="w-2 h-2 rounded-full bg-rose-400" />
                    <span>Uniform 20%</span>
                  </div>
                </td>
                <td className="py-3 px-4 text-lab-textMuted">Random independent sampling ($p=0.20$)</td>
                <td className="py-3 px-4 text-lab-textDim font-mono">Bernoulli selection</td>
                <td className="py-3 px-3 font-mono text-right tabular-nums text-slate-300">51.14%</td>
                <td className="py-3 px-3 font-mono text-right tabular-nums text-slate-300">23.31%</td>
                <td className="py-3 px-3 font-mono text-right tabular-nums text-rose-400 font-semibold">1.18%</td>
                <td className="py-3 px-4 font-mono text-right tabular-nums text-lab-textMuted">25.21%</td>
              </tr>

              <tr className="hover:bg-[#141820] transition-colors">
                <td className="py-3 px-4 font-medium text-amber-300">
                  <div className="flex items-center space-x-2">
                    <span className="w-2 h-2 rounded-full bg-amber-400" />
                    <span>Hybrid V5</span>
                  </div>
                </td>
                <td className="py-3 px-4 text-lab-textMuted">Geometric complexity & local voxel sparsity</td>
                <td className="py-3 px-4 text-lab-textDim font-mono">3-stage greedy quota</td>
                <td className="py-3 px-3 font-mono text-right tabular-nums text-slate-300">61.38%</td>
                <td className="py-3 px-3 font-mono text-right tabular-nums text-rose-400">7.86%</td>
                <td className="py-3 px-3 font-mono text-right tabular-nums text-amber-300">14.54%</td>
                <td className="py-3 px-4 font-mono text-right tabular-nums text-lab-textMuted">27.93%</td>
              </tr>

              <tr className="bg-[#121820] hover:bg-[#161d27] transition-colors">
                <td className="py-3 px-4 font-semibold text-emerald-300">
                  <div className="flex items-center space-x-2">
                    <span className="w-2 h-2 rounded-full bg-emerald-400" />
                    <span>V6 α=0.25</span>
                  </div>
                </td>
                <td className="py-3 px-4 text-slate-200">Density-aware sub-linear power damping $W_v = N_v^{0.25}$</td>
                <td className="py-3 px-4 text-emerald-400 font-mono">Hamilton-Webster integer</td>
                <td className="py-3 px-3 font-mono text-right tabular-nums font-semibold text-emerald-400">66.24%</td>
                <td className="py-3 px-3 font-mono text-right tabular-nums text-slate-200">19.42%</td>
                <td className="py-3 px-3 font-mono text-right tabular-nums font-semibold text-emerald-400">14.62%</td>
                <td className="py-3 px-4 font-mono text-right tabular-nums font-bold text-emerald-400">33.43%</td>
              </tr>

              <tr className="hover:bg-[#141820] transition-colors">
                <td className="py-3 px-4 font-medium text-sky-300">
                  <div className="flex items-center space-x-2">
                    <span className="w-2 h-2 rounded-full bg-sky-400" />
                    <span>V7 λ=0.25</span>
                  </div>
                </td>
                <td className="py-3 px-4 text-lab-textMuted">Convex combination of boundary score and density</td>
                <td className="py-3 px-4 text-sky-400 font-mono">Hamilton-Webster integer</td>
                <td className="py-3 px-3 font-mono text-right tabular-nums text-slate-300">65.33%</td>
                <td className="py-3 px-3 font-mono text-right tabular-nums text-slate-200">16.13%</td>
                <td className="py-3 px-3 font-mono text-right tabular-nums text-slate-200">12.18%</td>
                <td className="py-3 px-4 font-mono text-right tabular-nums text-sky-400">31.21%</td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
