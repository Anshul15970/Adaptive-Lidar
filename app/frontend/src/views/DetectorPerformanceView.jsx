import React, { useState, useEffect } from 'react';
import { fetchDetectorMetrics } from '../api';
import { Target, Trophy, TrendingUp, Info } from 'lucide-react';

export default function DetectorPerformanceView() {
  const [detectorMetrics, setDetectorMetrics] = useState(null);
  const [selectedClass, setSelectedClass] = useState('Car');
  const [metricType, setMetricType] = useState('3d_ap_r40'); // '3d_ap_r40' | 'bev_ap_r40'
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    async function loadMetrics() {
      try {
        const res = await fetchDetectorMetrics();
        setDetectorMetrics(res);
      } catch (err) {
        console.error('Failed to load detector metrics:', err);
      } finally {
        setLoading(false);
      }
    }
    loadMetrics();
  }, []);

  // Method order & display info
  const methods = [
    { id: 'full', name: 'Full Cloud (Upper Bound)', retention: '100%', color: '#94a3b8', isBaseline: false },
    { id: 'uniform', name: 'Uniform 20% (Baseline)', retention: '20%', color: '#f43f5e', isBaseline: true },
    { id: 'hybrid_v5', name: 'Hybrid V5', retention: '20%', color: '#f59e0b', isBaseline: false },
    { id: 'v6_alpha_0.25', name: 'V6 (α=0.25)', retention: '20%', color: '#10b981', isBaseline: false },
    { id: 'v7_lambda_0.25', name: 'V7 (λ=0.25)', retention: '20%', color: '#3b82f6', isBaseline: false },
  ];

  const getMetric = (methodId, cls, metricKey, difficulty) => {
    return detectorMetrics?.[methodId]?.classes?.[cls]?.[metricKey]?.[difficulty] ?? null;
  };

  const uniformMod = getMetric('uniform', selectedClass, metricType, 'moderate');

  return (
    <div className="flex-1 overflow-y-auto p-8 space-y-8 bg-lab-bg text-slate-200">
      {/* Editorial Title & Controls Header */}
      <div className="border-b border-lab-border pb-6 flex flex-wrap items-center justify-between gap-4">
        <div className="space-y-1 max-w-2xl">
          <div className="flex items-center space-x-2 text-xs font-mono text-sky-400">
            <Target className="w-4 h-4 text-sky-400" />
            <span className="uppercase">Downstream Evaluation &bull; Fixed Weights</span>
          </div>
          <h1 className="text-xl sm:text-2xl font-semibold tracking-tight text-white">
            Pretrained PointPillars 3D Perception Benchmark
          </h1>
          <p className="text-xs text-lab-textMuted leading-relaxed">
            Evaluated on all 3,769 KITTI validation frames using official 40-recall position interpolation (Moderate AP<sub>R40</sub>).
            Detector network weights are strictly frozen without retraining, isolating the downstream detection fidelity of each downsampling algorithm.
          </p>
        </div>

        {/* Class and Metric Selectors */}
        <div className="flex items-center space-x-2.5">
          {/* Class Selector */}
          <div className="flex items-center bg-lab-surface p-0.5 rounded border border-lab-border text-xs">
            {['Car', 'Pedestrian', 'Cyclist'].map((cls) => (
              <button
                key={cls}
                onClick={() => setSelectedClass(cls)}
                className={`px-3 py-1 rounded transition-colors ${
                  selectedClass === cls
                    ? 'bg-[#1a202c] text-white font-medium border border-[#2b3342]'
                    : 'text-lab-textMuted hover:text-slate-200'
                }`}
              >
                {cls}
              </button>
            ))}
          </div>

          {/* Metric Type Selector */}
          <div className="flex items-center bg-lab-surface p-0.5 rounded border border-lab-border text-xs">
            <button
              onClick={() => setMetricType('3d_ap_r40')}
              className={`px-3 py-1 rounded transition-colors ${
                metricType === '3d_ap_r40'
                  ? 'bg-[#1a202c] text-emerald-400 font-medium border border-[#2b3342]'
                  : 'text-lab-textMuted hover:text-slate-200'
              }`}
            >
              3D Box AP
            </button>
            <button
              onClick={() => setMetricType('bev_ap_r40')}
              className={`px-3 py-1 rounded transition-colors ${
                metricType === 'bev_ap_r40'
                  ? 'bg-[#1a202c] text-emerald-400 font-medium border border-[#2b3342]'
                  : 'text-lab-textMuted hover:text-slate-200'
              }`}
            >
              BEV AP
            </button>
          </div>
        </div>
      </div>

      {/* Difficulty Telemetry Strip */}
      <div className="rounded border border-lab-border bg-lab-surface p-4">
        <div className="text-[11px] font-semibold text-lab-textDim uppercase tracking-wider mb-3 flex items-center justify-between">
          <span>V6 (α=0.25) Performance Across Difficulty Levels ({selectedClass})</span>
          <span className="font-mono text-sky-400">{metricType === '3d_ap_r40' ? '3D Box AP (R40)' : 'BEV AP (R40)'}</span>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-6 divide-y md:divide-y-0 md:divide-x divide-lab-border">
          {['easy', 'moderate', 'hard'].map((difficulty, idx) => {
            const v6Val = getMetric('v6_alpha_0.25', selectedClass, metricType, difficulty);
            const uniVal = getMetric('uniform', selectedClass, metricType, difficulty);
            const delta = v6Val && uniVal ? v6Val - uniVal : 0;

            return (
              <div key={difficulty} className={idx > 0 ? 'pt-4 md:pt-0 md:pl-6 space-y-1' : 'space-y-1'}>
                <div className="flex items-center justify-between text-xs font-mono text-lab-textMuted uppercase">
                  <span>{difficulty}</span>
                  {difficulty === 'moderate' && (
                    <span className="text-[10px] text-sky-400 bg-[#161a22] px-1.5 py-0.2 rounded border border-[#202632]">
                      PRIMARY
                    </span>
                  )}
                </div>

                <div className="flex items-baseline space-x-2 my-0.5">
                  <span className="text-2xl font-semibold font-mono text-white tabular-nums">
                    {v6Val ? `${v6Val.toFixed(2)}%` : '-'}
                  </span>
                  <span className="text-xs text-lab-textDim font-mono">V6</span>
                </div>

                <div className="text-xs font-mono flex items-center space-x-1.5">
                  <span className="text-lab-textDim">vs Uniform:</span>
                  <span
                    className={`font-semibold tabular-nums flex items-center space-x-1 ${
                      delta >= 0 ? 'text-emerald-400' : 'text-rose-400'
                    }`}
                  >
                    <span>{delta >= 0 ? `+${delta.toFixed(2)}%` : `${delta.toFixed(2)}%`}</span>
                  </span>
                </div>
              </div>
            );
          })}
        </div>
      </div>

      {/* Main Benchmark AP Table */}
      <div className="space-y-3">
        <h2 className="text-sm font-semibold text-white tracking-tight flex items-center space-x-2">
          <Trophy className="w-4 h-4 text-sky-400" />
          <span>PointPillars {selectedClass} Detection Benchmark ({metricType === '3d_ap_r40' ? '3D Box AP' : 'Bird\'s Eye View AP'})</span>
        </h2>

        <div className="rounded border border-lab-border bg-lab-surface overflow-hidden">
          <table className="w-full text-xs font-mono text-left border-collapse">
            <thead>
              <tr className="border-b border-lab-border bg-[#0e1116] text-lab-textDim text-[11px] font-semibold">
                <th className="py-2.5 px-4 font-sans">Method</th>
                <th className="py-2.5 px-3">Sampling Quota</th>
                <th className="py-2.5 px-3 text-right">Easy AP</th>
                <th className="py-2.5 px-3 text-right font-semibold text-slate-300">Moderate AP (Primary)</th>
                <th className="py-2.5 px-3 text-right">Hard AP</th>
                <th className="py-2.5 px-4 text-right">Gain vs Uniform</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-lab-border text-slate-300">
              {methods.map((m) => {
                const easy = getMetric(m.id, selectedClass, metricType, 'easy');
                const mod = getMetric(m.id, selectedClass, metricType, 'moderate');
                const hard = getMetric(m.id, selectedClass, metricType, 'hard');
                const gain = mod && uniformMod && m.id !== 'uniform' && m.id !== 'full' ? mod - uniformMod : null;

                return (
                  <tr
                    key={m.id}
                    className={`hover:bg-[#141820] transition-colors ${
                      m.id === 'v6_alpha_0.25' ? 'bg-[#121820]' : ''
                    }`}
                  >
                    <td className="py-3 px-4 font-sans font-medium text-white flex items-center space-x-2">
                      <span className="w-2 h-2 rounded-full" style={{ backgroundColor: m.color }} />
                      <span>{m.name}</span>
                    </td>
                    <td className="py-3 px-3 text-lab-textMuted">{m.retention}</td>
                    <td className="py-3 px-3 text-right text-slate-300 tabular-nums">{easy ? `${easy.toFixed(2)}%` : '-'}</td>
                    <td className="py-3 px-3 text-right font-bold text-white text-sm tabular-nums">{mod ? `${mod.toFixed(2)}%` : '-'}</td>
                    <td className="py-3 px-3 text-right text-slate-300 tabular-nums">{hard ? `${hard.toFixed(2)}%` : '-'}</td>
                    <td className="py-3 px-4 text-right font-mono font-semibold tabular-nums">
                      {gain !== null ? (
                        <span className={gain >= 0 ? 'text-emerald-400' : 'text-rose-400'}>
                          {gain >= 0 ? `+${gain.toFixed(2)}%` : `${gain.toFixed(2)}%`}
                        </span>
                      ) : m.id === 'uniform' ? (
                        <span className="text-lab-textDim">BASELINE</span>
                      ) : (
                        <span className="text-lab-textDim">-</span>
                      )}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>

      {/* Deep-Dive Theoretical Insight Box */}
      <div className="rounded border border-lab-border bg-lab-surface p-5 space-y-3">
        <h3 className="text-xs font-semibold text-lab-textBright uppercase tracking-wider flex items-center space-x-2">
          <Info className="w-3.5 h-3.5 text-sky-400" />
          <span>Research Synthesis & Detector Dynamics</span>
        </h3>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6 text-xs text-slate-300 leading-relaxed pt-1">
          <div className="space-y-1">
            <span className="font-semibold text-rose-400 block">1. Uniform Baseline Vulnerability</span>
            <p className="text-lab-textMuted">
              Uniform random downsampling distributes point loss proportionally across ranges. For sparse objects like cyclists,
              points drop below feature extraction thresholds, collapsing Cyclist 3D AP to <strong className="text-white font-mono">1.18%</strong>.
            </p>
          </div>
          <div className="space-y-1">
            <span className="font-semibold text-amber-400 block">2. Hybrid V5 Over-Allocation</span>
            <p className="text-lab-textMuted">
              V5 geometry complexity sampling favors sparse high-contrast boundary points. While Cyclist AP recovers to 14.54%,
              dense pedestrian clusters are starved by greedy Stage 3 allocation, dropping Pedestrian AP to <strong className="text-white font-mono">7.86%</strong>.
            </p>
          </div>
          <div className="space-y-1">
            <span className="font-semibold text-emerald-400 block">3. Density-Aware V6 Balance</span>
            <p className="text-lab-textMuted">
              V6 sub-linear damping ($W_v = N_v^{0.25}$) suppresses foreground ground clusters while preserving small silhouettes.
              Achieves highest Car AP (<strong className="text-white font-mono">66.24%</strong>), Cyclist AP (<strong className="text-white font-mono">14.62%</strong>), and robust Pedestrians (<strong className="text-white font-mono">19.42%</strong>).
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}
