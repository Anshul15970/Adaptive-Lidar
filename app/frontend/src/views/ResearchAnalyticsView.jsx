import React, { useState, useEffect, useMemo } from 'react';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  BarElement,
  LineElement,
  PointElement,
  RadialLinearScale,
  Title,
  Tooltip,
  Legend,
  Filler,
} from 'chart.js';
import { Bar, Radar } from 'react-chartjs-2';
import {
  fetchGlobalMetrics,
  fetchDistanceMetrics,
  fetchDistributionMetrics,
} from '../api';
import {
  BarChart3,
  Layers,
  TrendingDown,
  AlertTriangle,
  Target,
  Trophy,
  Sliders,
  Download,
  Info,
  CheckCircle2,
  FileSpreadsheet,
  Car,
  User,
  Bike,
  Activity,
  Boxes,
  HelpCircle,
} from 'lucide-react';

// Register Chart.js elements
ChartJS.register(
  CategoryScale,
  LinearScale,
  BarElement,
  LineElement,
  PointElement,
  RadialLinearScale,
  Title,
  Tooltip,
  Legend,
  Filler
);

// Method colors & labels
const METHOD_COLORS = {
  full: '#94a3b8',        // Slate Silver
  uniform: '#f43f5e',     // Rose Red
  v5: '#f59e0b',          // Amber
  hybrid_v5: '#f59e0b',   // Amber
  'v6_alpha_0.25': '#10b981',// Emerald
  'v7_lambda_0.25': '#38bdf8',// Sky Blue
};

const METHOD_NAMES = {
  full: 'Full Cloud',
  uniform: 'Uniform 20%',
  v5: 'Hybrid V5',
  hybrid_v5: 'Hybrid V5',
  'v6_alpha_0.25': 'V6 α=0.25',
  'v7_lambda_0.25': 'V7 λ=0.25',
};

// Exact Measured PointPillars Moderate AP_R40 results from research evaluation
const DETECTOR_MODERATE_AP = {
  full: { Car: 72.9095, Pedestrian: 43.8000, Cyclist: 33.3889 },
  uniform: { Car: 51.1415, Pedestrian: 23.3063, Cyclist: 1.1832 },
  hybrid_v5: { Car: 61.3816, Pedestrian: 7.8629, Cyclist: 14.5416 },
  'v6_alpha_0.25': { Car: 66.2424, Pedestrian: 19.4154, Cyclist: 14.6179 },
  'v7_lambda_0.25': { Car: 65.3325, Pedestrian: 16.1286, Cyclist: 12.1795 },
};

// Common Chart.js styling
const chartBaseOptions = {
  responsive: true,
  maintainAspectRatio: false,
  plugins: {
    legend: {
      position: 'top',
      labels: {
        color: '#9ca3af',
        font: { family: 'Inter', size: 11, weight: '500' },
        boxWidth: 10,
        padding: 12,
      },
    },
    tooltip: {
      backgroundColor: '#101318',
      borderColor: '#242a36',
      borderWidth: 1,
      titleColor: '#f3f4f6',
      titleFont: { family: 'Inter', size: 12, weight: '600' },
      bodyColor: '#cbd5e1',
      bodyFont: { family: 'JetBrains Mono', size: 11 },
      padding: 10,
      cornerRadius: 4,
    },
  },
  scales: {
    x: {
      grid: { color: 'rgba(36, 42, 54, 0.4)' },
      ticks: { color: '#6b7280', font: { family: 'JetBrains Mono', size: 10 } },
    },
    y: {
      grid: { color: 'rgba(36, 42, 54, 0.4)' },
      ticks: { color: '#6b7280', font: { family: 'JetBrains Mono', size: 10 } },
    },
  },
};

export default function ResearchAnalyticsView() {
  const [activeSection, setActiveSection] = useState('overview'); // 'overview' | 'global' | 'distance' | 'distribution' | 'detector' | 'comparator'
  const [globalMetrics, setGlobalMetrics] = useState([]);
  const [distanceMetrics, setDistanceMetrics] = useState([]);
  const [distributionMetrics, setDistributionMetrics] = useState([]);
  const [loading, setLoading] = useState(true);

  // Filter states
  const [distanceClass, setDistanceClass] = useState('Car');
  const [distanceMetricType, setDistanceMetricType] = useState('mean'); // 'mean' | 'median' | 'weighted'
  const [distributionClass, setDistributionClass] = useState('Car');

  // Comparator selected methods
  const [comparedMethods, setComparedMethods] = useState(['uniform', 'hybrid_v5', 'v6_alpha_0.25', 'v7_lambda_0.25']);

  useEffect(() => {
    async function loadData() {
      setLoading(true);
      try {
        const [globalRes, distRes, distribRes] = await Promise.all([
          fetchGlobalMetrics(),
          fetchDistanceMetrics(),
          fetchDistributionMetrics(),
        ]);
        setGlobalMetrics(globalRes);
        setDistanceMetrics(distRes);
        setDistributionMetrics(distribRes);
      } catch (err) {
        console.error('Failed to load experimental research analytics data:', err);
      } finally {
        setLoading(false);
      }
    }
    loadData();
  }, []);

  // -------------------------------------------------------------
  // Chart 1: Global Retention Analysis
  // -------------------------------------------------------------
  const globalChartData = useMemo(() => {
    if (!globalMetrics.length) return null;
    const labels = globalMetrics.map((g) => g.method_name);
    return {
      labels,
      datasets: [
        {
          label: 'Weighted Retention %',
          data: globalMetrics.map((g) => g.weighted_retention_pct),
          backgroundColor: '#38bdf8cc',
          borderColor: '#38bdf8',
          borderWidth: 1,
        },
        {
          label: 'Mean Frame Retention %',
          data: globalMetrics.map((g) => g.mean_frame_retention_pct),
          backgroundColor: '#10b981cc',
          borderColor: '#10b981',
          borderWidth: 1,
        },
        {
          label: 'Median Frame Retention %',
          data: globalMetrics.map((g) => g.median_frame_retention_pct),
          backgroundColor: '#f59e0bcc',
          borderColor: '#f59e0b',
          borderWidth: 1,
        },
      ],
    };
  }, [globalMetrics]);

  // -------------------------------------------------------------
  // Chart 2: Distance-Wise Retention (Grouped by Near, Med, Far)
  // -------------------------------------------------------------
  const distanceChartData = useMemo(() => {
    if (!distanceMetrics.length) return null;
    const filtered = distanceMetrics.filter((d) => d.class === distanceClass);
    const groups = ['near_0_20m', 'medium_20_40m', 'far_over_40m'];
    const groupLabels = ['Near (0–20m)', 'Medium (20–40m)', 'Far (>40m)'];
    const methodsList = ['uniform', 'v5', 'v6_alpha_0.25', 'v7_lambda_0.25'];

    const datasets = methodsList.map((m) => {
      const color = METHOD_COLORS[m] || '#94a3b8';
      const data = groups.map((grp) => {
        const item = filtered.find((d) => d.method_id === m && d.distance_group === grp);
        if (!item) return 0;
        if (distanceMetricType === 'mean') return item.mean_retention_pct;
        if (distanceMetricType === 'median') return item.median_retention_pct;
        return item.weighted_retention_pct;
      });

      return {
        label: METHOD_NAMES[m] || m,
        data,
        backgroundColor: color + 'cc',
        borderColor: color,
        borderWidth: 1,
      };
    });

    return {
      labels: groupLabels,
      datasets,
    };
  }, [distanceMetrics, distanceClass, distanceMetricType]);

  // -------------------------------------------------------------
  // Chart 3: Object Retention Distribution (<10%, <20%, <40%, <60%)
  // -------------------------------------------------------------
  const distributionChartData = useMemo(() => {
    if (!distributionMetrics.length) return null;
    const filtered = distributionMetrics.filter((d) => d.class === distributionClass);
    const thresholds = ['< 10% Points', '< 20% Points', '< 40% Points', '< 60% Points'];
    const methodsList = ['uniform', 'v5', 'v6_alpha_0.25', 'v7_lambda_0.25'];

    const datasets = methodsList.map((m) => {
      const color = METHOD_COLORS[m] || '#94a3b8';
      const item = filtered.find((d) => d.method_id === m);
      const data = item
        ? [item.pct_below_10pct, item.pct_below_20pct, item.pct_below_40pct, item.pct_below_60pct]
        : [0, 0, 0, 0];

      return {
        label: METHOD_NAMES[m] || m,
        data,
        backgroundColor: color + 'cc',
        borderColor: color,
        borderWidth: 1,
      };
    });

    return {
      labels: thresholds,
      datasets,
    };
  }, [distributionMetrics, distributionClass]);

  // -------------------------------------------------------------
  // Chart 4: Detector Moderate AP_R40 Comparison
  // -------------------------------------------------------------
  const detectorChartData = useMemo(() => {
    const classes = ['Car', 'Pedestrian', 'Cyclist'];
    const methodsList = ['full', 'uniform', 'hybrid_v5', 'v6_alpha_0.25', 'v7_lambda_0.25'];

    const datasets = methodsList.map((m) => {
      const color = METHOD_COLORS[m] || '#94a3b8';
      const data = classes.map((cls) => DETECTOR_MODERATE_AP[m][cls]);

      return {
        label: METHOD_NAMES[m] || m,
        data,
        backgroundColor: color + 'cc',
        borderColor: color,
        borderWidth: 1,
      };
    });

    return {
      labels: classes,
      datasets,
    };
  }, []);

  // Mean AP (mAP) & Relative Retention against Full Cloud
  const mapAndRelativeData = useMemo(() => {
    const methodsList = ['full', 'uniform', 'hybrid_v5', 'v6_alpha_0.25', 'v7_lambda_0.25'];
    const fullMeanAP = (72.9095 + 43.8000 + 33.3889) / 3; // 50.0328%

    const mAPValues = methodsList.map((m) => {
      const scores = DETECTOR_MODERATE_AP[m];
      return (scores.Car + scores.Pedestrian + scores.Cyclist) / 3;
    });

    const relAPValues = mAPValues.map((val) => (val / fullMeanAP) * 100);

    return {
      labels: methodsList.map((m) => METHOD_NAMES[m] || m),
      mAPValues,
      relAPValues,
    };
  }, []);

  const toggleComparatorMethod = (methodId) => {
    setComparedMethods((prev) => {
      if (prev.includes(methodId)) {
        if (prev.length <= 2) return prev; // Keep at least 2 methods
        return prev.filter((m) => m !== methodId);
      } else {
        return [...prev, methodId];
      }
    });
  };

  const handleExportCSV = () => {
    let csvContent = 'data:text/csv;charset=utf-8,';
    csvContent += 'Method,Car_Moderate_AP_R40,Pedestrian_Moderate_AP_R40,Cyclist_Moderate_AP_R40,Mean_AP_R40,Relative_AP_Retention_Pct\n';
    const fullMeanAP = (72.9095 + 43.8000 + 33.3889) / 3;

    ['full', 'uniform', 'hybrid_v5', 'v6_alpha_0.25', 'v7_lambda_0.25'].forEach((m) => {
      const scores = DETECTOR_MODERATE_AP[m];
      const mAP = (scores.Car + scores.Pedestrian + scores.Cyclist) / 3;
      const rel = (mAP / fullMeanAP) * 100;
      csvContent += `${METHOD_NAMES[m]},${scores.Car},${scores.Pedestrian},${scores.Cyclist},${mAP.toFixed(4)},${rel.toFixed(2)}%\n`;
    });

    const encodedUri = encodeURI(csvContent);
    const link = document.createElement('a');
    link.setAttribute('href', encodedUri);
    link.setAttribute('download', 'adaptive_lidar_detector_moderate_ap_r40.csv');
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  return (
    <div className="flex-1 flex flex-col h-[calc(100vh-3rem)] overflow-hidden bg-lab-bg text-slate-100">
      {/* Sub-Header Tabs */}
      <div className="h-11 border-b border-lab-border bg-lab-surface px-5 flex items-center justify-between shrink-0 select-none">
        <div className="flex items-center space-x-1">
          {[
            { id: 'overview', label: 'Overview' },
            { id: 'global', label: 'Global Retention' },
            { id: 'distance', label: 'Distance Retention' },
            { id: 'distribution', label: 'Starvation Distribution' },
            { id: 'detector', label: 'Detector AP' },
            { id: 'comparator', label: 'Algorithm Comparator' },
          ].map((tab) => {
            const isActive = activeSection === tab.id;
            return (
              <button
                key={tab.id}
                onClick={() => setActiveSection(tab.id)}
                className={`px-3 py-1 rounded text-xs font-medium transition-colors ${
                  isActive
                    ? 'bg-[#1a202c] text-white font-medium border border-[#2b3342]'
                    : 'text-lab-textMuted hover:text-slate-200'
                }`}
              >
                {tab.label}
              </button>
            );
          })}
        </div>

        <button
          onClick={handleExportCSV}
          className="flex items-center space-x-1.5 px-2.5 py-1 rounded bg-[#141820] border border-lab-border text-xs text-lab-textMuted hover:text-white transition-colors"
        >
          <Download className="w-3.5 h-3.5" />
          <span>Export CSV</span>
        </button>
      </div>

      {/* Main Content Area */}
      <div className="flex-1 overflow-y-auto p-6 space-y-6">
        {/* ========================================================= */}
        {/* SECTION 1: OVERVIEW DASHBOARD                             */}
        {/* ========================================================= */}
        {activeSection === 'overview' && (
          <div className="space-y-6 max-w-6xl mx-auto">
            {/* Headline Statistics Telemetry Strip */}
            <div className="rounded border border-lab-border bg-lab-surface p-4">
              <div className="text-[11px] font-semibold text-lab-textDim uppercase tracking-wider mb-3">
                Research Analytics Overview
              </div>
              <div className="grid grid-cols-2 md:grid-cols-5 gap-6 divide-y md:divide-y-0 md:divide-x divide-lab-border">
                <div className="space-y-0.5">
                  <span className="text-[11px] text-lab-textMuted uppercase block">Validation Frames</span>
                  <span className="text-xl font-semibold font-mono text-white block tabular-nums">3,769</span>
                  <span className="text-[11px] text-lab-textDim block">KITTI val split</span>
                </div>

                <div className="pt-4 md:pt-0 md:pl-6 space-y-0.5">
                  <span className="text-[11px] text-lab-textMuted uppercase block">Original Points</span>
                  <span className="text-xl font-semibold font-mono text-white block tabular-nums">447,411,677</span>
                  <span className="text-[11px] text-lab-textDim block">Total LiDAR points</span>
                </div>

                <div className="pt-4 md:pt-0 md:pl-6 space-y-0.5">
                  <span className="text-[11px] text-lab-textMuted uppercase block">Sampled Points</span>
                  <span className="text-xl font-semibold font-mono text-emerald-400 block tabular-nums">89,480,814</span>
                  <span className="text-[11px] text-lab-textDim block">Retained points</span>
                </div>

                <div className="pt-4 md:pt-0 md:pl-6 space-y-0.5">
                  <span className="text-[11px] text-lab-textMuted uppercase block">Sampling Budget</span>
                  <span className="text-xl font-semibold font-mono text-sky-400 block tabular-nums">19.9997%</span>
                  <span className="text-[11px] text-lab-textDim block">Target: 20.00%</span>
                </div>

                <div className="pt-4 md:pt-0 md:pl-6 space-y-0.5">
                  <span className="text-[11px] text-lab-textMuted uppercase block">Algorithms</span>
                  <span className="text-xl font-semibold font-mono text-slate-300 block tabular-nums">4</span>
                  <span className="text-[11px] text-lab-textDim block">Excl. Full cloud</span>
                </div>
              </div>
            </div>

            {/* Experimental Setup Summary */}
            <div className="rounded border border-lab-border bg-lab-surface p-5 space-y-4">
              <div className="flex items-center space-x-2 border-b border-lab-border pb-3">
                <FileSpreadsheet className="w-4 h-4 text-sky-400" />
                <h3 className="text-sm font-semibold text-white">
                  Experimental Setup & Benchmark Protocols
                </h3>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-6 text-xs text-lab-textMuted leading-relaxed">
                <div className="space-y-3">
                  <div className="p-3.5 rounded bg-[#0a0c10] border border-lab-border">
                    <strong className="text-slate-200 block mb-1">1. Fixed Detector Checkpoint</strong>
                    Evaluated using a fixed, pretrained PointPillars architecture on the official KITTI 3D validation split.
                    <span className="text-slate-300 block mt-1">No detector retraining is performed.</span>
                    Performance variations directly reflect the perceptual quality of the downsampled points.
                  </div>

                  <div className="p-3.5 rounded bg-[#0a0c10] border border-lab-border">
                    <strong className="text-slate-200 block mb-1">2. Voxel Size & Budget Parity</strong>
                    All four downsampling algorithms operate with a standardized <span className="text-slate-200 font-mono">0.8m voxel size</span> and
                    enforce a strict <span className="text-slate-200 font-mono">20.0% retention target</span> (89,480,814 points out of 447,411,677).
                  </div>
                </div>

                <div className="space-y-3">
                  <div className="p-3.5 rounded bg-[#0a0c10] border border-lab-border">
                    <strong className="text-slate-200 block mb-1">3. Evaluated Downsamplers</strong>
                    <ul className="space-y-1 text-xs text-lab-textMuted list-disc list-inside">
                      <li><strong className="text-rose-400">Uniform 20%:</strong> Spatial voxel downsampling baseline.</li>
                      <li><strong className="text-amber-400">Hybrid V5:</strong> Geometry surface dispersion & voxel sparsity.</li>
                      <li><strong className="text-emerald-400">V6 α=0.25:</strong> Sub-linear density damping allocation.</li>
                      <li><strong className="text-sky-400">V7 λ=0.25:</strong> Combined score-density convex formulation.</li>
                    </ul>
                  </div>

                  <div className="p-3.5 rounded bg-[#0a0c10] border border-lab-border">
                    <strong className="text-slate-200 block mb-1">4. Ground-Truth Independence</strong>
                    Sampling decisions rely exclusively on unsupervised raw LiDAR coordinates and reflectance; ground-truth annotations are used solely for post-hoc bounding box containment and detector mAP scoring.
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* ========================================================= */}
        {/* SECTION 2: GLOBAL RETENTION ANALYSIS                      */}
        {/* ========================================================= */}
        {activeSection === 'global' && (
          <div className="space-y-6 max-w-6xl mx-auto">
            <div className="rounded border border-lab-border bg-lab-surface p-5">
              <div className="flex items-center justify-between mb-4">
                <div>
                  <h3 className="text-sm font-semibold text-white flex items-center space-x-2">
                    <BarChart3 className="w-4 h-4 text-sky-400" />
                    <span>Global Point Retention Parity Across All 3,769 Validation Frames</span>
                  </h3>
                  <p className="text-xs text-lab-textMuted mt-0.5">
                    Comparison of weighted, mean, median, minimum, and maximum retention percentages.
                  </p>
                </div>
                <span className="text-xs font-mono px-2 py-0.5 rounded bg-[#141820] border border-lab-border text-emerald-400">
                  TARGET: 20.00%
                </span>
              </div>

              {/* Chart */}
              <div className="h-72 w-full">
                {globalChartData && (
                  <Bar
                    data={globalChartData}
                    options={{
                      ...chartBaseOptions,
                      scales: {
                        ...chartBaseOptions.scales,
                        y: {
                          ...chartBaseOptions.scales.y,
                          min: 19.98,
                          max: 20.01,
                          title: { display: true, text: 'Retention (%)', color: '#9ca3af' },
                        },
                      },
                    }}
                  />
                )}
              </div>
            </div>

            {/* Exact Global Retention Metrics Table */}
            <div className="rounded border border-lab-border bg-lab-surface p-5 space-y-3">
              <h4 className="text-xs font-semibold text-lab-textDim uppercase tracking-wider">
                Verified Global Retention Statistics (From results/global_retention_summary_v5_v6_v7_comparison.csv)
              </h4>
              <div className="overflow-x-auto">
                <table className="w-full text-xs font-mono text-left border-collapse">
                  <thead>
                    <tr className="border-b border-lab-border bg-[#0e1116] text-lab-textDim text-[11px] font-semibold">
                      <th className="py-2.5 px-3 font-sans">Method</th>
                      <th className="py-2.5 px-3 text-right">Weighted Ret %</th>
                      <th className="py-2.5 px-3 text-right">Mean Frame %</th>
                      <th className="py-2.5 px-3 text-right">Median Frame %</th>
                      <th className="py-2.5 px-3 text-right">Min Frame %</th>
                      <th className="py-2.5 px-3 text-right">Max Frame %</th>
                      <th className="py-2.5 px-3 text-right">Exactly 20% Count</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-lab-border text-slate-300">
                    {globalMetrics.map((row) => (
                      <tr key={row.method_id} className="hover:bg-[#141820] transition-colors">
                        <td className="py-2.5 px-3 font-bold text-white flex items-center space-x-2 font-sans">
                          <span
                            className="w-2 h-2 rounded-full"
                            style={{ backgroundColor: METHOD_COLORS[row.method_id] }}
                          />
                          <span>{row.method_name}</span>
                        </td>
                        <td className="py-2.5 px-3 text-right font-bold text-emerald-400 tabular-nums">
                          {row.weighted_retention_pct.toFixed(5)}%
                        </td>
                        <td className="py-2.5 px-3 text-right text-slate-300 tabular-nums">{row.mean_frame_retention_pct.toFixed(5)}%</td>
                        <td className="py-2.5 px-3 text-right text-slate-300 tabular-nums">{row.median_frame_retention_pct.toFixed(5)}%</td>
                        <td className="py-2.5 px-3 text-right text-lab-textDim tabular-nums">{row.min_frame_retention_pct.toFixed(5)}%</td>
                        <td className="py-2.5 px-3 text-right text-lab-textDim tabular-nums">{row.max_frame_retention_pct.toFixed(5)}%</td>
                        <td className="py-2.5 px-3 text-right text-sky-400 tabular-nums">{row.frames_exactly_20pct.toLocaleString()}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        )}

        {/* ========================================================= */}
        {/* SECTION 3: DISTANCE-WISE RETENTION                        */}
        {/* ========================================================= */}
        {activeSection === 'distance' && (
          <div className="space-y-6 max-w-6xl mx-auto">
            <div className="rounded border border-lab-border bg-lab-surface p-5">
              <div className="flex flex-wrap items-center justify-between gap-4 mb-4">
                <div>
                  <h3 className="text-sm font-semibold text-white flex items-center space-x-2">
                    <TrendingDown className="w-4 h-4 text-sky-400" />
                    <span>Distance-Wise Object Point Retention</span>
                  </h3>
                  <p className="text-xs text-lab-textMuted mt-0.5">
                    Near (0–20 m), Medium (20–40 m), and Far (&gt;40 m) point retention profiles.
                  </p>
                </div>

                {/* Class & Metric Selectors */}
                <div className="flex flex-wrap items-center gap-2">
                  <div className="flex items-center space-x-0.5 bg-[#0a0c10] p-0.5 rounded border border-lab-border text-xs">
                    {['Car', 'Pedestrian', 'Cyclist'].map((cls) => (
                      <button
                        key={cls}
                        onClick={() => setDistanceClass(cls)}
                        className={`px-3 py-1 rounded transition-colors ${
                          distanceClass === cls
                            ? 'bg-[#1a202c] text-white font-medium border border-[#2b3342]'
                            : 'text-lab-textMuted hover:text-slate-200'
                        }`}
                      >
                        {cls}
                      </button>
                    ))}
                  </div>

                  <div className="flex items-center space-x-0.5 bg-[#0a0c10] p-0.5 rounded border border-lab-border text-xs">
                    {[
                      { id: 'mean', label: 'Mean %' },
                      { id: 'median', label: 'Median %' },
                      { id: 'weighted', label: 'Weighted %' },
                    ].map((m) => (
                      <button
                        key={m.id}
                        onClick={() => setDistanceMetricType(m.id)}
                        className={`px-2.5 py-1 rounded transition-colors ${
                          distanceMetricType === m.id
                            ? 'bg-[#1a202c] text-emerald-400 font-medium border border-[#2b3342]'
                            : 'text-lab-textMuted hover:text-slate-200'
                        }`}
                      >
                        {m.label}
                      </button>
                    ))}
                  </div>
                </div>
              </div>

              {/* Chart */}
              <div className="h-72 w-full">
                {distanceChartData && (
                  <Bar
                    data={distanceChartData}
                    options={{
                      ...chartBaseOptions,
                      scales: {
                        ...chartBaseOptions.scales,
                        y: {
                          ...chartBaseOptions.scales.y,
                          title: {
                            display: true,
                            text: `${distanceMetricType.toUpperCase()} RETENTION (%)`,
                            color: '#9ca3af',
                          },
                        },
                      },
                    }}
                  />
                )}
              </div>
            </div>

            {/* Exact Distance Retention Values Table */}
            <div className="rounded border border-lab-border bg-lab-surface p-5 space-y-3">
              <h4 className="text-xs font-semibold text-lab-textDim uppercase tracking-wider">
                Verified Distance Retention Breakdown for {distanceClass}
              </h4>
              <div className="overflow-x-auto">
                <table className="w-full text-xs font-mono text-left border-collapse">
                  <thead>
                    <tr className="border-b border-lab-border bg-[#0e1116] text-lab-textDim text-[11px] font-semibold">
                      <th className="py-2.5 px-3 font-sans">Range Band</th>
                      <th className="py-2.5 px-3 font-sans">Method</th>
                      <th className="py-2.5 px-3 text-right">Object Count</th>
                      <th className="py-2.5 px-3 text-right">Original Pts</th>
                      <th className="py-2.5 px-3 text-right">Retained Pts</th>
                      <th className="py-2.5 px-3 text-right font-sans font-semibold">Mean Ret %</th>
                      <th className="py-2.5 px-3 text-right">Median Ret %</th>
                      <th className="py-2.5 px-3 text-right">Weighted Ret %</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-lab-border text-slate-300">
                    {distanceMetrics
                      .filter((d) => d.class === distanceClass)
                      .map((row, idx) => (
                        <tr key={idx} className="hover:bg-[#141820] transition-colors">
                          <td className="py-2.5 px-3 text-white font-medium font-sans">
                            {row.distance_group === 'near_0_20m'
                              ? 'Near (0–20 m)'
                              : row.distance_group === 'medium_20_40m'
                              ? 'Medium (20–40 m)'
                              : 'Far (>40 m)'}
                          </td>
                          <td className="py-2.5 px-3 font-semibold text-slate-200 font-sans">{row.method_name}</td>
                          <td className="py-2.5 px-3 text-right text-lab-textDim tabular-nums">{row.object_count.toLocaleString()}</td>
                          <td className="py-2.5 px-3 text-right text-lab-textDim tabular-nums">{row.original_box_points.toLocaleString()}</td>
                          <td className="py-2.5 px-3 text-right text-slate-300 tabular-nums">{row.retained_box_points.toLocaleString()}</td>
                          <td className="py-2.5 px-3 text-right text-sky-400 font-semibold tabular-nums">{row.mean_retention_pct.toFixed(2)}%</td>
                          <td className="py-2.5 px-3 text-right text-slate-300 tabular-nums">{row.median_retention_pct.toFixed(2)}%</td>
                          <td className="py-2.5 px-3 text-right text-emerald-400 tabular-nums">{row.weighted_retention_pct.toFixed(2)}%</td>
                        </tr>
                      ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        )}

        {/* ========================================================= */}
        {/* SECTION 4: OBJECT RETENTION DISTRIBUTION (STARVATION)     */}
        {/* ========================================================= */}
        {activeSection === 'distribution' && (
          <div className="space-y-6 max-w-6xl mx-auto">
            <div className="rounded border border-lab-border bg-lab-surface p-5">
              <div className="flex flex-wrap items-center justify-between gap-4 mb-4">
                <div>
                  <h3 className="text-sm font-semibold text-white flex items-center space-x-2">
                    <AlertTriangle className="w-4 h-4 text-amber-400" />
                    <span>Object Retention Starvation Distribution ({distributionClass})</span>
                  </h3>
                  <p className="text-xs text-lab-textMuted mt-0.5">
                    Cumulative percentage of object instances falling below 10%, 20%, 40%, and 60% point retention.
                  </p>
                </div>

                {/* Class selector */}
                <div className="flex items-center space-x-0.5 bg-[#0a0c10] p-0.5 rounded border border-lab-border text-xs">
                  {['Car', 'Pedestrian', 'Cyclist'].map((cls) => (
                    <button
                      key={cls}
                      onClick={() => setDistributionClass(cls)}
                      className={`px-3 py-1 rounded transition-colors ${
                        distributionClass === cls
                          ? 'bg-[#1a202c] text-white font-medium border border-[#2b3342]'
                          : 'text-lab-textMuted hover:text-slate-200'
                      }`}
                    >
                      {cls}
                    </button>
                  ))}
                </div>
              </div>

              {/* Explanatory Callout Box */}
              <div className="mb-4 p-3 rounded bg-[#0a0c10] border border-lab-border text-xs text-lab-textMuted leading-relaxed flex items-start space-x-2.5">
                <Info className="w-4 h-4 text-sky-400 shrink-0 mt-0.5" />
                <div>
                  <strong className="text-slate-200">Interpretation Rule: </strong>
                  Lower percentages below a threshold indicate <strong className="text-white">fewer severely under-retained (starved) objects</strong>.
                  For example, under Uniform sampling, <strong className="text-white font-mono">98.40%</strong> of cars retain &lt;40% of their points (often 0–1 points at range), causing missed detections. Adaptive V6 (α=0.25) cuts this to <strong className="text-white font-mono">46.12%</strong>.
                </div>
              </div>

              {/* Chart */}
              <div className="h-72 w-full">
                {distributionChartData && (
                  <Bar
                    data={distributionChartData}
                    options={{
                      ...chartBaseOptions,
                      scales: {
                        ...chartBaseOptions.scales,
                        y: {
                          ...chartBaseOptions.scales.y,
                          max: 100,
                          title: { display: true, text: '% OF OBJECTS UNDER THRESHOLD', color: '#9ca3af' },
                        },
                      },
                    }}
                  />
                )}
              </div>
            </div>

            {/* Exact Distribution Table */}
            <div className="rounded border border-lab-border bg-lab-surface p-5 space-y-3">
              <h4 className="text-xs font-semibold text-lab-textDim uppercase tracking-wider">
                Verified Cumulative Retention Starvation Ratios ({distributionClass})
              </h4>
              <div className="overflow-x-auto">
                <table className="w-full text-xs font-mono text-left border-collapse">
                  <thead>
                    <tr className="border-b border-lab-border bg-[#0e1116] text-lab-textDim text-[11px] font-semibold">
                      <th className="py-2.5 px-3 font-sans">Method</th>
                      <th className="py-2.5 px-3 text-right">Total Objects</th>
                      <th className="py-2.5 px-3 text-right">&lt; 10% Points</th>
                      <th className="py-2.5 px-3 text-right">&lt; 20% Points</th>
                      <th className="py-2.5 px-3 text-right">&lt; 40% Points</th>
                      <th className="py-2.5 px-3 text-right">&lt; 60% Points</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-lab-border text-slate-300">
                    {distributionMetrics
                      .filter((d) => d.class === distributionClass)
                      .map((row) => (
                        <tr key={row.method_id} className="hover:bg-[#141820] transition-colors">
                          <td className="py-2.5 px-3 font-bold text-white flex items-center space-x-2 font-sans">
                            <span
                              className="w-2 h-2 rounded-full"
                              style={{ backgroundColor: METHOD_COLORS[row.method_id] }}
                            />
                            <span>{row.method_name}</span>
                          </td>
                          <td className="py-2.5 px-3 text-right text-lab-textDim tabular-nums">{row.object_count.toLocaleString()}</td>
                          <td className="py-2.5 px-3 text-right text-slate-300 tabular-nums">{row.pct_below_10pct.toFixed(2)}%</td>
                          <td className="py-2.5 px-3 text-right text-slate-300 tabular-nums">{row.pct_below_20pct.toFixed(2)}%</td>
                          <td
                            className={`py-2.5 px-3 text-right font-bold tabular-nums ${
                              row.pct_below_40pct > 80 ? 'text-rose-400' : 'text-emerald-400'
                            }`}
                          >
                            {row.pct_below_40pct.toFixed(2)}%
                          </td>
                          <td className="py-2.5 px-3 text-right text-lab-textDim tabular-nums">{row.pct_below_60pct.toFixed(2)}%</td>
                        </tr>
                      ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        )}

        {/* ========================================================= */}
        {/* SECTION 5: DETECTOR PERFORMANCE (MODERATE AP_R40)         */}
        {/* ========================================================= */}
        {activeSection === 'detector' && (
          <div className="space-y-6 max-w-6xl mx-auto">
            {/* Grouped AP Comparison Chart */}
            <div className="rounded border border-lab-border bg-lab-surface p-5">
              <div className="flex items-center justify-between mb-4">
                <div>
                  <h3 className="text-sm font-semibold text-white flex items-center space-x-2">
                    <Target className="w-4 h-4 text-sky-400" />
                    <span>PointPillars 3D Object Detection: Moderate AP_R40 Comparison</span>
                  </h3>
                  <p className="text-xs text-lab-textMuted mt-0.5">
                    Official 40-point interpolated Average Precision on Moderate difficulty benchmark setting.
                  </p>
                </div>
                <span className="text-xs font-mono px-2 py-0.5 rounded bg-[#141820] border border-lab-border text-slate-300">
                  MODERATE AP_R40
                </span>
              </div>

              <div className="h-72 w-full">
                <Bar
                  data={detectorChartData}
                  options={{
                    ...chartBaseOptions,
                    scales: {
                      ...chartBaseOptions.scales,
                      y: {
                        ...chartBaseOptions.scales.y,
                        max: 80,
                        title: { display: true, text: 'MODERATE AP_R40 (%)', color: '#9ca3af' },
                      },
                    },
                  }}
                />
              </div>
            </div>

            {/* Mean AP (mAP) & Relative Retention against Full Cloud */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              {/* Mean AP (mAP) */}
              <div className="rounded border border-lab-border bg-lab-surface p-5 flex flex-col justify-between">
                <div>
                  <h4 className="text-xs font-semibold text-white uppercase tracking-wider mb-1">
                    Mean Moderate AP_R40 (mAP Across Car, Pedestrian, Cyclist)
                  </h4>
                  <p className="text-[11px] text-lab-textDim mb-3">
                    Arithmetic mean across all three evaluated benchmark classes.
                  </p>
                </div>

                <div className="h-56 w-full">
                  <Bar
                    data={{
                      labels: mapAndRelativeData.labels,
                      datasets: [
                        {
                          label: 'Mean AP_R40 (%)',
                          data: mapAndRelativeData.mAPValues,
                          backgroundColor: [
                            '#94a3b8cc',
                            '#f43f5ecc',
                            '#f59e0bcc',
                            '#10b981cc',
                            '#38bdf8cc',
                          ],
                          borderWidth: 1,
                        },
                      ],
                    }}
                    options={{
                      ...chartBaseOptions,
                      scales: {
                        ...chartBaseOptions.scales,
                        y: {
                          ...chartBaseOptions.scales.y,
                          max: 60,
                          title: { display: true, text: 'mAP (%)', color: '#9ca3af' },
                        },
                      },
                    }}
                  />
                </div>
              </div>

              {/* Relative Retention vs Full Cloud Baseline */}
              <div className="rounded border border-lab-border bg-lab-surface p-5 flex flex-col justify-between">
                <div>
                  <h4 className="text-xs font-semibold text-emerald-400 uppercase tracking-wider mb-1">
                    Relative AP Retention Against Full Cloud Baseline (100% Pts)
                  </h4>
                  <p className="text-[11px] text-lab-textDim mb-3">
                    Percentage of original full-cloud detector accuracy retained at 20% point budget.
                  </p>
                </div>

                <div className="h-56 w-full">
                  <Bar
                    data={{
                      labels: mapAndRelativeData.labels,
                      datasets: [
                        {
                          label: 'Relative AP Retained (%)',
                          data: mapAndRelativeData.relAPValues,
                          backgroundColor: [
                            '#94a3b8cc',
                            '#f43f5ecc',
                            '#f59e0bcc',
                            '#10b981cc',
                            '#38bdf8cc',
                          ],
                          borderWidth: 1,
                        },
                      ],
                    }}
                    options={{
                      ...chartBaseOptions,
                      scales: {
                        ...chartBaseOptions.scales,
                        y: {
                          ...chartBaseOptions.scales.y,
                          max: 110,
                          title: { display: true, text: 'RELATIVE AP (%)', color: '#9ca3af' },
                        },
                      },
                    }}
                  />
                </div>
              </div>
            </div>

            {/* Official Measured Moderate AP_R40 Benchmark Table */}
            <div className="rounded border border-lab-border bg-lab-surface p-5 space-y-3">
              <h4 className="text-xs font-semibold text-lab-textDim uppercase tracking-wider">
                Official Benchmark Results Table (Moderate AP_R40)
              </h4>
              <div className="overflow-x-auto">
                <table className="w-full text-xs font-mono text-left border-collapse">
                  <thead>
                    <tr className="border-b border-lab-border bg-[#0e1116] text-lab-textDim text-[11px] font-semibold">
                      <th className="py-2.5 px-3 font-sans">Method</th>
                      <th className="py-2.5 px-3">Point Quota</th>
                      <th className="py-2.5 px-3 text-right">Car Moderate AP</th>
                      <th className="py-2.5 px-3 text-right">Pedestrian Moderate AP</th>
                      <th className="py-2.5 px-3 text-right">Cyclist Moderate AP</th>
                      <th className="py-2.5 px-3 text-right font-semibold">Mean AP (mAP)</th>
                      <th className="py-2.5 px-3 text-right">Relative Retained %</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-lab-border text-slate-300">
                    {[
                      { id: 'full', name: 'Full Cloud Baseline', quota: '100%' },
                      { id: 'uniform', name: 'Uniform 20% Baseline', quota: '20%' },
                      { id: 'hybrid_v5', name: 'Hybrid V5', quota: '20%' },
                      { id: 'v6_alpha_0.25', name: 'V6 α=0.25', quota: '20%' },
                      { id: 'v7_lambda_0.25', name: 'V7 λ=0.25', quota: '20%' },
                    ].map((row) => {
                      const scores = DETECTOR_MODERATE_AP[row.id];
                      const mAP = (scores.Car + scores.Pedestrian + scores.Cyclist) / 3;
                      const fullMeanAP = (72.9095 + 43.8000 + 33.3889) / 3;
                      const rel = (mAP / fullMeanAP) * 100;

                      return (
                        <tr
                          key={row.id}
                          className={`hover:bg-[#141820] transition-colors ${
                            row.id === 'v6_alpha_0.25' ? 'bg-[#121820]' : ''
                          }`}
                        >
                          <td className="py-3 px-3 font-semibold text-white flex items-center space-x-2 font-sans">
                            <span
                              className="w-2 h-2 rounded-full"
                              style={{ backgroundColor: METHOD_COLORS[row.id] }}
                            />
                            <span>{row.name}</span>
                          </td>
                          <td className="py-3 px-3 text-lab-textDim tabular-nums">{row.quota}</td>
                          <td className="py-3 px-3 text-right text-slate-200 tabular-nums">{scores.Car.toFixed(4)}%</td>
                          <td className="py-3 px-3 text-right text-slate-200 tabular-nums">{scores.Pedestrian.toFixed(4)}%</td>
                          <td
                            className={`py-3 px-3 text-right font-bold tabular-nums ${
                              row.id === 'uniform' ? 'text-rose-400 font-extrabold' : 'text-slate-200'
                            }`}
                          >
                            {scores.Cyclist.toFixed(4)}%
                          </td>
                          <td className="py-3 px-3 text-right text-sky-400 font-bold tabular-nums">{mAP.toFixed(4)}%</td>
                          <td className="py-3 px-3 text-right text-emerald-400 font-semibold tabular-nums">{rel.toFixed(2)}%</td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        )}

        {/* ========================================================= */}
        {/* SECTION 6: INTERACTIVE ALGORITHM COMPARATOR               */}
        {/* ========================================================= */}
        {activeSection === 'comparator' && (
          <div className="space-y-6 max-w-6xl mx-auto">
            {/* Controls */}
            <div className="rounded border border-lab-border bg-lab-surface p-5 space-y-4">
              <div className="flex flex-wrap items-center justify-between gap-4">
                <div>
                  <h3 className="text-sm font-semibold text-white flex items-center space-x-2">
                    <Sliders className="w-4 h-4 text-sky-400" />
                    <span>Interactive Head-to-Head Algorithm Comparator</span>
                  </h3>
                  <p className="text-xs text-lab-textMuted mt-0.5">
                    Select any two or more methods to compare point retention, distance curves, starvation risk, and detector AP.
                  </p>
                </div>

                {/* Method selector checkboxes */}
                <div className="flex flex-wrap items-center gap-1.5 bg-[#0a0c10] p-1 rounded border border-lab-border">
                  {[
                    { id: 'uniform', label: 'Uniform 20%' },
                    { id: 'hybrid_v5', label: 'Hybrid V5' },
                    { id: 'v6_alpha_0.25', label: 'V6 (α=0.25)' },
                    { id: 'v7_lambda_0.25', label: 'V7 (λ=0.25)' },
                  ].map((m) => {
                    const isChecked = comparedMethods.includes(m.id);
                    return (
                      <button
                        key={m.id}
                        onClick={() => toggleComparatorMethod(m.id)}
                        className={`flex items-center space-x-1.5 px-2.5 py-1 rounded text-xs transition-colors ${
                          isChecked
                            ? 'bg-[#1a202c] text-white border border-[#2b3342] font-medium'
                            : 'text-lab-textDim hover:text-slate-300'
                        }`}
                      >
                        <span
                          className={`w-1.5 h-1.5 rounded-full ${isChecked ? 'bg-sky-400' : 'bg-slate-600'}`}
                        />
                        <span>{m.label}</span>
                      </button>
                    );
                  })}
                </div>
              </div>

              {/* Comparative Feature Matrix */}
              <div className="overflow-x-auto pt-2">
                <table className="w-full text-xs font-mono text-left border-collapse">
                  <thead>
                    <tr className="border-b border-lab-border bg-[#0e1116] text-lab-textDim text-[11px] font-semibold">
                      <th className="py-2.5 px-3 font-sans">Criterion / Dimension</th>
                      {comparedMethods.map((m) => (
                        <th key={m} className="py-2.5 px-3 text-center font-sans font-semibold text-slate-200">
                          {METHOD_NAMES[m]}
                        </th>
                      ))}
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-lab-border text-slate-300">
                    {/* Dimension: Point Retention */}
                    <tr className="hover:bg-[#141820]">
                      <td className="py-2.5 px-3 font-medium text-slate-300 font-sans">Target Retention Budget</td>
                      {comparedMethods.map((m) => (
                        <td key={m} className="py-2.5 px-3 text-center text-slate-200 tabular-nums">20.00%</td>
                      ))}
                    </tr>
                    <tr className="hover:bg-[#141820]">
                      <td className="py-2.5 px-3 font-medium text-slate-300 font-sans">Weighted Retention Across 3,769 Frames</td>
                      {comparedMethods.map((m) => (
                        <td key={m} className="py-2.5 px-3 text-center text-emerald-400 font-semibold tabular-nums">19.9997%</td>
                      ))}
                    </tr>

                    {/* Dimension: Sampling Characteristics */}
                    <tr className="hover:bg-[#141820] bg-[#0c0f14]">
                      <td className="py-2.5 px-3 font-medium text-slate-300 font-sans">Voxel Size (m)</td>
                      {comparedMethods.map((m) => (
                        <td key={m} className="py-2.5 px-3 text-center text-sky-400 tabular-nums">0.8 m</td>
                      ))}
                    </tr>
                    <tr className="hover:bg-[#141820] bg-[#0c0f14]">
                      <td className="py-2.5 px-3 font-medium text-slate-300 font-sans">Density Weighting Factor (α)</td>
                      {comparedMethods.map((m) => (
                        <td key={m} className="py-2.5 px-3 text-center text-slate-200 tabular-nums">
                          {m === 'v6_alpha_0.25' ? '0.25' : m === 'v7_lambda_0.25' ? '0.25' : 'N/A'}
                        </td>
                      ))}
                    </tr>
                    <tr className="hover:bg-[#141820] bg-[#0c0f14]">
                      <td className="py-2.5 px-3 font-medium text-slate-300 font-sans">Score Convex Parameter (λ)</td>
                      {comparedMethods.map((m) => (
                        <td key={m} className="py-2.5 px-3 text-center text-slate-200 tabular-nums">
                          {m === 'v7_lambda_0.25' ? '0.25' : 'N/A'}
                        </td>
                      ))}
                    </tr>

                    {/* Dimension: Distance Retention */}
                    <tr className="hover:bg-[#141820]">
                      <td className="py-2.5 px-3 font-medium text-slate-300 font-sans">Far (&gt;40m) Car Mean Retention %</td>
                      {comparedMethods.map((m) => {
                        const item = distanceMetrics.find(
                          (d) => d.class === 'Car' && d.distance_group === 'far_over_40m' && (d.method_id === m || d.method_id === m.replace('hybrid_', ''))
                        );
                        return (
                          <td key={m} className="py-2.5 px-3 text-center text-sky-400 font-semibold tabular-nums">
                            {item ? `${item.mean_retention_pct.toFixed(1)}%` : '-'}
                          </td>
                        );
                      })}
                    </tr>
                    <tr className="hover:bg-[#141820]">
                      <td className="py-2.5 px-3 font-medium text-slate-300 font-sans">Far (&gt;40m) Cyclist Mean Retention %</td>
                      {comparedMethods.map((m) => {
                        const item = distanceMetrics.find(
                          (d) => d.class === 'Cyclist' && d.distance_group === 'far_over_40m' && (d.method_id === m || d.method_id === m.replace('hybrid_', ''))
                        );
                        return (
                          <td key={m} className="py-2.5 px-3 text-center text-sky-400 font-semibold tabular-nums">
                            {item ? `${item.mean_retention_pct.toFixed(1)}%` : '-'}
                          </td>
                        );
                      })}
                    </tr>

                    {/* Dimension: Object Retention Starvation */}
                    <tr className="hover:bg-[#141820] bg-[#0c0f14]">
                      <td className="py-2.5 px-3 font-medium text-slate-300 font-sans">Cars Starved (&lt;40% Points)</td>
                      {comparedMethods.map((m) => {
                        const item = distributionMetrics.find(
                          (d) => d.class === 'Car' && (d.method_id === m || d.method_id === m.replace('hybrid_', ''))
                        );
                        return (
                          <td
                            key={m}
                            className={`py-2.5 px-3 text-center font-bold tabular-nums ${
                              item && item.pct_below_40pct > 80 ? 'text-rose-400' : 'text-emerald-400'
                            }`}
                          >
                            {item ? `${item.pct_below_40pct.toFixed(1)}%` : '-'}
                          </td>
                        );
                      })}
                    </tr>

                    {/* Dimension: PointPillars Detector AP */}
                    <tr className="hover:bg-[#141820]">
                      <td className="py-2.5 px-3 font-medium text-slate-300 font-sans">Car Moderate AP</td>
                      {comparedMethods.map((m) => (
                        <td key={m} className="py-2.5 px-3 text-center text-slate-100 font-semibold tabular-nums">
                          {DETECTOR_MODERATE_AP[m]?.Car.toFixed(2)}%
                        </td>
                      ))}
                    </tr>
                    <tr className="hover:bg-[#141820]">
                      <td className="py-2.5 px-3 font-medium text-slate-300 font-sans">Pedestrian Moderate AP</td>
                      {comparedMethods.map((m) => (
                        <td key={m} className="py-2.5 px-3 text-center text-slate-100 font-semibold tabular-nums">
                          {DETECTOR_MODERATE_AP[m]?.Pedestrian.toFixed(2)}%
                        </td>
                      ))}
                    </tr>
                    <tr className="hover:bg-[#141820]">
                      <td className="py-2.5 px-3 font-medium text-slate-300 font-sans">Cyclist Moderate AP</td>
                      {comparedMethods.map((m) => (
                        <td
                          key={m}
                          className={`py-2.5 px-3 text-center font-bold tabular-nums ${
                            m === 'uniform' ? 'text-rose-400' : 'text-emerald-400'
                          }`}
                        >
                          {DETECTOR_MODERATE_AP[m]?.Cyclist.toFixed(2)}%
                        </td>
                      ))}
                    </tr>
                    <tr className="hover:bg-[#141820] bg-[#121820]">
                      <td className="py-2.5 px-3 font-semibold text-sky-400 font-sans">Overall Moderate mAP</td>
                      {comparedMethods.map((m) => {
                        const scores = DETECTOR_MODERATE_AP[m];
                        const mAP = (scores.Car + scores.Pedestrian + scores.Cyclist) / 3;
                        return (
                          <td key={m} className="py-2.5 px-3 text-center text-sky-300 font-bold text-sm tabular-nums">
                            {mAP.toFixed(2)}%
                          </td>
                        );
                      })}
                    </tr>
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
