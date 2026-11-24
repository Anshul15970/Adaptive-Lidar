import React, { useState, useEffect, useCallback, useRef } from 'react';
import ThreePointCloudViewer from '../components/ThreePointCloudViewer';
import PlaybackToolbar from '../components/PlaybackToolbar';
import {
  fetchFramePointsBinary,
  fetchGroundTruth,
  fetchFrameMetrics,
} from '../api';
import {
  Boxes,
  Palette,
  Eye,
  EyeOff,
  Columns,
  Box,
  RotateCcw,
  Sparkles,
  Info,
  BookOpen,
  X,
  Maximize2,
  Minimize2,
  ChevronLeft,
  ChevronRight,
  Shuffle,
  RefreshCw,
  Sliders,
  Check,
  Compass,
} from 'lucide-react';

// Structured, accurate research annotations matching the experimental study
const RESEARCH_ANNOTATIONS = {
  original: {
    id: 'original',
    name: 'Original Point Cloud',
    badge: 'Raw Sensor Baseline',
    color: '#94a3b8',
    budget: '100.0% (Uncompressed)',
    formula: 'P_{raw} = \\{(x, y, z, i)\\} \\subset \\mathbb{R}^4',
    concept: 'Unprocessed high-definition LiDAR sweep captured by the roof-mounted 64-beam Velodyne HDL-64E spinning sensor at 10 Hz.',
    characteristics: [
      'Acquires ~115k–130k raw 3D spatial points per frame.',
      'Exhibits severe natural 1/r² radial beam attenuation with distance.',
      'Dense oversampling on near-field road surfaces (<15m) contrasts with extreme sparsity at long range (>40m).',
      'Serves as the reference benchmark for PointPillars 3D detection evaluation.',
    ],
    detectorInsight: 'Baseline detector benchmark: PointPillars achieves 77.98% Car AP and 52.88% Pedestrian AP (Moderate) on uncompressed frames.',
  },
  uniform: {
    id: 'uniform',
    name: 'Uniform 20% Sampling',
    badge: 'Random Baseline',
    color: '#f43f5e',
    budget: '20.0% Target Budget',
    formula: 'P(p_i \\in S) = 0.20, \\quad \\forall p_i \\in P_{raw}',
    concept: 'Standard uniform random decimation where each point has an equal, independent 20% retention probability.',
    characteristics: [
      'Spatially uniform decimation independent of geometric complexity or distance.',
      'Applies the exact same decimation rate to dense road surfaces and critically sparse distant pedestrians.',
      'Disproportionately discards points on distant objects where returns are already scarce.',
      'Causes severe degradation in 3D bounding box regression and object classification.',
    ],
    detectorInsight: 'Significant detection penalty: Pedestrian Moderate AP drops sharply by ~5.0% compared to Original due to far-field point starvation.',
  },
  hybrid_v5: {
    id: 'hybrid_v5',
    name: 'Hybrid V5 Sampling',
    badge: 'Geometry & Sparsity',
    color: '#f59e0b',
    budget: '20.0% Target Budget',
    formula: 'S_i = f(\\sigma_n, N_v), \\quad v = 0.8\\text{m}',
    concept: 'Geometry complexity and voxel sparsity-based sampling operating over an 0.8m 3D voxel discretization.',
    characteristics: [
      'Discretizes point cloud into 0.8m voxels and computes local surface variation (eigenvalue curvature).',
      'Prioritizes retention in low-density voxels and high-curvature geometric surfaces.',
      'Aggressively prunes redundant points on flat, planar ground surfaces.',
      'Allocates a normalized per-frame budget targeting approximately 20% overall points.',
    ],
    detectorInsight: 'Substantially improves vehicle 3D detection AP over Uniform by preserving sharp vehicle edges and bounding boundaries.',
  },
  'v6_alpha_0.25': {
    id: 'v6_alpha_0.25',
    name: 'V6 Density-Aware (α=0.25)',
    badge: 'Density-Aware Adaptive',
    color: '#10b981',
    budget: '20.0% Target Budget',
    formula: 'w_i \\propto (1 + \\rho_i)^{-\\alpha}, \\quad \\alpha = 0.25',
    concept: 'Continuous density-aware adaptive downsampling inversely weighted by local neighborhood spatial density.',
    characteristics: [
      'Power-law weighting inversely penalizes points in ultra-dense spatial clusters.',
      'Distant LiDAR returns (>40m) receive near 100% point retention probability.',
      'Near-field road surfaces (<15m) are compressed to satisfy the overall 20% budget.',
      'Guarantees preservation of isolated pedestrians, cyclists, and distant vehicles.',
    ],
    detectorInsight: 'Recovers critical pedestrian and cyclist recall at long range, boosting Moderate and Hard AP without increasing point budget.',
  },
  'v7_lambda_0.25': {
    id: 'v7_lambda_0.25',
    name: 'V7 Score-Density (λ=0.25)',
    badge: 'Hybrid Score-Density',
    color: '#38bdf8',
    budget: '20.0% Target Budget',
    formula: 'S_i = (1 - \\lambda) S_{\\text{geom}}(i) + \\lambda S_{\\text{density}}(i), \\quad \\lambda = 0.25',
    concept: 'State-of-the-art hybrid formulation combining geometric curvature importance with distance-density compensation.',
    characteristics: [
      'Harmonizes 3D eigenvalue structure preservation with density-inversion weighting.',
      'Retains sharp geometric corners, edges, and object contours.',
      'Compensates for 1/r² beam attenuation across Near, Medium, and Far distance bands.',
      'Generates the most balanced spatial point distribution for voxelized pillar representations.',
    ],
    detectorInsight: 'Achieves the highest overall 3D detection AP on PointPillars across all evaluated algorithms (+1.5% to +4.5% AP over Uniform).',
  },
};

export default function PresentationView({
  currentFrameId,
  validationFrames = [],
  algorithms = [],
  selectedMethod = 'v6_alpha_0.25',
  onSelectMethod,
  onPreviousFrame,
  onNextFrame,
  onRandomFrame,
  onClosePresentation,
  playbackProps = {},
}) {
  // View presentation state
  const [viewMode, setViewMode] = useState('single'); // 'single' | 'comparison'
  const [compareLeftMethod, setCompareLeftMethod] = useState('uniform');
  const [compareRightMethod, setCompareRightMethod] = useState(selectedMethod || 'v6_alpha_0.25');

  // Visualization settings
  const [colorMode, setColorMode] = useState('elevation'); // 'elevation' | 'intensity' | 'distance' | 'solid'
  const [showGroundTruth, setShowGroundTruth] = useState(true);
  const [pointSize, setPointSize] = useState(1.8);
  const [cameraPreset, setCameraPreset] = useState(null);

  // Synchronized camera state for comparison view
  const [sharedCameraState, setSharedCameraState] = useState(null);

  // Research annotation drawer & active note
  const [showAnnotations, setShowAnnotations] = useState(false);
  const [activeAnnotationMethod, setActiveAnnotationMethod] = useState(selectedMethod || 'v6_alpha_0.25');

  // Fullscreen tracking
  const [isFullscreen, setIsFullscreen] = useState(Boolean(document.fullscreenElement));

  // Single view point cloud data
  const [singlePoints, setSinglePoints] = useState(null);
  const [singleMeta, setSingleMeta] = useState(null);

  // Comparison view point cloud data
  const [comparePoints, setComparePoints] = useState({});
  const [compareMeta, setCompareMeta] = useState({});

  // Ground truth & metrics
  const [groundTruthObjects, setGroundTruthObjects] = useState([]);
  const [frameMetrics, setFrameMetrics] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  // Keep right compare method aligned with selectedMethod if user changes it
  useEffect(() => {
    setCompareRightMethod(selectedMethod);
    setActiveAnnotationMethod(selectedMethod);
  }, [selectedMethod]);

  // Sync camera preset triggers
  const triggerCameraPreset = (preset) => {
    setCameraPreset(preset);
    setTimeout(() => setCameraPreset(null), 50);
  };

  const handleCameraChange = useCallback((camState) => {
    setSharedCameraState(camState);
  }, []);

  // Browser fullscreen toggle
  const toggleFullscreen = () => {
    if (!document.fullscreenElement) {
      document.documentElement.requestFullscreen().catch(() => {});
      setIsFullscreen(true);
    } else {
      document.exitFullscreen().catch(() => {});
      setIsFullscreen(false);
    }
  };

  // Monitor browser fullscreen change events
  useEffect(() => {
    const handleFsChange = () => {
      setIsFullscreen(Boolean(document.fullscreenElement));
    };
    document.addEventListener('fullscreenchange', handleFsChange);
    return () => document.removeEventListener('fullscreenchange', handleFsChange);
  }, []);

  // Load point clouds, ground truth, and metrics
  useEffect(() => {
    let isCancelled = false;

    async function loadPresentationData() {
      setLoading(true);
      setError(null);
      try {
        if (viewMode === 'single') {
          const [ptsRes, gtRes, metricsRes] = await Promise.all([
            fetchFramePointsBinary(currentFrameId, selectedMethod, 80000),
            fetchGroundTruth(currentFrameId, null, selectedMethod),
            fetchFrameMetrics(currentFrameId),
          ]);
          if (isCancelled) return;
          setSinglePoints(ptsRes.points);
          setSingleMeta(ptsRes.metadata);
          setGroundTruthObjects(gtRes.objects || []);
          setFrameMetrics(metricsRes);
        } else {
          // Comparison View
          const [leftPts, rightPts, gtRes, metricsRes] = await Promise.all([
            fetchFramePointsBinary(currentFrameId, compareLeftMethod, 50000),
            fetchFramePointsBinary(currentFrameId, compareRightMethod, 50000),
            fetchGroundTruth(currentFrameId),
            fetchFrameMetrics(currentFrameId),
          ]);
          if (isCancelled) return;
          setComparePoints({
            [compareLeftMethod]: leftPts.points,
            [compareRightMethod]: rightPts.points,
          });
          setCompareMeta({
            [compareLeftMethod]: leftPts.metadata,
            [compareRightMethod]: rightPts.metadata,
          });
          setGroundTruthObjects(gtRes.objects || []);
          setFrameMetrics(metricsRes);
        }
      } catch (err) {
        if (!isCancelled) {
          setError(err.message || 'Failed to load frame data');
        }
      } finally {
        if (!isCancelled) {
          setLoading(false);
        }
      }
    }

    loadPresentationData();
    return () => {
      isCancelled = true;
    };
  }, [currentFrameId, viewMode, selectedMethod, compareLeftMethod, compareRightMethod]);

  // Sync loading state with parent playback controller
  useEffect(() => {
    playbackProps.onLoadingChange?.(loading);
  }, [loading, playbackProps.onLoadingChange]);

  // Keyboard Shortcuts for Presentation Mode
  useEffect(() => {
    const handleKeyDown = (e) => {
      if (['INPUT', 'SELECT', 'TEXTAREA'].includes(document.activeElement?.tagName)) {
        return;
      }
      if (e.key === 'Escape') {
        e.preventDefault();
        onClosePresentation();
      } else if (e.key === 'a' || e.key === 'A') {
        e.preventDefault();
        setShowAnnotations((prev) => !prev);
      } else if (e.key === 'b' || e.key === 'B') {
        e.preventDefault();
        setShowGroundTruth((prev) => !prev);
      } else if (e.key === 'v' || e.key === 'V') {
        e.preventDefault();
        setViewMode((prev) => (prev === 'single' ? 'comparison' : 'single'));
      } else if (e.key === 'c' || e.key === 'C') {
        e.preventDefault();
        setColorMode((prev) => {
          const modes = ['elevation', 'intensity', 'distance', 'solid'];
          const nextIdx = (modes.indexOf(prev) + 1) % modes.length;
          return modes[nextIdx];
        });
      } else if (e.key === '1') {
        onSelectMethod('original');
      } else if (e.key === '2') {
        onSelectMethod('uniform');
      } else if (e.key === '3') {
        onSelectMethod('hybrid_v5');
      } else if (e.key === '4') {
        onSelectMethod('v6_alpha_0.25');
      } else if (e.key === '5') {
        onSelectMethod('v7_lambda_0.25');
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [onClosePresentation, onSelectMethod]);

  // Compute live quantitative metrics for active single method
  const origPoints = frameMetrics?.retention?.original_points || singleMeta?.totalPoints || 0;
  const activeMethodStats = frameMetrics?.retention?.methods?.[selectedMethod] || {
    sampled_points: singleMeta?.totalPoints || 0,
    retention_pct: selectedMethod === 'original' ? 100.0 : 20.0,
  };

  const activeAnnotation = RESEARCH_ANNOTATIONS[activeAnnotationMethod] || RESEARCH_ANNOTATIONS['v6_alpha_0.25'];

  return (
    <div className="fixed inset-0 z-50 bg-[#0a0c10] text-slate-100 flex flex-col overflow-hidden select-none">
      {/* 1. Ultra-Slim Presenter Header Ribbon */}
      <header className="h-11 bg-[#101318]/90 backdrop-blur-md border-b border-[#1f242e] px-4 flex items-center justify-between shrink-0 z-30">
        {/* Left: Branding & Frame Stepper */}
        <div className="flex items-center space-x-3">
          <div className="flex items-center space-x-2">
            <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse" />
            <span className="font-semibold text-xs tracking-wider uppercase text-slate-200">
              Adaptive LiDAR Lab
            </span>
            <span className="text-[10px] font-mono px-1.5 py-0.5 rounded bg-sky-950/60 text-sky-400 border border-sky-800/60 font-semibold tracking-wider">
              PRESENTATION COCKPIT
            </span>
          </div>

          <span className="text-[#1f242e]">|</span>

          {/* Frame Stepper */}
          <div className="flex items-center bg-[#0a0c10] rounded border border-[#1f242e] overflow-hidden">
            <button
              onClick={onPreviousFrame}
              title="Previous Frame [Left Arrow]"
              className="px-2 py-1 text-slate-400 hover:text-white hover:bg-[#161a22] transition-colors border-r border-[#1f242e]"
            >
              <ChevronLeft className="w-3.5 h-3.5" />
            </button>
            <div className="px-2.5 py-0.5 font-mono text-xs flex items-center space-x-1.5">
              <span className="text-[10px] text-slate-500 uppercase tracking-wider">FRAME</span>
              <span className="text-sky-400 font-semibold tracking-wider tabular-nums">
                {currentFrameId}
              </span>
            </div>
            <button
              onClick={onNextFrame}
              title="Next Frame [Right Arrow]"
              className="px-2 py-1 text-slate-400 hover:text-white hover:bg-[#161a22] transition-colors border-l border-[#1f242e]"
            >
              <ChevronRight className="w-3.5 h-3.5" />
            </button>
          </div>

          <button
            onClick={onRandomFrame}
            title="Random Frame"
            className="p-1.5 text-slate-400 hover:text-white hover:bg-[#141820] border border-[#1f242e] rounded transition-colors"
          >
            <Shuffle className="w-3.5 h-3.5" />
          </button>

          <span className="text-[11px] font-mono text-slate-500 hidden sm:inline">
            ({validationFrames.length.toLocaleString()} VAL FRAMES)
          </span>
        </div>

        {/* Center: View Mode Switcher (Single vs Dual Comparison) */}
        <div className="flex items-center bg-[#0a0c10] p-0.5 rounded border border-[#1f242e]">
          <button
            onClick={() => setViewMode('single')}
            className={`flex items-center space-x-1.5 px-3 py-1 rounded text-xs font-medium transition-colors ${
              viewMode === 'single'
                ? 'bg-[#181f2a] text-sky-400 shadow-sm border border-sky-800/40'
                : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            <Box className="w-3.5 h-3.5" />
            <span>Single 3D</span>
          </button>
          <button
            onClick={() => setViewMode('comparison')}
            className={`flex items-center space-x-1.5 px-3 py-1 rounded text-xs font-medium transition-colors ${
              viewMode === 'comparison'
                ? 'bg-[#181f2a] text-sky-400 shadow-sm border border-sky-800/40'
                : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            <Columns className="w-3.5 h-3.5" />
            <span>Dual Comparison</span>
          </button>
        </div>

        {/* Right: Annotations, Fullscreen, and Exit */}
        <div className="flex items-center space-x-2">
          {/* Research Annotations Toggle */}
          <button
            onClick={() => setShowAnnotations(!showAnnotations)}
            title="Toggle Research Annotations [A]"
            className={`flex items-center space-x-1.5 px-2.5 py-1 rounded text-xs font-medium transition-colors ${
              showAnnotations
                ? 'bg-amber-950/50 text-amber-300 border border-amber-700/60'
                : 'bg-[#0a0c10] text-slate-400 hover:text-slate-200 border border-[#1f242e]'
            }`}
          >
            <BookOpen className="w-3.5 h-3.5" />
            <span className="hidden md:inline">Research Context</span>
            <kbd className="text-[9px] font-mono px-1 py-0.2 bg-[#161a22] text-slate-400 rounded border border-[#2b3342]">
              A
            </kbd>
          </button>

          {/* Fullscreen Toggle */}
          <button
            onClick={toggleFullscreen}
            title={isFullscreen ? 'Exit Browser Fullscreen' : 'Enter Browser Fullscreen'}
            className="p-1.5 bg-[#0a0c10] text-slate-400 hover:text-slate-200 border border-[#1f242e] rounded transition-colors"
          >
            {isFullscreen ? <Minimize2 className="w-3.5 h-3.5" /> : <Maximize2 className="w-3.5 h-3.5" />}
          </button>

          {/* Exit Presentation Mode */}
          <button
            onClick={onClosePresentation}
            title="Exit Presentation Mode [Esc]"
            className="flex items-center space-x-1.5 px-3 py-1 rounded text-xs font-medium bg-rose-950/40 hover:bg-rose-900/50 text-rose-300 border border-rose-800/60 transition-colors shadow-sm"
          >
            <X className="w-3.5 h-3.5" />
            <span>Exit</span>
            <kbd className="text-[9px] font-mono px-1 py-0.2 bg-rose-900/60 text-rose-200 rounded border border-rose-700/50">
              ESC
            </kbd>
          </button>
        </div>
      </header>

      {/* 2. Main Presentation Canvas Workspace */}
      <div className="flex-1 relative overflow-hidden bg-[#0a0c10]">
        {/* Loading Indicator */}
        {loading && (
          <div className="absolute inset-0 bg-[#0a0c10]/75 backdrop-blur-sm flex items-center justify-center z-30 pointer-events-none">
            <div className="flex items-center space-x-2.5 px-4 py-2 rounded bg-[#101318] border border-[#1f242e] text-xs font-mono text-slate-200 shadow-xl">
              <RefreshCw className="w-4 h-4 text-sky-400 animate-spin" />
              <span>STREAMING HIGH-PRECISION POINT CLOUD...</span>
            </div>
          </div>
        )}

        {/* Error Alert */}
        {error && (
          <div className="absolute top-4 left-4 right-4 bg-rose-950/90 border border-rose-800 text-rose-200 text-xs font-mono p-3 rounded z-30 flex items-center space-x-2">
            <span>{error}</span>
          </div>
        )}

        {/* Viewport Render: Single 3D View vs Dual Comparison */}
        {viewMode === 'single' ? (
          <div className="w-full h-full relative">
            <ThreePointCloudViewer
              pointsData={singlePoints}
              groundTruthObjects={groundTruthObjects}
              colorMode={colorMode}
              pointSize={pointSize}
              showRangeRings={true}
              showGroundTruth={showGroundTruth}
              showPredictions={false}
              visibleClasses={{ Car: true, Pedestrian: true, Cyclist: true }}
              methodId={selectedMethod}
              renderMeta={singleMeta}
              cameraPreset={cameraPreset}
            />

            {/* Floating Telemetry HUD (Top-Left) */}
            <div className="absolute top-4 left-4 z-20 pointer-events-none flex flex-col space-y-2 max-w-sm">
              <div className="bg-[#0a0c10]/85 backdrop-blur-md border border-[#1f242e] p-3 rounded shadow-2xl space-y-2 pointer-events-auto">
                <div className="flex items-center justify-between border-b border-[#1f242e] pb-1.5">
                  <div className="flex items-center space-x-2">
                    <span
                      className="w-2.5 h-2.5 rounded-full"
                      style={{ backgroundColor: activeAnnotation.color }}
                    />
                    <span className="font-semibold text-xs text-white">
                      {activeAnnotation.name}
                    </span>
                  </div>
                  <span className="text-[10px] font-mono px-1.5 py-0.5 rounded bg-[#151922] text-slate-400 border border-[#232b38]">
                    {activeAnnotation.badge}
                  </span>
                </div>

                <div className="grid grid-cols-2 gap-2 text-[11px] font-mono">
                  <div>
                    <span className="text-slate-500 block text-[9px]">SAMPLED POINTS</span>
                    <span className="text-slate-200 font-semibold tabular-nums">
                      {(activeMethodStats.sampled_points || 0).toLocaleString()}
                    </span>
                  </div>
                  <div>
                    <span className="text-slate-500 block text-[9px]">ORIGINAL RAW</span>
                    <span className="text-slate-400 font-semibold tabular-nums">
                      {(origPoints || 0).toLocaleString()}
                    </span>
                  </div>
                  <div>
                    <span className="text-slate-500 block text-[9px]">RETENTION RATE</span>
                    <span className="text-sky-400 font-bold tabular-nums">
                      {activeMethodStats.retention_pct?.toFixed(2)}%
                    </span>
                  </div>
                  <div>
                    <span className="text-slate-500 block text-[9px]">VOXEL GRID</span>
                    <span className="text-emerald-400 font-semibold">0.80 m</span>
                  </div>
                </div>

                {groundTruthObjects.length > 0 && (
                  <div className="border-t border-[#1f242e] pt-1.5 flex items-center justify-between text-[10px] font-mono text-slate-400">
                    <span>3D GROUND TRUTH:</span>
                    <span className="text-emerald-400 font-semibold">
                      {groundTruthObjects.length} OBJECTS
                    </span>
                  </div>
                )}
              </div>
            </div>
          </div>
        ) : (
          /* Dual Comparison Viewport */
          <div className="w-full h-full grid grid-cols-2 gap-1 bg-[#101318] p-1 relative">
            {/* Left Viewport */}
            <div className="relative h-full bg-[#0a0c10] rounded overflow-hidden border border-[#1f242e]">
              {/* Method Selector Header Bar */}
              <div className="absolute top-3 left-3 right-3 z-20 flex items-center justify-between pointer-events-auto">
                <select
                  value={compareLeftMethod}
                  onChange={(e) => setCompareLeftMethod(e.target.value)}
                  className="bg-[#0a0c10]/90 backdrop-blur-md text-xs font-medium text-slate-200 px-2.5 py-1.5 rounded border border-[#232b38] focus:outline-none focus:border-sky-500 shadow-xl"
                >
                  {algorithms.map((algo) => (
                    <option key={algo.id} value={algo.id}>
                      {algo.name} ({algo.target_retention_pct}%)
                    </option>
                  ))}
                </select>

                <div className="bg-[#0a0c10]/90 backdrop-blur-md px-2.5 py-1 rounded border border-[#232b38] font-mono text-[10px] text-slate-300 shadow-xl">
                  <span className="text-slate-500">POINTS: </span>
                  <span className="text-sky-400 font-bold tabular-nums">
                    {(frameMetrics?.retention?.methods?.[compareLeftMethod]?.sampled_points ||
                      compareMeta[compareLeftMethod]?.totalPoints ||
                      0
                    ).toLocaleString()}
                  </span>
                  <span className="text-slate-500 ml-2">RET: </span>
                  <span className="text-emerald-400 font-bold tabular-nums">
                    {(
                      frameMetrics?.retention?.methods?.[compareLeftMethod]?.retention_pct ||
                      (compareLeftMethod === 'original' ? 100 : 20)
                    ).toFixed(1)}
                    %
                  </span>
                </div>
              </div>

              <ThreePointCloudViewer
                pointsData={comparePoints[compareLeftMethod]}
                groundTruthObjects={groundTruthObjects}
                colorMode={colorMode}
                pointSize={pointSize}
                showRangeRings={true}
                showGroundTruth={showGroundTruth}
                methodId={compareLeftMethod}
                renderMeta={compareMeta[compareLeftMethod]}
                syncCameraState={sharedCameraState}
                onCameraChange={handleCameraChange}
                cameraPreset={cameraPreset}
              />
            </div>

            {/* Right Viewport */}
            <div className="relative h-full bg-[#0a0c10] rounded overflow-hidden border border-[#1f242e]">
              {/* Method Selector Header Bar */}
              <div className="absolute top-3 left-3 right-3 z-20 flex items-center justify-between pointer-events-auto">
                <select
                  value={compareRightMethod}
                  onChange={(e) => {
                    setCompareRightMethod(e.target.value);
                    onSelectMethod(e.target.value);
                  }}
                  className="bg-[#0a0c10]/90 backdrop-blur-md text-xs font-medium text-slate-200 px-2.5 py-1.5 rounded border border-[#232b38] focus:outline-none focus:border-sky-500 shadow-xl"
                >
                  {algorithms.map((algo) => (
                    <option key={algo.id} value={algo.id}>
                      {algo.name} ({algo.target_retention_pct}%)
                    </option>
                  ))}
                </select>

                <div className="bg-[#0a0c10]/90 backdrop-blur-md px-2.5 py-1 rounded border border-[#232b38] font-mono text-[10px] text-slate-300 shadow-xl">
                  <span className="text-slate-500">POINTS: </span>
                  <span className="text-sky-400 font-bold tabular-nums">
                    {(frameMetrics?.retention?.methods?.[compareRightMethod]?.sampled_points ||
                      compareMeta[compareRightMethod]?.totalPoints ||
                      0
                    ).toLocaleString()}
                  </span>
                  <span className="text-slate-500 ml-2">RET: </span>
                  <span className="text-emerald-400 font-bold tabular-nums">
                    {(
                      frameMetrics?.retention?.methods?.[compareRightMethod]?.retention_pct ||
                      (compareRightMethod === 'original' ? 100 : 20)
                    ).toFixed(1)}
                    %
                  </span>
                </div>
              </div>

              <ThreePointCloudViewer
                pointsData={comparePoints[compareRightMethod]}
                groundTruthObjects={groundTruthObjects}
                colorMode={colorMode}
                pointSize={pointSize}
                showRangeRings={true}
                showGroundTruth={showGroundTruth}
                methodId={compareRightMethod}
                renderMeta={compareMeta[compareRightMethod]}
                syncCameraState={sharedCameraState}
                onCameraChange={handleCameraChange}
                cameraPreset={cameraPreset}
              />
            </div>
          </div>
        )}

        {/* 3. Floating Quick Demonstration Controls HUD (Top-Right) */}
        <div className="absolute top-4 right-4 z-20 flex flex-col space-y-2 pointer-events-auto">
          {/* Algorithm Quick Buttons (in Single View) */}
          {viewMode === 'single' && (
            <div className="bg-[#0a0c10]/90 backdrop-blur-md border border-[#1f242e] p-1 rounded shadow-2xl flex items-center space-x-1">
              {algorithms.map((algo, i) => {
                const isSelected = selectedMethod === algo.id;
                return (
                  <button
                    key={algo.id}
                    onClick={() => onSelectMethod(algo.id)}
                    title={`${algo.name} [HotKey ${i + 1}]`}
                    className={`flex items-center space-x-1 px-2.5 py-1 rounded text-xs font-medium transition-all ${
                      isSelected
                        ? 'bg-[#181f2a] text-white shadow border border-[#2b3342]'
                        : 'text-slate-400 hover:text-slate-200 hover:bg-[#12161f]'
                    }`}
                  >
                    <span
                      className="w-2 h-2 rounded-full"
                      style={{ backgroundColor: algo.color }}
                    />
                    <span>{algo.name}</span>
                    <span className="text-[9px] text-slate-500 font-mono ml-0.5">[{i + 1}]</span>
                  </button>
                );
              })}
            </div>
          )}

          {/* Quick Visualization Toggles */}
          <div className="bg-[#0a0c10]/90 backdrop-blur-md border border-[#1f242e] p-1.5 rounded shadow-2xl flex items-center justify-end space-x-2 text-xs">
            {/* Ground Truth Boxes Toggle */}
            <button
              onClick={() => setShowGroundTruth(!showGroundTruth)}
              title="Toggle 3D Ground-Truth Bounding Boxes [B]"
              className={`flex items-center space-x-1 px-2 py-1 rounded transition-colors ${
                showGroundTruth
                  ? 'bg-emerald-950/60 text-emerald-300 border border-emerald-800/60'
                  : 'text-slate-400 hover:text-slate-200 hover:bg-[#141820]'
              }`}
            >
              <Boxes className="w-3.5 h-3.5" />
              <span>3D Boxes</span>
            </button>

            <span className="text-[#1f242e]">|</span>

            {/* Color Mode Stepper */}
            <div className="flex items-center space-x-1">
              <Palette className="w-3.5 h-3.5 text-slate-500" />
              <select
                value={colorMode}
                onChange={(e) => setColorMode(e.target.value)}
                className="bg-[#12161f] text-slate-300 text-[11px] px-2 py-1 rounded border border-[#222936] focus:outline-none"
              >
                <option value="elevation">Elevation (Z)</option>
                <option value="intensity">Intensity</option>
                <option value="distance">Distance (R)</option>
                <option value="solid">Method Solid</option>
              </select>
            </div>

            <span className="text-[#1f242e]">|</span>

            {/* Camera Presets */}
            <div className="flex items-center space-x-0.5">
              <button
                onClick={() => triggerCameraPreset('perspective')}
                title="Perspective Camera"
                className="px-1.5 py-0.5 text-[10px] font-mono text-slate-400 hover:text-white rounded hover:bg-[#151922]"
              >
                3D
              </button>
              <button
                onClick={() => triggerCameraPreset('top')}
                title="Bird's Eye View (BEV)"
                className="px-1.5 py-0.5 text-[10px] font-mono text-slate-400 hover:text-white rounded hover:bg-[#151922]"
              >
                TOP
              </button>
              <button
                onClick={() => triggerCameraPreset('front')}
                title="Ego Front View"
                className="px-1.5 py-0.5 text-[10px] font-mono text-slate-400 hover:text-white rounded hover:bg-[#151922]"
              >
                FRONT
              </button>
              <button
                onClick={() => triggerCameraPreset('side')}
                title="Lateral Side View"
                className="px-1.5 py-0.5 text-[10px] font-mono text-slate-400 hover:text-white rounded hover:bg-[#151922]"
              >
                SIDE
              </button>
            </div>
          </div>
        </div>

        {/* 4. Research Methodology & Annotation Overlay Card (Collapsible) */}
        {showAnnotations && (
          <div className="absolute top-16 right-4 z-30 w-96 bg-[#0a0c10]/95 backdrop-blur-xl border border-[#232b38] rounded shadow-2xl p-4 flex flex-col space-y-3 pointer-events-auto">
            {/* Header with Selector */}
            <div className="flex items-center justify-between border-b border-[#1f242e] pb-2">
              <div className="flex items-center space-x-2">
                <BookOpen className="w-4 h-4 text-amber-400" />
                <span className="text-xs font-semibold text-white tracking-wide uppercase">
                  Research Methodology
                </span>
              </div>
              <button
                onClick={() => setShowAnnotations(false)}
                className="text-slate-400 hover:text-white transition-colors"
              >
                <X className="w-4 h-4" />
              </button>
            </div>

            {/* Method Tabs */}
            <div className="flex items-center space-x-1 overflow-x-auto pb-1 scrollbar-none">
              {algorithms.map((algo) => (
                <button
                  key={algo.id}
                  onClick={() => setActiveAnnotationMethod(algo.id)}
                  className={`px-2 py-0.5 rounded text-[10px] font-mono shrink-0 transition-colors ${
                    activeAnnotationMethod === algo.id
                      ? 'bg-[#181f2a] text-white border border-[#2c3545]'
                      : 'text-slate-400 hover:text-slate-200'
                  }`}
                >
                  {algo.name}
                </button>
              ))}
            </div>

            {/* Annotation Content */}
            <div className="space-y-2.5 text-xs">
              <div>
                <div className="flex items-center justify-between">
                  <span className="font-semibold text-slate-100">{activeAnnotation.name}</span>
                  <span
                    className="text-[10px] font-mono px-1.5 py-0.5 rounded border"
                    style={{
                      borderColor: activeAnnotation.color + '55',
                      color: activeAnnotation.color,
                      backgroundColor: activeAnnotation.color + '15',
                    }}
                  >
                    {activeAnnotation.badge}
                  </span>
                </div>
                <p className="text-slate-400 text-[11px] mt-1 leading-relaxed">
                  {activeAnnotation.concept}
                </p>
              </div>

              {/* Mathematical Formulation */}
              <div className="bg-[#12161f] border border-[#1f242e] p-2 rounded">
                <span className="text-[10px] font-mono text-slate-500 uppercase block mb-0.5">
                  Mathematical Formulation
                </span>
                <code className="text-[11px] font-mono text-sky-300 block">
                  {activeAnnotation.formula}
                </code>
              </div>

              {/* Sampling Characteristics */}
              <div>
                <span className="text-[10px] font-mono text-slate-500 uppercase block mb-1">
                  Sampling Characteristics
                </span>
                <ul className="space-y-1">
                  {activeAnnotation.characteristics.map((point, idx) => (
                    <li key={idx} className="flex items-start space-x-1.5 text-[11px] text-slate-300">
                      <span className="text-sky-400 shrink-0">•</span>
                      <span>{point}</span>
                    </li>
                  ))}
                </ul>
              </div>

              {/* 3D Detection Insight */}
              <div className="border-t border-[#1f242e] pt-2">
                <span className="text-[10px] font-mono text-emerald-400 uppercase block mb-0.5">
                  PointPillars 3D Perception Impact
                </span>
                <p className="text-[11px] text-slate-400 leading-relaxed">
                  {activeAnnotation.detectorInsight}
                </p>
              </div>
            </div>
          </div>
        )}
      </div>

      {/* 5. Minimalist Media Playback Transport Footer */}
      <div className="shrink-0 z-30">
        <PlaybackToolbar {...playbackProps} />
      </div>
    </div>
  );
}
