import React, { useState, useEffect } from 'react';
import ThreePointCloudViewer from '../components/ThreePointCloudViewer';
import PlaybackToolbar from '../components/PlaybackToolbar';
import {
  fetchFramePointsBinary,
  fetchGroundTruth,
  fetchPredictions,
  fetchFrameMetrics,
} from '../api';
import {
  Sliders,
  Palette,
  Eye,
  EyeOff,
  Boxes,
  CircleGauge,
  Car,
  User,
  Bike,
  RefreshCw,
  Target,
  ShieldAlert,
  ChevronLeft,
  ChevronRight,
  Shuffle,
  PanelRightClose,
  PanelRightOpen,
  Info,
  Layers,
  MonitorPlay,
} from 'lucide-react';

export default function ExplorerView({
  currentFrameId,
  algorithms,
  selectedMethod,
  onSelectMethod,
  onPreviousFrame,
  onNextFrame,
  onRandomFrame,
  onEnterPresentation,
  playbackProps = {},
}) {
  const [pointsData, setPointsData] = useState(null);
  const [renderMeta, setRenderMeta] = useState(null);
  const [groundTruthObjects, setGroundTruthObjects] = useState([]);
  const [predictionObjects, setPredictionObjects] = useState([]);
  const [frameMetrics, setFrameMetrics] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  // Inspector panel visibility
  const [isInspectorOpen, setIsInspectorOpen] = useState(true);

  // Explorer visual settings
  const [colorMode, setColorMode] = useState('elevation'); // 'elevation' | 'intensity' | 'distance' | 'solid'
  const [pointSize, setPointSize] = useState(1.8);
  const [showGroundTruth, setShowGroundTruth] = useState(true);
  const [showPredictions, setShowPredictions] = useState(false);
  const [showRangeRings, setShowRangeRings] = useState(true);
  const [maxRenderPoints, setMaxRenderPoints] = useState(60000);
  const [visibleClasses, setVisibleClasses] = useState({
    Car: true,
    Pedestrian: true,
    Cyclist: true,
  });

  // Load points, ground truth, and metrics whenever frame, method, or render cap changes
  useEffect(() => {
    let isCancelled = false;

    async function loadData() {
      setLoading(true);
      setError(null);
      try {
        const [pointsRes, gtRes, metricsRes] = await Promise.all([
          fetchFramePointsBinary(currentFrameId, selectedMethod, maxRenderPoints),
          fetchGroundTruth(currentFrameId, null, selectedMethod),
          fetchFrameMetrics(currentFrameId),
        ]);

        if (isCancelled) return;

        setPointsData(pointsRes.points);
        setRenderMeta(pointsRes.metadata);
        setGroundTruthObjects(gtRes.objects || []);
        setFrameMetrics(metricsRes);

        // Load predictions if toggled
        if (showPredictions) {
          const predRes = await fetchPredictions(currentFrameId, selectedMethod, 0.25);
          if (!isCancelled) {
            setPredictionObjects(predRes.predictions || []);
          }
        }
      } catch (err) {
        if (!isCancelled) {
          setError(err.message || 'Failed to load point cloud frame');
        }
      } finally {
        if (!isCancelled) {
          setLoading(false);
        }
      }
    }

    loadData();
    return () => {
      isCancelled = true;
    };
  }, [currentFrameId, selectedMethod, maxRenderPoints, showPredictions]);

  // Sync loading state with parent playback controller
  useEffect(() => {
    playbackProps.onLoadingChange?.(loading);
  }, [loading, playbackProps.onLoadingChange]);

  // Active method metadata
  const activeMethodMeta = algorithms.find((a) => a.id === selectedMethod) || {
    name: selectedMethod,
    description: '',
    target_retention_pct: 20.0,
  };

  const currentMethodStats = frameMetrics?.retention?.methods?.[selectedMethod] || {
    sampled_points: renderMeta?.totalPoints || 0,
    retention_pct: selectedMethod === 'original' ? 100.0 : 20.0,
  };

  const origPointCount = frameMetrics?.retention?.original_points || renderMeta?.totalPoints || 0;

  const toggleClass = (cls) => {
    setVisibleClasses((prev) => ({ ...prev, [cls]: !prev[cls] }));
  };

  return (
    <div className="flex-1 flex flex-col h-[calc(100vh-3rem)] overflow-hidden bg-lab-bg">
      {/* Compact Top Workstation Ribbon */}
      <div className="h-10 bg-lab-surface border-b border-lab-border px-3 flex items-center justify-between shrink-0 select-none gap-2">
        {/* Left: Compact Algorithm Selector */}
        <div className="flex items-center space-x-2 shrink-0">
          <div className="flex items-center space-x-0.5 bg-[#0a0c10] p-0.5 rounded border border-lab-border">
            {algorithms.map((algo) => {
              const isSelected = selectedMethod === algo.id;
              return (
                <button
                  key={algo.id}
                  onClick={() => onSelectMethod(algo.id)}
                  className={`flex items-center space-x-1.5 px-2 py-0.5 rounded text-[11px] transition-colors ${
                    isSelected
                      ? 'bg-[#1a202c] text-white font-medium border border-[#2b3342] shadow-subtle'
                      : 'text-lab-textMuted hover:text-slate-200 hover:bg-[#12161f]'
                  }`}
                >
                  <span
                    className="w-1.5 h-1.5 rounded-full"
                    style={{ backgroundColor: algo.color || '#38bdf8' }}
                  />
                  <span>{algo.name}</span>
                </button>
              );
            })}
          </div>

          {/* Quick Frame Stepper */}
          {onPreviousFrame && onNextFrame && (
            <div className="hidden md:flex items-center bg-[#0a0c10] rounded border border-lab-border overflow-hidden">
              <button
                onClick={onPreviousFrame}
                title="Previous Frame"
                className="px-1.5 py-0.5 text-lab-textMuted hover:text-white hover:bg-[#1a202c] transition-colors border-r border-lab-border"
              >
                <ChevronLeft className="w-3 h-3" />
              </button>
              <span className="px-2 font-mono text-[11px] text-sky-400 font-semibold tabular-nums">
                {currentFrameId}
              </span>
              <button
                onClick={onNextFrame}
                title="Next Frame"
                className="px-1.5 py-0.5 text-lab-textMuted hover:text-white hover:bg-[#1a202c] transition-colors border-l border-lab-border"
              >
                <ChevronRight className="w-3 h-3" />
              </button>
              {onRandomFrame && (
                <button
                  onClick={onRandomFrame}
                  title="Random Frame"
                  className="px-1.5 py-0.5 text-lab-textMuted hover:text-white hover:bg-[#1a202c] transition-colors border-l border-lab-border"
                >
                  <Shuffle className="w-3 h-3" />
                </button>
              )}
            </div>
          )}
        </div>

        {/* Center: Technical Information Strip */}
        <div className="hidden lg:flex items-center space-x-3 text-xs font-mono text-lab-textMuted overflow-hidden">
          <div className="flex items-center space-x-1">
            <span className="text-lab-textDim text-[11px]">ORIGINAL:</span>
            <span className="text-slate-200 font-medium tabular-nums">
              {origPointCount.toLocaleString()}
            </span>
          </div>

          <span className="text-lab-border">|</span>

          <div className="flex items-center space-x-1">
            <span className="text-lab-textDim text-[11px]">SAMPLED:</span>
            <span className="text-sky-400 font-medium tabular-nums">
              {(currentMethodStats.sampled_points || 0).toLocaleString()}
            </span>
          </div>

          <span className="text-lab-border">|</span>

          <div className="flex items-center space-x-1">
            <span className="text-lab-textDim text-[11px]">RETENTION:</span>
            <span
              className={`font-semibold tabular-nums ${
                selectedMethod === 'original'
                  ? 'text-slate-300'
                  : Math.abs(currentMethodStats.retention_pct - 20.0) < 0.05
                  ? 'text-emerald-400'
                  : 'text-amber-400'
              }`}
            >
              {Number(currentMethodStats.retention_pct || 0).toFixed(2)}%
            </span>
          </div>

          <span className="text-lab-border">|</span>

          <div className="flex items-center space-x-1">
            <span className="text-lab-textDim text-[11px]">VOXEL:</span>
            <span className="text-slate-200 font-medium">0.8m</span>
          </div>

          <span className="text-lab-border">|</span>

          <div className="flex items-center space-x-1">
            <span className="text-lab-textDim text-[11px]">OBJECTS:</span>
            <span className="text-emerald-400 font-medium tabular-nums">
              {groundTruthObjects.length} GT
            </span>
          </div>

          {renderMeta?.isDownsampled && (
            <>
              <span className="text-lab-border">|</span>
              <div className="flex items-center space-x-1 text-amber-400 text-[11px]">
                <CircleGauge className="w-3 h-3" />
                <span>CAP: {renderMeta.renderedPoints.toLocaleString()}</span>
              </div>
            </>
          )}
        </div>

        {/* Right: Presentation & Inspector Controls */}
        <div className="flex items-center space-x-2 shrink-0">
          {onEnterPresentation && (
            <button
              onClick={onEnterPresentation}
              title="Enter Presentation Mode [P]"
              className="flex items-center space-x-1 px-2 py-1 rounded text-xs bg-[#0a0c10] text-sky-400 hover:text-white hover:bg-sky-950/40 border border-lab-border hover:border-sky-800/60 transition-colors"
            >
              <MonitorPlay className="w-3.5 h-3.5" />
              <span className="text-[11px] hidden sm:inline">Present</span>
            </button>
          )}

          <button
            onClick={() => setIsInspectorOpen(!isInspectorOpen)}
            title={isInspectorOpen ? 'Collapse Inspector (Maximize Viewport)' : 'Open Inspector'}
            className={`flex items-center space-x-1.5 px-2 py-1 rounded text-xs transition-colors ${
              isInspectorOpen
                ? 'bg-[#1a202c] text-white border border-[#2b3342]'
                : 'bg-[#0a0c10] text-lab-textMuted hover:text-white border border-lab-border'
            }`}
          >
            {isInspectorOpen ? (
              <>
                <PanelRightClose className="w-3.5 h-3.5 text-sky-400" />
                <span className="text-[11px] hidden sm:inline">Controls</span>
              </>
            ) : (
              <>
                <PanelRightOpen className="w-3.5 h-3.5 text-slate-400" />
                <span className="text-[11px] hidden sm:inline">Controls</span>
              </>
            )}
          </button>
        </div>
      </div>

      {/* Main Workstation Body: Dominant 3D Viewport + Collapsible Inspector */}
      <div className="flex-1 relative flex overflow-hidden">
        {/* Dominant 3D Canvas Viewport + Bottom Playback Toolbar */}
        <div className="flex-1 h-full relative flex flex-col overflow-hidden">
          {/* 3D Canvas Container */}
          <div className="flex-1 relative overflow-hidden">
            {loading && (
              <div className="absolute inset-0 bg-lab-bg/80 backdrop-blur-sm flex items-center justify-center z-20">
                <div className="flex items-center space-x-2.5 px-4 py-2 rounded bg-lab-surface border border-lab-border text-xs font-mono text-slate-300">
                  <RefreshCw className="w-4 h-4 text-sky-400 animate-spin" />
                  <span>STREAMING POINT CLOUD [{selectedMethod.toUpperCase()}]</span>
                </div>
              </div>
            )}

            {error && (
              <div className="absolute top-3 left-3 right-3 bg-rose-950/90 border border-rose-800 text-rose-200 text-xs font-mono p-2.5 rounded z-20 flex items-center space-x-2">
                <ShieldAlert className="w-4 h-4 text-rose-400 shrink-0" />
                <span>{error}</span>
              </div>
            )}

            <ThreePointCloudViewer
              pointsData={pointsData}
              groundTruthObjects={groundTruthObjects}
              predictionObjects={predictionObjects}
              colorMode={colorMode}
              pointSize={pointSize}
              showRangeRings={showRangeRings}
              showGroundTruth={showGroundTruth}
              showPredictions={showPredictions}
              visibleClasses={visibleClasses}
              methodId={selectedMethod}
              renderMeta={renderMeta}
            />

            {/* Understated Method Context Pill */}
            <div className="absolute top-2.5 left-2.5 max-w-sm bg-lab-surface/90 backdrop-blur-sm p-2 rounded border border-lab-border z-10 pointer-events-none select-none">
              <div className="flex items-center space-x-2">
                <span
                  className="w-2 h-2 rounded-full"
                  style={{ backgroundColor: activeMethodMeta.color || '#38bdf8' }}
                />
                <span className="text-xs font-semibold text-slate-200">
                  {activeMethodMeta.name}
                </span>
                <span className="text-[10px] font-mono text-lab-textMuted bg-[#0a0c10] px-1.5 py-0.2 rounded border border-lab-border">
                  0.8m voxel &bull; ~20% budget
                </span>
              </div>
              {activeMethodMeta.description && (
                <p className="text-[11px] text-lab-textMuted mt-1 leading-snug">
                  {activeMethodMeta.description}
                </p>
              )}
            </div>
          </div>

          {/* Docked Media Transport Playback Toolbar */}
          <PlaybackToolbar
            isPlaying={playbackProps.isPlaying}
            onTogglePlay={playbackProps.onTogglePlay}
            onPreviousFrame={playbackProps.onPreviousFrame || onPreviousFrame}
            onNextFrame={playbackProps.onNextFrame || onNextFrame}
            onRestart={playbackProps.onRestart}
            playbackSpeed={playbackProps.playbackSpeed}
            onSpeedChange={playbackProps.onSpeedChange}
            isLooping={playbackProps.isLooping}
            onToggleLoop={playbackProps.onToggleLoop}
            currentFrameId={currentFrameId}
            currentIndex={playbackProps.currentIndex}
            totalFrames={playbackProps.totalFrames}
            onSeek={playbackProps.onSeek}
            isLoading={loading}
          />
        </div>

        {/* Compact Right Inspector Panel (Collapsible) */}
        {isInspectorOpen && (
          <aside className="w-72 border-l border-lab-border bg-lab-surface flex flex-col h-full z-10 shrink-0 select-none">
            {/* Inspector Header */}
            <div className="h-9 px-3 border-b border-lab-border flex items-center justify-between shrink-0">
              <div className="flex items-center space-x-1.5">
                <Sliders className="w-3.5 h-3.5 text-sky-400" />
                <span className="text-xs font-semibold text-lab-textBright tracking-tight">
                  Visualization Settings
                </span>
              </div>
              <button
                onClick={() => setIsInspectorOpen(false)}
                className="text-lab-textDim hover:text-white transition-colors"
                title="Collapse Inspector"
              >
                <PanelRightClose className="w-3.5 h-3.5" />
              </button>
            </div>

            <div className="flex-1 overflow-y-auto p-3 space-y-4 text-xs">
              {/* Color Mode Selection with Visual Indicators */}
              <div>
                <label className="text-lab-textDim uppercase text-[10px] font-semibold tracking-wider block mb-1.5 flex items-center justify-between">
                  <span>Color Mode</span>
                  <span className="font-mono text-sky-400 text-[10px] uppercase">{colorMode}</span>
                </label>
                <div className="grid grid-cols-2 gap-1 bg-[#0a0c10] p-1 rounded border border-lab-border">
                  {[
                    { id: 'elevation', label: 'Elevation (Z)', desc: 'Height -3m to +2m' },
                    { id: 'intensity', label: 'Intensity (I)', desc: 'Reflectance 0 to 1' },
                    { id: 'distance', label: 'Distance Range', desc: 'Radial 0m to 70m' },
                    { id: 'solid', label: 'Method Color', desc: 'Algorithm solid' },
                  ].map((mode) => (
                    <button
                      key={mode.id}
                      onClick={() => setColorMode(mode.id)}
                      className={`py-1.5 px-2 rounded text-left transition-colors flex flex-col ${
                        colorMode === mode.id
                          ? 'bg-[#1a202c] text-sky-400 font-medium border border-[#2b3342]'
                          : 'text-lab-textMuted hover:text-slate-200'
                      }`}
                    >
                      <span className="text-[11px] font-medium leading-none">{mode.label}</span>
                      <span className="text-[9px] text-lab-textDim mt-0.5 leading-none">{mode.desc}</span>
                    </button>
                  ))}
                </div>
              </div>

              {/* Point Size Slider */}
              <div>
                <div className="flex items-center justify-between text-lab-textDim uppercase text-[10px] font-semibold tracking-wider mb-1">
                  <span>Point Size</span>
                  <span className="font-mono text-slate-300 tabular-nums">{pointSize.toFixed(1)}px</span>
                </div>
                <input
                  type="range"
                  min="0.8"
                  max="5.0"
                  step="0.2"
                  value={pointSize}
                  onChange={(e) => setPointSize(parseFloat(e.target.value))}
                  className="w-full accent-sky-500 h-1 bg-[#1a202c] rounded cursor-pointer"
                />
              </div>

              {/* GPU Point Budget Cap */}
              <div>
                <div className="flex items-center justify-between text-lab-textDim uppercase text-[10px] font-semibold tracking-wider mb-1">
                  <span>GPU Render Budget</span>
                  <span className="font-mono text-amber-400">
                    {maxRenderPoints ? `${(maxRenderPoints / 1000).toFixed(0)}k` : 'Full'}
                  </span>
                </div>
                <select
                  value={maxRenderPoints || 0}
                  onChange={(e) => setMaxRenderPoints(parseInt(e.target.value, 10) || null)}
                  className="w-full bg-[#0a0c10] border border-lab-border rounded p-1.5 text-xs text-slate-200 focus:outline-none focus:border-sky-500 font-mono"
                >
                  <option value={30000}>30,000 Points (Fast / Low-End)</option>
                  <option value={60000}>60,000 Points (Recommended)</option>
                  <option value={100000}>100,000 Points (High Fidelity)</option>
                  <option value={0}>Full Cloud (Uncapped)</option>
                </select>
              </div>

              {/* Polar Range Rings Toggle */}
              <div className="flex items-center justify-between pt-1">
                <span className="text-lab-textDim uppercase text-[10px] font-semibold tracking-wider">
                  Range Rings (10m, 20m, 40m, 60m)
                </span>
                <button
                  onClick={() => setShowRangeRings(!showRangeRings)}
                  className={`text-[10px] font-mono px-2 py-0.5 rounded transition-colors ${
                    showRangeRings
                      ? 'bg-sky-950 text-sky-400 border border-sky-800'
                      : 'bg-[#151921] text-lab-textDim border border-lab-border'
                  }`}
                >
                  {showRangeRings ? 'ON' : 'OFF'}
                </button>
              </div>

              <div className="h-px bg-lab-border" />

              {/* Ground Truth 3D Bounding Boxes */}
              <div>
                <div className="flex items-center justify-between mb-1.5">
                  <label className="text-lab-textDim uppercase text-[10px] font-semibold tracking-wider flex items-center space-x-1">
                    <Boxes className="w-3.5 h-3.5 text-emerald-400" />
                    <span>3D Ground Truth</span>
                  </label>
                  <button
                    onClick={() => setShowGroundTruth(!showGroundTruth)}
                    className={`text-[10px] font-mono px-2 py-0.5 rounded transition-colors ${
                      showGroundTruth
                        ? 'bg-emerald-950 text-emerald-400 border border-emerald-800'
                        : 'bg-[#151921] text-lab-textDim border border-lab-border'
                    }`}
                  >
                    {showGroundTruth ? 'ON' : 'OFF'}
                  </button>
                </div>

                {/* Class Toggles */}
                <div className="space-y-1 bg-[#0a0c10] p-1.5 rounded border border-lab-border">
                  {[
                    { id: 'Car', label: 'Cars', icon: Car, color: 'text-emerald-400', count: groundTruthObjects.filter(o => o.class === 'Car').length },
                    { id: 'Pedestrian', label: 'Pedestrians', icon: User, color: 'text-amber-400', count: groundTruthObjects.filter(o => o.class === 'Pedestrian').length },
                    { id: 'Cyclist', label: 'Cyclists', icon: Bike, color: 'text-sky-400', count: groundTruthObjects.filter(o => o.class === 'Cyclist').length },
                  ].map((item) => {
                    const Icon = item.icon;
                    const isVisible = visibleClasses[item.id];
                    return (
                      <div
                        key={item.id}
                        onClick={() => toggleClass(item.id)}
                        className="flex items-center justify-between p-1 rounded hover:bg-[#151921] cursor-pointer"
                      >
                        <div className="flex items-center space-x-1.5">
                          <Icon className={`w-3.5 h-3.5 ${item.color}`} />
                          <span className="text-slate-300 font-medium text-[11px]">{item.label}</span>
                        </div>
                        <div className="flex items-center space-x-2">
                          <span className="text-[10px] font-mono text-lab-textDim tabular-nums">
                            {item.count} in frame
                          </span>
                          {isVisible ? (
                            <Eye className="w-3 h-3 text-slate-300" />
                          ) : (
                            <EyeOff className="w-3 h-3 text-lab-textDim" />
                          )}
                        </div>
                      </div>
                    );
                  })}
                </div>
              </div>

              {/* PointPillars Detector 3D Predictions Overlay */}
              <div>
                <div className="flex items-center justify-between mb-1">
                  <label className="text-lab-textDim uppercase text-[10px] font-semibold tracking-wider flex items-center space-x-1">
                    <Target className="w-3.5 h-3.5 text-indigo-400" />
                    <span>PointPillars Predictions</span>
                  </label>
                  <button
                    onClick={() => setShowPredictions(!showPredictions)}
                    className={`text-[10px] font-mono px-2 py-0.5 rounded transition-colors ${
                      showPredictions
                        ? 'bg-indigo-950 text-indigo-300 border border-indigo-800'
                        : 'bg-[#151921] text-lab-textDim border border-lab-border'
                    }`}
                  >
                    {showPredictions ? 'ON' : 'OFF'}
                  </button>
                </div>
                <p className="text-[10px] text-lab-textDim leading-tight">
                  Oriented 3D bounding boxes predicted by the frozen PointPillars network.
                </p>
              </div>
            </div>

            {/* Active Frame Objects Breakdown Drawer */}
            <div className="border-t border-lab-border p-2.5 bg-[#0c0e13] shrink-0">
              <div className="flex items-center justify-between mb-1">
                <span className="text-[10px] font-semibold text-lab-textDim uppercase tracking-wider">
                  Frame Objects ({groundTruthObjects.length})
                </span>
                <span className="text-[10px] font-mono text-sky-400"># {currentFrameId}</span>
              </div>
              <div className="max-h-32 overflow-y-auto space-y-1 pr-1">
                {groundTruthObjects.length === 0 ? (
                  <p className="text-[11px] text-lab-textDim italic">No annotated objects in this frame.</p>
                ) : (
                  groundTruthObjects.map((obj, i) => (
                    <div
                      key={i}
                      className="p-1 rounded bg-[#101318] border border-lab-border text-[10px] font-mono flex items-center justify-between"
                    >
                      <div>
                        <span
                          className={`font-semibold ${
                            obj.class === 'Car'
                              ? 'text-emerald-400'
                              : obj.class === 'Pedestrian'
                              ? 'text-amber-400'
                              : 'text-sky-400'
                          }`}
                        >
                          {obj.class}
                        </span>
                        <span className="text-lab-textDim ml-1 tabular-nums">{obj.distance_z_m}m</span>
                      </div>
                      {obj.contained_points !== null && (
                        <span className="text-slate-300 bg-[#161a22] px-1 py-0.2 rounded text-[10px] tabular-nums">
                          {obj.contained_points} pts
                        </span>
                      )}
                    </div>
                  ))
                )}
              </div>
            </div>
          </aside>
        )}
      </div>
    </div>
  );
}
