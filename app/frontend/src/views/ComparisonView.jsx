import React, { useState, useEffect, useCallback } from 'react';
import ThreePointCloudViewer from '../components/ThreePointCloudViewer';
import PlaybackToolbar from '../components/PlaybackToolbar';
import { fetchFramePointsBinary, fetchGroundTruth, fetchFrameMetrics } from '../api';
import {
  Grid2X2,
  Columns,
  RefreshCw,
  Boxes,
  ChevronLeft,
  ChevronRight,
  Shuffle,
  RotateCcw,
  MonitorPlay,
} from 'lucide-react';

export default function ComparisonView({
  currentFrameId,
  algorithms,
  onPreviousFrame,
  onNextFrame,
  onRandomFrame,
  onEnterPresentation,
  playbackProps = {},
}) {
  const [layoutMode, setLayoutMode] = useState('2-split'); // '2-split' | '4-quad'
  const [selectedMethods, setSelectedMethods] = useState(['uniform', 'v6_alpha_0.25', 'hybrid_v5', 'original']);
  const [pointSize, setPointSize] = useState(1.8);
  const [colorMode, setColorMode] = useState('elevation');
  const [showGroundTruth, setShowGroundTruth] = useState(true);
  const [maxRenderPoints, setMaxRenderPoints] = useState(35000); // Efficient cap for multi-view

  // Global camera preset state to broadcast to all viewports
  const [globalCameraPreset, setGlobalCameraPreset] = useState(null);

  // Loaded data maps
  const [cloudsData, setCloudsData] = useState({});
  const [cloudsMeta, setCloudsMeta] = useState({});
  const [groundTruthObjects, setGroundTruthObjects] = useState([]);
  const [frameMetrics, setFrameMetrics] = useState(null);
  const [loading, setLoading] = useState(true);

  // Synchronized camera state shared across all viewports
  const [sharedCameraState, setSharedCameraState] = useState(null);

  const handleCameraChange = useCallback((cameraState) => {
    setSharedCameraState(cameraState);
  }, []);

  const handleGlobalCameraPreset = (preset) => {
    setGlobalCameraPreset(preset);
    // Reset after tick so subsequent identical clicks trigger re-orientation
    setTimeout(() => setGlobalCameraPreset(null), 50);
  };

  const activeMethodsToLoad = layoutMode === '2-split' ? selectedMethods.slice(0, 2) : selectedMethods.slice(0, 4);

  // Load clouds for all visible viewports
  useEffect(() => {
    let isCancelled = false;

    async function loadMultiViewData() {
      setLoading(true);
      try {
        const promises = activeMethodsToLoad.map((m) =>
          fetchFramePointsBinary(currentFrameId, m, maxRenderPoints)
        );

        const [gtRes, metricsRes, ...pointResponses] = await Promise.all([
          fetchGroundTruth(currentFrameId),
          fetchFrameMetrics(currentFrameId),
          ...promises,
        ]);

        if (isCancelled) return;

        const newClouds = {};
        const newMeta = {};
        pointResponses.forEach((res, i) => {
          const m = activeMethodsToLoad[i];
          newClouds[m] = res.points;
          newMeta[m] = res.metadata;
        });

        setCloudsData(newClouds);
        setCloudsMeta(newMeta);
        setGroundTruthObjects(gtRes.objects || []);
        setFrameMetrics(metricsRes);
      } catch (err) {
        console.error('Failed to load comparison view data:', err);
      } finally {
        if (!isCancelled) setLoading(false);
      }
    }

    loadMultiViewData();
    return () => {
      isCancelled = true;
    };
  }, [currentFrameId, layoutMode, selectedMethods[0], selectedMethods[1], selectedMethods[2], selectedMethods[3], maxRenderPoints]);

  // Sync loading state with parent playback controller
  useEffect(() => {
    playbackProps.onLoadingChange?.(loading);
  }, [loading, playbackProps.onLoadingChange]);

  const handleMethodChange = (viewportIdx, newMethod) => {
    setSelectedMethods((prev) => {
      const copy = [...prev];
      copy[viewportIdx] = newMethod;
      return copy;
    });
  };

  const origCount = frameMetrics?.retention?.original_points || 0;

  return (
    <div className="flex-1 flex flex-col h-[calc(100vh-3rem)] overflow-hidden bg-lab-bg">
      {/* Top Workstation Comparison Toolbar */}
      <div className="h-10 bg-lab-surface border-b border-lab-border px-3 flex items-center justify-between shrink-0 select-none gap-2">
        {/* Left: Layout & Frame Stepper */}
        <div className="flex items-center space-x-2 shrink-0">
          {/* Layout Toggle */}
          <div className="flex items-center space-x-0.5 bg-[#0a0c10] p-0.5 rounded border border-lab-border">
            <button
              onClick={() => setLayoutMode('2-split')}
              className={`flex items-center space-x-1.5 px-2 py-0.5 rounded text-[11px] transition-colors ${
                layoutMode === '2-split'
                  ? 'bg-[#1a202c] text-white font-medium border border-[#2b3342] shadow-subtle'
                  : 'text-lab-textMuted hover:text-slate-200'
              }`}
            >
              <Columns className="w-3.5 h-3.5 text-sky-400" />
              <span>Dual Split</span>
            </button>
            <button
              onClick={() => setLayoutMode('4-quad')}
              className={`flex items-center space-x-1.5 px-2 py-0.5 rounded text-[11px] transition-colors ${
                layoutMode === '4-quad'
                  ? 'bg-[#1a202c] text-white font-medium border border-[#2b3342] shadow-subtle'
                  : 'text-lab-textMuted hover:text-slate-200'
              }`}
            >
              <Grid2X2 className="w-3.5 h-3.5 text-sky-400" />
              <span>Quad Grid</span>
            </button>
          </div>

          {/* Quick Frame Stepper */}
          {onPreviousFrame && onNextFrame && (
            <div className="hidden sm:flex items-center bg-[#0a0c10] rounded border border-lab-border overflow-hidden">
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

        {/* Center: Global Camera Controls & Visual Settings */}
        <div className="flex items-center space-x-2.5">
          {/* Synchronized Global Camera Presets */}
          <div className="hidden md:flex items-center space-x-0.5 bg-[#0a0c10] p-0.5 rounded border border-lab-border text-[11px] font-mono">
            <span className="text-lab-textDim px-1.5 text-[10px] uppercase font-sans">Camera:</span>
            <button
              onClick={() => handleGlobalCameraPreset('perspective')}
              className="px-2 py-0.5 text-slate-300 hover:text-white hover:bg-[#1a202c] rounded transition-colors"
              title="3D Perspective View (All Viewports)"
            >
              3D
            </button>
            <button
              onClick={() => handleGlobalCameraPreset('top')}
              className="px-2 py-0.5 text-slate-300 hover:text-white hover:bg-[#1a202c] rounded transition-colors"
              title="Bird's Eye View (All Viewports)"
            >
              BEV
            </button>
            <button
              onClick={() => handleGlobalCameraPreset('front')}
              className="px-2 py-0.5 text-slate-300 hover:text-white hover:bg-[#1a202c] rounded transition-colors"
              title="Driver Front View (All Viewports)"
            >
              POV
            </button>
            <button
              onClick={() => handleGlobalCameraPreset('side')}
              className="px-2 py-0.5 text-slate-300 hover:text-white hover:bg-[#1a202c] rounded transition-colors"
              title="Profile Side View (All Viewports)"
            >
              SIDE
            </button>
          </div>

          {/* Color Mode */}
          <div className="flex items-center space-x-0.5 bg-[#0a0c10] p-0.5 rounded border border-lab-border text-xs">
            {['elevation', 'intensity', 'distance', 'solid'].map((mode) => (
              <button
                key={mode}
                onClick={() => setColorMode(mode)}
                className={`px-2 py-0.5 rounded uppercase text-[10px] font-medium transition-colors ${
                  colorMode === mode
                    ? 'bg-[#1a202c] text-sky-400 font-semibold border border-[#2b3342]'
                    : 'text-lab-textMuted hover:text-slate-200'
                }`}
              >
                {mode === 'elevation' ? 'Elev' : mode === 'intensity' ? 'Int' : mode === 'distance' ? 'Dist' : 'Solid'}
              </button>
            ))}
          </div>

          {/* Point Size */}
          <div className="hidden lg:flex items-center space-x-1.5 bg-[#0a0c10] px-2 py-0.5 rounded border border-lab-border text-xs font-mono">
            <span className="text-lab-textDim text-[10px]">PT:</span>
            <input
              type="range"
              min="1.0"
              max="4.5"
              step="0.3"
              value={pointSize}
              onChange={(e) => setPointSize(parseFloat(e.target.value))}
              className="w-14 accent-sky-500 h-1 bg-[#1a202c] rounded cursor-pointer"
            />
            <span className="text-slate-300 tabular-nums text-[11px]">{pointSize.toFixed(1)}</span>
          </div>

          {/* Ground Truth Toggle */}
          <button
            onClick={() => setShowGroundTruth(!showGroundTruth)}
            className={`flex items-center space-x-1.5 px-2 py-0.5 rounded border text-xs font-mono transition-colors ${
              showGroundTruth
                ? 'bg-emerald-950 text-emerald-400 border-emerald-800'
                : 'bg-[#0a0c10] text-lab-textDim border-lab-border'
            }`}
          >
            <Boxes className="w-3 h-3" />
            <span className="text-[11px]">3D BOXES</span>
          </button>
        </div>

        {/* Right: Camera Sync Status & Present Button */}
        <div className="flex items-center space-x-2 shrink-0">
          <div className="flex items-center space-x-1.5 text-xs font-mono text-lab-textMuted">
            <span className="w-1.5 h-1.5 rounded-full bg-emerald-500" />
            <span className="text-[11px] hidden sm:inline">CAMERAS SYNCED</span>
          </div>

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
        </div>
      </div>

      {/* Synchronized Multi-Sensor Viewport Grid */}
      <div className="flex-1 relative overflow-hidden p-1 bg-lab-bg">
        {loading && (
          <div className="absolute inset-0 bg-lab-bg/75 backdrop-blur-sm flex items-center justify-center z-30 pointer-events-none">
            <div className="flex items-center space-x-2 px-3 py-1.5 bg-lab-surface rounded border border-lab-border text-xs font-mono text-slate-300">
              <RefreshCw className="w-3.5 h-3.5 text-sky-400 animate-spin" />
              <span>SYNCING VIEWPORTS...</span>
            </div>
          </div>
        )}

        <div
          className={`w-full h-full gap-1 grid ${
            layoutMode === '2-split' ? 'grid-cols-2' : 'grid-cols-2 grid-rows-2'
          }`}
        >
          {activeMethodsToLoad.map((method, idx) => {
            const algoMeta = algorithms.find((a) => a.id === method) || { name: method };
            const methodMetrics = frameMetrics?.retention?.methods?.[method];
            const sampledCount = methodMetrics?.sampled_points || cloudsMeta[method]?.totalPoints || 0;
            const retentionPct = methodMetrics?.retention_pct || (origCount > 0 ? (sampledCount / origCount) * 100 : 0);

            return (
              <div
                key={idx}
                className="relative rounded border border-lab-border bg-lab-surface overflow-hidden flex flex-col"
              >
                {/* Viewport Header Bar (Equal Importance & Clear Technical Info) */}
                <div className="h-8 px-2.5 bg-[#0e1116] border-b border-lab-border flex items-center justify-between z-10 shrink-0 select-none">
                  {/* Method Selector */}
                  <div className="flex items-center space-x-2">
                    <span
                      className="w-2 h-2 rounded-full shrink-0"
                      style={{ backgroundColor: algoMeta.color || '#38bdf8' }}
                    />
                    <select
                      value={method}
                      onChange={(e) => handleMethodChange(idx, e.target.value)}
                      className="bg-[#141820] border border-lab-border rounded px-2 py-0.5 text-xs font-medium text-slate-200 focus:outline-none focus:border-sky-500 cursor-pointer"
                    >
                      {algorithms.map((a) => (
                        <option key={a.id} value={a.id}>
                          {a.name}
                        </option>
                      ))}
                    </select>
                    <span className="text-[10px] font-mono text-lab-textDim hidden md:inline">
                      (0.8m voxel)
                    </span>
                  </div>

                  {/* Technical Information Readout */}
                  <div className="flex items-center space-x-2 text-[11px] font-mono">
                    <span className="text-lab-textDim">
                      PTS: <strong className="text-slate-200 tabular-nums">{sampledCount.toLocaleString()}</strong>
                      {origCount > 0 && (
                        <span className="text-lab-textDim ml-1">/ {origCount.toLocaleString()}</span>
                      )}
                    </span>
                    <span className="text-lab-border">|</span>
                    <span
                      className={`font-semibold tabular-nums ${
                        method === 'original'
                          ? 'text-slate-300'
                          : Math.abs(retentionPct - 20.0) < 0.05
                          ? 'text-emerald-400'
                          : 'text-amber-400'
                      }`}
                    >
                      {Number(retentionPct).toFixed(2)}%
                    </span>
                  </div>
                </div>

                {/* 3D Canvas */}
                <div className="flex-1 relative">
                  <ThreePointCloudViewer
                    pointsData={cloudsData[method]}
                    groundTruthObjects={groundTruthObjects}
                    colorMode={colorMode}
                    pointSize={pointSize}
                    showRangeRings={true}
                    showGroundTruth={showGroundTruth}
                    syncCameraState={sharedCameraState}
                    onCameraChange={handleCameraChange}
                    methodId={method}
                    renderMeta={cloudsMeta[method]}
                    cameraPreset={globalCameraPreset}
                  />
                </div>
              </div>
            );
          })}
        </div>
      </div>

      {/* Shared Playback Controller for All Comparison Panels */}
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
  );
}
