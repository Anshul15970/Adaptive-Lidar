import React, { useState, useEffect } from 'react';
import {
  Play,
  Pause,
  SkipBack,
  SkipForward,
  RotateCcw,
  Repeat,
  RefreshCw,
  Gauge,
} from 'lucide-react';

const SPEED_OPTIONS = [0.5, 1, 2, 4, 8];

export default function PlaybackToolbar({
  isPlaying,
  onTogglePlay,
  onPreviousFrame,
  onNextFrame,
  onRestart,
  playbackSpeed = 1,
  onSpeedChange,
  isLooping = true,
  onToggleLoop,
  currentFrameId,
  currentIndex = 0,
  totalFrames = 3769,
  onSeek,
  isLoading = false,
  className = '',
}) {
  const [isScrubbing, setIsScrubbing] = useState(false);
  const [scrubValue, setScrubValue] = useState(currentIndex);

  // Sync internal slider value when currentIndex changes from outside
  useEffect(() => {
    if (!isScrubbing) {
      setScrubValue(currentIndex);
    }
  }, [currentIndex, isScrubbing]);

  const handleSliderChange = (e) => {
    const val = parseInt(e.target.value, 10);
    setScrubValue(val);
  };

  const handleSliderMouseDown = () => {
    setIsScrubbing(true);
  };

  const handleSliderMouseUp = (e) => {
    setIsScrubbing(false);
    const val = parseInt(e.target.value, 10);
    onSeek(val);
  };

  const progressPct = totalFrames > 1 ? ((scrubValue / (totalFrames - 1)) * 100).toFixed(1) : 0;

  return (
    <div
      className={`h-11 bg-lab-surface/95 backdrop-blur-md border-t border-lab-border px-3.5 flex items-center justify-between select-none gap-3 text-xs z-20 ${className}`}
    >
      {/* Transport Controls (Left) */}
      <div className="flex items-center space-x-1.5 shrink-0">
        {/* Restart / Beginning */}
        <button
          onClick={onRestart}
          title="Restart Playback (First Validation Frame)"
          className="p-1.5 text-lab-textMuted hover:text-white hover:bg-[#1a202c] rounded transition-colors"
        >
          <RotateCcw className="w-3.5 h-3.5" />
        </button>

        {/* Previous Frame */}
        <button
          onClick={onPreviousFrame}
          title="Previous Frame (Left Arrow)"
          className="p-1.5 text-lab-textMuted hover:text-white hover:bg-[#1a202c] rounded transition-colors"
        >
          <SkipBack className="w-3.5 h-3.5" />
        </button>

        {/* Play / Pause Primary Button */}
        <button
          onClick={onTogglePlay}
          title={isPlaying ? 'Pause Playback (Space)' : 'Play Validation Frames (Space)'}
          className={`px-3 py-1.5 rounded font-medium flex items-center space-x-1.5 transition-colors shadow-subtle ${
            isPlaying
              ? 'bg-amber-600 hover:bg-amber-500 text-white'
              : 'bg-sky-600 hover:bg-sky-500 text-white'
          }`}
        >
          {isPlaying ? (
            <>
              <Pause className="w-3.5 h-3.5 fill-current" />
              <span className="text-[11px] font-semibold tracking-wide">PAUSE</span>
            </>
          ) : (
            <>
              <Play className="w-3.5 h-3.5 fill-current ml-0.5" />
              <span className="text-[11px] font-semibold tracking-wide">PLAY</span>
            </>
          )}
        </button>

        {/* Next Frame */}
        <button
          onClick={onNextFrame}
          title="Next Frame (Right Arrow)"
          className="p-1.5 text-lab-textMuted hover:text-white hover:bg-[#1a202c] rounded transition-colors"
        >
          <SkipForward className="w-3.5 h-3.5" />
        </button>

        <div className="h-4 w-px bg-lab-border mx-1" />

        {/* Loop Toggle */}
        <button
          onClick={onToggleLoop}
          title={isLooping ? 'Loop Playback: Enabled' : 'Loop Playback: Disabled'}
          className={`flex items-center space-x-1 px-2 py-1 rounded transition-colors ${
            isLooping
              ? 'bg-[#1a202c] text-sky-400 border border-[#2b3342]'
              : 'text-lab-textDim hover:text-slate-300'
          }`}
        >
          <Repeat className="w-3.5 h-3.5" />
          <span className="text-[10px] font-mono uppercase hidden sm:inline">Loop</span>
        </button>

        {/* Speed Selector */}
        <div className="flex items-center bg-[#0a0c10] p-0.5 rounded border border-lab-border text-[11px] font-mono">
          {SPEED_OPTIONS.map((spd) => (
            <button
              key={spd}
              onClick={() => onSpeedChange(spd)}
              className={`px-1.5 py-0.5 rounded transition-colors ${
                playbackSpeed === spd
                  ? 'bg-[#1a202c] text-sky-400 font-semibold border border-[#2b3342]'
                  : 'text-lab-textDim hover:text-slate-300'
              }`}
            >
              {spd}x
            </button>
          ))}
        </div>
      </div>

      {/* Timeline Scrubber Slider (Center) */}
      <div className="flex-1 flex items-center space-x-3 min-w-[180px]">
        {/* Frame Progress Readout */}
        <div className="flex items-center space-x-1.5 font-mono text-[11px] text-lab-textMuted shrink-0">
          <span className="text-lab-textDim text-[10px] uppercase hidden md:inline">FRAME:</span>
          <span className="text-sky-400 font-semibold tabular-nums">{currentFrameId}</span>
          <span className="text-lab-textDim text-[10px] hidden sm:inline">
            ({scrubValue + 1} / {totalFrames.toLocaleString()})
          </span>
        </div>

        {/* Range Scrubber Track */}
        <div className="flex-1 relative flex items-center">
          <input
            type="range"
            min={0}
            max={Math.max(0, totalFrames - 1)}
            value={scrubValue}
            onChange={handleSliderChange}
            onMouseDown={handleSliderMouseDown}
            onMouseUp={handleSliderMouseUp}
            onTouchStart={handleSliderMouseDown}
            onTouchEnd={handleSliderMouseUp}
            className="w-full accent-sky-500 h-1.5 bg-[#141820] rounded cursor-pointer transition-all"
            title={`Seek to frame index ${scrubValue + 1}`}
          />
        </div>

        {/* Progress Percentage */}
        <span className="font-mono text-[10px] text-lab-textDim shrink-0 tabular-nums hidden sm:inline">
          {progressPct}%
        </span>
      </div>

      {/* Playback Status & Notice (Right) */}
      <div className="flex items-center space-x-2.5 shrink-0 text-xs font-mono">
        {/* Loading Indicator */}
        {isLoading && (
          <div className="flex items-center space-x-1 text-sky-400 text-[11px]">
            <RefreshCw className="w-3 h-3 animate-spin" />
            <span className="hidden xl:inline text-[10px]">BUFFERING</span>
          </div>
        )}

        {/* Playback Status Badge */}
        <div className="flex items-center space-x-1.5 px-2 py-0.5 rounded bg-[#0a0c10] border border-lab-border text-[10px]">
          <span
            className={`w-1.5 h-1.5 rounded-full ${
              isPlaying ? 'bg-emerald-500' : 'bg-slate-500'
            }`}
          />
          <span className={isPlaying ? 'text-emerald-400 font-medium' : 'text-lab-textDim'}>
            {isPlaying ? 'PLAYING' : 'PAUSED'}
          </span>
        </div>

        <span className="text-[10px] text-lab-textDim uppercase hidden lg:inline">
          Validation Split
        </span>
      </div>
    </div>
  );
}

