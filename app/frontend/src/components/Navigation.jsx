import React, { useState } from 'react';
import {
  Compass,
  Box,
  SplitSquareVertical,
  BarChart3,
  Target,
  BookOpen,
  ChevronLeft,
  ChevronRight,
  Shuffle,
  Search,
  Layers,
  MonitorPlay,
} from 'lucide-react';

const TABS = [
  { id: 'overview', label: 'Overview', icon: Compass },
  { id: 'explorer', label: '3D Explorer', icon: Box },
  { id: 'comparison', label: 'Comparison', icon: SplitSquareVertical },
  { id: 'analysis', label: 'Research Analytics', icon: BarChart3 },
  { id: 'detector', label: 'Detector Benchmark', icon: Target },
  { id: 'methodology', label: 'Methodology', icon: BookOpen },
];

export default function Navigation({
  activeTab,
  onSelectTab,
  currentFrameId,
  totalFrames = 3769,
  onPreviousFrame,
  onNextFrame,
  onRandomFrame,
  onSearchFrame,
  onTogglePresentation,
}) {
  const [searchInput, setSearchInput] = useState('');

  const handleSearchSubmit = (e) => {
    e.preventDefault();
    if (searchInput.trim()) {
      onSearchFrame(searchInput.trim());
      setSearchInput('');
    }
  };

  return (
    <header className="h-12 border-b border-lab-border bg-lab-surface px-4 flex items-center justify-between select-none z-30 shrink-0">
      {/* Brand & Project Identity */}
      <div className="flex items-center space-x-3">
        <div className="flex items-center justify-center w-7 h-7 rounded bg-[#161b24] border border-[#232b38] text-sky-400">
          <Layers className="w-4 h-4" />
        </div>
        <div className="flex items-center space-x-2">
          <span className="font-semibold text-[13px] tracking-tight text-lab-textBright">
            Adaptive LiDAR Lab
          </span>
          <span className="text-[10px] font-mono px-1.5 py-0.5 rounded bg-[#161a22] text-lab-textMuted border border-[#202632]">
            KITTI 3D
          </span>
        </div>
      </div>

      {/* Navigation Tabs - Clean Desktop Workstation Tabs */}
      <nav className="flex items-center h-full">
        {TABS.map((tab) => {
          const Icon = tab.icon;
          const isActive = activeTab === tab.id;
          return (
            <button
              key={tab.id}
              onClick={() => onSelectTab(tab.id)}
              className={`relative flex items-center space-x-2 h-12 px-3.5 text-xs font-medium transition-colors ${
                isActive
                  ? 'text-lab-textBright bg-[#151922]'
                  : 'text-lab-textMuted hover:text-slate-200 hover:bg-[#12161f]'
              }`}
            >
              <Icon className={`w-3.5 h-3.5 ${isActive ? 'text-sky-400' : 'text-lab-textMuted'}`} />
              <span>{tab.label}</span>
              {isActive && (
                <div className="absolute bottom-0 left-0 right-0 h-[2px] bg-sky-500" />
              )}
            </button>
          );
        })}
      </nav>

      {/* Frame Scrubber & Status Controls */}
      <div className="flex items-center space-x-2.5">
        {/* Frame Search Form */}
        <form onSubmit={handleSearchSubmit} className="relative hidden xl:block">
          <input
            type="text"
            placeholder="Search frame (e.g. 000008)..."
            value={searchInput}
            onChange={(e) => setSearchInput(e.target.value)}
            className="w-44 bg-[#141820] text-xs font-mono text-slate-200 pl-7 pr-2.5 py-1 rounded border border-lab-border focus:border-sky-500 focus:outline-none transition-colors placeholder:text-lab-textDim"
          />
          <Search className="w-3.5 h-3.5 text-lab-textDim absolute left-2 top-1.5" />
        </form>

        {/* Frame Stepper Group */}
        <div className="flex items-center bg-[#141820] rounded border border-lab-border overflow-hidden">
          <button
            onClick={onPreviousFrame}
            title="Previous Frame"
            className="p-1 px-1.5 text-lab-textMuted hover:text-lab-textBright hover:bg-[#1c222c] transition-colors border-r border-lab-border"
          >
            <ChevronLeft className="w-3.5 h-3.5" />
          </button>

          <div className="px-2.5 py-0.5 font-mono text-xs text-center flex items-center space-x-1">
            <span className="text-[10px] text-lab-textDim uppercase">FRAME</span>
            <span className="text-sky-400 font-semibold tracking-wider tabular-nums">{currentFrameId}</span>
          </div>

          <button
            onClick={onNextFrame}
            title="Next Frame"
            className="p-1 px-1.5 text-lab-textMuted hover:text-lab-textBright hover:bg-[#1c222c] transition-colors border-l border-lab-border"
          >
            <ChevronRight className="w-3.5 h-3.5" />
          </button>
        </div>

        {/* Random Frame Jump */}
        <button
          onClick={onRandomFrame}
          title="Jump to Random Validation Frame"
          className="p-1.5 text-lab-textMuted hover:text-lab-textBright hover:bg-[#141820] border border-lab-border rounded transition-colors"
        >
          <Shuffle className="w-3.5 h-3.5" />
        </button>

        {/* System Dataset Split Indicator */}
        <div className="hidden lg:flex items-center space-x-1.5 text-[11px] font-mono text-lab-textMuted bg-[#141820] px-2.5 py-1 rounded border border-lab-border">
          <span className="w-1.5 h-1.5 rounded-full bg-emerald-500" />
          <span className="text-slate-300 tabular-nums">{totalFrames.toLocaleString()}</span>
          <span className="text-lab-textDim">VAL</span>
        </div>

        {/* Presentation Cockpit Mode Button */}
        <button
          onClick={onTogglePresentation}
          title="Enter Fullscreen Presentation Mode [P]"
          className="flex items-center space-x-1.5 px-2.5 py-1 text-xs font-medium text-sky-400 bg-sky-950/40 hover:bg-sky-900/60 border border-sky-800/60 rounded transition-colors shadow-sm"
        >
          <MonitorPlay className="w-3.5 h-3.5" />
          <span className="hidden sm:inline">Present</span>
          <kbd className="hidden md:inline text-[9px] font-mono px-1 py-0.2 bg-sky-900/60 text-sky-300 rounded border border-sky-700/50">
            P
          </kbd>
        </button>
      </div>
    </header>
  );
}
