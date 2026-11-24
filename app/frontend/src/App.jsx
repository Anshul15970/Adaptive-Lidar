import React, { useState, useEffect, useCallback } from 'react';
import Navigation from './components/Navigation';
import OverviewView from './views/OverviewView';
import ExplorerView from './views/ExplorerView';
import ComparisonView from './views/ComparisonView';
import SamplingAnalysisView from './views/SamplingAnalysisView';
import DetectorPerformanceView from './views/DetectorPerformanceView';
import MethodologyView from './views/MethodologyView';
import PresentationView from './views/PresentationView';
import { fetchAlgorithms, fetchValidationFrames } from './api';

export default function App() {
  const [activeTab, setActiveTab] = useState('explorer');
  const [currentFrameId, setCurrentFrameId] = useState('000001');
  const [validationFrames, setValidationFrames] = useState(['000001']);
  const [algorithms, setAlgorithms] = useState([
    { id: 'original', name: 'Original', target_retention_pct: 100.0, color: '#94a3b8' },
    { id: 'uniform', name: 'Uniform 20%', target_retention_pct: 20.0, color: '#f43f5e' },
    { id: 'hybrid_v5', name: 'Hybrid V5', target_retention_pct: 20.0, color: '#f59e0b' },
    { id: 'v6_alpha_0.25', name: 'V6 α=0.25', target_retention_pct: 20.0, color: '#10b981' },
    { id: 'v7_lambda_0.25', name: 'V7 λ=0.25', target_retention_pct: 20.0, color: '#38bdf8' },
  ]);
  const [selectedMethod, setSelectedMethod] = useState('v6_alpha_0.25');

  // Playback engine state
  const [isPlaying, setIsPlaying] = useState(false);
  const [playbackSpeed, setPlaybackSpeed] = useState(1);
  const [isLooping, setIsLooping] = useState(true);
  const [isFrameLoading, setIsFrameLoading] = useState(false);

  // Fullscreen Presentation Cockpit mode
  const [isPresentationMode, setIsPresentationMode] = useState(false);

  // Load algorithm metadata and validation frame list on app mount
  useEffect(() => {
    async function initApp() {
      try {
        const [algos, framesData] = await Promise.all([
          fetchAlgorithms(),
          fetchValidationFrames(),
        ]);
        if (algos && algos.length > 0) {
          setAlgorithms(algos);
        }
        if (framesData && framesData.frame_ids && framesData.frame_ids.length > 0) {
          setValidationFrames(framesData.frame_ids);
          setCurrentFrameId(framesData.default_frame || framesData.frame_ids[0]);
        }
      } catch (err) {
        console.error('Failed to initialize app data:', err);
      }
    }
    initApp();
  }, []);

  // Frame navigation helpers
  const handlePreviousFrame = useCallback(() => {
    const idx = validationFrames.indexOf(currentFrameId);
    if (idx > 0) {
      setCurrentFrameId(validationFrames[idx - 1]);
    } else {
      setCurrentFrameId(validationFrames[validationFrames.length - 1]);
    }
  }, [validationFrames, currentFrameId]);

  const handleNextFrame = useCallback(() => {
    const idx = validationFrames.indexOf(currentFrameId);
    if (idx >= 0 && idx < validationFrames.length - 1) {
      setCurrentFrameId(validationFrames[idx + 1]);
    } else {
      setCurrentFrameId(validationFrames[0]);
    }
  }, [validationFrames, currentFrameId]);

  const handleRandomFrame = useCallback(() => {
    if (validationFrames.length === 0) return;
    const randomIdx = Math.floor(Math.random() * validationFrames.length);
    setCurrentFrameId(validationFrames[randomIdx]);
  }, [validationFrames]);

  const handleSearchFrame = (query) => {
    const clean = query.trim();
    // Normalize to 6-digit zero padded if integer
    const target = /^\d+$/.test(clean) ? clean.padStart(6, '0') : clean;

    if (validationFrames.includes(target)) {
      setCurrentFrameId(target);
    } else {
      const match = validationFrames.find((f) => f.includes(clean));
      if (match) {
        setCurrentFrameId(match);
      } else {
        alert(`Validation frame "${clean}" not found in the 3,769 validation frames.`);
      }
    }
  };

  // Playback control actions
  const handleTogglePlay = useCallback(() => {
    setIsPlaying((prev) => !prev);
  }, []);

  const handleRestart = useCallback(() => {
    if (validationFrames.length > 0) {
      setCurrentFrameId(validationFrames[0]);
    }
  }, [validationFrames]);

  const handleSeek = useCallback((index) => {
    if (validationFrames.length === 0) return;
    const clamped = Math.max(0, Math.min(index, validationFrames.length - 1));
    setCurrentFrameId(validationFrames[clamped]);
  }, [validationFrames]);

  const handleSpeedChange = useCallback((spd) => {
    setPlaybackSpeed(spd);
  }, []);

  const handleToggleLoop = useCallback(() => {
    setIsLooping((prev) => !prev);
  }, []);

  // Continuous frame playback interval loop
  useEffect(() => {
    if (!isPlaying || validationFrames.length === 0) return;

    // Base step interval: 600ms at 1x, scaled by multiplier
    const intervalMs = Math.max(60, Math.round(600 / playbackSpeed));

    const timer = setTimeout(() => {
      // If frame is still fetching binary buffers, yield tick to avoid stacking requests
      if (isFrameLoading) return;

      const idx = validationFrames.indexOf(currentFrameId);
      if (idx === -1) return;

      if (idx < validationFrames.length - 1) {
        setCurrentFrameId(validationFrames[idx + 1]);
      } else if (isLooping) {
        setCurrentFrameId(validationFrames[0]);
      } else {
        setIsPlaying(false);
      }
    }, intervalMs);

    return () => clearTimeout(timer);
  }, [isPlaying, playbackSpeed, isLooping, currentFrameId, validationFrames, isFrameLoading]);

  // Global Keyboard Shortcuts (Space to play/pause, Arrows to step)
  useEffect(() => {
    const handleKeyDown = (e) => {
      if (['INPUT', 'SELECT', 'TEXTAREA'].includes(document.activeElement?.tagName)) {
        return;
      }
      if (e.code === 'Space') {
        e.preventDefault();
        handleTogglePlay();
      } else if (e.code === 'ArrowRight') {
        e.preventDefault();
        handleNextFrame();
      } else if (e.code === 'ArrowLeft') {
        e.preventDefault();
        handlePreviousFrame();
      } else if (e.code === 'KeyP') {
        e.preventDefault();
        setIsPresentationMode((prev) => !prev);
      } else if (e.code === 'Escape' && isPresentationMode) {
        e.preventDefault();
        setIsPresentationMode(false);
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [handleTogglePlay, handleNextFrame, handlePreviousFrame, isPresentationMode]);

  // Packaged playback props
  const currentFrameIndex = Math.max(0, validationFrames.indexOf(currentFrameId));
  const playbackProps = {
    isPlaying,
    playbackSpeed,
    isLooping,
    currentIndex: currentFrameIndex,
    totalFrames: validationFrames.length,
    onTogglePlay: handleTogglePlay,
    onPreviousFrame: handlePreviousFrame,
    onNextFrame: handleNextFrame,
    onRestart: handleRestart,
    onSpeedChange: handleSpeedChange,
    onToggleLoop: handleToggleLoop,
    onSeek: handleSeek,
    onLoadingChange: setIsFrameLoading,
  };

  return (
    <div className="flex flex-col h-screen w-screen overflow-hidden bg-lab-bg text-slate-100 antialiased">
      {/* Top Navigation */}
      <Navigation
        activeTab={activeTab}
        onSelectTab={setActiveTab}
        currentFrameId={currentFrameId}
        totalFrames={validationFrames.length}
        onPreviousFrame={handlePreviousFrame}
        onNextFrame={handleNextFrame}
        onRandomFrame={handleRandomFrame}
        onSearchFrame={handleSearchFrame}
        onTogglePresentation={() => setIsPresentationMode((prev) => !prev)}
      />

      {/* Main Content Area */}
      <main className="flex-1 flex overflow-hidden">
        {activeTab === 'overview' && (
          <OverviewView
            onSelectTab={setActiveTab}
            currentFrameId={currentFrameId}
          />
        )}

        {activeTab === 'explorer' && (
          <ExplorerView
            currentFrameId={currentFrameId}
            algorithms={algorithms}
            selectedMethod={selectedMethod}
            onSelectMethod={setSelectedMethod}
            onPreviousFrame={handlePreviousFrame}
            onNextFrame={handleNextFrame}
            onRandomFrame={handleRandomFrame}
            onEnterPresentation={() => setIsPresentationMode(true)}
            playbackProps={playbackProps}
          />
        )}

        {activeTab === 'comparison' && (
          <ComparisonView
            currentFrameId={currentFrameId}
            algorithms={algorithms}
            onPreviousFrame={handlePreviousFrame}
            onNextFrame={handleNextFrame}
            onRandomFrame={handleRandomFrame}
            onEnterPresentation={() => setIsPresentationMode(true)}
            playbackProps={playbackProps}
          />
        )}

        {activeTab === 'analysis' && (
          <SamplingAnalysisView
            currentFrameId={currentFrameId}
          />
        )}

        {activeTab === 'detector' && (
          <DetectorPerformanceView />
        )}

        {activeTab === 'methodology' && (
          <MethodologyView />
        )}
      </main>

      {/* Fullscreen Presentation Cockpit Overlay */}
      {isPresentationMode && (
        <PresentationView
          currentFrameId={currentFrameId}
          validationFrames={validationFrames}
          algorithms={algorithms}
          selectedMethod={selectedMethod}
          onSelectMethod={setSelectedMethod}
          onPreviousFrame={handlePreviousFrame}
          onNextFrame={handleNextFrame}
          onRandomFrame={handleRandomFrame}
          onClosePresentation={() => setIsPresentationMode(false)}
          playbackProps={playbackProps}
        />
      )}
    </div>
  );
}
