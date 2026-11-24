import React, { useEffect, useRef, useState, useCallback } from 'react';
import * as THREE from 'three';
import { Maximize2, RotateCcw, Compass, Eye, ShieldAlert, Sparkles, Layers } from 'lucide-react';

// Color map gradient helpers
function turboColormap(t) {
  // Approximate Turbo colormap: t in [0, 1]
  const r = Math.min(Math.max(1.5 * Math.sin(Math.PI * t), 0), 1);
  const g = Math.min(Math.max(1.8 * Math.sin(Math.PI * (t - 0.25)), 0), 1);
  const b = Math.min(Math.max(1.5 * Math.sin(Math.PI * (t - 0.6)), 0), 1);
  return [r, g, b];
}

function viridisColormap(t) {
  // Viridis approximation
  t = Math.max(0, Math.min(1, t));
  const r = 0.267 + 0.004 * t + 0.687 * t * t;
  const g = 0.004 + 0.88 * t - 0.5 * t * t;
  const b = 0.33 + 0.55 * t - 0.7 * t * t;
  return [Math.max(0, Math.min(1, r)), Math.max(0, Math.min(1, g)), Math.max(0, Math.min(1, b))];
}

function getMethodSolidColor(method) {
  switch (method) {
    case 'original':
      return [0.65, 0.7, 0.75]; // Silver
    case 'uniform':
      return [0.94, 0.27, 0.27]; // Red
    case 'hybrid_v5':
      return [0.96, 0.62, 0.07]; // Amber
    case 'v6_alpha_0.25':
      return [0.06, 0.75, 0.51]; // Emerald
    case 'v7_lambda_0.25':
      return [0.23, 0.51, 0.96]; // Blue
    default:
      return [0.06, 0.71, 0.83]; // Cyan
  }
}

export default function ThreePointCloudViewer({
  pointsData, // Float32Array of [x, y, z, i, ...] or Array of [x, y, z, i]
  groundTruthObjects = [],
  predictionObjects = [],
  colorMode = 'elevation', // 'elevation' | 'intensity' | 'distance' | 'solid'
  pointSize = 1.8,
  showGrid = true,
  showRangeRings = true,
  showGroundTruth = true,
  showPredictions = false,
  visibleClasses = { Car: true, Pedestrian: true, Cyclist: true },
  syncCameraState = null,
  onCameraChange = null,
  methodId = 'uniform',
  renderMeta = null,
  cameraPreset = null,
}) {
  const containerRef = useRef(null);
  const sceneRef = useRef(null);
  const rendererRef = useRef(null);
  const cameraRef = useRef(null);
  const pointsMeshRef = useRef(null);
  const gtLinesGroupRef = useRef(null);
  const predLinesGroupRef = useRef(null);
  const ringsGroupRef = useRef(null);

  const isInteractingRef = useRef(false);
  const cameraTargetRef = useRef(new THREE.Vector3(15, 0, 0));
  const previousMousePosition = useRef({ x: 0, y: 0 });

  // Camera preset handler
  const setCameraView = useCallback((preset) => {
    const camera = cameraRef.current;
    if (!camera) return;

    if (preset === 'perspective') {
      camera.position.set(-20, -22, 18);
      cameraTargetRef.current.set(20, 0, -1);
    } else if (preset === 'top') {
      camera.position.set(20, 0, 70);
      cameraTargetRef.current.set(20, 0, 0);
    } else if (preset === 'front') {
      camera.position.set(-8, 0, 1.7);
      cameraTargetRef.current.set(30, 0, 0);
    } else if (preset === 'side') {
      camera.position.set(20, -45, 10);
      cameraTargetRef.current.set(20, 0, 0);
    }
    camera.lookAt(cameraTargetRef.current);

    if (onCameraChange) {
      onCameraChange({
        position: camera.position.toArray(),
        target: cameraTargetRef.current.toArray(),
      });
    }
  }, [onCameraChange]);

  // Optional external preset trigger
  useEffect(() => {
    if (cameraPreset) {
      setCameraView(cameraPreset);
    }
  }, [cameraPreset, setCameraView]);

  // Sync incoming camera state from sibling comparison viewports
  useEffect(() => {
    if (!syncCameraState || isInteractingRef.current || !cameraRef.current) return;
    const { position, target } = syncCameraState;
    if (position && target) {
      cameraRef.current.position.set(position[0], position[1], position[2]);
      cameraTargetRef.current.set(target[0], target[1], target[2]);
      cameraRef.current.lookAt(cameraTargetRef.current);
    }
  }, [syncCameraState]);

  // Initial Scene Setup
  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;

    const width = container.clientWidth;
    const height = container.clientHeight;

    // 1. Scene
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x0a0c10);
    sceneRef.current = scene;

    // 2. Camera (KITTI convention: Z is UP, X is forward, Y is left)
    const camera = new THREE.PerspectiveCamera(50, width / height, 0.2, 800);
    camera.up.set(0, 0, 1); // Z is vertical UP
    camera.position.set(-18, -20, 15);
    cameraTargetRef.current.set(20, 0, -1);
    camera.lookAt(cameraTargetRef.current);
    cameraRef.current = camera;

    // 3. Renderer
    const renderer = new THREE.WebGLRenderer({ antialias: true, powerPreference: 'high-performance' });
    renderer.setSize(width, height);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.outputColorSpace = THREE.SRGBColorSpace;
    container.innerHTML = '';
    container.appendChild(renderer.domElement);
    rendererRef.current = renderer;

    // 4. Ground Grid & Concentric Range Circles
    const ringsGroup = new THREE.Group();
    ringsGroupRef.current = ringsGroup;

    // Polar range rings at 10m, 20m, 40m, 60m
    const radii = [10, 20, 40, 60];
    radii.forEach((r) => {
      const ringGeo = new THREE.BufferGeometry();
      const segments = 64;
      const pts = [];
      for (let i = 0; i <= segments; i++) {
        const theta = (i / segments) * Math.PI * 2;
        pts.push(r * Math.cos(theta), r * Math.sin(theta), -1.73);
      }
      ringGeo.setAttribute('position', new THREE.Float32BufferAttribute(pts, 3));
      const ringMat = new THREE.LineBasicMaterial({
        color: r === 20 ? 0x2b384c : 0x181f2a,
        transparent: true,
        opacity: r === 20 ? 0.6 : 0.4,
      });
      ringsGroup.add(new THREE.Line(ringGeo, ringMat));
    });

    // Coordinate Axes at Sensor origin (X: Red forward, Y: Green left, Z: Blue up)
    const axesHelper = new THREE.AxesHelper(3.0);
    axesHelper.position.set(0, 0, 0);
    ringsGroup.add(axesHelper);

    // Sensor origin representation (Velodyne sensor puck)
    const sensorGeo = new THREE.CylinderGeometry(0.3, 0.3, 0.25, 16);
    sensorGeo.rotateX(Math.PI / 2);
    const sensorMat = new THREE.MeshBasicMaterial({ color: 0x06b6d4, wireframe: true });
    const sensorMesh = new THREE.Mesh(sensorGeo, sensorMat);
    ringsGroup.add(sensorMesh);

    scene.add(ringsGroup);

    // 5. Groups for Bounding Boxes
    const gtGroup = new THREE.Group();
    gtLinesGroupRef.current = gtGroup;
    scene.add(gtGroup);

    const predGroup = new THREE.Group();
    predLinesGroupRef.current = predGroup;
    scene.add(predGroup);

    // 6. Animation loop
    let animId;
    const animate = () => {
      animId = requestAnimationFrame(animate);
      renderer.render(scene, camera);
    };
    animate();

    // 7. Mouse Orbit Controls
    let isDragging = false;
    let isPanning = false;

    const onMouseDown = (e) => {
      isInteractingRef.current = true;
      if (e.button === 0 && !e.shiftKey) {
        isDragging = true;
      } else if (e.button === 2 || (e.button === 0 && e.shiftKey)) {
        isPanning = true;
      }
      previousMousePosition.current = { x: e.clientX, y: e.clientY };
    };

    const onMouseMove = (e) => {
      if (!isDragging && !isPanning) return;
      const deltaX = e.clientX - previousMousePosition.current.x;
      const deltaY = e.clientY - previousMousePosition.current.y;
      previousMousePosition.current = { x: e.clientX, y: e.clientY };

      const cam = cameraRef.current;
      const target = cameraTargetRef.current;
      if (!cam) return;

      if (isDragging) {
        // Orbit around target
        const offset = new THREE.Vector3().subVectors(cam.position, target);
        let radius = offset.length();
        let theta = Math.atan2(offset.y, offset.x); // azimuth in XY
        let phi = Math.acos(Math.max(-1, Math.min(1, offset.z / radius))); // polar angle from Z

        theta -= deltaX * 0.008;
        phi = Math.max(0.05, Math.min(Math.PI - 0.05, phi - deltaY * 0.008));

        offset.x = radius * Math.sin(phi) * Math.cos(theta);
        offset.y = radius * Math.sin(phi) * Math.sin(theta);
        offset.z = radius * Math.cos(phi);

        cam.position.copy(target).add(offset);
        cam.lookAt(target);
      } else if (isPanning) {
        // Pan camera & target parallel to screen
        const panSpeed = 0.04;
        const forward = new THREE.Vector3().subVectors(target, cam.position).normalize();
        const right = new THREE.Vector3().crossVectors(forward, cam.up).normalize();
        const up = new THREE.Vector3().crossVectors(right, forward).normalize();

        const panOffset = new THREE.Vector3()
          .addScaledVector(right, -deltaX * panSpeed)
          .addScaledVector(up, deltaY * panSpeed);

        cam.position.add(panOffset);
        target.add(panOffset);
        cam.lookAt(target);
      }

      if (onCameraChange) {
        onCameraChange({
          position: cam.position.toArray(),
          target: target.toArray(),
        });
      }
    };

    const onMouseUp = () => {
      isDragging = false;
      isPanning = false;
      isInteractingRef.current = false;
    };

    const onWheel = (e) => {
      e.preventDefault();
      const cam = cameraRef.current;
      const target = cameraTargetRef.current;
      if (!cam) return;

      const offset = new THREE.Vector3().subVectors(cam.position, target);
      const zoomFactor = e.deltaY > 0 ? 1.08 : 0.92;
      const newDistance = offset.length() * zoomFactor;

      if (newDistance > 2 && newDistance < 350) {
        offset.multiplyScalar(zoomFactor);
        cam.position.copy(target).add(offset);
        cam.lookAt(target);

        if (onCameraChange) {
          onCameraChange({
            position: cam.position.toArray(),
            target: target.toArray(),
          });
        }
      }
    };

    const domElement = renderer.domElement;
    domElement.addEventListener('mousedown', onMouseDown);
    window.addEventListener('mousemove', onMouseMove);
    window.addEventListener('mouseup', onMouseUp);
    domElement.addEventListener('wheel', onWheel, { passive: false });
    domElement.addEventListener('contextmenu', (e) => e.preventDefault());

    // Resize handler
    const handleResize = () => {
      if (!containerRef.current || !rendererRef.current || !cameraRef.current) return;
      const w = containerRef.current.clientWidth;
      const h = containerRef.current.clientHeight;
      cameraRef.current.aspect = w / h;
      cameraRef.current.updateProjectionMatrix();
      rendererRef.current.setSize(w, h);
    };
    window.addEventListener('resize', handleResize);

    return () => {
      cancelAnimationFrame(animId);
      domElement.removeEventListener('mousedown', onMouseDown);
      window.removeEventListener('mousemove', onMouseMove);
      window.removeEventListener('mouseup', onMouseUp);
      domElement.removeEventListener('wheel', onWheel);
      window.removeEventListener('resize', handleResize);
      renderer.dispose();
    };
  }, []);

  // Update Points Geometry and Colors
  useEffect(() => {
    if (!sceneRef.current) return;

    // Clean up old point cloud mesh
    if (pointsMeshRef.current) {
      sceneRef.current.remove(pointsMeshRef.current);
      pointsMeshRef.current.geometry.dispose();
      pointsMeshRef.current.material.dispose();
      pointsMeshRef.current = null;
    }

    if (!pointsData) return;

    let numPoints = 0;
    let positions;
    let colors;

    if (pointsData instanceof Float32Array) {
      numPoints = Math.floor(pointsData.length / 4);
      positions = new Float32Array(numPoints * 3);
      colors = new Float32Array(numPoints * 3);

      const solidColor = getMethodSolidColor(methodId);

      for (let i = 0; i < numPoints; i++) {
        const x = pointsData[i * 4];
        const y = pointsData[i * 4 + 1];
        const z = pointsData[i * 4 + 2];
        const intensity = pointsData[i * 4 + 3];

        positions[i * 3] = x;
        positions[i * 3 + 1] = y;
        positions[i * 3 + 2] = z;

        let rgb = [0.1, 0.7, 0.9];
        if (colorMode === 'elevation') {
          // Normalize height z in [-2.5, 3.0]
          const t = Math.max(0, Math.min(1, (z + 2.5) / 5.5));
          rgb = turboColormap(t);
        } else if (colorMode === 'intensity') {
          const t = Math.max(0, Math.min(1, intensity));
          rgb = viridisColormap(t);
        } else if (colorMode === 'distance') {
          const dist = Math.sqrt(x * x + y * y + z * z);
          const t = Math.max(0, Math.min(1, dist / 60.0));
          rgb = turboColormap(1.0 - t);
        } else if (colorMode === 'solid') {
          rgb = solidColor;
        }

        colors[i * 3] = rgb[0];
        colors[i * 3 + 1] = rgb[1];
        colors[i * 3 + 2] = rgb[2];
      }
    } else if (Array.isArray(pointsData)) {
      numPoints = pointsData.length;
      positions = new Float32Array(numPoints * 3);
      colors = new Float32Array(numPoints * 3);
      const solidColor = getMethodSolidColor(methodId);

      for (let i = 0; i < numPoints; i++) {
        const pt = pointsData[i];
        const x = pt[0], y = pt[1], z = pt[2], intensity = pt[3] || 0;
        positions[i * 3] = x;
        positions[i * 3 + 1] = y;
        positions[i * 3 + 2] = z;

        let rgb = [0.1, 0.7, 0.9];
        if (colorMode === 'elevation') {
          const t = Math.max(0, Math.min(1, (z + 2.5) / 5.5));
          rgb = turboColormap(t);
        } else if (colorMode === 'intensity') {
          rgb = viridisColormap(Math.max(0, Math.min(1, intensity)));
        } else if (colorMode === 'distance') {
          const dist = Math.sqrt(x * x + y * y + z * z);
          rgb = turboColormap(1.0 - Math.max(0, Math.min(1, dist / 60.0)));
        } else if (colorMode === 'solid') {
          rgb = solidColor;
        }

        colors[i * 3] = rgb[0];
        colors[i * 3 + 1] = rgb[1];
        colors[i * 3 + 2] = rgb[2];
      }
    }

    if (numPoints === 0) return;

    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));

    const material = new THREE.PointsMaterial({
      size: pointSize,
      vertexColors: true,
      sizeAttenuation: true,
      transparent: true,
      opacity: 0.9,
    });

    const pointsMesh = new THREE.Points(geometry, material);
    sceneRef.current.add(pointsMesh);
    pointsMeshRef.current = pointsMesh;
  }, [pointsData, colorMode, pointSize, methodId]);

  // Update Ground Truth 3D Bounding Boxes
  useEffect(() => {
    if (!gtLinesGroupRef.current) return;
    const group = gtLinesGroupRef.current;

    // Clear previous boxes
    while (group.children.length > 0) {
      const obj = group.children[0];
      group.remove(obj);
      if (obj.geometry) obj.geometry.dispose();
      if (obj.material) obj.material.dispose();
    }

    if (!showGroundTruth || !groundTruthObjects || groundTruthObjects.length === 0) return;

    const classColors = {
      Car: 0x10b981,        // Emerald
      Pedestrian: 0xf59e0b, // Amber
      Cyclist: 0x06b6d4,    // Cyan
    };

    groundTruthObjects.forEach((gt) => {
      const cls = gt.class;
      if (!visibleClasses[cls]) return;

      const lines = gt.box_3d?.lines || [];
      const linePositions = [];

      lines.forEach((pair) => {
        const p1 = pair[0];
        const p2 = pair[1];
        linePositions.push(p1[0], p1[1], p1[2]);
        linePositions.push(p2[0], p2[1], p2[2]);
      });

      if (linePositions.length === 0) return;

      const geo = new THREE.BufferGeometry();
      geo.setAttribute('position', new THREE.Float32BufferAttribute(linePositions, 3));

      const color = classColors[cls] || 0x94a3b8;
      const mat = new THREE.LineBasicMaterial({
        color,
        linewidth: 2,
        transparent: true,
        opacity: 0.95,
      });

      const lineMesh = new THREE.LineSegments(geo, mat);
      group.add(lineMesh);
    });
  }, [groundTruthObjects, showGroundTruth, visibleClasses]);

  // Update Prediction 3D Boxes
  useEffect(() => {
    if (!predLinesGroupRef.current) return;
    const group = predLinesGroupRef.current;

    while (group.children.length > 0) {
      const obj = group.children[0];
      group.remove(obj);
      if (obj.geometry) obj.geometry.dispose();
      if (obj.material) obj.material.dispose();
    }

    if (!showPredictions || !predictionObjects || predictionObjects.length === 0) return;

    predictionObjects.forEach((pred) => {
      const cls = pred.class;
      if (!visibleClasses[cls]) return;

      const lines = pred.box_3d?.lines || [];
      const linePositions = [];

      lines.forEach((pair) => {
        const p1 = pair[0];
        const p2 = pair[1];
        linePositions.push(p1[0], p1[1], p1[2]);
        linePositions.push(p2[0], p2[1], p2[2]);
      });

      if (linePositions.length === 0) return;

      const geo = new THREE.BufferGeometry();
      geo.setAttribute('position', new THREE.Float32BufferAttribute(linePositions, 3));

      const mat = new THREE.LineDashedMaterial({
        color: 0xec4899, // Magenta pink for detector predictions
        dashSize: 0.6,
        gapSize: 0.3,
        linewidth: 2,
      });

      const lineMesh = new THREE.LineSegments(geo, mat);
      lineMesh.computeLineDistances();
      group.add(lineMesh);
    });
  }, [predictionObjects, showPredictions, visibleClasses]);

  // Toggle grid & range rings visibility
  useEffect(() => {
    if (ringsGroupRef.current) {
      ringsGroupRef.current.visible = showRangeRings;
    }
  }, [showRangeRings]);

  return (
    <div className="relative w-full h-full select-none overflow-hidden bg-lab-bg">
      {/* Three.js Canvas Container */}
      <div ref={containerRef} className="w-full h-full cursor-grab active:cursor-grabbing" />

      {/* Floating HUD Camera Presets & Reset */}
      <div className="absolute top-2.5 right-2.5 flex items-center space-x-1 bg-lab-surface/90 backdrop-blur-sm p-1 rounded border border-lab-border z-10">
        <button
          onClick={() => setCameraView('perspective')}
          title="3D Perspective"
          className="px-2 py-0.5 text-[11px] font-mono text-slate-300 hover:text-white hover:bg-[#1a202c] rounded transition-colors"
        >
          3D
        </button>
        <button
          onClick={() => setCameraView('top')}
          title="Bird's Eye View (BEV)"
          className="px-2 py-0.5 text-[11px] font-mono text-slate-300 hover:text-white hover:bg-[#1a202c] rounded transition-colors"
        >
          BEV
        </button>
        <button
          onClick={() => setCameraView('front')}
          title="Driver Front View"
          className="px-2 py-0.5 text-[11px] font-mono text-slate-300 hover:text-white hover:bg-[#1a202c] rounded transition-colors"
        >
          POV
        </button>
        <button
          onClick={() => setCameraView('side')}
          title="Profile Side View"
          className="px-2 py-0.5 text-[11px] font-mono text-slate-300 hover:text-white hover:bg-[#1a202c] rounded transition-colors"
        >
          SIDE
        </button>
        <div className="h-3.5 w-px bg-lab-border mx-0.5" />
        <button
          onClick={() => setCameraView('perspective')}
          title="Reset Camera"
          className="p-1 text-lab-textMuted hover:text-white hover:bg-[#1a202c] rounded transition-colors"
        >
          <RotateCcw className="w-3 h-3" />
        </button>
      </div>

      {/* Telemetry HUD Badge */}
      <div className="absolute bottom-2.5 left-2.5 flex items-center space-x-2 text-[11px] font-mono text-lab-textMuted bg-lab-surface/90 backdrop-blur-sm px-2.5 py-1 rounded border border-lab-border z-10">
        <span className="flex items-center space-x-1.5">
          <span className="w-1.5 h-1.5 rounded-full bg-sky-400" />
          <span className="text-slate-200 font-semibold">{methodId.toUpperCase()}</span>
        </span>
        {renderMeta && (
          <>
            <span className="text-lab-borderLight">|</span>
            <span>
              PTS: <strong className="text-slate-200 tabular-nums">{renderMeta.renderedPoints.toLocaleString()}</strong>
              {renderMeta.isDownsampled && (
                <span className="text-amber-400 ml-1">
                  (CAPPED @ {renderMeta.renderCap.toLocaleString()})
                </span>
              )}
            </span>
          </>
        )}
        <span className="text-lab-borderLight">|</span>
        <span className="text-sky-400 uppercase">{colorMode}</span>
      </div>

      {/* Dynamic Colormap Legend Bar */}
      <div className="absolute bottom-2.5 left-1/2 -translate-x-1/2 hidden sm:flex items-center space-x-2 text-[10px] font-mono bg-lab-surface/90 backdrop-blur-sm px-2.5 py-1 rounded border border-lab-border text-lab-textMuted z-10 pointer-events-none select-none">
        {colorMode === 'elevation' && (
          <div className="flex items-center space-x-1.5">
            <span className="text-sky-400 font-semibold">-3m</span>
            <div className="w-16 sm:w-24 h-1.5 rounded-sm bg-gradient-to-r from-blue-600 via-emerald-400 via-amber-400 to-rose-500" />
            <span className="text-rose-400 font-semibold">+2m</span>
            <span className="text-lab-textDim ml-1">Elevation (Z)</span>
          </div>
        )}
        {colorMode === 'intensity' && (
          <div className="flex items-center space-x-1.5">
            <span className="text-indigo-400 font-semibold">0.0</span>
            <div className="w-16 sm:w-24 h-1.5 rounded-sm bg-gradient-to-r from-indigo-950 via-teal-500 to-yellow-400" />
            <span className="text-yellow-400 font-semibold">1.0</span>
            <span className="text-lab-textDim ml-1">Intensity (I)</span>
          </div>
        )}
        {colorMode === 'distance' && (
          <div className="flex items-center space-x-1.5">
            <span className="text-emerald-400 font-semibold">0m</span>
            <div className="w-16 sm:w-24 h-1.5 rounded-sm bg-gradient-to-r from-emerald-500 via-sky-400 to-indigo-600" />
            <span className="text-indigo-400 font-semibold">70m</span>
            <span className="text-lab-textDim ml-1">Radial Range</span>
          </div>
        )}
        {colorMode === 'solid' && (
          <div className="flex items-center space-x-1.5">
            <span
              className="w-2 h-2 rounded-full"
              style={{
                backgroundColor: `rgb(${Math.round(getMethodSolidColor(methodId)[0] * 255)}, ${Math.round(getMethodSolidColor(methodId)[1] * 255)}, ${Math.round(getMethodSolidColor(methodId)[2] * 255)})`,
              }}
            />
            <span className="text-slate-200 uppercase font-semibold">{methodId}</span>
            <span className="text-lab-textDim ml-1">Solid Color</span>
          </div>
        )}
      </div>

      {/* Axes Legend */}
      <div className="absolute bottom-2.5 right-2.5 flex items-center space-x-2 text-[10px] font-mono bg-lab-surface/90 backdrop-blur-sm px-2 py-0.5 rounded border border-lab-border text-lab-textMuted z-10">
        <span className="text-rose-400 font-medium">+X (Fwd)</span>
        <span className="text-emerald-400 font-medium">+Y (Left)</span>
        <span className="text-sky-400 font-medium">+Z (Up)</span>
      </div>
    </div>
  );
}

