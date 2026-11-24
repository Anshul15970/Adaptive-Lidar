/**
 * API client for Adaptive LiDAR Lab Backend.
 */

const BASE_URL = '/api';

// In-memory client LRU caches for ultra-smooth playback and scrub seeking
const pointCache = new Map();
const MAX_POINT_CACHE_SIZE = 50;

const gtCache = new Map();
const MAX_GT_CACHE_SIZE = 60;

const metricsCache = new Map();
const MAX_METRICS_CACHE_SIZE = 60;

export async function fetchHealth() {
  const res = await fetch(`${BASE_URL}/health`);
  if (!res.ok) throw new Error(`Health check failed: ${res.statusText}`);
  return res.json();
}

export async function fetchAlgorithms() {
  const res = await fetch(`${BASE_URL}/algorithms`);
  if (!res.ok) throw new Error(`Failed to fetch algorithms: ${res.statusText}`);
  return res.json();
}

export async function fetchValidationFrames(limit = null, offset = 0, search = null) {
  const params = new URLSearchParams();
  if (limit) params.set('limit', limit);
  if (offset) params.set('offset', offset);
  if (search) params.set('search', search);

  const res = await fetch(`${BASE_URL}/frames?${params.toString()}`);
  if (!res.ok) throw new Error(`Failed to fetch validation frames: ${res.statusText}`);
  return res.json();
}

export async function fetchFramePoints(frameId, method = 'uniform', maxPoints = 60000) {
  const params = new URLSearchParams({ method });
  if (maxPoints !== null && maxPoints !== undefined) {
    params.set('max_points', maxPoints);
  }
  const res = await fetch(`${BASE_URL}/frames/${frameId}/points?${params.toString()}`);
  if (!res.ok) throw new Error(`Failed to load points for frame ${frameId}: ${res.statusText}`);
  return res.json();
}

/**
 * High-speed binary point cloud fetcher for direct GPU WebGL uploading.
 * Bypasses JSON parsing, transferring 24k-120k points in <5ms.
 */
export async function fetchFramePointsBinary(frameId, method = 'uniform', maxPoints = 60000) {
  const cacheKey = `${frameId}_${method}_${maxPoints}`;
  if (pointCache.has(cacheKey)) {
    return pointCache.get(cacheKey);
  }

  const params = new URLSearchParams({ method });
  if (maxPoints !== null && maxPoints !== undefined) {
    params.set('max_points', maxPoints);
  }
  const res = await fetch(`${BASE_URL}/frames/${frameId}/points/binary?${params.toString()}`);
  if (!res.ok) throw new Error(`Failed to load binary points for frame ${frameId}: ${res.statusText}`);

  const buffer = await res.arrayBuffer();
  const floatArray = new Float32Array(buffer);

  const metadata = {
    frameId: res.headers.get('X-Frame-ID') || frameId,
    method: res.headers.get('X-Method') || method,
    totalPoints: parseInt(res.headers.get('X-Total-Points') || '0', 10),
    renderedPoints: parseInt(res.headers.get('X-Rendered-Points') || '0', 10),
    isDownsampled: res.headers.get('X-Is-Downsampled') === 'true',
    samplingRatio: parseFloat(res.headers.get('X-Sampling-Ratio') || '1.0'),
    renderCap: res.headers.get('X-Render-Cap') ? parseInt(res.headers.get('X-Render-Cap'), 10) : null,
  };

  const payload = { points: floatArray, metadata };
  pointCache.set(cacheKey, payload);
  if (pointCache.size > MAX_POINT_CACHE_SIZE) {
    const oldestKey = pointCache.keys().next().value;
    pointCache.delete(oldestKey);
  }

  return payload;
}

export async function fetchMultiMethodCompare(frameId, methods = null, maxPoints = 30000) {
  const params = new URLSearchParams();
  if (methods) params.set('methods', methods.join(','));
  if (maxPoints) params.set('max_points', maxPoints);

  const res = await fetch(`${BASE_URL}/frames/${frameId}/compare?${params.toString()}`);
  if (!res.ok) throw new Error(`Multi-method comparison failed: ${res.statusText}`);
  return res.json();
}

export async function fetchFrameMetrics(frameId) {
  if (metricsCache.has(frameId)) {
    return metricsCache.get(frameId);
  }
  const res = await fetch(`${BASE_URL}/frames/${frameId}/metrics`);
  if (!res.ok) throw new Error(`Failed to fetch frame metrics: ${res.statusText}`);
  const data = await res.json();
  metricsCache.set(frameId, data);
  if (metricsCache.size > MAX_METRICS_CACHE_SIZE) {
    const oldest = metricsCache.keys().next().value;
    metricsCache.delete(oldest);
  }
  return data;
}

export async function fetchGroundTruth(frameId, classes = null, countPointsMethod = null) {
  const cacheKey = `${frameId}_${classes ? classes.join(',') : 'all'}_${countPointsMethod || 'none'}`;
  if (gtCache.has(cacheKey)) {
    return gtCache.get(cacheKey);
  }

  const params = new URLSearchParams();
  if (classes) params.set('classes', classes.join(','));
  if (countPointsMethod) params.set('count_points_method', countPointsMethod);

  const res = await fetch(`${BASE_URL}/frames/${frameId}/ground-truth?${params.toString()}`);
  if (!res.ok) throw new Error(`Failed to fetch ground truth: ${res.statusText}`);
  const data = await res.json();
  gtCache.set(cacheKey, data);
  if (gtCache.size > MAX_GT_CACHE_SIZE) {
    const oldest = gtCache.keys().next().value;
    gtCache.delete(oldest);
  }
  return data;
}

export async function fetchPredictions(frameId, method = 'v6_alpha_0.25', scoreThreshold = 0.25) {
  const params = new URLSearchParams({ method, score_threshold: scoreThreshold });
  const res = await fetch(`${BASE_URL}/frames/${frameId}/predictions?${params.toString()}`);
  if (!res.ok) throw new Error(`Failed to fetch predictions: ${res.statusText}`);
  return res.json();
}

export async function fetchGlobalMetrics() {
  const res = await fetch(`${BASE_URL}/metrics/global`);
  if (!res.ok) throw new Error(`Failed to fetch global metrics: ${res.statusText}`);
  return res.json();
}

export async function fetchDistanceMetrics(className = null, method = null) {
  const params = new URLSearchParams();
  if (className) params.set('class_name', className);
  if (method) params.set('method', method);

  const res = await fetch(`${BASE_URL}/metrics/distance?${params.toString()}`);
  if (!res.ok) throw new Error(`Failed to fetch distance metrics: ${res.statusText}`);
  return res.json();
}

export async function fetchDistributionMetrics(className = null) {
  const params = new URLSearchParams();
  if (className) params.set('class_name', className);

  const res = await fetch(`${BASE_URL}/metrics/distribution?${params.toString()}`);
  if (!res.ok) throw new Error(`Failed to fetch distribution metrics: ${res.statusText}`);
  return res.json();
}

export async function fetchDetectorMetrics() {
  const res = await fetch(`${BASE_URL}/metrics/detector`);
  if (!res.ok) throw new Error(`Failed to fetch detector metrics: ${res.statusText}`);
  return res.json();
}

