
import numpy as np
from scipy.spatial import cKDTree


def compute_features(xyz, k=20):
    """
    Compute local surface variation using batched covariance
    matrices and eigenvalue decomposition.
    Returns geometric complexity scores.
    """
    xyz = np.asarray(xyz, dtype=np.float64)
    n = len(xyz)

    if n == 0:
        return np.zeros(0, dtype=np.float32)

    k_eff = min(k, n - 1)
    if k_eff < 1:
        return np.zeros(n, dtype=np.float32)

    tree = cKDTree(xyz)
    _, indices = tree.query(xyz, k=k_eff + 1, workers=-1)

    # Exclude the point itself
    neighbors = xyz[indices[:, 1:]]

    # Center each neighbourhood
    centered = neighbors - neighbors.mean(axis=1, keepdims=True)

    # Batch covariance matrices: (n, 3, 3)
    covariance = np.einsum(
        "nki,nkj->nij", centered, centered, optimize=True
    ) / k_eff

    # Batched eigenvalues: (n, 3)
    eigenvalues = np.linalg.eigvalsh(covariance)
    eigenvalues = np.maximum(eigenvalues, 0.0)

    total = eigenvalues.sum(axis=1)
    complexity = np.divide(
        eigenvalues[:, 0],
        total,
        out=np.zeros(n, dtype=np.float64),
        where=total > 1e-12
    )

    return complexity.astype(np.float32)
