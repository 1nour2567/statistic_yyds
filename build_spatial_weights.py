"""
Build different spatial weight matrices for robustness tests.

This script creates:
1. Contiguity (adjacency) matrix
2. KNN (k=5) matrix
3. Inverse distance matrix (already provided)

Input files:
    data_processed/fujian_county_centroids_for_sdm.csv

Outputs:
    data_processed/fujian_spatial_weights_contiguity_matrix.csv
    data_processed/fujian_spatial_weights_knn5_matrix.csv
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path("/workspace")
DATA = ROOT / "data_processed"


def haversine(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    """Calculate the great circle distance between two points on the earth."""
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    r = 6371.0088  # Radius of earth in km
    return c * r


def build_contiguity_matrix(centroids: pd.DataFrame) -> np.ndarray:
    """Build contiguity matrix based on spatial proximity."""
    # For simplicity, we'll define contiguity based on distance threshold
    # Counties within 50 km are considered contiguous
    n = len(centroids)
    W = np.zeros((n, n))
    
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            dist = haversine(
                centroids.loc[i, "lon"], centroids.loc[i, "lat"],
                centroids.loc[j, "lon"], centroids.loc[j, "lat"]
            )
            if dist < 50:  # 50 km threshold
                W[i, j] = 1
    
    # Row-standardize
    row_sums = W.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1  # Avoid division by zero
    W = W / row_sums
    
    return W


def build_knn_matrix(centroids: pd.DataFrame, k: int = 5) -> np.ndarray:
    """Build KNN matrix."""
    n = len(centroids)
    W = np.zeros((n, n))
    
    for i in range(n):
        distances = []
        for j in range(n):
            if i == j:
                distances.append((float('inf'), j))
            else:
                dist = haversine(
                    centroids.loc[i, "lon"], centroids.loc[i, "lat"],
                    centroids.loc[j, "lon"], centroids.loc[j, "lat"]
                )
                distances.append((dist, j))
        
        # Sort by distance and select top k
        distances.sort()
        for _, j in distances[:k]:
            W[i, j] = 1
    
    # Row-standardize
    row_sums = W.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1  # Avoid division by zero
    W = W / row_sums
    
    return W


def main():
    # Load centroids data
    centroids = pd.read_csv(DATA / "fujian_county_centroids_for_sdm.csv")
    units = centroids["unit_id"].astype(str).tolist()
    
    # Build contiguity matrix
    print("Building contiguity matrix...")
    W_contiguity = build_contiguity_matrix(centroids)
    w_contiguity_df = pd.DataFrame(W_contiguity, index=units, columns=units)
    w_contiguity_df.to_csv(DATA / "fujian_spatial_weights_contiguity_matrix.csv", encoding="utf-8-sig")
    print(f"Contiguity matrix saved to {DATA / 'fujian_spatial_weights_contiguity_matrix.csv'}")
    
    # Build KNN matrix (k=5)
    print("Building KNN matrix (k=5)...")
    W_knn = build_knn_matrix(centroids, k=5)
    w_knn_df = pd.DataFrame(W_knn, index=units, columns=units)
    w_knn_df.to_csv(DATA / "fujian_spatial_weights_knn5_matrix.csv", encoding="utf-8-sig")
    print(f"KNN matrix saved to {DATA / 'fujian_spatial_weights_knn5_matrix.csv'}")
    
    print("All spatial weight matrices built successfully!")


if __name__ == "__main__":
    main()
