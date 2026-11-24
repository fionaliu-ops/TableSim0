"""Conservative height-clipped bounds for convex collision meshes."""
import numpy as np


def height_band_bounds(vertices, low, high):
    points = np.asarray(vertices, dtype=float)
    inside = points[(points[:, 2] >= low) & (points[:, 2] <= high)]
    sections = [inside]
    # Every hull edge is included among these segments. Using all vertex
    # pairs adds interior points but cannot enlarge the true convex slice.
    i, j = np.triu_indices(len(points), 1)
    a, b = points[i], points[j]
    dz = b[:, 2] - a[:, 2]
    for height in (low, high):
        valid = (np.abs(dz) > 1e-12) & ((a[:, 2] - height) * (b[:, 2] - height) <= 0)
        alpha = (height - a[valid, 2]) / dz[valid]
        sections.append(a[valid] + alpha[:, None] * (b[valid] - a[valid]))
    section = np.concatenate(sections)
    if len(section) == 0:
        return None
    lower, upper = section[:, :2].min(axis=0), section[:, :2].max(axis=0)
    return (lower + upper) / 2, upper - lower
