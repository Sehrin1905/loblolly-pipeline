# src/wood_stitch/deconvolve.py
import numpy as np
import cv2

# No published stain vectors exist for safranin/astra blue — 
# we have to estimate them from the image using Macenko method.

def rgb_to_od(img: np.ndarray, beta: float = 0.15) -> np.ndarray:
    """
    Convert RGB image to optical density (OD) space.
    Clips to avoid log(0).
    """
    img = img.astype(np.float64) / 255.0
    img = np.clip(img, 1e-6, 1.0)
    od = -np.log(img)
    # Remove background pixels (low OD = bright = no stain)
    od[np.all(od < beta, axis=2)] = 0
    return od


def estimate_stain_vectors_macenko(od: np.ndarray,
                                   alpha: float = 1.0,
                                   beta: float = 0.15) -> np.ndarray:
    """
    Estimate two stain vectors from OD image using Macenko PCA method.
    Returns (2, 3) array: [safranin_vector, astra_blue_vector].
    """
    # Flatten to (N, 3), keep only pixels with enough stain
    od_flat = od.reshape(-1, 3)
    od_flat = od_flat[np.all(od_flat > beta, axis=1)]

    if len(od_flat) < 10:
        raise ValueError("Not enough stained pixels found — check your image.")

    # PCA to find the 2D plane of max variance
    cov = np.cov(od_flat.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    # Take top 2 eigenvectors
    top2 = eigvecs[:, np.argsort(eigvals)[::-1][:2]]

    # Project onto the 2D plane
    proj = od_flat @ top2

    # Find the extreme angles in the plane
    angles = np.arctan2(proj[:, 1], proj[:, 0])
    phi1 = np.percentile(angles, alpha)
    phi2 = np.percentile(angles, 100 - alpha)

    # Convert angles back to 3D OD vectors
    v1 = top2 @ np.array([np.cos(phi1), np.sin(phi1)])
    v2 = top2 @ np.array([np.cos(phi2), np.sin(phi2)])

    # Normalize
    v1 = v1 / np.linalg.norm(v1)
    v2 = v2 / np.linalg.norm(v2)

    # Disambiguate: safranin absorbs more in blue channel (index 2)
    # astra blue absorbs more in red channel (index 0)
    if v1[2] < v2[2]:
        safranin, astra_blue = v1, v2
    else:
        safranin, astra_blue = v2, v1

    return np.array([safranin, astra_blue])


def build_unmixing_matrix(stain_vectors: np.ndarray) -> np.ndarray:
    """
    Build a 3x3 invertible unmixing matrix from 2 stain vectors.
    Adds an orthogonal residual row so the matrix is invertible.
    """
    s1 = stain_vectors[0]
    s2 = stain_vectors[1]
    # Residual = cross product of the two stain vectors
    residual = np.cross(s1, s2)
    residual = residual / np.linalg.norm(residual)
    stain_matrix = np.array([s1, s2, residual])
    return np.linalg.inv(stain_matrix.T)


def separate_stains(od: np.ndarray,
                    unmixing_matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Apply unmixing matrix to OD image.
    Returns (safranin, astra_blue, residual) concentration maps.
    """
    h, w, _ = od.shape
    od_flat = od.reshape(-1, 3)
    concentrations = (unmixing_matrix @ od_flat.T).T
    concentrations = np.clip(concentrations, 0, None)
    concentrations = concentrations.reshape(h, w, 3)
    safranin  = concentrations[:, :, 0]
    astra_blue = concentrations[:, :, 1]
    residual  = concentrations[:, :, 2]
    return safranin, astra_blue, residual


def deconvolve(img: np.ndarray) -> dict[str, np.ndarray]:
    """
    Full pipeline: RGB image → separated stain concentration maps.
    Returns dict with keys: 'od', 'safranin', 'astra_blue', 'residual', 'stain_vectors'
    """
    print("  Converting to optical density ...")
    od = rgb_to_od(img)

    print("  Estimating stain vectors (Macenko) ...")
    stain_vectors = estimate_stain_vectors_macenko(od)
    print(f"    Safranin vector:   {stain_vectors[0].round(4)}")
    print(f"    Astra blue vector: {stain_vectors[1].round(4)}")

    print("  Building unmixing matrix ...")
    unmixing_matrix = build_unmixing_matrix(stain_vectors)

    print("  Separating stains ...")
    safranin, astra_blue, residual = separate_stains(od, unmixing_matrix)

    return {
        "od": od,
        "safranin": safranin,
        "astra_blue": astra_blue,
        "residual": residual,
        "stain_vectors": stain_vectors
    }


if __name__ == "__main__":
    import sys

    mosaic_path = sys.argv[1] if len(sys.argv) > 1 else "data/mosaic.tif"

    print(f"Loading {mosaic_path} ...")
    img = cv2.imread(mosaic_path)

    print("Running color deconvolution ...")
    results = deconvolve(img)

    # Save each channel as a normalized grayscale image
    for key in ("safranin", "astra_blue", "residual"):
        channel = results[key]
        normalized = cv2.normalize(channel, None, 0, 255, cv2.NORM_MINMAX)
        out_path = f"data/{key}.png"
        cv2.imwrite(out_path, normalized.astype(np.uint8))
        print(f"  Saved → {out_path}")

    print("Done!")