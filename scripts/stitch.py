# scripts/stitch.py
import cv2
import numpy as np
from pathlib import Path
from itertools import combinations

def load_tiles(tile_dir: str) -> list[tuple[str, np.ndarray]]:
    """Load all tile images from a directory."""
    tile_dir = Path(tile_dir)
    tiles = []
    for ext in ("*.tif", "*.tiff", "*.png", "*.jpg"):
        for path in sorted(tile_dir.glob(ext)):
            img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                tiles.append((path.name, img))
    return tiles


def detect_and_match(img_a: np.ndarray, img_b: np.ndarray,
                     ratio: float = 0.75, min_matches: int = 10
                     ) -> tuple[np.ndarray | None, int]:
    """
    SIFT keypoints + Lowe ratio test + RANSAC homography.
    Returns (H_a_to_b, n_inliers) or (None, 0) if matching fails.
    """
    sift = cv2.SIFT_create()
    kp_a, des_a = sift.detectAndCompute(img_a, None)
    kp_b, des_b = sift.detectAndCompute(img_b, None)

    if des_a is None or des_b is None or len(kp_a) < min_matches:
        return None, 0

    # KNN match, Lowe ratio test
    bf = cv2.BFMatcher(cv2.NORM_L2)
    raw = bf.knnMatch(des_a, des_b, k=2)
    good = [m for m, n in raw if m.distance < ratio * n.distance]

    if len(good) < min_matches:
        return None, 0

    pts_a = np.float32([kp_a[m.queryIdx].pt for m in good])
    pts_b = np.float32([kp_b[m.trainIdx].pt for m in good])

    H, mask = cv2.findHomography(pts_a, pts_b,
                                 cv2.RANSAC,
                                 ransacReprojThreshold=4.0,
                                 confidence=0.995)
    n_inliers = int(mask.sum()) if mask is not None else 0
    return H, n_inliers


def build_pairwise_graph(tiles: list[tuple[str, np.ndarray]],
                         min_inliers: int = 15
                         ) -> dict[tuple[int,int], np.ndarray]:
    """
    Try all pairs; keep edges where RANSAC gives enough inliers.
    Only test pairs that are plausibly adjacent (overlap budget).
    """
    edges = {}
    for (i, (name_a, img_a)), (j, (name_b, img_b)) in combinations(enumerate(tiles), 2):
        H, n = detect_and_match(img_a, img_b)
        if H is not None and n >= min_inliers:
            edges[(i, j)] = H
            print(f"  {name_a} → {name_b}: {n} inliers")
    return edges


def chain_homographies(n_tiles: int,
                       edges: dict[tuple[int,int], np.ndarray]
                       ) -> list[np.ndarray | None]:
    """
    BFS from tile 0 to build each tile's transform into a global canvas frame.
    """
    import collections
    H_global = [None] * n_tiles
    H_global[0] = np.eye(3, dtype=np.float64)

    # Build adjacency list
    adj = {i: [] for i in range(n_tiles)}
    for (i, j), H in edges.items():
        adj[i].append((j, H))
        adj[j].append((i, np.linalg.inv(H)))

    queue = collections.deque([0])
    while queue:
        src = queue.popleft()
        for dst, H_src_to_dst in adj[src]:
            if H_global[dst] is None:
                H_global[dst] = H_global[src] @ np.linalg.inv(H_src_to_dst)
                queue.append(dst)

    return H_global


def composite(tiles: list[tuple[str, np.ndarray]],
              H_global: list[np.ndarray | None]) -> np.ndarray:
    """
    Warp all tiles to the global canvas, average-blend overlaps.
    """
    h0, w0 = tiles[0][1].shape[:2]

    # ── 1. Find canvas bounds ───────────────────────────────────────────────
    corners_all = []
    for idx, (_, img) in enumerate(tiles):
        if H_global[idx] is None:
            continue
        h, w = img.shape[:2]
        corners = np.float32([[0,0],[w,0],[w,h],[0,h]]).reshape(-1,1,2)
        warped  = cv2.perspectiveTransform(corners, H_global[idx])
        corners_all.append(warped)

    all_pts  = np.concatenate(corners_all, axis=0)
    x_min    = int(np.floor(all_pts[:,:,0].min()))
    y_min    = int(np.floor(all_pts[:,:,1].min()))
    x_max    = int(np.ceil (all_pts[:,:,0].max()))
    y_max    = int(np.ceil (all_pts[:,:,1].max()))

    offset   = np.array([[1,0,-x_min],[0,1,-y_min],[0,0,1]], dtype=np.float64)
    cw, ch   = x_max - x_min, y_max - y_min

    # ── 2. Accumulate weighted warp ─────────────────────────────────────────
    canvas   = np.zeros((ch, cw), dtype=np.float64)
    weight   = np.zeros((ch, cw), dtype=np.float64)

    for idx, (_, img) in enumerate(tiles):
        if H_global[idx] is None:
            continue
        H = offset @ H_global[idx]
        warped = cv2.warpPerspective(img.astype(np.float64), H, (cw, ch))
        mask   = cv2.warpPerspective(np.ones_like(img, dtype=np.float64), H, (cw, ch))
        canvas += warped * mask
        weight += mask

    # avoid div/0
    valid  = weight > 0
    result = np.zeros_like(canvas, dtype=np.uint8)
    result[valid] = np.clip(canvas[valid] / weight[valid], 0, 255).astype(np.uint8)
    return result


def stitch(tile_dir: str, out_path: str = "mosaic.tif") -> np.ndarray:
    print("Loading tiles …")
    tiles = load_tiles(tile_dir)
    print(f"  {len(tiles)} tiles found")

    print("Building pairwise matches …")
    edges = build_pairwise_graph(tiles)

    print("Chaining homographies …")
    H_global = chain_homographies(len(tiles), edges)

    unregistered = [tiles[i][0] for i, H in enumerate(H_global) if H is None]
    if unregistered:
        print(f"  WARNING: could not register {unregistered}")

    print("Compositing …")
    mosaic = composite(tiles, H_global)

    cv2.imwrite(out_path, mosaic)
    print(f"  Saved → {out_path}  ({mosaic.shape[1]}×{mosaic.shape[0]} px)")
    return mosaic


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("tile_dir")
    ap.add_argument("--out", default="mosaic.tif")
    args = ap.parse_args()
    stitch(args.tile_dir, args.out)
    