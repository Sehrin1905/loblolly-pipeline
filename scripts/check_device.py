"""Check actual device execution before processing a dataset (no model download)."""

import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", required=True, choices=["cpu", "mps", "cuda"])
    args = parser.parse_args()
    import torch

    x = torch.ones((32, 32), device=args.device)
    assert torch.allclose((x @ x).cpu(), torch.full((32, 32), 32.0))
    print(
        f"Executed tensor smoke test: torch={torch.__version__}, device={x.device}, CUDA runtime={torch.version.cuda}"
    )
