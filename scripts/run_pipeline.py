"""Run with: uv run --locked --extra inference python scripts/run_pipeline.py --config config.toml"""

from wood_stitch.pipeline import main

if __name__ == "__main__":
    raise SystemExit(main())
