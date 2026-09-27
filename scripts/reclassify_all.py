"""Legacy batch rewriting is disabled: use a fresh fingerprinted pipeline run."""

import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    parser.error("Use scripts/run_pipeline.py with a new run. Existing completed artifacts are immutable.")
