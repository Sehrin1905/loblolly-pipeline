import json
import os
from pathlib import Path
import subprocess
import tomllib

import pytest

from wood_stitch.pipeline import validate_config


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("path", [ROOT / "config.toml", *sorted((ROOT / "configs").glob("*.toml"))])
def test_committed_profiles_are_accepted(path):
    with path.open("rb") as stream:
        cfg = validate_config(tomllib.load(stream))
    assert len(cfg["pipeline"]["samples"]) == 1


@pytest.mark.parametrize("failure,expected_calls", [("sync", 1), ("device", 2), ("pipeline", 3)])
def test_batch_launch_stops_at_failure_and_propagates_status(tmp_path, failure, expected_calls):
    # Exercise the real shell script, without a scheduler, GPU, network, or R2.
    import sys

    binary = tmp_path / "uv"
    binary.write_text(
        f"#!{sys.executable}\n"
        "import json, os, sys\n"
        "args = sys.argv[1:]\n"
        "with open(os.environ['CALL_LOG'], 'a') as f:\n"
        "    f.write(json.dumps(args) + '\\n')\n"
        "stage = 'sync' if args[0] == 'sync' else "
        "('device' if 'scripts/check_device.py' in args else 'pipeline')\n"
        "sys.exit(37 if stage == os.environ['FAIL_STAGE'] else 0)\n"
    )
    binary.chmod(0o700)
    log = tmp_path / "calls.jsonl"
    env = dict(os.environ)
    env.pop("LOBLOLLY_CONFIG", None)
    env.update(
        PATH=f"{tmp_path}:{env['PATH']}",
        SLURM_SUBMIT_DIR=str(ROOT),
        CALL_LOG=str(log),
        FAIL_STAGE=failure,
    )
    result = subprocess.run(["bash", str(ROOT / "scripts/submit_pegasus.sh")], env=env, capture_output=True)
    assert result.returncode == 37
    calls = [json.loads(line) for line in log.read_text().splitlines()]
    assert len(calls) == expected_calls
    for args in calls:
        assert "--locked" in args
        assert args[args.index("--extra") + 1] == "inference"
    if failure == "pipeline":
        assert calls[-1][-2:] == ["--config", "configs/pegasus_ref636.toml"]
