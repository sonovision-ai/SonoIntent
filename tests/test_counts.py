"""Paper-number invariants for the public SonoIntent-Annotations package.

Set SONOINTENT_ANNOTATIONS to the package path to run against real data;
otherwise the test skips. (The package ships its own standalone verifier.)
"""
import os
import pytest

from sonointent.data.annotations import load_split

EXPECTED = {"train": 102248, "val": 5810, "test": 22870, "target_absent": 656}

def test_counts():
    repo = os.environ.get("SONOINTENT_ANNOTATIONS")
    if not repo or not os.path.exists(os.path.join(repo, "annotations", "train.jsonl")):
        pytest.skip("set SONOINTENT_ANNOTATIONS to a SonoIntent-Annotations package")
    for sp, n in EXPECTED.items():
        assert len(load_split(sp, repo)) == n, sp
    assert len({r["public_id"] for sp in EXPECTED for r in load_split(sp, repo)}) == 131584
