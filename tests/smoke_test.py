"""Synthetic smoke test: models, metrics and bootstrap without PhysioNet data."""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from models import BeatCNN, make_extra_trees, tree_features  # noqa: E402
from beats import waveform_stats, beat_class, MAPPING  # noqa: E402
from experiment import per_class_metrics, record_bootstrap  # noqa: E402


def main():
    # label mapping covers all declared symbols and nothing else maps
    assert beat_class("N") == 0 and beat_class("V") == 1 and beat_class("A") == 2
    assert beat_class("+") is None
    rng = np.random.default_rng(0)
    x = rng.normal(size=(64, 256)).astype(np.float32)
    rr = rng.random((64, 3)).astype(np.float32)
    y = rng.integers(0, 3, 64)
    for rrb in (False, True):
        m = BeatCNN(rr_branch=rrb)
        out = m(torch.tensor(x), torch.tensor(rr))
        assert out.shape == (64, 3)
    et = make_extra_trees(n_estimators=10)
    feats = tree_features(rr, waveform_stats(x))
    et.fit(feats, y)
    pred = et.predict(feats)
    res = per_class_metrics(y, pred)
    assert "macro_f1" in res
    recs = np.array([f"r{i//8}" for i in range(64)])
    ci = record_bootstrap(y, pred, recs, n_rep=20)
    assert len(ci) == 2
    print("SMOKE TEST PASSED")


if __name__ == "__main__":
    main()
