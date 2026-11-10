"""The two comparators: a 1-D CNN on raw 256-point beats (optionally with an
RR branch for the matched-input ablation requested in review) and Extra Trees
on RR + time-domain waveform statistics."""
import numpy as np
import torch
import torch.nn as nn
from sklearn.ensemble import ExtraTreesClassifier


class BeatCNN(nn.Module):
    def __init__(self, n_classes=3, rr_branch=False):
        super().__init__()
        self.rr_branch = rr_branch

        def block(cin, cout):
            return nn.Sequential(nn.Conv1d(cin, cout, 7, padding=3),
                                 nn.BatchNorm1d(cout), nn.ReLU(), nn.MaxPool1d(2))

        self.features = nn.Sequential(block(1, 16), block(16, 32), block(32, 64))
        self.pool = nn.AdaptiveAvgPool1d(1)
        feat_dim = 64 + (3 if rr_branch else 0)
        self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(feat_dim, n_classes))

    def forward(self, x, rr=None):
        z = self.pool(self.features(x.unsqueeze(1))).flatten(1)
        if self.rr_branch:
            z = torch.cat([z, rr], dim=1)
        return self.head(z)


def make_extra_trees(n_estimators=500, seed=1):
    return ExtraTreesClassifier(n_estimators=n_estimators, random_state=seed,
                                n_jobs=-1, class_weight=None)


def tree_features(rr, wave_stats):
    rr = np.nan_to_num(rr, nan=0.0)
    return np.column_stack([rr, wave_stats])
