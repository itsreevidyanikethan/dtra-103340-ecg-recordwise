"""Record-wise three-class benchmark: train on DS1, lock-test on DS2, then
evaluate externally on INCART/SVDB/SDDB. Record-grouped bootstrap CIs."""
import argparse
import json
import os
from collections import defaultdict

import numpy as np
import torch
import yaml
from sklearn.metrics import confusion_matrix, f1_score

from beats import extract_record, zscore_train_stats, waveform_stats
from models import BeatCNN, make_extra_trees, tree_features

CLASSES = ["Normal", "VEB", "Other"]


def gather(db_dir, records):
    xs, ys, rrs, recs = [], [], [], []
    for r in records:
        x, y, rr = extract_record(os.path.join(db_dir, str(r)))
        xs.append(x); ys.append(y); rrs.append(rr)
        recs.append(np.full(len(y), str(r)))
    return (np.concatenate(xs), np.concatenate(ys),
            np.concatenate(rrs), np.concatenate(recs))


def per_class_metrics(y, pred):
    cm = confusion_matrix(y, pred, labels=[0, 1, 2])
    out = {"confusion_matrix": cm.tolist()}
    for i, c in enumerate(CLASSES):
        tp = cm[i, i]; fn = cm[i].sum() - tp; fp = cm[:, i].sum() - tp
        out[c] = {"support": int(cm[i].sum()),
                  "sensitivity": float(tp / (tp + fn)) if tp + fn else None,
                  "ppv": float(tp / (tp + fp)) if tp + fp else None,
                  "f1": float(2 * tp / (2 * tp + fn + fp)) if 2 * tp + fn + fp else 0.0}
    out["macro_f1"] = float(f1_score(y, pred, labels=[0, 1, 2], average="macro", zero_division=0))
    return out


def record_bootstrap(y, pred, records, n_rep=2000, seed=1):
    rng = np.random.default_rng(seed)
    uniq = np.unique(records)
    idx = {r: np.where(records == r)[0] for r in uniq}
    f1s = []
    for _ in range(n_rep):
        rows = np.concatenate([idx[r] for r in rng.choice(uniq, len(uniq), replace=True)])
        f1s.append(f1_score(y[rows], pred[rows], labels=[0, 1, 2], average="macro", zero_division=0))
    lo, hi = np.percentile(f1s, [2.5, 97.5])
    return [float(lo), float(hi)]


def train_cnn(x, y, rr, seed, cfg):
    torch.manual_seed(seed); np.random.seed(seed)
    model = BeatCNN(rr_branch=cfg["rr_branch"])
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    crit = torch.nn.CrossEntropyLoss()
    xt = torch.tensor(x); yt = torch.tensor(y); rrt = torch.tensor(np.nan_to_num(rr))
    n = len(y)
    for epoch in range(cfg["max_epochs"]):
        order = np.random.permutation(n)
        for i in range(0, n, cfg["batch_size"]):
            j = order[i:i + cfg["batch_size"]]
            opt.zero_grad()
            crit(model(xt[j], rrt[j]), yt[j]).backward()
            opt.step()
    return model


@torch.no_grad()
def cnn_predict(model, x, rr, batch=1024):
    model.eval()
    out = []
    rrt = torch.tensor(np.nan_to_num(rr))
    xt = torch.tensor(x)
    for i in range(0, len(x), batch):
        out.append(model(xt[i:i + batch], rrt[i:i + batch]).argmax(1).numpy())
    return np.concatenate(out)


def main(cfg_path="configs/protocol.yaml", data_root="data", out="outputs"):
    os.makedirs(out, exist_ok=True)
    cfg = yaml.safe_load(open(cfg_path))
    tr = cfg["split"]["train_records"]; te = cfg["split"]["test_records"]
    x_tr, y_tr, rr_tr, _ = gather(f"{data_root}/mitdb", tr)
    x_te, y_te, rr_te, rec_te = gather(f"{data_root}/mitdb", te)
    mean, std = zscore_train_stats(x_tr)
    x_tr = (x_tr - mean) / std
    x_te = (x_te - mean) / std

    json.dump({"mean": mean, "std": std}, open(f"{out}/train_stats.json", "w"))
    results = {"class_counts_train": np.bincount(y_tr, minlength=3).tolist(),
               "class_counts_test": np.bincount(y_te, minlength=3).tolist()}

    # CNN across registered seeds
    cnn_cfg = cfg["models"]["cnn"]
    for seed in cnn_cfg["seeds"]:
        model = train_cnn(x_tr, y_tr, rr_tr, seed, cnn_cfg)
        pred = cnn_predict(model, x_te, rr_te)
        results[f"cnn_seed{seed}"] = per_class_metrics(y_te, pred)
        results[f"cnn_seed{seed}"]["macro_f1_ci95_record_bootstrap"] = \
            record_bootstrap(y_te, pred, rec_te,
                             cfg["evaluation"]["bootstrap"]["replicates"])
        torch.save(model.state_dict(), f"{out}/cnn_seed{seed}.pt")

    # Extra Trees on RR + waveform statistics
    et = make_extra_trees(cfg["models"]["extra_trees"]["n_estimators"],
                          cfg["models"]["extra_trees"]["seed"])
    et.fit(tree_features(rr_tr, waveform_stats(x_tr)), y_tr)
    pred = et.predict(tree_features(rr_te, waveform_stats(x_te)))
    results["extra_trees"] = per_class_metrics(y_te, pred)
    results["extra_trees"]["macro_f1_ci95_record_bootstrap"] = \
        record_bootstrap(y_te, pred, rec_te,
                         cfg["evaluation"]["bootstrap"]["replicates"])

    json.dump(results, open(f"{out}/mitdb_results.json", "w"), indent=2)
    print(json.dumps({k: v for k, v in results.items() if "class_counts" in k}, indent=2))
    print("done; external evaluation: run src/external_eval.py per database")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="data")
    a = ap.parse_args()
    main(data_root=a.data_root)
