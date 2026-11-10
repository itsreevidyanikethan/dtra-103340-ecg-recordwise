"""External evaluation on INCART / SVDB / SDDB with frozen MIT-BIH models.

Reports the exact lead used per record, per-class supports, confusion matrices
and record-grouped bootstrap CIs. For INCART, note that several recordings may
derive from fewer Holter subjects; record-level resampling is reported, with a
caveat, exactly as the review requested.
"""
import argparse
import json
import os

import numpy as np
import torch
import wfdb
import yaml

from beats import extract_record, pick_lead, waveform_stats
from experiment import CLASSES, cnn_predict, per_class_metrics, record_bootstrap
from models import BeatCNN, make_extra_trees, tree_features


def main(db, data_root="data", out="outputs", cfg_path="configs/protocol.yaml"):
    cfg = yaml.safe_load(open(cfg_path))
    db_dir = f"{data_root}/{db}"
    records = [r for r in sorted(os.listdir(db_dir)) if r.endswith(".hea")]
    records = [r[:-4] for r in records]
    xs, ys, rrs, recs, leads = [], [], [], [], {}
    for r in records:
        try:
            rec = wfdb.rdrecord(os.path.join(db_dir, r))
            leads[r] = rec.sig_name[pick_lead(rec)]
            x, y, rr = extract_record(os.path.join(db_dir, r))
        except Exception as e:
            print(f"skipping {r}: {e}")
            continue
        if len(y) == 0:
            continue
        xs.append(x); ys.append(y); rrs.append(rr)
        recs.append(np.full(len(y), r))
    x = np.concatenate(xs); y = np.concatenate(ys)
    rr = np.concatenate(rrs); rec_ids = np.concatenate(recs)

    stats = json.load(open(f"{out}/train_stats.json"))
    x = (x - stats["mean"]) / stats["std"]

    results = {"database": db, "usable_records": len(set(rec_ids)),
               "lead_used_per_record": leads,
               "class_supports": np.bincount(y, minlength=3).tolist()}
    for seed in cfg["models"]["cnn"]["seeds"]:
        m = BeatCNN(rr_branch=cfg["models"]["cnn"]["rr_branch"])
        m.load_state_dict(torch.load(f"{out}/cnn_seed{seed}.pt"))
        pred = cnn_predict(m, x.astype(np.float32), rr)
        results[f"cnn_seed{seed}"] = per_class_metrics(y, pred)
        results[f"cnn_seed{seed}"]["macro_f1_ci95_record_bootstrap"] = \
            record_bootstrap(y, pred, rec_ids,
                             cfg["evaluation"]["bootstrap"]["replicates"])
    json.dump(results, open(f"{out}/{db}_results.json", "w"), indent=2)
    print(f"wrote {out}/{db}_results.json")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", choices=["incartdb", "svdb", "sddb"], required=True)
    ap.add_argument("--data-root", default="data")
    a = ap.parse_args()
    main(a.db, a.data_root)
