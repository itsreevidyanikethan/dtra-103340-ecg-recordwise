# DTRA-103340 — Record-Wise Three-Class ECG Benchmark: Reproduction Package

Code skeleton for the leakage-controlled, record-independent benchmark of
three-class ECG beat classification (Normal / VEB / Other) with internal
MIT-BIH evaluation and external evaluation on INCART, SVDB and SDDB
(Digital Technologies Research and Applications, Article ID 103340).

**Status:** the structural protocol (record-wise De Chazal split, three-class
mapping with its AAMI reference table, 256-point beat windows, CNN vs Extra
Trees comparators including the matched-input RR-branch ablation requested in
review, record-grouped bootstrap) is implemented. Numeric hyperparameters in
`configs/protocol.yaml` are marked for locking against the revised manuscript
before results are published from this package.

## Datasets (PhysioNet; not redistributed)
```bash
pip install -r requirements.txt
for db in mitdb incartdb svdb sddb; do python -I src/download.py --db $db; done
```

## Reproduce
```bash
python -I src/experiment.py                     # DS1 train, DS2 locked test
python -I src/external_eval.py --db incartdb
python -I src/external_eval.py --db svdb
python -I src/external_eval.py --db sddb
python -I tests/smoke_test.py                   # fast synthetic check
```

## Review-driven design notes
- `classes.mapping` lists every beat symbol in the custom "Other" class, with
  the standard AAMI EC57 superclasses alongside (Reviewers 1, 3, 4).
- Records 201/202 share a subject: co-assign them in the config, or report the
  split as record-disjoint rather than patient-independent (Reviewer 4).
- `models.cnn.rr_branch: true` runs the matched-input ablation in which the
  CNN also receives pre-/post-RR intervals (Reviewers 1, 2, 3).
- Beat locations come from reference annotations, not end-to-end detection.
- External runs record the exact lead used per record and report
  record-grouped bootstrap CIs; the INCART Holter caveat is printed with them.

## License
Code: MIT (see LICENSE). The PhysioNet databases remain under their own
licences and must be downloaded from PhysioNet directly.
