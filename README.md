# DTRA-103340 — Record-Wise Three-Class ECG Benchmark: Reproduction Package

## Verification and artifact status

This repository currently contains source code and a protocol configuration.
The original experiment checkpoints, row-level manifests and prediction archives
are not included. Synthetic smoke tests check software execution only; they do
not reproduce or validate the manuscript's numerical results. See
[REPRODUCIBILITY.md](REPRODUCIBILITY.md) for the remaining evidence and code gaps.

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
for db in mitdb incartdb svdb sddb; do python src/download.py --db $db; done
```

## Reproduce
```bash
python src/experiment.py                     # DS1 train, DS2 locked test
python src/external_eval.py --db incartdb
python src/external_eval.py --db svdb
python src/external_eval.py --db sddb
python tests/smoke_test.py                   # fast synthetic check
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

## Check this code release

Run these commands from the repository root, after installing the dependencies:

```bash
python tests/smoke_test.py
python tests/test_checksums.py
python make_checksums.py --verify SOURCE_CHECKSUMS.txt
```

`SOURCE_CHECKSUMS.txt` covers the distributed source and validation records.
`validation/validation.json` records the actual synthetic checks performed for
this code update and their software versions; it is not the original study
environment. The `python` commands intentionally omit isolated mode (`-I`),
because training/evaluation scripts import neighbouring modules under `src/`.

Once authentic run outputs have been recovered and reviewed, verify their bytes
separately:

```bash
python make_checksums.py outputs --output CHECKSUMS.txt
python make_checksums.py --verify CHECKSUMS.txt
```

The generator fails for a missing or empty artifact directory. Do not describe a
source checksum manifest as verification of unavailable experiment outputs.
Keep newly generated results labelled as new runs until their provenance and
agreement with the manuscript have been assessed. Large checkpoints should be
archived separately with an immutable identifier and checksums. The `.gitignore`
excludes local datasets and model outputs from accidental source commits.
