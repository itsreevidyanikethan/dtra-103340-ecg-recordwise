# DTRA-103340 — Record-grouped ECG beat classification

The supplied package labelled its numerical settings as placeholders and no
final manuscript or original experiment archive was supplied. This repository
therefore implements a **new reconstruction benchmark**. It must not be presented
as reproduction of unavailable original results.

## New protocol

`configs/reconstruction.yaml` defines a separate protocol. Records 201 and 202
are kept together in development because they share a subject. Development is
split into training and validation groups before fitting; the test records do
not select models. Scalar normalisation statistics are fitted only on training
beats. CNN checkpoints use validation macro F1 and early stopping. Both waveform
and waveform-plus-RR variants run with three seeds, alongside saved Extra Trees.
RR intervals use beat annotations only. Missing RR values are zero-imputed.

This uses reference annotation locations, including the following beat for the
post-RR feature. It is offline beat classification, not causal real-time beat
detection, heart-attack prediction or survival analysis.

## Run

```bash
pip install -r requirements.txt
python src/download.py --db mitdb
python src/reconstruct.py --threads 2
python src/download.py --db svdb
python src/download.py --db incartdb
python src/download.py --db sddb
python src/reconstruct.py --external svdb --threads 2
python src/reconstruct.py --external incartdb --threads 2
python src/reconstruct.py --external sddb --threads 2
python make_checksums.py outputs --output CHECKSUMS.txt
python make_checksums.py --verify CHECKSUMS.txt
```

The downloader pins PhysioNet version 1.0.0 and verifies each downloaded file
against its official SHA-256 manifest. For SDDB it includes ONLY records with
audited `.atr` labels; unaudited `.ari` annotations are not substituted.
Download provenance lists all excluded records.

Training snapshots permit interruption and resumption in the same environment.
Keep source/config/data unchanged when resuming. A completed internal run writes
`mitdb_complete.json`; external evaluation is a separate explicit stage.

## Outputs and limits

`outputs/` contains the split manifest, training statistics, model checkpoints,
validation histories, record audits, saved labels/probabilities/sample positions,
and group-bootstrap replicates. A freeze manifest protects selected models from
changes during evaluation. Record clusters on external sets are not necessarily
independent patients: INCART has multiple recordings per Holter subject.

A complete new internal and external run was executed on 11 November 2026. See [REPRODUCIBILITY.md](REPRODUCIBILITY.md) for all seven models and limitations. The old
`configs/protocol.yaml` and `src/experiment.py` are retained as supplied-protocol
references, not recommended as the completed reconstruction entry point.

## Tests and licences

```bash
python tests/smoke_test.py
python tests/test_reconstruction.py
python tests/test_checksums.py
python make_checksums.py --verify SOURCE_CHECKSUMS.txt
```

Use wfdb >=4.3.1 with NumPy 2; an older version failed to read real annotations
in this environment. Tests cover non-beat RR exclusion, whole-group bootstrap
equivalence, grouped validation, saved models and freeze tamper detection.
Code: MIT. Obtain data from PhysioNet and follow each database's licence and
attribution requirements; raw signals are not included in this repository.

Official data sources: [MIT-BIH](https://physionet.org/content/mitdb/1.0.0/), [SVDB](https://physionet.org/content/svdb/1.0.0/), [INCART](https://physionet.org/content/incartdb/1.0.0/), [SDDB](https://physionet.org/content/sddb/1.0.0/).
