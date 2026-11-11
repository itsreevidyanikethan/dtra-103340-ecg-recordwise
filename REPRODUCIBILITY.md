# Reproducibility status — 11 November 2026

No final ECG manuscript or original experiment artifacts were supplied. The
original protocol file explicitly used placeholder settings. This is a new
benchmark implementation and cannot establish original manuscript agreement.

## Changes and checks

The new protocol co-assigns MIT-BIH records 201 and 202 to development because
they belong to the same known subject. Training/validation splitting is grouped;
held-out test records are not used for checkpoint selection. This differs from
the supplied record-only partition and must be reported as a new experiment.

Both CNN input variants use validation macro F1, early stopping, three seeds
and saved optimizer/random-state snapshots. Extra Trees is fitted on the same
training portion and saved. The scalar waveform normalisation statistics use
training beats only. The RR extractor ignores non-beat annotations and records
boundary/non-finite exclusions. Predictions include reference sample positions.

Regression tests cover the RR correction, group-bootstrap equivalence and a
small complete pipeline with freezing and tamper detection. Real MIT-BIH
annotations initially exposed an incompatibility between wfdb 4.1.2 and NumPy 2;
wfdb was upgraded to 4.3.1 and real-data loading succeeded.

## External evaluation constraints

The downloader verifies version-1.0.0 file hashes against PhysioNet's published
SHA-256 list. SDDB has audited `.atr` annotations for only 12 records:
30, 31, 32, 34, 35, 36, 41, 45, 46, 49, 51 and 52. Other records are explicitly
excluded; unaudited `.ari` annotations are not accepted as ground truth.

External bootstrap units are records, not verified independent patients.
INCART contains 75 recordings from 32 Holter subjects, so a record-level
interval does not fully address patient dependence. Reference beat locations
and future RR information are used; this is offline classification. There is
no heart-attack, survival or deployment validation.

## Actual execution

All six CNNs and the Extra Trees comparator were fitted. The validation-selected
models were frozen and evaluated on the held-out MIT-BIH records and on SVDB,
INCART and the 12 audited-label SDDB records. Per-beat predictions, model weights,
validation histories, data audits and 2,000 group-bootstrap draws per model and
database are saved. Recalculation from saved predictions verified the metrics,
confusion matrices and bootstrap intervals; model hashes still match the freeze.

| Model | MIT-BIH macro F1 | SVDB macro F1 | INCART macro F1 | SDDB macro F1 |
|---|---:|---:|---:|---:|
| waveform_seed1 | 0.589083 | 0.510665 | 0.323642 | 0.366146 |
| waveform_seed2 | 0.551130 | 0.491284 | 0.315018 | 0.425402 |
| waveform_seed3 | 0.546109 | 0.484539 | 0.342270 | 0.413980 |
| waveform_rr_seed1 | 0.561716 | 0.472076 | 0.332914 | 0.410682 |
| waveform_rr_seed2 | 0.589309 | 0.469498 | 0.328366 | 0.399988 |
| waveform_rr_seed3 | 0.579107 | 0.519870 | 0.337567 | 0.428260 |
| extra_trees | 0.546817 | 0.483447 | 0.460340 | 0.486250 |

These are new benchmark results. No model was selected or retrained using
external results. The limitations above remain: different reconstruction
protocol, reference beat locations, future RR information, custom class mapping,
selected SDDB annotation coverage and imperfect external subject clustering.

Small results and provenance are in `runs/reconstruction_2026-11-11/`.
The separate evidence archive contains model weights, predictions and bootstrap
draws; source is preserved in its referenced GitHub commit. No original-run
agreement or persistent archive DOI is claimed. Source checksums verify bytes,
not clinical validity.
