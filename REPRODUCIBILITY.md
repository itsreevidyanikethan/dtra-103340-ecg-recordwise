# Reproducibility status for DTRA-103340

This repository is a protocol skeleton with synthetic software checks. Numeric
hyperparameters in the supplied protocol are explicitly placeholders. No
real-dataset experiment was run or verified for this maintenance update.

## Evidence required

| Evidence | Status in this repository |
|---|---|
| Final manuscript-aligned protocol, seeds and hyperparameters | Not supplied; current values require confirmation |
| Original executed source and environment | Original-run provenance not verified |
| Record/subject split records and source-data integrity audit | Missing |
| Fitted checkpoints, validation histories and comparator artifacts | Missing |
| Per-beat labels, record IDs, predictions and external evaluation outputs | Missing |
| Original bootstrap records, metrics and artifact checksums | Missing |
| Persistent archive DOI | Missing |

## Implementation limits identified

The supplied training and test record lists contain records 201 and 202 on
opposite sides, and the source README notes that those records share a subject.
The present split therefore must not be described as patient-independent.
The configured early-stopping patience is not used by `train_cnn` and no
validation split is implemented. The Extra Trees model is not saved, and the
external evaluator evaluates CNN checkpoints only. Per-class F1 values and
row-level predictions are not exported. Normalisation uses a scalar mean and
standard deviation from pooled training beats, while configuration/module text
refers to per-record normalisation. A final manuscript for this package was not
provided with the two chest-X-ray manuscripts.

No protocol, split or model-selection choice was changed during this update.
Resolve these points against the final study design before undertaking or
publishing a full run. Synthetic tests cover model execution and bootstrap
functions, not scientific validity or external generalisation.
