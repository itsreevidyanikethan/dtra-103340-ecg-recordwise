"""Beat extraction: annotation-supplied R peaks, 256-point resampled windows,
per-record training z-scoring, RR features, and the three-class label mapping."""
import numpy as np
import wfdb
from scipy.signal import resample

MAPPING = {
    "Normal": set("NLRej"),
    "VEB": set("VE"),
    "Other": set("AaJSF/fQ"),
}
CLASS_INDEX = {"Normal": 0, "VEB": 1, "Other": 2}


def beat_class(symbol):
    for cls, syms in MAPPING.items():
        if symbol in syms:
            return CLASS_INDEX[cls]
    return None  # non-beat annotation


def pick_lead(record):
    names = [n.upper() for n in record.sig_name]
    return names.index("MLII") if "MLII" in names else 0


def extract_record(path, window_s=0.7, out_points=256):
    """Return beats (n,256), labels (n,), rr features (n,3) for one record.

    Beat locations come from the reference annotations (not end-to-end
    detection), as stated in the manuscript.
    """
    rec = wfdb.rdrecord(path)
    ann = wfdb.rdann(path, "atr")
    sig = rec.p_signal[:, pick_lead(rec)]
    fs = rec.fs
    half = int(window_s * fs / 2)
    samples, labels, rrs = [], [], []
    peaks = ann.sample
    for k, (pos, sym) in enumerate(zip(peaks, ann.symbol)):
        cls = beat_class(sym)
        if cls is None or pos - half < 0 or pos + half >= len(sig):
            continue
        seg = resample(sig[pos - half:pos + half], out_points)
        pre = (pos - peaks[k - 1]) / fs if k > 0 else np.nan
        post = (peaks[k + 1] - pos) / fs if k + 1 < len(peaks) else np.nan
        local = np.diff(peaks[max(0, k - 5):k + 1]).mean() / fs if k > 0 else np.nan
        samples.append(seg)
        labels.append(cls)
        rrs.append([pre, post, local])
    x = np.asarray(samples, dtype=np.float32)
    return x, np.asarray(labels), np.asarray(rrs, dtype=np.float32)


def zscore_train_stats(x_train):
    return float(x_train.mean()), float(x_train.std())


def waveform_stats(x):
    """Time-domain morphological statistics for the tree model."""
    return np.column_stack([
        x.mean(1), x.std(1), x.min(1), x.max(1),
        np.abs(np.diff(x, axis=1)).mean(1),
        ((x[:, 1:] * x[:, :-1]) < 0).sum(1),            # zero crossings
        np.argmax(x, axis=1) / x.shape[1],
        np.argmin(x, axis=1) / x.shape[1],
    ]).astype(np.float32)
