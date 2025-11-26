"""Pupil HHT simulation, decomposition, and visualization."""

import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from PyEMD import EMD
from scipy.signal import hilbert
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


from scipy.stats import f_oneway
# optional, only if you want to actually fit a classifier:
from sklearn.linear_model import LogisticRegression

from scipy.stats import f_oneway
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score



DEFAULT_OUTDIR = "plots"


# def simulate_pupil(fs: int, duration: float, seed: int = 0):
#     """Simulate a pupil-diameter time series with moderate drift and events."""
#     rng = np.random.default_rng(seed)
#     t = np.linspace(0, duration, int(fs * duration), endpoint=False)

#     baseline = 4.0 + 0.1 * np.sin(2 * np.pi * 0.03 * t)  # reduced drift amplitude
#     physio = 0.2 * np.sin(2 * np.pi * 0.25 * t)
#     faster = 0.08 * np.sin(2 * np.pi * 0.6 * t)

#     events = np.zeros_like(t)
#     for ev_t in [10, 20, 32, 45]:
#         events += -0.4 * np.exp(-0.5 * ((t - ev_t) / 0.4) ** 2)

#     noise = 0.05 * rng.standard_normal(len(t))

#     pupil = baseline + physio + faster + events + noise
#     return t, pupil

# def simulate_pupil(fs: int,
#                    duration: float,
#                    seed: int = 0,
#                    event_scale: float = 1.0,
#                    hf_scale: float = 1.0):
#     """Simulate a pupil-diameter time series with drift and events.

#     event_scale: scales the size of event-related dips (arousal-like).
#     hf_scale   : scales the amplitude of the faster oscillation (~0.6–1 Hz).
#     """
#     rng = np.random.default_rng(seed)
#     t = np.linspace(0, duration, int(fs * duration), endpoint=False)

#     baseline = 4.0 + 0.1 * np.sin(2 * np.pi * 0.03 * t)
#     physio = 0.2 * np.sin(2 * np.pi * 0.25 * t)
#     faster = hf_scale * 0.08 * np.sin(2 * np.pi * 0.6 * t)

#     events = np.zeros_like(t)
#     for ev_t in [10, 20, 32, 45]:
#         events += event_scale * (-0.4 * np.exp(-0.5 * ((t - ev_t) / 0.4) ** 2))

#     noise = 0.05 * rng.standard_normal(len(t))

#     pupil = baseline + physio + faster + events + noise
#     return t, pupil

def simulate_pupil(fs: int,
                   duration: float,
                   seed: int = 0,
                   event_scale: float = 1.0,
                   hf_scale: float = 1.0,
                   noise_scale: float = 0.05,
                   event_times=None):
    """
    Simulate a pupil-diameter time series with drift, oscillations, and events.

    event_scale : scales size of event-related dips (arousal-like).
    hf_scale    : scales amplitude of the faster oscillation (~0.6 Hz).
    noise_scale : overall noise amplitude.
    event_times : list of event times in seconds; if None, defaults to [10,20,32,45].
    """
    rng = np.random.default_rng(seed)
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)

    if event_times is None:
        event_times = [10, 20, 32, 45]

    baseline = 4.0 + 0.1 * np.sin(2 * np.pi * 0.03 * t)
    physio = 0.2 * np.sin(2 * np.pi * 0.25 * t)
    faster = hf_scale * 0.08 * np.sin(2 * np.pi * 0.6 * t)

    events = np.zeros_like(t)
    for ev_t in event_times:
        events += event_scale * (-0.4 * np.exp(-0.5 * ((t - ev_t) / 0.4) ** 2))

    noise = noise_scale * rng.standard_normal(len(t))

    pupil = baseline + physio + faster + events + noise
    return t, pupil


def pretty_print_dict(title, d):
    print(f"\n{title}")
    print("-" * len(title))
    for k, v in d.items():
        print(f"{k:12s} : {v:10.6f}")


def emd_decompose(signal):
    """Run EMD on the signal and return IMFs as (n_imfs, n_samples)."""
    emd = EMD()
    imfs = emd(signal)
    if imfs.ndim == 1:
        imfs = imfs[np.newaxis, :]
    return imfs

def load_pupil_csv(path, time_col=0, pupil_col=1, skiprows=1):
    """
    Load pupil data from a simple CSV file.

    Assumes:
        - time is in column `time_col` (seconds),
        - pupil size is in column `pupil_col`,
        - first row is a header (skiprows=1 by default).

    Returns:
        t        : time vector (seconds)
        pupil    : pupil-size vector
        fs_est   : estimated sampling rate (Hz) from median dt
    """
    data = np.loadtxt(path, delimiter=",", skiprows=skiprows)
    t = data[:, time_col]
    pupil = data[:, pupil_col]

    dt = np.diff(t)
    fs_est = 1.0 / np.median(dt)

    return t, pupil, fs_est


def compute_imf_stats(imfs, fs):
    """Compute mean instantaneous frequency, relative energy, and class for each IMF.

    - Baseline classification is based purely on very low mean frequency (< 0.03 Hz).
    - High-frequency noise is classified when mean frequency > 2.5 Hz.
    - Everything in between is considered 'physio'.
    - Relative energy is computed so that non-baseline IMFs are normalized
      against the total energy of all non-baseline IMFs, while baseline
      IMFs are normalized against the total energy of all IMFs.
    """
    stats = []
    if imfs.size == 0:
        return stats

    n_imfs = imfs.shape[0]

    # Per-IMF energies
    energies = np.sum(imfs**2, axis=1)
    total_energy = float(np.sum(energies))

    # We'll later normalize non-baseline IMFs w.r.t. the sum of their energies.
    # For now, just compute frequencies and energies.
    mean_freqs = []

    for idx, imf in enumerate(imfs):
        analytic = hilbert(imf)
        amp = np.abs(analytic)
        phase = np.unwrap(np.angle(analytic))
        inst_freq = np.diff(phase) * fs / (2 * np.pi)

        if amp.size > 1:
            mask = amp[1:] > np.percentile(amp[1:], 60)
            if np.any(mask):
                mean_freq = float(np.mean(inst_freq[mask]))
            else:
                mean_freq = np.nan
        else:
            mean_freq = np.nan

        mean_freqs.append(mean_freq)

    mean_freqs = np.array(mean_freqs)

    # First pass: classify by frequency ONLY
    classes = []
    for mf in mean_freqs:
        if np.isnan(mf):
            classes.append("unknown")
        elif mf < 0.03:
            classes.append("baseline")
        elif mf > 2.5:
            classes.append("noise")
        else:
            classes.append("physio")

    # Compute energy normalization:
    # - non-baseline IMFs: normalized by total energy of all non-baseline IMFs
    # - baseline IMFs: normalized by total energy of all IMFs
    non_baseline_mask = np.array([c != "baseline" for c in classes])
    non_baseline_energy = float(np.sum(energies[non_baseline_mask])) or 1.0

    for idx in range(n_imfs):
        mf = mean_freqs[idx]
        cls = classes[idx]
        energy = float(energies[idx])

        if cls == "baseline":
            # fraction of total energy in the entire signal
            rel_energy = energy / total_energy if total_energy > 0 else np.nan
        else:
            # fraction of energy among all non-baseline IMFs
            rel_energy = energy / non_baseline_energy if non_baseline_energy > 0 else np.nan

        stats.append(
            {
                "index": idx + 1,
                "mean_freq_hz": mf,
                "rel_energy": rel_energy,
                "classification": cls,
            }
        )

    return stats

def imf_power_vector(imfs):
    """
    Return absolute power per IMF as a 1D array of length n_imfs.
    """
    if imfs.size == 0:
        return np.zeros(0)
    return np.sum(imfs**2, axis=1)

def compute_imf_power(imfs):
    """
    Per-IMF power = mean squared amplitude.
    Returns array of shape (n_imfs,).
    """
    if imfs.size == 0:
        return np.array([])
    return np.mean(imfs**2, axis=1)


def define_pupil_bands():
    return {
        "VLF": (0.01, 0.04),
        "LF":  (0.04, 0.15),
        "MF":  (0.15, 0.40),
        "HF":  (0.40, 1.50),  # widened to catch IMF 5 at ~1.08 Hz
    }


def compute_band_powers(stats, imfs):
    """
    Compute band power for each neurophysiological band by summing powers
    of IMFs whose mean frequency falls inside the band.
    Returns:
        band_powers: dict[band_name] -> float
        imf_power:   np.ndarray of shape (n_imfs,)
    """
    bands = define_pupil_bands()
    imf_power = compute_imf_power(imfs)

    # Map IMF index -> mean frequency
    mean_freqs = np.array([s["mean_freq_hz"] for s in stats])

    band_powers = {}
    for name, (f_lo, f_hi) in bands.items():
        in_band = (mean_freqs >= f_lo) & (mean_freqs < f_hi)
        band_powers[name] = float(np.sum(imf_power[in_band]))

    return band_powers, imf_power


def compute_band_ratios(band_powers):
    """
    Compute common band ratios, e.g. LF/HF, (LF+MF)/HF, etc.
    Returns dict with ratios; if denominator is zero, ratio is np.nan.
    """
    def safe_ratio(num, den):
        return float(num / den) if den > 0 else np.nan

    lf = band_powers.get("LF", 0.0)
    mf = band_powers.get("MF", 0.0)
    hf = band_powers.get("HF", 0.0)
    vlf = band_powers.get("VLF", 0.0)

    ratios = {
        "LF_HF": safe_ratio(lf, hf),
        "MF_HF": safe_ratio(mf, hf),
        "VLF_LF": safe_ratio(vlf, lf),
        "LFplusMF_HF": safe_ratio(lf + mf, hf),
    }
    return ratios

def compute_event_locked_imf_average(t, imfs, event_times, fs, t_pre=1.0, t_post=3.0):
    """
    Build event-locked averages for each IMF.

    Args:
        t           : time vector (seconds), shape (n_samples,)
        imfs        : array (n_imfs, n_samples)
        event_times : list of event times (seconds)
        fs          : sampling rate (Hz)
        t_pre       : time before event (s)
        t_post      : time after event (s)

    Returns:
        t_epoch     : relative time axis, shape (n_epoch_samples,)
        imf_evoked  : array (n_imfs, n_epoch_samples) = mean across events
    """
    if imfs.size == 0 or len(event_times) == 0:
        return np.array([]), np.array([])

    n_imfs, n_samples = imfs.shape
    n_pre = int(round(t_pre * fs))
    n_post = int(round(t_post * fs))
    win_len = n_pre + n_post

    # relative time axis centered on the event
    t_epoch = np.linspace(-t_pre, t_post, win_len, endpoint=False)

    # collect epochs: shape (n_events, n_imfs, win_len)
    epochs = []
    for ev_t in event_times:
        center_idx = np.argmin(np.abs(t - ev_t))
        start = center_idx - n_pre
        end = center_idx + n_post
        if start < 0 or end > n_samples:
            # skip events too close to edges
            continue
        epochs.append(imfs[:, start:end])

    if not epochs:
        return t_epoch, np.zeros((n_imfs, win_len))

    epochs = np.stack(epochs, axis=0)  # (n_events, n_imfs, win_len)
    imf_evoked = np.mean(epochs, axis=0)  # average over events

    return t_epoch, imf_evoked


def plot_event_locked_imfs(t_epoch, imf_evoked, save=False, show=True, outdir=DEFAULT_OUTDIR):
    """
    Plot event-locked average for each IMF.
    """
    if t_epoch.size == 0 or imf_evoked.size == 0:
        return

    n_imfs = imf_evoked.shape[0]
    fig, axes = plt.subplots(n_imfs, 1, figsize=(8, 1.3 * n_imfs), sharex=True)

    if n_imfs == 1:
        axes = [axes]

    for i in range(n_imfs):
        axes[i].plot(t_epoch, imf_evoked[i, :])
        axes[i].axvline(0.0, color="red", linestyle="--", linewidth=1)
        axes[i].set_ylabel(f"IMF {i+1}")

    axes[-1].set_xlabel("Time relative to event (s)")
    fig.suptitle("Event-locked IMF averages", y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    if save:
        os.makedirs(outdir, exist_ok=True)
        plt.savefig(os.path.join(outdir, "event_locked_imfs.png"), dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)
        
# def run_arousal_trial_simulation(n_trials_per_cond=40,
#                                  fs=100,
#                                  duration=60.0,
#                                  base_seed=100):
#     """
#     Simulate many trials under two conditions (low vs high arousal),
#     compute band powers/ratios per trial, run ANOVA, and fit
#     a logistic regression arousal classifier.

#     Returns (X, y, feature_names, model)
#     """
#     # Features: 4 band powers + 4 band ratios
#     feature_names = [
#         "VLF_power", "LF_power", "MF_power", "HF_power",
#         "LF_HF", "MF_HF", "VLF_LF", "LFplusMF_HF",
#     ]

#     X_list = []
#     y_list = []         # 0 = low arousal, 1 = high arousal
#     cond_names = []     # for readability only

#     # Define two conditions with different arousal-like parameters
#     conditions = [
#         ("low",  0.8, 0.8),   # (label, event_scale, hf_scale)
#         ("high", 1.3, 1.5),   # bigger/faster response for high arousal
#     ]

#     trial_index = 0
#     for arousal_label, event_scale, hf_scale in conditions:
#         for k in range(n_trials_per_cond):
#             seed = base_seed + trial_index
#             trial_index += 1

#             # 1) simulate signal
#             t, pupil = simulate_pupil(
#                 fs=fs,
#                 duration=duration,
#                 seed=seed,
#                 event_scale=event_scale,
#                 hf_scale=hf_scale,
#             )

#             # 2) EMD + stats
#             imfs = emd_decompose(pupil)
#             stats = compute_imf_stats(imfs, fs)

#             # 3) band powers & ratios
#             band_powers, _ = compute_band_powers(stats, imfs)
#             band_ratios = compute_band_ratios(band_powers)

#             # 4) assemble feature vector
#             feat_vec = np.array([
#                 band_powers.get("VLF", 0.0),
#                 band_powers.get("LF", 0.0),
#                 band_powers.get("MF", 0.0),
#                 band_powers.get("HF", 0.0),
#                 band_ratios.get("LF_HF", np.nan),
#                 band_ratios.get("MF_HF", np.nan),
#                 band_ratios.get("VLF_LF", np.nan),
#                 band_ratios.get("LFplusMF_HF", np.nan),
#             ], dtype=float)

#             X_list.append(feat_vec)
#             y_list.append(0 if arousal_label == "low" else 1)
#             cond_names.append(arousal_label)

#     X = np.vstack(X_list)
#     y = np.array(y_list)

#     # Replace any NaN ratios with 0 so the model can handle them
#     X = np.nan_to_num(X, nan=0.0)

#     # ---------- ANOVA per feature ----------
#     print("\n=== One-way ANOVA: low vs high arousal per feature ===")
#     low_mask = (y == 0)
#     high_mask = (y == 1)

#     for j, fname in enumerate(feature_names):
#         F, p = f_oneway(X[low_mask, j], X[high_mask, j])
#         print(f"{fname:12s}  F = {F:7.3f}, p = {p: .3e}")

#     # ---------- Predictive model (logistic regression) ----------
#     X_train, X_test, y_train, y_test = train_test_split(
#         X, y, test_size=0.3, stratify=y, random_state=0
#     )

#     clf = make_pipeline(
#         StandardScaler(),
#         LogisticRegression(max_iter=2000)
#     )
#     clf.fit(X_train, y_train)

#     y_pred = clf.predict(X_test)
#     acc = accuracy_score(y_test, y_pred)
#     print("\n=== Logistic regression arousal classifier ===")
#     print(f"Test accuracy: {acc:.3f}")

#     # Inspect feature weights in standardized space
#     lr = clf.named_steps["logisticregression"]
#     coefs = lr.coef_[0]
#     print("\nFeature coefficients (positive => higher in high arousal):")
#     for fname, w in sorted(zip(feature_names, coefs), key=lambda x: -abs(x[1])):
#         print(f"{fname:12s}: {w: .3f}")

#     return X, y, feature_names, clf

def cohens_d(x, y):
    """Compute Cohen's d for two independent samples."""
    nx, ny = len(x), len(y)
    vx, vy = np.var(x, ddof=1), np.var(y, ddof=1)
    pooled_sd = np.sqrt(((nx-1)*vx + (ny-1)*vy) / (nx + ny - 2))
    if pooled_sd == 0:
        return 0.0
    return (np.mean(x) - np.mean(y)) / pooled_sd

def fdr_bh(pvals):
    """Benjamini-Hochberg FDR correction."""
    pvals = np.array(pvals)
    n = len(pvals)
    order = np.argsort(pvals)
    ranked_p = pvals[order]
    qvals = np.empty(n)
    cummin = 1.0
    for i in reversed(range(n)):
        rank = i + 1
        q = ranked_p[i] * n / rank
        cummin = min(cummin, q)
        qvals[i] = cummin
    # return in original order
    qvals_original = np.empty(n)
    qvals_original[order] = qvals
    return qvals_original


def run_arousal_trial_simulation(n_trials_per_cond=40,
                                 fs=100,
                                 duration=60.0,
                                 base_seed=100,
                                 save_plots=False,
                                 show=True,
                                 outdir=DEFAULT_OUTDIR):
    """
    Simulate many trials under two conditions (low vs high arousal),
    compute band powers/ratios per trial, run ANOVA, and fit
    a logistic regression arousal classifier.

    Returns (X, y, feature_names, model)
    """
    feature_names = [
        "VLF_power", "LF_power", "MF_power", "HF_power",
        "LF_HF", "MF_HF", "VLF_LF", "LFplusMF_HF",
    ]

    X_list = []
    y_list = []

    # Slightly closer conditions + more noise
    # label, event_scale, hf_scale, noise_scale
    conditions = [
        ("low",  0.9, 1.0, 0.07),
        ("high", 1.1, 1.3, 0.09),
    ]

    base_events = np.array([10.0, 20.0, 32.0, 45.0])
    jitter_std = 0.25  # seconds

    trial_index = 0
    X_imf_list = []

    for arousal_label, event_scale, hf_scale, noise_scale in conditions:
        for _ in range(n_trials_per_cond):
            seed = base_seed + trial_index
            trial_index += 1
            rng = np.random.default_rng(seed)

            # jitter event times per trial
            event_times = base_events + rng.normal(0.0, jitter_std, size=base_events.shape)

            # 1) simulate signal
            t, pupil = simulate_pupil(
                fs=fs,
                duration=duration,
                seed=seed,
                event_scale=event_scale,
                hf_scale=hf_scale,
                noise_scale=noise_scale,
                event_times=event_times.tolist(),
            )

            # 2) EMD + stats
            imfs = emd_decompose(pupil)
            stats = compute_imf_stats(imfs, fs)
            
            # IMF-level power for this trial
            p_vec = imf_power_vector(imfs)
            X_imf_list.append(p_vec)


            # 3) band powers & ratios
            band_powers, _ = compute_band_powers(stats, imfs)
            band_ratios = compute_band_ratios(band_powers)

            # 4) assemble feature vector
            feat_vec = np.array([
                band_powers.get("VLF", 0.0),
                band_powers.get("LF", 0.0),
                band_powers.get("MF", 0.0),
                band_powers.get("HF", 0.0),
                band_ratios.get("LF_HF", np.nan),
                band_ratios.get("MF_HF", np.nan),
                band_ratios.get("VLF_LF", np.nan),
                band_ratios.get("LFplusMF_HF", np.nan),
            ], dtype=float)

            X_list.append(feat_vec)
            y_list.append(0 if arousal_label == "low" else 1)
    
    # --- assemble IMF power matrix (n_trials x max_n_imfs) ---
    max_imfs = max(len(v) for v in X_imf_list)
    n_trials = len(X_imf_list)
    X_imf = np.zeros((n_trials, max_imfs))
    for i, p_vec in enumerate(X_imf_list):
        X_imf[i, :len(p_vec)] = p_vec


    X = np.vstack(X_list)
    y = np.array(y_list)
    X = np.nan_to_num(X, nan=0.0)

    # ANOVA
    print("\n=== One-way ANOVA: low vs high arousal per feature ===")
    low_mask = (y == 0)
    high_mask = (y == 1)

    for j, fname in enumerate(feature_names):
        F, p = f_oneway(X[low_mask, j], X[high_mask, j])
        print(f"{fname:12s}  F = {F:7.3f}, p = {p: .3e}")
        
    # # --- IMF-level ANOVA: which IMFs differ with arousal? ---
    # print("\n=== One-way ANOVA: IMF power per arousal condition ===")
    # for imf_idx in range(max_imfs):
    #     F_imf, p_imf = f_oneway(
    #         X_imf[low_mask, imf_idx],
    #         X_imf[high_mask, imf_idx],
    #     )
    #     print(f"IMF {imf_idx+1:2d}  power  F = {F_imf:7.3f}, p = {p_imf: .3e}")
    
    # --- IMF-level ANOVA: low vs high arousal ---
    print("\n=== IMF-Level Arousal Effects (ANOVA + Effect Sizes + FDR) ===")

    p_values = []
    F_values = []
    d_values = []
    mean_low = []
    mean_high = []

    for imf_idx in range(max_imfs):
        x_low = X_imf[low_mask, imf_idx]
        x_high = X_imf[high_mask, imf_idx]

        F_imf, p_imf = f_oneway(x_low, x_high)
        d_imf = cohens_d(x_low, x_high)

        p_values.append(p_imf)
        F_values.append(F_imf)
        d_values.append(d_imf)
        mean_low.append(np.mean(x_low))
        mean_high.append(np.mean(x_high))

    # --- FDR correction ---
    q_values = fdr_bh(p_values)

    # --- Print summary table ---
    print(f"{'IMF':<5}{'F':>10}{'p':>14}{'q(FDR)':>14}{'d':>10}{'Mean Low':>14}{'Mean High':>14}")
    for i in range(max_imfs):
        print(
            f"{i+1:<5}{F_values[i]:10.3f}"
            f"{p_values[i]:14.3e}{q_values[i]:14.3e}"
            f"{d_values[i]:10.3f}"
            f"{mean_low[i]:14.4f}{mean_high[i]:14.4f}"
        )
    # --- Bar plot of effect sizes for significant IMFs ---
    sig_mask = q_values < 0.05

    if np.any(sig_mask):
        plt.figure(figsize=(10, 5))
        sig_indices = np.where(sig_mask)[0]
        plt.bar(sig_indices + 1, np.array(d_values)[sig_mask], color="tab:red")
        plt.axhline(0, color="black", linewidth=1)
        plt.xlabel("IMF Index")
        plt.ylabel("Cohen's d (Low - High)")  # sign per our cohens_d(x_low, x_high)
        plt.title("Significant IMF-Level Effect Sizes (FDR q < 0.05)")
        plt.tight_layout()
        if save_plots:
            os.makedirs(outdir, exist_ok=True)
            plt.savefig(os.path.join(outdir, "imf_effect_sizes.png"), dpi=300, bbox_inches="tight")
        if show:
            plt.show()
        else:
            plt.close()



    # classifier
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=0
    )

    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000)
    )
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print("\n=== Logistic regression arousal classifier ===")
    print(f"Test accuracy: {acc:.3f}")

    lr = clf.named_steps["logisticregression"]
    coefs = lr.coef_[0]
    print("\nFeature coefficients (positive => higher in high arousal):")
    for fname, w in sorted(zip(feature_names, coefs), key=lambda x: -abs(x[1])):
        print(f"{fname:12s}: {w: .3f}")

    return X, y, feature_names, clf


def anova_on_band_powers(band_power_by_condition):
    """
    Run one-way ANOVA on band powers across conditions.

    Args:
        band_power_by_condition: dict[str, np.ndarray]
            Example:
                {
                    "low_load": np.array([... band powers per trial ...]),
                    "high_load": np.array([...])
                }

    Returns:
        results: dict with F and p-values for each band, per ANOVA.
    """
    results = {}
    for band_name in band_power_by_condition:
        # collect per-condition arrays for this band
        arrays = [vals for vals in band_power_by_condition[band_name].values()]
        F, p = f_oneway(*arrays)
        results[band_name] = {"F": float(F), "p": float(p)}
    return results

def fit_arousal_model(feature_matrix, labels):
    """
    Fit a simple logistic regression classifier to predict arousal.

    Args:
        feature_matrix : np.ndarray, shape (n_samples, n_features)
            e.g., concatenated band powers, band ratios, IMF powers, etc.
        labels         : np.ndarray, shape (n_samples,)
            Binary or multi-class arousal labels.

    Returns:
        model : fitted sklearn LogisticRegression instance
    """
    model = LogisticRegression(max_iter=1000)
    model.fit(feature_matrix, labels)
    return model


def print_imf_table(stats, save=False, outdir=DEFAULT_OUTDIR):
    """Print IMF summary and optionally save to a text file."""
    lines = ["IMF  MeanFreq(Hz)  RelEnergy   Class"]
    for entry in stats:
        mf = entry["mean_freq_hz"]
        mf_str = f"{mf:10.3f}" if not np.isnan(mf) else "    nan   "
        re = entry["rel_energy"]
        lines.append(f"{entry['index']:>3d}  {mf_str}    {re:8.3f}   {entry['classification']}")

    table_text = "\n".join(lines)
    print(table_text)

    if save:
        os.makedirs(outdir, exist_ok=True)
        path = os.path.join(outdir, "imf_stats_table.txt")
        with open(path, "w", encoding="utf-8") as f:
            f.write(table_text + "\n")


def reconstruct_from_imfs(imfs, indices):
    """Reconstruct signal from selected IMF indices (0-based)."""
    if len(indices) == 0:
        return np.zeros(imfs.shape[1])
    return imfs[indices].sum(axis=0)


def plot_signal_and_imfs(t, signal, imfs, save=False, show=True, outdir=DEFAULT_OUTDIR):
    """Plot original signal and its IMFs."""
    n_imfs = imfs.shape[0]
    fig, axes = plt.subplots(n_imfs + 1, 1, figsize=(10, 1.4 * (n_imfs + 1)), sharex=True)

    axes[0].plot(t, signal, color="black")
    axes[0].set_title("Original Signal")
    axes[0].set_ylabel("Amplitude")

    for idx, imf in enumerate(imfs):
        axes[idx + 1].plot(t, imf)
        axes[idx + 1].set_ylabel(f"IMF {idx + 1}")

    axes[-1].set_xlabel("Time (s)")
    fig.tight_layout()

    if save:
        os.makedirs(outdir, exist_ok=True)
        plt.savefig(os.path.join(outdir, "imfs.png"), dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_reconstruction(t, original, reconstructed, selected_indices, save=False, show=True, outdir=DEFAULT_OUTDIR):
    """Plot original signal and reconstruction from selected IMFs."""
    fig = plt.figure(figsize=(10, 3))
    plt.plot(t, original, label="Original", color="black")
    plt.plot(t, reconstructed, label="Reconstructed (physio IMFs)", linewidth=2, color="tab:orange")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.title(f"Reconstruction from IMFs {selected_indices}")
    plt.legend()
    plt.tight_layout()

    if save:
        os.makedirs(outdir, exist_ok=True)
        plt.savefig(os.path.join(outdir, "reconstruction.png"), dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)


def compute_hilbert_spectrum(imfs, fs, f_max, classifications, n_time=300, n_freq=120):
    """Build time-frequency-amplitude grid (Hilbert spectrum), skipping baseline IMFs."""
    n_samples = imfs.shape[1] if imfs.size else 0
    t_full = np.arange(n_samples) / fs if n_samples else np.array([0.0])

    t_grid = np.linspace(t_full[0], t_full[-1], n_time) if n_samples else np.array([0.0])
    f_grid = np.linspace(0, f_max, n_freq)
    H = np.zeros((n_freq, n_time))

    if not imfs.size:
        return t_grid, f_grid, H

    for imf, cls in zip(imfs, classifications):
        if cls == "baseline":
            continue
        analytic = hilbert(imf)
        amp = np.abs(analytic)
        phase = np.unwrap(np.angle(analytic))
        inst_freq = np.abs(np.diff(phase) * fs / (2 * np.pi))
        t_mid = t_full[1:]
        if inst_freq.size == 0:
            continue

        ti = ((t_mid - t_grid[0]) / (t_grid[-1] - t_grid[0]) * (n_time - 1)).astype(int)
        fi = (inst_freq / f_max * (n_freq - 1)).astype(int)
        ti = np.clip(ti, 0, n_time - 1)
        fi = np.clip(fi, 0, n_freq - 1)

        for tt, ff, amp_val in zip(ti, fi, amp[1:]):
            H[ff, tt] += amp_val

    return t_grid, f_grid, H


def plot_hilbert_spectrum(t_grid, f_grid, H, save=False, show=True, outdir=DEFAULT_OUTDIR):
    """Plot the Hilbert spectrum as a 2D image."""
    fig = plt.figure(figsize=(10, 4))
    extent = [t_grid[0], t_grid[-1], f_grid[0], f_grid[-1]]
    plt.imshow(np.log1p(H), aspect="auto", origin="lower", extent=extent, cmap="viridis")
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")
    plt.title("Hilbert Spectrum (log1p amplitude)")
    cbar = plt.colorbar()
    cbar.set_label("Amplitude (log1p)")
    plt.tight_layout()

    if save:
        os.makedirs(outdir, exist_ok=True)
        plt.savefig(os.path.join(outdir, "hilbert_spectrum.png"), dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_hht_3d(t_grid, f_grid, H, save=False, show=True, outdir=DEFAULT_OUTDIR):
    """Plot the 3D Hilbert-Huang spectrum surface (log1p amplitude)."""
    Tg, Fg = np.meshgrid(t_grid, f_grid)
    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(Tg, Fg, np.log1p(H), rstride=2, cstride=2, cmap="viridis")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    ax.set_zlabel("log1p Amplitude")
    ax.set_title("Hilbert-Huang Spectrum (3D)")
    fig.tight_layout()

    if save:
        os.makedirs(outdir, exist_ok=True)
        plt.savefig(os.path.join(outdir, "hht_3d.png"), dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)
        
def plot_feature_distributions(
    X,
    y,
    feature_names,
    features_to_show=None,
    save=False,
    show=True,
    outdir=DEFAULT_OUTDIR,
):
    """
    Visualize feature distributions for low vs high arousal.

    X : (n_trials, n_features)
    y : (n_trials,)  0 = low, 1 = high
    feature_names : list of feature-name strings
    features_to_show : list of names to plot; if None, plot all
    """
    if features_to_show is None:
        features_to_show = feature_names

    low_mask = (y == 0)
    high_mask = (y == 1)

    n_feats = len(features_to_show)
    fig, axes = plt.subplots(n_feats, 1, figsize=(6, 2.3 * n_feats), sharex=False)

    if n_feats == 1:
        axes = [axes]

    for ax, fname in zip(axes, features_to_show):
        j = feature_names.index(fname)
        data_low = X[low_mask, j]
        data_high = X[high_mask, j]

        ax.boxplot(
            [data_low, data_high],
            tick_labels=["low", "high"],
            showmeans=True,
        )
        ax.set_title(fname)
        ax.set_ylabel("value")

    axes[-1].set_xlabel("arousal condition")
    fig.suptitle("Feature distributions by arousal", y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    if save:
        os.makedirs(outdir, exist_ok=True)
        plt.savefig(os.path.join(outdir, "feature_distributions.png"),
                    dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)



def parse_args():
    parser = argparse.ArgumentParser(description="Simulate pupil signal and run HHT analysis.")
    parser.add_argument("--duration", type=float, default=60.0, help="Duration in seconds (default: 60)")
    parser.add_argument("--fs", type=int, default=100, help="Sampling rate in Hz (default: 100)")
    parser.add_argument("--f-max", type=float, default=3.0, help="Max frequency for Hilbert spectrum (default: 3.0)")
    parser.add_argument("--seed", type=int, default=0, help="Random seed (default: 0)")
    parser.add_argument("--no-3d", action="store_true", help="Disable 3D Hilbert-Huang plot")
    parser.add_argument("--save-plots", action="store_true", help="Save plots to ./plots/")
    parser.add_argument("--no-show", action="store_true", help="Suppress displaying plots")
    return parser.parse_args()


def main():
    args = parse_args()
    outdir = DEFAULT_OUTDIR
    
    # --- OPTION: use a real pupil CSV instead of simulation ---
    # Uncomment and adjust when you have real data:
    #
    # real_path = "my_pupil_file.csv"
    # t, pupil, fs_est = load_pupil_csv(
    #     real_path,
    #     time_col=0,     # adjust to your file
    #     pupil_col=1,    # adjust to your file
    #     skiprows=1,     # header lines
    #     delimiter=",",  # or "\t"
    # )
    # args.fs = int(round(fs_est))
    #
    # # If you use real data, comment out the simulate_pupil call below,
    # # and just pass (t, pupil) into the rest of the pipeline.

    t, pupil = simulate_pupil(fs=args.fs, duration=args.duration, seed=args.seed)
    imfs = emd_decompose(pupil)

    stats = compute_imf_stats(imfs, fs=args.fs)
    print_imf_table(stats, save=args.save_plots, outdir=outdir if args.save_plots else DEFAULT_OUTDIR)

    band_powers, imf_power = compute_band_powers(stats, imfs)
    band_ratios = compute_band_ratios(band_powers)
    pretty_print_dict("Band powers", band_powers)
    pretty_print_dict("Band ratios", band_ratios)

    
    classifications = [s["classification"] for s in stats]
    physio_indices = [i for i, s in enumerate(stats) if s["classification"] == "physio"]
    if len(physio_indices) == 0:
        print("Warning: No physiological IMFs selected with current criteria; reconstruction is zero.")
    reconstructed = reconstruct_from_imfs(imfs, physio_indices)

    plot_signal_and_imfs(
        t,
        pupil,
        imfs,
        save=args.save_plots,
        show=not args.no_show,
        outdir=outdir,
    )

    plot_reconstruction(
        t,
        pupil,
        reconstructed,
        [i + 1 for i in physio_indices],
        save=args.save_plots,
        show=not args.no_show,
        outdir=outdir,
    )

    t_grid, f_grid, H = compute_hilbert_spectrum(imfs, fs=args.fs, f_max=args.f_max, classifications=classifications)
    plot_hilbert_spectrum(
        t_grid,
        f_grid,
        H,
        save=args.save_plots,
        show=not args.no_show,
        outdir=outdir,
    )

    if not args.no_3d:
        plot_hht_3d(
            t_grid,
            f_grid,
            H,
            save=args.save_plots,
            show=not args.no_show,
            outdir=outdir,
        )
        
    event_times = [10, 20, 32, 45]  # in seconds; for real data these would be trial onsets, etc.
    t_epoch, imf_evoked = compute_event_locked_imf_average(
        t, imfs, event_times, fs=args.fs, t_pre=1.0, t_post=3.0
    )
    plot_event_locked_imfs(
        t_epoch, imf_evoked,
        save=args.save_plots,
        show=not args.no_show,
        outdir=outdir,
    )
    
    print("\n\n############ AROUSAL TRIAL SIMULATION ############")
    X, y, feature_names, clf = run_arousal_trial_simulation(
        n_trials_per_cond=40,
        fs=args.fs,
        duration=args.duration,
        base_seed=1000,
        save_plots=args.save_plots,
        show=not args.no_show,
        outdir=outdir if args.save_plots else DEFAULT_OUTDIR,
        )


    # NEW: sanity plots for a few key features
    features_to_show = ["LF_power", "HF_power", "VLF_LF"]
    plot_feature_distributions(
        X,
        y,
        feature_names,
        features_to_show=features_to_show,
        save=args.save_plots,
        show=not args.no_show,
        outdir=outdir,
    )



if __name__ == "__main__":
    main()
