import inspect
import numpy as np
import plotly.graph_objects as go
from dash import Dash, Input, Output, State, dash_table, dcc, html
from scipy.stats import f_oneway
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from pupil_hht_sim import (
    cohens_d,
    compute_band_powers,
    compute_band_ratios,
    compute_event_locked_imf_average,
    compute_imf_stats,
    emd_decompose,
    fdr_bh,
    imf_power_vector,
    reconstruct_from_imfs,
    run_arousal_trial_simulation,
    simulate_pupil,
)


BOOTSTRAP_THEME = "https://cdn.jsdelivr.net/npm/bootswatch@5.3.2/dist/darkly/bootstrap.min.css"
PLOT_LAYOUT = {
    "template": "plotly_dark",
    "paper_bgcolor": "rgba(0,0,0,0)",
    "plot_bgcolor": "rgba(0,0,0,0)",
}

app = Dash(__name__, external_stylesheets=[BOOTSTRAP_THEME])
app.title = "Pupil HHT Lab"


def safe_number(value, default, cast_func):
    try:
        return cast_func(value)
    except (TypeError, ValueError):
        return default


def parse_event_list(text):
    if not text:
        return []
    items = str(text).replace(";", ",").split(",")
    events = []
    for item in items:
        item = item.strip()
        if not item:
            continue
        try:
            events.append(float(item))
        except ValueError:
            continue
    return events


def empty_fig(message):
    fig = go.Figure()
    fig.add_annotation(text=message, showarrow=False, font={"color": "#cccccc", "size": 14})
    fig.update_layout(**PLOT_LAYOUT, margin={"l": 40, "r": 20, "t": 40, "b": 40})
    return fig


def build_signal_fig(t, signal):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t, y=signal, mode="lines", name="Pupil"))
    fig.update_layout(
        **PLOT_LAYOUT,
        margin={"l": 60, "r": 20, "t": 40, "b": 50},
        xaxis_title="Time (s)",
        yaxis_title="Pupil size (a.u.)",
        title="Original simulated pupil signal",
    )
    return fig


def build_imf_fig(t, imfs):
    if imfs.size == 0:
        return empty_fig("No IMFs available for this run.")

    fig = go.Figure()
    offset = 0.35
    for idx, imf in enumerate(imfs):
        fig.add_trace(
            go.Scatter(
                x=t,
                y=imf + idx * offset,
                mode="lines",
                name=f"IMF {idx + 1}",
                hovertemplate="t=%{x:.2f}s<br>value=%{y:.3f}",
            )
        )
    fig.update_layout(
        **PLOT_LAYOUT,
        margin={"l": 60, "r": 20, "t": 40, "b": 50},
        xaxis_title="Time (s)",
        yaxis_title="IMFs (stacked offsets)",
        title="Empirical Mode Decomposition (stacked)",
        showlegend=True,
    )
    return fig


def build_reconstruction_fig(t, original, reconstructed):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t, y=original, mode="lines", name="Original"))
    fig.add_trace(
        go.Scatter(
            x=t,
            y=reconstructed,
            mode="lines",
            name="Physio-only reconstruction",
            line={"width": 3},
        )
    )
    fig.update_layout(
        **PLOT_LAYOUT,
        margin={"l": 60, "r": 20, "t": 40, "b": 50},
        xaxis_title="Time (s)",
        yaxis_title="Amplitude",
        title="Reconstruction from physiological IMFs",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "x": 0},
    )
    return fig


def build_imf_table(stats):
    columns = [
        {"name": "IMF", "id": "index"},
        {"name": "Mean freq (Hz)", "id": "mean_freq_hz"},
        {"name": "Rel energy", "id": "rel_energy"},
        {"name": "Class", "id": "classification"},
    ]
    data = []
    for row in stats:
        mf = row.get("mean_freq_hz")
        mf_val = None if mf is None or np.isnan(mf) else round(float(mf), 4)
        rel = row.get("rel_energy")
        rel_val = None if rel is None or np.isnan(rel) else round(float(rel), 4)
        data.append(
            {
                "index": row.get("index"),
                "mean_freq_hz": mf_val,
                "rel_energy": rel_val,
                "classification": row.get("classification"),
            }
        )
    return data, columns


def build_event_locked_fig(t_epoch, imf_evoked):
    if t_epoch.size == 0 or imf_evoked.size == 0:
        return empty_fig("No valid event-locked windows (check event times and duration).")

    fig = go.Figure()
    offset = 0.35
    for idx in range(imf_evoked.shape[0]):
        fig.add_trace(
            go.Scatter(
                x=t_epoch,
                y=imf_evoked[idx] + idx * offset,
                mode="lines",
                name=f"IMF {idx + 1}",
            )
        )
    fig.add_shape(
        type="line",
        x0=0,
        x1=0,
        y0=np.min(imf_evoked) - offset,
        y1=np.max(imf_evoked) + imf_evoked.shape[0] * offset,
        line={"color": "#ff6b6b", "dash": "dash"},
    )
    fig.update_layout(
        **PLOT_LAYOUT,
        margin={"l": 60, "r": 20, "t": 40, "b": 50},
        xaxis_title="Time relative to event (s)",
        yaxis_title="IMFs (stacked offsets)",
        title="Event-locked IMF averages",
        showlegend=True,
    )
    return fig


def build_feature_boxplot_fig(X, y, feature_names, features_to_show):
    if X.size == 0:
        return empty_fig("No trials simulated yet.")

    fig = go.Figure()
    low_mask = y == 0
    high_mask = y == 1

    for fname in features_to_show:
        if fname not in feature_names:
            continue
        idx = feature_names.index(fname)
        fig.add_trace(go.Box(y=X[low_mask, idx], name=f"{fname} (low)", boxmean=True))
        fig.add_trace(go.Box(y=X[high_mask, idx], name=f"{fname} (high)", boxmean=True))

    fig.update_layout(
        **PLOT_LAYOUT,
        boxmode="group",
        margin={"l": 60, "r": 20, "t": 40, "b": 70},
        xaxis_title="Feature / arousal condition",
        yaxis_title="Value",
        title="Band-power features by arousal condition",
        legend={"orientation": "h", "yanchor": "bottom", "y": -0.2, "x": 0},
    )
    return fig


def build_imf_effect_fig(imf_effects):
    if not imf_effects:
        return empty_fig("No IMF effect sizes available.")

    sig_effects = [e for e in imf_effects if e["q"] < 0.05]
    if not sig_effects:
        fig = empty_fig("No significant IMF-level effects (q < 0.05).")
        return fig

    x_vals = [e["index"] for e in sig_effects]
    y_vals = [e["d"] for e in sig_effects]
    hover_text = [
        f"IMF {e['index']}<br>d={e['d']:.3f}<br>p={e['p']:.3e}<br>q={e['q']:.3e}"
        for e in sig_effects
    ]

    fig = go.Figure()
    fig.add_trace(go.Bar(x=x_vals, y=y_vals, text=hover_text, hoverinfo="text", marker_color="#ff8c42"))
    fig.add_hline(y=0, line_color="#888", line_width=1)
    fig.update_layout(
        **PLOT_LAYOUT,
        margin={"l": 60, "r": 20, "t": 40, "b": 50},
        xaxis_title="IMF index",
        yaxis_title="Cohen's d (low - high)",
        title="Significant IMF-level effect sizes",
    )
    return fig


def render_summary(acc, coef_pairs):
    lines = [html.P(f"Classifier accuracy (test set): {acc:.3f}")]
    coef_items = [
        html.Li(f"{name}: {weight:.3f}") for name, weight in coef_pairs
    ]
    lines.append(html.P("Logistic regression coefficients (scaled features):"))
    lines.append(html.Ul(coef_items))
    return lines


def simulate_arousal_trials(
    n_trials_per_cond,
    fs,
    duration,
    base_seed,
    low_event_scale,
    low_hf_scale,
    low_noise_scale,
    high_event_scale,
    high_hf_scale,
    high_noise_scale,
):
    feature_names = [
        "VLF_power",
        "LF_power",
        "MF_power",
        "HF_power",
        "LF_HF",
        "MF_HF",
        "VLF_LF",
        "LFplusMF_HF",
    ]

    base_events = np.array([10.0, 20.0, 32.0, 45.0])
    jitter_std = 0.25

    X_rows = []
    y_rows = []
    imf_power_rows = []
    trial_index = 0

    conditions = [
        ("low", low_event_scale, low_hf_scale, low_noise_scale),
        ("high", high_event_scale, high_hf_scale, high_noise_scale),
    ]

    for arousal_label, event_scale, hf_scale, noise_scale in conditions:
        for _ in range(n_trials_per_cond):
            seed = base_seed + trial_index
            trial_index += 1
            rng = np.random.default_rng(seed)
            event_times = base_events + rng.normal(0.0, jitter_std, size=base_events.shape)

            t, pupil = simulate_pupil(
                fs=fs,
                duration=duration,
                seed=seed,
                event_scale=event_scale,
                hf_scale=hf_scale,
                noise_scale=noise_scale,
                event_times=event_times.tolist(),
            )
            imfs = emd_decompose(pupil)
            stats = compute_imf_stats(imfs, fs)

            band_powers, _ = compute_band_powers(stats, imfs)
            band_ratios = compute_band_ratios(band_powers)

            feat_vec = np.array(
                [
                    band_powers.get("VLF", 0.0),
                    band_powers.get("LF", 0.0),
                    band_powers.get("MF", 0.0),
                    band_powers.get("HF", 0.0),
                    band_ratios.get("LF_HF", np.nan),
                    band_ratios.get("MF_HF", np.nan),
                    band_ratios.get("VLF_LF", np.nan),
                    band_ratios.get("LFplusMF_HF", np.nan),
                ],
                dtype=float,
            )

            X_rows.append(feat_vec)
            y_rows.append(0 if arousal_label == "low" else 1)
            imf_power_rows.append(imf_power_vector(imfs))

    X = np.vstack(X_rows) if X_rows else np.array([])
    y = np.array(y_rows) if y_rows else np.array([])
    X = np.nan_to_num(X, nan=0.0) if X.size else X

    max_imfs = max((len(vec) for vec in imf_power_rows), default=0)
    imf_power_matrix = np.zeros((len(imf_power_rows), max_imfs))
    for idx, vec in enumerate(imf_power_rows):
        imf_power_matrix[idx, : len(vec)] = vec

    # classifier
    acc = np.nan
    coef_pairs = []
    clf = None
    if X.size:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, stratify=y, random_state=0
        )
        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        coefs = clf.named_steps["logisticregression"].coef_[0]
        coef_pairs = sorted(
            zip(feature_names, coefs),
            key=lambda x: -abs(x[1]),
        )

    # IMF effect sizes
    imf_effects = []
    if max_imfs > 0 and y.size:
        low_mask = y == 0
        high_mask = y == 1
        p_vals = []
        d_vals = []
        f_vals = []
        for imf_idx in range(max_imfs):
            x_low = imf_power_matrix[low_mask, imf_idx]
            x_high = imf_power_matrix[high_mask, imf_idx]
            F_imf, p_imf = f_oneway(x_low, x_high)
            d_imf = cohens_d(x_low, x_high)
            f_vals.append(F_imf)
            p_vals.append(p_imf)
            d_vals.append(d_imf)
        q_vals = fdr_bh(p_vals)
        for idx in range(max_imfs):
            imf_effects.append(
                {
                    "index": idx + 1,
                    "F": f_vals[idx],
                    "p": p_vals[idx],
                    "q": q_vals[idx],
                    "d": d_vals[idx],
                }
            )

    return X, y, feature_names, clf, acc, coef_pairs, imf_effects


def run_arousal_with_parameters(
    n_trials_per_cond,
    fs,
    duration,
    base_seed,
    low_event_scale,
    low_hf_scale,
    low_noise_scale,
    high_event_scale,
    high_hf_scale,
    high_noise_scale,
):
    # Try to call the provided utility with extended parameters if it supports them.
    call_kwargs = {
        "n_trials_per_cond": n_trials_per_cond,
        "fs": fs,
        "duration": duration,
        "base_seed": base_seed,
    }
    extended_kwargs = {
        "low_event_scale": low_event_scale,
        "low_hf_scale": low_hf_scale,
        "low_noise_scale": low_noise_scale,
        "high_event_scale": high_event_scale,
        "high_hf_scale": high_hf_scale,
        "high_noise_scale": high_noise_scale,
    }

    sig = inspect.signature(run_arousal_trial_simulation)
    supports_extended = all(name in sig.parameters for name in extended_kwargs)
    if supports_extended:
        try:
            run_arousal_trial_simulation(**call_kwargs, **extended_kwargs)
        except Exception:
            # ignore errors; fall back to the in-app simulation below
            pass

    return simulate_arousal_trials(
        n_trials_per_cond,
        fs,
        duration,
        base_seed,
        low_event_scale,
        low_hf_scale,
        low_noise_scale,
        high_event_scale,
        high_hf_scale,
        high_noise_scale,
    )


control_row_style = {
    "display": "flex",
    "flexWrap": "wrap",
    "gap": "0.75rem",
    "alignItems": "flex-end",
}

card_style = {
    "backgroundColor": "rgba(40,40,40,0.85)",
    "padding": "1rem",
    "borderRadius": "8px",
    "boxShadow": "0 2px 8px rgba(0,0,0,0.4)",
}


app.layout = html.Div(
    [
        html.Div(
            [
                html.H1("Pupil HHT Lab", style={"color": "#f8f9fa"}),
                html.P(
                    "Interactive sandbox for simulating pupil signals, HHT decompositions, "
                    "event-locked IMFs, and arousal-driven stats.",
                    style={"color": "#ced4da", "maxWidth": "900px"},
                ),
            ],
            style={"padding": "1.5rem 1rem"},
        ),
        dcc.Tabs(
            id="tabs",
            value="single",
            children=[
                dcc.Tab(
                    label="Single Trial HHT",
                    value="single",
                    children=[
                        html.Div(
                            [
                                html.Div(
                                    [
                                        html.Label("fs (Hz)"),
                                        dcc.Input(id="fs-single", type="number", value=100, min=1, step=1),
                                    ]
                                ),
                                html.Div(
                                    [
                                        html.Label("Duration (s)"),
                                        dcc.Input(id="duration-single", type="number", value=60, min=1, step=1),
                                    ]
                                ),
                                html.Div(
                                    [
                                        html.Label("Seed"),
                                        dcc.Input(id="seed-single", type="number", value=0, step=1),
                                    ]
                                ),
                                html.Div(
                                    [
                                        html.Label("event_scale"),
                                        dcc.Input(
                                            id="event-scale-single", type="number", value=1.0, step=0.1, min=0
                                        ),
                                    ]
                                ),
                                html.Div(
                                    [
                                        html.Label("hf_scale"),
                                        dcc.Input(
                                            id="hf-scale-single", type="number", value=1.0, step=0.1, min=0
                                        ),
                                    ]
                                ),
                                html.Div(
                                    [
                                        html.Label("noise_scale"),
                                        dcc.Input(
                                            id="noise-scale-single", type="number", value=0.05, step=0.01, min=0
                                        ),
                                    ]
                                ),
                                html.Button("Run simulation", id="run-single", n_clicks=0, className="btn btn-primary"),
                            ],
                            style=control_row_style | {"padding": "1rem"},
                        ),
                        html.Div(
                            [
                                dcc.Graph(id="single-signal-fig"),
                                dcc.Graph(id="single-imf-fig"),
                                dcc.Graph(id="single-recon-fig"),
                                html.Div(
                                    dash_table.DataTable(
                                        id="imf-table",
                                        style_table={"overflowX": "auto"},
                                        style_header={
                                            "backgroundColor": "#343a40",
                                            "color": "white",
                                            "fontWeight": "bold",
                                        },
                                        style_cell={
                                            "backgroundColor": "rgba(0,0,0,0)",
                                            "color": "#e9ecef",
                                            "padding": "6px",
                                            "border": "1px solid #495057",
                                        },
                                    ),
                                    style=card_style | {"margin": "0 1rem 1.5rem 1rem"},
                                ),
                            ]
                        ),
                    ],
                ),
                dcc.Tab(
                    label="Event-Locked IMFs",
                    value="event",
                    children=[
                        html.Div(
                            [
                                html.Div(
                                    [
                                        html.Label("fs (Hz)"),
                                        dcc.Input(id="fs-event", type="number", value=100, min=1, step=1),
                                    ]
                                ),
                                html.Div(
                                    [
                                        html.Label("Duration (s)"),
                                        dcc.Input(id="duration-event", type="number", value=60, min=1, step=1),
                                    ]
                                ),
                                html.Div(
                                    [
                                        html.Label("Seed"),
                                        dcc.Input(id="seed-event", type="number", value=0, step=1),
                                    ]
                                ),
                                html.Div(
                                    [
                                        html.Label("event_scale"),
                                        dcc.Input(id="event-scale-event", type="number", value=1.0, step=0.1, min=0),
                                    ]
                                ),
                                html.Div(
                                    [
                                        html.Label("hf_scale"),
                                        dcc.Input(id="hf-scale-event", type="number", value=1.0, step=0.1, min=0),
                                    ]
                                ),
                                html.Div(
                                    [
                                        html.Label("noise_scale"),
                                        dcc.Input(id="noise-scale-event", type="number", value=0.05, step=0.01, min=0),
                                    ]
                                ),
                                html.Div(
                                    [
                                        html.Label("Event times (s)"),
                                        dcc.Input(
                                            id="event-times-input",
                                            type="text",
                                            value="10,20,32,45",
                                            placeholder="comma-separated",
                                            style={"minWidth": "180px"},
                                        ),
                                    ]
                                ),
                                html.Div(
                                    [
                                        html.Label("t_pre (s)"),
                                        dcc.Input(id="tpre-event", type="number", value=1.0, step=0.1, min=0),
                                    ]
                                ),
                                html.Div(
                                    [
                                        html.Label("t_post (s)"),
                                        dcc.Input(id="tpost-event", type="number", value=3.0, step=0.1, min=0),
                                    ]
                                ),
                                html.Button(
                                    "Compute event-locked IMFs",
                                    id="run-event",
                                    n_clicks=0,
                                    className="btn btn-primary",
                                ),
                            ],
                            style=control_row_style | {"padding": "1rem"},
                        ),
                        html.Div(
                            [
                                dcc.Graph(id="event-imf-fig"),
                            ],
                            style={"padding": "0 1rem 1.5rem 1rem"},
                        ),
                    ],
                ),
                dcc.Tab(
                    label="Arousal Simulation & Stats",
                    value="arousal",
                    children=[
                        html.Div(
                            [
                                html.Div(
                                    [
                                        html.Label("Trials per condition"),
                                        dcc.Input(id="ntrials-arousal", type="number", value=40, min=1, step=1),
                                    ]
                                ),
                                html.Div(
                                    [
                                        html.Label("low_event_scale"),
                                        dcc.Input(id="low-event-scale", type="number", value=0.9, step=0.05, min=0),
                                    ]
                                ),
                                html.Div(
                                    [
                                        html.Label("low_hf_scale"),
                                        dcc.Input(id="low-hf-scale", type="number", value=1.0, step=0.05, min=0),
                                    ]
                                ),
                                html.Div(
                                    [
                                        html.Label("low_noise_scale"),
                                        dcc.Input(id="low-noise-scale", type="number", value=0.07, step=0.01, min=0),
                                    ]
                                ),
                                html.Div(
                                    [
                                        html.Label("high_event_scale"),
                                        dcc.Input(id="high-event-scale", type="number", value=1.1, step=0.05, min=0),
                                    ]
                                ),
                                html.Div(
                                    [
                                        html.Label("high_hf_scale"),
                                        dcc.Input(id="high-hf-scale", type="number", value=1.3, step=0.05, min=0),
                                    ]
                                ),
                                html.Div(
                                    [
                                        html.Label("high_noise_scale"),
                                        dcc.Input(id="high-noise-scale", type="number", value=0.09, step=0.01, min=0),
                                    ]
                                ),
                                html.Div(
                                    [
                                        html.Label("fs (Hz)"),
                                        dcc.Input(id="fs-arousal", type="number", value=100, min=1, step=1),
                                    ]
                                ),
                                html.Div(
                                    [
                                        html.Label("Duration (s)"),
                                        dcc.Input(id="duration-arousal", type="number", value=60, min=1, step=1),
                                    ]
                                ),
                                html.Button(
                                    "Run arousal simulation",
                                    id="run-arousal",
                                    n_clicks=0,
                                    className="btn btn-primary",
                                ),
                            ],
                            style=control_row_style | {"padding": "1rem"},
                        ),
                        html.Div(
                            [
                                dcc.Graph(id="arousal-features-fig"),
                                dcc.Graph(id="imf-effect-fig"),
                                html.Div(id="arousal-summary", style=card_style | {"margin": "0 1rem 1.5rem 1rem"}),
                            ]
                        ),
                    ],
                ),
            ],
        ),
    ],
    style={"backgroundColor": "#121212"},
)


@app.callback(
    Output("single-signal-fig", "figure"),
    Output("single-imf-fig", "figure"),
    Output("single-recon-fig", "figure"),
    Output("imf-table", "data"),
    Output("imf-table", "columns"),
    Input("run-single", "n_clicks"),
    State("fs-single", "value"),
    State("duration-single", "value"),
    State("seed-single", "value"),
    State("event-scale-single", "value"),
    State("hf-scale-single", "value"),
    State("noise-scale-single", "value"),
)
def update_single_trial(_, fs, duration, seed, event_scale, hf_scale, noise_scale):
    fs = safe_number(fs, 100, int)
    duration = safe_number(duration, 60.0, float)
    seed = safe_number(seed, 0, int)
    event_scale = safe_number(event_scale, 1.0, float)
    hf_scale = safe_number(hf_scale, 1.0, float)
    noise_scale = safe_number(noise_scale, 0.05, float)

    t, pupil = simulate_pupil(
        fs=fs,
        duration=duration,
        seed=seed,
        event_scale=event_scale,
        hf_scale=hf_scale,
        noise_scale=noise_scale,
    )
    imfs = emd_decompose(pupil)
    stats = compute_imf_stats(imfs, fs)

    classifications = [s["classification"] for s in stats]
    physio_indices = [i for i, cls in enumerate(classifications) if cls == "physio"]
    reconstructed = (
        reconstruct_from_imfs(imfs, physio_indices) if imfs.size and physio_indices else np.zeros_like(pupil)
    )

    table_data, table_cols = build_imf_table(stats)
    return (
        build_signal_fig(t, pupil),
        build_imf_fig(t, imfs),
        build_reconstruction_fig(t, pupil, reconstructed),
        table_data,
        table_cols,
    )


@app.callback(
    Output("event-imf-fig", "figure"),
    Input("run-event", "n_clicks"),
    State("fs-event", "value"),
    State("duration-event", "value"),
    State("seed-event", "value"),
    State("event-scale-event", "value"),
    State("hf-scale-event", "value"),
    State("noise-scale-event", "value"),
    State("event-times-input", "value"),
    State("tpre-event", "value"),
    State("tpost-event", "value"),
)
def update_event_locked(_, fs, duration, seed, event_scale, hf_scale, noise_scale, event_text, t_pre, t_post):
    fs = safe_number(fs, 100, int)
    duration = safe_number(duration, 60.0, float)
    seed = safe_number(seed, 0, int)
    event_scale = safe_number(event_scale, 1.0, float)
    hf_scale = safe_number(hf_scale, 1.0, float)
    noise_scale = safe_number(noise_scale, 0.05, float)
    t_pre = safe_number(t_pre, 1.0, float)
    t_post = safe_number(t_post, 3.0, float)

    event_times = parse_event_list(event_text)
    t, pupil = simulate_pupil(
        fs=fs,
        duration=duration,
        seed=seed,
        event_scale=event_scale,
        hf_scale=hf_scale,
        noise_scale=noise_scale,
        event_times=event_times if event_times else None,
    )
    imfs = emd_decompose(pupil)
    t_epoch, imf_evoked = compute_event_locked_imf_average(
        t, imfs, event_times if event_times else [], fs=fs, t_pre=t_pre, t_post=t_post
    )
    return build_event_locked_fig(t_epoch, imf_evoked)


@app.callback(
    Output("arousal-features-fig", "figure"),
    Output("imf-effect-fig", "figure"),
    Output("arousal-summary", "children"),
    Input("run-arousal", "n_clicks"),
    State("ntrials-arousal", "value"),
    State("low-event-scale", "value"),
    State("low-hf-scale", "value"),
    State("low-noise-scale", "value"),
    State("high-event-scale", "value"),
    State("high-hf-scale", "value"),
    State("high-noise-scale", "value"),
    State("fs-arousal", "value"),
    State("duration-arousal", "value"),
)
def update_arousal(_, n_trials, low_event_scale, low_hf_scale, low_noise_scale, high_event_scale, high_hf_scale, high_noise_scale, fs, duration):
    n_trials = safe_number(n_trials, 40, int)
    low_event_scale = safe_number(low_event_scale, 0.9, float)
    low_hf_scale = safe_number(low_hf_scale, 1.0, float)
    low_noise_scale = safe_number(low_noise_scale, 0.07, float)
    high_event_scale = safe_number(high_event_scale, 1.1, float)
    high_hf_scale = safe_number(high_hf_scale, 1.3, float)
    high_noise_scale = safe_number(high_noise_scale, 0.09, float)
    fs = safe_number(fs, 100, int)
    duration = safe_number(duration, 60.0, float)

    X, y, feature_names, clf, acc, coef_pairs, imf_effects = run_arousal_with_parameters(
        n_trials,
        fs,
        duration,
        base_seed=100,
        low_event_scale=low_event_scale,
        low_hf_scale=low_hf_scale,
        low_noise_scale=low_noise_scale,
        high_event_scale=high_event_scale,
        high_hf_scale=high_hf_scale,
        high_noise_scale=high_noise_scale,
    )

    features_to_show = [f for f in ["LF_power", "HF_power", "VLF_LF"] if f in feature_names]
    feature_fig = build_feature_boxplot_fig(X, y, feature_names, features_to_show)
    imf_fig = build_imf_effect_fig(imf_effects)
    summary_children = render_summary(acc if not np.isnan(acc) else 0.0, coef_pairs)
    return feature_fig, imf_fig, summary_children


if __name__ == "__main__":
    # Dash 2.16+ prefers app.run; keep debug for live reloads.
    app.run(debug=True)
