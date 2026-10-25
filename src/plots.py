"""Consistent, sober figure style + result figures. Every figure is saved as PNG with a caption."""
import os

import numpy as np

COLORS = {"vanilla": "#4C72B0", "cifg": "#DD8452", "wmc": "#55A868", "lstmp": "#8172B3"}
LABELS = {"vanilla": "Vanilla LSTM", "cifg": "CIFG-LSTM", "wmc": "WMC-LSTM", "lstmp": "LSTMP"}
CAPTIONS = {}


def setup_style():
    import matplotlib.pyplot as plt
    plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 200, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.grid": True, "grid.alpha": .3,
                         "font.size": 10, "axes.titlesize": 11, "legend.frameon": False})


def save(fig, fig_dir, name, caption):
    import matplotlib.pyplot as plt
    os.makedirs(fig_dir, exist_ok=True)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, f"{name}.png"), bbox_inches="tight")
    CAPTIONS[name] = caption
    try:
        get_ipython  # noqa: F821
        plt.show()
    except NameError:
        pass
    plt.close(fig)
    return caption


def learning_curves(results, fig_dir, name="curves_main"):
    import matplotlib.pyplot as plt
    setup_style()
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for r in results:
        h, c = r["history"], COLORS.get(r["cell"], None)
        ep = [x["epoch"] for x in h]
        ax[0].plot(ep, [x["train_loss"] for x in h], "--", color=c, label=f"{LABELS.get(r['cell'], r['cell'])} train")
        ax[0].plot(ep, [x["val_loss"] for x in h], "-", color=c, label=f"{LABELS.get(r['cell'], r['cell'])} val")
        ax[1].plot(ep, [x["val_token_acc"] for x in h], "-o", ms=3, color=c, label=LABELS.get(r["cell"], r["cell"]))
        ax[0].axvline(r["best_epoch"], color=c, lw=.6, alpha=.5)
    ax[0].set(xlabel="epoch", ylabel="cross-entropy", title="Training (dashed) and validation (solid) loss")
    ax[1].set(xlabel="epoch", ylabel="next-token accuracy", title="Validation token accuracy (teacher forcing)")
    for a in ax:
        a.xaxis.get_major_locator().set_params(integer=True); a.legend(fontsize=8)
    return save(fig, fig_dir, name, "Learning curves; thin vertical lines mark the selected (best validation loss) epoch.")


def bar_compare(df, fig_dir, name="metrics_main"):
    import matplotlib.pyplot as plt
    setup_style()
    metrics = [("test_bleu", "test BLEU"), ("test_chrf", "test chrF"), ("params_total", "trainable parameters"),
               ("s_per_epoch", "seconds / epoch")]
    fig, ax = plt.subplots(1, 4, figsize=(14, 3.6))
    for a, (m, lab) in zip(ax, metrics):
        cols = [COLORS.get(c, "grey") for c in df["cell"]]
        a.bar([LABELS.get(c, c) for c in df["cell"]], df[m], color=cols)
        for i, v in enumerate(df[m]):
            a.text(i, v, f"{v:,.2f}" if m != "params_total" else f"{v/1e6:.2f}M", ha="center", va="bottom", fontsize=8)
        a.set_title(lab); a.tick_params(axis="x", rotation=20)
    return save(fig, fig_dir, name, "Test BLEU/chrF of the selected checkpoints, model size and training cost per epoch.")


def perf_vs_complexity(df, fig_dir, name="perf_vs_params"):
    import matplotlib.pyplot as plt
    setup_style()
    fig, ax = plt.subplots(figsize=(6, 4))
    for _, r in df.iterrows():
        ax.scatter(r["params_recurrent"] / 1e6, r["test_chrf"], s=60, color=COLORS.get(r["cell"], "grey"))
        ax.annotate(f"{LABELS.get(r['cell'], r['cell'])}", (r["params_recurrent"] / 1e6, r["test_chrf"]),
                    textcoords="offset points", xytext=(5, 5), fontsize=8)
    ax.set(xlabel="recurrent parameters (encoder + decoder, millions)", ylabel="test chrF",
           title="Translation quality vs recurrent complexity")
    return save(fig, fig_dir, name, "Each point is one architecture trained with the same configuration.")


def length_buckets(bucket_df, fig_dir, name="bleu_by_length"):
    import matplotlib.pyplot as plt
    setup_style()
    fig, ax = plt.subplots(figsize=(7, 4))
    buckets = list(dict.fromkeys(bucket_df["bucket"]))
    cells = list(dict.fromkeys(bucket_df["cell"]))
    w = .8 / len(cells)
    for k, c in enumerate(cells):
        sub = bucket_df[bucket_df["cell"] == c].set_index("bucket").reindex(buckets)
        ax.bar(np.arange(len(buckets)) + k * w, sub["chrf"], w, color=COLORS.get(c), label=LABELS.get(c, c))
    ax.set_xticks(np.arange(len(buckets)) + w * (len(cells) - 1) / 2)
    ax.set_xticklabels([f"{b}\n(n={int(bucket_df[bucket_df['bucket'] == b]['n'].iloc[0])})" for b in buckets])
    ax.set(xlabel="English source length (tokens)", ylabel="test chrF", title="Quality by sentence length"); ax.legend()
    return save(fig, fig_dir, name, "chrF on test sentences grouped by source length (n = sentences per bucket).")


def hparam_effects(df_h, fig_dir, name="hparam_effects"):
    """df_h: rows with columns factor, value, cell, val_chrf (validation only!)."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    setup_style()

    def sort_key(v):
        s = str(v).replace(" (base)", "")
        try:
            return (0, float(s), "")
        except ValueError:
            return (1, 0.0, s)

    order = [c for c in COLORS if c in set(df_h["cell"])]
    factors = list(dict.fromkeys(df_h["factor"]))
    ncols = min(4, len(factors)); nrows = (len(factors) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.6 * ncols, 3.3 * nrows), squeeze=False)
    for a, f in zip(axes.flat, factors):
        sub = df_h[df_h["factor"] == f]
        vals = sorted(dict.fromkeys(sub["value"].astype(str)), key=sort_key)
        cells = [c for c in order if c in set(sub["cell"])]
        w = .8 / len(cells)
        for k, c in enumerate(cells):
            s = sub[sub["cell"] == c].assign(v=lambda x: x["value"].astype(str)).set_index("v").reindex(vals)
            a.bar(np.arange(len(vals)) + k * w, s["val_chrf"], w, color=COLORS.get(c))
        a.set_xticks(np.arange(len(vals)) + w * (len(cells) - 1) / 2); a.set_xticklabels(vals, rotation=15)
        a.set_title(f); a.set_ylabel("validation chrF")
    for a in list(axes.flat)[len(factors):]:
        a.axis("off")
    fig.legend(handles=[Patch(color=COLORS[c], label=LABELS[c]) for c in order],
               loc="upper center", ncol=len(order), bbox_to_anchor=(0.5, 1.06))
    return save(fig, fig_dir, name, "One-factor-at-a-time study: validation chrF when ONE hyperparameter changes "
                                    "(the baseline value is marked '(base)'); test data are not used here.")
