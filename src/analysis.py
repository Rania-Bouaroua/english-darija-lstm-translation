"""Post-hoc analysis of saved test predictions (results/predictions/<run>.csv)."""
import os
import re

import numpy as np
import pandas as pd

BUCKETS = [(1, 5, "1-5"), (6, 10, "6-10"), (11, 15, "11-15"), (16, 10_000, "16+")]


def load_predictions(out_dir, run_name):
    return pd.read_csv(os.path.join(out_dir, "predictions", f"{run_name}.csv"), keep_default_na=False)


def bleu_by_length(pred_df, cell):
    import sacrebleu
    rows = []
    for lo, hi, name in BUCKETS:
        sub = pred_df[(pred_df["src_len"] >= lo) & (pred_df["src_len"] <= hi)]
        if len(sub) == 0:
            continue
        h, r = sub["hypothesis"].tolist(), [sub["reference"].tolist()]
        rows.append({"cell": cell, "bucket": name, "n": len(sub),
                     "bleu": sacrebleu.corpus_bleu(h, r, tokenize="none").score,
                     "chrf": sacrebleu.corpus_chrf(h, r).score})
    return pd.DataFrame(rows)


def _has_repetition(toks):
    if any(toks[i] == toks[i + 1] == toks[i + 2] for i in range(len(toks) - 2)):
        return True
    bigrams = list(zip(toks, toks[1:]))
    return len(bigrams) >= 4 and len(set(bigrams)) <= len(bigrams) / 2


def error_profile(pred_df, cell):
    """Automatic, transparent indicators (heuristics, not a human judgement)."""
    hyp = pred_df["hypothesis"].map(str.split)
    ref = pred_df["reference"].map(str.split)
    ratio = np.array([len(h) / max(1, len(r)) for h, r in zip(hyp, ref)])
    return {"cell": cell, "n": len(pred_df),
            "exact_match_%": 100 * float(np.mean(pred_df["hypothesis"] == pred_df["reference"])),
            "empty_output_%": 100 * float(np.mean([len(h) == 0 for h in hyp])),
            "too_short_(<50%_ref_len)_%": 100 * float(np.mean(ratio < .5)),
            "too_long_(>150%_ref_len)_%": 100 * float(np.mean(ratio > 1.5)),
            "repetition_%": 100 * float(np.mean([_has_repetition(h) for h in hyp])),
            "contains_<unk>_%": 100 * float(np.mean(["<unk>" in h for h in hyp])),
            "mean_len_ratio_hyp/ref": float(ratio.mean())}


def qualitative_table(pred_dfs, n=12, seed=0, max_src_len=None):
    """pred_dfs: {cell: DataFrame}. Same test sentences for every model, picked at random
    (seeded) -> not cherry-picked."""
    base = next(iter(pred_dfs.values()))
    idx = base.index if max_src_len is None else base.index[base["src_len"] <= max_src_len]
    pick = pd.Series(idx).sample(min(n, len(idx)), random_state=seed).sort_values().tolist()
    tab = base.loc[pick, ["english", "reference"]].copy()
    for cell, df in pred_dfs.items():
        tab[cell] = df.loc[pick, "hypothesis"].values
    return tab.reset_index(drop=True)


def sentence_chrf(pred_df):
    import sacrebleu
    return [sacrebleu.sentence_chrf(h, [r]).score for h, r in zip(pred_df["hypothesis"], pred_df["reference"])]
