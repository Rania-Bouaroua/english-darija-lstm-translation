"""Exploratory data analysis: statistics (JSON) + figures (PNG), all computed from the data."""
import json
import os
import re
from collections import Counter

import numpy as np
import pandas as pd

from .data import source_key, tokenize_darija, tokenize_en
from .plots import save, setup_style

ARABIC_RE = re.compile(r"[\u0600-\u06FF]")
LATIN_RE = re.compile(r"[A-Za-z]")
DIGIT_IN_WORD_RE = re.compile(r"[A-Za-z]*[2-9][A-Za-z]+|[A-Za-z]+[2-9]")


def raw_profile(df_raw):
    prof = {"rows": len(df_raw), "columns": list(df_raw.columns),
            "missing_per_column": {c: int(df_raw[c].isna().sum()) for c in df_raw.columns}}
    return prof


def dataset_stats(prep):
    d, sp, rep = prep["clean"], prep["splits"], prep["report"]
    tr = sp["train"]
    src_tok = [tokenize_en(s) for s in tr["src"]]
    tgt_tok = [tokenize_darija(t) for t in tr["tgt"]]
    ls, lt = np.array([len(x) for x in src_tok]), np.array([len(x) for x in tgt_tok])
    cs, ct = Counter(t for x in src_tok for t in x), Counter(t for x in tgt_tok for t in x)
    keys = d["src"].map(source_key)
    groups = keys.value_counts()
    tgt_lc = Counter(t.lower() for x in tgt_tok for t in x)

    def oov(split, tok_fn, vocab):
        toks = [t for s in sp[split]["src" if tok_fn is tokenize_en else "tgt"] for t in tok_fn(s)]
        return float(np.mean([t not in vocab.stoi for t in toks])) if toks else 0.0

    script_latin = d["tgt"].map(lambda s: bool(LATIN_RE.search(s))).mean()
    script_arabic = d["tgt"].map(lambda s: bool(ARABIC_RE.search(s))).mean()
    src_has_arabic = d["src"].map(lambda s: bool(ARABIC_RE.search(s))).mean()
    ratio = (lt + 1) / (ls + 1)
    stats = {
        "cleaning": rep,
        "pairs_after_cleaning": len(d),
        "split_sizes": {s: len(sp[s]) for s in sp},
        "unique_english_sources": int(len(groups)),
        "english_sources_with_several_translations": int((groups > 1).sum()),
        "pairs_whose_source_has_several_translations": int(groups[groups > 1].sum()),
        "max_translations_for_one_source": int(groups.max()),
        "english_len_tokens": {"mean": float(ls.mean()), "median": float(np.median(ls)),
                               "p95": float(np.percentile(ls, 95)), "max": int(ls.max())},
        "darija_len_tokens": {"mean": float(lt.mean()), "median": float(np.median(lt)),
                              "p95": float(np.percentile(lt, 95)), "max": int(lt.max())},
        "length_correlation_pearson": float(np.corrcoef(ls, lt)[0, 1]),
        "share_target_truncated_at_max_len": float(np.mean(lt > prep["config"]["max_len"] - 2)),
        "vocab_types_train": {"english": len(cs), "darija": len(ct),
                              "darija_if_lowercased": len(tgt_lc)},
        "vocab_after_min_freq": {"english": len(prep["src_vocab"]), "darija": len(prep["tgt_vocab"])},
        "hapax_share_of_types": {"english": float(np.mean([c == 1 for c in cs.values()])),
                                 "darija": float(np.mean([c == 1 for c in ct.values()]))},
        "oov_token_rate": {
            "val_english": oov("val", tokenize_en, prep["src_vocab"]),
            "test_english": oov("test", tokenize_en, prep["src_vocab"]),
            "val_darija": oov("val", tokenize_darija, prep["tgt_vocab"]),
            "test_darija": oov("test", tokenize_darija, prep["tgt_vocab"])},
        "script": {"target_with_latin_letters": float(script_latin),
                   "target_with_arabic_letters": float(script_arabic),
                   "source_with_arabic_letters": float(src_has_arabic),
                   "darija_tokens_with_digit_letters_(3,7,9...)": float(np.mean(
                       [bool(DIGIT_IN_WORD_RE.fullmatch(t)) for x in tgt_tok for t in x]))},
        "noise_indicators": {
            "pairs_source_equals_target": int((d["src"].str.lower() == d["tgt"].str.lower()).sum()),
            "pairs_length_ratio_gt_3_or_lt_0.33": int(((ratio > 3) | (ratio < 1 / 3)).sum()),
            "single_token_targets": int(np.sum(lt == 1))},
        "top_darija_tokens": ct.most_common(25),
        "top_english_tokens": cs.most_common(25),
    }
    return stats, dict(ls=ls, lt=lt, cs=cs, ct=ct, groups=groups, ratio=ratio, train=tr)


def noisy_examples(prep, k=10):
    d = prep["clean"].copy()
    d["ls"] = d["src"].map(lambda s: len(tokenize_en(s)))
    d["lt"] = d["tgt"].map(lambda s: len(tokenize_darija(s)))
    d["ratio"] = (d["lt"] + 1) / (d["ls"] + 1)
    out = pd.concat([d.sort_values("ratio").head(k // 2), d.sort_values("ratio").tail(k // 2)])
    return out[["src", "tgt", "ls", "lt", "ratio"]]


def several_translations_examples(prep, k=5):
    d = prep["clean"].copy()
    d["key"] = d["src"].map(source_key)
    multi = d.groupby("key").filter(lambda g: len(g) > 2)
    return multi.sort_values("key").groupby("key").head(3).head(3 * k)[["src", "tgt"]]


def make_figures(stats, aux, fig_dir, max_len):
    import matplotlib.pyplot as plt
    setup_style()
    os.makedirs(fig_dir, exist_ok=True)
    ls, lt, cs, ct = aux["ls"], aux["lt"], aux["cs"], aux["ct"]
    caps = {}

    fig, ax = plt.subplots(figsize=(7, 4))
    bins = np.arange(0, min(60, max(ls.max(), lt.max())) + 2)
    ax.hist(ls.clip(max=bins[-1]), bins=bins, alpha=.6, label="English")
    ax.hist(lt.clip(max=bins[-1]), bins=bins, alpha=.6, label="Darija (Latin)")
    ax.axvline(max_len - 2, color="k", ls="--", lw=1, label=f"truncation (max_len={max_len})")
    ax.set(xlabel="tokens per sentence", ylabel="sentence pairs", title="Sentence length distribution (train)")
    ax.legend()
    caps["eda_lengths"] = save(fig, fig_dir, "eda_lengths",
                               "Token length of English sources and Darija targets in the training split.")

    fig, ax = plt.subplots(figsize=(5, 5))
    jitter = lambda a: a + np.random.default_rng(0).uniform(-.3, .3, len(a))
    ax.scatter(jitter(ls), jitter(lt), s=2, alpha=.15)
    m = int(np.percentile(np.r_[ls, lt], 99.5))
    ax.plot([0, m], [0, m], "k--", lw=1); ax.set(xlim=(0, m), ylim=(0, m))
    ax.set(xlabel="English length (tokens)", ylabel="Darija length (tokens)",
           title=f"Length correlation (Pearson r = {stats['length_correlation_pearson']:.2f})")
    caps["eda_length_scatter"] = save(fig, fig_dir, "eda_length_scatter",
                                      "Source vs target length per pair (jittered); dashed line = equal length.")

    fig, ax = plt.subplots(figsize=(7, 4))
    for name, c in (("English", cs), ("Darija", ct)):
        f = sorted(c.values(), reverse=True)
        ax.loglog(range(1, len(f) + 1), f, label=f"{name} ({len(f):,} types)")
    ax.set(xlabel="rank", ylabel="frequency", title="Zipf plot of training tokens"); ax.legend()
    caps["eda_zipf"] = save(fig, fig_dir, "eda_zipf",
                            "Rank-frequency of tokens (log-log). A long tail means many rare words.")

    fig, ax = plt.subplots(figsize=(7, 4))
    g = aux["groups"].clip(upper=6).value_counts().sort_index()
    ax.bar([str(i) if i < 6 else "6+" for i in g.index], g.values)
    ax.set(xlabel="number of Darija translations of the same English sentence",
           ylabel="English sentences", title="One-to-many translations")
    ax.set_yscale("log")
    caps["eda_multi_translations"] = save(fig, fig_dir, "eda_multi_translations",
                                          "How many distinct Darija translations each (normalised) English source has.")

    fig, ax = plt.subplots(figsize=(7, 4))
    ks = ["val_english", "test_english", "val_darija", "test_darija"]
    ax.bar(ks, [100 * stats["oov_token_rate"][k] for k in ks])
    ax.set(ylabel="% tokens mapped to <unk>", title="Out-of-vocabulary rate (vocab fitted on train)")
    caps["eda_oov"] = save(fig, fig_dir, "eda_oov",
                           "Share of validation/test tokens absent from the training vocabulary (min_freq applied).")
    return caps


def run_eda(df_raw, prep, out_dir="results/eda", fig_dir="figures"):
    os.makedirs(out_dir, exist_ok=True)
    stats, aux = dataset_stats(prep)
    stats["raw_profile"] = raw_profile(df_raw)
    caps = make_figures(stats, aux, fig_dir, prep["config"]["max_len"])
    with open(os.path.join(out_dir, "eda_stats.json"), "w", encoding="utf8") as fh:
        json.dump(stats, fh, ensure_ascii=False, indent=2)
    noisy_examples(prep).to_csv(os.path.join(out_dir, "noisy_examples.csv"), index=False)
    several_translations_examples(prep).to_csv(os.path.join(out_dir, "multi_translation_examples.csv"), index=False)
    return stats, caps
