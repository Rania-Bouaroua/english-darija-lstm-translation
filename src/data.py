"""
Data pipeline for English -> Moroccan Darija translation (imomayiz/darija-english, DODa project).

Facts verified on the Hugging Face page (Oct 2026):
  * two configurations: 'sentences' (87,785 rows) and 'submissions'  -> we use 'sentences'
  * the 'sentences' configuration has ONE split, itself named 'sentences' (not 'train')
  * columns: 'darija' (Latin script / Arabizi), 'eng' (nullable), 'darija_ar' (Arabic script)
  => there is no official train/val/test split: we build our own (see group_split).

Design decisions
  * English is lower-cased. Latin Darija is NOT lower-cased: in the DODa transliteration,
    capital letters encode emphatic consonants (T, D, S ...), e.g. 'ghanmreD', 'TTayaran'.
  * Splits are made by GROUP of normalised English source sentence, so the same English
    sentence (or a trivial variant: case / punctuation) never appears in two splits, even
    when it has several Darija translations -> no source-side leakage.
  * Vocabularies are fitted on the training split only.
"""
import random
import re
import unicodedata
from collections import Counter

import pandas as pd
import torch
from torch.utils.data import Dataset, Sampler

HF_NAME, HF_CONFIG = "imomayiz/darija-english", "sentences"
PAD, BOS, EOS, UNK = "<pad>", "<bos>", "<eos>", "<unk>"
SPECIALS = [PAD, BOS, EOS, UNK]
PAD_ID, BOS_ID, EOS_ID, UNK_ID = 0, 1, 2, 3
TOKEN_RE = re.compile(r"\w+|[^\w\s]", flags=re.UNICODE)


# ----------------------------------------------------------------------------- loading
def load_raw(local_path=None):
    """Returns (DataFrame, info). local_path: optional CSV/Parquet with the same columns
    (useful offline); otherwise downloads the 'sentences' configuration from Hugging Face."""
    if local_path:
        df = pd.read_parquet(local_path) if str(local_path).endswith(".parquet") else pd.read_csv(local_path)
        info = {"source": str(local_path), "splits": ["file"]}
    else:
        from datasets import load_dataset
        dsd = load_dataset(HF_NAME, HF_CONFIG)          # config name is REQUIRED
        split = "train" if "train" in dsd else list(dsd.keys())[0]   # actual name: 'sentences'
        df = dsd[split].to_pandas()
        info = {"source": f"{HF_NAME}/{HF_CONFIG}", "splits": list(dsd.keys()), "used_split": split}
    info.update(rows=len(df), columns=list(df.columns))
    return df, info


def resolve_columns(df, target_script="latin"):
    """Find source/target columns instead of assuming them."""
    def pick(cands):
        for c in cands:
            if c in df.columns:
                return c
        raise ValueError(f"Columns {list(df.columns)} do not contain any of {cands}")
    src = pick(["eng", "english", "en"])
    tgt = pick(["darija_ar", "darija_arabic"]) if target_script == "arabic" else pick(["darija", "darija_latin"])
    return src, tgt


def normalize_text(s):
    s = unicodedata.normalize("NFC", str(s))
    return re.sub(r"\s+", " ", s).strip()


def clean(df, src_col, tgt_col):
    """Drop missing / empty values and exact duplicate pairs. Returns (clean_df, report)."""
    rep = {"rows_raw": len(df),
           "missing_source": int(df[src_col].isna().sum()),
           "missing_target": int(df[tgt_col].isna().sum())}
    d = df[[src_col, tgt_col]].dropna().copy()
    d.columns = ["src", "tgt"]
    d["src"], d["tgt"] = d["src"].map(normalize_text), d["tgt"].map(normalize_text)
    bad = d["src"].str.lower().isin(["", "nan", "none"]) | d["tgt"].str.lower().isin(["", "nan", "none"])
    rep["empty_after_strip"] = int(bad.sum())
    d = d[~bad]
    n = len(d)
    d = d.drop_duplicates(subset=["src", "tgt"]).reset_index(drop=True)
    rep["exact_duplicate_pairs_removed"] = n - len(d)
    rep["rows_clean"] = len(d)
    return d, rep


# ----------------------------------------------------------------------------- tokenisation
def tokenize_en(s):
    return TOKEN_RE.findall(s.lower())


def tokenize_darija(s, lowercase=False):
    return TOKEN_RE.findall(s.lower() if lowercase else s)


def source_key(s):
    """Normalised English used for grouping (case, punctuation and spaces ignored)."""
    return " ".join(re.findall(r"\w+", s.lower()))


# ----------------------------------------------------------------------------- splitting
def group_split(d, val_frac=0.1, test_frac=0.1, seed=42, max_pairs=None):
    """Deterministic split where all pairs sharing the same normalised English source
    go to the same split. max_pairs optionally keeps a random subset of GROUPS
    (stratification is not needed: translation has no class labels)."""
    d = d.copy()
    d["key"] = d["src"].map(source_key)
    groups = sorted(d["key"].unique())
    random.Random(seed).shuffle(groups)
    sizes = d.groupby("key").size().to_dict()
    if max_pairs:
        kept, tot = [], 0
        for g in groups:
            if tot >= max_pairs:
                break
            kept.append(g); tot += sizes[g]
        groups = kept
    total = sum(sizes[g] for g in groups)
    assign, n_test, n_val = {}, 0, 0
    for g in groups:
        if n_test < test_frac * total:
            assign[g] = "test"; n_test += sizes[g]
        elif n_val < val_frac * total:
            assign[g] = "val"; n_val += sizes[g]
        else:
            assign[g] = "train"
    d = d[d["key"].isin(assign)]
    d["split"] = d["key"].map(assign)
    out = {s: d[d["split"] == s].reset_index(drop=True) for s in ("train", "val", "test")}
    # leakage check (must be empty by construction)
    ktr, kva, kte = (set(out[s]["key"]) for s in ("train", "val", "test"))
    assert not (ktr & kva) and not (ktr & kte) and not (kva & kte), "source leakage between splits"
    return out


# ----------------------------------------------------------------------------- vocabulary
class Vocab:
    def __init__(self, token_lists, min_freq=2, max_size=None):
        counts = Counter(t for toks in token_lists for t in toks)
        words = [w for w, c in counts.most_common(max_size) if c >= min_freq and w not in SPECIALS]
        self.itos = SPECIALS + words
        self.stoi = {w: i for i, w in enumerate(self.itos)}
        self.counts = counts

    def __len__(self):
        return len(self.itos)

    def encode(self, toks):
        return [self.stoi.get(t, UNK_ID) for t in toks]

    def decode(self, ids):
        """ids -> tokens, stops at EOS, drops PAD/BOS, keeps <unk> visible."""
        out = []
        for i in ids:
            if i == EOS_ID:
                break
            if i in (PAD_ID, BOS_ID):
                continue
            out.append(self.itos[i])
        return out


class TranslationDataset(Dataset):
    def __init__(self, df, src_vocab, tgt_vocab, max_len=50, darija_lowercase=False):
        self.src_text, self.tgt_text = df["src"].tolist(), df["tgt"].tolist()
        self.src_tok = [tokenize_en(s) for s in self.src_text]
        self.tgt_tok = [tokenize_darija(t, darija_lowercase) for t in self.tgt_text]
        self.src = [src_vocab.encode(t[:max_len - 1]) + [EOS_ID] for t in self.src_tok]
        self.tgt = [[BOS_ID] + tgt_vocab.encode(t[:max_len - 2]) + [EOS_ID] for t in self.tgt_tok]
        self.lengths = [len(s) for s in self.src]
        self.refs = [" ".join(t) for t in self.tgt_tok]   # untruncated tokenised references

    def __len__(self):
        return len(self.src)

    def __getitem__(self, i):
        return self.src[i], self.tgt[i], i


def collate(batch):
    def pad(seqs):
        x = torch.full((len(seqs), max(map(len, seqs))), PAD_ID, dtype=torch.long)
        for i, s in enumerate(seqs):
            x[i, :len(s)] = torch.tensor(s)
        return x
    src, tgt, idx = zip(*batch)
    return pad(src), pad(tgt), torch.tensor(idx)


class BucketBatchSampler(Sampler):
    """Batches of similar source length -> less padding -> faster Python-level unrolling."""

    def __init__(self, lengths, batch_size, shuffle=True, seed=0, bucket_factor=50):
        self.lengths, self.bs, self.shuffle = lengths, batch_size, shuffle
        self.bucket, self.rng = batch_size * bucket_factor, random.Random(seed)

    def __iter__(self):
        idx = list(range(len(self.lengths)))
        if self.shuffle:
            self.rng.shuffle(idx)
        batches = []
        for s in range(0, len(idx), self.bucket):
            chunk = sorted(idx[s:s + self.bucket], key=lambda i: self.lengths[i])
            batches += [chunk[j:j + self.bs] for j in range(0, len(chunk), self.bs)]
        if self.shuffle:
            self.rng.shuffle(batches)
        return iter(batches)

    def __len__(self):
        return (len(self.lengths) + self.bs - 1) // self.bs


def prepare(df_raw, target_script="latin", max_pairs=None, val_frac=0.1, test_frac=0.1,
            seed=42, min_freq=2, max_len=50, darija_lowercase=False):
    """Full pipeline. Returns a dict with splits, vocabularies, datasets and reports."""
    src_col, tgt_col = resolve_columns(df_raw, target_script)
    d, report = clean(df_raw, src_col, tgt_col)
    splits = group_split(d, val_frac, test_frac, seed, max_pairs)
    tr = splits["train"]
    src_vocab = Vocab([tokenize_en(s) for s in tr["src"]], min_freq)
    tgt_vocab = Vocab([tokenize_darija(t, darija_lowercase) for t in tr["tgt"]], min_freq)
    ds = {s: TranslationDataset(splits[s], src_vocab, tgt_vocab, max_len, darija_lowercase) for s in splits}
    report.update(columns_used={"source": src_col, "target": tgt_col},
                  split_sizes={s: len(splits[s]) for s in splits},
                  src_vocab=len(src_vocab), tgt_vocab=len(tgt_vocab))
    return dict(clean=d, splits=splits, src_vocab=src_vocab, tgt_vocab=tgt_vocab, ds=ds,
                report=report, config=dict(target_script=target_script, max_pairs=max_pairs,
                                           val_frac=val_frac, test_frac=test_frac, seed=seed,
                                           min_freq=min_freq, max_len=max_len,
                                           darija_lowercase=darija_lowercase))
