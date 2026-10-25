"""End-to-end test on SYNTHETIC data with the real column format (darija, eng, darija_ar).
Checks: cleaning, leakage-free split, vocabulary, training step, overfitting a tiny set,
greedy decoding, BLEU/chrF computation, resume logic.  Run: python -m tests.test_pipeline"""
import random
import shutil
import tempfile

import pandas as pd
import torch

from src.data import (EOS_ID, group_split, clean, prepare, source_key, tokenize_darija)
from src.model import Seq2Seq
from src.train import RunConfig, train_and_evaluate

LEX = {"i": "ana", "want": "bghit", "to": "n", "eat": "akol", "go": "mchi", "home": "ldar",
       "today": "lyoum", "tomorrow": "ghda", "big": "kbir", "house": "Dar", "my": "dyali",
       "friend": "Sa7bi", "is": "rah", "sick": "mreD", "plane": "Tyyara", "we": "7na"}


def fake_df(n=1500, seed=0):
    rng, rows = random.Random(seed), []
    words = list(LEX)
    for _ in range(n):
        s = [rng.choice(words) for _ in range(rng.randint(2, 9))]
        eng = " ".join(s).capitalize() + rng.choice(["", "!", "?", "."])
        dar = " ".join(LEX[w] for w in s)
        rows.append({"darija": dar, "eng": eng, "darija_ar": "ا"})
    rows += [{"darija": "safi", "eng": None, "darija_ar": "صافي"}] * 5            # missing source
    rows += [rows[0]] * 3                                                          # exact duplicates
    rows += [{"darija": "ana bghit nakol", "eng": "I want to eat", "darija_ar": ""},  # variants
             {"darija": "bghit nakol", "eng": "i want to eat!", "darija_ar": ""}]
    return pd.DataFrame(rows)


def test_cleaning_and_split():
    df = fake_df()
    d, rep = clean(df, "eng", "darija")
    assert rep["missing_source"] == 5 and rep["exact_duplicate_pairs_removed"] >= 3
    sp = group_split(d, seed=1)
    keys = {s: set(sp[s]["src"].map(source_key)) for s in sp}
    assert not (keys["train"] & keys["test"]) and not (keys["train"] & keys["val"])
    # 'I want to eat' and 'i want to eat!' must be in the same split
    k = source_key("I want to eat")
    assert sum(k in keys[s] for s in keys) == 1
    assert tokenize_darija("ghanmreD") == ["ghanmreD"]           # case preserved
    print("cleaning + leakage-free split + case preservation OK")


def test_overfit_and_decode():
    torch.manual_seed(0)
    prep = prepare(fake_df(400), max_len=20, min_freq=1)
    ds = prep["ds"]["train"]
    batch = [ds[i] for i in range(16)]
    from src.data import collate
    src, tgt, _ = collate(batch)
    for cell in ("vanilla", "cifg", "wmc"):
        model = Seq2Seq(len(prep["src_vocab"]), len(prep["tgt_vocab"]), cell, 32, 64, 1, 0.0)
        opt = torch.optim.Adam(model.parameters(), 1e-2)
        crit = torch.nn.CrossEntropyLoss(ignore_index=0)
        for _ in range(150):
            opt.zero_grad()
            lg = model(src, tgt)
            loss = crit(lg.reshape(-1, lg.size(-1)), tgt[:, 1:].reshape(-1))
            loss.backward(); opt.step()
        model.eval()
        hyp = model.greedy(src, 25)
        gold = [[t for t in s[1:] if t not in (0, EOS_ID)] for s in tgt.tolist()]
        acc = sum(h == g for h, g in zip(hyp, gold)) / len(gold)
        assert loss.item() < 0.1 and acc > 0.8, (cell, loss.item(), acc)
        # scheduled-sampling path runs too
        model.train(); model(src, tgt, teacher_forcing=0.5)
        print(f"{cell}: memorises 16 pairs (loss={loss.item():.3f}, exact greedy={acc:.2f})  OK")


def test_full_run_and_resume():
    tmp = tempfile.mkdtemp()
    try:
        prep = prepare(fake_df(1200), max_len=20)
        for cell in ("vanilla", "cifg", "wmc"):
            cfg = RunConfig(run_name=f"t_{cell}", cell=cell, emb=32, hidden=64, epochs=3, batch_size=32)
            r = train_and_evaluate(cfg, prep, out_dir=tmp, ckpt_dir=tmp + "/ck", verbose=False)
            assert 0 <= r["test"]["bleu"] <= 100 and r["params"]["total"] > 0
            print(f"{cell}: full run OK  test BLEU={r['test']['bleu']:.1f} chrF={r['test']['chrf']:.1f} "
                  f"params={r['params']['total']:,}")
        r2 = train_and_evaluate(RunConfig(run_name="t_wmc", cell="wmc"), prep, out_dir=tmp, verbose=False)
        assert r2["config"]["hidden"] == 64           # loaded, not retrained
        print("resume (skip finished run) OK")
    finally:
        shutil.rmtree(tmp)


def run_all():
    test_cleaning_and_split(); test_overfit_and_decode(); test_full_run_and_resume()
    print("ALL PIPELINE TESTS PASSED")


if __name__ == "__main__":
    run_all()
