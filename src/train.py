"""
Training and evaluation of one configuration.

Protocol (identical for every run):
  * model selection (early stopping, best checkpoint) on VALIDATION cross-entropy
  * after training: validation BLEU/chrF (used to compare hyperparameters) and
    test loss / perplexity / token accuracy / BLEU / chrF (reported once, never used for tuning)
  * BLEU and chrF are case-sensitive (case is meaningful in Latin Darija) and computed on the
    same tokenisation for hypotheses and references (sacrebleu tokenize='none').
A run whose result file already exists is loaded instead of re-trained (resume after a
Colab disconnection).
"""
import copy
import json
import math
import os
import platform
import random
import time
from dataclasses import asdict, dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .data import PAD_ID, BucketBatchSampler, collate
from .model import Seq2Seq, count_parameters, init_weights

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover
    tqdm = lambda x, **k: x


@dataclass
class RunConfig:
    run_name: str
    cell: str = "vanilla"            # vanilla | cifg | wmc | lstmp
    group: str = "main"
    emb: int = 256
    hidden: int = 256
    layers: int = 1
    proj: Optional[int] = None       # LSTMP only
    dropout: float = 0.3
    optimizer: str = "adam"          # adam | rmsprop | sgd
    lr: float = 1e-3
    weight_decay: float = 0.0
    batch_size: int = 128
    epochs: int = 15
    patience: int = 3
    clip: float = 1.0
    teacher_forcing: float = 1.0
    init: str = "default"            # default | xavier | orthogonal
    forget_bias: float = 0.0
    max_decode_len: int = 50
    seed: int = 42
    max_train_batches: Optional[int] = None   # debug / quick mode only
    notes: str = ""
    extra: dict = field(default_factory=dict)


def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def environment():
    env = {"python": platform.python_version(), "torch": torch.__version__,
           "cuda": torch.version.cuda, "device": "cpu"}
    if torch.cuda.is_available():
        env["device"] = torch.cuda.get_device_name(0)
    return env


def make_optimizer(cfg, params):
    if cfg.optimizer == "adam":
        return torch.optim.Adam(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    if cfg.optimizer == "rmsprop":
        return torch.optim.RMSprop(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    if cfg.optimizer == "sgd":
        return torch.optim.SGD(params, lr=cfg.lr, momentum=0.9, weight_decay=cfg.weight_decay)
    raise ValueError(cfg.optimizer)


def loaders(data, batch_size, seed):
    ds = data["ds"]
    mk = lambda d, bs, sh: DataLoader(d, collate_fn=collate,
                                      batch_sampler=BucketBatchSampler(d.lengths, bs, sh, seed))
    return mk(ds["train"], batch_size, True), mk(ds["val"], 256, False), mk(ds["test"], 256, False)


@torch.no_grad()
def evaluate_loss(model, loader, device):
    """Teacher-forced cross-entropy, perplexity and next-token accuracy (PAD ignored)."""
    model.eval()
    crit = nn.CrossEntropyLoss(ignore_index=PAD_ID, reduction="sum")
    tot, n, correct = 0.0, 0, 0
    for src, tgt, _ in loader:
        src, tgt = src.to(device), tgt.to(device)
        logits = model(src, tgt, 1.0)
        gold = tgt[:, 1:]
        tot += crit(logits.reshape(-1, logits.size(-1)), gold.reshape(-1)).item()
        m = gold != PAD_ID
        n += int(m.sum()); correct += int(((logits.argmax(-1) == gold) & m).sum())
    loss = tot / max(1, n)
    return {"loss": loss, "ppl": math.exp(min(loss, 50)), "token_acc": correct / max(1, n)}


@torch.no_grad()
def translate(model, dataset, tgt_vocab, device, max_len=50, batch_size=256):
    """Greedy translation of a whole split, in the dataset's original order."""
    model.eval()
    order = sorted(range(len(dataset)), key=lambda i: dataset.lengths[i])
    hyps = [None] * len(dataset)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()
    for s in range(0, len(order), batch_size):
        idx = order[s:s + batch_size]
        src, _, _ = collate([dataset[i] for i in idx])
        for i, ids in zip(idx, model.greedy(src.to(device), max_len)):
            hyps[i] = " ".join(tgt_vocab.decode(ids))
    if device.type == "cuda":
        torch.cuda.synchronize()
    secs = time.time() - t0
    return hyps, {"decode_seconds": secs, "sentences_per_second": len(dataset) / max(secs, 1e-9)}


def corpus_scores(hyps, refs):
    import sacrebleu
    bleu = sacrebleu.corpus_bleu(hyps, [refs], tokenize="none")
    chrf = sacrebleu.corpus_chrf(hyps, [refs])
    return {"bleu": bleu.score, "chrf": chrf.score, "bleu_bp": bleu.bp,
            "bleu_precisions": [round(p, 2) for p in bleu.precisions]}


def train_and_evaluate(cfg: RunConfig, data, out_dir="results", ckpt_dir="checkpoints", verbose=True):
    run_file = os.path.join(out_dir, "runs", f"{cfg.run_name}.json")
    if os.path.exists(run_file):
        if verbose:
            print(f"[{cfg.run_name}] already done -> loaded from {run_file}")
        with open(run_file, encoding="utf8") as fh:
            return json.load(fh)
    os.makedirs(os.path.dirname(run_file), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "predictions"), exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)

    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_dl, val_dl, test_dl = loaders(data, cfg.batch_size, cfg.seed)
    model = Seq2Seq(len(data["src_vocab"]), len(data["tgt_vocab"]), cfg.cell, cfg.emb,
                    cfg.hidden, cfg.layers, cfg.dropout, cfg.proj)
    init_weights(model, cfg.init, cfg.forget_bias)
    model.to(device)
    params = count_parameters(model)
    opt = make_optimizer(cfg, model.parameters())
    crit = nn.CrossEntropyLoss(ignore_index=PAD_ID)
    if verbose:
        print(f"[{cfg.run_name}] cell={cfg.cell} params={params['total']:,} "
              f"(recurrent enc+dec={params['encoder_recurrent'] + params['decoder_recurrent']:,}) device={device}")

    history, best, best_state, best_epoch, bad = [], float("inf"), None, 0, 0
    t_train = time.time()
    for epoch in range(1, cfg.epochs + 1):
        model.train(); t0 = time.time(); tot, n = 0.0, 0
        bar = tqdm(train_dl, desc=f"{cfg.run_name} ep{epoch}", leave=False, disable=not verbose)
        for b, (src, tgt, _) in enumerate(bar):
            if cfg.max_train_batches and b >= cfg.max_train_batches:
                break
            src, tgt = src.to(device), tgt.to(device)
            opt.zero_grad(set_to_none=True)
            logits = model(src, tgt, cfg.teacher_forcing)
            loss = crit(logits.reshape(-1, logits.size(-1)), tgt[:, 1:].reshape(-1))
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite loss at epoch {epoch}, batch {b} (lr too high?)")
            loss.backward()
            gnorm = nn.utils.clip_grad_norm_(model.parameters(), cfg.clip)
            opt.step()
            tot += loss.item(); n += 1
            if hasattr(bar, "set_postfix") and b % 20 == 0:
                bar.set_postfix(loss=f"{loss.item():.3f}")
        val = evaluate_loss(model, val_dl, device)
        rec = {"epoch": epoch, "train_loss": tot / max(1, n), "val_loss": val["loss"],
               "val_ppl": val["ppl"], "val_token_acc": val["token_acc"],
               "grad_norm": float(gnorm), "epoch_seconds": time.time() - t0}
        history.append(rec)
        if verbose:
            print(f"  ep{epoch:02d} train_loss={rec['train_loss']:.3f} val_loss={val['loss']:.3f} "
                  f"val_ppl={val['ppl']:.1f} val_acc={val['token_acc']:.3f} ({rec['epoch_seconds']:.0f}s)")
        if val["loss"] < best - 1e-4:
            best, best_epoch, bad = val["loss"], epoch, 0
            best_state = copy.deepcopy(model.state_dict())
            torch.save(best_state, os.path.join(ckpt_dir, f"{cfg.run_name}.pt"))
        else:
            bad += 1
            if bad >= cfg.patience:
                if verbose:
                    print(f"  early stopping (best epoch {best_epoch})")
                break
    train_seconds = time.time() - t_train
    model.load_state_dict(best_state)

    # ---- final evaluation of the best checkpoint
    val_loss = evaluate_loss(model, val_dl, device)
    test_loss = evaluate_loss(model, test_dl, device)
    val_hyps, _ = translate(model, data["ds"]["val"], data["tgt_vocab"], device, cfg.max_decode_len)
    test_hyps, speed = translate(model, data["ds"]["test"], data["tgt_vocab"], device, cfg.max_decode_len)
    val_scores = corpus_scores(val_hyps, data["ds"]["val"].refs)
    test_scores = corpus_scores(test_hyps, data["ds"]["test"].refs)
    if verbose:
        msg = f"  BEST ep{best_epoch} | val BLEU={val_scores['bleu']:.2f} chrF={val_scores['chrf']:.2f}"
        if not cfg.group.startswith("hp"):   # hyperparameter runs are judged on validation only
            msg += f" | TEST BLEU={test_scores['bleu']:.2f} chrF={test_scores['chrf']:.2f} ppl={test_loss['ppl']:.1f}"
        print(msg)

    tds = data["ds"]["test"]
    pd.DataFrame({"english": tds.src_text, "reference": tds.refs, "hypothesis": test_hyps,
                  "src_len": [len(t) for t in tds.src_tok]}).to_csv(
        os.path.join(out_dir, "predictions", f"{cfg.run_name}.csv"), index=False)
    result = {"run_name": cfg.run_name, "group": cfg.group, "cell": cfg.cell, "config": asdict(cfg),
              "params": params, "best_epoch": best_epoch, "epochs_run": len(history),
              "history": history, "train_seconds": train_seconds,
              "seconds_per_epoch": float(np.mean([h["epoch_seconds"] for h in history])),
              "val": {**val_loss, **val_scores}, "test": {**test_loss, **test_scores, **speed},
              "data_config": data["config"], "data_report": data["report"], "env": environment()}
    with open(run_file, "w", encoding="utf8") as fh:
        json.dump(result, fh, ensure_ascii=False, indent=2)
    del model, opt
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def results_table(results):
    rows = []
    for r in results:
        rows.append({"run": r["run_name"], "group": r["group"], "cell": r["cell"],
                     "params_total": r["params"]["total"],
                     "params_recurrent": r["params"]["encoder_recurrent"] + r["params"]["decoder_recurrent"],
                     "best_epoch": r["best_epoch"], "s_per_epoch": round(r["seconds_per_epoch"], 1),
                     "val_loss": round(r["val"]["loss"], 4), "val_bleu": round(r["val"]["bleu"], 2),
                     "val_chrf": round(r["val"]["chrf"], 2), "test_loss": round(r["test"]["loss"], 4),
                     "test_ppl": round(r["test"]["ppl"], 2), "test_token_acc": round(r["test"]["token_acc"], 4),
                     "test_bleu": round(r["test"]["bleu"], 2), "test_chrf": round(r["test"]["chrf"], 2),
                     "decode_sent_per_s": round(r["test"]["sentences_per_second"], 1)})
    return pd.DataFrame(rows)
