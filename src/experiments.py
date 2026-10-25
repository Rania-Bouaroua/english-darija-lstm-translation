"""
Experiment plan.

1) MAIN comparison: Vanilla (baseline) vs CIFG vs WMC, identical configuration.
2) HYPERPARAMETER study, one factor at a time around the BASE configuration, ordered by
   priority. Runs are launched in that order until the time budget is exhausted, so the most
   informative experiments are done first and the notebook never runs for an unbounded time.
   Model choice in this study uses VALIDATION scores only.
"""
import time
from dataclasses import replace

from .train import RunConfig, train_and_evaluate

BASE = dict(emb=256, hidden=256, layers=1, dropout=0.3, optimizer="adam", lr=1e-3,
            batch_size=128, epochs=15, patience=3, clip=1.0, teacher_forcing=1.0,
            init="default", forget_bias=0.0, seed=42)
CELLS = ["vanilla", "cifg", "wmc"]

# (priority, factor, value, cells it is applied to)
HPARAM_PLAN = [
    (1, "lr", 3e-4, CELLS),
    (1, "lr", 3e-3, CELLS),
    (2, "optimizer", "rmsprop", CELLS),
    (3, "hidden", 128, CELLS),
    (4, "batch_size", 64, ["vanilla"]),
    (4, "dropout", 0.1, ["vanilla"]),
    (4, "emb", 128, ["vanilla"]),
    (5, "init", "orthogonal", ["vanilla"]),
]


def main_configs(base=None, **over):
    b = {**BASE, **(base or {}), **over}
    return [RunConfig(run_name=f"main__{c}", cell=c, group="main", **b) for c in CELLS]


def hparam_configs(base=None, plan=None, **over):
    b = {**BASE, **(base or {}), **over}
    out = []
    for prio, factor, value, cells in sorted(plan or HPARAM_PLAN, key=lambda x: x[0]):
        for c in cells:
            cfg = RunConfig(run_name=f"hp__{factor}={value}__{c}", cell=c, group=f"hp_{factor}", **b)
            out.append(replace(cfg, **{factor: value}, notes=f"priority {prio}"))
    return out


def run_with_budget(configs, data, budget_minutes, out_dir, ckpt_dir, log=print):
    """Runs configs in order; does not START a new run once the budget is spent.
    Finished runs (result file exists) are loaded instantly and cost nothing."""
    start, results, skipped = time.time(), [], []
    for cfg in configs:
        spent = (time.time() - start) / 60
        if spent > budget_minutes:
            skipped.append(cfg.run_name)
            continue
        log(f"--- {cfg.run_name}  (budget used: {spent:.1f}/{budget_minutes} min)")
        try:
            results.append(train_and_evaluate(cfg, data, out_dir, ckpt_dir))
        except FloatingPointError as e:   # e.g. divergence with a too-large learning rate
            log(f"    run diverged and was recorded as failed: {e}")
            results.append({"run_name": cfg.run_name, "failed": str(e), "cell": cfg.cell,
                            "group": cfg.group, "config": vars(cfg)})
    if skipped:
        log(f"Time budget reached: {len(skipped)} runs not started -> {skipped}")
    return results, skipped
