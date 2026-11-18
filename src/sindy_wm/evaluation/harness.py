"""The common evaluation harness of the final exploration (docs/final_exploration/).

Every experiment fits a model through the same splits and scores it with the same metrics. Results
go to one append-only table, `runs/final_exploration/results.csv`, one row per experiment x split x
fold, keyed by a config hash.

Splits (all by battle, fixed seed):

    cv        5 folds of exp06 `train`: fit on 4/5, score on the other 1/5. Mean ± std over
              folds is the screening number. Fits on 4/5 also give the coefficient stability.
    val       fit on all of `train`, score on exp06 `val`. `val` was used for every choice since
              exp06, so this is a second check, not the screening number.
    size      held-out battle size: fit on `train` battles with <= 25 units in total (81%),
              score on those with >= 26 (the largest 19%; 26-33 units).
    stalkers  fit on all of `train` (marines), score on exp07 `train` (stalkers, 1500 battles),
              scaled without refitting. Only for models that can be scaled.
    test2     Phase 3 only: exp06 `test2`, `test2_cf` and exp07 `test2`. Loading them needs
              the environment variable FINAL_EXPLORATION_PHASE3=1.

Metrics, per split:

    horizon   from start points every 8th step (as in the gates): health error, margin R²,
              alive error (units), distance error at 1, 2, 4, 8 s; one-step (0.125 s) R² of the
              change in health and distance.
    battle    one closed-loop rollout per battle from step 0 to the end, under the real intent
              sequence: normalised RMSE of health over the real battle; winner accuracy; error of
              the true winner's remaining health at the real end; duration error; share of
              rollouts with no end within 60 s; Brier score of the win probability (if the model
              gives one, otherwise of its 0/1 call); blow-up share (non-finite, negative, alive or
              health above the start).
    model     SINDy terms, all fitted numbers, and the coefficients (to coefficients.csv).

A model is any object with `fit(data) -> self`, `rollout(B, stalkers=False) -> (N, T+1, 5)`,
`complexity() -> {"terms": int, "numbers": int}`, `coefficients() -> {name: value}` and,
optionally, `win_probability(B) -> (N,)`.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import subprocess
import time
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path

import numpy as np
import pandas as pd

from sindy_wm.models.intent_model import DT, add_controls, add_rates, make_batch
from sindy_wm.paths import DATA_DIR, PROJECT_ROOT

PROCESSED = DATA_DIR / "processed"
CACHE = PROCESSED / "final_exploration"
OUT = PROJECT_ROOT / "runs" / "final_exploration"
RESULTS, COEFS = OUT / "results.csv", OUT / "coefficients.csv"

SPLIT_SEED = 20261118
N_FOLDS = 5
SIZE_LIMIT = 25  # `size` split: fit on <= 25 units in total, score on more
HORIZONS = (1, 2, 4, 8)
BATTLE_STEPS = 480  # 60 s, the collection's limit
END_ALIVE = 0.05  # a side is eliminated below half a unit (alive 1 = ten units)
END_HEALTH = 1e-3
PHASE3_FLAG = "FINAL_EXPLORATION_PHASE3"
SPLITS = ("cv", "val", "size", "stalkers")


# ------------------------------------------------------------------------------------------- data
def _guard(name: str) -> None:
    if name.startswith("test2") and os.environ.get(PHASE3_FLAG) != "1":
        raise PermissionError(f"{name} is the Phase 3 test set; set {PHASE3_FLAG}=1 to read it")


def load_states(name: str, experiment: str = "exp06_intents") -> pd.DataFrame:
    """A team-state table with controls; `<prefix>cf` joins the six counterfactual sets.

    Counterfactual episodes get ids 10_000 * (k + 1) + episode, so `ep % 10_000` is the seed group.
    """
    _guard(name)
    folder = PROCESSED / experiment
    if name.endswith("cf"):
        parts = []
        for k, f in enumerate(sorted(folder.glob(f"{name}_*.parquet"))):
            d = pd.read_parquet(f).reset_index()
            d["episode_idx"] += 10_000 * (k + 1)
            parts.append(d.set_index(["episode_idx", "step"]))
        return add_controls(pd.concat(parts))
    return add_controls(pd.read_parquet(folder / f"{name}.parquet"))


def rows_with_rates(name: str, window: int = 9, experiment: str = "exp06_intents") -> pd.DataFrame:
    """All steps with smoothed rates (per battle, so no battle sees another), cached on disk."""
    _guard(name)
    path = CACHE / f"rates_{experiment}_{name}_w{window}.parquet"
    if path.exists():
        return pd.read_parquet(path)
    rows = add_rates(load_states(name, experiment), window=window)
    CACHE.mkdir(parents=True, exist_ok=True)
    rows.to_parquet(path)
    return rows


def both_alive(rows: pd.DataFrame) -> pd.DataFrame:
    return rows[(rows["alive_blue"] > 0) & (rows["alive_red"] > 0)]


def select(table: pd.DataFrame, episodes) -> pd.DataFrame:
    return table[table.index.get_level_values(0).isin(np.asarray(episodes))]


def episodes(table: pd.DataFrame) -> np.ndarray:
    return table.index.get_level_values(0).unique().to_numpy()


def folds(train_episodes: np.ndarray, k: int = N_FOLDS, seed: int = SPLIT_SEED) -> list[np.ndarray]:
    """k disjoint sets of battles (the held-out part of each fold)."""
    perm = np.random.default_rng(seed).permutation(np.sort(train_episodes))
    return np.array_split(perm, k)


def battle_sizes(table: pd.DataFrame) -> pd.Series:
    first = table.groupby(level=0)[["n_blue", "n_red"]].first()
    return first["n_blue"] + first["n_red"]


# ---------------------------------------------------------------------------------------- batches
def horizon_batch(states: pd.DataFrame) -> dict:
    """Start points every 8th step with 8 s of truth (as in the gates)."""
    B = make_batch(states)
    B["group"] = B["ep"] % 10_000
    return B


def battle_batch(states: pd.DataFrame) -> dict:
    """One start per battle at step 0, with truth to the battle's end and the real outcome."""
    with np.errstate(invalid="ignore", divide="ignore"):  # the last-slope field is undefined at step 0
        B = make_batch(states, horizon=BATTLE_STEPS, every=10**9, history=0, min_future=1)
    B["group"] = B["ep"] % 10_000
    won = states.groupby(level=0)["battle_won"].first()
    B["won"] = won.loc[B["ep"]].to_numpy().astype(bool)
    valid = ~np.isnan(B["Y"][:, :, 0])
    B["end"] = valid.sum(1) - 1  # index of the last real step
    return B


@dataclass
class FitData:
    """What a model may fit on: one split's training battles. Everything is built on first use."""

    states: pd.DataFrame  # team states with controls (all steps)
    rates: pd.DataFrame  # the same steps with d_<col> rates (SG 9), both sides alive
    raw_rates: pd.DataFrame = field(repr=False, default=None)  # with rates, all steps
    seed: int = 0

    @cached_property
    def batch(self) -> dict:
        return horizon_batch(self.states)

    def subsample(self, n: int = 20_000, seed: int | None = None) -> dict:
        B = self.batch
        rng = np.random.default_rng(self.seed if seed is None else seed)
        idx = rng.choice(len(B["X0"]), min(n, len(B["X0"])), replace=False)
        return {
            k: (v[idx] if isinstance(v, np.ndarray) and len(v) == len(B["X0"]) else v) for k, v in B.items()
        }


# ---------------------------------------------------------------------------------------- metrics
def _r2(y, p) -> float:
    y, p = np.asarray(y, float), np.asarray(p, float)
    den = np.sum((y - y.mean()) ** 2)
    return float(1 - np.sum((p - y) ** 2) / den) if den > 0 else float("nan")


def horizon_metrics(Y: np.ndarray, B: dict) -> dict:
    Yt, out = B["Y"], {}
    ok1 = ~np.isnan(Yt[:, 1, 0])
    for j, name in ((0, "health_blue"), (1, "health_red"), (4, "distance")):
        out[f"step_r2_{name}"] = _r2(Yt[ok1, 1, j] - Yt[ok1, 0, j], Y[ok1, 1, j] - Y[ok1, 0, j])
    for h in HORIZONS:
        n = round(h / DT)
        ok = ~np.isnan(Yt[:, n, 0])
        real = Yt[ok, n, 0] - Yt[ok, n, 1] - (Yt[ok, 0, 0] - Yt[ok, 0, 1])
        pred = Y[ok, n, 0] - Y[ok, n, 1] - (Y[ok, 0, 0] - Y[ok, 0, 1])
        out[f"health_{h}s"] = float(np.abs(Y[ok, n, :2] - Yt[ok, n, :2]).mean())
        out[f"margin_r2_{h}s"] = _r2(real, pred)
        out[f"alive_{h}s"] = float(10 * np.abs(Y[ok, n, 2:4] - Yt[ok, n, 2:4]).mean())
        out[f"distance_{h}s"] = float(np.abs(Y[ok, n, 4] - Yt[ok, n, 4]).mean())
    return out


def rollout_end(Y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """First step at which a side is eliminated in a rollout (or -1), and whether Blue won it.

    With no elimination within the rollout, Blue "wins" if it has more health at the last step.
    """
    out_b = (Y[:, :, 2] < END_ALIVE) | (Y[:, :, 0] <= END_HEALTH)
    out_r = (Y[:, :, 3] < END_ALIVE) | (Y[:, :, 1] <= END_HEALTH)
    either = out_b | out_r
    has_end = either.any(1)
    t_end = np.where(has_end, either.argmax(1), -1)
    idx = np.clip(t_end, 0, None)
    rows = np.arange(len(Y))
    at_end_b, at_end_r = out_b[rows, idx], out_r[rows, idx]
    margin_end = Y[rows, idx, 0] - Y[rows, idx, 1]
    margin_last = Y[:, -1, 0] - Y[:, -1, 1]
    blue_won = np.where(
        has_end,
        np.where(at_end_b & at_end_r, margin_end > 0, at_end_r),
        margin_last > 0,
    )
    return t_end, blue_won


def blown_up(Y: np.ndarray, X0: np.ndarray) -> np.ndarray:
    finite = np.isfinite(Y).all((1, 2))
    Yf = np.nan_to_num(Y)
    neg = (Yf < -1e-9).any((1, 2))
    alive_up = (Yf[:, :, 2:4] > X0[:, None, 2:4] + 1e-9).any((1, 2))
    health_up = (Yf[:, :, 0:2] > X0[:, None, 0:2] + 1e-6).any((1, 2))
    return ~finite | neg | alive_up | health_up


def battle_metrics(Y: np.ndarray, B: dict, p_win: np.ndarray | None = None) -> dict:
    Yt, end, won = B["Y"], B["end"], B["won"]
    n = len(Yt)
    rows = np.arange(n)
    start = np.maximum(Yt[:, 0, :2], 1e-6)
    mask = np.arange(Yt.shape[1])[None, :] <= end[:, None]
    err = (Y[:, :, :2] - Yt[:, :, :2]) / start[:, None, :]
    err = np.where(mask[:, :, None], err, 0.0)
    nrmse = np.sqrt((err**2).sum((1, 2)) / (2 * (end + 1)))
    t_end, blue_won = rollout_end(Y)
    winner_side = np.where(won, 0, 1)
    remaining_err = np.abs(Y[rows, end, winner_side] - Yt[rows, end, winner_side])
    has_end = t_end >= 0
    duration_err = np.abs(np.where(has_end, t_end, Y.shape[1] - 1) - end) * DT
    p = blue_won.astype(float) if p_win is None else np.asarray(p_win, float)
    return {
        "battle_health_nrmse": float(np.nanmean(nrmse)),
        "winner_acc": float(np.mean(blue_won == won)),
        "brier": float(np.mean((p - won) ** 2)),
        "final_health_err": float(np.nanmean(remaining_err)),
        "duration_err_s": float(np.mean(duration_err)),
        "no_end_rate": float(np.mean(~has_end)),
        "blowup_rate": float(np.mean(blown_up(Y, B["X0"]))),
        "n_battles": int(n),
    }


# ------------------------------------------------------------------------------------------ table
def config_hash(config: dict) -> str:
    return hashlib.sha256(json.dumps(config, sort_keys=True, default=str).encode()).hexdigest()[:10]


def _git() -> tuple[str, bool]:
    commit = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, cwd=PROJECT_ROOT
    )
    status = subprocess.run(
        ["git", "status", "--porcelain"], capture_output=True, text=True, cwd=PROJECT_ROOT
    )
    return commit.stdout.strip(), bool(status.stdout.strip())


def done_keys() -> set[tuple[str, str, int]]:
    if not RESULTS.exists():
        return set()
    r = pd.read_csv(RESULTS, usecols=["config_hash", "split", "fold"])
    return set(zip(r["config_hash"], r["split"], r["fold"], strict=True))


def _append(path: Path, frame: pd.DataFrame) -> None:
    """Append rows under an exclusive lock, so parallel runs can share the table."""
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / ".lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            _append_unlocked(path, frame)
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def _append_unlocked(path: Path, frame: pd.DataFrame) -> None:
    if path.exists():
        cols = pd.read_csv(path, nrows=0).columns.tolist()
        new = [c for c in frame.columns if c not in cols]
        if new:  # a new metric: rewrite the header once, keeping every old row unchanged
            old = pd.read_csv(path)
            pd.concat([old, frame], ignore_index=True).to_csv(path, index=False)
            return
        frame.reindex(columns=cols).to_csv(path, mode="a", header=False, index=False)
    else:
        frame.to_csv(path, index=False)


# ----------------------------------------------------------------------------------------- runner
@dataclass
class Experiment:
    """One configuration: `make()` returns an unfitted model; `config` is what gets hashed."""

    name: str  # e.g. "ref/A" or "E/ensemble-stlsq"
    make: callable
    config: dict
    axis: str = "ref"
    note: str = ""

    @property
    def hash(self) -> str:
        return config_hash({"name": self.name, **self.config})


def _split_data(split: str, fold: int, cache: dict) -> tuple[FitData, dict, dict, bool]:
    """FitData, horizon batch, battle batch, stalkers flag for one split (and fold)."""
    if "train_states" not in cache:
        cache["train_states"] = load_states("train")
        cache["train_rates"] = rows_with_rates("train")
    S, R = cache["train_states"], cache["train_rates"]
    eps = episodes(S)
    if split == "cv":
        held = folds(eps)[fold]
        fit_eps = np.setdiff1d(eps, held)
        ev = select(S, held)
    elif split == "val":
        fit_eps = eps
        ev = cache.setdefault("val_states", load_states("val"))
    elif split == "size":
        sizes = battle_sizes(S)
        fit_eps = sizes.index[sizes <= SIZE_LIMIT].to_numpy()
        ev = select(S, sizes.index[sizes > SIZE_LIMIT].to_numpy())
    elif split == "stalkers":
        fit_eps = eps
        ev = cache.setdefault("stalker_states", load_states("train", "exp07_stalkers"))
    elif split.startswith("test2"):
        fit_eps = eps
        name, exp = {
            "test2": ("test2", "exp06_intents"),
            "test2_cf": ("test2_cf", "exp06_intents"),
            "test2_stalkers": ("test2", "exp07_stalkers"),
        }[split]
        ev = load_states(name, exp)
    else:
        raise ValueError(split)
    key = (split, fold, "fit")
    if key not in cache:
        fit_states = S if len(fit_eps) == len(eps) else select(S, fit_eps)
        fit_rates = R if len(fit_eps) == len(eps) else select(R, fit_eps)
        cache[key] = FitData(states=fit_states, rates=both_alive(fit_rates), raw_rates=fit_rates, seed=fold)
    bkey = (split, fold, "batches")
    if bkey not in cache:
        cache[bkey] = (horizon_batch(ev), battle_batch(ev))
    return cache[key], *cache[bkey], split in ("stalkers", "test2_stalkers")


def run(
    experiment: Experiment,
    splits=SPLITS,
    cache: dict | None = None,
    verbose: bool = True,
    cv_folds=None,
) -> pd.DataFrame:
    """Fit and score one experiment on each split (each fold for cv); append new rows to the table.

    Rows already in the table (same config hash, split and fold) are skipped, never overwritten.
    """
    cache = {} if cache is None else cache
    done = done_keys()
    commit, dirty = _git()
    rows, coefs = [], []
    for split in splits:
        for fold in (range(N_FOLDS) if cv_folds is None else cv_folds) if split == "cv" else [0]:
            if (experiment.hash, split, fold) in done:
                continue
            model = experiment.make()
            if split.endswith("stalkers") and not getattr(model, "can_scale", False):
                continue
            data, Bh, Bb, stalkers = _split_data(split, fold, cache)
            t0 = time.time()
            model.fit(data)
            t_fit = time.time() - t0
            Yh = model.rollout(Bh, stalkers=stalkers)
            Yb = model.rollout(Bb, stalkers=stalkers)
            p_win = (
                model.win_probability(Bb, stalkers=stalkers) if hasattr(model, "win_probability") else None
            )
            row = {
                "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                "commit": commit,
                "dirty": dirty,
                "axis": experiment.axis,
                "experiment": experiment.name,
                "config_hash": experiment.hash,
                "split": split,
                "fold": fold,
                **{f"n_{k}": v for k, v in model.complexity().items()},
                **horizon_metrics(Yh, Bh),
                **battle_metrics(Yb, Bb, p_win),
                "fit_seconds": round(t_fit, 1),
                "config": json.dumps(experiment.config, sort_keys=True, default=str),
                "note": experiment.note,
            }
            rows.append(row)
            for k, v in model.coefficients().items():
                coefs.append(
                    {
                        "experiment": experiment.name,
                        "config_hash": experiment.hash,
                        "split": split,
                        "fold": fold,
                        "coefficient": k,
                        "value": float(v),
                    }
                )
            if verbose:
                print(
                    f"{experiment.name:32s} {split:9s} fold {fold}  health 4 s {row['health_4s']:.4f}  "
                    f"margin R² 4 s {row['margin_r2_4s']:+.3f}  "
                    f"battle nRMSE {row['battle_health_nrmse']:.3f}  "
                    f"winner {row['winner_acc']:.3f}  terms {row.get('n_terms')}  ({t_fit:.0f} s)",
                    flush=True,
                )
            _append(RESULTS, pd.DataFrame([row]))
            if coefs:
                _append(COEFS, pd.DataFrame(coefs))
                coefs = []
    return pd.DataFrame(rows)


# -------------------------------------------------------------------------------------- summaries
KEY_METRICS = [
    "health_4s",
    "margin_r2_2s",
    "margin_r2_4s",
    "margin_r2_8s",
    "alive_4s",
    "distance_4s",
    "battle_health_nrmse",
    "winner_acc",
    "brier",
    "final_health_err",
    "duration_err_s",
    "no_end_rate",
    "blowup_rate",
    "n_terms",
    "n_numbers",
]


def load_results() -> pd.DataFrame:
    return pd.read_csv(RESULTS) if RESULTS.exists() else pd.DataFrame()


def summarize(results: pd.DataFrame | None = None, split: str = "cv", metrics=KEY_METRICS) -> pd.DataFrame:
    """Mean ± std over folds per experiment (latest config hash per experiment name)."""
    r = load_results() if results is None else results
    r = r[r["split"] == split]
    latest = r.groupby("experiment")["time"].transform("max")
    hashes = r[r["time"] == latest].groupby("experiment")["config_hash"].first()
    r = r[r["config_hash"].isin(hashes.values)]
    metrics = [m for m in metrics if m in r]
    g = r.groupby(["axis", "experiment"])[metrics]
    mean, std = g.mean(), g.std()
    out = pd.concat({"mean": mean, "std": std}, axis=1).swaplevel(axis=1).sort_index(axis=1)
    out[("folds", "n")] = g.size()
    return out


def stability(split: str = "cv") -> pd.DataFrame:
    """Per coefficient over folds: how often it is non-zero, its mean, and std / |mean|."""
    if not COEFS.exists():
        return pd.DataFrame()
    c = pd.read_csv(COEFS)
    c = c[c["split"] == split]
    g = c.groupby(["experiment", "coefficient"])["value"]
    out = pd.DataFrame({"kept": g.apply(lambda v: float((v != 0).mean())), "mean": g.mean(), "std": g.std()})
    out["cv"] = out["std"] / out["mean"].abs()
    return out


def pareto_plot(
    split: str = "cv", x: str = "n_terms", y: str = "battle_health_nrmse", path: Path | None = None
):
    """Complexity against error, one point per experiment (mean ± std over folds), front highlighted."""
    import matplotlib.pyplot as plt

    s = summarize(split=split, metrics=[x, y])
    s = s.dropna(subset=[(x, "mean"), (y, "mean")])
    fig, ax = plt.subplots(figsize=(7, 4.5))
    xs, ys, es = s[(x, "mean")].to_numpy(), s[(y, "mean")].to_numpy(), s[(y, "std")].fillna(0).to_numpy()
    names = [e for _, e in s.index]
    front = [i for i in range(len(xs)) if not any((xs[j] <= xs[i]) & (ys[j] < ys[i]) for j in range(len(xs)))]
    ax.errorbar(xs, ys, yerr=es, fmt="o", color="0.6", ms=4, lw=1, zorder=1)
    order = sorted(front, key=lambda i: xs[i])
    ax.plot(xs[order], ys[order], "-o", color="C0", ms=5, zorder=2, label="Pareto front")
    for i, n in enumerate(names):
        ax.annotate(n, (xs[i], ys[i]), fontsize=7, xytext=(3, 3), textcoords="offset points")
    ax.set_xscale("symlog", linthresh=10)
    ax.set_xlabel({"n_terms": "SINDy terms", "n_numbers": "fitted numbers"}.get(x, x))
    ax.set_ylabel(y)
    ax.set_title(f"Complexity vs error ({split})")
    ax.legend(frameon=False)
    fig.tight_layout()
    if path is not None:
        fig.savefig(path, dpi=150)
    return fig
