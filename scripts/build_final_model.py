"""Rebuild a final-exploration model from scratch (exp06 `train` only) and print its equations.

Config: configs/final_model.yaml. Writes data/processed/final_exploration/final_<name>.pkl and, with
--bootstrap N, coefficient intervals from N refits on battles resampled with replacement (closures
held fixed, as in gate G3a) to docs/final_exploration/final_<name>_bootstrap.csv.

    uv run --extra explore scripts/build_final_model.py                  the model named in the config
    uv run --extra explore scripts/build_final_model.py --model B --bootstrap 30
"""

import argparse
import copy
import multiprocessing
import os
import pickle

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402

from sindy_wm.evaluation import harness as H  # noqa: E402
from sindy_wm.evaluation.explore_models import AliveVariantA, CombinedA, resample_battles  # noqa: E402
from sindy_wm.evaluation.harness_models import ModelA, ModelB, ModelS  # noqa: E402
from sindy_wm.paths import PROJECT_ROOT  # noqa: E402

CONFIG = PROJECT_ROOT / "configs" / "final_model.yaml"
DOCS = PROJECT_ROOT / "docs" / "final_exploration"


def SeparateK() -> AliveVariantA:
    """A*-2k (final model from 19 Nov): A's separate k_B, k_R with A*'s reach-regressor alive law."""
    return AliveVariantA(law="reach")


MODELS = {"A*2k": SeparateK, "A*": CombinedA, "A": ModelA, "B": ModelB, "S": ModelS}


def train_data(seed: int = 0) -> H.FitData:
    states, rates = H.load_states("train"), H.rows_with_rates("train")
    return H.FitData(states=states, rates=H.both_alive(rates), raw_rates=rates, seed=seed)


def describe(model) -> str:
    if hasattr(model, "m"):
        lines = []
        m = model.m
        for eq in ("health_blue", "health_red", "distance"):
            terms = " ".join(f"{c:+.4f}·{n}" for n, c in zip(m.names, m.coef[eq], strict=True) if c)
            lines.append(f"  ({eq})' = {terms}")
        for side in ("blue", "red"):
            p = " ".join(f"{v:+.4f}·{k}" for k, v in m.p[side].items() if v)
            lines.append(f"  (ln alive_{side})' = ({p}) · (ln health_{side})'")
        return "\n".join(lines)
    return "\n".join(f"  {k} = {v:+.4f}" for k, v in model.coefficients().items())


def _boot(args):
    name, b, P = args
    data = train_data(seed=b)
    rates = resample_battles(data.rates, 20_000 + b)
    states = resample_battles(data.states, 20_000 + b)
    boot = H.FitData(states=states, rates=rates, raw_rates=rates, seed=b)
    model = MODELS[name]()
    if hasattr(model, "P"):
        model.P = copy.deepcopy(P)  # closures held fixed (G3a convention)
    return model.fit(boot).coefficients()


def main() -> None:
    cfg = yaml.safe_load(CONFIG.read_text())
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default=cfg["model"], choices=list(MODELS))
    parser.add_argument("--bootstrap", type=int, default=cfg.get("bootstrap", 0))
    parser.add_argument("--workers", type=int, default=9)
    args = parser.parse_args()

    model = MODELS[args.model]().fit(train_data())
    H.CACHE.mkdir(parents=True, exist_ok=True)
    out = H.CACHE / f"final_{args.model.replace('*', 'star')}.pkl"
    pickle.dump(model, open(out, "wb"))
    print(f"{args.model}: {model.complexity()}  ->  {out}")
    print(describe(model))
    if args.bootstrap:
        P = getattr(model, "P", None)  # None for B / S: their inner A refits closures per resample
        jobs = [(args.model, b, P) for b in range(args.bootstrap)]
        with multiprocessing.get_context("spawn").Pool(args.workers) as pool:
            fits = pool.map(_boot, jobs)
        full = model.coefficients()
        rows = []
        for k in sorted({k for f in fits for k in f} | set(full)):
            vals = np.array([f.get(k, 0.0) for f in fits], float)
            rows.append(
                {
                    "coefficient": k,
                    "value": full.get(k, 0.0),
                    "kept": float((vals != 0).mean()),
                    "q2.5": float(np.quantile(vals, 0.025)),
                    "q97.5": float(np.quantile(vals, 0.975)),
                    "std": float(vals.std()),
                }
            )
        table = pd.DataFrame(rows)
        path = DOCS / f"final_{args.model.replace('*', 'star')}_bootstrap.csv"
        table.to_csv(path, index=False)
        print(table.round(4).to_string(index=False))
        print("wrote", path)


if __name__ == "__main__":
    main()
