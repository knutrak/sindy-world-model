# Notebooks

Grouped by stage of the project, roughly in the order the work was done. Every notebook finds the project
through `sindy_wm.paths.PROJECT_ROOT`, so it can be run from its own folder. The `.py` files next to the newer
notebooks are their sources (percent format); rebuild a notebook with
`uv run --extra explore python scripts/run_notebook.py <source>.py <notebook>.ipynb <folder>`.

| folder | what | start with |
|---|---|---|
| `00_overview/` | tours of the whole project: recap, model overview, findings summary, how the model code works | `00_recap` |
| `01_early_experiments/` | exp01-exp05: reading an experiment, the aggression / step_mul sweep and first SINDy fits, Lanchester drivers, engagement from distance | `read_experiment` |
| `02_sindyc/` | SINDYc with intents (exp06): pilot, world model, counterfactuals and planner, switches and transfer, why hold wins, reach as a state, what limits the model, derivatives | `sindyc_02_world_model` |
| `03_final_exploration/` | the final marine model (A\*-2k) as claims with caveats, with A, A\*, B, S | `final_01_model_showcase` |
| `04_marauders_mixed/` | marauders alone and mixed marine + marauder armies; SMAClite's range limit of 6 | `mixed_01_marauders` |
| `05_generalisation/` | one equation-discovery pipeline across marines, marauders, mixed and uneven armies; the held-out test | `generalisation_01` |
| `lib/` | helper modules imported by notebooks, scripts and `sindy_wm.evaluation.harness_models`: `sindyc_07_lab.py` (parameter-dict simulator), `weak_sindy.py` (weak-form fitting) | — |

Notes and reports that go with them are in `docs/` (`final_exploration/`, `mixed/`, `generalisation/`).
