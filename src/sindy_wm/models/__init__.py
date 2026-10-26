"""Models of team-level combat dynamics, and how they are fitted, simulated and scored.

    trajectories   to_trajectory, derivative_data: Level A table -> fitting data
    sindy          fit_sindy: polynomial SINDy (a pysindy model)
    lanchester     LanchesterModel, fit_lanchester: the classical square law
    gated          GatedModel, fit_gated: SINDy with every term gated by the opponent
    simulation     simulate, simulate_to_elimination
    metrics        evaluate, summarize_eval, evaluate_horizons, healing_share
    runs           save_run

All three model types share one interface (`predict`, `simulate`, `coefficients`,
`get_feature_names`, `equations`), so everything in `simulation`, `metrics`,
`runs` and `evaluation.plots.plot_predictions` works on any of them. Typical use:

    trajectories = {i: to_trajectory(states.loc[i], ...) for i in index.index}
    model = fit_gated(trajectories, train_ids, STATE_COLUMNS, SMOOTH_WINDOW, degree, threshold)
    metrics = summarize_eval(model, test_ids, trajectories, STATE_COLUMNS, index["battle_won"])
    horizons = evaluate_horizons(model, test_ids, trajectories, STATE_COLUMNS)
    save_run(RUNS_DIR, name, config, model, STATE_COLUMNS, metrics, per_episode, figures)
"""
