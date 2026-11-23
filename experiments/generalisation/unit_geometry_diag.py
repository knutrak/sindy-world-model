"""U1 failure diagnostic (docs/generalisation/14): speed states and headings of moving units, 100 battles."""

import numpy as np
import pandas as pd

from sindy_wm.envs.scenarios import load_map_info
from sindy_wm.io import storage
from sindy_wm.paths import RAW_DIR

out = []
for scen, exp, mp in (("marines", "exp06_intents", "10m_vs_10m"), ("mixed", "exp08_mixed", "mixed_m_mrd")):
    ap = np.array(load_map_info(mp).attack_point, float)
    s = storage.load_snapshots(RAW_DIR / exp / "train_0", list(range(100)))
    s = s.assign(key=s.team + "_" + s.unit_id.astype(str))
    for _ep, g in s.groupby("episode_idx"):
        steps = np.sort(g.step.unique())
        for t0, t1 in zip(steps[:-1], steps[1:], strict=True):
            a = g[(g.step == t0) & g.alive].set_index("key")
            b = g[(g.step == t1) & g.alive].set_index("key")
            if a.team.nunique() < 2:
                continue
            for k, u in a.iterrows():
                if k not in b.index:
                    continue
                v = b.loc[k, ["vx", "vy"]].to_numpy(float)
                sp = np.linalg.norm(v)
                en = a[a.team != u.team]
                d = np.hypot(en.x - u.x, en.y - u.y)
                j = d.idxmin()

                def cos(p, u=u, v=v, sp=sp):
                    w = p - np.array([u.x, u.y])
                    n = np.linalg.norm(w)
                    return np.nan if sp < 1e-6 or n < 1e-6 else float(v @ w / (sp * n))

                tgt = np.nan
                if u.target_id >= 0:
                    tk = ("red_" if u.team == "blue" else "blue_") + str(int(u.target_id))
                    if tk in a.index:
                        tgt = cos(a.loc[tk, ["x", "y"]].to_numpy(float))
                out.append(
                    (
                        scen,
                        u.team,
                        sp,
                        cos(en.loc[j, ["x", "y"]].to_numpy(float)),
                        tgt,
                        cos(ap) if u.team == "red" else np.nan,
                    )
                )
t = pd.DataFrame(out, columns=["scen", "team", "speed", "cos_nearest", "cos_target", "cos_attackpoint"])
t["state"] = np.where(t.speed < 0.3, "stopped", np.where(t.speed > 2.9, "full speed", "in between"))
pd.set_option("display.width", 200)
print(t.groupby(["scen", "team"]).state.value_counts(normalize=True).unstack().round(3))
w = t[t.state == "full speed"]
print(w.groupby(["scen", "team"])[["cos_nearest", "cos_target", "cos_attackpoint"]].median().round(3))
print(
    w.groupby(["scen", "team"])
    .apply(
        lambda x: pd.Series(
            {"cos_nearest>0.9": (x.cos_nearest > 0.9).mean(), "cos_ap>0.9": (x.cos_attackpoint > 0.9).mean()}
        )
    )
    .round(3)
)
