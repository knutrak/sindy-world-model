# The frozen pipeline's models, fitted on each scenario's train set (21 Nov)

Script: `experiments/generalisation/frozen_models.py`; config: `configs/generalisation_pipeline.yaml`.
R terms: enemy units in reach (×(1 − wd) for Blue attackers), s[t]: share of fire on type t
(proportional rule). danger·wd, walk_r·wd: the withdraw movement rule. Units: target units per second.

## marines

```
H_B^marine' = -0.1770·R[marine]*s[marine]
H_R^marine' = -0.1861·R[marine]*s[marine]
(ln A_B^marine)' = (0.662) · (ln H_B^marine)'
(ln A_R^marine)' = (0.562 +0.260·focus) · (ln H_R^marine)'
d' = -0.485·walk_r*wd +0.575·danger*wd -6.008·close*adv -3.183·close*hold*far -5.915·close*hold*contact
```

reach closure rho: {'marine': 5.75}; complexity {'terms': 10, 'closure numbers': 6}

## marauders

```
H_B^marauder' = -0.1245·R[marauder]*s[marauder]
H_R^marauder' = -0.1257·R[marauder]*s[marauder]
(ln A_B^marauder)' = (0.614) · (ln H_B^marauder)'
(ln A_R^marauder)' = (0.535 +0.159·focus) · (ln H_R^marauder)'
d' = -0.337·walk_r*wd +0.915·danger*wd -5.613·close*adv -3.417·close*hold*far -5.285·close*hold*contact
```

reach closure rho: {'marauder': 6.0}; complexity {'terms': 10, 'closure numbers': 6}

## mixed

```
H_B^marine' = -0.1784·R[marine]*s[marine] -0.1651·R[marauder]*s[marine]
H_B^marauder' = -0.0600·R[marine]*s[marauder] -0.1127·R[marauder]*s[marauder]
H_R^marine' = -0.1892·R[marine]*s[marine] -0.1719·R[marauder]*s[marine]
H_R^marauder' = -0.0599·R[marine]*s[marauder] -0.1170·R[marauder]*s[marauder]
(ln A_B^marine)' = (0.578) · (ln H_B^marine)'
(ln A_B^marauder)' = (0.444) · (ln H_B^marauder)'
(ln A_R^marine)' = (0.506 +0.155·focus) · (ln H_R^marine)'
(ln A_R^marauder)' = (0.376 +0.139·focus) · (ln H_R^marauder)'
d' = -0.340·walk_r*wd +0.708·danger*wd -5.434·close*adv -3.081·close*hold*far -5.263·close*hold*contact
```

reach closure rho: {'marine': 5.84375, 'marauder': 6.0}; complexity {'terms': 19, 'closure numbers': 24}
