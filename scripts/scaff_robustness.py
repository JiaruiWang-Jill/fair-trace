"""Robustness engine for the SCAFF planted-sensitivity / localization experiment.

This module is the executable core behind
``08XX_FAIR-TRACE_SCAFF_localization_robustness.ipynb``. It exists to fix four defects
found while reproducing the 2026-08-16 prototype (see
``0817_FAIR-TRACE_SCAFF_crossover_summary.ipynb``):

1. ``top1_gap`` is unusable as Rank's primary coordinate: its within-condition repeat-noise
   floor (0.328 movies / 0.370 restaurants in the shared run) exceeds the direct effect it
   estimates (0.094 / 0.156), and being 0/1 it is also blind to a true effect that does not
   happen to change the top item.  -> ``rank_metric_comparison``
2. Repair removability is circular when scored on the repaired stage's own output: the final
   divergence measure *is* the Rank distance, so repairing Rank zeroes it by construction
   whatever the true fault location.  -> ``repair_table``
3. "100% localization" carries no confidence interval and came from one seed.
   -> ``localization_with_ci``, ``seed_sweep``
4. The detection margin tau = 0.10 was hand-set.  -> ``calibrate_margin``, ``margin_sweep``

Plus one reporting distinction: an Explain- or Memory-planted fault cannot move a single-turn
endpoint *at all*, because both stages sit downstream of the ranked list. That is structural
blindness, not low power, and the two are separated by ``masking_table`` / ``two_turn_masking``.

The substrate is the one-domain movie re-implementation of the 0817 notebook, extended with
the stochasticity the deterministic version lacked -- without it there is no noise floor to
study and defect (1) cannot be exhibited at all.

Design invariants worth preserving:

* No LangChain, no LLM, no API key. Stages are transparent functions over explicit parent-state
  dictionaries, so every ``Z_s`` is addressable, holdable and replayable, and it is provable by
  inspection that a stage saw the descriptor only through ``fires()``.
* The crossover holds the stage's exogenous noise ``eps`` fixed across all four cells, matching
  ``Y_s^{i,j} = phi_s(f_s(Z_s^i, A=j; eps))``. Varying eps across cells would fold noise into
  ``D^A``; the noise coordinate is measured *between repeats* instead.
* The planted faults are unit tests of the measurement instrument. They are not estimates of
  real-world disparity and assert nothing about any group.
"""

from __future__ import annotations

import itertools
import math
import zlib
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

STAGES = ["Elicit", "Retrieve", "Rank", "Explain", "Memory"]
DESCRIPTORS = ("man", "woman")
TOP_K = 5
GENRES = ["mystery", "scifi", "comedy", "drama"]
TONES = ["cerebral", "romantic", "playful"]

DEFAULT_SEED = 42
DEFAULT_REPEATS = 8
DEFAULT_N_PER_STAGE = 8

# Stochasticity. Tuned so this substrate sits in the shared run's regime rather than matching
# it exactly (it is one domain, not two, and its catalog is larger): Retrieve repeat-noise
# 0.185 vs 0.166, Rank direct effect 0.089 vs 0.094, Rank natural divergence 0.328 vs 0.396.
# The defect has to be reproduced before it can be shown fixed.
RANK_JITTER = 0.10
# Gumbel temperature for retrieval. 0 = deterministic popularity order, 1 = exact
# popularity-weighted sampling. Tuned so the repeat-noise floors land near the shared run's
# (Retrieve 0.166, Rank top1_gap 0.328) -- the defect must be reproduced before it is fixed.
RETRIEVE_TEMP = 0.06

LEGACY_MARGIN = 0.10  # the hand-set tau this work replaces


# --------------------------------------------------------------------------------------
# substrate
# --------------------------------------------------------------------------------------

def make_catalog(n: int = 60, seed: int = DEFAULT_SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame([{
        "item_id": f"m{i:02d}",
        "genre": GENRES[i % len(GENRES)],
        "tone": TONES[i % len(TONES)],
        "pace": ["slow", "medium", "fast"][i % 3],
        "is_horror": bool(i % 7 == 0),
        "popularity": int(rng.integers(10, 100)),
    } for i in range(n)])


def make_scenarios(n_per_stage: int = DEFAULT_N_PER_STAGE) -> pd.DataFrame:
    """One scenario per (planted stage, replicate), plus controls with nothing planted."""
    rows = []
    for k, (stage, _) in enumerate(itertools.product(["none"] + STAGES, range(n_per_stage))):
        genre = GENRES[k % len(GENRES)]
        rows.append({
            "scenario_id": f"s{k:02d}",
            "message": f"I want a {genre} movie, and please avoid horror.",
            # the latent profile: the reference used only for harm, never for detection
            "true_preferences": {"genre": genre, "tone": "cerebral", "avoid_horror": True},
            "revealed_preferences": {"genre": genre, "avoid_horror": True},
            "planted_stage": stage,
            "planted_target": DESCRIPTORS[k % 2],
        })
    return pd.DataFrame(rows)


def draw_eps(rng: np.random.Generator, n_items: int) -> Dict[str, np.ndarray]:
    """Exogenous noise, keyed by catalog position so it is well defined across candidate sets.

    Held fixed across the four crossover cells; varied across repeats.
    """
    return {
        "rank_jitter": rng.normal(0.0, 1.0, size=n_items),
        "retrieve_u": rng.random(n_items),
    }


def _coerce(value: str):
    """Parse a memory fact / reason tag value back to its original type.

    Carry-over serializes the preference state to ``key=value`` strings; reading it back
    without coercion silently turns ``avoid_horror=True`` into the string ``"True"``, which is
    truthy everywhere but breaks arithmetic downstream.
    """
    if value in ("True", "False"):
        return value == "True"
    return value


def _sid_seed(scenario_id: str) -> int:
    """Stable per-scenario seed. ``hash()`` is salted per process and would silently break
    reproducibility across runs, which is precisely what this module exists to establish."""
    return zlib.crc32(scenario_id.encode()) % (2 ** 31)


def fires(scenario, stage: str, descriptor: str) -> bool:
    """The only way a stage sees the descriptor at all: it is this scenario's planted fault."""
    return scenario["planted_stage"] == stage and scenario["planted_target"] == descriptor


def elicit(parent, scenario, descriptor, catalog, eps):
    prefs = dict(parent["revealed"])
    if fires(scenario, "Elicit", descriptor):
        prefs["tone"] = "romantic"          # unsupported: the latent tone is cerebral
    # Memory channel (inert on turn 1, where no descriptor fact can exist yet): a retained
    # protected descriptor makes the *next* turn's elicitation descriptor-sensitive. This is
    # the documented harm mechanism of memory retention and is an explicit modelling choice.
    if "descriptor" in prefs:
        prefs["tone"] = "romantic" if prefs["descriptor"] == "woman" else "cerebral"
    return {"ask": "tone" not in prefs, "question_target": "tone", "pref_state": prefs}


def retrieve(parent, scenario, descriptor, catalog, eps):
    prefs = parent["pref_state"]
    pool = catalog[catalog["genre"] == prefs.get("genre")]
    if prefs.get("avoid_horror"):
        pool = pool[~pool["is_horror"]]
    if "tone" in prefs:
        matched = pool[pool["tone"] == prefs["tone"]]
        pool = matched if len(matched) else pool
    # Explain channel (inert on turn 1): an unsupported "general audience" framing that the
    # user echoes back broadens retrieval into a popularity shortcut on the next turn.
    if prefs.get("audience") == "general":
        pool = catalog.nlargest(max(len(pool), 1), "popularity")
    if fires(scenario, "Retrieve", descriptor):
        pool = catalog.nlargest(max(len(pool), 1), "popularity")   # popularity shortcut
    if not len(pool):
        pool = catalog
    if RETRIEVE_TEMP > 0 and len(pool) > TOP_K:
        # Gumbel-perturbed popularity order: a mild, tunable amount of retrieval
        # stochasticity, rather than the full weighted resample that would make successive
        # repeats almost disjoint.
        u = np.clip(eps["retrieve_u"][pool.index.to_numpy()], 1e-12, 1 - 1e-12)
        gumbel = -np.log(-np.log(u))
        keys = np.log(pool["popularity"].to_numpy()) + RETRIEVE_TEMP * gumbel
        pool = pool.iloc[np.argsort(-keys)]
    return {"candidates": list(pool["item_id"])[:TOP_K]}


def item_score(row, prefs) -> float:
    return (1.5 * (row["genre"] == prefs.get("genre"))
            + 1.0 * (row["tone"] == prefs.get("tone"))
            - 2.0 * (row["is_horror"] and prefs.get("avoid_horror")))


def rank(parent, scenario, descriptor, catalog, eps):
    cands = catalog[catalog["item_id"].isin(parent["candidates"])].copy()
    if not len(cands):
        return {"ranked": []}
    cands["score"] = [item_score(r, parent["pref_state"]) for _, r in cands.iterrows()]
    if fires(scenario, "Rank", descriptor):
        cands["score"] = cands["score"] + 0.02 * cands["popularity"]   # popularity boost
    cands["score"] = cands["score"] + RANK_JITTER * eps["rank_jitter"][cands.index.to_numpy()]
    cands = cands.sort_values(["score", "item_id"], ascending=[False, True])
    return {"ranked": list(cands["item_id"])}


def explain(parent, scenario, descriptor, catalog, eps):
    prefs = parent["pref_state"]
    tags = [f"genre={prefs.get('genre')}"]
    if "tone" in prefs:
        tags.append(f"tone={prefs['tone']}")
    if fires(scenario, "Explain", descriptor):
        tags.append("audience=general")                          # unsupported framing
    return {"top_item": parent["ranked"][0] if parent["ranked"] else None,
            "reason_tags": sorted(tags)}


def memory(parent, scenario, descriptor, catalog, eps):
    facts = [f"{k}={v}" for k, v in sorted(parent["pref_state"].items())]
    if fires(scenario, "Memory", descriptor):
        facts.append(f"descriptor={descriptor}")                  # retains the descriptor
    return {"facts": sorted(facts)}


STAGE_FN = {"Elicit": elicit, "Retrieve": retrieve, "Rank": rank,
            "Explain": explain, "Memory": memory}


def parent_state(stage: str, scenario, traj):
    """Z_s: the legitimate parent state. Never contains the protected descriptor."""
    if stage == "Elicit":
        return {"message": scenario["message"], "revealed": scenario["revealed_preferences"]}
    if stage == "Retrieve":
        return {"pref_state": traj["Elicit"]["pref_state"]}
    if stage == "Rank":
        return {"pref_state": traj["Elicit"]["pref_state"],
                "candidates": traj["Retrieve"]["candidates"]}
    if stage in ("Explain", "Memory"):
        return {"pref_state": traj["Elicit"]["pref_state"], "ranked": traj["Rank"]["ranked"]}
    raise KeyError(stage)


def run_stage(stage, parent, scenario, descriptor, catalog, eps):
    return STAGE_FN[stage](parent, scenario, descriptor, catalog, eps)


def run_trajectory(scenario, descriptor, catalog, eps):
    traj = {}
    for stage in STAGES:
        traj[stage] = run_stage(stage, parent_state(stage, scenario, traj),
                                scenario, descriptor, catalog, eps)
    return traj


# --------------------------------------------------------------------------------------
# distances
# --------------------------------------------------------------------------------------

def jaccard_distance(a: Sequence, b: Sequence) -> float:
    a, b = set(a), set(b)
    return 0.0 if not (a | b) else 1.0 - len(a & b) / len(a | b)


def top1_gap(x: Sequence[str], y: Sequence[str]) -> float:
    return float(list(x[:1]) != list(y[:1]))


def topk_jaccard(x: Sequence[str], y: Sequence[str], k: int = 3) -> float:
    return jaccard_distance(list(x)[:k], list(y)[:k])


def rbo_distance(x: Sequence[str], y: Sequence[str], p: float = 0.9) -> float:
    """1 - rank-biased overlap. Top-weighted, graded, handles unequal lists."""
    x, y = list(x), list(y)
    if not x and not y:
        return 0.0
    depth = max(len(x), len(y))
    sx, sy, total, weight_sum = set(), set(), 0.0, 0.0
    for d in range(1, depth + 1):
        if d <= len(x):
            sx.add(x[d - 1])
        if d <= len(y):
            sy.add(y[d - 1])
        overlap = len(sx & sy) / d
        w = p ** (d - 1)
        total += w * overlap
        weight_sum += w
    return 1.0 - (total / weight_sum if weight_sum else 0.0)


def kendall_tau_distance(x: Sequence[str], y: Sequence[str]) -> float:
    """Normalized Kendall tau distance over the union, absent items ranked last.

    Implemented directly rather than via scipy so the notebook stays numpy+pandas only.
    """
    x, y = list(x), list(y)
    union = list(dict.fromkeys(x + y))
    n = len(union)
    if n < 2:
        return 0.0
    px = {item: (x.index(item) if item in x else len(x)) for item in union}
    py = {item: (y.index(item) if item in y else len(y)) for item in union}
    discordant = 0
    for a, b in itertools.combinations(union, 2):
        sx = np.sign(px[a] - px[b])
        sy = np.sign(py[a] - py[b])
        if sx * sy < 0:
            discordant += 1
    return discordant / (n * (n - 1) / 2)


def _dcg(items: Sequence[str], rel: Dict[str, float], k: int) -> float:
    return sum(rel.get(it, 0.0) / math.log2(i + 2) for i, it in enumerate(list(items)[:k]))


def ndcg_delta(x: Sequence[str], y: Sequence[str], rel: Dict[str, float], k: int = 5) -> float:
    """|NDCG(x) - NDCG(y)| against the latent-profile relevance. Requires an oracle."""
    ideal_items = sorted(rel, key=lambda it: -rel[it])
    ideal = _dcg(ideal_items, rel, k)
    if ideal <= 0:
        return 0.0
    return abs(_dcg(x, rel, k) - _dcg(y, rel, k)) / ideal


RANK_METRICS = {
    "top1_gap": lambda x, y, rel: top1_gap(x, y),
    "top3_jaccard": lambda x, y, rel: topk_jaccard(x, y, k=3),
    "rbo_p90": lambda x, y, rel: rbo_distance(x, y, p=0.9),
    "kendall_tau": lambda x, y, rel: kendall_tau_distance(x, y),
    "ndcg_delta": lambda x, y, rel: ndcg_delta(x, y, rel, k=5),
}
DEFAULT_RANK_METRIC = "rbo_p90"


def relevance(scenario, catalog) -> Dict[str, float]:
    """Oracle relevance from the latent profile -- used only for NDCG, never for detection."""
    prefs = scenario["true_preferences"]
    return {r["item_id"]: max(0.0, item_score(r, prefs)) for _, r in catalog.iterrows()}


def distance(stage, x, y, rel=None, rank_metric: str = DEFAULT_RANK_METRIC) -> float:
    """d_s: the primary coordinate per stage."""
    if stage == "Elicit":
        return jaccard_distance([f"{k}={v}" for k, v in x["pref_state"].items()],
                                [f"{k}={v}" for k, v in y["pref_state"].items()])
    if stage == "Retrieve":
        return jaccard_distance(x["candidates"], y["candidates"])
    if stage == "Rank":
        return RANK_METRICS[rank_metric](x["ranked"], y["ranked"], rel or {})
    if stage == "Explain":
        return jaccard_distance(x["reason_tags"], y["reason_tags"])
    if stage == "Memory":
        return jaccard_distance(x["facts"], y["facts"])
    raise KeyError(stage)


# --------------------------------------------------------------------------------------
# the audit
# --------------------------------------------------------------------------------------

def audit(scenarios: pd.DataFrame, catalog: pd.DataFrame, seed: int = DEFAULT_SEED,
          repeats: int = DEFAULT_REPEATS, rank_metric: str = DEFAULT_RANK_METRIC,
          rank_metrics: Sequence[str] = (),
          shared_eps: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Run the 2x2 crossover with repeats.

    Returns (coords, pairs). ``coords`` carries one row per (scenario, stage, repeat) with the
    natural / direct / inherited coordinates; ``pairs`` carries one row per (scenario, repeat)
    with the endpoint comparison. Noise is measured *between* repeats and attached to coords.

    ``shared_eps`` controls the single most consequential implementation choice in the whole
    method. The formalism writes the crossover as
    ``Y_s^{i,j} = phi_s(f_s(Z_s^i, A=j; eps))`` -- one ``eps``, shared by all four cells. An
    implementation that instead lets each cell draw its own randomness turns every difference
    between cells into a mixture of the effect and the sampling noise, so ``D^A`` inherits the
    stage's full noise floor. Setting ``shared_eps=False`` reproduces that regime, which is
    what makes it measurable rather than merely asserted.

    ``rank_metrics`` optionally computes the Rank coordinates under several metrics at once,
    adding ``natural__<m>`` / ``direct__<m>`` / ``inherited__<m>`` columns. This is what makes
    the metric comparison an apples-to-apples one: the same trajectories, scored differently.
    """
    a, ap = DESCRIPTORS
    n_items = len(catalog)
    extra = list(rank_metrics)
    coord_rows, pair_rows, natural_cache = [], [], {}

    for _, scenario in scenarios.iterrows():
        rel = relevance(scenario, catalog)
        rng = np.random.default_rng((seed, _sid_seed(scenario["scenario_id"])))

        for r in range(repeats):
            eps = draw_eps(rng, n_items)
            traj = {arm: run_trajectory(scenario, arm, catalog, eps) for arm in (a, ap)}
            natural_cache[(scenario["scenario_id"], r)] = traj
            Z = {arm: {s: parent_state(s, scenario, traj[arm]) for s in STAGES} for arm in (a, ap)}

            per_stage = []
            for stage in STAGES:
                # Y[i, j]: fed arm i's parent state, shown descriptor j, same eps throughout
                if shared_eps:
                    cell_eps = {(i, j): eps for i in (a, ap) for j in (a, ap)}
                else:
                    cell_eps = {(i, j): draw_eps(rng, n_items)
                                for i in (a, ap) for j in (a, ap)}
                Y = {(i, j): run_stage(stage, Z[i][stage], scenario, j, catalog,
                                       cell_eps[(i, j)])
                     for i in (a, ap) for j in (a, ap)}

                def coords_for(metric):
                    d = lambda u, v: distance(stage, u, v, rel, metric)
                    return (d(Y[(a, a)], Y[(ap, ap)]),
                            0.5 * (d(Y[(a, a)], Y[(a, ap)]) + d(Y[(ap, a)], Y[(ap, ap)])),
                            0.5 * (d(Y[(a, a)], Y[(ap, a)]) + d(Y[(a, ap)], Y[(ap, ap)])))

                nat, direct, inherited = coords_for(rank_metric)
                row = {"scenario_id": scenario["scenario_id"], "stage": stage, "repeat": r,
                       "planted_stage": scenario["planted_stage"],
                       "natural": nat, "direct": direct, "inherited": inherited}
                for m in extra:
                    if stage == "Rank":
                        n_m, d_m, i_m = coords_for(m)
                    else:
                        n_m, d_m, i_m = nat, direct, inherited
                    row[f"natural__{m}"], row[f"direct__{m}"], row[f"inherited__{m}"] = n_m, d_m, i_m
                per_stage.append(row)
                coord_rows.append(row)

            endpoint = distance("Rank", traj[a]["Rank"], traj[ap]["Rank"], rel, rank_metric)
            pair_rows.append({
                "scenario_id": scenario["scenario_id"], "repeat": r,
                "planted_stage": scenario["planted_stage"],
                "endpoint_distance": endpoint,
            })

    coords = pd.DataFrame(coord_rows)
    pairs = pd.DataFrame(pair_rows)

    # noise: within-condition distance between distinct repeats of the *same* arm.
    noise_rows = []
    for _, scenario in scenarios.iterrows():
        sid = scenario["scenario_id"]
        rel = relevance(scenario, catalog)
        for stage in STAGES:
            vals, vals_by_metric = [], {m: [] for m in extra}
            for r1, r2 in itertools.combinations(range(repeats), 2):
                for arm in (a, ap):
                    x = natural_cache[(sid, r1)][arm][stage]
                    y = natural_cache[(sid, r2)][arm][stage]
                    vals.append(distance(stage, x, y, rel, rank_metric))
                    for m in extra:
                        vals_by_metric[m].append(
                            distance(stage, x, y, rel, m) if stage == "Rank" else vals[-1])
            row = {"scenario_id": sid, "stage": stage, "noise": float(np.mean(vals))}
            for m in extra:
                row[f"noise__{m}"] = float(np.mean(vals_by_metric[m]))
            noise_rows.append(row)

    noise = pd.DataFrame(noise_rows)
    coords = coords.merge(noise, on=["scenario_id", "stage"], how="left")
    return coords, pairs


def classify(row, margin: float) -> str:
    """The prototype's diagnosis rule, with the margin made an explicit argument."""
    d, z, n = row["direct"] > margin, row["inherited"] > margin, row["natural"] > margin
    if d and z:
        return "mixed"
    if d:
        return "direct"
    if n and z:
        return "inherited"
    return "invariant"


def scenario_coords(coords: pd.DataFrame, metric_suffix: str = "") -> pd.DataFrame:
    """Average the per-repeat coordinates up to one row per (scenario, stage)."""
    cols = [f"natural{metric_suffix}", f"direct{metric_suffix}",
            f"inherited{metric_suffix}", f"noise{metric_suffix}"]
    out = (coords.groupby(["scenario_id", "planted_stage", "stage"], as_index=False)[cols].mean())
    return out.rename(columns={c: c.replace(metric_suffix, "") for c in cols}) \
        if metric_suffix else out


# --------------------------------------------------------------------------------------
# (1) rank-metric comparison
# --------------------------------------------------------------------------------------

def _auc(pos: np.ndarray, neg: np.ndarray) -> float:
    """Mann-Whitney AUC, ties counted as half. No scipy dependency."""
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    wins = (pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum()
    return float(wins / (len(pos) * len(neg)))


def rank_metric_comparison(coords: pd.DataFrame, metrics: Sequence[str]) -> pd.DataFrame:
    """For each candidate Rank coordinate: effect size, noise floor, separation, detection AUC.

    ``effect`` is the mean direct effect at Rank on Rank-planted scenarios. ``floor`` is the
    residual ``D^A`` at Rank where nothing was planted -- the quantity a detection threshold has
    to clear, and the one the original write-up conflated with ``repeat_noise`` (the
    within-condition repeat distance, which bounds how finely ``D^nat`` can be read but does not
    by itself contaminate ``D^A`` when ``eps`` is shared across cells). ``auc`` asks the
    operational question directly: ranking scenarios by Rank's ``D^A``, how cleanly are
    Rank-planted scenarios separated from every other scenario?
    """
    rows = []
    for m in metrics:
        rank_rows = scenario_coords(coords, f"__{m}")
        rank_rows = rank_rows[rank_rows["stage"] == "Rank"]
        planted = rank_rows[rank_rows["planted_stage"] == "Rank"]
        other = rank_rows[rank_rows["planted_stage"] != "Rank"]
        effect = float(planted["direct"].mean())
        noise = float(rank_rows["noise"].mean())
        floor = float(other["direct"].mean())   # residual D^A where nothing was planted
        rows.append({
            "metric": m,
            "effect": effect,                    # mean D^A at Rank, Rank-planted scenarios
            "floor": floor,                      # mean D^A at Rank, everything else
            "separation": effect / floor if floor > 0 else float("inf"),
            "auc": _auc(planted["direct"].to_numpy(), other["direct"].to_numpy()),
            "repeat_noise": noise,               # within-condition, bounds D^nat not D^A
            "effect_over_noise": effect / noise if noise > 0 else float("inf"),
            "usable": bool(effect > floor),
        })
    return pd.DataFrame(rows).sort_values(
        ["auc", "separation"], ascending=False).reset_index(drop=True)


# --------------------------------------------------------------------------------------
# (4) margin calibration
# --------------------------------------------------------------------------------------

def calibrate_margin(coords: pd.DataFrame, quantile: float = 0.95) -> float:
    """Derive tau from the control scenarios' own noise distribution.

    The control scenarios have nothing planted, so every D^A they show is instrument noise.
    Taking an upper quantile of the per-scenario maximum D^A sets tau at the level a
    no-fault scenario exceeds only (1 - quantile) of the time -- a false-positive budget
    rather than a hand-set number.
    """
    sc = scenario_coords(coords)
    controls = sc[sc["planted_stage"] == "none"]
    if not len(controls):
        raise ValueError("no control scenarios: cannot calibrate a margin")
    per_scenario_max = controls.groupby("scenario_id")["direct"].max()
    return float(np.quantile(per_scenario_max.to_numpy(), quantile))


def localize(coords: pd.DataFrame, margin: float) -> pd.DataFrame:
    """Per scenario: the stage carrying the largest D^A, or 'none' if none clears the margin."""
    sc = scenario_coords(coords)
    best = (sc.sort_values("direct", ascending=False)
            .groupby("scenario_id", as_index=False).first())
    best["predicted_stage"] = np.where(best["direct"] > margin, best["stage"], "none")
    best["correct"] = best["predicted_stage"] == best["planted_stage"]
    return best


def margin_sweep(coords: pd.DataFrame, grid: Sequence[float] = ()) -> pd.DataFrame:
    """Accuracy / precision / recall of planted-stage detection as tau varies."""
    grid = list(grid) if len(grid) else list(np.round(np.arange(0.0, 0.605, 0.01), 3))
    rows = []
    for tau in grid:
        best = localize(coords, tau)
        planted = best[best["planted_stage"] != "none"]
        flagged = best[best["predicted_stage"] != "none"]
        rows.append({
            "tau": tau,
            "accuracy": float(best["correct"].mean()),
            # recall: of scenarios that really had a plant, how many were located correctly
            "recall": float(planted["correct"].mean()) if len(planted) else float("nan"),
            # precision: of scenarios we flagged, how many named the right stage
            "precision": float(flagged["correct"].mean()) if len(flagged) else float("nan"),
            "false_positive_rate": float(
                (best[best["planted_stage"] == "none"]["predicted_stage"] != "none").mean()),
            "n_flagged": int(len(flagged)),
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------
# (3) localization with a confidence interval
# --------------------------------------------------------------------------------------

def wilson_interval(k: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    """Wilson score interval -- the right one at proportions near 1, where Wald degenerates."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    denom = 1 + z ** 2 / n
    centre = (p + z ** 2 / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def localization_with_ci(coords: pd.DataFrame, margin: float, n_boot: int = 2000,
                         seed: int = DEFAULT_SEED) -> Dict[str, object]:
    best = localize(coords, margin)
    correct = best["correct"].to_numpy()
    n, k = len(correct), int(correct.sum())
    rng = np.random.default_rng(seed)
    boot = np.array([rng.choice(correct, size=n, replace=True).mean() for _ in range(n_boot)])
    lo_w, hi_w = wilson_interval(k, n)
    return {
        "accuracy": k / n,
        "n_scenarios": n,
        "n_correct": k,
        "wilson_lo": lo_w, "wilson_hi": hi_w,
        "boot_lo": float(np.quantile(boot, 0.025)),
        "boot_hi": float(np.quantile(boot, 0.975)),
        "confusion": pd.crosstab(best["planted_stage"], best["predicted_stage"]),
        "misses": best[~best["correct"]],
    }


# --------------------------------------------------------------------------------------
# (2) repair, scored without circularity
# --------------------------------------------------------------------------------------

def _patched_trajectory(scenario, repair_stage, base, catalog, eps, arm, other):
    """Replace repair_stage's output with the counterpart arm's and re-run its descendants."""
    patched, idx = {}, STAGES.index(repair_stage)
    for k, stage in enumerate(STAGES):
        if k < idx:
            patched[stage] = base[arm][stage]
        elif k == idx:
            patched[stage] = base[other][stage]                  # the repair itself
        else:
            patched[stage] = run_stage(stage, parent_state(stage, scenario, patched),
                                       scenario, arm, catalog, eps)   # descendants re-run
    return patched


def repair_table(scenarios: pd.DataFrame, catalog: pd.DataFrame, planted_stage: str,
                 seed: int = DEFAULT_SEED, repeats: int = DEFAULT_REPEATS,
                 rank_metric: str = DEFAULT_RANK_METRIC,
                 endpoint: str = "Rank") -> pd.DataFrame:
    """Repair each stage in turn on faults planted at ``planted_stage``.

    Two scorings, both non-circular, because the original one was not:

    * ``fixed_endpoint`` -- divergence at ``endpoint`` (default Rank, the paper's core outcome),
      reported ONLY for repairs strictly upstream of it. Scoring a Rank repair on the Rank
      distance is what made the original table meaningless: pasting in the counterpart's ranked
      list zeroes that distance by construction, whatever the true fault location.
    * ``downstream_mean`` -- mean divergence over every stage strictly after the repaired stage.
      Defined for all repairs except the last stage, and never includes the repaired stage.
    """
    a, ap = DESCRIPTORS
    n_items, ep_idx = len(catalog), STAGES.index(endpoint)
    subset = scenarios[scenarios["planted_stage"] == planted_stage]
    rows = []

    for _, scenario in subset.iterrows():
        rel = relevance(scenario, catalog)
        rng = np.random.default_rng((seed, _sid_seed(scenario["scenario_id"])))
        for r in range(repeats):
            eps = draw_eps(rng, n_items)
            base = {arm: run_trajectory(scenario, arm, catalog, eps) for arm in (a, ap)}

            def diverge(traj_x, traj_y, stages):
                if not stages:
                    return float("nan")
                return float(np.mean([distance(s, traj_x[s], traj_y[s], rel, rank_metric)
                                      for s in stages]))

            for repair_stage in STAGES:
                idx = STAGES.index(repair_stage)
                patched = _patched_trajectory(scenario, repair_stage, base, catalog, eps, a, ap)
                downstream = STAGES[idx + 1:]

                fixed_before = fixed_after = float("nan")
                if idx < ep_idx:
                    fixed_before = diverge(base[a], base[ap], [endpoint])
                    fixed_after = diverge(patched, base[ap], [endpoint])

                rows.append({
                    "scenario_id": scenario["scenario_id"], "repeat": r,
                    "planted_stage": planted_stage, "repair_stage": repair_stage,
                    "fixed_before": fixed_before, "fixed_after": fixed_after,
                    "fixed_removability": fixed_before - fixed_after,
                    "downstream_before": diverge(base[a], base[ap], downstream),
                    "downstream_after": diverge(patched, base[ap], downstream),
                    "downstream_removability": (diverge(base[a], base[ap], downstream)
                                                - diverge(patched, base[ap], downstream)),
                    "scoreable_at_endpoint": idx < ep_idx,
                    "n_downstream": len(downstream),
                })

    df = pd.DataFrame(rows)
    agg = (df.groupby("repair_stage")[
        ["fixed_before", "fixed_after", "fixed_removability",
         "downstream_before", "downstream_after", "downstream_removability"]]
        .mean().reindex(STAGES))
    # Normalized: the fraction of the divergence that was actually there to remove. Raw
    # removability is not comparable across repair stages, because each one is scored over a
    # different downstream stage set; the fraction is.
    with np.errstate(invalid="ignore", divide="ignore"):
        agg["fixed_frac"] = agg["fixed_removability"] / agg["fixed_before"]
        agg["downstream_frac"] = agg["downstream_removability"] / agg["downstream_before"]
    agg["scoreable_at_endpoint"] = [STAGES.index(s) < ep_idx for s in agg.index]
    agg["n_downstream"] = [len(STAGES) - STAGES.index(s) - 1 for s in agg.index]
    return agg


def legacy_repair_matrix(scenarios: pd.DataFrame, catalog: pd.DataFrame,
                         seed: int = DEFAULT_SEED, repeats: int = DEFAULT_REPEATS,
                         rank_metric: str = DEFAULT_RANK_METRIC) -> pd.DataFrame:
    """The same matrix under the original circular scoring, so the defect shows a number.

    Every cell is scored on the Rank distance, including the cell that repairs Rank itself --
    which is what lets a repair of the ranker report full removability for a fault planted at
    Elicit.
    """
    rows = {}
    for planted in STAGES:
        tab = legacy_repair_table(scenarios, catalog, planted, seed=seed, repeats=repeats,
                                  rank_metric=rank_metric)
        with np.errstate(invalid="ignore", divide="ignore"):
            rows[planted] = tab["removability"] / tab["before"]
    return pd.DataFrame(rows).T.reindex(STAGES)[STAGES]


def diagnosis_recovery(coords: pd.DataFrame, margin: float) -> pd.DataFrame:
    """Per-scenario diagnosis *at the planted stage*, which is the reading that means anything.

    A corpus-mean diagnosis is degenerate once tau is calibrated near zero: every stage carries
    a nonzero mean D^A simply because some scenarios plant there. The diagnosis rule delivers a
    per-conversation verdict, so it has to be scored per conversation.
    """
    sc = scenario_coords(coords)
    sc["diagnosis"] = sc.apply(lambda r: classify(r, margin), axis=1)
    at_plant = sc[sc["stage"] == sc["planted_stage"]]
    tab = (pd.crosstab(at_plant["planted_stage"], at_plant["diagnosis"])
           .reindex(STAGES).fillna(0).astype(int))
    tab["n"] = tab.sum(axis=1)
    for col in ("direct", "mixed"):
        if col not in tab:
            tab[col] = 0
    tab["flagged_direct_or_mixed"] = (tab["direct"] + tab["mixed"]) / tab["n"]
    return tab


def repair_matrix(scenarios: pd.DataFrame, catalog: pd.DataFrame, seed: int = DEFAULT_SEED,
                  repeats: int = DEFAULT_REPEATS, rank_metric: str = DEFAULT_RANK_METRIC,
                  column: str = "downstream_frac") -> pd.DataFrame:
    """Planted stage x repaired stage. The diagonal is the claim; the off-diagonal is the control.

    A trustworthy repair experiment puts mass on the diagonal only: repairing the stage the
    crossover implicates removes the downstream disparity, and repairing any other stage does
    not. Scored on stages strictly downstream of the repair, so no cell is circular.
    """
    rows = {}
    for planted in STAGES:
        tab = repair_table(scenarios, catalog, planted, seed=seed, repeats=repeats,
                           rank_metric=rank_metric)
        rows[planted] = tab[column]
    return pd.DataFrame(rows).T.reindex(STAGES)[STAGES]


def legacy_repair_table(scenarios: pd.DataFrame, catalog: pd.DataFrame, planted_stage: str,
                        seed: int = DEFAULT_SEED, repeats: int = DEFAULT_REPEATS,
                        rank_metric: str = DEFAULT_RANK_METRIC) -> pd.DataFrame:
    """The original, circular scoring -- kept so the defect can be shown rather than asserted."""
    a, ap = DESCRIPTORS
    n_items = len(catalog)
    subset = scenarios[scenarios["planted_stage"] == planted_stage]
    rows = []
    for _, scenario in subset.iterrows():
        rel = relevance(scenario, catalog)
        rng = np.random.default_rng((seed, _sid_seed(scenario["scenario_id"])))
        for r in range(repeats):
            eps = draw_eps(rng, n_items)
            base = {arm: run_trajectory(scenario, arm, catalog, eps) for arm in (a, ap)}
            before = distance("Rank", base[a]["Rank"], base[ap]["Rank"], rel, rank_metric)
            for repair_stage in STAGES:
                patched = _patched_trajectory(scenario, repair_stage, base, catalog, eps, a, ap)
                after = distance("Rank", patched["Rank"], base[ap]["Rank"], rel, rank_metric)
                rows.append({"repair_stage": repair_stage, "before": before, "after": after,
                             "removability": before - after})
    return (pd.DataFrame(rows).groupby("repair_stage")[["before", "after", "removability"]]
            .mean().reindex(STAGES))


# --------------------------------------------------------------------------------------
# endpoint masking: structural blindness vs low power
# --------------------------------------------------------------------------------------

def masking_table(coords: pd.DataFrame, pairs: pd.DataFrame, margin: float,
                  endpoint_margin: float = 0.05) -> Dict[str, object]:
    """Process sensitivity vs endpoint difference, split by whether the endpoint *could* move.

    Explain and Memory sit downstream of the ranked list, so in a single-turn pipeline a fault
    planted there cannot change the endpoint at all. Pooling those pairs into one masking rate
    conflates "the audit is underpowered" with "the endpoint is blind by construction". They are
    reported separately here, and the structural claim is verified rather than asserted.
    """
    sc = scenario_coords(coords)
    sens = (sc.assign(flag=sc["direct"] > margin)
            .groupby(["scenario_id", "planted_stage"], as_index=False)["flag"].any()
            .rename(columns={"flag": "process_sensitive"}))
    ep = (pairs.groupby(["scenario_id", "planted_stage"], as_index=False)["endpoint_distance"]
          .mean())
    df = sens.merge(ep, on=["scenario_id", "planted_stage"])
    df["endpoint_different"] = df["endpoint_distance"] > endpoint_margin
    df["downstream_of_endpoint"] = df["planted_stage"].isin(["Explain", "Memory"])

    def crosstab(frame):
        return pd.crosstab(
            frame["process_sensitive"].map({True: "process sensitive",
                                            False: "no direct process sensitivity"}),
            frame["endpoint_different"].map({True: "endpoint different",
                                             False: "endpoint equivalent"}))

    can_move = df[~df["downstream_of_endpoint"]]
    blind = df[df["downstream_of_endpoint"]]
    masked = int(((can_move["process_sensitive"]) & (~can_move["endpoint_different"])).sum())
    return {
        "all": crosstab(df),
        "endpoint_can_move": crosstab(can_move),
        "structurally_blind": crosstab(blind),
        "masked_rate_power": masked / len(can_move) if len(can_move) else float("nan"),
        "n_can_move": len(can_move),
        "n_blind": len(blind),
        # the structural claim, verified: no Explain/Memory-planted pair moves the endpoint
        "blind_max_endpoint_distance": float(blind["endpoint_distance"].max()) if len(blind) else 0.0,
        "frame": df,
    }


def two_turn_masking(scenarios: pd.DataFrame, catalog: pd.DataFrame, margin: float,
                     seed: int = DEFAULT_SEED, repeats: int = DEFAULT_REPEATS,
                     rank_metric: str = DEFAULT_RANK_METRIC,
                     endpoint_margin: float = 0.05) -> pd.DataFrame:
    """Re-run the endpoint comparison with a follow-up turn, so Explain/Memory can reach it.

    Carry-over model, stated explicitly because it is a modelling choice and not a measurement:
    the agent's memory facts and the reason tags it gave are both echoed into the next turn's
    revealed preferences -- the user continues the conversation from what the agent said and
    stored. That gives a retained descriptor and an unsupported framing a causal path to the
    turn-2 ranking, which they structurally cannot have within turn 1.
    """
    a, ap = DESCRIPTORS
    n_items = len(catalog)
    rows = []
    for _, scenario in scenarios.iterrows():
        rel = relevance(scenario, catalog)
        rng = np.random.default_rng((seed, _sid_seed(scenario["scenario_id"])))
        for r in range(repeats):
            eps1 = draw_eps(rng, n_items)
            eps2 = draw_eps(rng, n_items)
            t2 = {}
            for arm in (a, ap):
                traj1 = run_trajectory(scenario, arm, catalog, eps1)
                revealed = dict(scenario["revealed_preferences"])
                for fact in traj1["Memory"]["facts"] + traj1["Explain"]["reason_tags"]:
                    k, _, v = fact.partition("=")
                    revealed[k] = _coerce(v)
                sc2 = dict(scenario)
                sc2["revealed_preferences"] = revealed
                t2[arm] = run_trajectory(sc2, arm, catalog, eps2)
            rows.append({
                "scenario_id": scenario["scenario_id"],
                "planted_stage": scenario["planted_stage"],
                "turn2_endpoint_distance": distance("Rank", t2[a]["Rank"], t2[ap]["Rank"],
                                                    rel, rank_metric),
            })
    df = pd.DataFrame(rows)
    out = (df.groupby(["scenario_id", "planted_stage"], as_index=False)["turn2_endpoint_distance"]
           .mean())
    out["turn2_endpoint_different"] = out["turn2_endpoint_distance"] > endpoint_margin
    return out


# --------------------------------------------------------------------------------------
# seed sweep
# --------------------------------------------------------------------------------------

def seed_sweep(seeds: Sequence[int], n_per_stage: int = DEFAULT_N_PER_STAGE,
               repeats: int = DEFAULT_REPEATS, rank_metric: str = DEFAULT_RANK_METRIC,
               quantile: float = 0.95) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Re-run the whole audit across seeds. Returns (per-seed summary, per-seed coordinates)."""
    summaries, coord_frames = [], []
    for s in seeds:
        catalog = make_catalog(seed=s)
        scenarios = make_scenarios(n_per_stage)
        coords, pairs = audit(scenarios, catalog, seed=s, repeats=repeats,
                              rank_metric=rank_metric)
        tau = calibrate_margin(coords, quantile)
        loc = localization_with_ci(coords, tau, n_boot=500, seed=s)
        legacy = localize(coords, LEGACY_MARGIN)
        summaries.append({
            "seed": s,
            "tau_calibrated": tau,
            "accuracy": loc["accuracy"],
            "boot_lo": loc["boot_lo"], "boot_hi": loc["boot_hi"],
            "accuracy_at_legacy_tau": float(legacy["correct"].mean()),
        })
        sc = scenario_coords(coords).assign(seed=s)
        coord_frames.append(sc)
    return pd.DataFrame(summaries), pd.concat(coord_frames, ignore_index=True)
