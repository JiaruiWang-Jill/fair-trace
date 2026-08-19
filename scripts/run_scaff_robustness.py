"""Run the canonical SCAFF robustness configuration and emit every paper-bound number.

    python scripts/run_scaff_robustness.py

Writes ``paper/scaff_robustness_results.json`` (every number the .tex quotes, so each one is
traceable to a run rather than copied forward) and the figure
``paper/figures/fig_scaff_robustness.pdf`` / ``.png``.

The configuration is fixed here rather than passed in, so that "re-run it and the numbers
match" is a meaningful check.
"""

from __future__ import annotations

import json
import pathlib
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import scaff_robustness as R  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT_JSON = ROOT / "paper" / "scaff_robustness_results.json"
OUT_FIG = ROOT / "paper" / "figures" / "fig_scaff_robustness"

N_PER_STAGE = 20           # 6 conditions x 20 = 120 scenarios
REPEATS = 8
SEEDS = [42, 43, 44, 45, 46]
METRICS = list(R.RANK_METRICS)
PRIMARY_METRIC = "ndcg_delta"
ENDPOINT_MARGIN = 0.02
BOOT = 4000


def _records(df: pd.DataFrame) -> list:
    return json.loads(df.reset_index().to_json(orient="records"))


def main() -> dict:
    catalog = R.make_catalog()
    scenarios = R.make_scenarios(N_PER_STAGE)
    out = {
        "config": {
            "n_scenarios": int(len(scenarios)), "n_per_stage": N_PER_STAGE,
            "repeats": REPEATS, "seeds": SEEDS, "catalog_size": int(len(catalog)),
            "rank_jitter": R.RANK_JITTER, "retrieve_temp": R.RETRIEVE_TEMP,
            "primary_rank_metric": PRIMARY_METRIC, "legacy_margin": R.LEGACY_MARGIN,
            "endpoint_margin": ENDPOINT_MARGIN, "n_bootstrap": BOOT,
        }
    }

    # ---- the two crossover regimes ---------------------------------------------------
    for regime, shared in (("shared_eps", True), ("independent_eps", False)):
        coords, pairs = R.audit(scenarios, catalog, repeats=REPEATS,
                                rank_metric="top1_gap", rank_metrics=METRICS,
                                shared_eps=shared)
        sc = R.scenario_coords(coords)
        out[regime] = {
            "coordinates_legacy_metric": _records(
                sc.groupby("stage")[["natural", "direct", "inherited", "noise"]]
                .mean().reindex(R.STAGES).round(4)),
            "rank_metric_comparison": _records(
                R.rank_metric_comparison(coords, METRICS).round(4)),
        }
        per_metric = []
        for m in METRICS:
            co, _ = R.audit(scenarios, catalog, repeats=REPEATS, rank_metric=m,
                            shared_eps=shared)
            tau = R.calibrate_margin(co)
            loc = R.localization_with_ci(co, tau, n_boot=BOOT)
            legacy = R.localize(co, R.LEGACY_MARGIN)
            per_metric.append({
                "metric": m, "tau_calibrated": round(tau, 4),
                "accuracy": round(loc["accuracy"], 4),
                "boot_lo": round(loc["boot_lo"], 4), "boot_hi": round(loc["boot_hi"], 4),
                "wilson_lo": round(loc["wilson_lo"], 4), "wilson_hi": round(loc["wilson_hi"], 4),
                "accuracy_at_legacy_tau": round(float(legacy["correct"].mean()), 4),
            })
        out[regime]["localization_by_metric"] = per_metric

    # ---- primary configuration: shared eps, calibrated tau, primary metric -----------
    coords, pairs = R.audit(scenarios, catalog, repeats=REPEATS, rank_metric=PRIMARY_METRIC)
    tau = R.calibrate_margin(coords)
    loc = R.localization_with_ci(coords, tau, n_boot=BOOT)
    sc = R.scenario_coords(coords)
    out["primary"] = {
        "tau_calibrated": round(tau, 4),
        "coordinates": _records(
            sc.groupby("stage")[["natural", "direct", "inherited", "noise"]]
            .mean().reindex(R.STAGES).round(4)),
        "diagnosis_at_planted_stage": _records(R.diagnosis_recovery(coords, tau).round(4)),
        "localization": {k: (round(v, 4) if isinstance(v, float) else v)
                         for k, v in loc.items() if k not in ("confusion", "misses")},
        "confusion": _records(loc["confusion"]),
        "n_misses": int(len(loc["misses"])),
    }

    # ---- (4) margin sweep -------------------------------------------------------------
    out["margin_sweep"] = _records(
        R.margin_sweep(coords, np.round(np.arange(0.0, 0.402, 0.02), 3)).round(4))

    # ---- (2) repair -------------------------------------------------------------------
    out["repair"] = {
        "downstream_frac": _records(
            R.repair_matrix(scenarios, catalog, repeats=REPEATS,
                            rank_metric=PRIMARY_METRIC, column="downstream_frac").round(4)),
        "downstream_before": _records(
            R.repair_matrix(scenarios, catalog, repeats=REPEATS,
                            rank_metric=PRIMARY_METRIC, column="downstream_before").round(4)),
        "legacy_circular_frac": _records(
            R.legacy_repair_matrix(scenarios, catalog, repeats=REPEATS,
                                   rank_metric=PRIMARY_METRIC).round(4)),
        "elicit_detail": _records(
            R.repair_table(scenarios, catalog, "Elicit", repeats=REPEATS,
                           rank_metric=PRIMARY_METRIC).round(4)),
    }

    # ---- masking: structural blindness vs low power ------------------------------------
    mask = R.masking_table(coords, pairs, tau, endpoint_margin=ENDPOINT_MARGIN)
    two = R.two_turn_masking(scenarios, catalog, tau, repeats=REPEATS,
                             rank_metric=PRIMARY_METRIC, endpoint_margin=ENDPOINT_MARGIN)
    out["masking"] = {
        "all": _records(mask["all"]),
        "endpoint_can_move": _records(mask["endpoint_can_move"]),
        "structurally_blind": _records(mask["structurally_blind"]),
        "masked_rate_power_only": round(mask["masked_rate_power"], 4),
        "n_can_move": mask["n_can_move"], "n_blind": mask["n_blind"],
        "blind_max_endpoint_distance": round(mask["blind_max_endpoint_distance"], 6),
        "two_turn_by_planted_stage": _records(
            two.groupby("planted_stage")[["turn2_endpoint_distance",
                                          "turn2_endpoint_different"]].mean().round(4)),
    }

    # ---- (3) seed sweep ----------------------------------------------------------------
    summary, _ = R.seed_sweep(SEEDS, n_per_stage=N_PER_STAGE, repeats=REPEATS,
                              rank_metric=PRIMARY_METRIC)
    out["seed_sweep"] = {
        "per_seed": _records(summary.round(4)),
        "accuracy_min": round(float(summary["accuracy"].min()), 4),
        "accuracy_max": round(float(summary["accuracy"].max()), 4),
        "accuracy_mean": round(float(summary["accuracy"].mean()), 4),
        "accuracy_at_legacy_tau_mean": round(float(summary["accuracy_at_legacy_tau"].mean()), 4),
    }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, indent=2))
    make_figure(out)
    return out


def make_figure(res: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Drawn 1:1 at ACM single-column manuscript text width (~6.9in); slide-sized figures
    # drop to unreadable type at \linewidth in acmart.
    fig, axes = plt.subplots(1, 3, figsize=(6.9, 2.3))
    ink = {"direct": "#D55E00", "inherited": "#7B52AB", "noise": "#5A5A5A"}

    ax = axes[0]
    cmp_ind = pd.DataFrame(res["independent_eps"]["rank_metric_comparison"])
    order = cmp_ind.sort_values("auc")
    ax.barh(range(len(order)), order["auc"], color="#4C72B0", height=0.6)
    ax.axvline(0.5, color="black", lw=0.8, ls=":")
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(order["metric"], fontsize=6)
    ax.set_xlabel("detection AUC", fontsize=7)
    ax.set_xlim(0.4, 1.0)
    ax.set_title("(a) Rank coordinate,\nnoisy crossover", fontsize=7)

    ax = axes[1]
    sw = pd.DataFrame(res["margin_sweep"])
    ax.plot(sw["tau"], sw["accuracy"], color="#4C72B0", lw=1.2, label="accuracy")
    ax.plot(sw["tau"], sw["recall"], color="#D55E00", lw=1.0, ls="--", label="recall")
    ax.plot(sw["tau"], sw["false_positive_rate"], color="#5A5A5A", lw=1.0, ls=":",
            label="false pos.")
    ax.axvline(res["config"]["legacy_margin"], color="#C44E52", lw=0.9)
    ax.text(res["config"]["legacy_margin"] + 0.012, 1.16, r"hand-set $\tau$", fontsize=5.5,
            color="#C44E52")
    ax.set_xlabel(r"margin $\tau$", fontsize=7)
    ax.set_ylim(-0.05, 1.28)
    ax.set_title("(b) the margin was\nnever calibrated", fontsize=7)
    ax.legend(fontsize=5.5, frameon=False, loc="lower left", ncol=1)

    ax = axes[2]
    rep = pd.DataFrame(res["repair"]["downstream_frac"]).set_index("index").reindex(R.STAGES)
    leg = pd.DataFrame(res["repair"]["legacy_circular_frac"]).set_index("index").reindex(R.STAGES)
    idx = np.arange(2)
    vals_true = [leg.loc["Elicit", "Elicit"], rep.loc["Elicit", "Elicit"]]
    vals_wrong = [leg.loc["Elicit", "Retrieve"], rep.loc["Elicit", "Retrieve"]]
    ax.bar(idx - 0.19, vals_true, width=0.38, color=ink["direct"], label="repair Elicit (true)")
    ax.bar(idx + 0.19, vals_wrong, width=0.38, color=ink["inherited"],
           label="repair Retrieve (faithful)")
    ax.set_xticks(idx)
    ax.set_xticklabels(["scored on the\nrepaired stage", "scored strictly\ndownstream"],
                       fontsize=6)
    ax.set_ylabel("fraction removed", fontsize=7)
    ax.set_ylim(0, 1.32)
    ax.set_title("(c) repair, fault at Elicit", fontsize=7)
    ax.legend(fontsize=5.5, frameon=False, loc="upper center", ncol=1,
              handlelength=1.2, borderpad=0.1)

    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(labelsize=6)
    fig.tight_layout()
    OUT_FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(f"{OUT_FIG}.pdf", bbox_inches="tight")
    fig.savefig(f"{OUT_FIG}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    res = main()
    print(f"wrote {OUT_JSON.relative_to(ROOT)}")
    print(f"wrote {OUT_FIG.relative_to(ROOT)}.pdf/.png")
    p = res["primary"]
    print(f"\ntau* = {p['tau_calibrated']}  (hand-set was {res['config']['legacy_margin']})")
    L = p["localization"]
    print(f"localization {L['accuracy']:.1%} "
          f"[{L['boot_lo']:.3f}, {L['boot_hi']:.3f}] boot, "
          f"[{L['wilson_lo']:.3f}, {L['wilson_hi']:.3f}] Wilson, n={L['n_scenarios']}")
    s = res["seed_sweep"]
    print(f"across seeds: {s['accuracy_min']:.3f}-{s['accuracy_max']:.3f} "
          f"(mean {s['accuracy_mean']:.3f}); at the hand-set tau: {s['accuracy_at_legacy_tau_mean']:.3f}")
