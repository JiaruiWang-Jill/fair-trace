#!/usr/bin/env python3
"""Analyse the qwen-on-MovieLens-1M audit and write figures plus a results summary.

Safe to run against a partial run: it reports how many users completed and labels
the output accordingly.
"""
from __future__ import annotations

import json, math, sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "fair_trace_outputs_ml1m_qwen"
STAGES = ["Elicit", "Retrieve", "Rank", "Explain", "Memory"]
MARGIN = 0.10
INK, MUTED, LINE = "#131A22", "#5C6B75", "#C3D0D6"
KEY, WARN, SOFT = "#0F6E7D", "#C24A32", "#9FB3BD"
plt.rcParams["font.family"] = "DejaVu Sans"


def population_std(vals):
    vals = [v for v in vals if v == v]
    if not vals:
        return float("nan")
    m = sum(vals) / len(vals)
    return math.sqrt(sum((v - m) ** 2 for v in vals) / len(vals))


def main() -> None:
    sens = pd.read_csv(OUT / "sensitivity.csv")
    sim = pd.read_csv(OUT / "similarity.csv")
    cons = pd.read_csv(OUT / "consequence.csv")
    n_users = sens["user"].nunique()
    lines: list[str] = []
    P = lines.append

    P(f"# qwen3.5:9b on MovieLens-1M — results\n")
    P(f"Users completed: **{n_users}**  ·  sensitivity rows {len(sens)}  ·  "
      f"similarity rows {len(sim)}  ·  consequence rows {len(cons)}\n")
    man = None
    if (OUT / "run_manifest.json").exists():
        man = json.loads((OUT / "run_manifest.json").read_text())
        # A manifest left by an earlier, smaller run must not be read as this run finishing.
        if man.get("n_users") != n_users:
            man = None
    if man:
        P(f"Run finished cleanly. Model calls {man['usage']['calls']}, "
          f"cache hits {man['usage']['cache_hits']}, retries {man['usage']['retries']}.\n")
    else:
        P("**The run was still in progress when this was written**, so these numbers "
          "cover the users completed so far.\n")

    # ── 1. localisation, raw vs noise-adjusted ─────────────────────────────
    sens["adj"] = sens["direct"] - sens["noise"]
    rows = []
    for (u, planted), g in sens.groupby(["user", "planted_stage"]):
        per_stage = g.groupby("stage")[["direct", "adj"]].mean()
        for rule in ("direct", "adj"):
            best = per_stage[rule].idxmax()
            score = per_stage[rule].max()
            rows.append({"user": u, "true": planted, "rule": rule,
                         "pred": best if score > MARGIN else "none", "score": score})
    loc = pd.DataFrame(rows)
    acc = {r: (sub["true"] == sub["pred"]).mean() for r, sub in loc.groupby("rule")}
    P("## Localisation\n")
    P(f"| rule | accuracy | correct |")
    P(f"|---|---|---|")
    for rule, label in [("direct", "argmax raw direct"), ("adj", "argmax direct − noise")]:
        sub = loc[loc["rule"] == rule]
        P(f"| {label} | {acc[rule]:.1%} | {int((sub['true']==sub['pred']).sum())}/{len(sub)} |")
    P(f"\nChance with six labels is 16.7%.\n")

    adj = loc[loc["rule"] == "adj"]
    labels = ["none"] + STAGES
    cm = pd.crosstab(adj["true"], adj["pred"]).reindex(index=labels, columns=labels, fill_value=0)
    wrong = int(sum(cm.loc[t, p] for t in labels for p in labels if t != p and p != "none"))
    missed = int(sum(cm.loc[t, "none"] for t in labels if t != "none"))
    P(f"Under the noise-adjusted rule: {wrong} errors name the wrong stage, "
      f"{missed} are missed detections.\n")

    fig, ax = plt.subplots(figsize=(6.6, 5.6))
    ax.imshow(cm.values, cmap="Blues")
    ax.set_xticks(range(len(labels)), labels, rotation=35, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    for i in range(len(labels)):
        for j in range(len(labels)):
            v = int(cm.values[i, j])
            if v:
                ax.text(j, i, v, ha="center", va="center", fontsize=10,
                        color="white" if v > cm.values.max() / 2 else INK)
    ax.set_xlabel("predicted stage"); ax.set_ylabel("planted stage")
    ax.set_title(f"qwen3.5:9b on MovieLens-1M\nargmax(direct − noise): "
                 f"{acc['adj']:.1%} of {n_users} users")
    plt.tight_layout(); plt.savefig(OUT / "qwen_ml1m_localisation.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    # ── 2. is the direct effect above its own noise floor? ─────────────────
    sens["is_planted"] = sens["stage"] == sens["planted_stage"]
    grp = sens.groupby("is_planted")[["direct", "noise", "adj"]].mean()
    P("## Signal against the noise floor\n")
    P("| rows | direct | noise | direct − noise |")
    P("|---|---|---|---|")
    for flag, label in [(True, "stage carrying the planted fault"), (False, "every other stage")]:
        if flag in grp.index:
            r = grp.loc[flag]
            n = int((sens["is_planted"] == flag).sum())
            P(f"| {label} ({n}) | {r['direct']:.4f} | {r['noise']:.4f} | {r['adj']:+.4f} |")
    planted_rows = sens[sens["is_planted"]]
    clears = int((planted_rows["direct"] > planted_rows["noise"]).sum())
    P(f"\nPlanted stages whose direct effect clears their own noise floor: "
      f"**{clears} of {len(planted_rows)}**.\n")

    by_stage = sens.groupby(["stage", "is_planted"])[["direct", "noise"]].mean()
    fig, ax = plt.subplots(figsize=(9, 4.4))
    xs = np.arange(len(STAGES)); w = 0.27
    for off, flag, color, lab in [(-w, True, WARN, "direct, planted here"),
                                  (0, False, SOFT, "direct, elsewhere")]:
        vals = [by_stage.loc[(s, flag), "direct"] if (s, flag) in by_stage.index else 0 for s in STAGES]
        ax.bar(xs + off, vals, w, color=color, label=lab)
    nz = [by_stage.loc[(s, True), "noise"] if (s, True) in by_stage.index else 0 for s in STAGES]
    ax.bar(xs + w, nz, w, color=LINE, edgecolor=MUTED, label="noise floor")
    ax.axhline(MARGIN, ls="--", lw=1, color=INK, label=f"margin {MARGIN}")
    ax.set_xticks(xs, STAGES); ax.set_ylabel("primary-coordinate distance")
    ax.set_title("A served model has real jitter: the noise floor is not zero")
    ax.legend(fontsize=9); ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout(); plt.savefig(OUT / "qwen_ml1m_signal_vs_noise.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    # ── 3. dispersion ──────────────────────────────────────────────────────
    P("## Group dispersion\n")
    disp_rows = []
    for col in ("sim_item", "sim_pref"):
        per_value = sim.groupby(["attribute", "stage", "value"])[col].mean().reset_index(name="bar")
        for (attr, stage), g in per_value.groupby(["attribute", "stage"]):
            v = g["bar"].tolist()
            disp_rows.append({"attribute": attr, "stage": stage, "coordinate": col,
                              "n_values": len(v), "SNSR": max(v) - min(v), "SNSV": population_std(v)})
    disp = pd.DataFrame(disp_rows)
    disp["SNSR_over_2"] = disp["SNSR"] / 2
    disp["identity_holds"] = np.isclose(disp["SNSV"], disp["SNSR_over_2"], atol=1e-12)
    disp.to_csv(OUT / "dispersion.csv", index=False)
    for attr in sorted(disp["attribute"].unique()):
        sub = disp[disp["attribute"] == attr]
        P(f"- `{attr}`: identity SNSV = SNSR/2 holds on "
          f"{int(sub['identity_holds'].sum())} of {len(sub)} rows")
    for col in ("sim_item", "sim_pref"):
        sub = disp[disp["coordinate"] == col]
        P(f"- `{col}`: mean SNSR {sub['SNSR'].mean():.4f}, max {sub['SNSR'].max():.4f}")
    P("")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), sharey=True)
    for ax, col in zip(axes, ["sim_item", "sim_pref"]):
        sub = disp[disp["coordinate"] == col]
        xs = np.arange(len(STAGES)); w = 0.35
        for off, attr, color in zip([-w / 2, w / 2], sorted(disp["attribute"].unique()), [KEY, WARN]):
            vals = [sub[(sub["attribute"] == attr) & (sub["stage"] == s)]["SNSR"].mean() for s in STAGES]
            ax.bar(xs + off, np.nan_to_num(vals), w, color=color, label=attr)
        ax.set_xticks(xs, STAGES, rotation=20)
        ax.set_title("item level" if col == "sim_item" else "true-preference level")
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("SNSR"); axes[0].legend(fontsize=9)
    plt.tight_layout(); plt.savefig(OUT / "qwen_ml1m_snsr.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    # ── 4. consequence ─────────────────────────────────────────────────────
    P("## Consequence\n")
    side = (sens.groupby("stage")[["natural", "direct", "inherited", "noise"]].mean()
            .join(cons.groupby("stage")[["benefit", "benefit_neutral", "benefit_delta"]].mean())
            .reindex(STAGES))
    side.to_csv(OUT / "sensitivity_vs_consequence.csv")
    P("```")
    P(side.round(4).to_string())
    P("```\n")

    truth = (cons.groupby(["stage", "matches_real"])["benefit"].mean().unstack()
             .reindex(STAGES))
    if truth.shape[1] == 2:
        truth.columns = ["stated attribute is false", "stated attribute is true"]
        truth["gap"] = truth.iloc[:, 1] - truth.iloc[:, 0]
        P("Benefit when the prompt states the user's real attribute versus a counterfactual one:\n")
        P("```"); P(truth.round(4).to_string()); P("```\n")

    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.axvline(MARGIN, color=WARN, lw=1.0, ls="--", label=f"margin {MARGIN}")
    ax.scatter(side["direct"], side["benefit_delta"], s=80, color=KEY, zorder=3)
    for s, r in side.iterrows():
        ax.annotate(s, (r["direct"], r["benefit_delta"]), fontsize=9,
                    xytext=(6, 4), textcoords="offset points")
    ax.set_xlabel("$D_s^A$ (sensitivity)"); ax.set_ylabel("benefit delta vs neutral")
    ax.set_title("Sensitivity against consequence, qwen on MovieLens-1M")
    ax.legend(fontsize=9); ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout(); plt.savefig(OUT / "qwen_ml1m_sensitivity_vs_consequence.png",
                                    dpi=160, bbox_inches="tight")
    plt.close(fig)

    P("## Figures\n")
    for p in sorted(OUT.glob("qwen_ml1m_*.png")):
        P(f"- `{p.name}`")

    (OUT / "RESULTS.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nwrote {OUT/'RESULTS.md'} and {len(list(OUT.glob('qwen_ml1m_*.png')))} figures")


if __name__ == "__main__":
    main()
