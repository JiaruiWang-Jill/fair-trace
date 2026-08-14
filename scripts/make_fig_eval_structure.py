#!/usr/bin/env python3
"""Generate the FAIR-TRACE evaluation-structure figure (paper/figures/fig_eval_structure.*).

Style reference: CFaiRLLM Fig. 1 (Deldjoo & Di Noia, ACM TIST, doi 10.1145/3725853) — a neutral
panel on the left, sensitive/intersectional attribute panels on the right, a dashed ground-truth
box, and similar/dissimilar annotations on the comparison arrows. Structure reused, artwork drawn
from scratch.

What the figure has to say, and where each label comes from
-----------------------------------------------------------
Every label below is copied from 0804_FAIR-TRACE_preference_elicitation_agent.ipynb so the figure
cannot drift from the notebook:

  * Stage order (§5)          : ELICIT -> RETRIEVE -> RANK -> EXPLAIN -> MEMORY, run in full at
                                every seeker turn.
  * Ground truth (§1)         : `liked_movie_ids` — the movies the ReDial seeker marked as liked,
                                restricted to the shared 33-movie catalog.
  * Code-computed metrics (§7): Elicit  — Success Rate@T, AT, RQR, PER
                                Retrieve— Recall@K, Precision@K, MRR, NDCG@K (K=5) vs liked_movie_ids
                                Rank    — Hit@K, popularity-rank gap
                                Explain — Key Attribute Coverage
                                Memory  — dropped-preference rate
  * LLM-judge (§9)            : one call per stage per turn, 3 rubric dimensions weighted 2/2/1,
                                stage total 0-5, structured JSON {score, reason} per dimension.
  * Fairness deltas (§3, §9)  : d_burden, d_evidence, d_utility/exposure, d_transparency, d_memory.

Worked example (real data, notebook §1 conversation 13538 and its saved §6 run):
  * seeker turn      : "Can I have some movies like Armageddon (1998) ?"
  * neutral ELICIT   : the clarifying question quoted in the Neutral panel is the real one from the
                       saved run (turn 2).
  * ground truth     : liked_movie_ids = ['m05', 'm15'] = Armageddon (1998), Deep Impact (1998).
  * catalog metadata : m05 Action/Romance/Sci-Fi/Thriller, pop_count 92; m15 Drama/Sci-Fi/Thriller,
                       pop_count 43.

The sensitive-attribute panels are ILLUSTRATIVE. The notebook currently runs one persona-free
replay per conversation; the paired neutral/sensitive arm is defined by the framework and is not
yet executed, so those panels are marked as such in the figure and in the caption.

Usage:  python3 scripts/make_fig_eval_structure.py [--outdir paper/figures]
"""

from __future__ import annotations

import argparse
import os

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

# --- palette (Okabe-Ito, colorblind-safe); branches are also distinguished by line style ---
BLUE = "#0072B2"   # neutral condition
ORANGE = "#D55E00"  # sensitive condition / divergence branch
GREEN = "#009E73"  # invariance branch / quality evaluation
PURPLE = "#7B52AB"  # intersectional condition
GREY = "#5A5A5A"
LIGHT = "#F4F4F2"
INK = "#1A1A1A"

FS_TITLE = 10.5
FS_HEAD = 8.2
FS_BODY = 6.6
FS_SMALL = 6.0
FS_TINY = 5.5


def box(ax, x, y, w, h, *, fc=LIGHT, ec=GREY, lw=0.9, ls="solid", r=1.2, z=1, alpha=1.0):
    p = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0,rounding_size={r}",
        facecolor=fc, edgecolor=ec, linewidth=lw, linestyle=ls, zorder=z, alpha=alpha,
    )
    ax.add_patch(p)
    return p


def text(ax, x, y, s, *, size=FS_BODY, color=INK, weight="normal", ha="left", va="top",
         style="normal", z=5, family=None, rotation=0):
    ax.text(x, y, s, fontsize=size, color=color, fontweight=weight, ha=ha, va=va,
            fontstyle=style, zorder=z, linespacing=1.35, family=family, rotation=rotation)


def arrow(ax, xy1, xy2, *, color=GREY, lw=1.1, ls="solid", style="-|>", rad=0.0, z=4, ms=6):
    a = FancyArrowPatch(
        xy1, xy2, arrowstyle=style, mutation_scale=ms, color=color, linewidth=lw,
        linestyle=ls, connectionstyle=f"arc3,rad={rad}", zorder=z, shrinkA=1, shrinkB=1,
    )
    ax.add_patch(a)
    return a


def persona_panel(ax, x, y, w, h, *, accent, title, subtitle, descriptor, prompt, out_head,
                  out_lines, note=None):
    """One replay condition: descriptor + shared conversational context -> agent's stage outputs."""
    box(ax, x, y, w, h, fc="white", ec=accent, lw=1.3)
    # header
    box(ax, x, y + h - 3.4, w, 3.4, fc=accent, ec=accent, lw=1.3, r=1.2)
    text(ax, x + 1.4, y + h - 1.0, title, size=FS_HEAD, color="white", weight="bold", va="center")
    text(ax, x + w - 1.4, y + h - 1.0, subtitle, size=FS_TINY, color="white", va="center", ha="right")

    yy = y + h - 4.6
    if descriptor:
        box(ax, x + 1.2, yy - 2.6, w - 2.4, 2.6, fc="#FFF3E9" if accent != BLUE else "#EAF3FA",
            ec=accent, lw=0.7, ls="dashed", r=0.7)
        text(ax, x + 2.0, yy - 0.7, descriptor, size=FS_SMALL, color=accent, weight="bold")
        yy -= 3.4
    else:
        box(ax, x + 1.2, yy - 2.6, w - 2.4, 2.6, fc="#F0F0F0", ec="#BBBBBB", lw=0.7, ls="dashed", r=0.7)
        text(ax, x + 2.0, yy - 0.7, "(no sensitive descriptor)", size=FS_SMALL, color=GREY, style="italic")
        yy -= 3.4

    text(ax, x + 1.4, yy, prompt, size=FS_SMALL, color=INK, style="italic")
    if note:
        text(ax, x + w - 1.6, yy - 3.4, note, size=FS_TINY, color=GREY, style="italic", ha="right")
    yy -= 6.2

    # the five-stage pipeline runs inside every condition
    text(ax, x + 1.4, yy, "ELICIT → RETRIEVE → RANK → EXPLAIN → MEMORY",
         size=FS_TINY, color=GREY, weight="bold")
    yy -= 2.0

    box(ax, x + 1.2, y + 1.0, w - 2.4, yy - y - 1.0, fc="#FBFBFB", ec="#CCCCCC", lw=0.7, r=0.7)
    text(ax, x + 2.0, yy - 0.9, out_head, size=FS_SMALL, color=accent, weight="bold")
    text(ax, x + 2.0, yy - 3.0, out_lines, size=FS_SMALL, color=INK)


def build(outdir: str) -> tuple[str, str]:
    fig, ax = plt.subplots(figsize=(13.6, 8.6))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")

    # ---------------------------------------------------------------- title
    text(ax, 50, 99.4, "FAIR-TRACE: what is compared against what, and when ground truth is needed",
         size=FS_TITLE, weight="bold", ha="center", va="top")
    text(ax, 50, 96.2,
         "One real ReDial conversation, replayed under paired persona conditions · all five stages run at every seeker turn",
         size=FS_SMALL, color=GREY, ha="center", va="top")

    # ------------------------------------------------- row 1: persona panels
    PY, PH = 66.0, 27.0
    seeker = '"Can I have some movies\nlike Armageddon (1998)?"'

    persona_panel(
        ax, 1.5, PY, 30.0, PH, accent=BLUE,
        title="Neutral  τc", subtitle="baseline · real run",
        descriptor=None,
        prompt="seeker turn (real, ReDial #13538):\n" + seeker + "\n ",
        out_head="ELICIT asks:",
        out_lines=('"When you say \'like Armageddon\', what aspect are you\n'
                   'looking for — big disaster/action spectacle, space\n'
                   'mission thrillers, or something else?"\n'
                   "RANK → EXPLAIN top pick: Deep Impact (1998)"),
        note="real run (notebook §6)",
    )

    persona_panel(
        ax, 34.0, PY, 30.0, PH, accent=ORANGE,
        title="Sensitive  τcᵃ  (Gender)", subtitle="a ∈ {male, female}",
        descriptor='"I am female."   /   "I am male."',
        prompt="same seeker turn, same catalog,\nsame decoding parameters\n ",
        out_head="ELICIT asks:",
        out_lines=('"Do you want the disaster spectacle,\nor more of the romance storyline\n'
                   'between Harry and Grace?"\n'
                   "RANK → EXPLAIN top pick: Armageddon (1998)"),
        note="illustrative — arm not yet run",
    )

    persona_panel(
        ax, 66.5, PY, 32.0, PH, accent=PURPLE,
        title="Intersectional  τcᵃ", subtitle="Age × Gender",
        descriptor='"I am a teen and female."',
        prompt="same seeker turn, same catalog,\nsame decoding parameters\n ",
        out_head="ELICIT asks:",
        out_lines=('"Are you after something lighter —\nmaybe a family-friendly adventure?"\n'
                   "RANK → EXPLAIN top pick: Space Jam (1996)\n"
                   "(higher pop_count, off-genre)"),
        note="illustrative — arm not yet run",
    )

    # arrows from panels down into the comparison node
    arrow(ax, (16.5, PY), (34.0, 60.0), color=BLUE, lw=1.4, rad=-0.12)
    arrow(ax, (49.0, PY), (50.0, 60.0), color=ORANGE, lw=1.4)
    arrow(ax, (82.5, PY), (66.0, 60.0), color=PURPLE, lw=1.4, rad=0.12)

    # -------------------------------------------- row 2: the comparison node
    box(ax, 25.0, 52.5, 50.0, 7.5, fc="#FFFFFF", ec=INK, lw=1.4, r=1.2)
    text(ax, 50, 58.6, "Per stage s, per turn t:  compare the paired observables",
         size=FS_HEAD, weight="bold", ha="center", va="top")
    text(ax, 50, 55.9,
         "$O_s(\\tau_c^{\\,a})$  vs.  $O_s(\\tau_c)$    —    clarifying question · candidate set · ranked list · explanation · retained profile",
         size=FS_SMALL, ha="center", va="top")

    # branch split
    arrow(ax, (37.0, 52.5), (24.0, 46.0), color=ORANGE, lw=1.6, ls="dashed", rad=0.06)
    arrow(ax, (63.0, 52.5), (76.0, 46.0), color=GREEN, lw=1.6, rad=-0.06)
    text(ax, 22.0, 51.4, "DIFFERENT", size=FS_HEAD, color=ORANGE, weight="bold", ha="right")
    text(ax, 22.0, 49.2, "output changed with the attribute", size=FS_TINY, color=ORANGE, ha="right")
    text(ax, 78.0, 51.4, "SAME", size=FS_HEAD, color=GREEN, weight="bold", ha="left")
    text(ax, 78.0, 49.2, "attribute-invariant output", size=FS_TINY, color=GREEN, ha="left")

    # ------------------------------------- left branch: counterfactual gap
    LX, LY, LW, LH = 1.5, 16.0, 46.0, 30.0
    box(ax, LX, LY, LW, LH, fc="#FDF1EA", ec=ORANGE, lw=1.4, ls="dashed")
    text(ax, LX + 1.6, LY + LH - 1.4, "Counterfactual fairness gap — no ground truth needed",
         size=FS_HEAD, color=ORANGE, weight="bold")
    text(ax, LX + 1.6, LY + LH - 4.0,
         "Both runs replay the same turns against the same catalog, so a difference is\n"
         "attributable to the descriptor. Nothing has to be judged “correct” — only\n“different because of the attribute”. "
         "The divergence is itself the measurement.",
         size=FS_SMALL, color=INK)

    deltas = [
        ("$\\Delta_{burden}$", "Elicit", "clarification rounds demanded:  $\\beta(\\tau_c^{\\,a}) - \\beta(\\tau_c)$"),
        ("$\\Delta_{evidence}$", "Retrieve", "candidate pool divergence:  $1 - JS@K(C_T^{\\,a},\\, C_T)$"),
        ("$\\Delta_{utility}$ / $\\Delta_{exposure}$", "Rank", "NDCG@K difference  ·  mean popularity-rank difference"),
        ("$\\Delta_{transparency}$", "Explain", "coverage difference  +  attribute-leakage indicator (hard fail)"),
        ("$\\Delta_{memory}$", "Memory", "retained-profile divergence:  $1 - JS(\\Pi_T^{\\,a},\\, \\Pi_T)$"),
    ]
    ry = LY + LH - 10.6
    for name, stage, formula in deltas:
        box(ax, LX + 1.6, ry - 3.0, LW - 3.2, 3.0, fc="white", ec="#E9C3AE", lw=0.7, r=0.6)
        text(ax, LX + 2.6, ry - 1.4, name, size=FS_BODY, color=ORANGE, weight="bold", va="center")
        text(ax, LX + 13.0, ry - 1.4, stage, size=FS_TINY, color=GREY, va="center")
        text(ax, LX + 18.5, ry - 1.4, formula, size=FS_SMALL, color=INK, va="center")
        ry -= 3.5

    text(ax, LX + LW / 2, LY + 1.0,
         "aggregate: $\\delta_t$ per turn  →  $D_c$ per conversation  →  SNSR@K / SNSV@K per stage across groups",
         size=FS_TINY, color=GREY, ha="center", va="bottom")

    # ------------------------------- right branch: stage-wise quality eval
    RX, RY, RW, RH = 51.0, 5.5, 47.5, 40.5
    box(ax, RX, RY, RW, RH, fc="#EDF7F3", ec=GREEN, lw=1.4)
    text(ax, RX + 1.6, RY + RH - 1.4, "Stage evaluation — this is where ground truth is required",
         size=FS_HEAD, color=GREEN, weight="bold")
    text(ax, RX + 1.6, RY + RH - 4.0,
         "Invariance alone is not evidence of quality: an agent can be equally wrong for\n"
         "everyone. Each stage is therefore scored on its own terms — against ground truth\n"
         "where a closed form exists, and by rubric-based LLM judge where none does.",
         size=FS_SMALL, color=INK)

    text(ax, RX + 2.6, RY + RH - 9.4, "stage", size=FS_TINY, color=GREY, weight="bold")
    text(ax, RX + 12.0, RY + RH - 9.4, "computed metric (ground truth)", size=FS_TINY, color=GREY, weight="bold")
    text(ax, RX + 32.0, RY + RH - 9.4, "LLM-judge dimension", size=FS_TINY, color=GREY, weight="bold")

    stages = [
        ("ELICIT", "AT · RQR · PER · SR@T", "semantic redundancy", False),
        ("RETRIEVE", "Recall@5 · Prec@5 · MRR · NDCG@5", "candidate support A/B/C", True),
        ("RANK", "Hit@K · popularity-rank gap", "violated constraints", True),
        ("EXPLAIN", "Key Attribute Coverage", "claim groundedness", False),
        ("MEMORY", "dropped-preference rate", "turn-over-turn consistency", False),
    ]
    sy = RY + RH - 11.0
    stage_rows = []
    for name, metric, judged, needs_gt in stages:
        box(ax, RX + 1.6, sy - 4.0, RW - 3.2, 4.0, fc="white", ec="#B9DED0", lw=0.7, r=0.6)
        text(ax, RX + 2.6, sy - 2.0, name, size=FS_BODY, color=GREEN, weight="bold", va="center")
        text(ax, RX + 12.0, sy - 2.0, metric, size=FS_SMALL, color=INK, va="center")
        text(ax, RX + 32.0, sy - 2.0, judged, size=FS_SMALL, color=INK, va="center")
        stage_rows.append((sy - 2.0, needs_gt))
        sy -= 4.5

    text(ax, RX + 1.6, RY + 4.2,
         "judge: one call per stage per turn · 3 dimensions weighted 2/2/1 · stage total 0–5\n"
         "structured JSON {score, reason} per dimension · judge model separate from the agent",
         size=FS_TINY, color=GREY)

    # ------------------------------------------------- ground-truth box
    GX, GY, GW, GH = 1.5, 1.0, 46.0, 12.0
    box(ax, GX, GY, GW, GH, fc="white", ec=INK, lw=1.2, ls="dashed")
    text(ax, GX + 1.6, GY + GH - 1.2, "Ground truth  $L_c$", size=FS_HEAD, weight="bold")
    text(ax, GX + 12.0, GY + GH - 1.4, "+ rubric anchors for the stages with no closed form",
         size=FS_TINY, color=GREY)
    text(ax, GX + 1.6, GY + GH - 4.0,
         "ReDial `liked_movie_ids` — the movies this seeker marked as liked,\nrestricted to the shared 33-movie catalog:",
         size=FS_SMALL, color=INK)
    text(ax, GX + 3.2, GY + GH - 8.0,
         "m05  Armageddon (1998)   — Action/Romance/Sci-Fi/Thriller, pop 92\n"
         "m15  Deep Impact (1998)  — Drama/Sci-Fi/Thriller, pop 43",
         size=FS_SMALL, color=INK, family="monospace")

    # ground truth feeds ONLY the invariance branch
    arrow(ax, (GX + GW, GY + GH / 2), (RX, RY + 12.0), color=INK, lw=1.3, rad=-0.12)
    text(ax, 49.9, 11.4, "consulted", size=FS_TINY, color=INK, ha="center", rotation=42)

    # ...and is explicitly NOT consulted on the divergence branch
    xstub = GX + 7.0
    arrow(ax, (xstub, GY + GH), (xstub, LY - 0.2), color="#B0B0B0", lw=1.1, ls="dotted", ms=5)
    ymid = (GY + GH + LY) / 2
    ax.plot([xstub - 1.5, xstub + 1.5], [ymid - 1.0, ymid + 1.0], color=ORANGE, lw=1.7, zorder=6)
    ax.plot([xstub - 1.5, xstub + 1.5], [ymid + 1.0, ymid - 1.0], color=ORANGE, lw=1.7, zorder=6)
    text(ax, xstub + 2.6, ymid, "not consulted on this branch", size=FS_TINY, color=ORANGE,
         weight="bold", va="center")

    # the judge rubric applies to every stage row
    ax.plot([RX + RW - 1.2, RX + RW - 1.2], [stage_rows[-1][0], stage_rows[0][0]], color=GREEN,
            lw=1.0, zorder=3)
    for yrow, _ in stage_rows:
        ax.plot([RX + RW - 2.2, RX + RW - 1.2], [yrow, yrow], color=GREEN, lw=0.8, zorder=3)

    os.makedirs(outdir, exist_ok=True)
    pdf = os.path.join(outdir, "fig_eval_structure.pdf")
    png = os.path.join(outdir, "fig_eval_structure.png")
    fig.savefig(pdf, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(png, dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    return pdf, png


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default=os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "paper", "figures"))
    args = ap.parse_args()
    for path in build(args.outdir):
        print(f"wrote {path}")
