#!/usr/bin/env python3
"""Generate the SCAFF stage-crossover figure in four alternative designs.

  paper/figures/fig_scaff_v1_crossover_matrix.{pdf,png}
  paper/figures/fig_scaff_v2_stage_tape.{pdf,png}
  paper/figures/fig_scaff_v3_worked_example.{pdf,png}
  paper/figures/fig_scaff_v4_causal_dag.{pdf,png}
  paper/figures/fig_scaff_paper.{pdf,png}       (print-fit variant, 1:1 at ACM text width)
  paper/figures/fig_scaff_contact_sheet.png     (all versions side by side, for choosing)

Style reference: CFaiRLLM Fig. 1 (Deldjoo & Di Noia, ACM TIST, doi 10.1145/3725853) --
paired prompt bubbles that differ only in the sensitive descriptor, a dashed box holding the
user's true preferences, and a shouted verdict label on the right of each comparison. That
grammar is reused; all artwork here is drawn from scratch and the verdicts are SCAFF's
(DIRECT / INHERITED / INVARIANT) rather than CFaiRLLM's Fair!/Unfair!.

WHERE EVERY NUMBER COMES FROM
-----------------------------
All quantities are read off a local re-execution of the shared prototype notebook
(FAIR_TRACE_SCAFF_TwoDataset_Toy_Colab.ipynb, Drive 144lBKEV781_uBMWDHFCYNBYi0Mod-z4h,
posted to #fair-trace on 2026-08-16). The re-execution reproduced the notebook's own saved
outputs exactly, so the constants below are measurements, not illustrations.

  * Worked example: scenario ``mov_001``, seed 42, 4 repeats. Seeker message
    "I want a mystery movie, and please avoid horror."; true_preferences
    {genre: mystery, tone: cerebral, pace: medium, avoid_horror: True}; revealed
    {genre: mystery, avoid_horror: True}; planted_stage = Elicit, planted_target = man.
    Per-stage primary coordinates (natural, direct, inherited):
        Elicit   0.333 / 0.333 / 0.000     <- the origin, directly descriptor-sensitive
        Retrieve 0.200 / 0.000 / 0.200
        Rank     0.750 / 0.000 / 0.750
        Explain  0.500 / 0.000 / 0.500
        Memory   0.333 / 0.000 / 0.333
    Stage outputs on that scenario: Rank top-1 flips m18 (man) vs m02 (woman); the man arm's
    Explain carries the extra reason tag ``tone=romantic``, which the latent profile does not
    support (true tone is cerebral).
  * Second worked example: ``mov_002``, planted_stage = Retrieve -- Elicit 0.000/0.000/0.000,
    Retrieve 0.833/0.833/0.000, Rank 1.000/0.000/1.000.
  * Corpus means (movies dataset, 24 scenarios x 4 repeats): see CORPUS below.
  * Diagnosis rule: DIRECT_EFFECT_MARGIN = 0.10, applied to the primary coordinate per stage
    (Elicit pref_jaccard, Retrieve candidate_jaccard, Rank top1_gap, Explain reason_jaccard,
    Memory fact_jaccard); mixed = direct and inherited both above margin.
  * Localization accuracy 100% (48 scenarios, both datasets); masked-process rate 0.365
    (movies) / 0.438 (restaurants) at ENDPOINT_EQUIV_MARGIN = 0.15; repair removability
    0.576 (movies) / 0.588 (restaurants) for the true planted stage vs 0.170 / 0.187 for a
    wrong stage.

Usage:  python3 scripts/make_fig_scaff.py [--outdir paper/figures] [--only v1,v3]
"""

from __future__ import annotations

import argparse
import os

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon

# --- palette: same Okabe-Ito basis as make_fig_eval_structure.py so the two figures match ---
BLUE = "#0072B2"    # condition A = a   (reference arm)
ORANGE = "#D55E00"  # condition A = a'  / direct sensitivity
GREEN = "#009E73"   # invariance / repair
PURPLE = "#7B52AB"  # inherited-state dependence
GOLD = "#E69F00"    # oracle / latent truth
GREY = "#5A5A5A"
MUTED = "#8C8C8C"
LIGHT = "#F4F4F2"
INK = "#1A1A1A"

FS_TITLE = 11.0
FS_HEAD = 8.4
FS_BODY = 6.8
FS_SMALL = 6.1
FS_TINY = 5.4
FS_MICRO = 4.8

STAGES = ["Elicit", "Retrieve", "Rank", "Explain", "Memory"]

# scenario mov_001 -- planted at Elicit (see module docstring)
MOV001 = {
    "Elicit":   (0.333, 0.333, 0.000),
    "Retrieve": (0.200, 0.000, 0.200),
    "Rank":     (0.750, 0.000, 0.750),
    "Explain":  (0.500, 0.000, 0.500),
    "Memory":   (0.333, 0.000, 0.333),
}
# scenario mov_002 -- planted at Retrieve
MOV002 = {
    "Elicit":   (0.000, 0.000, 0.000),
    "Retrieve": (0.833, 0.833, 0.000),
    "Rank":     (1.000, 0.000, 1.000),
    "Explain":  (0.000, 0.000, 0.000),
    "Memory":   (0.000, 0.000, 0.000),
}
# corpus means, movies dataset
CORPUS = {
    "Elicit":   (0.056, 0.056, 0.000),
    "Retrieve": (0.205, 0.136, 0.069),
    "Rank":     (0.396, 0.094, 0.302),
    "Explain":  (0.225, 0.067, 0.158),
    "Memory":   (0.156, 0.056, 0.101),
}
MARGIN = 0.10


def diagnose(nat: float, direct: float, inherited: float, margin: float = MARGIN) -> str:
    """The notebook's classify() rule, restated so the figure cannot drift from it."""
    d, z, n = direct > margin, inherited > margin, nat > margin
    if d and z:
        return "mixed"
    if d:
        return "direct"
    if n and z:
        return "inherited"
    return "invariant"


DIAG_COLOR = {"direct": ORANGE, "inherited": PURPLE, "mixed": "#B5651D", "invariant": GREEN}
DIAG_LABEL = {"direct": "DIRECT", "inherited": "INHERITED", "mixed": "MIXED",
              "invariant": "INVARIANT"}


# ---------------------------------------------------------------- drawing primitives
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


def arrow(ax, xy1, xy2, *, color=GREY, lw=1.1, ls="solid", style="-|>", rad=0.0, z=4, ms=7):
    a = FancyArrowPatch(
        xy1, xy2, arrowstyle=style, mutation_scale=ms, color=color, linewidth=lw,
        linestyle=ls, connectionstyle=f"arc3,rad={rad}", zorder=z, shrinkA=1.5, shrinkB=1.5,
    )
    ax.add_patch(a)
    return a


def bubble(ax, x, y, w, h, body, *, ec=INK, fc="white", tail="left", head=None,
           head_color=INK, lw=1.1, size=FS_SMALL):
    """A CFaiRLLM-style rounded speech bubble with a small triangular tail."""
    box(ax, x, y, w, h, fc=fc, ec=ec, lw=lw, r=1.0, z=2)
    ty = y + h * 0.66
    if tail == "left":
        pts = [(x, ty + 1.1), (x, ty - 1.1), (x - 2.2, ty - 2.4)]
    else:
        pts = [(x + w, ty + 1.1), (x + w, ty - 1.1), (x + w + 2.2, ty - 2.4)]
    ax.add_patch(Polygon(pts, closed=True, facecolor=fc, edgecolor=ec, linewidth=lw, zorder=2))
    yy = y + h - 1.3
    if head:
        text(ax, x + 1.4, yy, head, size=FS_SMALL, color=head_color, weight="bold")
        yy -= 2.3
    text(ax, x + 1.4, yy, body, size=size, color=INK)


def chip(ax, x, y, label, color, *, w=None, h=2.6, size=FS_TINY, fc=None):
    w = w or max(9.0, 1.35 * len(label))
    box(ax, x, y, w, h, fc=fc or color, ec=color, lw=0.9, r=0.7, z=6)
    text(ax, x + w / 2, y + h / 2, label, size=size, color="white" if fc is None else color,
         weight="bold", ha="center", va="center", z=7)
    return w


def verdict(ax, x, y, label, color, *, size=FS_HEAD, ha="left"):
    """CFaiRLLM's shouted right-hand verdict ('Fair!'/'UnFair!'), retargeted to SCAFF."""
    text(ax, x, y, label, size=size, color=color, weight="bold", ha=ha, va="center")


def coord_bars(ax, x, y, w, h, nat, direct, inherited, *, labels=True, size=FS_MICRO):
    """Three small horizontal bars: natural, direct, inherited, all on [0,1]."""
    rows = [("$D^{\\mathrm{nat}}$", nat, GREY),
            ("$D^{A}$", direct, ORANGE),
            ("$D^{Z}$", inherited, PURPLE)]
    bh = h / 3.4
    gap = (h - 3 * bh) / 2.0
    for k, (name, val, col) in enumerate(rows):
        by = y + h - (k + 1) * bh - k * gap
        ax.plot([x, x + w], [by + bh / 2, by + bh / 2], color="#E4E4E4", lw=bh * 3.1,
                solid_capstyle="butt", zorder=2)
        if val > 0:
            ax.plot([x, x + w * val], [by + bh / 2, by + bh / 2], color=col, lw=bh * 3.1,
                    solid_capstyle="butt", zorder=3)
        if labels:
            text(ax, x - 0.8, by + bh / 2, name, size=size, color=col, ha="right", va="center")
            text(ax, x + w + 0.7, by + bh / 2, f"{val:.2f}", size=size, color=col, va="center",
                 family="monospace")
    # the decision margin
    mx = x + w * MARGIN
    ax.plot([mx, mx], [y - 0.2, y + h + 0.2], color=INK, lw=0.6, ls=(0, (2, 2)), zorder=5)


def save(fig, outdir, stem):
    os.makedirs(outdir, exist_ok=True)
    pdf = os.path.join(outdir, f"{stem}.pdf")
    png = os.path.join(outdir, f"{stem}.png")
    fig.savefig(pdf, bbox_inches="tight", pad_inches=0.03)
    fig.savefig(png, dpi=300, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    return pdf, png


def canvas(w, h):
    fig, ax = plt.subplots(figsize=(w, h))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")
    return fig, ax


def footer(ax, extra=""):
    text(ax, 50, 0.6,
         "Prototype run: 24 scenarios x 4 repeats per dataset, seed 42, no API key; "
         "planted faults are unit tests of the instrument, not claims about group preferences. "
         + extra,
         size=FS_MICRO, color=MUTED, ha="center", va="bottom", style="italic")


# =============================================================== V1: crossover matrix
def build_v1(outdir):
    """Paired prompts on the left (CFaiRLLM grammar), the 2x2 crossover on the right."""
    fig, ax = canvas(13.6, 7.4)

    text(ax, 50, 99.4,
         "SCAFF: holding the inherited state fixed separates a stage that reads the protected "
         "descriptor from one that merely inherits its effect",
         size=FS_TITLE, weight="bold", ha="center", va="top")
    text(ax, 50, 96.0,
         "Same seeker message, same catalog, same decoding; the two arms differ only in the "
         "model-visible descriptor $A$",
         size=FS_SMALL, color=GREY, ha="center", va="top")

    # ---------------------------------------------------------- left: the paired arms
    text(ax, 2.0, 91.0, "(a)  Paired trajectories", size=FS_HEAD, weight="bold")

    bubble(ax, 6.0, 76.0, 32.0, 11.5,
           'I want a mystery movie,\nand please avoid horror.\n'
           '(revealed: genre=mystery, avoid_horror)',
           ec=BLUE, head="$A=a$   \u201cI am a man.\u201d", head_color=BLUE)
    bubble(ax, 6.0, 62.0, 32.0, 11.5,
           'I want a mystery movie,\nand please avoid horror.\n'
           '(revealed: genre=mystery, avoid_horror)',
           ec=ORANGE, head="$A=a'$   \u201cI am a woman.\u201d", head_color=ORANGE)

    # the oracle, drawn as CFaiRLLM draws the true-preference box: dashed, set apart
    box(ax, 6.0, 43.0, 32.0, 17.5, fc="#FFF9EC", ec=GOLD, lw=1.1, ls="dashed", r=1.0)
    text(ax, 7.4, 59.0, "Latent profile  $\\Omega$  (oracle, not shown to the agent)",
         size=FS_SMALL, color="#8A6100", weight="bold")
    text(ax, 8.6, 55.6,
         "genre  = mystery\ntone   = cerebral\npace   = medium\navoid_horror = true",
         size=FS_SMALL, color=INK, family="monospace")
    text(ax, 7.4, 46.6,
         "consulted only to ask whether a difference did harm --\nnever to detect that one exists",
         size=FS_MICRO, color="#8A6100", style="italic")

    # the pipeline the two arms run through
    text(ax, 2.0, 42.5, "(b)  Both arms run the same five stages", size=FS_HEAD, weight="bold")
    sx, sw, sgap = 4.0, 6.4, 1.1
    for k, st in enumerate(STAGES):
        x = sx + k * (sw + sgap)
        planted = st == "Elicit"
        box(ax, x, 34.5, sw, 5.2, fc="#FDF1EA" if planted else "white",
            ec=ORANGE if planted else GREY, lw=1.3 if planted else 0.9, r=0.8)
        text(ax, x + sw / 2, 37.1, st.upper(), size=FS_TINY, weight="bold", ha="center",
             va="center", color=ORANGE if planted else INK)
        if k < len(STAGES) - 1:
            arrow(ax, (x + sw, 37.1), (x + sw + sgap, 37.1), color=GREY, lw=0.9, ms=5)
    text(ax, 4.0, 32.6,
         "$\\uparrow$ planted fault lives here: on the $A=a$ arm, ELICIT adds the "
         "unsupported preference $\\mathtt{tone=romantic}$",
         size=FS_MICRO, color=ORANGE)

    # natural divergence -- what an endpoint-free but crossover-free audit would report
    text(ax, 2.0, 28.0, "(c)  Natural divergence $D^{\\mathrm{nat}}_s$ alone: four stages "
                        "look implicated", size=FS_HEAD, weight="bold")
    bx, bw = 6.0, 22.0
    for k, st in enumerate(STAGES):
        nat = MOV001[st][0]
        y = 23.0 - k * 3.9
        text(ax, bx - 0.8, y + 1.0, st, size=FS_TINY, color=INK, ha="right", va="center")
        ax.plot([bx, bx + bw], [y + 1.0, y + 1.0], color="#E4E4E4", lw=5.2,
                solid_capstyle="butt", zorder=2)
        if nat > 0:
            ax.plot([bx, bx + bw * nat], [y + 1.0, y + 1.0], color=GREY, lw=5.2,
                    solid_capstyle="butt", zorder=3)
        text(ax, bx + bw + 0.8, y + 1.0, f"{nat:.2f}", size=FS_MICRO, color=GREY, va="center",
             family="monospace")
    text(ax, 31.0, 18.0,
         "Retrieve, Rank, Explain and\nMemory all diverge. None of\nthem read $A$.\n"
         "A per-stage difference is\nnot a per-stage cause.",
         size=FS_SMALL, color=ORANGE, style="italic")

    # ------------------------------------------------- right: the 2x2 crossover itself
    PX = 46.0
    ax.plot([PX - 2.5, PX - 2.5], [3.0, 93.0], color=BLUE, lw=1.6, ls=(0, (6, 4)), zorder=1)

    text(ax, PX, 91.0, "(d)  The stage crossover $Y^{i,j}_s = \\phi_s(f_s(Z^i_s, A{=}j))$",
         size=FS_HEAD, weight="bold")
    text(ax, PX, 88.0,
         "$i$ = which arm's inherited parent state the stage is fed.   "
         "$j$ = which descriptor the stage itself is shown.",
         size=FS_SMALL, color=GREY)

    # the 2x2 grid
    GX, GY, CW, CH = PX + 9.0, 62.0, 15.0, 8.6
    text(ax, GX + CW, 85.6, "descriptor shown to the stage", size=FS_TINY, color=GREY,
         ha="center", weight="bold")
    text(ax, GX + CW * 0.5, 83.4, "$A=a$", size=FS_BODY, color=BLUE, ha="center", weight="bold")
    text(ax, GX + CW * 1.5, 83.4, "$A=a'$", size=FS_BODY, color=ORANGE, ha="center",
         weight="bold")
    text(ax, GX - 2.2, GY + CH + CH / 2, "$Z^{a}_s$", size=FS_BODY, color=BLUE, ha="right",
         va="center", weight="bold")
    text(ax, GX - 2.2, GY + CH / 2, "$Z^{a'}_s$", size=FS_BODY, color=ORANGE, ha="right",
         va="center", weight="bold")
    text(ax, GX - 7.0, GY + CH, "inherited\nparent state", size=FS_TINY, color=GREY,
         ha="center", va="center", weight="bold", rotation=90)

    cells = [((0, 0), "$Y^{a,a}$", "natural arm $a$", BLUE, "#EAF3FA"),
             ((0, 1), "$Y^{a,a'}$", "crossover", MUTED, "white"),
             ((1, 0), "$Y^{a',a}$", "crossover", MUTED, "white"),
             ((1, 1), "$Y^{a',a'}$", "natural arm $a'$", ORANGE, "#FDF1EA")]
    for (row, col), lab, sub, ec, fc in cells:
        x = GX + col * CW
        y = GY + (1 - row) * CH
        box(ax, x, y, CW, CH, fc=fc, ec=ec, lw=1.2, r=0.8)
        text(ax, x + CW / 2, y + CH * 0.62, lab, size=FS_HEAD, ha="center", va="center",
             weight="bold", color=ec if ec != MUTED else INK)
        text(ax, x + CW / 2, y + CH * 0.24, sub, size=FS_MICRO, color=GREY, ha="center",
             va="center", style="italic")

    # the two contrasts read off the grid
    arrow(ax, (GX + CW * 0.62, GY + CH * 1.5), (GX + CW * 1.38, GY + CH * 1.5),
          color=ORANGE, lw=1.5, style="<|-|>", ms=6)
    arrow(ax, (GX + CW * 0.62, GY + CH * 0.5), (GX + CW * 1.38, GY + CH * 0.5),
          color=ORANGE, lw=1.5, style="<|-|>", ms=6)
    arrow(ax, (GX + CW * 0.5, GY + CH * 1.22), (GX + CW * 0.5, GY + CH * 0.78),
          color=PURPLE, lw=1.5, style="<|-|>", ms=6)
    arrow(ax, (GX + CW * 1.5, GY + CH * 1.22), (GX + CW * 1.5, GY + CH * 0.78),
          color=PURPLE, lw=1.5, style="<|-|>", ms=6)

    text(ax, GX + CW * 2 + 2.0, GY + CH * 1.5,
         "$D^{A}_s$  rows: state fixed,\ndescriptor swapped\n= direct sensitivity",
         size=FS_SMALL, color=ORANGE, va="center", weight="bold")
    text(ax, GX + CW, GY - 2.0,
         "$D^{Z}_s$  columns: descriptor fixed, state swapped  =  inherited dependence",
         size=FS_SMALL, color=PURPLE, ha="center", va="top", weight="bold")
    text(ax, GX + CW, GY - 5.6,
         "and the diagonal, $D^{\\mathrm{nat}}_s = d_s(Y^{a,a}, Y^{a',a'})$, is what a "
         "stage-wise audit\nwithout the crossover sees: both effects added together",
         size=FS_MICRO, color=GREY, ha="center", va="top", style="italic")

    # ----------------------------------------- the resulting per-stage verdicts
    text(ax, PX, 49.0, "(e)  Same run, decomposed: the fault localizes to one stage",
         size=FS_HEAD, weight="bold")
    ry = 44.0
    for st in STAGES:
        nat, d, z = MOV001[st]
        dg = diagnose(nat, d, z)
        box(ax, PX, ry - 6.2, 50.0, 6.2, fc="white", ec="#DDDDDD", lw=0.7, r=0.7)
        text(ax, PX + 1.4, ry - 3.1, st.upper(), size=FS_TINY, weight="bold", va="center",
             color=DIAG_COLOR[dg])
        coord_bars(ax, PX + 13.5, ry - 5.8, 18.0, 5.4, nat, d, z)
        verdict(ax, PX + 36.5, ry - 3.1, DIAG_LABEL[dg] + "!", DIAG_COLOR[dg])
        ry -= 6.8

    text(ax, PX, 9.6,
         "Elicit is the only stage whose output moves when the descriptor is swapped under a "
         "fixed parent state ($D^{A}=0.33$).\n"
         "Everything downstream has $D^{A}=0$: the divergence is real but inherited. Repairing "
         "Elicit and rerunning descendants\n"
         "removes 0.58 of the final ranking gap; repairing any other stage removes 0.17.",
         size=FS_SMALL, color=INK, va="top")

    footer(ax, "Worked example: scenario mov_001, planted at Elicit.")
    return save(fig, outdir, "fig_scaff_v1_crossover_matrix")


# =================================================================== V2: stage tape
def build_v2(outdir):
    """The pipeline as a horizontal tape; each stage carries its own decomposition."""
    fig, ax = canvas(14.0, 6.6)

    text(ax, 50, 99.0,
         "Stage-Crossover Attribution: where protected-attribute dependence enters the "
         "pipeline, and where it is only passed along",
         size=FS_TITLE, weight="bold", ha="center", va="top")
    text(ax, 50, 95.4,
         "Each stage is re-run four times -- its own parent state and the other arm's, "
         "crossed with each descriptor -- so its divergence splits into a direct and an "
         "inherited part",
         size=FS_SMALL, color=GREY, ha="center", va="top")

    # ------------------------------------------------------ the tape
    n = len(STAGES)
    x0, colw, gap = 2.5, 17.4, 2.4
    TOP, TH = 78.0, 9.0

    text(ax, x0, 93.2, "$A$ (protected descriptor) is visible to every stage; "
                       "$Z_s$ (legitimate parent state) flows left to right",
         size=FS_SMALL, color=GREY, style="italic")

    for k, st in enumerate(STAGES):
        x = x0 + k * (colw + gap)
        nat, d, z = MOV001[st]
        dg = diagnose(nat, d, z)
        col = DIAG_COLOR[dg]

        # stage header
        box(ax, x, TOP, colw, TH, fc="white", ec=col, lw=1.5, r=0.9)
        box(ax, x, TOP + TH - 3.2, colw, 3.2, fc=col, ec=col, lw=1.5, r=0.9)
        text(ax, x + colw / 2, TOP + TH - 1.6, st.upper(), size=FS_HEAD, color="white",
             weight="bold", ha="center", va="center")
        text(ax, x + colw / 2, TOP + 2.6, {
            "Elicit": "ask? + preference state",
            "Retrieve": "candidate set",
            "Rank": "ranked list",
            "Explain": "reason tags",
            "Memory": "retained facts",
        }[st], size=FS_MICRO, color=GREY, ha="center", va="center", style="italic")

        # descriptor injection arrow, from a shared A rail above
        arrow(ax, (x + colw / 2, 90.5), (x + colw / 2, TOP + TH), color=ORANGE, lw=1.0,
              ls="dashed", ms=5)

        # state flow
        if k < n - 1:
            arrow(ax, (x + colw, TOP + TH / 2 - 1.5), (x + colw + gap, TOP + TH / 2 - 1.5),
                  color=GREY, lw=1.4, ms=7)
            text(ax, x + colw + gap / 2, TOP + TH / 2 + 1.0, f"$Z_{{{STAGES[k+1][:3]}}}$",
                 size=FS_MICRO, color=GREY, ha="center", va="bottom")

        # decomposition
        box(ax, x, 47.0, colw, 27.5, fc="#FAFAFA", ec="#E0E0E0", lw=0.7, r=0.8)
        text(ax, x + colw / 2, 72.6, "scenario mov_001", size=FS_MICRO, color=GREY,
             ha="center", va="top", style="italic")
        coord_bars(ax, x + 4.2, 61.0, 9.4, 8.4, nat, d, z, size=FS_MICRO)
        chip(ax, x + colw / 2 - 6.5, 55.5, DIAG_LABEL[dg], col, w=13.0, h=3.0)
        text(ax, x + colw / 2, 53.4, {
            "direct": "reads $A$ itself",
            "inherited": "passes on Elicit's error",
            "invariant": "no effect",
            "mixed": "both channels active",
        }[dg], size=FS_MICRO, color=col, ha="center", va="top")
        rel = ">" if d > MARGIN else r"\leq"
        text(ax, x + colw / 2, 50.2, f"$D^{{A}} {rel} 0.10$",
             size=FS_MICRO, color=GREY, ha="center", va="top")

    # the shared A rail
    ax.plot([x0 + colw / 2, x0 + (n - 1) * (colw + gap) + colw / 2], [90.5, 90.5],
            color=ORANGE, lw=1.2, zorder=3)
    text(ax, x0 + colw / 2 - 1.0, 90.5, "$A$", size=FS_BODY, color=ORANGE, weight="bold",
         ha="right", va="center")

    # ------------------------------------------------------ second row: corpus means
    text(ax, 2.5, 42.5, "Corpus means over the same design (movies, 24 scenarios x 4 repeats, "
                        "faults planted uniformly across stages)",
         size=FS_HEAD, weight="bold")
    for k, st in enumerate(STAGES):
        x = x0 + k * (colw + gap)
        nat, d, z = CORPUS[st]
        dg = diagnose(nat, d, z)
        box(ax, x, 22.0, colw, 16.5, fc="white", ec="#DDDDDD", lw=0.8, r=0.8)
        text(ax, x + colw / 2, 36.6, st.upper(), size=FS_TINY, weight="bold", ha="center",
             va="top", color=DIAG_COLOR[dg])
        coord_bars(ax, x + 4.2, 25.6, 9.4, 8.0, nat, d, z, size=FS_MICRO)
        text(ax, x + colw / 2, 24.4, DIAG_LABEL[dg].lower(), size=FS_MICRO,
             color=DIAG_COLOR[dg], ha="center", va="top", weight="bold")

    # ------------------------------------------------------ what it buys
    box(ax, 2.5, 4.0, 95.0, 15.0, fc="#F3F8FB", ec=BLUE, lw=1.0, r=1.0)
    text(ax, 4.2, 17.6, "What the decomposition buys, measured on this run",
         size=FS_HEAD, color=BLUE, weight="bold")
    facts = [
        ("Localization", "100%", "of 48 scenarios: the stage with the largest $D^{A}$ is the "
                                 "planted stage (or none, when nothing was planted)"),
        ("Masking", "36.5 / 43.8%", "of pairs are process-sensitive yet endpoint-equivalent "
                                    "(movies / restaurants) -- invisible to an output-only audit"),
        ("Repair", "0.58 vs 0.17", "final-gap removability when the implicated stage is "
                                   "repaired vs. when a wrong stage is"),
    ]
    fy = 13.6
    for name, val, why in facts:
        text(ax, 5.6, fy, name, size=FS_SMALL, color=BLUE, weight="bold")
        text(ax, 18.0, fy, val, size=FS_SMALL, color=INK, weight="bold", family="monospace")
        text(ax, 31.0, fy, why, size=FS_SMALL, color=INK)
        fy -= 3.6

    footer(ax)
    return save(fig, outdir, "fig_scaff_v2_stage_tape")


# ============================================================== V3: worked example
def build_v3(outdir):
    """Closest to CFaiRLLM Fig. 1: left, the concrete dialogue; right, the reading rule."""
    fig, ax = canvas(13.4, 7.8)

    text(ax, 50, 99.3,
         "Reading one paired trace: an unsupported preference invented at ELICIT "
         "surfaces as divergence at four later stages",
         size=FS_TITLE, weight="bold", ha="center", va="top")

    ax.plot([49.0, 49.0], [3.0, 94.5], color=BLUE, lw=1.6, ls=(0, (6, 4)), zorder=1)

    # -------------------------------------------------- LEFT: the trace
    text(ax, 2.0, 94.0, "The two arms, stage by stage", size=FS_HEAD, weight="bold")

    bubble(ax, 5.5, 84.0, 40.0, 8.4,
           "I want a mystery movie, and please avoid horror.",
           ec=GREY, head="seeker turn -- identical in both arms", head_color=GREY,
           size=FS_SMALL)

    rows = [
        ("ELICIT",
         'ask "preferred tone?"\npreference state += tone=romantic',
         'ask "preferred tone?"\npreference state unchanged',
         0.333, "the descriptor is read here"),
        ("RETRIEVE",
         "filters: genre=mystery, avoid_horror\ncandidates: 5 items",
         "filters: genre=mystery, avoid_horror\ncandidates: 5 items",
         0.200, "same filters; pool shifts only\nvia the changed state"),
        ("RANK",
         "top-1 = m18",
         "top-1 = m02",
         0.750, "top-1 flips"),
        ("EXPLAIN",
         "reason tags: genre=mystery,\n            tone=romantic",
         "reason tags: genre=mystery",
         0.500, "cites the invented preference"),
        ("MEMORY",
         "facts retained include tone=romantic",
         "facts retained: no tone fact",
         0.333, "the error is written down\nand carried into later turns"),
    ]
    y = 81.5
    for name, left, right, nat, note in rows:
        nlines = max(left.count("\n"), note.count("\n")) + 1
        h = 6.0 + 1.8 * nlines + 1.9 * note.count("\n")
        y -= h + 1.5
        box(ax, 5.5, y, 40.0, h, fc="white", ec="#DDDDDD", lw=0.8, r=0.8)
        text(ax, 6.8, y + h - 1.2, name, size=FS_TINY, weight="bold", color=INK)
        text(ax, 6.8, y + h - 4.0, left, size=FS_MICRO, color=BLUE, family="monospace")
        text(ax, 26.5, y + h - 1.2, "$A=a$ (man)", size=FS_MICRO, color=BLUE, weight="bold")
        text(ax, 26.5, y + h - 4.0, right, size=FS_MICRO, color=ORANGE, family="monospace")
        text(ax, 44.2, y + h - 1.2, "$A=a'$ (woman)", size=FS_MICRO, color=ORANGE,
             weight="bold", ha="right")
        text(ax, 6.8, y + 1.2 + 1.9 * note.count("\n"),
             f"$D^{{\\mathrm{{nat}}}} = {nat:.2f}$   {note}",
             size=FS_MICRO, color=GREY, style="italic")

    box(ax, 5.5, 9.0, 40.0, 10.5, fc="#FFF9EC", ec=GOLD, lw=1.1, ls="dashed", r=0.9)
    text(ax, 6.9, 18.0, "Latent profile $\\Omega$: tone = cerebral", size=FS_SMALL,
         color="#8A6100", weight="bold")
    text(ax, 6.9, 14.8,
         "so tone=romantic is not merely different, it is wrong -- which is what\n"
         "upgrades a procedural difference into a measurable harm. Without $\\Omega$\n"
         "we would report sensitivity and leave quality unidentified.",
         size=FS_MICRO, color=INK)

    # -------------------------------------------------- RIGHT: the rule
    RX = 52.0
    text(ax, RX, 94.0, "How each stage is classified", size=FS_HEAD, weight="bold")

    box(ax, RX, 64.5, 46.0, 27.0, fc="white", ec=INK, lw=1.1, r=0.9)
    text(ax, RX + 1.6, 89.8, "Two interventions per stage, one margin", size=FS_SMALL,
         weight="bold")
    text(ax, RX + 1.6, 86.4,
         "$D^{A}_s$: hold the parent state $Z_s$ at what this arm actually reached,\n"
         "swap only the descriptor the stage sees.\n"
         "$D^{Z}_s$: hold the descriptor fixed, swap the parent state.",
         size=FS_SMALL, color=INK)
    rule = [("$D^{A} > \\tau$, $D^{Z} > \\tau$", "MIXED", "both channels carry the effect"),
            ("$D^{A} > \\tau$", "DIRECT", "the stage itself reads $A$"),
            ("$D^{A} \\leq \\tau$, $D^{Z} > \\tau$", "INHERITED",
             "the effect arrives from upstream"),
            ("otherwise", "INVARIANT", "no dependence at this stage")]
    ry = 78.0
    for cond, lab, why in rule:
        col = DIAG_COLOR[lab.lower()]
        text(ax, RX + 2.6, ry, cond, size=FS_MICRO, color=INK, va="center", family="monospace")
        chip(ax, RX + 17.0, ry - 1.2, lab, col, w=11.5, h=2.4, size=FS_MICRO)
        text(ax, RX + 29.8, ry, why, size=FS_MICRO, color=GREY, va="center")
        ry -= 3.6
    text(ax, RX + 1.6, 65.9, "$\\tau = 0.10$, the direct-effect margin; the within-arm repeat "
                             "noise is reported alongside so a margin is not read as signal.",
         size=FS_MICRO, color=GREY, style="italic")

    # applied to the trace
    text(ax, RX, 61.0, "Applied to the trace on the left", size=FS_HEAD, weight="bold")
    ry = 56.5
    for st in STAGES:
        nat, d, z = MOV001[st]
        dg = diagnose(nat, d, z)
        box(ax, RX, ry - 6.8, 46.0, 6.8, fc="white", ec="#DDDDDD", lw=0.7, r=0.7)
        text(ax, RX + 1.4, ry - 3.4, st.upper(), size=FS_TINY, weight="bold", va="center",
             color=DIAG_COLOR[dg])
        coord_bars(ax, RX + 13.0, ry - 6.2, 16.0, 5.6, nat, d, z)
        verdict(ax, RX + 34.0, ry - 3.4, DIAG_LABEL[dg] + "!", DIAG_COLOR[dg])
        ry -= 7.4

    box(ax, RX, 6.0, 46.0, 14.0, fc="#EDF7F3", ec=GREEN, lw=1.0, r=0.9)
    text(ax, RX + 1.6, 18.6, "The consequence for repair", size=FS_SMALL, color=GREEN,
         weight="bold")
    text(ax, RX + 1.6, 15.4,
         "Replace ELICIT's output with the counterpart arm's and rerun every\n"
         "descendant: the final ranking gap falls by 0.58. Doing the same at a stage\n"
         "the crossover exonerates removes 0.17. A diagnosis that cannot be acted on\n"
         "differently from a guess is not a diagnosis.",
         size=FS_MICRO, color=INK)

    footer(ax, "Scenario mov_001; a second scenario planted at Retrieve localizes there "
               "instead ($D^{A}_{\\mathrm{Retrieve}} = 0.83$).")
    return save(fig, outdir, "fig_scaff_v3_worked_example")


# ================================================================ V4: causal DAG
def build_v4(outdir):
    """The formal object: which edges each intervention cuts, and what repair does."""
    fig, ax = canvas(13.2, 6.2)

    text(ax, 50, 99.0,
         "The two SCAFF interventions as edge deletions: $D^{A}_s$ cuts the descriptor's "
         "direct edge into stage $s$; $D^{Z}_s$ cuts the path through its parents",
         size=FS_TITLE, weight="bold", ha="center", va="top")

    def panel(px, pw, title, subtitle, accent, *, active, cut, note):
        """One copy of the DAG; ``active`` names the edges drawn live, ``cut`` the severed ones."""
        box(ax, px, 24.0, pw, 62.0, fc="white", ec=accent, lw=1.3, r=1.0)
        box(ax, px, 80.0, pw, 6.0, fc=accent, ec=accent, lw=1.3, r=1.0)
        text(ax, px + pw / 2, 83.0, title, size=FS_HEAD, color="white", weight="bold",
             ha="center", va="center")
        text(ax, px + pw / 2, 77.6, subtitle, size=FS_MICRO, color=GREY, ha="center", va="top",
             style="italic")

        # stage chain along the bottom of the panel
        n = len(STAGES)
        w = (pw - 6.0) / n
        cy = 46.0
        pos = {}
        for k, st in enumerate(STAGES):
            x = px + 3.0 + k * w
            pos[st] = (x + w / 2, cy)
            focus = st == "Rank"
            box(ax, x + 0.5, cy - 3.2, w - 1.0, 6.4,
                fc="#FFF3E9" if focus else "#F7F7F7",
                ec=accent if focus else GREY, lw=1.3 if focus else 0.8, r=0.7)
            text(ax, x + w / 2, cy, st[:4] if w < 9 else st, size=FS_MICRO,
                 weight="bold" if focus else "normal", ha="center", va="center",
                 color=accent if focus else INK)
            if k < n - 1:
                nxt = px + 3.0 + (k + 1) * w
                # the Z-edge feeding Rank is the one D^Z_Rank manipulates
                is_target_edge = STAGES[k + 1] == "Rank"
                live = not (is_target_edge and "Z" in cut)
                arrow(ax, (x + w - 0.5, cy), (nxt + 0.5, cy),
                      color=PURPLE if is_target_edge else GREY,
                      lw=1.5 if is_target_edge else 0.9,
                      ls="solid" if live else (0, (1.5, 1.8)),
                      ms=6)
                if is_target_edge:
                    text(ax, (x + w + nxt) / 2, cy - 4.6, "$Z_{\\mathrm{Rank}}$",
                         size=FS_MICRO, color=PURPLE, ha="center", va="top")

        # A node above, with an edge into every stage
        ay = 68.0
        box(ax, px + pw / 2 - 5.0, ay - 3.0, 10.0, 6.0, fc="#FDF1EA", ec=ORANGE, lw=1.3, r=0.7)
        text(ax, px + pw / 2, ay, "$A$", size=FS_BODY, color=ORANGE, weight="bold",
             ha="center", va="center")
        for st in STAGES:
            sx, sy = pos[st]
            is_target = st == "Rank"
            live = not (is_target and "A" in cut)
            arrow(ax, (px + pw / 2, ay - 3.0), (sx, sy + 3.2),
                  color=ORANGE if is_target else "#F0C4AC",
                  lw=1.5 if is_target else 0.8,
                  ls="solid" if live else (0, (1.5, 1.8)),
                  ms=5, rad=0.12 if sx < px + pw / 2 else -0.12)

        # scissors marker on whichever edge is cut
        if "A" in cut:
            mx, my = (px + pw / 2 + pos["Rank"][0]) / 2, (ay + pos["Rank"][1]) / 2 + 1.0
            for dx, dy in ((-1.6, -1.3), (-1.6, 1.3)):
                ax.plot([mx + dx, mx - dx], [my + dy, my - dy], color=ORANGE, lw=1.6, zorder=8)
            text(ax, mx + 3.0, my, "held fixed", size=FS_MICRO, color=ORANGE, va="center",
                 weight="bold")
        if "Z" in cut:
            k = STAGES.index("Rank")
            mx = pos[STAGES[k - 1]][0] + (pos["Rank"][0] - pos[STAGES[k - 1]][0]) / 2
            my = cy
            for dx, dy in ((-1.0, -1.2), (1.0, -1.2)):
                ax.plot([mx + dx, mx - dx], [my + dy, my - dy], color=PURPLE, lw=1.6, zorder=8)
            text(ax, mx - 1.6, my + 4.6, "swapped", size=FS_MICRO, color=PURPLE, ha="right",
                 weight="bold")

        text(ax, px + 2.5, 38.0, active, size=FS_SMALL, color=accent, weight="bold")
        text(ax, px + 2.5, 34.5, note, size=FS_MICRO, color=INK)

    panel(2.0, 30.0, "Natural  $D^{\\mathrm{nat}}_s$",
          "run both arms end to end, compare stage $s$", GREY,
          active="both channels open",
          cut=set(),
          note="What a stage-wise audit without the\ncrossover reports. A large value at\n"
               "RANK is compatible with RANK\nnever having seen $A$.")

    panel(35.0, 30.0, "Direct  $D^{A}_s$",
          "fix $Z_s$, swap only the descriptor at $s$", ORANGE,
          active="descriptor channel isolated",
          cut={"A"},
          note="Non-zero only if stage $s$ itself\nresponds to $A$. This is the\n"
               "quantity that localizes a fault --\nand the one a procedural-fairness\n"
               "claim needs.")

    panel(68.0, 30.0, "Inherited  $D^{Z}_s$",
          "fix the descriptor, swap the parent state", PURPLE,
          active="state channel isolated",
          cut={"Z"},
          note="Non-zero when the stage is merely\nfaithful to an input that upstream\n"
               "already corrupted. Real disparity,\nwrong place to intervene.")

    # ------------------------------------------------------ bottom strip
    box(ax, 2.0, 3.0, 96.0, 14.0, fc="#F3F8FB", ec=BLUE, lw=1.0, r=1.0)
    text(ax, 3.6, 15.6, "And the test that the decomposition is not just bookkeeping",
         size=FS_HEAD, color=BLUE, weight="bold")
    text(ax, 3.6, 12.0,
         "Replace the implicated stage's output with the counterpart arm's and rerun every "
         "descendant. If the attribution is right, the downstream gap should collapse -- and "
         "it does: removability 0.58 at the stage $D^{A}$ implicates,\n"
         "0.17 at a stage it exonerates (movies; 0.59 vs 0.19 restaurants). Localization "
         "recovers the planted stage in 48/48 scenarios. In 36.5% of pairs the process is "
         "descriptor-sensitive while the final ranking stays\n"
         "within the endpoint-equivalence margin -- the cases an output-only audit scores as "
         "fair.",
         size=FS_SMALL, color=INK)

    footer(ax)
    return save(fig, outdir, "fig_scaff_v4_causal_dag")


# =========================================== PAPER: print-fit, 1:1 at text width
def build_paper(outdir):
    """The variant actually used in the manuscript.

    v1--v4 are drawn at ~13.5in wide, which is right for a slide or a screen but wrong for a
    paper: included at \\linewidth in acmart's single-column manuscript mode (~6.75in) they are
    downscaled by half, taking 6pt type to 3pt. This variant is drawn at the target width so
    the scale factor is 1 and the type sizes below are the sizes that reach the page. Paying
    for that means carrying less: the paired-prompt panel and the natural-divergence bar chart
    are dropped, and what remains is the crossover itself and the verdicts it produces.
    """
    fig, ax = canvas(6.9, 4.35)

    # at this figure size one y-unit is ~3.1pt, so a text line costs ~2.7 units. Sizes are
    # deliberately near the ACM minimum for figure text and are 1:1 on the page.
    T_TITLE, T_HEAD, T_BODY, T_SMALL = 7.8, 6.8, 6.0, 5.4

    ax.text(50, 98.6,
            "Holding the inherited state fixed separates a stage that reads the descriptor\n"
            "from one that only inherits its effect",
            fontsize=T_TITLE, fontweight="bold", ha="center", va="top", linespacing=1.3,
            color=INK)

    # ------------------------------------------------------- left: the 2x2 crossover
    GX, GY, CW, CH = 12.0, 44.0, 15.5, 12.0
    ax.text(24.0, 84.0, "(a)  the crossover  $Y^{i,j}_s = \\phi_s(f_s(Z^i_s, A{=}j))$",
            fontsize=T_HEAD, fontweight="bold", ha="center", va="top", color=INK)

    text(ax, GX + CW, 76.6, "descriptor shown to the stage", size=T_SMALL, color=GREY,
         ha="center", weight="bold")
    text(ax, GX + CW * 0.5, 71.8, "$A=a$", size=T_BODY, color=BLUE, ha="center", weight="bold")
    text(ax, GX + CW * 1.5, 71.8, "$A=a'$", size=T_BODY, color=ORANGE, ha="center",
         weight="bold")
    text(ax, GX - 1.6, GY + CH * 1.5, "$Z^{a}_s$", size=T_BODY, color=BLUE, ha="right",
         va="center", weight="bold")
    text(ax, GX - 1.6, GY + CH * 0.5, "$Z^{a'}_s$", size=T_BODY, color=ORANGE, ha="right",
         va="center", weight="bold")
    text(ax, GX - 8.4, GY + CH, "inherited\nparent state", size=T_SMALL, color=GREY,
         ha="center", va="center", weight="bold", rotation=90)

    for (row, col), lab, ec, fc in [((0, 0), "$Y^{a,a}$", BLUE, "#EAF3FA"),
                                    ((0, 1), "$Y^{a,a'}$", MUTED, "white"),
                                    ((1, 0), "$Y^{a',a}$", MUTED, "white"),
                                    ((1, 1), "$Y^{a',a'}$", ORANGE, "#FDF1EA")]:
        x, y = GX + col * CW, GY + (1 - row) * CH
        box(ax, x, y, CW, CH, fc=fc, ec=ec, lw=1.1, r=0.9)
        text(ax, x + CW / 2, y + CH / 2, lab, size=T_BODY, ha="center", va="center",
             weight="bold", color=ec if ec != MUTED else INK)

    # the two contrasts
    for yy in (GY + CH * 1.5, GY + CH * 0.5):
        arrow(ax, (GX + CW * 0.68, yy), (GX + CW * 1.32, yy), color=ORANGE, lw=1.3,
              style="<|-|>", ms=5)
    for xx in (GX + CW * 0.5, GX + CW * 1.5):
        arrow(ax, (xx, GY + CH * 1.28), (xx, GY + CH * 0.72), color=PURPLE, lw=1.3,
              style="<|-|>", ms=5)

    text(ax, GX + CW, GY - 2.2,
         "$D^{A}_s$ rows: state fixed, descriptor swapped  $=$  direct",
         size=T_SMALL, color=ORANGE, ha="center", va="top", weight="bold")
    text(ax, GX + CW, GY - 6.4,
         "$D^{Z}_s$ cols: descriptor fixed, state swapped  $=$  inherited",
         size=T_SMALL, color=PURPLE, ha="center", va="top", weight="bold")
    text(ax, GX + CW, GY - 10.6,
         "diagonal $=D^{\\mathrm{nat}}_s$: both channels at once",
         size=T_SMALL, color=GREY, ha="center", va="top", style="italic")

    # ------------------------------------------- right: per-stage decomposition
    RX = 51.0
    ax.text(RX + 23.5, 84.0, "(b)  one scenario, fault planted at Elicit",
            fontsize=T_HEAD, fontweight="bold", ha="center", va="top", color=INK)

    ry = 79.0
    for st in STAGES:
        nat, d, z = MOV001[st]
        dg = diagnose(nat, d, z)
        box(ax, RX, ry - 9.6, 47.0, 9.6, fc="white", ec="#DDDDDD", lw=0.6, r=0.6)
        text(ax, RX + 1.4, ry - 4.8, st.upper(), size=T_SMALL, weight="bold", va="center",
             color=DIAG_COLOR[dg])
        coord_bars(ax, RX + 14.0, ry - 9.0, 16.0, 8.2, nat, d, z, size=T_SMALL - 0.6)
        verdict(ax, RX + 37.0, ry - 4.8, DIAG_LABEL[dg] + "!", DIAG_COLOR[dg], size=T_SMALL)
        ry -= 10.8

    # ------------------------------------------------------------- bottom strip
    box(ax, 2.0, 3.0, 96.0, 21.0, fc="#F7F9FA", ec="#CFD8DC", lw=0.8, r=0.8)
    text(ax, 3.8, 21.6,
         "Elicit adds an unsupported preference on the $A{=}a$ arm. Four later stages diverge; "
         "none of them reads $A$.",
         size=T_SMALL, color=INK, weight="bold")
    text(ax, 3.8, 17.0,
         "Only Elicit has $D^{A}_s > 0$, so only Elicit is implicated. Repairing it and "
         "rerunning descendants removes 0.58 of the\n"
         "final ranking gap against 0.17 for a stage the crossover exonerates; localization "
         "recovers the planted stage in 48/48\n"
         "scenarios. The planted fault is a unit test of the instrument, not a claim about "
         "group preferences.",
         size=T_SMALL, color=INK)

    return save(fig, outdir, "fig_scaff_paper")


# =============================================================== contact sheet
def contact_sheet(outdir, stems):
    import matplotlib.image as mpimg

    n = len(stems)
    fig, axes = plt.subplots(n, 1, figsize=(11.0, 5.4 * n))
    if n == 1:
        axes = [axes]
    for ax, stem in zip(axes, stems):
        ax.imshow(mpimg.imread(os.path.join(outdir, f"{stem}.png")))
        ax.set_title(stem, fontsize=9, color=GREY, loc="left")
        ax.axis("off")
    fig.tight_layout()
    out = os.path.join(outdir, "fig_scaff_contact_sheet.png")
    fig.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return out


BUILDERS = {"v1": build_v1, "v2": build_v2, "v3": build_v3, "v4": build_v4,
            "paper": build_paper}
STEMS = {
    "v1": "fig_scaff_v1_crossover_matrix",
    "v2": "fig_scaff_v2_stage_tape",
    "v3": "fig_scaff_v3_worked_example",
    "v4": "fig_scaff_v4_causal_dag",
    "paper": "fig_scaff_paper",
}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default=os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "paper", "figures"))
    ap.add_argument("--only", default="v1,v2,v3,v4,paper")
    args = ap.parse_args()

    keys = [k.strip() for k in args.only.split(",") if k.strip()]
    for key in keys:
        for path in BUILDERS[key](args.outdir):
            print(f"wrote {path}")
    print(f"wrote {contact_sheet(args.outdir, [STEMS[k] for k in keys])}")
