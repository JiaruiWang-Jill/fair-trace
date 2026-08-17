#!/usr/bin/env python3
"""Generate the high-level evaluation-structure figure in five alternative designs.

  paper/figures/fig_paired_a_two_rails.{pdf,png}
  paper/figures/fig_paired_b_mirror_ladder.{pdf,png}
  paper/figures/fig_paired_c_split_screen.{pdf,png}
  paper/figures/fig_paired_d_fork_converge.{pdf,png}
  paper/figures/fig_paired_e_turn_loop.{pdf,png}
  paper/figures/fig_paired_contact_sheet.png     (all five, for choosing)

WHAT THIS FIGURE IS FOR
-----------------------
One job: let a reader who has never seen the paper understand the evaluation design at a glance.
Two paired trajectories that differ only in the protected descriptor (man / woman), each run
through all five stages, compared stage by stage rather than only at the final recommendation.

Deliberately NOT in this figure: the stage crossover, direct vs inherited dependence, distance
functions, thresholds, oracles, or any measured value. Those belong to the method figure
(scripts/make_fig_scaff.py) and to the results tables. Anything here that looked like a number
would invite the reader to start judging results before they know what is being compared.

Every design is drawn 1:1 at ACM text width (about 6.9in), so `\\includegraphics[width=\\linewidth]`
applies no scaling and the type sizes below are the sizes that reach the page.

Colours: the two arms are green and purple from the Okabe-Ito set. Blue/orange or blue/pink would
map the arms onto a gender colour convention, which is the last thing a fairness figure should do.
Blue is reserved for what is held identical across the two arms.

The illustrative outputs are the movie scenario of the controlled prototype, stated in words
rather than as data: Elicit asks about tone in both arms but on one arm records a mood the user
never expressed, and that difference then shows up in the ranking, the explanation and the
retained profile. Item titles are written as "film X"/"film Y" rather than invented, so nothing
here can be misread as a result.

Usage:  python3 scripts/make_fig_eval_paired.py [--outdir paper/figures] [--only a,c]
"""

from __future__ import annotations

import argparse
import os

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import Arc, Circle, FancyArrowPatch, FancyBboxPatch, Polygon

# --- palette ---------------------------------------------------------------
ARM_A = "#009E73"   # arm A -- "I am a man."
ARM_B = "#7B52AB"   # arm B -- "I am a woman."
SHARED = "#0072B2"  # everything held identical across the two arms
GREY = "#5A5A5A"
MUTED = "#8C8C8C"
INK = "#1A1A1A"
PANEL = "#F7F7F5"
TINT_A = "#E6F5F0"
TINT_B = "#F0EAF7"
TINT_S = "#EAF3FA"

# type sizes, 1:1 on the page
T_TITLE = 8.4
T_SUB = 6.4
T_BODY = 6.4
T_SMALL = 5.8
T_TINY = 5.1

STAGES = ["Elicit", "Retrieve", "Rank", "Explain", "Memory"]

# what the audit looks at, per stage. Plain language, no symbols.
OBSERVABLE_2L = {
    "Elicit": "what it asks,\nwhat it records",
    "Retrieve": "which items\nit pulls",
    "Rank": "the order it\nputs them in",
    "Explain": "the reasons\nit gives",
    "Memory": "what it keeps\nabout the user",
}
OBSERVABLE_1L = {
    "Elicit": "what it asks, and what it records",
    "Retrieve": "which items it pulls",
    "Rank": "the order it puts them in",
    "Explain": "the reasons it gives",
    "Memory": "what it keeps about the user",
}

ARM_A_LABEL = "“I am a man.”"
ARM_B_LABEL = "“I am a woman.”"
REQUEST_2L = "“I want a mystery movie,\nand please avoid horror.”"
REQUEST_3L = "“I want a mystery\nmovie, and please\navoid horror.”"
REQUEST_1L = "“I want a mystery movie, and please avoid horror.”"

HELD = "same request  ·  same catalog  ·  same model  ·  same decoding"


# ---------------------------------------------------------------- primitives
def box(ax, x, y, w, h, *, fc="white", ec=GREY, lw=0.9, ls="solid", r=1.0, z=1, alpha=1.0):
    p = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0,rounding_size={r}",
        facecolor=fc, edgecolor=ec, linewidth=lw, linestyle=ls, zorder=z, alpha=alpha,
    )
    ax.add_patch(p)
    return p


def text(ax, x, y, s, *, size=T_BODY, color=INK, weight="normal", ha="left", va="top",
         style="normal", z=5, family=None, rotation=0, spacing=1.35):
    ax.text(x, y, s, fontsize=size, color=color, fontweight=weight, ha=ha, va=va,
            fontstyle=style, zorder=z, linespacing=spacing, family=family, rotation=rotation)


def arrow(ax, xy1, xy2, *, color=GREY, lw=1.0, ls="solid", style="-|>", rad=0.0, z=4, ms=6):
    a = FancyArrowPatch(
        xy1, xy2, arrowstyle=style, mutation_scale=ms, color=color, linewidth=lw,
        linestyle=ls, connectionstyle=f"arc3,rad={rad}", zorder=z, shrinkA=1.0, shrinkB=1.0,
    )
    ax.add_patch(a)
    return a


def bubble(ax, x, y, w, h, body, *, ec=INK, fc="white", tail="left", lw=1.0, size=T_SMALL,
           color=INK):
    box(ax, x, y, w, h, fc=fc, ec=ec, lw=lw, r=0.8, z=2)
    ty = y + h * 0.60
    dx = -1.8 if tail == "left" else 1.8
    xe = x if tail == "left" else x + w
    ax.add_patch(Polygon([(xe, ty + 0.9), (xe, ty - 0.9), (xe + dx, ty - 2.0)], closed=True,
                         facecolor=fc, edgecolor=ec, linewidth=lw, zorder=2))
    text(ax, x + w / 2, y + h / 2, body, size=size, color=color, ha="center", va="center")


def person(ax, cx, cy, *, color, r=1.5):
    """A minimal head-and-shoulders glyph, so the arms read as two users, not two configs."""
    ax.add_patch(Circle((cx, cy + r * 1.05), r * 0.60, facecolor="white", edgecolor=color,
                        linewidth=1.0, zorder=6))
    ax.add_patch(Arc((cx, cy - r * 0.60), r * 2.2, r * 2.2, theta1=25, theta2=155,
                     edgecolor=color, linewidth=1.0, zorder=6))


def output_pill(ax, x, y, w, h, color, tint, *, label=None, size=T_TINY):
    """One arm's output at one stage. Tinted rather than empty so it reads as a filled slot."""
    box(ax, x, y, w, h, fc=tint, ec=color, lw=1.0, r=0.6, z=2)
    if label:
        text(ax, x + w / 2, y + h / 2, label, size=size, color=color, ha="center", va="center")


def canvas(w, h):
    fig, ax = plt.subplots(figsize=(w, h))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")
    return fig, ax


def save(fig, outdir, stem):
    os.makedirs(outdir, exist_ok=True)
    pdf = os.path.join(outdir, f"{stem}.pdf")
    png = os.path.join(outdir, f"{stem}.png")
    fig.savefig(pdf, bbox_inches="tight", pad_inches=0.03)
    fig.savefig(png, dpi=300, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    return pdf, png


def held_note(ax, y, *, x=50, size=T_SMALL):
    text(ax, x, y, HELD, size=size, color=SHARED, weight="bold", ha="center")


# ======================================================== A: two parallel rails
def build_a(outdir):
    """Two horizontal trajectories, compared column by column. The most literal reading."""
    fig, ax = canvas(6.9, 4.0)

    text(ax, 50, 98.6,
         "We run the same conversation twice, changing only who the user says they are,\n"
         "and compare what the agent does at every stage",
         size=T_TITLE, weight="bold", ha="center", spacing=1.3)

    # the shared request
    person(ax, 8.5, 58.5, color=SHARED, r=2.0)
    box(ax, 0.5, 41.0, 16.5, 12.5, fc=TINT_S, ec=SHARED, lw=1.0, r=0.8)
    text(ax, 8.75, 47.2, REQUEST_3L, size=T_TINY, color=INK, ha="center", va="center",
         spacing=1.25)
    text(ax, 8.75, 39.2, "one real request", size=T_TINY, color=SHARED, ha="center", va="top",
         style="italic")

    x0, cw, gap = 21.0, 13.4, 2.2
    ROW_A, ROW_B, BH = 63.0, 28.0, 9.0

    # the two rails
    for label, ry, col, above in ((ARM_A_LABEL, ROW_A, ARM_A, True),
                                  (ARM_B_LABEL, ROW_B, ARM_B, False)):
        text(ax, x0, ry + BH + 1.4 if above else ry - 1.6, label, size=T_SMALL, color=col,
             weight="bold", ha="left", va="bottom" if above else "top")
        for k, st in enumerate(STAGES):
            x = x0 + k * (cw + gap)
            box(ax, x, ry, cw, BH, fc="white", ec=col, lw=1.2, r=0.8)
            text(ax, x + cw / 2, ry + BH / 2, st.upper(), size=T_SMALL, color=col,
                 weight="bold", ha="center", va="center")
            if k < len(STAGES) - 1:
                arrow(ax, (x + cw, ry + BH / 2), (x + cw + gap, ry + BH / 2), color=col,
                      lw=1.0, ms=5)

    arrow(ax, (17.3, 50.5), (x0 - 0.8, ROW_A + BH / 2), color=ARM_A, lw=1.2, rad=-0.18)
    arrow(ax, (17.3, 44.0), (x0 - 0.8, ROW_B + BH / 2), color=ARM_B, lw=1.2, rad=0.18)

    # the comparison, once per stage, fed from both rails
    CY, CH = 47.9, 4.8
    text(ax, 19.0, CY + CH / 2, "compare", size=T_TINY, color=INK, weight="bold", ha="center",
         va="center", rotation=90)
    for k, st in enumerate(STAGES):
        x = x0 + k * (cw + gap)
        xc = x + cw / 2
        box(ax, x + 0.4, CY, cw - 0.8, CH, fc=PANEL, ec="#D6D6D6", lw=0.7, r=0.6)
        text(ax, xc, CY + CH / 2, OBSERVABLE_2L[st], size=T_TINY, color=INK, ha="center",
             va="center", spacing=1.2)
        arrow(ax, (xc, ROW_A), (xc, CY + CH), color=INK, lw=1.1, ms=5)
        arrow(ax, (xc, ROW_B + BH), (xc, CY), color=INK, lw=1.1, ms=5)

    held_note(ax, 18.0)
    text(ax, 50, 13.0,
         "Because everything except the descriptor is held identical, a difference in any column "
         "is attributable to the descriptor.\n"
         "An audit that reads only the last column cannot see the first four.",
         size=T_SMALL, color=INK, ha="center", spacing=1.35)

    return save(fig, outdir, "fig_paired_a_two_rails")


# ---------------------------------------------------------- glyphs used by B
def stage_glyph(ax, cx, cy, stage, color, arm):
    """A schematic of the artifact a stage produces.

    The two arms are drawn slightly differently exactly where the illustrative scenario says
    they differ (order at Rank, an extra reason at Explain, an extra retained fact at Memory)
    and identically where it says they agree.
    """
    if stage == "Elicit":
        box(ax, cx - 8.0, cy - 2.4, 16.0, 4.8, fc="white", ec=color, lw=0.9, r=1.4, z=3)
        text(ax, cx, cy, "“which tone?”", size=T_TINY, color=color, ha="center",
             va="center", z=4)
        text(ax, cx + 9.4, cy,
             "+ a mood\nnever stated" if arm == "a" else "records only\nwhat was said",
             size=T_TINY, color=color, ha="left", va="center", style="italic", spacing=1.15,
             z=4)
    elif stage == "Retrieve":
        for i, w in enumerate((9.0, 7.4, 8.2, 6.6)):
            yy = cy + 2.7 - i * 1.8
            ax.plot([cx - 4.5, cx - 4.5 + w], [yy, yy], color=color, lw=1.7,
                    solid_capstyle="round", zorder=3)
        text(ax, cx + 6.2, cy, "same\nshortlist", size=T_TINY, color=color, ha="left",
             va="center", style="italic", spacing=1.15, z=4)
    elif stage == "Rank":
        widths = (8.5, 6.0, 4.0) if arm == "a" else (6.0, 8.5, 4.0)
        for i, w in enumerate(widths):
            yy = cy + 2.6 - i * 2.6
            text(ax, cx - 7.6, yy, f"{i + 1}.", size=T_TINY, color=color, va="center", z=4)
            ax.plot([cx - 5.0, cx - 5.0 + w], [yy, yy], color=color, lw=2.0,
                    solid_capstyle="round", zorder=3)
        text(ax, cx + 5.4, cy, "different\ntop pick", size=T_TINY, color=color, ha="left",
             va="center", style="italic", spacing=1.15, z=4)
    elif stage == "Explain":
        lines = (8.5, 7.0, 5.5) if arm == "a" else (8.5, 7.0)
        for i, w in enumerate(lines):
            yy = cy + 2.4 - i * 1.9
            ax.plot([cx - 7.5, cx - 7.5 + w], [yy, yy], color=color, lw=1.5,
                    solid_capstyle="round", zorder=3)
        text(ax, cx + 2.4, cy,
             "cites the\nmood too" if arm == "a" else "cites the\ngenre only",
             size=T_TINY, color=color, ha="left", va="center", style="italic", spacing=1.15,
             z=4)
    else:  # Memory
        box(ax, cx - 8.4, cy - 3.0, 9.6, 6.0, fc="white", ec=color, lw=0.9, r=0.6, z=3)
        n = 3 if arm == "a" else 2
        for i in range(n):
            yy = cy + 1.6 - i * 1.6
            ax.plot([cx - 7.2, cx - 2.0], [yy, yy], color=color, lw=1.3,
                    solid_capstyle="round", zorder=4)
        text(ax, cx + 2.4, cy,
             "keeps the\ninvented mood" if arm == "a" else "keeps only\nwhat was said",
             size=T_TINY, color=color, ha="left", va="center", style="italic", spacing=1.15,
             z=4)


# ====================================================== B: mirrored stage ladder
def build_b(outdir):
    """The five stages down the spine, the two arms facing each other across it."""
    fig, ax = canvas(6.9, 5.2)

    text(ax, 50, 98.8, "The evaluation unit is a pair of trajectories, not a single answer",
         size=T_TITLE, weight="bold", ha="center")
    text(ax, 50, 95.2,
         "identical conversation, identical catalog — the two arms differ only in the descriptor "
         "the user gives",
         size=T_SUB, color=GREY, ha="center")

    # the shared request, feeding both arms
    box(ax, 34.0, 84.0, 32.0, 7.6, fc=TINT_S, ec=SHARED, lw=1.0, r=0.8)
    text(ax, 50, 87.8, REQUEST_2L, size=T_TINY, color=INK, ha="center", va="center")
    person(ax, 6.0, 85.6, color=ARM_A, r=1.8)
    text(ax, 9.6, 86.2, ARM_A_LABEL, size=T_SMALL, color=ARM_A, weight="bold", va="center")
    person(ax, 94.0, 85.6, color=ARM_B, r=1.8)
    text(ax, 90.4, 86.2, ARM_B_LABEL, size=T_SMALL, color=ARM_B, weight="bold", va="center",
         ha="right")
    arrow(ax, (33.6, 87.8), (22.0, 87.8), color=SHARED, lw=1.0, ms=5)
    arrow(ax, (66.4, 87.8), (78.0, 87.8), color=SHARED, lw=1.0, ms=5)

    # column headers
    text(ax, 19.0, 79.6, "what this arm produced", size=T_TINY, color=ARM_A, ha="center",
         weight="bold")
    text(ax, 50.0, 79.6, "the stage — compared here", size=T_TINY, color=INK, ha="center",
         weight="bold")
    text(ax, 81.0, 79.6, "what this arm produced", size=T_TINY, color=ARM_B, ha="center",
         weight="bold")

    y0, rh, rgap = 76.0, 11.6, 2.6
    for k, st in enumerate(STAGES):
        y = y0 - k * (rh + rgap) - rh
        yc = y + rh / 2

        # the arms' artifacts
        box(ax, 3.0, y + 0.6, 32.0, rh - 1.2, fc="white", ec=ARM_A, lw=1.0, r=0.7)
        stage_glyph(ax, 15.0, yc, st, ARM_A, "a")
        box(ax, 65.0, y + 0.6, 32.0, rh - 1.2, fc="white", ec=ARM_B, lw=1.0, r=0.7)
        stage_glyph(ax, 77.0, yc, st, ARM_B, "b")

        # the stage itself
        box(ax, 38.0, y, 24.0, rh, fc=PANEL, ec=GREY, lw=1.1, r=0.8, z=3)
        text(ax, 50, yc + 2.0, st.upper(), size=T_BODY, weight="bold", ha="center",
             va="center", color=INK, z=4)
        text(ax, 50, yc - 2.0, OBSERVABLE_2L[st], size=T_TINY, color=GREY, ha="center",
             va="center", style="italic", spacing=1.2, z=4)

        arrow(ax, (35.4, yc), (37.6, yc), color=ARM_A, lw=1.1, ms=5)
        arrow(ax, (64.6, yc), (62.4, yc), color=ARM_B, lw=1.1, ms=5)

        if k < len(STAGES) - 1:
            arrow(ax, (50, y), (50, y - rgap), color=GREY, lw=1.0, ms=5)

    held_note(ax, 6.8)
    text(ax, 50, 2.8,
         "Five comparisons per turn instead of one at the end, so a difference is located at the "
         "stage where it appears.",
         size=T_SMALL, color=INK, ha="center")

    return save(fig, outdir, "fig_paired_b_mirror_ladder")


# ============================================= C: split screen, concrete outputs
def build_c(outdir):
    """The concrete version: what the two arms actually produced, stage by stage."""
    fig, ax = canvas(6.9, 4.3)

    text(ax, 50, 98.8, "What the audit compares: the two arms side by side, at every stage",
         size=T_TITLE, weight="bold", ha="center")

    box(ax, 22.0, 87.5, 56.0, 7.0, fc=TINT_S, ec=SHARED, lw=1.0, r=0.8)
    text(ax, 50, 91.0, REQUEST_1L, size=T_TINY, color=INK, ha="center", va="center")
    text(ax, 50, 86.0, "one request, replayed twice", size=T_TINY, color=SHARED, ha="center",
         va="top", style="italic")

    person(ax, 5.5, 78.5, color=ARM_A, r=1.8)
    text(ax, 9.0, 79.2, ARM_A_LABEL, size=T_SMALL, color=ARM_A, weight="bold", va="center")
    person(ax, 94.5, 78.5, color=ARM_B, r=1.8)
    text(ax, 91.0, 79.2, ARM_B_LABEL, size=T_SMALL, color=ARM_B, weight="bold", va="center",
         ha="right")

    rows = [
        ("Elicit",   "asks which tone the user wants",
                     "asks which tone the user wants", "same"),
        ("Retrieve", "the same shortlist of mysteries",
                     "the same shortlist of mysteries", "same"),
        ("Rank",     "puts film X first",
                     "puts film Y first", "differs"),
        ("Explain",  "cites the genre, and a mood\nthe user never mentioned",
                     "cites the genre only", "differs"),
        ("Memory",   "keeps the invented mood",
                     "keeps only what was said", "differs"),
    ]

    y = 74.5
    for st, left, right, verdict in rows:
        nl = max(left.count("\n"), right.count("\n")) + 1
        h = 7.6 + 2.8 * (nl - 1)
        y -= h + 2.4
        bubble(ax, 3.0, y, 37.0, h, left, ec=ARM_A, tail="right", size=T_TINY, color=INK)
        bubble(ax, 60.0, y, 37.0, h, right, ec=ARM_B, tail="left", size=T_TINY, color=INK)

        box(ax, 42.0, y, 16.0, h, fc=PANEL, ec="#D6D6D6", lw=0.8, r=0.6)
        text(ax, 50, y + h * 0.63, st.upper(), size=T_TINY, weight="bold", ha="center",
             va="center", color=INK)
        text(ax, 50, y + h * 0.28, verdict, size=T_TINY, ha="center", va="center",
             color=GREY if verdict == "same" else INK,
             weight="normal" if verdict == "same" else "bold")

    text(ax, 50, 17.5,
         "Only the descriptor differs between the two columns, so every row compares the same\n"
         "decision taken under the same conditions.",
         size=T_SMALL, color=INK, ha="center")
    held_note(ax, 11.5)
    text(ax, 50, 5.5,
         "An output-only audit sees the RANK row and nothing above or below it.",
         size=T_SMALL, color=INK, ha="center")

    return save(fig, outdir, "fig_paired_c_split_screen")


# ============================================== D: fork through a shared pipeline
def build_d(outdir):
    """One request forks into two arms that traverse the same pipeline; compare per stage."""
    fig, ax = canvas(6.9, 4.0)

    text(ax, 50, 98.6, "One conversation, two protected conditions, five points of comparison",
         size=T_TITLE, weight="bold", ha="center")

    # the request and the fork
    person(ax, 8.75, 71.5, color=SHARED, r=1.9)
    box(ax, 0.5, 54.0, 16.5, 12.5, fc=TINT_S, ec=SHARED, lw=1.0, r=0.8)
    text(ax, 8.75, 60.2, REQUEST_3L, size=T_TINY, color=INK, ha="center", va="center",
         spacing=1.25)

    text(ax, 25.5, 82.0, "the only thing we change", size=T_TINY, color=INK, ha="center",
         va="bottom", weight="bold")
    box(ax, 18.0, 72.0, 15.0, 7.0, fc=TINT_A, ec=ARM_A, lw=1.2, r=0.7)
    text(ax, 25.5, 75.5, ARM_A_LABEL, size=T_TINY, color=ARM_A, weight="bold", ha="center",
         va="center")
    box(ax, 18.0, 44.0, 15.0, 7.0, fc=TINT_B, ec=ARM_B, lw=1.2, r=0.7)
    text(ax, 25.5, 47.5, ARM_B_LABEL, size=T_TINY, color=ARM_B, weight="bold", ha="center",
         va="center")
    arrow(ax, (17.2, 63.5), (17.9, 73.0), color=ARM_A, lw=1.2, rad=-0.25)
    arrow(ax, (17.2, 57.5), (17.9, 50.0), color=ARM_B, lw=1.2, rad=0.25)

    # the shared pipeline
    x0, cw, gap = 36.0, 11.4, 1.3
    PY, PH = 38.0, 42.0
    LA, LB, LH = 57.5, 44.0, 5.6
    for k, st in enumerate(STAGES):
        x = x0 + k * (cw + gap)
        xc = x + cw / 2
        box(ax, x, PY, cw, PH, fc=PANEL, ec=GREY, lw=1.0, r=0.8)
        text(ax, xc, PY + PH - 1.8, st.upper(), size=T_TINY, weight="bold", ha="center",
             va="top", color=INK)
        text(ax, xc, PY + PH - 6.4, OBSERVABLE_2L[st], size=T_TINY, color=GREY, ha="center",
             va="top", style="italic", spacing=1.2)

        output_pill(ax, x + 1.2, LA, cw - 2.4, LH, ARM_A, TINT_A)
        output_pill(ax, x + 1.2, LB, cw - 2.4, LH, ARM_B, TINT_B)
        arrow(ax, (xc, LA - 0.5), (xc, LB + LH + 0.5), color=INK, lw=1.1, style="<|-|>", ms=6)
        text(ax, xc, PY + 3.4, "compare", size=T_TINY, color=INK, ha="center", va="center",
             weight="bold")

        if k < len(STAGES) - 1:
            arrow(ax, (x + cw, LA + LH / 2), (x + cw + gap, LA + LH / 2), color=ARM_A, lw=1.0,
                  ms=4.5)
            arrow(ax, (x + cw, LB + LH / 2), (x + cw + gap, LB + LH / 2), color=ARM_B, lw=1.0,
                  ms=4.5)

    arrow(ax, (33.2, 75.5), (x0 + 1.0, LA + LH / 2), color=ARM_A, lw=1.2, rad=-0.12)
    arrow(ax, (33.2, 47.5), (x0 + 1.0, LB + LH / 2), color=ARM_B, lw=1.2, rad=0.12)
    text(ax, 25.5, 62.5, "each arm then runs\nall five stages", size=T_TINY, color=GREY,
         ha="center", va="center", style="italic", spacing=1.2)

    text(ax, 50, 26.5,
         "Both arms traverse the same pipeline, so each stage is compared on its own terms "
         "rather than folded into the final answer.",
         size=T_SMALL, color=INK, ha="center")
    held_note(ax, 20.5)
    text(ax, 50, 13.5,
         "Prior consumer-fairness audits compare only what leaves the last stage.\n"
         "The four earlier comparisons are what this paper adds.",
         size=T_SMALL, color=INK, ha="center")

    return save(fig, outdir, "fig_paired_d_fork_converge")


# ================================================ E: the same pair, across turns
def build_e(outdir):
    """Emphasises that the five stages repeat every turn and memory carries the pair forward."""
    fig, ax = canvas(6.9, 4.8)

    text(ax, 50, 98.6,
         "The five stages run again at every turn, and each arm carries its own memory forward",
         size=T_TITLE, weight="bold", ha="center")
    text(ax, 50, 94.4,
         "so the pair is compared five times per turn, and a difference can persist, fade, or "
         "grow as the conversation goes on",
         size=T_SUB, color=GREY, ha="center")

    person(ax, 27.0, 87.0, color=ARM_A, r=1.6)
    text(ax, 30.0, 87.6, ARM_A_LABEL, size=T_TINY, color=ARM_A, weight="bold", va="center")
    person(ax, 58.0, 87.0, color=ARM_B, r=1.6)
    text(ax, 61.0, 87.6, ARM_B_LABEL, size=T_TINY, color=ARM_B, weight="bold", va="center")

    x0, cw, gap = 13.0, 14.4, 2.6
    LH = 5.2
    panels = ((56.0, "turn $t$"), (19.0, "turn $t+1$"))
    for ty, tlabel in panels:
        box(ax, 10.5, ty - 4.0, 5 * cw + 4 * gap + 5.0, 26.5, fc=PANEL, ec="#DFDFDF", lw=0.8,
            r=0.8)
        text(ax, 97.0, ty + 23.7, tlabel, size=T_SMALL, color=INK, weight="bold",
             ha="right", va="bottom")
        for k, st in enumerate(STAGES):
            x = x0 + k * (cw + gap)
            xc = x + cw / 2
            text(ax, xc, ty + 20.0, st.upper(), size=T_TINY, weight="bold", ha="center",
                 va="center", color=INK)
            output_pill(ax, x, ty + 12.5, cw, LH, ARM_A, TINT_A)
            output_pill(ax, x, ty + 0.4, cw, LH, ARM_B, TINT_B)
            arrow(ax, (xc, ty + 12.1), (xc, ty + 6.0), color=INK, lw=1.1, style="<|-|>", ms=6)
            if k < len(STAGES) - 1:
                arrow(ax, (x + cw, ty + 12.5 + LH / 2), (x + cw + gap, ty + 12.5 + LH / 2),
                      color=ARM_A, lw=1.0, ms=4.5)
                arrow(ax, (x + cw, ty + 0.4 + LH / 2), (x + cw + gap, ty + 0.4 + LH / 2),
                      color=ARM_B, lw=1.0, ms=4.5)

    # memory carries forward, routed in the gap between the two turns
    xm = x0 + 4 * (cw + gap) + cw / 2
    xe = x0 + cw / 2
    yr = 47.0
    ax.plot([xm, xm], [52.0, yr], color=GREY, lw=1.1, ls=(0, (3, 2.5)), zorder=3)
    ax.plot([xm, xe], [yr, yr], color=GREY, lw=1.1, ls=(0, (3, 2.5)), zorder=3)
    arrow(ax, (xe, yr), (xe, 44.0), color=GREY, lw=1.1, ls=(0, (3, 2.5)), ms=6)
    text(ax, (xm + xe) / 2, 48.4,
         "what Memory kept is what Elicit starts from next turn — separately in each arm",
         size=T_TINY, color=GREY, ha="center", va="bottom", style="italic")

    text(ax, 50, 11.5, "five comparisons per turn, one per stage", size=T_TINY, color=INK,
         ha="center", weight="bold")
    held_note(ax, 7.2)
    text(ax, 50, 2.6,
         "The unit of analysis is the pair of trajectories over the whole conversation, not a "
         "single recommendation.",
         size=T_SMALL, color=INK, ha="center")

    return save(fig, outdir, "fig_paired_e_turn_loop")


# =============================================================== contact sheet
def contact_sheet(outdir, stems):
    import matplotlib.image as mpimg

    n = len(stems)
    fig, axes = plt.subplots(n, 1, figsize=(9.0, 5.4 * n))
    if n == 1:
        axes = [axes]
    for ax, stem in zip(axes, stems):
        ax.imshow(mpimg.imread(os.path.join(outdir, f"{stem}.png")))
        ax.set_title(stem, fontsize=9, color=GREY, loc="left")
        ax.axis("off")
    fig.tight_layout()
    out = os.path.join(outdir, "fig_paired_contact_sheet.png")
    fig.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return out


BUILDERS = {"a": build_a, "b": build_b, "c": build_c, "d": build_d, "e": build_e}
STEMS = {
    "a": "fig_paired_a_two_rails",
    "b": "fig_paired_b_mirror_ladder",
    "c": "fig_paired_c_split_screen",
    "d": "fig_paired_d_fork_converge",
    "e": "fig_paired_e_turn_loop",
}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default=os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "paper", "figures"))
    ap.add_argument("--only", default="a,b,c,d,e")
    args = ap.parse_args()

    keys = [k.strip() for k in args.only.split(",") if k.strip()]
    for key in keys:
        for path in BUILDERS[key](args.outdir):
            print(f"wrote {path}")
    print(f"wrote {contact_sheet(args.outdir, [STEMS[k] for k in keys])}")
