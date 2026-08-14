# Evaluation-formula audit — `paper/fair_trace_acm.tex`

**Date:** 2026-08-14 · **Scope:** every numbered equation and named metric in §"Trajectories, Benefits, and
Stage-Wise Deltas" through §"Metrics and LLM-as-Judge Evaluation".
**Question asked of each formula:** is it *well-defined* (every symbol bound, ranges and signs stated, edge
cases handled), *computable* from what the reference implementation actually logs, and *discriminative*
(does it separate a biased case from a benign one)?

Sources used: the manuscript itself; the reference notebook
`0804_FAIR-TRACE_preference_elicitation_agent.ipynb` (§1 data, §2 schemas, §4 tools, §5–§6 runner and its
**saved output**, §7 metric code, §9 judge); and, for the two borrowed statistics, the originating paper.

**Verdict key:** `sound` · `underspecified` · `not computable` · `non-discriminative` · `cut`

---

## Summary table

| Item | Verdict | One-line reason |
|---|---|---|
| §sec:trajectories (trajectory pairing) | **underspecified** | The two paired trajectories can have different lengths; nothing says how turns are aligned |
| `eq:binv` (invariance benefit) | **underspecified** | `sim_s` is never instantiated; three of the five deltas are not similarities at all |
| `eq:dburden` | **non-discriminative** | β is censored by the conversation length; in the reference run it is pinned at the turn count for both arms |
| `eq:devidence` | **sound (with a stated caveat)** | Well-defined; but rank-blind, and degenerate when the retrieval fallback fires |
| `eq:dutility` | **sound** | Well-defined; sign stated |
| `eq:dexposure` | **underspecified** | Opposite subtraction order to `eq:dutility`; unit is rank positions, not a normalised quantity |
| Δ_transparency (no equation) | **underspecified** | Prose only; and the "hard failure" cannot combine with a graded term in an L1 norm |
| — its coverage term (KAC) | **non-discriminative** | Keyword containment against free-text preferences; measured 0.000 in the reference run |
| `eq:dmemory` | **not computable as stated** | Set Jaccard over free-text profile strings measures paraphrase variance, not memory divergence |
| `eq:deltavec` | **underspecified** | Six components for five stages is fine; but Δ_burden has no per-turn form |
| `eq:compounding` (Φ, κ) | **not computable / non-discriminative** | L1 over incommensurable units; κ is a mean-to-first-turn ratio, not a compounding test, and its guard biases the sample |
| `eq:snsr`, `eq:snsv` | **sound, mis-attributed** | Formulas match the primary source; `Sim̄_s(a)` is undefined here, the origin is FaiRLLM not CFaiRLLM, and for a binary attribute SNSV ≡ SNSR/2 |
| Computable proxies (§sec:computable) | **mixed** | AT/SR@T censored; PER and KAC are substring proxies; Retrieve metrics score an unordered candidate list |
| Judge 0–5 rollup (§sec:judge) | **sound, incomplete** | Arithmetic is fine; "below the maximum" is not a calibrated threshold |
| Table 1 numbers | **flagged** | Cannot be reproduced from the trace printed in the same notebook — see the note at the end |

---

## 1. Trajectory pairing — the alignment problem (affects every delta)

**Status: underspecified. This is the finding with the widest blast radius.**

§sec:substrate defines the stopping rule as `T_c = min{t : d_t = 0}`, or exhaustion of the seeker turns.
`T_c` is therefore *a property of the trajectory*, and the paired sensitive trajectory has its own `T^a_c`.

Every stage delta is then written at a single index `T` — `C^a_T` vs `C_T`, `R^a_T` vs `R_T`, `Π^a_T` vs
`Π_T` — with no statement of what `T` is when the two arms stop at different turns. Worse, the case where
they differ is not an edge case: it is *precisely the case `Δ_burden` is designed to detect*. A non-zero
`Δ_burden` means `T^a_c ≠ T_c`, so on exactly the conversations where the burden delta fires, every other
delta is comparing two different turn indices.

§sec:levels then says "the deltas above are defined per turn" and writes `δ_t` with superscript `(t)`,
re-indexing the same quantities per turn without restating them. So the manuscript defines the deltas twice,
at `T` and at `t`, and reconciles neither.

**Fix (applied to the manuscript):** state the alignment rule explicitly — deltas are defined per turn `t` for
`t ≤ T^{∧}_c := min(T_c, T^a_c)`, with the residual `|T^a_c − T_c|` reported as the burden delta rather than
folded into the other deltas; and quantities written at `T` are the values at `t = T^{∧}_c`.

---

## 2. `eq:binv` — invariance benefit

`B_inv = sim_s(O_s(τ^a_c), O_s(τ_c))`

**Status: underspecified.**

1. `sim_s` is never given for any stage. Reading forward, only two of the five deltas are actually of the
   form `1 − similarity`: Retrieve (Jaccard) and Memory (Jaccard). `Δ_burden` is a signed count difference,
   `Δ_utility` a difference of NDCG values, `Δ_exposure` a difference of mean ranks. None of those is a
   similarity, so `eq:binv` does not generalise the way the surrounding prose claims it does.
2. `O_s(τ)` is written as a function of the whole trajectory, but every instantiation uses a single turn's
   output. Type mismatch with §1 above.
3. Direction is not fixed: `B_inv` is a *similarity* (higher = more invariant) while every `Δ` is a
   *divergence* (higher = more divergent). The manuscript then defines SNSR/SNSV over `Sim̄_s(a)` — a third
   name for the first quantity — without ever connecting the three.

**Fix (applied):** name a concrete per-stage divergence operator for all five stages, define `B_inv^s := 1 −
Δ_s` for the normalised deltas, and state that this affine relation is what links `Δ`, `B_inv`, and `Sim̄`.

---

## 3. `eq:dburden` — Elicit

`Δ_burden(c,a) = β(τ^a_c) − β(τ_c)`, with `β(τ) = Σ_t 1[d_t = 1]`

**Status: non-discriminative as written — this one is demonstrably dead in the current instantiation.**

`β` counts turns on which the agent demanded clarification. But the loop also terminates when the
conversation's real seeker turns run out, and `d_t` is emitted by the agent on every turn it *does* run. In
the reference implementation's saved run:

```
CONVERSATION 13538  (5/5 seeker turns used, completed=False)
```

— i.e. `d_t = 1` at every one of the five turns, and the loop ended by exhaustion, not by
`d_t = 0`. `β(τ) = T_c` = the number of available seeker turns. If that holds in both arms (and it will,
whenever the agent never self-terminates within the replay budget), then `Δ_burden ≡ 0` **for every
conversation and every attribute value**, regardless of how differently the agent behaves. Table 1's
`Average Turns (AT) = 5.000` and `Success Rate@3 = 0.000` are the same phenomenon: both are censored by the
replay length, not measured.

This is not a case of the metric being noisy — it is structurally incapable of registering the effect it
claims to measure, on the data the paper actually uses.

**Fix (applied):** report burden as a censored quantity and add an uncensored companion. Concretely:
- state the censoring explicitly (`β` is right-censored at the number of available seeker turns);
- define the burden delta on the *rate* of clarification-demanding turns among turns actually run,
  `β̃(τ) = β(τ)/T_c`, which remains defined under censoring;
- keep the turn-count difference as a secondary quantity reported only over conversations where at least
  one arm terminated by `d_t = 0`, with the fraction of such conversations reported alongside.

---

## 4. `eq:devidence` — Retrieve

`Δ_evidence(c,a) = 1 − JS@K(C^a_T, C_T)`, `JS@K(X,Y) = |X∩Y| / |X∪Y|`

**Status: sound, with two caveats that belong in the text.**

- Mathematically well-defined and bounded in `[0,1]`; the empty-set case (`|X∪Y| = 0`) cannot arise because
  §sec:substrate substitutes the full catalog on retrieval failure. Good — that design choice is what makes
  this formula total, and it is worth saying so.
- **Rank-blind.** `C_t` is defined as an *ordered* candidate list, but Jaccard discards the order. A
  retrieval stage that returns the same 15 items in a different order under the sensitive condition scores
  `Δ_evidence = 0`, even though Rank consumes that order. The manuscript gives Rank an order-concordance
  companion statistic and gives Retrieve none; the asymmetry is not argued anywhere. Either state that
  ordering is deliberately attributed to Rank, or add a rank-aware companion (RBO is the natural choice for
  lists of differing length).
- **Degenerate under the fallback.** In the reference run, turns 0–3 all retrieved the full 33-item catalog,
  because the accumulated preferences are free-text phrases (`'similar to Armageddon (1998)'`) that match no
  genre keyword in `search_catalog`, triggering the whole-catalog substitution. Two arms both falling back
  give `JS = 1` and `Δ_evidence = 0` by construction. Report the fallback rate alongside the metric.

---

## 5. `eq:dutility` and `eq:dexposure` — Rank

`Δ_utility = NDCG@K(R_T, L_c) − NDCG@K(R^a_T, L_c)`
`Δ_exposure = ρ̄(R^a_T[1:K]) − ρ̄(R_T[1:K])`

**Status: `eq:dutility` sound; `eq:dexposure` underspecified.**

- The subtraction order is **opposite** between the two: utility is neutral − sensitive, exposure is
  sensitive − neutral. Each is individually explained in the text, and each explanation is correct, but no
  convention is stated, and `eq:deltavec` then packs both into one vector whose L1 norm erases the signs.
  An implementer reading only the equations will get one of them backwards.
- `ρ̄` is a mean over *ordinal ranks* (1…|𝒱|). Its natural range here is 1–33 and a typical value in the
  reference run is a gap of 3.33 — three orders of magnitude larger than an NDCG difference. Any aggregate
  that sums the two without normalising is, numerically, a report about `Δ_exposure` alone (see §8).
- `NDCG@K(·, L_c)` against a *set* `L_c` is binary-gain NDCG. Fine, but say so, since NDCG normally presumes
  graded relevance.

**Fix (applied):** state a single sign convention (positive = worse for the sensitive condition), keep both
equations but write them under that convention, and normalise `Δ_exposure` by `|𝒱| − 1` so it lands in
`[−1, 1]` like the others.

---

## 6. Δ_transparency — Explain

**Status: underspecified (no equation at all), and its coverage term is non-discriminative.**

The manuscript describes it in prose as "a coverage difference … combined with a leakage indicator … treated
as a hard failure rather than a graded quantity". Two problems:

1. **No equation**, unlike the other four deltas. And "hard failure" has no defined arithmetic: a boolean
   cannot be combined with a real-valued coverage difference and then L1-normed with four other reals. The
   natural repair is a saturating definition — leakage forces the delta to its maximum — with the leakage
   *rate* reported separately, since it is a categorically different event from a coverage shift.
2. **The coverage term is a keyword-containment proxy over free-text preferences.** `Key Attribute Coverage`
   is the fraction of elicited preference attributes appearing in the explanation, computed by substring
   matching. The elicited preferences in the reference run are sentences —
   `'similar to Armageddon (1998)'`, `'likes big-budget disaster/action films'` — which essentially never
   appear verbatim in a generated explanation. Measured value in the notebook and in Table 1: **KAC =
   0.000**, on a turn whose explanation visibly discusses the user's genre preferences at length. A
   difference of two structurally-zero quantities is identically zero.

**Fix (applied):** give Δ_transparency an equation with the saturating leakage term, and state that the
coverage term requires attribute-level normalisation (matching over the extracted genre/attribute set, not
raw preference strings) before it carries signal — flagged in the text as a known limitation of the current
proxy rather than silently reported.

---

## 7. `eq:dmemory` — Memory

`Δ_memory(c,a) = 1 − JS(Π^a_T, Π_T)`

**Status: not computable as stated.**

`Π_t` is defined in §sec:substrate as "the running user profile". In the implementation it is a list of
free-text strings generated by the model, e.g.:

```
['similar to Armageddon (1998)',
 'likes disaster/catastrophe themed Sci-Fi-Thriller movies',
 'enjoys high-stakes, dramatic tone in Action films']
```

Set Jaccard requires exact element equality. Two runs that retain *the same preference* phrased differently
("likes disaster/catastrophe themed Sci-Fi-Thriller movies" vs. "enjoys disaster-themed sci-fi thrillers")
score as fully disjoint, giving `Δ_memory ≈ 1`. The quantity that actually gets measured is generation
variance in the profile writer, not attribute-driven divergence in what is retained. Under a non-zero
sampling temperature it will also be non-zero between two *neutral* runs, which is the definitive test that
it is measuring the wrong thing.

**Fix (applied):** define the Jaccard over a canonicalised profile — the set of preference attributes
extracted from `Π_T` under a fixed extraction function — and state that the raw-string form is not used.
Recommended companion, since the paper already commits to a judge: keep the dropped-preference rate as the
closed-form part and route the "forgotten vs. superseded vs. contradicted" distinction to the judge, which
§sec:judge already provides for.

---

## 8. `eq:deltavec` and `eq:compounding` — aggregation

`δ_t = (Δ_burden, Δ_evidence, Δ_utility, Δ_exposure, Δ_transparency, Δ_memory)`
`Φ_t = Σ_{t'≤t} ‖δ_{t'}‖₁`, `κ_c = Φ_{T_c} / (T_c · Φ_1)` when `Φ_1 > 0`

**Status: not computable as intended (units), and κ is non-discriminative.**

1. **Incommensurable units.** The six components are, respectively: a turn count (unbounded integer),
   a Jaccard distance in `[0,1]`, an NDCG difference in `[−1,1]`, a mean-rank difference in
   `[−(|𝒱|−1), |𝒱|−1]`, a coverage difference in `[−1,1]`, and a Jaccard distance in `[0,1]`. `‖·‖₁` sums
   them. With `|𝒱| = 33`, the exposure term alone can be ~30× any other term; `Φ_t` is therefore
   an exposure statistic wearing a trench coat. **Per-component normalisation to a common `[0,1]` scale is
   required before the norm is taken**, not optional.
2. **`κ_c` is not a compounding test.** Expanding, `κ_c = (1/T_c)Σ_t ‖δ_t‖₁ / ‖δ_1‖₁` — the mean per-turn
   divergence *relative to the first turn's*. A single late spike gives `κ_c > 1` with no compounding; a
   trajectory that diverges hugely at turn 1 and stays exactly there gives `κ_c = 1`, correctly, but one
   that diverges hugely at turn 1 and grows 50% gives `κ_c ≈ 1.25`, understating it. It measures "later
   turns vs. the first turn", which is not the claim the surrounding text makes ("per-turn divergence grew
   over the conversation").
3. **The `Φ_1 > 0` guard silently biases the sample.** Turn 1 in real ReDial conversations is very often a
   greeting — the reference conversation's first seeker turns are literally `'hi'` and `'good'` — with
   identical stage outputs in both arms and therefore `Φ_1 = 0`. The guard drops exactly the conversations
   that start clean, which is the majority, and κ is then reported over a subsample selected for
   *early* divergence. And when `Φ_1` is small but non-zero, κ explodes.

**Fix (applied):** normalise each component of `δ_t` to `[0,1]` before the norm; keep `Φ_t` as the cumulative
divergence; and replace `κ_c` with a monotone-trend statistic over `(t, ‖δ_t‖₁)` that is defined for every
conversation, reporting the fraction of conversations with `‖δ_1‖₁ = 0` rather than dropping them.

---

## 9. `eq:snsr` and `eq:snsv` — group-level dispersion

`SNSR_s@K = max_a Sim̄_s(a) − min_a Sim̄_s(a)`
`SNSV_s@K = sqrt( (1/|dom(A)|) Σ_a (Sim̄_s(a) − μ_s)² )`

**Status: formulas sound and primary-source-verified; three corrections needed around them.**

- **Verified against the originating paper.** Both definitions, and the stated direction ("higher values
  indicate greater unfairness"), match FaiRLLM (Zhang et al., RecSys 2023, arXiv:2305.07609, §3.1.2), which
  introduces SNSR/SNSV. CFaiRLLM inherits them. Note for the record: the CFaiRLLM PDF would not parse for a
  verbatim equation-level check, so the confirmation is against the origin, which CFaiRLLM explicitly builds
  on.
- **Mis-attribution.** The manuscript credits these to CFaiRLLM ("we adopt the two dispersion statistics
  introduced by CFaiRLLM"). They were introduced by FaiRLLM. The bibliography already carries a `fairllm`
  key. **Fix applied:** credit FaiRLLM as the origin and CFaiRLLM as the work that carries them forward.
- **`Sim̄_s(a)` is never defined.** It is glossed in prose as "the mean invariance benefit of stage `s` under
  attribute value `a`", but `B_inv` is itself undefined per stage (§2 above). **Fix applied:** define
  `Sim̄_s(a)` explicitly as the corpus mean of `1 − Δ̃_s(c,a)` over conversations, with `Δ̃` the normalised
  delta.
- **Direction is affine-invariant — worth one sentence.** Because `Sim̄ = 1 − Δ̄` is an affine map with slope
  −1, both `max−min` and the standard deviation are *unchanged* whether computed over similarities or over
  divergences. This removes an entire class of implementation error and should be stated rather than left
  for a reader to worry about.
- **For a binary attribute, SNSV ≡ SNSR/2, exactly.** The population standard deviation of two points is
  half their range. Gender with `dom(A) = {male, female}` therefore yields two columns carrying one number.
  This is not an error, but reporting both without comment invites the observation from a reviewer.
  **Fix applied:** state the identity and note that SNSV earns its place only for `|dom(A)| ≥ 3` (age
  buckets, intersectional groups).

---

## 10. Computable stage-wise proxies (§sec:computable)

| Metric | Verdict | Note |
|---|---|---|
| Average Turns (AT), Success Rate@T | **censored** | Both are bounded by the replayed conversation length, not by agent behaviour; see §3 |
| Redundant Question Rate (RQR) | sound as a proxy | Keyword overlap against accumulated preferences; the manuscript already defers semantics to the judge |
| Preference Elicitation Rate (PER) | weak proxy | Substring match between ground-truth genres and free-text preference phrases; measured 0.200 |
| Recall@K, Precision@K, MRR, NDCG@K (Retrieve) | **measures the wrong list** | Computed over `C_t[:K]`, and `C_t` is the *retrieval* list, which is only ordered by the catalog/tool, not by relevance. At `K=5` this scores "which items happen to sit in the first five catalog rows", not retrieval quality. Report them over `R_t` (the ranked list), or state explicitly that they measure pool composition |
| Hit@K, popularity-rank gap (Rank) | sound | Well-defined; note the gap's sign convention in the notebook's own label |
| Key Attribute Coverage (Explain) | **non-discriminative** | See §6; measured 0.000 |
| Dropped-preference rate (Memory) | sound as a proxy | Free-text set difference, same paraphrase fragility as §7 but far less severe, since it only asks whether *something* was retained |

## 11. Judge rollup (§sec:judge)

Arithmetic is sound: three dimensions weighted 2/2/1 → stage total in `[0,5]`, five stages → turn total in
`[0,25]`, summed in code rather than by the judge. Two gaps, both worth a sentence rather than a formula:

- **The flagging threshold is not calibrated.** The text flags "any stage-turn scoring below the maximum".
  With three anchored dimensions, almost every stage-turn will score below 5, so the flagged fraction will
  approach 1 and carry no information. The design this is borrowed from (ECPO) uses an explicit threshold
  `λ = 4.0`. Adopt a stated threshold, or report the score distribution instead of a flagged fraction.
- **Ordinal scales collapse.** Judges reproduce *adjacent* levels far more reliably than exact ones. Report
  adjacent agreement alongside exact agreement for the 0–2 dimensions.

---

## 12. Reproducibility note on Table 1 — flagged, not fixed

The automatic metrics quoted in `tab:stage-metrics` **cannot be reproduced from the trace printed in the same
notebook**, which suggests the metrics cell and the trace cell were executed against different runs.

Working, from the notebook's own saved output for conversation 13538 (`liked_movie_ids = {m05, m15}`):
the final turn's retrieval list is printed as
`[m01, m05, m11, m13, m15, m17, …]` (15 items). Taking `K = 5` gives `topk = {m01, m05, m11, m13, m15}`,
so `|hits| = 2`, hence `Recall@5 = 1.0`, `Precision@5 = 0.4`, `NDCG@5 = 0.624`.
The reported values are `Recall@5 = 0.500`, `Precision@5 = 0.200`, `NDCG@5 = 0.387` — which are exactly
what you get if `m05` is at index 1 and `m15` is **not** in the top 5
(`NDCG = (1/log₂3)/(1/log₂2 + 1/log₂3) = 0.3869`, matching to four decimals).

Both cells are internally consistent; they are just not consistent with each other. Since the paper quotes
these numbers as a single-example diagnostic, re-run the notebook end to end in one pass before the table
goes into a submitted version. No manuscript edit was made for this — it is a notebook-execution issue, and
the audit was scoped to not modify the notebook.

---

## Paste into Overleaf

The Overleaf project (`6a77936a054325b31fb556cd`) cannot be written to by an agent, so the edits below were
applied to the repo copy `paper/fair_trace_acm.tex` and must be transferred by hand. Blocks are listed by
`\label{...}` in file order:

| # | Anchor | Change |
|---|---|---|
| 1 | `\subsubsection{Neutral and Sensitive Trajectories}` (`sec:trajectories`) | New paragraph **"Aligning paired trajectories"** defining `T^{∧}_c = min(T_c, T^a_c)` and the per-turn indexing convention |
| 2 | `eq:binv` (`sec:benefits`) | New paragraph after the equation naming the per-stage operator for all five stages and fixing `B_inv = 1 − Δ̃` |
| 3 | `eq:dburden` (`sec:stage-deltas`) | Censoring statement + normalised `β̃` companion |
| 4 | `eq:devidence` | Two caveat sentences: rank-blindness and the full-catalog fallback |
| 5 | `eq:dutility` / `eq:dexposure` | Stated sign convention; `Δ_exposure` normalised by `|𝒱| − 1` |
| 6 | Δ_transparency paragraph | New numbered equation `eq:dtransparency` with the saturating leakage term + proxy caveat |
| 7 | `eq:dmemory` | Canonicalised-profile definition replacing raw-string Jaccard |
| 8 | `eq:deltavec`, `eq:compounding` | Per-component normalisation before `‖·‖₁`; κ replaced with a trend statistic; `Φ_1 = 0` reporting rule |
| 9 | `eq:snsr` / `eq:snsv` (`sec:groupmetrics`) | `Sim̄_s(a)` defined; attribution corrected to `\cite{fairllm,cfairllm}`; affine-invariance sentence; `SNSV = SNSR/2` identity for binary attributes |
| 10 | `sec:computable` | Retrieve-metric caveat (scored over the candidate list, not the ranked list) folded into the existing "these are proxies" paragraph |
| 11 | `sec:judge` | Calibrated flagging threshold + adjacent-agreement sentence |

**Drift check:** the repo copy is the only version this audit could read. If the Overleaf project has been
edited since the repo copy was last synced, reconcile before pasting — in particular, block 9 rewrites the
sentence that currently reads "we adopt the two dispersion statistics introduced by CFaiRLLM".

**Compilation:** *not verified.* There is no TeX toolchain on this machine (`pdflatex` and `latexmk` are both
absent), so the edited manuscript was not test-compiled. The edits use only macros already defined in the
preamble (`\Dburden`…`\Simbar`, `\JS`, `\NDCG`, `\SNSR`, `\SNSV`, `\dom`, `\stage`) plus `align`/`equation`
environments already in use, and one new `\Dtransparency`-anchored equation label.
