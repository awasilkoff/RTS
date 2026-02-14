# CLAUDE.md — ruc (Paper 3)

This subdirectory implements a **new Robust Unit Commitment (RUC) formulation** for Paper 3, building on the ARUC/DARUC infrastructure from Paper 2.

## Policies (inherited from root CLAUDE.md)

**Commit Policy:** After completing a major change, automatically commit and push with a descriptive message. Do not wait for the user to ask.

**Execution Policy:** For any script that will take >1 minute to run: create the script, tell the user to run it manually, suggest the command as a single-line one-liner, and wait for pasted results. Do NOT execute long-running tasks directly.

**Formatting:** Always provide shell commands as single-line one-liners (no backslash continuations).

## Shared Infrastructure (from parent repo)

This formulation reuses:
- `models.py` — `DAMData` Pydantic model (canonical data container)
- `io_rts.py` — RTS-GMLC ETL pipeline
- `dam_model.py` — Deterministic DAM model (baseline)
- `network_ptdf.py` — DC power flow PTDF computation
- `RTS_Data/` — RTS-GMLC test case data
- `uncertainty_sets_refactored/` — Learned covariance, conformal prediction, uncertainty set generation

## Directory Structure

```
ruc/
├── CLAUDE.md          # This file
├── ruc_model.py       # New RUC formulation (Gurobi model builder)
├── run_ruc.py         # End-to-end RUC pipeline
└── ...                # Additional modules as needed
```

## Paper Outline (Variation 1: "The Diagnosis")

**Working title:** Notification-Gated Reliability Unit Commitment: Least-Distortionary Out-of-Market Interventions

**Central thesis:** Standard DARUC over-commits because it optimizes decisions that aren't its to make. The gating set — derived from physical notification times and the sequencing of decision processes — restricts reliability interventions to the non-deferrable subset. Robustness (from paper 2) ensures the gated commitments are sufficient; it is a tool, not the main contribution.

**Formulation approach:** Two-phase decomposition.
- **Phase 1:** Deterministic Gated LD-RUC (MIP) — mean forecast, incremental-variable objective over gating window, objective-only gating (non-gated variables free for MUT/feasibility).
- **Phase 2:** Robust Feasibility Verification (SOCP) — fix all binaries from Phase 1, check worst-case dispatch feasibility via LDR + ellipsoidal uncertainty set, minimize slack.
- **Augmentation:** If Phase 2 fails, CCG iteration (add worst-case scenario to Phase 1, re-solve) converges to optimal robust solution. In the case study, report Phase 2 pass/fail and robustness gap.

### Section 1: Introduction
- Reliability as a market failure → operators commit outside the market
- Current heuristics are overly conservative → over-commitment, price distortion
- This paper: identify the *minimal* set of non-deferrable commitment decisions
- Robust feasibility under worst-case uncertainty (subordinate to gating idea)
- **Contributions:**
  1. Notification-gated formulation (gating set from lead times + decision sequencing)
  2. Minimal-intervention objective (startup + no-load for gated decisions only)
  3. Two-phase decomposition: deterministic gated MIP + robust SOCP verification
  4. Generalization across horizons (MDRA, DARUC, ID-RUC as parameterizations)
  5. Numerical evaluation vs standard DARUC and integrated ARUC

### Section 2: Literature Review
- **(a) RUC in practice:** ISO/RTO operational descriptions, FERC reports, SPP/MISO/CAISO market manuals. RUC is widespread but formulations are proprietary; academic literature rarely models the sequential decision structure.
- **(b) Robust/stochastic UC:** Bertsimas, Litvinov, Jiang et al. Position: we are not proposing a new uncertainty model (paper 2), but a new *decision scope* within which robustness is enforced.
- **(c) Market design & out-of-market actions:** Uplift, price formation, convex hull pricing. Brief treatment for motivation.
- **Gap:** No existing formulation uses notification times to restrict reliability optimization to non-deferrable decisions.

### Section 3: The Reliability Commitment Problem
- **3.1 Operational Timeline:** SPP as exemplar. MDRA (advisory, 4-7 day) → DAMKT (noon OD-1) → DARUC (5pm OD-1) → ID-RUC (intraday). Timeline figure.
- **3.2 UC Formulation:** State the canonical constraint set X^UC *once*. Power balance, transmission (PTDF), block dispatch, renewable limits, capacity, ramp, logic, MUT/MDT, initial conditions.
- **3.3 DAMKT:** Full-cost objective (energy + commitment costs) subject to X^UC.
- **3.4 Standard DARUC:** Same X^UC, commitment-cost-only objective (no energy terms). Describe as a modification of 3.3.
- **3.5 Why Standard DARUC Over-Commits:** The diagnostic argument.
  - DARUC optimizes u_{i,t} for all i,t — but most will be revisited by the next market
  - May commit early to avoid expensive later startups, even when those decisions can wait
  - Result: premature commitments that displace market-cleared resources
  - Core issue: the formulation lacks a notion of *deferrability*

### Section 4: Notification-Gated Reliability Commitment (THE CONTRIBUTION)

- **4.1 Gating Sets:** T_i^gate := {t ∈ T^k | t − L_i^SU < t^next}. Only startup decisions that cannot be deferred. Note: gating applies to v_{i,t} (startups), not u_{i,t} (status) — already-committed units aren't gated. Gating window is contiguous from t=1 (if t is gated, t-1 is also gated).

- **4.2 Market-Reliability Decomposition:**
  - Total: u_{i,t} = u^m_{i,t} + u'_{i,t}, with u'_{i,t} ∈ {0,1}, u' ≤ 1 − u^m
  - v_{i,t}, w_{i,t} derived from total u via logic constraint (not decomposed separately)
  - No de-commitment: u_{i,t} ≥ u^m_{i,t} ∀i,t
  - Incremental startup identification: v'_{i,t} captures startups not in market solution

- **4.3 Phase 1 — Deterministic Gated LD-RUC (MIP):**
  - **Given:** u^m (DAMKT commitment), μ (mean renewable forecast), T_i^gate
  - **Variables:** u, v, w (binary), p (continuous), s (continuous) — total variables
  - **Objective:** min Σ_i Σ_{t ∈ T_i^gate} [C^SU v'_{i,t} + C^NL u'_{i,t}] + M Σ_t s_t
    - Penalizes only *incremental* startups and commitments in the gating window
  - **Constraints:** X^UC (full horizon) + no-decommitment (u ≥ u^m) + mean forecast
  - **Objective-only gating:** Non-gated u'_{i,t} are free (not forced to zero). Needed because MUT from a gated startup may propagate beyond the gating window. Optimizer won't add pointless non-gated commitments (no objective benefit; Pmin makes them costly).
  - **Output:** Candidate commitment schedule û = u^m + u'

- **4.4 Phase 2 — Robust Feasibility Verification (SOCP):**
  - **Given:** û, v̂, ŵ from Phase 1 (all fixed — no integers), Σ, ρ (from paper 2)
  - **Variables:** p^0 (nominal dispatch), Z (LDR coefficients), s (slack) — all continuous
  - **Dispatch policy:** p_{i,t}(r) = p^0_{i,t} + Σ_k Z_{i,t,k} r_k
  - **Objective:** min Σ_t s_t (feasibility check: 0 = robust-sufficient, >0 = gap)
  - **Robust constraints** (∀r: r^T Σ^{-1} r ≤ ρ^2): power balance, capacity, ramp, transmission, wind limits — each converted to SOC via standard robust counterpart
  - **No binary variables** → SOCP, fast to solve
  - **Output:** feasible/infeasible + robustness gap (Σ s_t)

- **4.5 Augmentation (if Phase 2 fails):**
  - **CCG iteration:** Phase 2 returns worst-case r*. Add scenario μ + r* to Phase 1 constraints. Re-solve Phase 1 → re-check Phase 2. Finite convergence to optimal robust solution.
  - **Theoretical completeness:** CCG converges to the same solution as the monolithic robust gated MIP. State in a remark or proposition.
  - **Practical reporting:** In the case study, report Phase 2 pass/fail rate and robustness gap magnitude. The gap = marginal cost of robustness beyond the gated deterministic solution.

- **4.6 Complete Formulation Summary:** Assemble Phase 1 + Phase 2 + CCG in one clean block or algorithm listing.

### Section 5: Generalization Across Decision Stages
- MDRA, DARUC, ID-RUC are all instances with different k and t^next:
  | Process | t^next | Effect |
  |---------|--------|--------|
  | MDRA | first period of OD-1 | gates long-lead resources |
  | DA-RUC | first period of OD+1 | gates medium-lead resources |
  | ID-RUC | current period + lead | gates short-lead resources |
- Standard DARUC = limiting case where t^next → ∞ (all decisions non-deferrable)
- The framework nests current practice as a special case

### Section 6: Case Study
- **6.1 Setup:** RTS-GMLC, SPP wind data, uncertainty sets from paper 2. Notification time assumptions for generator fleet.
- **6.2 Standard DARUC vs LD-RUC:** Commitment count differences, which units differ (expect: long-lead identical, short-lead differ), cost comparison, feasibility verification.
- **6.3 Comparison with Integrated ARUC:** Cost, curtailment, compute time. Key point: LD-RUC is least-interventionist, not cheapest.
- **6.4 Robust Verification Results:** Phase 2 pass/fail rate across scenarios. Robustness gap when it fails. How many CCG iterations needed (if any). Key finding: does the deterministic gated solution tend to be robust-sufficient?
- **6.5 Sensitivity to Notification Times:** Vary L_i^SU or t^next. Show gated set changes. Limiting cases: L_i→0 (nothing gated, pure market), L_i→∞ (everything gated, standard DARUC).

### Section 7: Conclusion
- Gating set: simple but consequential modeling decision
- Standard DARUC implicitly treats all decisions as non-deferrable
- LD-RUC restricts interventions to what physics requires
- Two-phase decomposition separates commitment (MIP) from robust verification (SOCP)
- Generalizes across decision horizons
- Future work: LMP/uplift analysis (Variation 3 framing), stochastic/CC extensions

## Status

Paper outline established. Drafting in progress (`Paper_Files/paper.tex`).
