# ARUC Model Structural Optimization Suggestions

Saved from analysis session 2026-03-09. These are mathematically equivalent reformulations that reduce model size without changing solution quality.

## Ranked by Impact (Low Risk Only)

| # | Change | Vars Removed | Constraints Removed | Risk |
|---|--------|-------------|-------------------|------|
| 1 | Inline y variables into SOC | ~18k | ~19k | None (mathematically identical) |
| 2 | Eliminate p0_block when B=1 | ~3.7k | ~7.4k | Low (check downstream usage) |
| 3 | Pre-compute sparse Cholesky + PTDF structure | 0 | 0 | None (build time only, ~20% faster) |
| 4 | z_gen upper bound tightening | 0 | 0 | None (valid inequality) |

Items 1+2 together would remove ~40% of variables and ~45% of constraints from a typical model.

---

## 1. Inline y Variables into SOC Constraints

**Current:** For each generator SOC, creates K auxiliary variables `y_gen[i,t,k]` with K equality constraints defining `y = L^T @ Z`, then one SOC `z >= ||y||`.

**Proposed:** Use `m.addGenConstrNorm()` or pass the linear expressions directly into the quadratic constraint, eliminating the y variables and their definition constraints entirely.

**Locations:**
- `aruc_model.py:410–512` — y_gen (generators Pmin/Pmax)
- `aruc_model.py:689–763` — y_line (line flows)
- `aruc_model.py:786–842` — y_wind (wind availability)
- `aruc_model.py:863–885` — y_cost (worst-case cost)

**Savings:** ~17,856 auxiliary variables + ~17,856 equality constraints eliminated.

---

## 2. Eliminate p0_block When B=1 (Single Cost Block)

**Current:** When B=1, creates `p0_block[i,t,0]` variables identical to `p0[i,t]`, plus capacity constraints and aggregation constraints.

**Proposed:** Special-case B=1: use `p0[i,t]` directly in cost expressions, skip p0_block variables.

**Locations:**
- `aruc_model.py:146–149` — assertion
- `aruc_model.py:280–281` — variable creation
- `aruc_model.py:426–435` — block capacity constraints
- `aruc_model.py:513–525` — aggregation constraints

**Savings:** ~3,696 variables + ~1,848 constraints.

---

## 3. Pre-compute Sparse Cholesky Structure

**Current:** Inner loops iterate over all K×K entries of L (Cholesky factor), checking `abs(coef) < 1e-10` every time.

**Proposed:** Pre-compute `nonzero_L[k] = [(j, L[j,k]) for j where |L[j,k]| > 1e-10]` once per period, then iterate only nonzero entries.

**Locations:**
- `aruc_model.py:495–504` — y_gen definitions
- `aruc_model.py:737–746` — y_line definitions
- `aruc_model.py:798–809` — y_wind definitions
- `aruc_model.py:864–873` — y_cost definitions

**Savings:** ~20% faster model construction time.

---

## 4. z_gen Upper Bound Tightening (Valid Inequality)

**Current:** `z_gen[i,t]` has no explicit upper bound (only the SOC and Pmax constraints bind it).

**Proposed:** Add `z_gen[i,t] <= Pmax[i] - Pmin[i]` as a valid inequality. This tightens the LP relaxation and can improve branch-and-bound performance.

**Savings:** Better LP bounds → potentially faster solve times. No reduction in model size.

---

## Additional Items (Higher Risk / Lower Priority)

### y_line Pre-filtering
Some lines have zero PTDF coupling to all z-eligible generators. Their SOC cones are trivially `z_line >= 0`. Could skip creating y_line variables and SOC constraints for those lines.
- **Potential savings:** 5–10% of line SOC variables/constraints
- **Risk:** Low, but requires pre-computation step

### Z Connectivity Pruning
If thermal unit i has negligible PTDF coupling to all wind buses, `Z[i,t,k]` could be fixed to 0.
- **Potential savings:** 20–40% of Z variables (problem-dependent)
- **Risk:** Medium — requires network distance analysis and may affect solution quality

### Exclude Redundant ZNet Constraint
The zero-net-response constraint `sum_i Z[i,t,k] = 0` for all K dimensions has one redundant dimension (K-1 constraints suffice).
- **Potential savings:** 24 constraints (0.05%)
- **Risk:** Negligible savings; Gurobi presolve likely handles this

### Constraint Name Shortening
f-strings like `f"y_gen_def_i{i}_t{t}_k{k_idx}"` create ~35k named constraints. Shorter names reduce memory.
- **Potential savings:** 10–15% memory reduction
- **Risk:** Harder to debug
