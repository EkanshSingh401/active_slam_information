# Formulation → Code

Every equation in formulation draft v0.1, where it lives, and how it's verified.

## The one derivation you need to understand

The draft scores a viewpoint by the drop in entropy of the scored marginal:

$$\Delta I = \log\det\Sigma_S - \log\det\Sigma_S^+$$

Computed naively that means a full Kalman update on a ~250-dim singular
matrix per candidate. The code never does that. Instead:

$$
\Sigma_S^+ = \Sigma_S - B A^{-1} B^\top,\qquad
A = H\Sigma H^\top + V,\qquad B = T\Sigma H^\top
$$

Apply Sylvester's determinant theorem, $\det(I + XY) = \det(I + YX)$:

$$
\boxed{\;\Delta I = \log\det A - \log\det D,\qquad
D = H\,\Sigma_{\cdot|S}\,H^\top + V\;}
$$

$$
\Sigma_{\cdot|S} = \Sigma - \Sigma T^\top \Sigma_S^{-1} T \Sigma
$$

Three consequences, all of which the code relies on:

1. **$A$ and $D$ are $m\times m$** where $m = 2\times$(visible landmarks) —
   small, and independent of state size.
2. **Both are strictly PD because $V$ is**, so the gain is well defined even
   though the full $\Sigma$ has a rank deficit of 6. The only state-size
   matrix ever factored is $\Sigma_S$.
3. **$\Sigma_{\cdot|S}$ doesn't depend on the candidate**, so it's computed
   once per planning cycle. $D$ is the innovation covariance *given* the
   scored state — which is why its log-det is what's left after scoring.

## Equation → code

| Formulation | Code | Notes |
|---|---|---|
| $\mathbf{M}(\mathbf{x}) = \sum_j H_j^\top V_j^{-1} H_j$ (§3) | never formed | Information form is avoided entirely; the gain is computed in measurement space via $A, D$ |
| $H_j = \partial h_j/\partial \mathbf{x}$ (§3) | `predict_measurement()` | block sparse: IMU pose (2×6) + one block per landmark representation column |
| candidate pose error (§1, common prior) | `H_imu.leftCols<3>() = H_th_c * R_delta` | $\delta\theta_c = R_\Delta\,\delta\theta_{now}$, $\delta p_c = \delta p_{now}$ |
| landmark chain rule (§7) | `H_pf * r.J` over `lm.rep` | native anchored-inverse-depth coords; no transform in the planner |
| $V_j$, appearance noise (§3, §6 of draft) | `NoiseModel::sigma_px()` | v1 = `ConstantNoiseModel`, must equal `up_msckf_sigma_px` |
| $\Sigma_S = T\Sigma T^\top$ (§4) | `InformationPrior` ctor | fails loudly (`ok()==false`) if $T$ scores a degenerate direction |
| selection $\mathbf{S}$ (§4) | `T_selection()` in tests | special case of $T$ |
| metric map (XYZ) | `T_metric()` in tests | landmark rows = representation Jacobian |
| $\Sigma_{\cdot\mid S}$ | `Sigma_cond_ = Sigma - WᵀW`, $W = L^{-1}T\Sigma$ | once per cycle |
| $\Delta I$ (§5) | `evaluate()` → `logdet(A) - logdet(D)` | two $m\times m$ Cholesky |
| $\log\det\Sigma_S^+$ | `posterior_logdet` | **rank by this**, not by `delta` (§5, Table I) |
| square-root log-det (§5) | `logdet_spd()` = $2\sum\log L_{ii}$ | |
| virtual frontier landmarks (§6) | `augment_with_virtual_landmarks()`, `augment_scoring_map()` | prior $\sigma_l^2 I_3$, uncorrelated; rep = identity |

## A result the draft should absorb

**$\Delta I$ is invariant to $T \to KT$ for any invertible $K$** (verified to
9 significant figures after a random rescaling spanning 4 orders of
magnitude). So the score doesn't care about units — radians vs. metres vs.
inverse depth mixing in one matrix is not a problem, unlike trace (A-optimality),
which is unit-dependent. This is a second, independent argument for
D-optimality alongside the entropy argument in §5.

The flip side: the invariance is only to reparameterizing *within* the scored
quantity. Choosing **what** to score is a real modeling decision. Anchored
inverse depth → global XYZ is **not** such a reparameterization, because the
XYZ position also depends on the anchor clone, which is outside a pure
selection. Scoring "the landmark parameters" and scoring "the metric map" give
different numbers. The draft should commit to the metric map (`T_metric`),
since that's what "map uncertainty" means.

## Verification

| Test | Result | What it proves |
|---|---|---|
| structural degeneracy | deficit = 6, IMU+newest-clone map rejected | synthetic state reproduces the diagnostic |
| fast vs. reference | worst rel. err **5.9e-15** over 40 cases | the Sylvester derivation, on singular $\Sigma$ |
| unit invariance | identical to 9 s.f. | D-optimality is coordinate-free |
| monotonicity | $\Delta I \ge 0$, ↑ with #obs, ↓ with $\sigma$ | sanity |
| frontier pull | ↑ with $\sigma_l$ | exploration term behaves |
| Jacobians vs. FD | worst rel. err **6.4e-11**, 50 poses, non-identity extrinsics | **internal** consistency with the stated convention |
| timing | 0.64 ms/candidate, n=250, 54 visible (x86) | fits budget; re-measure on Orin NX |

**Not yet verified: that the stated convention matches OpenVINS.** The FD test
checks the code against the convention written in `view_geometry.hpp`. Whether
that convention is OpenVINS's is checked only by the cross-check in the
integration step. Until that passes, no score from this code is trustworthy.

## v1 approximations, stated

- **Common prior.** No process noise along the motion to the candidate. Every
  candidate is scored against the same prior, so gain-ranking equals
  entropy-ranking. Adding propagation breaks that equivalence — rank by
  `posterior_logdet`.
- **Greedy, horizon 1.** One viewpoint, not a trajectory. Non-myopic
  planning (reduced value iteration, Atanasov et al.) is future work.
- **Time offset** not in $H$ (zero Jacobian for a future measurement).
- **Normalized-coordinate noise** $\sigma_{px}/\bar f$. Equivalent to
  OpenVINS's pixel-space noise only up to distortion, which is near zero
  on the D455.

## Second scoring mode: pose-marginal information + coverage (day 2)

$$\text{score}(c) = \Delta I_{\text{pose}}(c) + \lambda\, V_{\text{new}}(c)$$

**Pose term.** Scoring map $T_{\text{pose}} = [\,0 \;\cdots\; I_6 \;\cdots\; 0\,]$ selecting the
6-dof IMU pose (orientation error, position) — every landmark, real and virtual, is outside
$T$ and therefore **marginalized out**: the scored covariance is
$\Sigma_S = T_{\text{pose}}\Sigma T_{\text{pose}}^\top$, the pose marginal of the joint. The
Sylvester identity above holds for any full-row-rank $T$, so the fast path is unchanged:
$\log\det\Sigma_S^+ = \log\det\Sigma_S - (\log\det A - \log\det D)$ with $A, D$ built from the
same $H$ (real and virtual landmark blocks included in $H$; they still carry information *to*
the pose through their correlation and the shared measurement).

With travel the prior differs per candidate (OpenVINS propagation over the travel time,
clone at the candidate). The pose term is measured against the **current** pose covariance:

$$\Delta I_{\text{pose}}(c) = \log\det\Sigma_{\text{pose}}(\text{now}) - \log\det\Sigma^{+}_{\text{pose}}(c)$$
$$= \underbrace{\log\det\Sigma_{\text{pose}}(\text{now}) - \log\det\Sigma_{\text{pose}}^{\text{prop}}(c)}_{\le 0:\ \text{cost of getting there}}
   + \underbrace{\log\det\Sigma_{\text{pose}}^{\text{prop}}(c) - \log\det\Sigma^{+}_{\text{pose}}(c)}_{\ge 0:\ \text{measurement gain}}$$

so a far candidate must earn back the uncertainty its travel adds. Code:
`exploration_planner.cpp` (`planner_type: pose_cov`), `Tpose` over the propagated,
virtual-augmented covariance; `posterior_logdet` from `evaluate()`; `log det Σ_pose(now)` by
Cholesky of the current IMU pose block.

**Coverage term.** $V_{\text{new}}(c)$ = unknown volume in the candidate's camera frustum,
ray-cast through the mapper's coarse known-space map (`active_slam_sim/coverage_gain.hpp`):
each ray $(x,y,1)$ on the image plane subtends
$d\Omega = dx\,dy/(1+x^2+y^2)^{3/2}$; unknown steps $[r_1,r_2]$ before the first occupied cell
add $d\Omega\,(r_2^3-r_1^3)/3$. Units m³; $\lambda$ in nats/m³.

**Verification.**
| test | result |
|---|---|
| pose-marginal fast vs brute-force Kalman update, T = pose (6), real + virtual landmarks, 20 states | worst rel. err 2.2e-14 |
| coverage, empty map vs $\Omega R^3/3$ | 0.01% |
| coverage, wall plane at 3 m vs pyramid volume | 0.04% |
| coverage, all known free | 0 m³ |
