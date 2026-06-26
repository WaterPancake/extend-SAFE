# When Does Multitask VLA Failure Detection Actually Work?

### An evaluation audit of SAFE on OpenVLA / LIBERO

*Working results page. Every number below was recomputed from checkpoints and
cached features in this repository; see §7 for the exact scripts and artifacts.*

---

## Abstract

SAFE (TRI, NeurIPS 2025) detects task failures of vision-language-action (VLA)
models by training a lightweight probe on the model's frozen hidden states, and
reports that the detector **generalizes across tasks**. We reproduce SAFE on
OpenVLA / LIBERO and audit that claim under a deployment-relevant protocol. We
find that (i) the reported pooled AUC **overstates** honest per-task performance
by **+0.09–0.11**; (ii) in this benchmark **"failure" is operationally
"timeout"** (a length-only predictor scores AUC = 1.0); (iii) a genuine early
failure signal exists *in-distribution* but **does not transfer** to unseen
tasks; and (iv) under leave-one-task-out (LOTO), there is **no operating point
with both a calibrated false-positive rate and useful lead time** — at an FPR
that actually holds (~1%), the detector catches 5–14% of failures and fires at
78–98% of the rollout, i.e. essentially at the timeout it is meant to pre-empt.
We additionally show that fusing representations across network depth yields **no
complementary cross-task failure signal** (ΔAUC within ±0.03). We report two
preliminary claims of our own that **deflated under hardening**, and we propose a
corrected evaluation protocol.

---

## 1. Background

A VLA failure detector should answer, *early and on tasks it was not trained on,*
"is this rollout going to fail?" SAFE trains a probe (linear, an MLP with
cumulative-sum scoring, or an LSTM) on the last-token hidden state of a frozen
OpenVLA policy and scores a rollout by the running maximum of the probe output.
Its headline evidence is a **pooled** ROC-AUC over rollouts from held-out tasks.
Our question is narrow and practical: *does that number mean what a practitioner
would assume it means?*

## 2. Setup and data

- **Policy / benchmark:** OpenVLA rolled out on LIBERO (10 tasks). We probe the
  frozen last-token hidden state at 9 transformer layers {1, 4, 8, 12, 16, 20,
  24, 28, 32}.
- **Data:** 1,000 rollouts (an initial 500 + a 500-rollout extension). Each
  rollout stores per-step hidden states and a binary success/failure label.
- **The label structure (central to everything below):** every *failure* rollout
  runs to exactly 520 steps — the episode time limit — while successes terminate
  early. "Failure," as labeled, means *"did not succeed before the clock ran
  out."*
- **Probes & scoring:** SAFE-MLP (cumsum, 2-layer), SAFE-LSTM (1-layer),
  linear. `falert_early` = max probe score up to `task_min_step` (SAFE's
  earliest-stop cap), the **honest** metric; `falert_end` = uncapped.
- **Hyperparameters:** the paper's *selection-table* values (not the repo's
  pre-sweep defaults): both lr = 1e-4, token = last; **MLP λ = 1e-2, LSTM λ = 1**;
  1000 epochs, effective batch 512.

## 3. Methodology

The audit rests on one move: **separate genuine, early, transferable failure
signal from three confounds** — task identity, rollout length / elapsed time, and
survival bias. Each tool isolates one.

1. **Pooled vs. macro per-task AUC.** Pooling lets a probe partly answer *which
   task is this* (via score scale/length) instead of *will it fail*. We report
   the macro average of within-task AUCs alongside the pooled number.
2. **Length-only / task-identity baselines.** Because failure ≡ timeout, rollout
   length alone is a baseline; its AUC bounds how much of a probe's score is the
   clock.
3. **`falert_early` (the `task_min_step` cap).** Capping every rollout at its
   task's shortest length removes the survival advantage — all compared rollouts
   are still alive.
4. **Fixed-absolute-timestep decoupling.** Compare instantaneous scores at the
   *same* absolute timestep; with elapsed time held constant, any AUC > 0.5 is
   genuine *state* signal, not a clock.
5. **Leave-One-Task-Out (LOTO).** 10 folds, each task held out once. The rigorous
   test of cross-task generalization; we hardened every headline number under it.
6. **Conformal detection-time.** Calibrate an alarm threshold on seen-task
   validation at a target FPR, then measure **realized FPR, catch rate, and lead
   time** on held-out tasks — does SAFE's calibrated guarantee transfer?
7. **Multi-layer complementarity gate.** Linear CKA between layers; an
   incremental-value test (does adding layer-*k* features, via PCA or supervised
   PLS, beat the best single layer on held-out tasks?); and a UMAP / cross-task
   kNN nonlinear check.

All early-window and multi-layer tests pool features over `[0, task_min_step]`
so survival bias cannot contribute, and evaluate on held-out tasks.

## 4. Results

### 4.1 Pooled AUC reproduces SAFE — and inflates the honest metric by ~+0.1

We reproduce SAFE's reported unseen-task pooled AUC, then show it overstates the
macro per-task AUC on the *same probes* (layer 32, seeds 0/1/2, n = 1000;
pooled values match the stored metric to |Δ| < 1e-4).

| Probe | Pooled AUC | Macro per-task AUC | **Inflation** | SAFE paper (pooled, unseen) |
|---|---|---|---|---|
| SAFE-MLP  | 0.747 ± 0.052 | 0.660 ± 0.030 | **+0.087** | 0.735 |
| SAFE-LSTM | 0.755 ± 0.080 | 0.646 ± 0.043 | **+0.109** | 0.725 |

The gap is task-identity / scale / length structure that pooling rewards. Our
pooled numbers slightly exceed the paper's, so this is not a reproduction
failure — it is the *same* result, read more honestly.

### 4.2 "Failure" is "timeout"

Every failure rollout is exactly 520 steps; a **length-only predictor scores
AUC = 1.0** on every fold (and the length-leakage flag is set on every run).
Uncapped `falert_end` AUC sits at 0.96–0.99 — almost all of it survival/elapsed
time. This is why `falert_early` is the only defensible headline. (Fig. 1:
per-task success-length distributions; failures are absent because they all sit
at the 520-step cap.)

### 4.3 The honest ceiling is ~0.66, and it is wildly task-dependent

Under LOTO (10 folds, layer 32), the macro per-task `falert_early` AUC — where
each fold's test set is a *single* held-out task, so the fold mean **is** the
macro per-task AUC:

| Probe | LOTO macro AUC | Per-task range |
|---|---|---|
| SAFE-MLP  | **0.672 ± 0.113** | 0.41 (task 5) → 0.82 (tasks 0, 7) |
| SAFE-LSTM | **0.652 ± 0.123** | 0.48 (task 5) → 0.87 (task 0) |

Task 5 is *anti-predictive* (AUC < 0.5) for both families; tasks 0 and 7 are
easy. A single pooled number hides this entirely.

### 4.4 The early signal is real in-distribution but does not transfer

Fixed-absolute-timestep test: *in-distribution*, deep-layer MLP macro AUC is
~0.60–0.62 as early as step 5–20 and rises to 0.72–0.78 by step ~120 — genuine
failure-relevant *state* exists early. On **held-out tasks** it **collapses to
chance (0.50–0.58) until ~25% into the rollout**, reaching only ~0.59–0.63 even
late. The early signal is task-specific. The bottleneck is *cross-task
generalization of early signal*, not its existence.

### 4.5 No actionable early warning (the strongest surviving result)

Conformal early-warning, hardened under LOTO (layer 32):

| Probe | Target FPR | Realized FPR | Catch rate | Alarm fires at (% of rollout) |
|---|---|---|---|---|
| MLP  | 1%  | **0.0%** | **14%** | **98%** |
| MLP  | 5%  | 8.2% | 69% | 86% |
| MLP  | 10% | 17.7% | 90% | 74% |
| LSTM | 1%  | **0.6%** | **5%** | **78%** |
| LSTM | 5%  | 8.7% | 44% | 60% |
| LSTM | 10% | 17.5% | 69% | 51% |

At an FPR that actually holds (~1%), the detector catches 5–14% of failures and
fires at 78–98% of the rollout — coincident with the timeout it is supposed to
pre-empt. Catching a majority of failures requires accepting 8–18% realized FPR,
and even then alarms fire past the halfway mark. **There is no operating point
that is simultaneously well-calibrated and timely.**

### 4.6 Fusing layers does not help (closed, negative)

Representations decorrelate across depth (linear CKA: shallow {1,4} vs. deep
{12–32} ≈ 0.45 cross-block, while the deep block is 0.85–0.99 internally
redundant) — which *looked* like headroom. It is not: in the incremental-value
test (base = layer-32 score, AUC 0.672; add layer-*k* on held-out tasks), every
layer lands within **±0.03 ΔAUC**, and the *decorrelated shallow layers give
zero-to-negative ΔAUC even under a supervised PLS probe that explicitly hunts the
missed failure direction.

| Method | Shallow layers (1/4/8) ΔAUC | Best deep layer ΔAUC | Verdict |
|---|---|---|---|
| PCA-100 (unsupervised) | −0.015 / −0.014 / −0.030 | +0.028 (L20) | within noise |
| PLS-5 onto L32 residual (supervised) | −0.001 / +0.011 / −0.003 | +0.024 (L20) | within noise |

A nonlinear check agrees: UMAP of pooled features organizes cleanly **by task**,
not failure (Fig. 2), and cross-task kNN label-AUC is flat ~0.63–0.67 across all
layers — the same ~0.66 ceiling. **The non-redundant content is not
failure-relevant; the failure signal lives in the redundant deep block and is
saturated.**

## 5. Two of our own claims that deflated under hardening

In the spirit of the audit, we report where our *preliminary* results were too
strong:

| Claim | Preliminary (3-seed pilot) | Hardened (LOTO + correct hyperparameters) |
|---|---|---|
| Conformal calibration "fails to transfer" | target 5% → **25.5%** realized (~5×) | target 5% → **8.2%** realized (**~1.6×**) |
| Seed-2 length-shift "collapse to chance" | macro AUC **0.514** | macro AUC **0.639** (recovered) |

The conformal overshoot is real but **modest** (~1.6×, not catastrophic); the
"length-fitting collapse" was an undertraining/data artifact (200 epochs, wrong
MLP λ, half the data) that recovers with correct training (across-seed macro
range shrinks 0.26 → 0.07). **Do not headline these.** The robust survivors are
§4.1–4.6.

## 6. Discussion: a corrected protocol

The method-paper gate — *early failure signal that generalizes to unseen tasks
under a calibration that holds* — fails here: the signal is genuine but
task-specific, calibration overshoots modestly on unseen tasks, and no operating
point is both honest and timely. The defensible contribution is therefore an
**evaluation audit and a corrected protocol**. When reporting multitask VLA
failure detection, report:

1. **Macro per-task AUC** alongside pooled (the gap is the task-identity tax);
2. **Length-only and task-identity baselines** (here, length alone = 1.0);
3. A **fixed-absolute-timestep** early metric (clock-decoupled);
4. **Realized — not target — FPR and lead time on held-out tasks.**

A method revival needs *new data* that changes the gate: rollouts that **save
per-step actions** and **terminate on / label genuine failure onset** (to break
the failure ≡ timeout confound), plus off-policy / unsafe trajectories — the
prerequisites for the latent-reachability direction below.

## 7. Reproducibility

Numbers in this page were recomputed in-session from repository artifacts:

| Result | Script | Artifact |
|---|---|---|
| Pooled vs. macro (§4.1) | `scripts/score_layer_pooled_macro.py` | `runs/openvla_{mlp,lstm}_seed012_l32_opt/` |
| LOTO macro (§4.3) | (10-fold mean of `test_falert_early`) | `runs/openvla_{mlp,lstm}_loto_l32/ablation_results.csv` |
| failure ≡ timeout (§4.2) | — (`length_only_roc_auc`) | same ablation CSVs (= 1.0 every row) |
| Detection-time (§4.5) | `scripts/conformal_detection_time.py` | `runs/openvla_layer_correlations/loto_detection_time_{mlp,lstm}_l32.csv` |
| Walk-backs (§5) | same | `.../detection_time_mlp_l32.csv` (pilot) |
| Multi-layer (§4.6) | `scripts/analyze_layer_complementarity{,_supervised}.py`, `scripts/umap_layer_failure.py` | `runs/openvla_layer_correlations/complementarity_cache.npz` |
| Disentanglement (§4.4) | `scripts/analyze_proxy_disentangle.py` | `runs/openvla_layer_correlations/` |

**Figures:** `docs/success_length_violin.png` (Fig. 1), `docs/umap_by_task.png`
(Fig. 2), `docs/umap_by_label.png`.

---

## Appendix: ongoing direction — latent-space safety certificates

Motivated by §4.4–4.5 (static probes carry no transferable early warning), we are
prototyping a **latent reach-avoid** view of failure: rather than classify each
frame, learn a value function over the VLA's latent that estimates *whether
failure is still avoidable*. Stage 1 (an offline reachability certificate via a
discounted safety-Bellman backup) is buildable on the existing latents; Stage 2
(an action-conditioned latent safety filter) requires the new data collection
noted in §6. This is design-stage work; no results are claimed yet.
