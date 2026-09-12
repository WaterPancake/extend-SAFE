# When Pooled AUC Is Not Early Warning

2026-08-03

- [Abstract](#abstract)
- [1. Evidence selection](#1-evidence-selection)
- [2. Scope and experimental setup](#2-scope-and-experimental-setup)
  - [2.1 Policy, benchmark, and
    rollouts](#21-policy-benchmark-and-rollouts)
  - [2.2 Detectors and splits](#22-detectors-and-splits)
  - [2.3 Keep training, ranking, and calibration
    separate](#23-keep-training-ranking-and-calibration-separate)
- [3. Results](#3-results)
  - [3.1 The outcome label is perfectly recoverable from rollout
    length](#31-the-outcome-label-is-perfectly-recoverable-from-rollout-length)
  - [3.2 Pooled AUC is dominated by cross-task
    comparisons](#32-pooled-auc-is-dominated-by-cross-task-comparisons)
  - [3.3 LOTO performance is modest and task
    dependent](#33-loto-performance-is-modest-and-task-dependent)
  - [3.4 SAFE functional conformal prediction does not yield a useful
    operating
    point](#34-safe-functional-conformal-prediction-does-not-yield-a-useful-operating-point)
- [4. Supporting findings](#4-supporting-findings)
  - [4.1 Extra layers do not provide a reliable held-out
    gain](#41-extra-layers-do-not-provide-a-reliable-held-out-gain)
  - [4.2 Time-resolved fusion does not improve ranking; a residual
    warning signal is
    horizon-dependent](#42-time-resolved-fusion-does-not-improve-ranking-a-residual-warning-signal-is-horizon-dependent)
  - [4.3 Latent geometry is organized strongly by
    task](#43-latent-geometry-is-organized-strongly-by-task)
- [5. What the evidence supports](#5-what-the-evidence-supports)
  - [Supported](#supported)
  - [Not supported](#not-supported)
- [6. Recommended evaluation
  protocol](#6-recommended-evaluation-protocol)
- [7. Limitations](#7-limitations)
- [8. Reproducibility and artifact
  map](#8-reproducibility-and-artifact-map)
  - [Featured artifacts](#featured-artifacts)
  - [Commands for the final headline
    analyses](#commands-for-the-final-headline-analyses)
  - [References](#references)

## Abstract

[SAFE](https://arxiv.org/abs/2506.09937) proposes failure probes over
frozen vision-language-action (VLA) representations and evaluates
transfer to unseen tasks. We reproduced the OpenVLA/LIBERO probe setting
and then hardened the evaluation around the deployment question: **does
the detector warn early on an unseen task while maintaining an
acceptable false-positive rate?**

On 1,000 rollouts from 10 LIBERO tasks, the SAFE-style pooled
early-window ROC-AUC is strong (0.747 for MLP and 0.755 for LSTM), but
macro within-task AUC is lower (0.660 and 0.646). An exact pair
decomposition explains why: 71.1% of the comparisons entering pooled AUC
are cross-task comparisons, and those comparisons are easier than
within-task comparisons. Under 10-fold leave-one-task-out (LOTO), macro
early-window AUC is 0.672 for MLP and 0.652 for LSTM, with wide
variation across tasks.

The decisive test is SAFE’s time-varying functional conformal procedure
under LOTO. At operating points with at most 5% realized false-positive
rate, the benchmark-compatible earliest-stop evaluation catches only
8.5% of failures for MLP and 6.9% for LSTM. Counting missed failures as
alarm-at-end, the mean alarm position is 95.5% and 95.7% of the
evaluation window. Even when the full rollout is available, the
corresponding catch rates are only 23.8% and 17.5%. Reaching roughly 50%
catch requires 8.9% realized FPR for MLP and 19.0% for LSTM, and
warnings remain late.

The evidence is strong enough for a narrow conclusion: **in this
reproduction, pooled ROC-AUC does not imply reliable, timely, cross-task
failure warning.** It does not establish that SAFE is invalid in every
policy, benchmark, or data regime.

## 1. Evidence selection

This report does not treat every generated artifact as equally
probative. The featured evidence was selected for held-out-task
validity, resistance to the known timeout shortcut, statistical
transparency, and reproducibility from saved checkpoints.

| Evidence | Assessment | Use in this report |
|----|----|----|
| Functional CP under 10-fold LOTO, 10 CP splits, 8 alpha values | **Primary.** Directly tests false alarms, failure coverage, and warning time with SAFE’s functional procedure. | Headline result |
| LOTO `falert_early` AUC on all 10 tasks | **Primary.** Every task is held out once; task bootstrap exposes heterogeneity. | Headline result |
| Exact pooled-AUC pair decomposition on finalized checkpoints | **Primary mechanism, limited population inference.** Exact for the observed splits; only three held-out tasks per original split. | Headline explanation with caveat |
| Rollout-length distribution and length-only AUC | **Primary diagnostic.** Directly establishes the timeout shortcut in uncapped evaluation. | Headline diagnostic |
| Incremental layer-value tests under LOTO | **Mixed secondary result.** Most tests are null; the late-layer trio beats shuffled features but not layer 32 or functional CP. | Supporting result |
| CKA and UMAP latent geometry | **Explanatory only.** Useful for diagnosing task structure, but embeddings are not evidence of detector performance. | Appendix/support |
| Fixed-timestep and progress-proxy analyses | **Supporting.** Useful disentanglement checks, but less direct than the final LOTO functional-CP evaluation. | Not headlined |
| Earlier constant-threshold detection-time tables | **Superseded.** They used a simpler split-conformal threshold rather than SAFE’s functional procedure. | Excluded from conclusions |
| Earlier pilot and undertrained runs | **Development evidence.** Valuable for generating hypotheses, not for final effect sizes. | Excluded from conclusions |

Earlier prose drafts and their constant-threshold conformal tables were
consolidated into this report. Their older conformal numbers are
superseded by `evaluate_functional_cp_loto.py`.

## 2. Scope and experimental setup

### 2.1 Policy, benchmark, and rollouts

- Policy: [OpenVLA](https://arxiv.org/abs/2406.09246).
- Benchmark: 10 tasks from [LIBERO](https://arxiv.org/abs/2306.03310).
- Data: 1,000 rollouts, exactly 100 per task; 527 successes and 473
  failures.
- Sources: `data/rollouts/openVLA-last-layer` and
  `data/rollouts/openVLA-last-layer-2`, each containing 500 distinct
  rollouts.
- Representation: the last action-token hidden state, with layer 32 used
  for the final detector comparison. The layer-complementarity study
  uses layers 1, 4, 8, 12, 16, 20, 24, 28, and 32.

The two rollout shards use different historical storage schemas: the
first preserves the action-token axis and the second stores an
already-selected last token. `MixedSchemaLayerDataset` reads both
without rewriting the source files and exposes a common
4,096-dimensional layer-32 trajectory.

### 2.2 Detectors and splits

The finalized layer-32 probes use the paper-selected OpenVLA settings:

| Probe     | Learning rate | Regularization | Architecture                        |
|-----------|--------------:|---------------:|-------------------------------------|
| SAFE-MLP  |          1e-4 |           1e-2 | two-layer MLP, cumulative-sum score |
| SAFE-LSTM |          1e-4 |              1 | one-layer LSTM, all-history score   |

Generalization is evaluated with 10 LOTO folds. Fold *k* trains and
calibrates on the other nine tasks, then tests on task *k*. Every
reported LOTO macro value first aggregates within held-out task and then
averages equally across the 10 tasks.

### 2.3 Keep training, ranking, and calibration separate

Three stages answer different questions:

1.  **Probe training** produces a scalar failure score trajectory from
    frozen hidden states.
2.  **Ranking evaluation** computes ROC-AUC from a rollout score. The
    primary score, `falert_early`, is the maximum score up to the task’s
    shortest rollout (`task_min_step`). `falert_end` uses the full
    rollout.
3.  **Conformal calibration** converts score trajectories into
    time-varying alarm thresholds. It governs thresholded metrics such
    as realized FPR, failure catch rate, and detection time; it does not
    explain ROC-AUC.

The earliest-stop cap follows SAFE’s benchmark evaluation, but it is
retrospective: it uses a per-task minimum rollout length known from the
dataset. It is therefore a useful comparability check, not a deployable
stopping policy. The full-end evaluation is also reported, but it is
vulnerable to elapsed-time and survival information.

## 3. Results

### 3.1 The outcome label is perfectly recoverable from rollout length

Every failed rollout lasts exactly 520 steps, whereas successful
rollouts may terminate earlier. Consequently, a length-only classifier
obtains ROC-AUC 1.0 for every held-out task. This does not make
early-window evaluation invalid; it does make uncapped, full-rollout
discrimination a poor measure of early semantic failure detection.

<img src="success_length_violin.png"
data-fig-alt="Violin plots of successful rollout lengths for ten tasks, all below or approaching the 520-step line used by every failure."
alt="Successful rollout lengths by task. Every failure is at the 520-step timeout." />

This diagnostic is the reason the report treats `falert_early` and
functional alarm timing as primary, and `falert_end` as a sensitivity
analysis.

### 3.2 Pooled AUC is dominated by cross-task comparisons

ROC-AUC is a pairwise ranking statistic. When rollouts from several
tasks are pooled, most failure-success pairs come from different tasks.
Those pairs can be ranked correctly because of task-dependent score
scale, failure prevalence, or trajectory structure even when within-task
ranking is only modest.

| Probe | Pooled AUC | Macro within-task | Within-task micro | Between-task AUC | Cross-task pair share | Pooled - macro |
|----|---:|---:|---:|---:|---:|---:|
| MLP | 0.747 | 0.660 | 0.688 | 0.768 | 71.1% | +0.087 |
| LSTM | 0.755 | 0.646 | 0.668 | 0.789 | 71.1% | +0.109 |

<img src="pooled_auc_decomposition.png"
data-fig-alt="Grouped bars show that between-task AUC exceeds macro within-task AUC in all three splits for both LSTM and MLP."
alt="Exact decomposition of pooled ROC-AUC into within-task and between-task pair rankings for each original split." />

For each model and split, the decomposition is exact: pooled AUC is the
pair-count-weighted average of within-task and between-task pair
accuracy. The observed gap is positive in all six model-split
combinations. Split 1 is the clearest case: task 8 has a high failure
rate and high detector scores, while tasks 5 and 7 have more successes
and lower score scales, making cross-task pairs especially easy.

The inferential boundary matters. Each original split contains only
three held-out tasks. A task-clustered, label-stratified bootstrap gives
intervals for the pooled-minus-macro gap that overlap zero within each
split. The decomposition therefore establishes the **mechanism in these
observed splits**, but the report does not claim a precisely estimated
population-wide inflation constant. The per-split values and bootstrap
intervals are preserved in
[`pair_decomposition_by_split.csv`](results_audit/pair_decomposition_by_split.csv)
and
[`pair_decomposition_bootstrap_summary.csv`](results_audit/pair_decomposition_bootstrap_summary.csv).

### 3.3 LOTO performance is modest and task dependent

| Probe | LOTO macro early AUC | 95% task-bootstrap CI | Across-task SD |  Task range |
|-------|---------------------:|----------------------:|---------------:|------------:|
| MLP   |                0.672 |      \[0.599, 0.738\] |          0.113 | 0.413-0.817 |
| LSTM  |                0.652 |      \[0.580, 0.730\] |          0.123 | 0.476-0.870 |

The variation is operationally important. Both probes are below chance
on task 5, while tasks 0 and 7 are comparatively easy. A pooled score
cannot expose this failure mode because task-dependent scale contributes
to the pooled ranking. The full per-task table is available in
[`loto_per_task_auc.csv`](results_audit/loto_per_task_auc.csv).

### 3.4 SAFE functional conformal prediction does not yield a useful operating point

We reimplemented the official functional-conformal path used by SAFE:

- take successful seen-task validation score trajectories;
- extend-align variable-length trajectories to a fixed 520-step horizon;
- randomly split successful calibration trajectories 30/70;
- fit the mean trajectory and SAFE `Tfunc` modulation on the first
  portion;
- calibrate a one-sided upper band on the second portion;
- alarm at the first threshold crossing on the unseen held-out task.

The evaluation sweeps 8 alpha values and repeats the random 30/70
functional-CP split for 10 seeds. Within each held-out task we average
over CP seeds; across tasks we report an equal-weight macro mean and a
10,000-replicate task bootstrap.

<img src="functional_cp_loto_tradeoff.png"
data-fig-alt="Four tradeoff plots for LSTM and MLP at earliest-stop and full-end evaluation. Low false-positive points have low catch rates and late effective alarms."
alt="Failure catch rate versus realized false-positive rate. Color records effective alarm position, assigning missed failures to the end of the window." />

The most informative constraint-selected points are:

| Evaluation | Probe | Selection rule | Alpha | Realized FPR | Failure catch | Effective alarm position |
|----|----|----|---:|---:|---:|---:|
| Earliest stop | MLP | best point with FPR \<= 5% | 0.075 | 4.8% | 8.5% | 95.5% |
| Earliest stop | LSTM | best point with FPR \<= 5% | 0.010 | 3.8% | 6.9% | 95.7% |
| Full end | MLP | best point with FPR \<= 5% | 0.025 | 2.6% | 23.8% | 96.2% |
| Full end | LSTM | best point with FPR \<= 5% | 0.010 | 4.4% | 17.5% | 91.8% |
| Full end | MLP | lowest-FPR point with catch \>= 50% | 0.050 | 8.9% | 52.0% | 89.6% |
| Full end | LSTM | lowest-FPR point with catch \>= 50% | 0.075 | 19.0% | 57.0% | 68.4% |

“Effective alarm position” is the first-alarm fraction for caught
failures and 1.0 for missed failures. This prevents a misleading
caught-only summary: a detector can appear early by catching a small,
easy subset while missing most failures. Under earliest-stop evaluation,
LSTM needs 37.5% realized FPR to catch 56.7% of failures; MLP never
reaches 50% catch in the tested alpha range.

The conclusion is stable across CP split seeds, but transfer is
heterogeneous across tasks. For example, at nominal alpha 0.01 the LSTM
earliest-stop FPR is near zero on several tasks but 29.6% on task 2.
Functional conformal calibration on seen tasks therefore should not be
described as a uniform guarantee on a new task.

## 4. Supporting findings

### 4.1 Extra layers do not provide a reliable held-out gain

The layer study asks a stricter question than whether layer
representations are different: does layer *k* improve unseen-task AUC
after the layer-32 detector score is already known? A logistic combiner
is fit on seen tasks using either PCA-100 features or five supervised
PLS directions targeting the layer-32 residual, then evaluated on the
held-out task.

The best mean improvement is layer 20 at approximately +0.024 AUC for
both methods. Its 95% intervals cross zero: \[-0.011, 0.056\] for
PCA-100 and \[-0.009, 0.054\] for residual PLS. Shallow layers are
zero-to-negative. The appropriate conclusion is **no measurable
complementary held-out signal in these tests**, not proof that
complementary information cannot exist.

### 4.2 Time-resolved fusion does not improve ranking; a residual warning signal is horizon-dependent

The pooled PCA/PLS analysis could miss nonlinear or short-lived
information. We therefore first tested the strongest individual
candidate, layer 20, with a causal time-resolved model. A shared
projection dynamically gates layers 20 and 32 at each timestep before an
LSTM. Sharing all parameters makes the one-layer and two-layer versions
exactly capacity matched at 164,481 trainable parameters. Training and
evaluation stop at `task_min_step`, and a negative control replaces
layer 20 with another rollout from the same task and data partition.
Because all rollouts from a task share the early-window length, this
retains task and progress structure while breaking episode alignment.

The completed follow-up now uses two ordered 500-rollout shards: 1,000
rollouts total and 100 for each of tasks 0–9. All files have the same
compact schema: layers 20 and 32, the last action token, 4,096 features
per layer, and temporal stride 4. The first shard was normalized from
token-preserving artifacts and the second from artifacts that had
already selected the last token; the two portable schema and inventory
audits are published with the results. This remains a targeted secondary
experiment with newly trained dynamic monitors, not a replacement for
the report’s primary finalized MLP/LSTM analysis.

Three fixed training seeds are averaged within task before tasks are
bootstrapped. This separates optimizer instability from held-out-task
uncertainty.

| Monitor | Macro held-out AUC | SD across tasks | Delta vs layer 32 | 95% task-bootstrap CI | Positive tasks |
|----|---:|---:|---:|---:|---:|
| Layer 32 | 0.673 | 0.083 | – | – | – |
| Dynamic layer 20 + 32 | 0.670 | 0.088 | -0.004 | \[-0.020, 0.012\] | 5/10 |
| Dynamic shuffled 20 + 32 | 0.670 | 0.084 | -0.003 | \[-0.018, 0.015\] | 4/10 |
| Frozen-32 + layer-20 residual | 0.681 | 0.097 | +0.007 | \[-0.005, 0.018\] | 7/10 |
| Frozen-32 + shuffled residual | 0.679 | 0.087 | +0.005 | \[-0.00002, 0.011\] | 6/10 |

The direct fusion delta varies substantially across the three
optimization runs (+0.010, -0.025, and +0.003). After averaging seeds
within task, its effect is slightly negative, only 5/10 task-level
contrasts are positive, and the paired exact sign-flip test is not
significant ($p=0.670$). Real fusion is also indistinguishable from
shuffled fusion: -0.001 AUC (95% CI \[-0.015, 0.014\], $p=0.928$). The
learned real-data gate nevertheless assigns 69.8% of its mass to layer
20 on average. Doubling the sample therefore removes the earlier
500-rollout descriptive gain rather than strengthening it.

The residual model closes a second loophole: perhaps layer 20 should
only correct, rather than replace, layer-32 information. The layer-32
model is frozen and the auxiliary branch is initialized to an exact zero
correction. The real residual has the largest descriptive AUC, but its
+0.007 gain over layer 32 is unresolved and it exceeds the
capacity-matched shuffled residual by only +0.002 (95% CI \[-0.008,
0.011\], $p=0.721$). The AUC evidence therefore does not attribute the
residual model’s apparent gain to episode-aligned layer 20.

Functional conformal evaluation tests whether the small ranking change
becomes a useful warning change. In addition to the benchmark-compatible
retrospective `task_min_step` window, we calibrate and evaluate at
stride-adjusted counterparts of the predeclared 50/100/148-step windows:
13, 25, and 37 processed samples. The shortest processed rollout has
length 37, so these fixed-window results use no padding, score
extension, or rollout exclusion.

At independently grid-selected operating points with at most 5% macro
realized FPR, the direct layer-32 comparison is:

| Evaluation window | Layer-32 FPR | Layer-32 catch | Layer-20+32 FPR | Layer-20+32 catch | Catch-rate delta (95% task-bootstrap CI) |
|----|---:|---:|---:|---:|---:|
| Retrospective earliest stop | 0.037 | 0.136 | 0.029 | 0.100 | -0.036 \[-0.068, 0.002\] |
| Fixed 50-step counterpart | 0.037 | 0.089 | 0.043 | 0.081 | -0.008 \[-0.056, 0.047\] |
| Fixed 100-step counterpart | 0.031 | 0.082 | 0.036 | 0.081 | -0.002 \[-0.038, 0.039\] |
| Fixed 148-step counterpart | 0.046 | 0.137 | 0.030 | 0.094 | -0.043 \[-0.088, -0.003\] |

The conditional intervals above hold the independently selected alpha
grid points fixed. Direct fusion catches no more failures in any window
and has less effective lead in every window. It also does not resolve
against shuffled fusion: real-minus-shuffled catch-rate intervals cross
zero retrospectively and at all three fixed horizons.

The residual formulation produces one limited positive result:

| Evaluation window | Layer-32 FPR / catch | Residual FPR / catch | Shuffled residual FPR / catch |
|----|---:|---:|---:|
| Retrospective earliest stop | 0.037 / 0.136 | 0.044 / 0.172 | 0.040 / 0.148 |
| Fixed 50-step counterpart | 0.037 / 0.089 | 0.042 / 0.095 | 0.036 / 0.088 |
| Fixed 100-step counterpart | 0.031 / 0.082 | 0.043 / 0.105 | 0.033 / 0.084 |
| Fixed 148-step counterpart | 0.046 / 0.137 | 0.038 / 0.112 | 0.048 / 0.144 |

Retrospectively, the residual improves catch by +0.036 over layer 32
(95% CI \[0.017, 0.057\], exact sign-flip $p=0.011$) and +0.025 over
shuffled residual (95% CI \[0.006, 0.043\], $p=0.044$). Its
effective-lead fraction also improves by +0.018 over layer 32 (95% CI
\[0.005, 0.032\]), while realized FPR rises by 0.8 percentage points and
remains below 5%. This is evidence that aligned layer 20 can affect the
residual model’s retrospective warning behavior.

It is not a robust deployment result. At the fixed 50-step horizon the
residual contrast is unresolved. At 100 steps, catch improves by +0.022
but realized FPR also rises by +0.012; the real-versus-shuffled catch
interval narrowly crosses zero. At 148 steps the residual catches fewer
failures and warns later. The operating points were selected from the
same alpha grid on these held-out results, and the many
architecture/window comparisons were not multiplicity adjusted. The
appropriate consequence is a preregistered residual-only replication,
not a claim that multiple layers improve SAFE.

An older three-task split had shown a larger gain for a late-layer
mixer, so we also tested layers 24, 28, and 32. The original development
set contained 450 usable rollouts across nine tasks. The newly added
second shard provides an independent 500-rollout, ten-task replication
with the same model, within-partition shuffled control, three training
seeds, and LOTO protocol:

| Dataset | Layer-32 AUC | Real 24+28+32 AUC | Shuffled AUC | Real minus layer 32 (95% CI) | Real minus shuffled (95% CI) |
|----|---:|---:|---:|---:|---:|
| Development: 450 rollouts, 9 tasks | 0.586 | 0.609 | 0.573 | +0.023 \[-0.008, 0.049\] | +0.036 \[0.002, 0.070\] |
| Independent shard: 500 rollouts, 10 tasks | 0.681 | 0.703 | 0.667 | +0.022 \[-0.010, 0.053\] | +0.037 \[-0.004, 0.079\] |

The replication reproduces both point estimates almost exactly. Direct
gains are positive on 7/10 replication tasks, but the task-bootstrap
interval crosses zero and the exact sign-flip test is unresolved
($p=0.229$). The real model beats shuffled fusion on only 5/10
replication tasks ($p=0.134$). Gate behavior also replicates: the
development model assigns 46.0%, 41.7%, and 12.3% mean weight to layers
24, 28, and 32; the independent model assigns 43.3%, 44.9%, and 11.8%.
These repeated effect sizes and gate weights strengthen the narrow claim
that layers 24 and 28 contain complementary episode-level information,
while the task-level uncertainty still does not establish better
ranking.

Functional conformal evaluation on the independent shard again fails to
turn that information into warning utility:

| Monitor | Realized FPR | Catch rate | Effective lead fraction |
|----|---:|---:|---:|
| Layer 32 | 0.047 | 0.114 | 0.052 |
| Dynamic layers 24 + 28 + 32 | 0.036 | 0.090 | 0.043 |
| Shuffled layers 24 + 28 + 32 | 0.047 | 0.082 | 0.050 |

At the fixed 50-step counterpart, real fusion’s catch delta versus layer
32 is +0.010 (95% CI \[-0.012, 0.033\]); at 100 and 148 steps it is
-0.004 and -0.015, with both intervals crossing zero.
Real-versus-shuffled catch intervals also cross zero in every window.
Moreover, one of the three training seeds has no alpha-grid point below
5% realized FPR for any variant, demonstrating material calibration
instability.

The independent replication therefore makes the mechanistic signal more
credible but also replicates the operational negative result. It
supports one narrow statement—late layers carry some complementary
information—but **does not support claiming a better SAFE monitor or a
broad multilayer campaign**.

### 4.3 Latent geometry is organized strongly by task

Linear CKA shows that depth changes the representation: L1-L32 CKA is
0.466 and L4-L32 is 0.512, while L28-L32 is 0.980. The UMAP
visualization likewise forms clearer task clusters in later layers. This
explains how a pooled evaluator can benefit from task identity, but it
is only a diagnostic; UMAP geometry is not a performance metric.

Linear CKA captures only second-order (covariance) similarity. To test
whether nonlinear structure changes the picture, we also computed
RBF-kernel CKA with a median-heuristic bandwidth on the same
1,000-rollout, 9-layer, rollout-pooled features
([`rbf_cka_table.csv`](results_audit/rbf_cka_table.csv),
[`cka_linear_vs_rbf.csv`](results_audit/cka_linear_vs_rbf.csv)):

| Pair | Linear CKA | RBF CKA | Difference |
|-----|----------:-------:|-----------:|
| L1–L32 | 0.466 | 0.670 | +0.204 |
| L4–L32 | 0.512 | 0.726 | +0.214 |
| L8–L32 | 0.727 | 0.865 | +0.138 |
| L20–L32 | 0.883 | 0.953 | +0.070 |
| L24–L32 | 0.957 | 0.975 | +0.018 |
| L28–L32 | 0.980 | 0.987 | +0.007 |

RBF CKA is uniformly higher than linear CKA, and the gap is largest for
shallow–deep pairs. This means that shallow and deep layers share more
nonlinear structure than linear CKA suggests. The deep block (L24–L32)
remains highly similar under both kernels (0.957 linear, 0.975 RBF), and
the shallow–deep de-correlation is stronger under linear CKA, so the
qualitative conclusion — representations de-correlate with depth —
holds under both. The RBF result does not overturn the linear CKA
finding, but it does show that the linear kernel overestimates the
shallow–deep representational distance.

<img src="umap_by_task.png"
data-fig-alt="Nine UMAP panels, one per hidden layer, showing increasingly separated clusters for the ten tasks."
alt="UMAP of early-window rollout-pooled features, colored by task." />

## 5. What the evidence supports

### Supported

1.  In this OpenVLA/LIBERO reproduction, pooled early-window ROC-AUC is
    materially higher than equal-weight within-task AUC.
2.  Most pooled AUC comparisons are between tasks, and those comparisons
    are easier than within-task comparisons in every observed original
    split.
3.  Cross-task early ranking is modest and highly task dependent under
    LOTO.
4.  SAFE’s functional conformal procedure does not produce an operating
    point that is simultaneously low-FPR, high-recall, and timely on
    held-out tasks.
5.  Capacity-matched direct fusion does not improve layer-32 held-out
    AUC or functional-CP warning. A larger residual model improves one
    retrospective low-FPR operating point, including against its
    shuffled control, but the effect does not persist across fixed
    horizons and its AUC gain is unresolved.

### Not supported

1.  That SAFE is invalid in all settings, policies, or benchmarks.
2.  That pooled AUC is mathematically wrong; it answers a different,
    mixture-level ranking question.
3.  That the detector has no state information about failure. The audit
    shows limited transfer and poor alarm utility, not absence of all
    signal.
4.  That functional conformal prediction is generally invalid. The
    result is a distribution-transfer failure in this task-held-out
    reproduction.
5.  That additional layers can never help. Several real-versus-shuffled
    contrasts suggest complementary signal, but none of the tested
    methods establishes a robust fixed-horizon warning improvement over
    layer 32.

## 6. Recommended evaluation protocol

A credible multitask VLA failure-detection result should report:

1.  **Pooled and macro per-task AUC together**, plus the fraction and
    accuracy of between-task pairs.
2.  **LOTO or another task-clustered split**, with per-task results and
    task-level uncertainty.
3.  **Length-only, elapsed-time, and task-identity baselines** whenever
    episode termination depends on outcome.
4.  **A predeclared fixed horizon** in addition to any retrospective
    `task_min_step` benchmark cap.
5.  **Thresholded deployment metrics**: realized held-out-task FPR,
    catch rate, and alarm time with misses included.
6.  **Calibration sensitivity** across random conformal splits and
    held-out tasks.
7.  **Incremental held-out value** before claiming that another layer,
    modality, or feature family is complementary.

## 7. Limitations

- The audit covers one policy family and 10 LIBERO tasks; 10 task
  clusters still produce wide uncertainty intervals.
- Failure is an episode outcome, not an annotated onset. Because every
  failure reaches the 520-step limit, this dataset cannot identify when
  failure first became unavoidable.
- `task_min_step` is benchmark-compatible but retrospective. A
  deployment claim needs fixed, predeclared horizons or online stopping
  rules.
- The predeclared 50/100/148-step fixed-horizon sensitivity and the public
  per-timestep score bundle for the primary 1,000-rollout MLP/LSTM analysis
  are included. The secondary 1,000-rollout layer-20
  follow-up includes stride-adjusted versions of those fixed horizons,
  but its newly trained dynamic models and stride-4 inputs cannot
  substitute for the finalized primary probes.
- Functional CP matches the published SAFE code path, but this is an
  independent reproduction rather than an author-verified reanalysis.
- The pair decomposition uses the three original task-held-out splits.
  It is exact for those samples but underpowered for population-level
  inference over tasks.
- The current rollout artifacts do not provide failure-onset labels or
  the action-conditioned counterfactual data needed to evaluate recovery
  or reachability certificates.
- The layer-20/32 time-resolved follow-up now covers 1,000 rollouts over
  all 10 tasks. Its residual result was discovered across several
  architecture/window comparisons and needs preregistered replication.
  The original 450-rollout, nine-task late-layer result has an
  independent 500-rollout, ten-task replication, but the two shards
  cannot currently be combined into a single 1,000-rollout late-layer
  analysis because the first surviving compact shard lacks layers 24/28.

## 8. Reproducibility and artifact map

### Featured artifacts

| Finding | Script | Tracked artifact |
|----|----|----|
| Timeout structure | `scripts/plot_success_length_dist.py` | [`success_length_violin.png`](success_length_violin.png) |
| Pooled/macro pair decomposition | `scripts/score_layer_pooled_macro.py`, `scripts/analyze_pooled_macro_decomposition.py` | [`pooled_auc_decomposition.png`](pooled_auc_decomposition.png), [`pair_decomposition_descriptive.csv`](results_audit/pair_decomposition_descriptive.csv) |
| LOTO early AUC | finalized LOTO checkpoint summaries | [`loto_macro_ci.csv`](results_audit/loto_macro_ci.csv), [`loto_per_task_auc.csv`](results_audit/loto_per_task_auc.csv) |
| Primary checkpoint/split contract | `scripts/verify_primary_checkpoints.py` | [`primary_checkpoint_audit.json`](results_audit/primary_checkpoint_audit.json) |
| Functional CP under LOTO | `scripts/evaluate_functional_cp_loto.py`, `scripts/evaluate_conformal.py` | [`functional_cp_loto_report.md`](results_audit/functional_cp_loto_report.md), [`functional_cp_loto_macro.csv`](results_audit/functional_cp_loto_macro.csv), [`functional_cp_loto_per_task.csv`](results_audit/functional_cp_loto_per_task.csv) |
| Added-layer value | layer-complementarity analyses | [`layer_incremental_delta_summary.csv`](results_audit/layer_incremental_delta_summary.csv) |
| Layer-20/32 ten-task follow-up | `scripts/evaluate_dynamic_layer_loto.py`, `scripts/evaluate_dynamic_layer_functional_cp.py`, `scripts/summarize_dynamic_layer_seeds.py` | [`shard-0 manifest`](results_audit/dynamic_layer_data_manifest_1000_shard0.json), [`shard-1 manifest`](results_audit/dynamic_layer_data_manifest_1000_shard1.json), [`dynamic_layer_loto_1000_summary.json`](results_audit/dynamic_layer_loto_1000_summary.json), [`dynamic_layer_loto_1000_per_task.csv`](results_audit/dynamic_layer_loto_1000_per_task.csv), [`dynamic_layer_functional_cp_1000_macro.csv`](results_audit/dynamic_layer_functional_cp_1000_macro.csv), [`dynamic_layer_functional_cp_1000_operating_points.csv`](results_audit/dynamic_layer_functional_cp_1000_operating_points.csv), [`dynamic_layer_functional_cp_1000_comparisons.csv`](results_audit/dynamic_layer_functional_cp_1000_comparisons.csv) |
| Late-layer nine-task follow-up | `scripts/evaluate_dynamic_layer_loto.py`, `scripts/evaluate_dynamic_layer_functional_cp.py` | [`dynamic_layer_data_manifest.json`](results_audit/dynamic_layer_data_manifest.json), [`dynamic_late_layer_loto_450_summary.json`](results_audit/dynamic_late_layer_loto_450_summary.json), [`dynamic_late_layer_functional_cp_macro.csv`](results_audit/dynamic_late_layer_functional_cp_macro.csv), [`dynamic_late_layer_functional_cp_operating_points.csv`](results_audit/dynamic_late_layer_functional_cp_operating_points.csv) |
| Independent late-layer replication | `scripts/evaluate_dynamic_layer_loto.py`, `scripts/evaluate_dynamic_layer_functional_cp.py`, `scripts/summarize_dynamic_layer_seeds.py` | [`replication manifest`](results_audit/dynamic_late_layer_replication_data_manifest.json), [`dynamic_late_layer_replication_loto_summary.json`](results_audit/dynamic_late_layer_replication_loto_summary.json), [`dynamic_late_layer_replication_loto_per_task.csv`](results_audit/dynamic_late_layer_replication_loto_per_task.csv), [`dynamic_late_layer_replication_functional_cp_macro.csv`](results_audit/dynamic_late_layer_replication_functional_cp_macro.csv), [`dynamic_late_layer_replication_functional_cp_operating_points.csv`](results_audit/dynamic_late_layer_replication_functional_cp_operating_points.csv), [`dynamic_late_layer_replication_functional_cp_comparisons.csv`](results_audit/dynamic_late_layer_replication_functional_cp_comparisons.csv) |
| Representation similarity | layer-complementarity analyses | [`linear_cka_table.csv`](results_audit/linear_cka_table.csv), [`umap_by_task.png`](umap_by_task.png) |

### Commands for the final headline analyses

``` bash
uv run python scripts/score_layer_pooled_macro.py \
  --root data/rollouts/openVLA-last-layer data/rollouts/openVLA-last-layer-2 \
  --model-type mlp --checkpoint-dir runs/openvla_mlp_seed012_l32_opt \
  --layer 32 --seeds 0 1 2 \
  --output runs/results_audit/final_mlp_test_scores.csv

uv run python scripts/score_layer_pooled_macro.py \
  --root data/rollouts/openVLA-last-layer data/rollouts/openVLA-last-layer-2 \
  --model-type lstm --checkpoint-dir runs/openvla_lstm_seed012_l32_opt \
  --layer 32 --seeds 0 1 2 \
  --output runs/results_audit/final_lstm_test_scores.csv

uv run python scripts/analyze_pooled_macro_decomposition.py \
  --scores runs/results_audit/final_mlp_test_scores.csv \
           runs/results_audit/final_lstm_test_scores.csv \
  --output-dir runs/results_audit/pair_decomposition \
  --figure docs/pooled_auc_decomposition.png \
  --n-bootstrap 10000 --bootstrap-seed 20260714

uv run python scripts/evaluate_functional_cp_loto.py \
  --root data/rollouts/openVLA-last-layer data/rollouts/openVLA-last-layer-2 \
  --mlp-checkpoint-dir runs/openvla_mlp_loto_l32 \
  --lstm-checkpoint-dir runs/openvla_lstm_loto_l32 \
  --output-dir runs/results_audit/functional_cp_loto \
  --alphas 0.01,0.025,0.05,0.075,0.1,0.15,0.2,0.3 \
  --cp-seeds 0,1,2,3,4,5,6,7,8,9 \
  --horizon 520 --batch-size 8 --task-bootstrap 10000 \
  --bootstrap-seed 20260716

# Normalize the second 500-rollout shard to match the first compact shard:
uv run python scripts/prepare_openvla_layer_subset.py \
  --input-root data/processed/OpenVLA_500_2 \
  --output-root data/processed/openVLA_500_2_l20_l32_last_s4 \
  --layers 20,32 --token-index -1 --temporal-stride 4 \
  --expected-rollouts 500 --expected-tasks 0,1,2,3,4,5,6,7,8,9 \
  --expected-rollouts-per-task 50 --minimum-length 37

# Compose a stable ordered union with hard links (no duplicate tensor storage):
mkdir -p data/processed/openVLA_1000_l20_l32_last_s4/shard0 \
         data/processed/openVLA_1000_l20_l32_last_s4/shard1
cp -Rl data/processed/openVLA_500_l20_l32_last_s4/. \
       data/processed/openVLA_1000_l20_l32_last_s4/shard0/
cp -Rl data/processed/openVLA_500_2_l20_l32_last_s4/. \
       data/processed/openVLA_1000_l20_l32_last_s4/shard1/

uv run python scripts/evaluate_dynamic_layer_loto.py \
  --root data/processed/openVLA_1000_l20_l32_last_s4 \
  --output-dir runs/openvla_dynamic_layer_loto_1000 \
  --variants baseline32,layer20_32,shuffled20_32,residual20_32,shuffled_residual20_32 \
  --epochs 30 --patience 5 --batch-size 16 \
  --projection-dim 32 --hidden-dim 64 --seed 20260722

# Repeat as runs/openvla_dynamic_layer_loto_1000_seed1 and ..._seed2 with
# --seed 20260723 and --seed 20260724, respectively.

uv run python scripts/evaluate_dynamic_layer_functional_cp.py \
  --root data/processed/openVLA_1000_l20_l32_last_s4 \
  --checkpoint-dir runs/openvla_dynamic_layer_loto_1000 \
  --output-dir runs/openvla_dynamic_layer_loto_1000/functional_cp \
  --alphas 0.01,0.025,0.05,0.075,0.1,0.15,0.2,0.3 \
  --cp-seeds 0,1,2,3,4,5,6,7,8,9 \
  --horizon 130 --fixed-horizons 13,25,37 \
  --task-bootstrap 20000 --bootstrap-seed 20260722

# Repeat functional CP for the other two checkpoint directories, then aggregate:
uv run python scripts/summarize_dynamic_layer_seeds.py \
  --summary runs/openvla_dynamic_layer_loto_1000/summary.json \
            runs/openvla_dynamic_layer_loto_1000_seed1/summary.json \
            runs/openvla_dynamic_layer_loto_1000_seed2/summary.json \
  --cp-raw runs/openvla_dynamic_layer_loto_1000/functional_cp/functional_cp_raw.csv \
           runs/openvla_dynamic_layer_loto_1000_seed1/functional_cp/functional_cp_raw.csv \
           runs/openvla_dynamic_layer_loto_1000_seed2/functional_cp/functional_cp_raw.csv \
  --output-dir runs/openvla_dynamic_layer_loto_1000_seed_aggregate \
  --n-bootstrap 20000 --bootstrap-seed 20260722

# Independent late-layer replication: repeat training and functional CP for
# seeds 20260722, 20260723, and 20260724, then aggregate as above.
uv run python scripts/prepare_openvla_layer_subset.py \
  --input-root data/processed/OpenVLA_500_2 \
  --output-root data/processed/openVLA_500_2_l24_l28_l32_last_s4 \
  --layers 24,28,32 --token-index -1 --temporal-stride 4 \
  --expected-rollouts 500 --expected-tasks 0,1,2,3,4,5,6,7,8,9 \
  --expected-rollouts-per-task 50 --minimum-length 37

uv run python scripts/evaluate_dynamic_layer_loto.py \
  --root data/processed/openVLA_500_2_l24_l28_l32_last_s4 \
  --dataset-layers 24,28,32 \
  --output-dir runs/openvla_dynamic_late_loto_500_2_repl \
  --variants baseline32,late24_28_32,shuffled_late24_28_32 \
  --epochs 30 --patience 5 --batch-size 16 \
  --projection-dim 32 --hidden-dim 64 --seed 20260722

uv run python scripts/evaluate_dynamic_layer_functional_cp.py \
  --root data/processed/openVLA_500_2_l24_l28_l32_last_s4 \
  --dataset-layers 24,28,32 \
  --checkpoint-dir runs/openvla_dynamic_late_loto_500_2_repl \
  --output-dir runs/openvla_dynamic_late_loto_500_2_repl/functional_cp \
  --horizon 130 --fixed-horizons 13,25,37 \
  --task-bootstrap 20000 --bootstrap-seed 20260722
```

### References

- Gu, Q., Ju, Y., Sun, S., et al. [SAFE: Multitask Failure Detection for
  Vision-Language-Action Models](https://arxiv.org/abs/2506.09937),
  2025.
- [Official SAFE implementation](https://github.com/vla-safe/SAFE).
- Kim, M. J., Pertsch, K., Karamcheti, S., et al. [OpenVLA: An
  Open-Source Vision-Language-Action
  Model](https://arxiv.org/abs/2406.09246), 2024.
- Liu, B., Zhu, Y., Gao, C., et al. [LIBERO: Benchmarking Knowledge
  Transfer for Lifelong Robot
  Learning](https://arxiv.org/abs/2306.03310), 2023.
