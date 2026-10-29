# How much motion data does SuperMimic need to generalize?

A held-out data-scaling study of **SuperMimic** ([arXiv:2610.28480](https://arxiv.org/abs/2610.28480), CoRL 2026)
on BONES-SEED, built on [LaoYang1994/SuperMimic-Rep](https://github.com/LaoYang1994/SuperMimic-Rep) (itself a fork of
[CuboiLeo/SuperMimic-Reproduced](https://github.com/CuboiLeo/SuperMimic-Reproduced), MIT).

The paper's data-scaling result (Fig. 3) trains on 1, 7, 40 and 264 hours of BONES-SEED and evaluates **on the training
motions only**. It reports that larger datasets take longer to reach a target return and that the benefit plateaus
around 40 h. It does not ask the question a user of the method cares about: **does more data make the policy, and the
world model, better on motions they have never seen?** Community reproductions report held-out results for the full
dataset only, and none report world-model error on held-out data (see [Related results](#related-results)). This study
measures both across data sizes, under one protocol.

## Findings

Four runs on nested subsets of BONES-SEED (1 h, 7 h, 40 h, all 254 h), identical networks, hyperparameters and compute
(40,000 iterations each), evaluated every 4,000 iterations on three held-out sets carved out before any subset was
drawn. Headline numbers average the last three checkpoints (32k–40k).

| Run | Data | Unseen motions | Unseen performers | LAFAN1 (other capture source) | Own training motions |
|---|---:|---:|---:|---:|---:|
| D1 | 1 h | 0.867 | 0.872 | 0.624 | 0.960 |
| D7 | 7 h | 0.909 | 0.918 | 0.706 | 0.940 |
| D40 | 40 h | **0.926** | 0.929 | 0.755 | 0.942 |
| Dall | 254 h | **0.926** | 0.930 | **0.779** | 0.919 |

Mean normalized return; min–max ranges, failure rates and best checkpoints are in
[`results/results.md`](results/results.md).

![Tracking return vs training data](figures/fig1_heldout_vs_data.png)

1. **In-distribution generalization saturates at about 40 hours.** On held-out BONES-SEED motions and performers,
   return rises by +0.042 from 1 h to 7 h and +0.017 from 7 h to 40 h, then by **0.000** from 40 h to 254 h. The paper's
   plateau on training motions also holds for motions the policy never saw.
2. **Out-of-distribution transfer improves much more, and may keep improving past 40 h.** On LAFAN1, a different
   capture source the runs never trained on, return rises from 0.624 (1 h) to 0.779 (254 h), far beyond any plausible
   seed noise, and the failure rate halves (16% → 7%). Whether it still improves *after* 40 h is less certain: the
   +0.024 step from 40 h to 254 h is the size of the seed-to-seed spreads reported in the community (0.007–0.03), and
   D40's LAFAN1 score moved between 0.725 and 0.771 over its last three checkpoints. Dall is ahead of D40 on LAFAN1 at
   every checkpoint from 8k to 40k (9 of 10), which is suggestive, not conclusive.
3. **The generalization gap closes with data.** The gap between a run's own training motions and unseen motions shrinks
   from 0.093 (1 h) to 0.031 (7 h), 0.016 (40 h) and about zero (254 h). (The training column uses a 2,000-clip sample of
   each run's own subset, so it differs in clips as well as policy; the held-out columns are like-for-like.)
4. **Small data gives a lower ceiling, not a collapse.** D1 plateaus at about 0.865 on unseen motions by 20k iterations
   and does not decline through 40k, but it never reaches 0.90 and fails 4% of unseen-performer segments (0.1% for 40 h
   and 254 h).

![Learning curves](figures/fig2_learning_curves.png)

### The world model

5. **A 1-hour dataset gives a world model that looks excellent on its own data and is 6× worse off it.** Each run's
   world model is scored on the same pooled held-out transitions (8-step open-loop error, the paper's metric):

   | Run | Held-out windows (pooled) | Own training rollouts | Ratio |
   |---|---:|---:|---:|
   | D1 | 44.5 | 7.2 | 6.2× |
   | D7 | 21.3 | 9.8 | 2.2× |
   | D40 | 17.9 | 10.7 | 1.7× |
   | Dall | 17.7 | 13.0 | 1.4× |

   Normalized 8-step error ×10⁻³. Error on its own rollouts *rises* with data (more varied data is harder to fit),
   which reproduces the paper's Fig. 3(c) (about 7, 7.5, 11, 10). Error on held-out windows falls 2.5× from 1 h to 40 h and
   then flattens, exactly where held-out return flattens.
6. **D1's world model overfits before its policy does.** Its held-out error is lowest at 8k iterations (31) and climbs to
   47 by 40k while its training error keeps falling; the other runs keep improving to the end. D1's held-out *return*
   does not drop over the same period. (At each checkpoint the windows come from that step's policies, so the window set
   shifts over training; all runs are always scored on the same windows.)

![World-model error](figures/fig3_wm_error.png)

### Compute to target

7. **The paper's compute-to-target ranking reverses on held-out motions.** Iterations to a mean return of 0.85:

   | Run | Own training motions | Unseen motions |
   |---|---:|---:|
   | D1 | **5.7k** | 10.4k |
   | D7 | 7.4k | 7.8k |
   | D40 | 6.7k | 7.4k |
   | Dall | 9.3k | **7.3k** |

   On training motions smaller datasets get there first, the paper's Fig. 3(b) trend. On unseen motions it is the other
   way round, and D1 never reaches 0.90. "Smaller data is cheaper" is a property of measuring on the training set.

![Iterations to target](figures/fig4_compute_to_target.png)

## Setup

- **Method and code:** SuperMimic as reproduced in SuperMimic-Rep, paper networks and hyperparameters, the paper's
  observation setting (`beyondmimic_seq`). One H100 SXM per run, 2.3 iterations/s, about 5 GPU-hours per run.
- **Training length:** 40,000 iterations on the paper's 320,000-iteration learning-rate schedule (the schedule is kept;
  training stops early), checkpoints every 4,000. This covers the stage where data matters most; results at convergence
  may differ.
- **Data:** BONES-SEED G1 retargets, the paper's filters (129,511 clips, 263.08 h, matching the paper within 3 clips).
- **Held-out sets** ([`make_splits.py`](make_splits.py)), carved out before any training subset:
  - *Unseen performers:* every clip of randomly chosen performers (2,293 clips, 4.3 h).
  - *Unseen motions:* every clip, by any performer, of randomly chosen motion groups (2,006 clips, 4.4 h). A motion and
    its mirrored copy (`*_M`, which BONES-SEED stores under a separate `content_name`) are held out together, otherwise
    the mirror leaks the motion into training. Very common motions (>100 clips) are never held out, so one motion cannot
    dominate the set.
  - *LAFAN1:* all 40 G1-retargeted clips (2.45 h), a different capture source.
  - Every set is cut into 10-second segments, each played once from its first frame (paper Appendix F protocol).
- **Training subsets** ([`make_subsets.py`](make_subsets.py)): one shuffle of the remaining 254 h, cut at 1, 7 and 40 h,
  so each subset contains the smaller ones.
- **Training-motion evaluation:** a fixed sample of up to 2,000 clips of each run's own subset, same segment protocol.
- **World-model error** ([`wm_error.py`](wm_error.py)): every run's world model scored on the same pooled transitions
  recorded while all four policies played the held-out sets; open-loop on logged actions, normalized as in the paper.
- **Evaluation noise:** re-evaluating one checkpoint changed its return by 0.0007; checkpoint-to-checkpoint variation is
  about ±0.005 (the bracketed ranges in [`results/results.md`](results/results.md)).

## Caveats

- **One seed per data size,** as in the paper (which reports single runs, noting that training is very stable across
  seeds) and most community reproductions. Trends spanning the whole 1 h → 254 h range are far larger than the
  seed-to-seed spreads reported elsewhere (0.007–0.03); differences between D40 and Dall are not.
- **40,000 of 320,000 iterations.** These are mid-training results, not converged ones.
- **"Unseen motion" is only as clean as the dataset's labels.** Two different `content_name`s can be near-identical
  movements, so the motion split partly tests interpolation to similar motions. LAFAN1 is the clean
  out-of-distribution test.
- **Two operational blemishes.** D40 was interrupted at iteration 6,400 and resumed from its 4,000 checkpoint (the
  replay buffer is not checkpointed and was refilled). Dall shared its GPU with a stray process for a few minutes, which
  affects its wall-clock time but not its iterations. Neither should move the results.

## Related results

Checked on 2026-10-29 against the community list [awesome-supermimic](https://github.com/miao-org/awesome-supermimic)
and the reproductions it links. Held-out results exist only for models trained on the full dataset, and no entry reports
world-model error on held-out data. The closest comparisons:

- [realmimic](https://huggingface.co/kevinhenryz/realmimic) (full BONES-SEED, two seeds): LAFAN1 10-second segments,
  return 0.821–0.828 with 4–6% failures. Our Dall run reaches 0.779 with 7% failures at 40,000 iterations; the gap is
  consistent with our shorter training.
- [ScaleBFM_Plus](https://github.com/Hxxxz0/ScaleBFM_Plus): 99.5% success on 2,575 held-out motions. Our 40 h and 254 h
  runs fail 0.1–0.5% of held-out BONES-SEED segments (99.5–99.9% success), under a different split.

## Reproduce

[`RUNBOOK.md`](RUNBOOK.md) has every command, from download to figures. In short: build the BONES-SEED library
(section 1), make the splits and subsets (2), build the libraries (3), train and evaluate each run with
`--schedule.stop-iteration 40000` (5–6), then `post_runs.sh` (7). `analyze.py` regenerates every table and figure here
from the logs and evaluation summaries. CPU tests: `python -m pytest scaling/tests`.

## How this was built

I did this in about a day and a half, deliberately with an AI agent (Claude) doing most of the reading, coding and
debugging, to see how far that speeds up a real research question.

- **Choosing the question** (Wednesday morning): the agent surveyed the paper and the community reproductions to find extensions that had not been done. I chose the held-out scaling question and cut the scope from
  a full grid to four runs and one seed.
- **Code** (Wednesday): the agent wrote the scaling code in `scaling/` (splits, subsets, segmenting, held-out
  evaluation, pooled world-model error, analysis, tests) and the runbook. I set up the accounts, rented the GPUs and
  ran every command, pasting the output back.
- **Runs** (Wednesday evening to Thursday morning): data preparation took about 1.5 hours, then a smoke test, then four
  runs on two H100s overnight, about 27 rented GPU-hours in all.
- **Checking:** several problems were caught by checking real output against expectations rather than trusting the
  code: the mirrored-motion leak (found by inspecting the metadata before splitting), a command-line flag that silently
  dropped two held-out sets, a wrong metadata path, and the interrupted D40 run.
- **Write-up** (Thursday): the agent wrote the analysis and a first draft of this page from the results; I reviewed and
  edited it.
